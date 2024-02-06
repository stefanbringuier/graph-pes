from __future__ import annotations

import torch
from graph_pes.core import GraphPESModel
from graph_pes.data import AtomicGraph
from graph_pes.nn import MLP, HaddamardProduct, PerSpeciesEmbedding
from jaxtyping import Float
from torch import Tensor, nn

from .distances import Bessel, PolynomialEnvelope


class InteractionBlock(nn.Module):
    def __init__(
        self,
        radial_features: int,
        internal_dim: int,
        cutoff: float,
    ):
        super().__init__()
        self.internal_dim = internal_dim

        self.filter_generator = HaddamardProduct(
            nn.Sequential(
                Bessel(radial_features, cutoff),
                nn.Linear(radial_features, internal_dim * 3),
            ),
            PolynomialEnvelope(cutoff),
        )

        self.φ = MLP(
            [internal_dim, internal_dim, internal_dim * 3],
            activation=nn.SiLU(),
        )

    def forward(
        self,
        vector_embeddings: Float[Tensor, "graph.n_atoms self.internal_dim 3"],
        scalar_embeddings: Float[Tensor, "graph.n_atoms self.internal_dim"],
        graph: AtomicGraph,
    ) -> tuple[
        Float[Tensor, "graph.n_atoms self.internal_dim 3"],
        Float[Tensor, "graph.n_atoms self.internal_dim"],
    ]:
        central_atoms, neighbours = graph.neighbour_index
        d = graph.neighbour_distances
        unit_vectors = graph.neighbour_vectors / d.unsqueeze(-1)

        # continous filter message creation
        x_ij = self.filter_generator(d) * self.φ(scalar_embeddings)
        a, b, c = torch.split(x_ij, self.internal_dim, dim=-1)

        # simple sum over neighbours to get scalar messages
        Δs = torch.zeros_like(scalar_embeddings)
        Δs.scatter_add_(0, neighbours, a)

        # create vector messages
        v_ij = b * unit_vectors + c * vector_embeddings[neighbours]

        Δv = torch.zeros_like(vector_embeddings)
        Δv.scatter_add_(0, neighbours, v_ij)

        return Δv, Δs


class VectorLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._linear = nn.Linear(in_features, out_features)

    def forward(
        self, x: Float[Tensor, "... self.in_features 3"]
    ) -> Float[Tensor, "... self.out_features 3"]:
        # a hack to swap the vector and channel dimensions
        return self._linear(x.transpose(-1, -2)).transpose(-1, -2)


class UpdateBlock(nn.Module):
    def __init__(self, internal_dim: int):
        super().__init__()
        self.internal_dim = internal_dim
        self.U = VectorLinear(internal_dim, internal_dim)
        self.V = VectorLinear(internal_dim, internal_dim)
        self.mlp = MLP(
            [internal_dim * 2, internal_dim, internal_dim * 3],
            activation=nn.SiLU(),
        )

    def forward(
        self,
        vector_embeddings: Float[Tensor, "batch self.internal_dim 3"],
        scalar_embeddings: Float[Tensor, "batch self.internal_dim"],
    ) -> tuple[
        Float[Tensor, "batch self.internal_dim 3"],
        Float[Tensor, "batch self.internal_dim"],
    ]:
        u = self.U(vector_embeddings)
        v = self.V(vector_embeddings)

        # stack scalar message and the norm of v
        m = torch.cat([scalar_embeddings, v.norm(dim=-1)], dim=-1)
        m = self.mlp(m)

        # split the update into 3 parts
        a, b, c = torch.split(m, self.internal_dim, dim=-1)

        # vector update:
        Δv = u * a.unsqueeze(-1)

        # scalar update:
        dot = torch.sum(u * v, dim=1, keepdim=True)  # u . v
        Δs = b + c * dot

        return Δv, Δs


class PaiNN(GraphPESModel):
    def __init__(
        self,
        internal_dim: int = 32,
        radial_features: int = 20,
        layers: int = 3,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.internal_dim = internal_dim
        self.layers = layers
        self.interactions: list[InteractionBlock] = nn.ModuleList(
            [
                InteractionBlock(radial_features, internal_dim, cutoff)
                for _ in range(layers)
            ]
        )  # type: ignore
        self.updates: list[UpdateBlock] = nn.ModuleList(
            [UpdateBlock(internal_dim) for _ in range(layers)]
        )  # type: ignore
        self.z_embedding = PerSpeciesEmbedding(internal_dim)
        self.read_out = MLP(
            [internal_dim, internal_dim, 1],
            activation=nn.SiLU(),
        )

    def predict_local_energies(
        self, graph: AtomicGraph
    ) -> Float[Tensor, "1 graph.n_atoms"]:
        vector_embeddings = torch.zeros(
            (graph.n_atoms, self.internal_dim, 3),
            device=graph.Z.device,
        )
        scalar_embeddings = self.z_embedding(graph.Z)

        for interaction, update in zip(self.interactions, self.updates):
            Δv, Δs = interaction(vector_embeddings, scalar_embeddings, graph)
            vector_embeddings += Δv
            scalar_embeddings += Δs

            Δv, Δs = update(
                vector_embeddings,
                scalar_embeddings,
            )
            vector_embeddings += Δv
            scalar_embeddings += Δs

        return self.read_out(scalar_embeddings)
