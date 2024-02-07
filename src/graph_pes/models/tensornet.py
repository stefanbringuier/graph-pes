from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..data import AtomicGraph
from ..nn import MLP, PerSpeciesEmbedding
from .distances import CosineEnvelope, ExponentialRBF


class EdgeEmbedding(nn.Module):
    r"""
    Generates embeddings for each (directed) edge in the graph, incorporating
    the species of each atom, and the vector between them.

    1. generate initial edge embedding components:

       * :math:`I_0^{(ij)} = \text{Id}`
       * :math:`A_0^{(ij)} = \begin{pmatrix} 
         0 & \hat{r}_{ij}^z & - \hat{r}_{ij}^y \\
         - \hat{r}_{ij}^z & 0 & \hat{r}_{ij}^x \\
         \hat{r}_{ij}^y & - \hat{r}_{ij}^x & 0
         \end{pmatrix}`
       * :math:`S_0^{(ij)} = \hat{r}_{ij} \cdot \hat{r}_{ij}^T -
         \text{Tr}(\hat{r}_{ij} \cdot \hat{r}_{ij}^T) \text{Id}`

    2. generate an embedding of the ordered pair of species:

    .. math::
    
       h_Z^{(ij)} = f(z_i, z_j) = \text{Linear}(\text{embed}(z_i) 
       || \text{embed}(z_j))

    3. expand the edge vectors in to an exponential radial basis:

    .. math::
      
         h_{R,n}^{(ij)} = \exp(\beta_n \cdot (\exp(- r_{ij}) - \mu_k)^2)

    4. combine all edge embeddings:

    .. math::

        X^{ij} = \phi(r_{ij}) \cdot h_Z^{(ij)} \cdot \left ( 
        \text{Dense}(h_R^{(ij)}) \cdot I_0^{(ij)} +
        \text{Dense}(h_R^{(ij)}) \cdot A_0^{(ij)} +
        \text{Dense}(h_R^{(ij)}) \cdot S_0^{(ij)}
        \right)
    """

    def __init__(
        self,
        radial_features: int,
        embedding_size: int,
        cutoff: float,
    ):
        super().__init__()
        self.embedding_size = embedding_size

        self.z_embedding = PerSpeciesEmbedding(embedding_size)
        self.radial_expansion = ExponentialRBF(radial_features, cutoff)
        self.envelope = CosineEnvelope(cutoff)
        self.z_map = nn.Linear(2 * embedding_size, embedding_size, bias=False)
        self.r_map = nn.Linear(radial_features, 3 * embedding_size)

    def forward(
        self, graph: AtomicGraph
    ) -> tuple[
        Float[Tensor, "graph.n_edges self.embedding_size 3 3"],
        Float[Tensor, "graph.n_edges self.embedding_size 3 3"],
        Float[Tensor, "graph.n_edges self.embedding_size 3 3"],
    ]:
        # 1. generate initial edge embedding components:
        I_0, A_0, S_0 = self._initial_edge_embeddings(graph)  # (E, 1, 3, 3)

        # 2. encode atomic species
        # - get embeddings per atom
        h_z_atom = self.z_embedding(graph.Z)  # (N, C)
        # - combine atom embeddings for each edge
        h_z_edge = h_z_atom[graph.neighbour_index]  # (E, 2, C)
        # - concat along last axis
        h_z_edge = h_z_edge.reshape(-1, 2 * h_z_atom.shape[-1])  # (E, 2C)
        # - ... and map to embedding size
        h_z_edge = self.z_map(h_z_edge)  # (E, C)

        # 3. embed edge distances
        expansion_r = self.radial_expansion(graph.neighbour_distances)  # (E, K)
        env = self.envelope(graph.neighbour_distances)  # (E,)

        # combine information into coefficients
        c = (
            self.r_map(expansion_r)
            * env[..., None]
            * h_z_edge.repeat(1, 3)  # (E, 3C)
        )  # (E, 3C)
        c_I, c_A, c_S = torch.split(
            c[..., None, None], self.embedding_size, dim=1
        )  # (E, C, 1, 1)

        return c_I * I_0, c_A * A_0, c_S * S_0

    def _initial_edge_embeddings(
        self, graph: AtomicGraph
    ) -> tuple[
        Float[Tensor, "graph.n_edge 1 3 3"],
        Float[Tensor, "graph.n_edge 1 3 3"],
        Float[Tensor, "graph.n_edge 1 3 3"],
    ]:
        r_hat = graph.neighbour_vectors / graph.neighbour_distances[..., None]
        eye = torch.eye(3, device=graph.Z.device)
        I_ij = torch.repeat_interleave(eye[None, ...], graph.n_edges, dim=0)
        A_ij = vector_to_skew_vector(r_hat)
        S_ij = vector_to_symmetric_tensor(r_hat)
        return (
            I_ij.view(graph.n_edges, 1, 3, 3),
            A_ij.view(graph.n_edges, 1, 3, 3),
            S_ij.view(graph.n_edges, 1, 3, 3),
        )


def vector_to_skew_vector(
    v: Float[Tensor, "... 3"]
) -> Float[Tensor, "... 3 3"]:
    """Creates a skew-symmetric tensor from a vector."""
    batch_size = v.size(0)
    zero = torch.zeros(batch_size, device=v.device, dtype=v.dtype)
    tensor = torch.stack(
        (
            zero,
            -v[:, 2],
            v[:, 1],
            v[:, 2],
            zero,
            -v[:, 0],
            -v[:, 1],
            v[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)


def vector_to_symmetric_tensor(
    v: Float[Tensor, "... 3"]
) -> Float[Tensor, "... 3 3"]:
    """Creates a symmetric tensor from a vector."""
    tensor = torch.matmul(v.unsqueeze(-1), v.unsqueeze(-2))
    Id = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[
        ..., None, None
    ] * torch.eye(3, 3, device=tensor.device, dtype=tensor.dtype)
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - Id
    return S


def frobenius_norm(t):
    return (t**2).sum((-2, -1))


class TensorLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._linear = nn.Linear(in_features, out_features, bias=False)

    def forward(
        self, x: Float[Tensor, "... self.in_features 3 3"]
    ) -> Float[Tensor, "... self.out_features 3 3"]:
        return self._linear(x.transpose(-1, -3)).transpose(-1, -3)


class Embedding(nn.Module):
    def __init__(
        self,
        radial_features: int,
        embedding_size: int,
        cutoff: float,
    ):
        super().__init__()
        self.edge_embedding = EdgeEmbedding(
            radial_features, embedding_size, cutoff
        )
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.mlp = MLP(
            layers=[
                embedding_size,
                2 * embedding_size,
                3 * embedding_size,
            ],
            activation=nn.SiLU(),
            activate_last=True,
        )
        self.W_I = TensorLinear(embedding_size, embedding_size)
        self.W_A = TensorLinear(embedding_size, embedding_size)
        self.W_S = TensorLinear(embedding_size, embedding_size)

    def forward(
        self, graph: AtomicGraph
    ) -> Float[Tensor, "graph.n_atoms self.embedding_size 3 3"]:
        I_edge, A_edge, S_edge = self.edge_embedding(graph)  # (E, C, 3, 3)

        # sum over edges
        central_atom, neighbours = graph.neighbour_index  # (E,)

        def sum_per_atom(edge_tensor: Tensor) -> Tensor:
            shape = (graph.n_atoms, self.edge_embedding.embedding_size, 3, 3)
            atom_tensor = torch.zeros(*shape, device=graph.Z.device)
            atom_tensor.scatter_add_(
                0,
                central_atom[:, None, None, None].expand_as(edge_tensor),
                edge_tensor,
            )
            return atom_tensor

        I_atom = sum_per_atom(I_edge)  # (N, C, 3, 3)
        A_atom = sum_per_atom(A_edge)  # (N, C, 3, 3)
        S_atom = sum_per_atom(S_edge)  # (N, C, 3, 3)

        norms = frobenius_norm(I_atom + A_atom + S_atom)  # (N, C)
        norms = self.layer_norm(norms)
        coefficients = self.mlp(norms)  # (N, 3C)

        c_I, c_A, c_S = torch.split(
            coefficients[..., None, None],
            self.edge_embedding.embedding_size,
            dim=1,
        )  # (N, C, 1, 1)

        X = (
            c_I * self.W_I(I_atom)
            + c_A * self.W_A(A_atom)
            + c_S * self.W_S(S_atom)
        )  # (N, C, 3, 3)

        return X
