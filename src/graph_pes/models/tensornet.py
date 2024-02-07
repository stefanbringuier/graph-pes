from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..data import AtomicGraph
from ..nn import PerSpeciesEmbedding
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
        cutoff: float,
        embedding_size: int,
    ):
        super().__init__()
        self.embedding_size = embedding_size

        self.z_embedding = PerSpeciesEmbedding(embedding_size)
        self.radial_expansion = ExponentialRBF(embedding_size, cutoff)
        self.envelope = CosineEnvelope(cutoff)
        self.z_map = nn.Linear(2 * embedding_size, embedding_size, bias=False)
        self.r_map = nn.Linear(embedding_size, 3 * embedding_size)

    def forward(
        self, graph: AtomicGraph
    ) -> Float[Tensor, "graph.n_edges self.embedding_size 3 3"]:
        # 1. generate initial edge embedding components:
        I_0, A_0, S_0 = self._initial_edge_embeddings(graph)  # (E, 3, 3)

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
        c_r = self.r_map(expansion_r)[..., None, None]  # (E, 3C, 1, 1)
        c_r_I, c_r_A, c_r_S = torch.split(
            c_r, self.embedding_size, dim=1
        )  # (E, C, 1, 1)

        # 4. combine all edge embeddings
        env = self.envelope(graph.neighbour_distances)[..., None]  # (E, 1)
        return (
            h_z_edge[..., None, None]
            * env[..., None, None]
            * (
                c_r_I * I_0.view(-1, 1, 3, 3)
                + c_r_A * A_0.view(-1, 1, 3, 3)
                + c_r_S * S_0.view(-1, 1, 3, 3)
            )
        )

    def _initial_edge_embeddings(
        self, graph: AtomicGraph
    ) -> tuple[
        Float[Tensor, "graph.n_edge 3 3"],
        Float[Tensor, "graph.n_edge 3 3"],
        Float[Tensor, "graph.n_edge 3 3"],
    ]:
        r_hat = graph.neighbour_vectors / graph.neighbour_distances[..., None]
        eye = torch.eye(3, device=graph.Z.device)
        I_ij = torch.repeat_interleave(eye[None, ...], graph.n_edges, dim=0)
        A_ij = vector_to_skew_vector(r_hat)
        S_ij = vector_to_symmetric_tensor(r_hat)
        return I_ij, A_ij, S_ij


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
