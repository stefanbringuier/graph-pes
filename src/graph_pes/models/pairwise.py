from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from graph_pes.core import GraphPESModel
from graph_pes.data import AtomicGraph
from graph_pes.nn import MLP, PositiveParameter
from torch import nn
from torch_geometric.utils import scatter

from .distances import Bessel, DistanceExpansion, Envelope, PolynomialEnvelope


class PairPotential(GraphPESModel, ABC):
    r"""
    An abstract base class for PES models that calculate system energy as
    a sum over pairwise interactions:

    .. math::
        E = \sum_{i < j} V(r_{ij}, Z_i, Z_j)

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    and :math:`Z_i` and :math:`Z_j` are their atomic numbers.
    """

    @abstractmethod
    def interaction(
        self, r: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
    ) -> torch.Tensor:
        """compute the pairwise interaction between atoms i and j"""

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        central_atoms, neighbours = graph.neighbour_index
        distances = graph.neighbour_distances

        Z_i, Z_j = graph.Z[central_atoms], graph.Z[neighbours]
        V = self.interaction(
            distances.view(-1, 1), Z_i.view(-1, 1), Z_j.view(-1, 1)
        )

        # sum over the neighbours and divide by 2 to avoid double counting
        return scatter(V.squeeze(), central_atoms, dim=0, reduce="sum") / 2


class LennardJones(PairPotential):
    r"""
    A pair potential of the form:

    .. math::
        \varepsilon_{ij} = 4 \varepsilon \left[ \left( \frac{\sigma}{r_{ij}}
        \right)^{12} - \left( \frac{\sigma}{r_{ij}} \right)^{6} \right]

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`.

    Parameters
    ----------
    epsilon : Optional[float]
        The depth of the potential.
    sigma : Optional[float]
        The distance at which the potential is zero.

    Attributes
    ----------
    epsilon : torch.nn.Parameter
        The depth of the potential.
    sigma : torch.nn.Parameter
        The distance at which the potential is zero.
    """

    def __init__(self, epsilon: float = 1.0, sigma: float = 0.5):
        super().__init__()
        self.epsilon = PositiveParameter(epsilon)
        self.sigma = PositiveParameter(sigma)

    def interaction(
        self, r: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
    ):
        """
        Evaluate the pair potential.

        Parameters
        ----------
        r : torch.Tensor
            The pair-wise distances between the atoms.
        """
        x = self.sigma / r
        return 4 * self.epsilon * (x**12 - x**6)


class SimplePP(PairPotential):
    r"""
    .. math::
        V(r) = \text{Envelope}(r) \odot \text{MLP}(\text{Expansion}(r))
    """

    def __init__(
        self,
        hidden_layers: list[int],
        cutoff: float,
        radial_features: int = 8,
        activation: str | torch.nn.Module = "CELU",
        expansion: type[DistanceExpansion] = Bessel,
        envelope: Callable[[float], Envelope] = PolynomialEnvelope,
    ):
        super().__init__()
        self.envelope = envelope(cutoff)
        self.raw_interaction = nn.Sequential(
            expansion(radial_features, cutoff),
            MLP([radial_features] + hidden_layers + [1], activation),
        )

    def interaction(
        self, r: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
    ) -> torch.Tensor:
        return self.raw_interaction(r).view(-1, 1) * self.envelope(r)


# class LearnablePairPotential(PairPotential):
#     def __init__(
#         self,
#         hidden_layers: list[int],
#         cutoff: float = 3.7,
#         r_features: int = 6,
#         z_features: int = 4,
#         activation: str = "CELU",
#     ):
#         super().__init__()

#         self.envelope = PolynomialEnvelope(cutoff)
#         self.r_embedding = Bessel(r_features, cutoff)
#         self.z_embedding = torch.nn.Embedding(MAX_Z, z_features)

#         layers = [r_features + 2 * z_features] + hidden_layers + [1]
#         self.mlp = MLP(layers, activation)

#     def pair_potential(
#         self, distances: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
#     ) -> torch.Tensor:
#         e = self.envelope(distances)

#         # embed the pairwise distance and atomic numbers
#         # application of the envelope here ensures smoothness wrt.
#         # atoms entering and leaving the cutoff shell
#         h_r = self.r_embedding(distances) * e
#         h_i = self.z_embedding(Z_i).squeeze()
#         h_j = self.z_embedding(Z_j).squeeze()

#         # concatenate the features and predict the pair potential
#         h = torch.cat([h_r, h_i, h_j], dim=-1)
#         V = self.mlp(h)

#         # the second envelope application ensures that the potential
#         # has the correct asymptotic behaviour (V -> 0 as r -> cutoff)
#         return V * e


# class MultiLennardJones(PairPotential):
#     r"""
#     A multi-component Lennard Jones potential:

#     .. math::
#         U_{ij}(r) = 4\epsilon_{ij} \left[ \left( \frac{\sigma_{ij}}{r}
#          \right)^{12} - \left( \frac{\sigma_{ij}}{r} \right)^6 \right]

#     where both :math:`\epsilon_{ij}` and :math:`\sigma_{ij}` are functions of
#     the atomic numbers :math:`Z_i` and :math:`Z_j`:
#     """

#     def __init__(
#         self,
#         epsilon: torch.Tensor | float = 1.0,
#         sigma: torch.Tensor | float = 0.5,
#     ):
#         super().__init__()

#         if isinstance(epsilon, float):
#             _log_epsilon = torch.log(torch.ones(MAX_Z, MAX_Z) * epsilon)
#         else:
#             if not len(epsilon.shape) == 2:
#                 raise ValueError(
#                     f"epsilon must be a 2D tensor, got {epsilon.shape}"
#                 )
#             if not (epsilon > 0).all():
#                 raise ValueError("epsilon must be positive for all pairs i,j")
#             _log_epsilon = torch.log(epsilon)
#         self._log_epsilon = torch.nn.Parameter(_log_epsilon,
# requires_grad=True)

#         if isinstance(sigma, float):
#             _log_sigma = torch.log(torch.ones(MAX_Z, MAX_Z) * sigma)
#         else:
#             if not len(sigma.shape) == 2:
#                 raise ValueError(
#                     f"sigma must be a 2D tensor, got {sigma.shape}"
#                 )
#             if not (sigma > 0).all():
#                 raise ValueError("sigma must be positive for all pairs i,j")
#             _log_sigma = torch.log(sigma)
#         self._log_sigma = torch.nn.Parameter(_log_sigma, requires_grad=True)

#         self.offset = torch.nn.Parameter(torch.zeros(MAX_Z, MAX_Z))

#     @property
#     def epsilon(self):
#         """The depth of the LJ well for each pair of elements"""
#         return torch.exp(self._log_epsilon)

#     @property
#     def sigma(self):
#         """The position of the LJ minimum for each pair of elements"""
#         return torch.exp(self._log_sigma)

#     def pair_potential(
#         self, distances: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
#     ) -> torch.Tensor:
#         # to keep the potential symmetric, we use the average of the two
#         # values for epsilon and sigma
#         epsilon = self.epsilon[Z_i, Z_j] + self.epsilon[Z_j, Z_i]
#         sigma = self.sigma[Z_i, Z_j] + self.sigma[Z_j, Z_i]

#         x = sigma / distances
#         lj = 4 * epsilon * (x**12 - x**6)

#         # we add an aribtrary offset to the potential
#         return lj + (self.offset[Z_i, Z_j] + self.offset[Z_j, Z_i]) / 2
