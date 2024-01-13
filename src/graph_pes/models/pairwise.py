from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from graph_pes.core import EnergySummation, GraphPESModel
from graph_pes.data import AtomicGraph
from graph_pes.data.batching import AtomicGraphBatch
from graph_pes.nn import MLP, PositiveParameter
from graph_pes.transform import PerAtomShift
from jaxtyping import Float
from torch import Tensor, nn
from torch_geometric.utils import scatter

from .distances import Bessel, DistanceExpansion, Envelope, PolynomialEnvelope


class PairPotential(GraphPESModel, ABC):
    r"""
    An abstract base class for PES models that calculate system energy as
    a sum over pairwise interactions:

    .. math::
        E = \sum_{i, j} V(r_{ij}, Z_i, Z_j)

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    and :math:`Z_i` and :math:`Z_j` are their atomic numbers.

    This can be recast as a sum over local energy contributions,
    :math:`E = \sum_i \varepsilon_i`, according to:

    .. math::
        \varepsilon_i = \frac{1}{2} \sum_j V(r_{ij}, Z_i, Z_j)
    """

    @abstractmethod
    def interaction(
        self,
        r: Float[Tensor, "E"],
        Z_i: Float[Tensor, "E"],
        Z_j: Float[Tensor, "E"],
    ) -> Float[Tensor, "E"]:
        """
        Compute the interactions between pairs of atoms, given their
        distances and atomic numbers.

        Parameters
        ----------
        r
            The pair-wise distances between the atoms.
        Z_i
            The atomic numbers of the central atoms.
        Z_j
            The atomic numbers of the neighbours.

        Returns
        -------
        V: Float[Tensor, "E"]
            The pair-wise interactions.
        """

    def predict_local_energies(
        self, graph: AtomicGraph
    ) -> Float[Tensor, "graph.n_edges"]:
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
        V(r_{ij}, Z_i, Z_j) = V(r_{ij}) = 4 \varepsilon \left[ \left(
        \frac{\sigma}{r_{ij}} \right)^{12} - \left( \frac{\sigma}{r_{ij}}
        \right)^{6} \right]

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`.
    Internally, :math:`\varepsilon` and :math:`\sigma` are stored as
    :class:`PositiveParameter` instances, which ensures that they are
    strictly positive.

    Attributes
    ----------
    epsilon: :class:`PositiveParameter <graph_pes.nn.PositiveParameter>`
        The depth of the potential.
    sigma: :class:`PositiveParameter <graph_pes.nn.PositiveParameter>`
        The distance at which the potential is zero.
    """

    def __init__(self):
        super().__init__()
        self.epsilon = PositiveParameter(0.1)
        self.sigma = PositiveParameter(1.0)

        # epsilon is a scaling term, so only need to learn a shift
        # parameter (rather than a shift and scale)
        self._energy_summation = EnergySummation(local_transform=PerAtomShift())

    def interaction(
        self, r: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
    ):
        """
        Evaluate the pair potential.

        Parameters
        ----------
        r : torch.Tensor
            The pair-wise distances between the atoms.
        Z_i : torch.Tensor
            The atomic numbers of the central atoms. (unused)
        Z_j : torch.Tensor
            The atomic numbers of the neighbours. (unused)
        """
        x = self.sigma / r
        return 4 * self.epsilon * (x**12 - x**6)

    def pre_fit(self, graph: AtomicGraphBatch):
        super().pre_fit(graph)

        # set the potential depth to be shallow
        self.epsilon = PositiveParameter(0.01)

        # set the distance at which the potential is zero to be
        # close to the minimum pair-wise distance
        d = torch.quantile(graph.neighbour_distances, 0.01)
        self.sigma = PositiveParameter(d)


class Morse(PairPotential):
    r"""
    A pair potential of the form:

    .. math::
        V(r_{ij}, Z_i, Z_j) = V(r_{ij}) = D (1 - e^{-a(r_{ij} - r_0)})^2

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    and :math:`D`, :math:`a` and :math:`r_0` control the depth, width and
    center of the potential well, respectively. Internally, these are stored
    as :class:`PositiveParameter` instances.

    Attributes
    ----------
    D: :class:`PositiveParameter <graph_pes.nn.PositiveParameter>`
        The depth of the potential.
    a: :class:`PositiveParameter <graph_pes.nn.PositiveParameter>`
        The width of the potential.
    r0: :class:`PositiveParameter <graph_pes.nn.PositiveParameter>`
        The center of the potential.
    """

    def __init__(self):
        super().__init__()
        self.D = PositiveParameter(0.1)
        self.a = PositiveParameter(1.0)
        self.r0 = PositiveParameter(0.5)

        # D is a scaling term, so only need to learn a shift
        # parameter (rather than a shift and scale)
        self._energy_summation = EnergySummation(local_transform=PerAtomShift())

    def interaction(
        self, r: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
    ):
        """
        Evaluate the pair potential.

        Parameters
        ----------
        r : torch.Tensor
            The pair-wise distances between the atoms.
        Z_i : torch.Tensor
            The atomic numbers of the central atoms. (unused)
        Z_j : torch.Tensor
            The atomic numbers of the neighbours. (unused)
        """
        return self.D * (1 - torch.exp(-self.a * (r - self.r0))) ** 2

    def pre_fit(self, graph: AtomicGraphBatch):
        super().pre_fit(graph)

        # set the potential depth to be shallow
        self.D = PositiveParameter(0.1)

        # set the center of the well to be close to the minimum pair-wise
        # distance
        d = torch.quantile(graph.neighbour_distances, 0.01)
        self.r0 = PositiveParameter(d)

        # set the width to be broad
        self.a = PositiveParameter(0.5)


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
