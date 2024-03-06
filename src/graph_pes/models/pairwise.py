from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from graph_pes.core import GraphPESModel
from graph_pes.data import (
    AtomicGraph,
    AtomicGraphBatch,
    keys,
    neighbour_distances,
    sum_over_neighbours,
)
from graph_pes.transform import PerAtomShift
from graph_pes.util import pytorch_repr, to_significant_figures
from jaxtyping import Float
from torch import Tensor


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

    Subclasses should implement :meth:`interaction`.
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

    def predict_local_energies(self, graph: AtomicGraph) -> Tensor:
        """
        Predict the local energies as half the sum of the pair-wise
        interactions that each atom participates in.
        """

        # avoid tuple unpacking to keep torchscript happy
        central_atoms = graph[keys.NEIGHBOUR_INDEX][0]
        neighbours = graph[keys.NEIGHBOUR_INDEX][1]
        distances = neighbour_distances(graph)

        Z_i = graph[keys.ATOMIC_NUMBERS][central_atoms]
        Z_j = graph[keys.ATOMIC_NUMBERS][neighbours]

        V = self.interaction(
            distances.view(-1, 1), Z_i.view(-1, 1), Z_j.view(-1, 1)
        )  # (E, 1)

        # sum over the neighbours
        energies = sum_over_neighbours(V.squeeze(), graph)

        # divide by 2 to avoid double counting
        return energies / 2


class LennardJones(PairPotential):
    r"""
    A pair potential of the form:

    .. math::
        V(r_{ij}, Z_i, Z_j) = V(r_{ij}) = 4 \varepsilon \left[ \left(
        \frac{\sigma}{r_{ij}} \right)^{12} - \left( \frac{\sigma}{r_{ij}}
        \right)^{6} \right]

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    and :math:`\varepsilon` and :math:`\sigma` are strictly positive
    paramters that control the depth and width of the potential well,

    Parameters
    ----------
    epsilon:
        The maximum depth of the potential.
    sigma:
        The distance at which the potential is zero.
    """

    def __init__(self, epsilon: float = 0.1, sigma: float = 1.0):
        super().__init__()
        self._log_epsilon = torch.nn.Parameter(torch.tensor(epsilon).log())
        self._log_sigma = torch.nn.Parameter(torch.tensor(sigma).log())

        # epsilon is a scaling term, so only need to learn a shift
        self.energy_transform = PerAtomShift()

    @property
    def epsilon(self):
        return self._log_epsilon.exp()

    @property
    def sigma(self):
        return self._log_sigma.exp()

    # don't use Z_i and Z_j, but include them for consistency with the
    # abstract method
    def interaction(self, r: torch.Tensor, Z_i=None, Z_j=None):
        """
        Evaluate the pair potential.

        Parameters
        ----------
        r
            The pair-wise distances between the atoms.
        """

        x = self.sigma / r
        return 4 * self.epsilon * (x**12 - x**6)

    def pre_fit(self, graph: AtomicGraphBatch):
        assert "energy" in graph

        # epsilon is already a scaling term, so we only need to
        # learn a shift parameter (rather than a shift and scale):
        transform = PerAtomShift().fit(graph["energy"], graph)

        # This transform maps total energies to a distribution with
        # 0 mean. We want to go the other way (i.e. transform our raw
        # predictions (with ~0 mean) into total energies, and so we use
        # the inverse of the transform.
        self.energy_transform = transform.inverse()

        # set the distance at which the potential is zero to be
        # close to the minimum pair-wise distance
        d = torch.quantile(neighbour_distances(graph), 0.01)
        self._log_sigma = torch.nn.Parameter(d.log())

    def __repr__(self):
        return pytorch_repr(
            "LennardJones",
            _modules={
                "epsilon": to_significant_figures(self.epsilon.item(), 3),
                "sigma": to_significant_figures(self.sigma.item(), 3),
                "energy_transform": self.energy_transform,
            },
        )


class Morse(PairPotential):
    r"""
    A pair potential of the form:

    .. math::
        V(r_{ij}, Z_i, Z_j) = V(r_{ij}) = D (1 - e^{-a(r_{ij} - r_0)})^2

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`,
    and :math:`D`, :math:`a` and :math:`r_0` are strictly positive parameters
    that control the depth, width and center of the potential well respectively.
    """

    def __init__(self, D: float = 0.1, a: float = 3.0, r0: float = 1.0):
        super().__init__()
        self._log_D = torch.nn.Parameter(torch.tensor(D).log())
        self._log_a = torch.nn.Parameter(torch.tensor(a).log())
        self._log_r0 = torch.nn.Parameter(torch.tensor(r0).log())

        # D is a scaling term, so only need to learn a shift
        # parameter (rather than a shift and scale)
        self.energy_transform = PerAtomShift()

    @property
    def D(self):
        return self._log_D.exp()

    @property
    def a(self):
        return self._log_a.exp()

    @property
    def r0(self):
        return self._log_r0.exp()

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
        assert "energy" in graph
        transform = PerAtomShift().fit(graph["energy"], graph)
        self.energy_transform = transform.inverse()

        # set the potential depth to be shallow
        self._log_D = torch.nn.Parameter(torch.tensor(0.1).log())

        # set the center of the well to be close to the minimum pair-wise
        # distance
        d = torch.quantile(neighbour_distances(graph), 0.01)
        self._log_r0 = torch.nn.Parameter(d.log())

        # set the width to be "reasonable"
        self._log_a = torch.nn.Parameter(torch.tensor(3.0).log())

    def __repr__(self):
        return pytorch_repr(
            "Morse",
            _modules={
                "D": to_significant_figures(self.D.item(), 3),
                "a": to_significant_figures(self.a.item(), 3),
                "r0": to_significant_figures(self.r0.item(), 3),
                "energy_transform": self.energy_transform,
            },
        )
