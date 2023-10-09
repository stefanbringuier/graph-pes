from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from graph_pes.data import AtomicGraph
from graph_pes.data.batching import AtomicGraphBatch, sum_per_structure
from graph_pes.models.transforms import LocalEnergyTransform


class GraphPESModel(nn.Module, ABC):
    r"""
    An abstract base class for all graph-based models of the PES that
    make predictions of the total energy of a structure as a sum
    of local contributions:

    .. math::
        E = f(\mathcal{G}) = \sum_i \varepsilon(\mathcal{N}_i)

    To create such a model, implement :meth:`predict_local_energies`,
    which takes a :class:`AtomicGraph` and returns a per-atom prediction
    of the local energy. Under the hood, :class:`GraphPESModel` uses
    these predictions to compute the total energy of the structure
    (complete with a learnable per-species shift and scale) in the
    forward pass.

    .. note::
        All :class:`GraphPESModel` instances are also instances of
        :class:`torch.nn.Module`. This allows for easy optimisation
        of parameters, and automated save/load functionality.
    """

    @abstractmethod
    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        """
        Predict the (standardized) local energy for each atom in the structure,
        as represented by `graph`.

        Parameters
        ----------
        graph : AtomicGraph
            The graph representation of the structure.
        """

    def forward(self, graph: AtomicGraph | AtomicGraphBatch):
        """
        Predict the total energy of the structure/s.

        Parameters
        ----------
        graph : AtomicGraph
            The graph representation of the structure/s.
        """
        raw_local_energies = self.predict_local_energies(graph).squeeze()
        local_energies = self._transform(raw_local_energies, graph.Z)
        return sum_per_structure(local_energies, graph)

    def __init__(self):
        super().__init__()
        self._transform = LocalEnergyTransform()
        self._transform_has_been_fit = False

    def fit_transform_to(self, train_data: list[AtomicGraph]) -> None:
        """
        Fit the local energy transform to the provided data.

        Parameters
        ----------
        train_data : list[AtomicGraph]
            The training data.
        """
        self._transform.fit_to(train_data)
        self._transform_has_been_fit = True

    def __repr__(self):
        return nn.Module.__repr__(self)


class MessagePassingPESModel(GraphPESModel, MessagePassing, ABC):
    # takes care of the complicated __init__ method

    def __init__(self, aggr: str = "add"):
        MessagePassing.__init__(self, aggr=aggr)
        self._transform = LocalEnergyTransform()
        self._transform_has_been_fit = False


class PairPotential(MessagePassingPESModel, ABC):
    r"""
    An abstract base class models that sum over pair-wise interactions
    to model the total energy of a structure. We define this as:

    .. math::
        E = \frac{1}{2} \sum_{i \neq j} V(r_{ij}, Z_i, Z_j)

    where :math:`V` is the pair potential, :math:`r_{ij}` is the pair-wise
    distance between atoms :math:`i` and :math:`j`, and :math:`Z_i` and
    :math:`Z_j` are the atomic numbers of atoms :math:`i` and :math:`j`.
    
    .. note::
        For a symmetrical pair potential, this is equivalent to:

        .. math::
            E = \sum_{i<j} V(r_{ij}, Z_i, Z_j)

    We can re-arrange the above to recast a pair potential as a 
    local energy model:

    .. math::
        \begin{align*}
        E &= \sum_i \frac{1}{2} \sum_{j \in \mathcal{N}_i} V(r_{ij}, Z_i, Z_j) \\
        &= \sum_i \varepsilon_i

    where :math:`\varepsilon_i` is the local energy of atom :math:`i`, and
    the neighbourhood :math:`\mathcal{N}_i` of atom :math:`i` effectively
    defines the cutoff radius of the pair potential. This in turn can be
    interpreted as a primitive message passing scheme, where the message
    passed from atom :math:`j` to atom :math:`i` is the pair potential,
    aggregation is a sum and update is identity.
    Given this correspondence, we implement :class:`PairPotential` as a
    subclass of both :class:`GraphPESModel` and :class:`MessagePassing`.

    To create such a model, implement :meth:`pair_potential`.
    """

    def __init__(self):
        MessagePassingPESModel.__init__(self, aggr="add")

    @abstractmethod
    def pair_potential(
        self,
        distances: torch.Tensor,
        Z_i: torch.Tensor,
        Z_j: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Evaluate the pair potential.

        Parameters
        ----------
        distances : torch.Tensor
            The pair-wise distances between the atoms.
        Z_i : torch.Tensor
            The atomic numbers of atom :math:`i`.
        Z_j : torch.Tensor
            The atomic numbers of atom :math:`j`.
        """

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        # TODO handle the case where we have an isolated node
        return self.propagate(
            graph.neighbour_index,
            distances=graph.neighbour_distances.unsqueeze(-1),
            Z=graph.Z.unsqueeze(-1),
        )

    def message(
        self, distances: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
    ) -> torch.Tensor:
        return self.pair_potential(distances, Z_i, Z_j)


class SimplePairPotential(PairPotential, ABC):
    r"""
    A :class:`PairPotential` where the pair-potential doesn't depend on
    the atomic numbers of the atoms:

    .. math::
        V(r_{ij}, Z_i, Z_j) = V(r_{ij})
    """

    @abstractmethod
    def pair_potential(self, distances: torch.Tensor) -> torch.Tensor:
        r"""
        Evaluate the pair potential.

        Parameters
        ----------
        distances : torch.Tensor
            The pair-wise distances between the atoms.
        """

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        return self.propagate(
            graph.neighbour_index,
            distances=graph.neighbour_distances.unsqueeze(-1),
        )

    def message(self, distances: torch.Tensor) -> torch.Tensor:
        return self.pair_potential(distances)

    # TOOD: add _from seqeuntial method
