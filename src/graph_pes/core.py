from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from graph_pes.data import AtomicGraph
from graph_pes.data.batching import AtomicGraphBatch, sum_per_structure
from graph_pes.transform import (
    Chain,
    Identity,
    PerAtomScale,
    PerAtomShift,
    Transform,
)
from graph_pes.util import Property, differentiate, require_grad
from jaxtyping import Float
from torch import Tensor, nn


class GraphPESModel(nn.Module, ABC):
    r"""
    An abstract base class for all graph-based, energy-conserving models of the
    PES that make predictions of the total energy of a structure as a sum
    of local contributions:

    .. math::
        E(\mathcal{G}) = \sum_i \varepsilon_i

    To create such a model, implement :meth:`predict_local_energies`,
    which takes an :class:`AtomicGraph`, or an :class:`AtomicGraphBatch`,
    and returns a per-atom prediction of the local energy. For a simple example,
    see :class:`LennardJones <graph_pes.models.pairwise.LennardJones>`.

    Under the hood, :class:`GraphPESModel` contains an
    :class:`EnergySummation` module, which is responsible for
    summing over local energies to obtain the total energy/ies,
    with optional transformations of the local and total energies.
    By default, this learns a per-species, local energy offset and scale.

    .. note::
        All :class:`GraphPESModel` instances are also instances of
        :class:`torch.nn.Module`. This allows for easy optimisation
        of parameters, and automated save/load functionality.
    """

    # TODO: fix this for the case of an isolated atom, either by itself
    # or within a batch: perhaps that should go in sum_per_structure?
    # or maybe default to a local scale followed by a global peratomshift?
    @abstractmethod
    def predict_local_energies(
        self, graph: AtomicGraph | AtomicGraphBatch
    ) -> Float[Tensor, "graph.n_atoms"]:
        """
        Predict the (standardized) local energy for each atom in the graph.

        Parameters
        ----------
        graph
            The graph representation of the structure/s.
        """

    def forward(self, graph: AtomicGraph | AtomicGraphBatch):
        """
        Predict the total energy of the structure/s.

        Parameters
        ----------
        graph : AtomicGraph
            The graph representation of the structure/s.
        """

        # local predictions
        local_energies = self.predict_local_energies(graph).squeeze()

        # sum over atoms to get total energy
        return self.energy_summation(local_energies, graph)

    def __init__(self):
        super().__init__()
        self.energy_summation = EnergySummation()

    def __add__(self, other: GraphPESModel) -> Ensemble:
        """
        A convenient way to create a summation of two models.

        Examples
        --------
        >>> TwoBody() + ThreeBody()
        Ensemble([TwoBody(), ThreeBody()], aggregation=sum)
        """

        return Ensemble([self, other], aggregation="sum")

    def pre_fit(self, graphs: AtomicGraphBatch, energy_label: str = "energy"):
        """
        Perform optional pre-processing of the training data.

        By default, this fits a :class:`graph_pes.transform.PerAtomShift`
        and :class:`graph_pes.transform.PerAtomScale` to the energies
        of the training data, such that, before training, a unit-Normal
        output by the underlying model will result in energy predictions
        that are distributed according to the training data.

        Parameters
        ----------
        graphs
            The training data.
        """
        self.energy_summation.fit_to_graphs(graphs, energy_label)


class EnergySummation(nn.Module):
    def __init__(
        self,
        local_transform: Transform | None = None,
        total_transform: Transform | None = None,
    ):
        super().__init__()

        # if both None, default to a per-species, local energy offset
        if local_transform is None and total_transform is None:
            local_transform = Chain(
                [PerAtomShift(), PerAtomScale()], trainable=True
            )
        self.local_transform: Transform = local_transform or Identity()
        self.total_transform: Transform = total_transform or Identity()

    def forward(self, local_energies: torch.Tensor, graph: AtomicGraphBatch):
        local_energies = self.local_transform.inverse(local_energies, graph)
        total_E = sum_per_structure(local_energies, graph)
        total_E = self.total_transform.inverse(total_E, graph)
        return total_E

    def fit_to_graphs(
        self,
        graphs: AtomicGraphBatch | list[AtomicGraph],
        energy_label: str = "energy",
    ):
        if not isinstance(graphs, AtomicGraphBatch):
            graphs = AtomicGraphBatch.from_graphs(graphs)

        for transform in [self.local_transform, self.total_transform]:
            transform.fit(graphs[energy_label], graphs)


class Ensemble(GraphPESModel):
    """
    An ensemble of :class:`GraphPESModel` models.

    Parameters
    ----------
    models
        the models to ensemble.
    aggregation
        the method of aggregating the predictions of the models.
    weights
        scalar weights for combining each model's prediction.
    trainable_weights
        whether the weights are trainable.

    Examples
    --------

    >>> from graph_pes.models.pairwise import LennardJones
    >>> from graph_pes.models.schnet import SchNet
    >>> from graph_pes.core import Ensemble
    >>> # create an ensemble of two models
    >>> # equivalent to Ensemble([LennardJones(), SchNet()], aggregation="sum")
    >>> ensemble = LennardJones() + SchNet()

    See Also
    --------
    :meth:`GraphPESModel.__add__`
    """

    def __init__(
        self,
        models: list[GraphPESModel],
        aggregation: Literal["mean", "sum"] = "mean",
        weights: list[float] | None = None,
        trainable_weights: bool = False,
    ):
        super().__init__()
        self.models: list[GraphPESModel] = nn.ModuleList(models)  # type: ignore
        self.aggregation = aggregation
        self.weights = nn.Parameter(
            torch.tensor(
                weights or [1.0] * len(models), requires_grad=trainable_weights
            )
        )

        # use the energy summation of each model separately
        self.energy_summation = None

    def predict_local_energies(self, graph: AtomicGraph | AtomicGraphBatch):
        raise NotImplementedError(
            "Ensemble models don't have a single local energy prediction."
        )

    def forward(self, graph: AtomicGraph | AtomicGraphBatch):
        predictions: Tensor = sum(
            w * model(graph) for w, model in zip(self.weights, self.models)
        )  # type: ignore
        if self.aggregation == "mean":
            return predictions / self.weights.sum()
        else:
            return predictions

    def __repr__(self):
        info = [str(self.models), f"aggregation={self.aggregation}"]
        if self.weights.requires_grad:
            info.append(f"weights={self.weights.tolist()}")
        info = "\n  ".join(info)
        return f"Ensemble(\n  {info}\n)"


# TODO: add training flag to this so that we don't create the graph needlessly
# when in eval mode
# TODO: perhaps this should be a method of GraphPESModel?
def get_predictions(
    pes: GraphPESModel,
    structure: AtomicGraph | AtomicGraphBatch | list[AtomicGraph],
    property_labels: dict[Property, str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Evaluate the `pes` on `structure` to get the labels requested.

    Parameters
    ----------
    pes
        The PES to use.
    structure
        The atomic structure to evaluate.
    property_labels
        The names of the properties to return. If None, all available
        properties are returned.

    Returns
    -------
    dict[str, torch.Tensor]
        The requested properties.

    Examples
    --------
    >>> # TODO

    """

    if isinstance(structure, list):
        structure = AtomicGraphBatch.from_graphs(structure)

    if property_labels is None:
        property_labels = {
            Property.ENERGY: "energy",
            Property.FORCES: "forces",
        }
        if structure.has_cell:
            property_labels[Property.STRESS] = "stress"

    else:
        if Property.STRESS in property_labels and not structure.has_cell:
            raise ValueError("Can't predict stress without cell information.")

    predictions = {}

    # setup for calculating stress:
    if Property.STRESS in property_labels:
        # The virial stress tensor is the gradient of the total energy wrt
        # an infinitesimal change in the cell parameters.
        # We therefore add this change to the cell, such that
        # we can calculate the gradient wrt later if required.
        #
        # See <> TODO: find reference
        actual_cell = structure.cell
        change_to_cell = torch.zeros_like(actual_cell, requires_grad=True)
        symmetric_change = 0.5 * (
            change_to_cell + change_to_cell.transpose(-1, -2)
        )
        structure.cell = actual_cell + symmetric_change
    else:
        change_to_cell = torch.zeros_like(structure.cell)

    # use the autograd machinery to auto-magically calculate forces and stress
    # from the energy
    with require_grad(structure._positions), require_grad(change_to_cell):
        energy = pes(structure)

        if Property.ENERGY in property_labels:
            predictions[property_labels[Property.ENERGY]] = energy

        if Property.FORCES in property_labels:
            dE_dR = differentiate(energy, structure._positions)
            predictions[property_labels[Property.FORCES]] = -dE_dR

        if Property.STRESS in property_labels:
            stress = differentiate(energy, change_to_cell)
            predictions[property_labels[Property.STRESS]] = stress

    return predictions
