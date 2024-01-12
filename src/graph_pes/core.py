from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import NamedTuple

import torch
from graph_pes.data import AtomicGraph
from graph_pes.data.batching import AtomicGraphBatch, sum_per_structure
from graph_pes.transform import (
    Chain,
    Identity,
    PerSpeciesOffset,
    PerSpeciesScale,
    Transform,
)
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
        return self._energy_summation(local_energies, graph)

    def __init__(self):
        super().__init__()
        self._energy_summation = EnergySummation()

    # def __repr__(self):
    #     # modified from torch.nn.Module.__repr__
    #     # changes:
    #     # - don't print any modules that start with _

    #     # We treat the extra repr like the sub-module, one item per line
    #     extra_lines = []
    #     extra_repr = self.extra_repr()
    #     # empty string will be split into list ['']
    #     if extra_repr:
    #         extra_lines = extra_repr.split("\n")
    #     child_lines = []
    #     for key, module in self._modules.items():
    #         if key.startswith("_"):
    #             continue
    #         mod_str = repr(module)
    #         mod_str = nn.modules.module._addindent(mod_str, 2)
    #         child_lines.append("(" + key + "): " + mod_str)
    #     lines = extra_lines + child_lines

    #     main_str = self._get_name() + "("
    #     if lines:
    #         # simple one-liner info, which most builtin Modules will use
    #         if len(extra_lines) == 1 and not child_lines:
    #             main_str += extra_lines[0]
    #         else:
    #             main_str += "\n  " + "\n  ".join(lines) + "\n"

    #     main_str += ")"
    #     return main_str

    def __add__(self, other: GraphPESModel) -> Ensemble:
        """
        A convenient way to create a summation of two models.

        Examples
        --------
        >>> TwoBody() + ThreeBody()
        Ensemble([TwoBody(), ThreeBody()], aggregation=sum)
        """

        return Ensemble([self, other], mean=False)


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
                [PerSpeciesScale(), PerSpeciesOffset()], trainable=True
            )
        self.local_transform = local_transform or Identity()
        self.total_transform = total_transform or Identity()

    def forward(self, local_energies: torch.Tensor, graph: AtomicGraphBatch):
        local_energies = self.local_transform(local_energies, graph)
        total_E = sum_per_structure(local_energies, graph)
        total_E = self.total_transform(total_E, graph)
        return total_E

    def fit_to_graphs(
        self,
        graphs: AtomicGraphBatch | list[AtomicGraph],
        energy_key: str = "energy",
    ):
        if not isinstance(graphs, AtomicGraphBatch):
            graphs = AtomicGraphBatch.from_graphs(graphs)

        if energy_key in graphs.structure_labels:
            energies = graphs.structure_labels[energy_key]
        elif energy_key in graphs.atom_labels:
            energies = graphs.atom_labels[energy_key]
        else:
            raise KeyError(
                f"Could not find {energy_key} in either "
                f"structure_labels or atom_labels"
            )

        for transform in [self.local_transform, self.total_transform]:
            transform.fit_to_target(energies, graphs)


class Ensemble(GraphPESModel):
    def __init__(self, models: list[GraphPESModel], mean: bool = True):
        super().__init__()
        self.models: list[GraphPESModel] = nn.ModuleList(models)  # type: ignore
        self.mean = mean

    def predict_local_energies(self, graph: AtomicGraph | AtomicGraphBatch):
        s = sum(m.predict_local_energies(graph).squeeze() for m in self.models)
        return s / len(self.models) if self.mean else s

    def __repr__(self):
        aggregation = "mean" if self.mean else "sum"
        return f"Ensemble({self.models}, aggregation={aggregation})"


@contextmanager
def require_grad(tensor: torch.Tensor):
    # check if in a torch.no_grad() context: if so,
    # raise an error
    if not torch.is_grad_enabled():
        raise RuntimeError(
            "Autograd is disabled, but you are trying to "
            "calculate gradients. Please wrap your code in "
            "a torch.enable_grad() context."
        )

    req_grad = tensor.requires_grad
    tensor.requires_grad_(True)
    yield
    tensor.requires_grad_(req_grad)


class Prediction(NamedTuple):
    energy: torch.Tensor
    forces: torch.Tensor
    stress: torch.Tensor | None = None


def get_predictions(pes: GraphPESModel, structure: AtomicGraph) -> Prediction:
    """
    Evaluate the `pes`  on `structure` to obtain the total energy,
    forces on each atom, and optionally a stress tensor (if the
    structure has a unit cell).

    Parameters
    ----------
    pes : PES
        The PES to use.
    structure : AtomicStructure
        The atomic structure to evaluate.
    """

    if not structure.has_cell:
        # don't care about stress
        with require_grad(structure._positions):
            total_energy = pes(structure)
            dE_dR = get_gradient(total_energy, (structure._positions,))[0]
        return Prediction(total_energy, -dE_dR)

    # the virial stress tensor is the gradient of the total energy wrt
    # an infinitesimal change in the cell parameters
    actual_cell = structure.cell
    change_to_cell = torch.zeros_like(actual_cell, requires_grad=True)
    symmetric_change = 0.5 * (change_to_cell + change_to_cell.transpose(-1, -2))
    structure.cell = actual_cell + symmetric_change
    with require_grad(structure._positions):
        total_energy = pes(structure)
        (dE_dR, dCell_dR) = get_gradient(
            total_energy, (structure._positions, change_to_cell)
        )

    structure.cell = actual_cell
    volume = torch.det(actual_cell).view(-1, 1, 1)
    stress = -dCell_dR / volume

    return Prediction(total_energy, -dE_dR, stress)


def energy_and_forces(pes: GraphPESModel, structure: AtomicGraph):
    """
    Evaluate the `pes`  on `structure` to obtain both the
    total energy and the forces on each atom.

    Parameters
    ----------
    pes : PES
        The PES to use.
    structure : AtomicStructure
        The atomic structure to evaluate.

    Returns
    -------
    EnergyAndForces
        The energy of the structure and forces on each atom.
    """

    # use the autograd machinery to auto-magically
    # calculate forces for (almost) free
    structure._positions.requires_grad_(True)
    energy = pes(structure)
    dE_dR = get_gradient(energy, (structure._positions,))[0]
    structure._positions.requires_grad_(False)
    return dict(energy=energy.squeeze(), forces=-dE_dR)


def get_gradient(
    energy: torch.Tensor,
    things: tuple[torch.Tensor, ...],
):
    grads = torch.autograd.grad(
        energy.sum(),
        things,
        create_graph=True,
        allow_unused=True,
    )

    return tuple(
        output
        if output is not None
        else torch.zeros_like(input, requires_grad=True)
        for (output, input) in zip(grads, things)
    )
