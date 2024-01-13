from __future__ import annotations

import warnings
from typing import Iterable

import ase
import numpy as np
import torch
from ase.neighborlist import neighbor_list
from jaxtyping import Float, Int, Shaped
from torch import Tensor

from ..util import as_possible_tensor, shape_repr


class AtomicGraph:
    """
    A graph representation of an atomic structure, compatible with
    PyTorch/Geometric/Lightning.

    .. note::
        We use `jaxtyping <https://docs.kidger.site/jaxtyping/>`_ notation
        to annotate all tensor shapes. Each :class:`AtomicGraph` contains
        :math:`N` atoms and :math:`E` edges

    Parameters
    ----------
    Z
        The atomic numbers of the atoms.
    positions
        The positions of the atoms.
    neighbour_index
        An edge list specifying neighbouring atoms :math:`j`
        for each central atom :math:`i`.
    cell
        The unit cell of the structure.
    neighbour_offsets
        An edge list specifying the periodic offset of neighbouring atoms
        :math:`j` for each central atom :math:`i`.
    atom_labels
        Additional, user-defined, per-atom labels.
    edge_labels
        Additional, user-defined, per-edge labels.
    structure_labels
        Additional, user-defined, per-structure labels.
    """

    def __init__(
        self,
        Z: Int[Tensor, "N"],
        positions: Float[Tensor, "N 3"],
        neighbour_index: Int[Tensor, "2 E"],
        cell: Float[Tensor, "3 3"],
        neighbour_offsets: Int[Tensor, "E 3"],
        atom_labels: dict[str, Shaped[Tensor, "N ..."]] | None = None,
        edge_labels: dict[str, Shaped[Tensor, "E ..."]] | None = None,
        structure_labels: dict[str, Tensor] | None = None,
    ):
        self.Z = Z
        self.neighbour_index = neighbour_index
        self.cell = cell
        self.neighbour_offsets = neighbour_offsets
        self.atom_labels = atom_labels or {}
        self.edge_labels = edge_labels or {}
        self.structure_labels = structure_labels or {}

        # we store positions privately, such that we can warn users
        # who access them directly, via the .position property below,
        # in case they naïvely use them to calculate neighbour vectors
        # or distances, which will be incorrect for periodic systems.
        self._positions = positions

    @classmethod
    def from_isolated_structure(
        cls,
        Z: Int[Tensor, "N"],
        positions: Float[Tensor, "N 3"],
        neighbour_index: Int[Tensor, "2 E"],
        atom_labels: dict[str, Shaped[Tensor, "N ..."]] | None = None,
        edge_labels: dict[str, Shaped[Tensor, "E ..."]] | None = None,
        structure_labels: dict[str, Tensor] | None = None,
    ) -> AtomicGraph:
        """
        Create a graph from a structure with no periodic boundary conditions.

        Parameters
        ----------
        Z
            The atomic numbers of the atoms.
        positions
            The positions of the atoms.
        neighbour_index
            An edge list specifying neighbouring atoms :math:`j`
            for each central atom :math:`i`.
        atom_labels
            Additional, user-defined, per-atom labels.
        edge_labels
            Additional, user-defined, per-edge labels.
        structure_labels
            Additional, user-defined, per-structure labels.
        """
        cell = torch.FloatTensor(torch.zeros((3, 3)))
        neighbour_offsets = torch.Tensor(
            torch.zeros((neighbour_index.shape[1], 3))
        )

        return cls(
            Z=Z,
            positions=positions,
            neighbour_index=neighbour_index,
            cell=cell,
            neighbour_offsets=neighbour_offsets,
            atom_labels=atom_labels,
            edge_labels=edge_labels,
            structure_labels=structure_labels,
        )

    @property
    def has_cell(self) -> bool:
        """Whether the structure has a unit cell."""
        return not torch.all(self.cell == 0).item()

    @property
    def positions(self):
        if self.has_cell:
            # raise a warning if the user accesses the positions directly,
            # since the most likely cause for this is that they are attempting
            # to calculate neighbour vectors/distances as:
            #   neighbour_vectors = positions[j] - positions[i]
            # which will be incorrect for periodic systems.

            warnings.warn(
                "Are you using `AtomicGraph.positions` to calculate "
                "neighbour vectors/distances?\n"
                "Use `AtomicGraph.neighbour_vectors` and "
                "`AtomicGraph.neighbour_distances` instead: for periodic "
                "systems, neighbour_vectors != positions[j] - positions[i]!",
                stacklevel=2,
            )
            # TODO link to docs

        return self._positions

    @property
    def neighbour_vectors(self) -> Float[Tensor, "E 3"]:
        """
        The vectors from central atoms :math:`i` to neighbours :math:`j`,
        respecting any periodic boundary conditions.
        """
        i, j = self.neighbour_index
        if not self.has_cell:
            return self._positions[j] - self._positions[i]

        neighbour_positions = (
            self._positions[j] + self.neighbour_offsets.float() @ self.cell
        )
        return neighbour_positions - self._positions[i]

    @property
    def neighbour_distances(self) -> Float[Tensor, "E"]:
        """
        The distances from central atoms :math:`i` to neighbours :math:`j`,
        respecting any periodic boundary conditions.
        """
        return self.neighbour_vectors.norm(dim=-1)

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the structure."""
        return self.Z.shape[0]

    @property
    def n_edges(self) -> int:
        """The number of edges in the graph."""
        return self.neighbour_index.shape[1]

    def to(self, device: torch.device | str) -> AtomicGraph:
        """Create a copy of this graph on the specified device."""

        def process_dict(d):
            return {key: value.to(device) for key, value in d.items()}

        return AtomicGraph(
            Z=self.Z.to(device),
            positions=self._positions.to(device),
            neighbour_index=self.neighbour_index.to(device),
            cell=self.cell.to(device),
            neighbour_offsets=self.neighbour_offsets.to(device),
            atom_labels=process_dict(self.atom_labels),
            edge_labels=process_dict(self.edge_labels),
            structure_labels=process_dict(self.structure_labels),
        )

    def get_labels(self, key: str) -> Tensor:
        """
        Get the labels for the specified key.

        Parameters
        ----------
        key
            The name of the label to retrieve.
        """
        for labels in (
            self.atom_labels,
            self.edge_labels,
            self.structure_labels,
        ):
            if key in labels:
                return labels[key]
        raise KeyError(f"Could not find label '{key}'")

    def is_local_property(self, x: Tensor) -> bool:
        """
        Check whether a tensor is a local property, based on its shape.

        Parameters
        ----------
        x
            The tensor to check.
        """
        return len(x.shape) > 0 and (x.shape[0] == self.n_atoms)

    def __repr__(self) -> str:
        device = self.Z.device
        _dict = dict(
            Z=self.Z,
            positions=self._positions,
            neighbour_index=self.neighbour_index,
            **self.atom_labels,
            **self.edge_labels,
            **self.structure_labels,
        )
        return f"AtomicGraph({shape_repr(_dict, sep=', ')}, device={device})"


def convert_to_atomic_graph(
    structure: ase.Atoms, cutoff: float, labels: list[str] | None = None
) -> AtomicGraph:
    """
    Convert an ASE Atoms object to an AtomicGraph.

    Parameters
    ----------
    structures
        The ASE Atoms object.
    cutoff
        The cutoff distance for neighbour finding.
    labels
        The names of any additional labels to include in the graph.
        These must be present in either the `atoms.info`
        or `atoms.arrays` dict. If not provided, all possible labels
        will be included.
    """

    atom_info, structure_info = extract_information(structure, labels)
    i, j, offsets = neighbor_list("ijS", structure, cutoff)
    return AtomicGraph(
        Z=torch.LongTensor(structure.numbers),
        positions=torch.FloatTensor(structure.positions),
        neighbour_index=torch.LongTensor(np.vstack([i, j])),
        cell=torch.FloatTensor(structure.cell.array),
        neighbour_offsets=torch.LongTensor(offsets),
        atom_labels=atom_info,
        structure_labels=structure_info,
    )


def extract_information(
    atoms: ase.Atoms,
    atom_keys: list[str] | None = None,
    structure_keys: list[str] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    Extract any additional information from the atoms object.

    Parameters
    ----------
    atoms
        The ASE Atoms object.
    atom_keys
        The names of any additional per-atom labels to include in the graph.
        If not provided, all possible labels will be included.
    structure_keys
        The names of any additional per-structure labels to include in the
        graph. If not provided, all possible labels will be included.

    Returns
    -------
    atom_info
        The per-atom labels as tensors
    structure_info
        The per-structure labels as tensors
    """

    return (
        extract_tensors(
            atoms.arrays, atom_keys, ignore=["numbers", "positions"]
        ),
        extract_tensors(atoms.info, structure_keys, ignore=["cell", "pbc"]),
    )


def extract_tensors(
    info_dict: dict[str, object],
    labels: list[str] | None,
    ignore: list[str] | None = None,
) -> dict[str, Tensor]:
    """
    Grab any tensors from the info dict.

    If labels is None, all items will be included. We warn if any of the
    items are not tensors, but continue anyway.
    If labels are provided, they must be present in the info dict,
    and be able to be converted to tensors.
    """

    strict = labels is not None
    ignore = ignore or []

    if strict:
        missing = set(labels) - set(info_dict.keys())
        if missing:
            raise KeyError(
                f"The following labels are not present in the info dict: "
                f"{', '.join(missing)}"
            )

    labels = labels if labels is not None else list(info_dict.keys())
    tensor_dict = {}

    for key, value in info_dict.items():
        if key not in labels or key in ignore:
            continue

        maybe_tensor = as_possible_tensor(value)

        if maybe_tensor is None:
            if strict:
                raise ValueError(
                    f"Label '{key}' is not a tensor and cannot be included."
                )
        else:
            if maybe_tensor.dtype == torch.float64:
                maybe_tensor = maybe_tensor.float()
            tensor_dict[key] = maybe_tensor

    return tensor_dict


def convert_to_atomic_graphs(
    structures: Iterable[ase.Atoms] | ase.Atoms,
    cutoff: float,
    labels: list[str] | None = None,
) -> list[AtomicGraph]:
    """
    Convert a collection of ASE Atoms into a list of AtomicGraphs.

    Parameters
    ----------
    structures
        The ASE Atoms objects.
    cutoff
        The cutoff distance for neighbour finding.
    labels
        The names of any additional labels to include in the graph.
        These must be present in either the `atoms.info`
        or `atoms.arrays` dict. If not provided, all possible labels
        will be included.
    """
    if isinstance(structures, ase.Atoms):
        structures = [structures]
    return [convert_to_atomic_graph(s, cutoff, labels) for s in structures]
