"""Defines :class:`AtomicGraph`, the central data object of GraphPES."""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from graph_pes.util import shape_repr


class AtomicGraph:
    r"""
    A graph representation of an atomic structure.
    Fully compatible with PyTorch/Geometric/Lightning.

    Parameters
    ----------
    Z: torch.Tensor
        The atomic numbers of the atoms. Shape: (n_atoms,)
    positions: torch.Tensor
        The positions of the atoms. Shape: (n_atoms, 3)
    neighbour_index: torch.LongTensor
        An edge list specifying neighbouring atoms :math:`j`
        for each central atom :math:`i`. Shape: (2, n_edges)
    unit_cell: torch.Tensor | None
        The unit cell of the structure. Shape: (3, 3)
    neighbour_offsets: torch.Tensor | None
        Offsets to apply to neighbour :math:`j` positions to account for
        periodic boundary conditions such that

        .. code-block:: python

            i, j = neighbour_index
            positions_i = positions[i]
            positions_j = positions[j] + neighbour_offsets @ unit_cell
            vectors_i_to_j = positions_j - positions_i

        Shape: (n_edges, 3)
    labels: dict[str, torch.Tensor]
        Additional, user defined labels for the structure/atoms/edges.

    Additional Properties:
    ----------------------
    neighbour_vectors: torch.Tensor
        The vectors from central atom :math:`i` to neighbours :math:`j`.
        Shape: (n_edges, 3)
    neighbour_distances: torch.Tensor
        The distances from central atom :math:`i` to neighbours :math:`j`.
        Shape: (n_edges,)
    n_atoms: int
        The number of atoms in the structure.
    n_edges: int
        The number of edges in the graph.
    """

    def __init__(
        self,
        Z: torch.ShortTensor,
        positions: torch.FloatTensor,
        neighbour_index: torch.LongTensor,
        cell: torch.FloatTensor | None = None,
        neighbour_offsets: torch.ShortTensor | None = None,
        **labels: torch.Tensor,
    ):
        # sanitise inputs:
        if neighbour_offsets is not None and cell is None:
            raise ValueError(
                "If neighbour_offsets is provided, cell must also be provided."
            )

        self.Z = Z
        """Atomic numbers of the atoms. Shape: (`n_atoms`,)"""

        self.neighbour_index = neighbour_index
        """An edge list specifying neighbouring atoms :math:`j`
        for each central atom :math:`i`. Shape: (2, n_edges)"""

        nothing_cell = torch.FloatTensor(torch.zeros((3, 3)))
        cell = cell if cell is not None else nothing_cell
        self.cell = cell
        """The unit cell of the structure. Shape: (3, 3)"""

        zero_offsets = torch.ShortTensor(
            torch.zeros((neighbour_index.shape[1], 3), dtype=torch.short)
        )
        neighbour_offsets = (
            neighbour_offsets if neighbour_offsets is not None else zero_offsets
        )
        self.neighbour_offsets = neighbour_offsets
        """Offsets to apply to neighbour :math:`j` positions to
        account for periodic boundary conditions such that

        .. code-block:: python

            i, j = neighbour_index
            positions_i = positions[i]
            positions_j = positions[j] + neighbour_offsets @ unit_cell
            vectors_i_to_j = positions_j - positions_i

        Shape: (n_edges, 3)"""

        self.labels = labels
        """Additional, user defined labels for the structure/atoms/edges."""

        # we store positions privately, such that we can warn users
        # who access them directly, via the .position property below,
        # in case they naïvely use them to calculate neighbour vectors
        # or distances, which will be incorrect for periodic systems.
        self._positions = positions

    @property
    def has_cell(self) -> bool:
        """Whether the structure has a unit cell."""
        return not torch.all(self.cell == 0).item()

    @property
    def positions(self):
        """The positions of the atoms. Shape: (n_atoms, 3)"""

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
        return self._positions

    @property
    def neighbour_vectors(self) -> torch.Tensor:
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
    def neighbour_distances(self) -> torch.Tensor:
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
        labels = {k: v.to(device) for k, v in self.labels.items()}

        return AtomicGraph(
            Z=self.Z.to(device),  # type: ignore
            positions=self._positions.to(device),  # type: ignore
            neighbour_index=self.neighbour_index.to(device),  # type: ignore
            cell=self.cell.to(device),  # type: ignore
            neighbour_offsets=self.neighbour_offsets.to(device),  # type: ignore
            **labels,
        )

    def __repr__(self) -> str:
        device = self.Z.device
        _dict = dict(
            Z=self.Z,
            positions=self._positions,
            neighbour_index=self.neighbour_index,
            **self.labels,
        )
        return f"AtomicGraph({shape_repr(_dict, sep=', ')}, device={device})"


def convert_to_atomic_graph(
    atoms: Atoms, cutoff: float, labels: list[str] | None = None
) -> AtomicGraph:
    """
    Convert an ASE Atoms object to an AtomicGraph.

    Parameters
    ----------
    atoms : Atoms
        The ASE Atoms object.
    cutoff : float
        The cutoff distance for neighbour finding.
    labels : list[str]
        The names of any additional labels to include in the graph.
        These must be present in either the `atoms.info`
        or `atoms.arrays` dict. If not provided, all possible labels
        will be included.
    """

    # in all this, we need to ensure we move from numpy float64 to
    # torch.float (which is float32 by default)

    labels_dict = extract_information(atoms, labels)

    i, j, offsets = neighbor_list("ijS", atoms, cutoff)

    return AtomicGraph(
        Z=torch.ShortTensor(atoms.numbers),
        positions=torch.FloatTensor(atoms.positions),
        neighbour_index=torch.LongTensor(np.vstack([i, j])),
        cell=torch.FloatTensor(atoms.cell.array),
        neighbour_offsets=torch.ShortTensor(offsets),
        **labels_dict,
    )


def extract_information(
    atoms: Atoms, labels: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """
    Extract any additional information from the atoms object.

    Parameters
    ----------
    atoms : Atoms
        The ASE Atoms object.
    labels : list[str]
        The names of any additional labels to include in the graph.
        These must be present in either the `atoms.info`
        or `atoms.arrays` dict. If not provided, all possible labels
        will be included.
    """

    add_all = labels is None
    labels = labels or []

    labels_dict = {}

    to_ignore = ["numbers", "positions", "cell", "pbc"]
    other_ignored = []

    for info_dict in [atoms.info, atoms.arrays]:
        for key, value in info_dict.items():
            if not add_all and key not in labels:
                continue
            if key in to_ignore:
                continue

            t = as_possible_tensor(value)

            if t is None:
                if add_all:
                    other_ignored.append(key)
                    continue
                else:
                    raise ValueError(
                        f"Label '{key}' is not a tensor and cannot be included."
                    )

            labels_dict[key] = t.float()

    if other_ignored:
        warnings.warn(
            f"The following labels are not tensors and will be ignored: "
            f"{', '.join(other_ignored)}",
            stacklevel=2,
        )
    return labels_dict


def as_possible_tensor(value: object) -> torch.Tensor | None:
    """
    Convert a value to a tensor if possible.

    Parameters
    ----------
    value : object
        The value to convert.
    """

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)

    if isinstance(value, (int, float)):
        return torch.tensor([value])

    try:
        return torch.tensor(value)

    except Exception:
        return None


def convert_to_atomic_graphs(
    structures: Iterable[Atoms],
    cutoff: float,
    labels: list[str] | None = None,
) -> list[AtomicGraph]:
    """
    Convert a collection of ASE `Atoms` into a list of `AtomicGraph`s.

    Parameters
    ----------
    atoms : Iterable[Atoms]
        The ASE Atoms objects.
    cutoff : float
        The cutoff distance for neighbour finding.
    labels : list[str]
        The names of any additional labels to include in the graph.
        These must be present in either the `atoms.info`
        or `atoms.arrays` dict. If not provided, all possible labels
        will be included.
    """
    return [convert_to_atomic_graph(s, cutoff, labels) for s in structures]
