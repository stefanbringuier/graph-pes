"""Defines :class:`AtomicGraph`, the central data object of GraphPES."""

from __future__ import annotations

import warnings

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
        A symmetric edge list specifying neighbouring atoms :math:`j`
        for each central atom :math:`i`. Shape: (2, n_edges)
    neighbour_offsets: torch.Tensor | None
        Offsets to apply to neighbour :math:`j` positions to account for
        periodic boundary conditions such that

        .. code-block:: python

            i, j = neighbour_index
            positions_i = positions[i]
            positions_j = positions[j] + neighbour_offsets
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
        Z: torch.Tensor,
        positions: torch.Tensor,
        neighbour_index: torch.LongTensor,
        neighbour_offsets: torch.Tensor | None = None,
        **labels: torch.Tensor,
    ):
        self.Z = Z
        """Atomic numbers of the atoms. Shape: (`n_atoms`,)"""

        self.neighbour_index = neighbour_index
        """An edge list specifying neighbouring atoms :math:`j`
        for each central atom :math:`i`. Shape: (2, n_edges)"""

        zero_offsets = torch.zeros((neighbour_index.shape[1], 3))
        self.neighbour_offsets = (
            zero_offsets if neighbour_offsets is None else neighbour_offsets
        )
        """Offsets to apply to neighbour :math:`j` positions to account for
        periodic boundary conditions such that

        .. code-block:: python

            i, j = neighbour_index
            neighbour_positions = positions[j] + neighbour_offsets
            neighbour_vectors = neighbour_positions - positions[i]

        Shape: (n_edges, 3)."""

        self.labels = labels
        """Additional, user defined labels for the structure/atoms/edges."""

        # we store positions privately, such that we can warn users
        # who access them directly, via the .position property below,
        # in case they naïvely use them to calculate neighbour vectors
        # or distances, which will be incorrect for periodic systems.
        self._positions = positions

    @property
    def positions(self):
        """The positions of the atoms. Shape: (n_atoms, 3)"""

        # raise a warning if the user accesses the positions directly,
        # since the most likely cause for this is that they are attempting
        # to calculate neighbour vectors/distances as:
        #   neighbour_vectors = positions[j] - positions[i]
        # which will be incorrect for periodic systems.

        warnings.warn(
            "Are you using `AtomicGraph.positions` to calculate "
            "neighbour vectors/distances? Use `AtomicGraph.neighbour_vectors` "
            "and `AtomicGraph.neighbour_distances` instead: for periodic "
            "systems, neighbour_vectors != positions[j] - positions[i]!",
            UserWarning,
        )
        return self._positions

    @property
    def neighbour_vectors(self) -> torch.Tensor:
        """
        The vectors from central atoms :math:`i` to neighbours :math:`j`,
        respecting any periodic boundary conditions.
        """

        i, j = self.neighbour_index
        neighbour_positions = self._positions[j] + self.neighbour_offsets
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
        """Get a copy of this graph on the specified device."""

        labels = {k: v.to(device) for k, v in self.labels.items()}
        return AtomicGraph(
            Z=self.Z.to(device),
            positions=self._positions.to(device),
            neighbour_index=self.neighbour_index.to(device),  # type: ignore
            neighbour_offsets=self.neighbour_offsets.to(device),
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
    neighbour_offsets = torch.tensor(
        offsets @ atoms.cell.array, dtype=torch.float
    )

    return AtomicGraph(
        Z=torch.LongTensor(atoms.numbers),
        positions=torch.tensor(atoms.positions, dtype=torch.float),
        neighbour_index=torch.LongTensor(np.vstack([i, j])),
        neighbour_offsets=neighbour_offsets,
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

    for info_dict in [atoms.info, atoms.arrays]:
        for key, value in info_dict.items():
            if not add_all and not key in labels:
                continue
            if key in to_ignore:
                continue

            t = as_possible_tensor(value)

            if t is None:
                if add_all:
                    warnings.warn(
                        f"Label '{key}' is not a tensor and will be ignored."
                    )
                    continue
                else:
                    raise ValueError(
                        f"Label '{key}' is not a tensor and cannot be included."
                    )

            labels_dict[key] = t

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
    structures: list[Atoms],
    cutoff: float,
    labels: list[str] | None = None,
) -> list[AtomicGraph]:
    return [convert_to_atomic_graph(s, cutoff, labels) for s in structures]
