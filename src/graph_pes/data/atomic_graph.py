from __future__ import annotations

import warnings
from typing import Iterable

import ase
import numpy as np
import torch
from ase.neighborlist import neighbor_list
from jaxtyping import Float, Int
from torch import Tensor

from ..util import as_possible_tensor, shape_repr


class AtomicGraph:
    """
    A graph representation of an atomic structure, compatible with
    PyTorch/Geometric/Lightning.

    .. note::
        We use `jaxtyping <https://docs.kidger.site/jaxtyping/>`_ notation
        to annotate all tensor shapes. Each AtomicGraph has :math:`N` atoms
        and :math:`E` edges

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
    labels
        Additional, user defined labels for the structure/atoms/edges.
    """

    def __init__(
        self,
        Z: Int[Tensor, "N"],
        positions: Float[Tensor, "N 3"],
        neighbour_index: Int[Tensor, "2 E"],
        cell: Float[Tensor, "3 3"],
        neighbour_offsets: Int[Tensor, "E 3"],
        **labels: Tensor,
    ):
        self.Z = Z
        self.neighbour_index = neighbour_index
        self.cell = cell
        self.neighbour_offsets = neighbour_offsets
        self.labels = labels

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
        **labels: torch.Tensor,
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
        labels
            Additional, user defined labels for the structure/atoms/edges.
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
            **labels,
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

    # in all this, we need to ensure we move from numpy float64 to
    # torch.float (which is float32 by default)

    labels_dict = extract_information(structure, labels)
    i, j, offsets = neighbor_list("ijS", structure, cutoff)
    return AtomicGraph(
        Z=torch.ShortTensor(structure.numbers),
        positions=torch.FloatTensor(structure.positions),
        neighbour_index=torch.LongTensor(np.vstack([i, j])),
        cell=torch.FloatTensor(structure.cell.array),
        neighbour_offsets=torch.ShortTensor(offsets),
        **labels_dict,
    )


def extract_information(
    atoms: ase.Atoms, labels: list[str] | None = None
) -> dict[str, torch.Tensor]:
    """
    Extract any additional information from the atoms object.

    Parameters
    ----------
    atoms
        The ASE Atoms object.
    labels
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

    possible_matches = set(atoms.info.keys()) | set(atoms.arrays.keys())
    if any(label not in possible_matches for label in labels):
        raise KeyError(
            f"Labels {labels} are not present in the atoms object."
            f"Possible matches are: {possible_matches}"
        )

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


def convert_to_atomic_graphs(
    structures: Iterable[ase.Atoms],
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
    return [convert_to_atomic_graph(s, cutoff, labels) for s in structures]
