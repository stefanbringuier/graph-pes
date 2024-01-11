from __future__ import annotations

import warnings
from typing import Sequence

import torch
from jaxtyping import Float, Int, Shaped
from torch import Tensor
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.utils import scatter

from ..util import pairs
from .atomic_graph import AtomicGraph


def sum_per_structure(
    x: torch.Tensor, graph: AtomicGraph | AtomicGraphBatch
) -> torch.Tensor:
    """
    Sum a per-atom property into a per-structure property, respecting
    any batch structure of the input graph.

    Parameters
    ----------
    x
        The per-atom property to sum.
    graph
        The graph to sum over.
    """
    if isinstance(graph, AtomicGraphBatch):
        # we have more than one structure: sum over local energies to
        # get a total energy for each structure
        return scatter(x, graph.batch, dim=0, reduce="sum")
    else:
        # we only have one structure: sum over all the atoms
        return x.sum()


class AtomicDataLoader(TorchDataLoader):
    r"""
    A data loader for merging :class:`AtomicGraph` objects into
    :class:`AtomicGraphBatch` objects.

    Parameters
    ----------
    dataset: Sequence[AtomicGraph]
        The dataset to load.
    batch_size: int
        The batch size.
    shuffle: bool
        Whether to shuffle the dataset.
    **kwargs:
        Additional keyword arguments are passed to the underlying
        :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Sequence[AtomicGraph],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            warnings.warn(
                "GraphPES uses a custom collate_fn (`collate_atomic_graphs`), "
                "are you sure you want to override this?",
                stacklevel=2,
            )

        collate_fn = kwargs.pop("collate_fn", _collate_atomic_graphs)

        super().__init__(
            dataset,  # type: ignore
            batch_size,
            shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )


class AtomicGraphBatch(AtomicGraph):
    """
    A disconnected graph representing multiple atomic structures.

    .. note::
        We use `jaxtyping <https://docs.kidger.site/jaxtyping/>`_ notation
        to annotate all tensor shapes. Each AtomicGraphBatch contains
        :math:`B` structures, with a total of :math:`Nb` atoms
        and :math:`Eb` edges.

    Parameters
    ----------
    Z
        The atomic numbers of the atoms.
    positions
        The positions of the atoms.
    neighbour_index
        An edge list specifying neighbouring atoms :math:`j`
        for each central atom :math:`i`.
    cells
        The unit cells of the structure.
    neighbour_offsets
        An edge list specifying the periodic offset of neighbouring atoms
        :math:`j` for each central atom :math:`i`.
    batch
        The structure index of each atom, e.g. `[0, 0, 0, 1, 1, 1, 1, 2, 2, 2]`.
        In this case, the batch contains 3 structures, with 3, 4 and 3 atoms
    ptr
        A pointer to the start and end of each structure in the batch.
        In the above example, this would be `[0, 3, 7, 10]`.
    labels
        Additional, user defined labels for the structure/atoms/edges.
    """

    def __init__(
        self,
        Z: Int[Tensor, "Nb"],
        positions: Float[Tensor, "Nb 3"],
        neighbour_index: Int[Tensor, "2 Eb"],
        cells: Float[Tensor, "B 3 3"],
        neighbour_offsets: Int[Tensor, "Eb 3"],
        batch: Int[Tensor, "Nb"],
        ptr: Int[Tensor, "B+1"],
        atom_labels: dict[str, Shaped[Tensor, "Nb ..."]] | None = None,
        edge_labels: dict[str, Shaped[Tensor, "Eb ..."]] | None = None,
        structure_labels: dict[str, Shaped[Tensor, "B, ..."]] | None = None,
    ):
        super().__init__(
            Z=Z,
            positions=positions,
            neighbour_index=neighbour_index,
            cell=cells,
            neighbour_offsets=neighbour_offsets,
            atom_labels=atom_labels,
            edge_labels=edge_labels,
            structure_labels=structure_labels,
        )

        self.batch = batch
        self.ptr = ptr

    @property
    def neighbour_vectors(self) -> torch.Tensor:
        """
        The vectors from central atoms :math:`i` to neighbours :math:`j`,
        respecting any periodic boundary conditions.
        """

        # more involved than the single structure case, because each
        # cell is different

        # easy case of no cell:
        i, j = self.neighbour_index
        if not self.has_cell:
            return self._positions[j] - self._positions[i]

        # otherwise calculate offsets on a per-structure basis
        actual_offsets = torch.zeros((self.neighbour_index.shape[1], 3))
        for batch, (start, end) in enumerate(pairs(self.ptr)):  # type: ignore
            mask = (i >= start) & (i < end)
            actual_offsets[mask] = (
                self.neighbour_offsets[mask].float() @ self.cell[batch]
            )

        return self._positions[j] - self._positions[i] + actual_offsets

    @property
    def n_structures(self) -> int:
        """The number of structures within this batch."""
        return self.ptr.shape[0] - 1

    @property
    def structure_sizes(self) -> torch.Tensor:
        """The number of atoms in each structure within this batch."""
        return self.ptr[1:] - self.ptr[:-1]

    def to(self, device: torch.device | str) -> AtomicGraphBatch:
        def tensor_dict_to_device(
            d: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return {k: v.to(device) for k, v in d.items()}

        return AtomicGraphBatch(
            Z=self.Z.to(device),
            positions=self._positions.to(device),
            neighbour_index=self.neighbour_index.to(device),
            batch=self.batch.to(device),
            cells=self.cells.to(device),
            neighbour_offsets=self.neighbour_offsets.to(device),
            ptr=self.ptr.to(device),
            structure_labels=tensor_dict_to_device(self.structure_labels),
            atom_labels=tensor_dict_to_device(self.atom_labels),
            edge_labels=tensor_dict_to_device(self.edge_labels),
        )

    @property
    def cells(self) -> torch.Tensor:
        return self.cell

    @classmethod
    def from_graphs(cls, graphs: list[AtomicGraph]) -> AtomicGraphBatch:
        return _collate_atomic_graphs(graphs)


def _collate_atomic_graphs(graphs: list[AtomicGraph]) -> AtomicGraphBatch:
    # easy properties: just cat these together
    Z = torch.cat([g.Z for g in graphs])
    positions = torch.cat([g._positions for g in graphs])
    cells = torch.stack([g.cell for g in graphs])
    neighbour_offsets = torch.cat([g.neighbour_offsets for g in graphs])

    # per-atom and per-edge labels are concatenated along the first axis
    per_atom_labels = {
        k: torch.cat([g.atom_labels[k] for g in graphs])
        for k in graphs[0].atom_labels
    }
    per_edge_labels = {
        k: torch.cat([g.edge_labels[k] for g in graphs])
        for k in graphs[0].edge_labels
    }

    # per-structure labels are concatenated along a new batch axis (0)
    per_structure_labels = {
        k: torch.stack([g.structure_labels[k] for g in graphs])
        for k in graphs[0].structure_labels
    }

    # standard way to calculate the batch and ptr properties
    batch = torch.cat(
        [torch.full((g.Z.shape[0],), i) for i, g in enumerate(graphs)]
    )
    ptr = torch.tensor([0] + [g.Z.shape[0] for g in graphs]).cumsum(dim=0)

    # use the ptr to increment the neighbour index appropriately
    neighbour_index = torch.cat(
        [g.neighbour_index + ptr[i] for i, g in enumerate(graphs)], dim=1
    )

    return AtomicGraphBatch(
        Z=Z,
        positions=positions,
        neighbour_index=neighbour_index,
        cells=cells,
        neighbour_offsets=neighbour_offsets,
        batch=batch,
        ptr=ptr,
        atom_labels=per_atom_labels,
        edge_labels=per_edge_labels,
        structure_labels=per_structure_labels,
    )
