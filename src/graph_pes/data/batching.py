from __future__ import annotations

import warnings
from typing import Sequence, cast

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.utils import scatter

from .atomic_graph import AtomicGraph


def sum_per_structure(
    property: torch.Tensor, graph: AtomicGraph | AtomicGraphBatch
) -> torch.Tensor:
    if isinstance(graph, AtomicGraphBatch):
        # we have more than one structure: sum over local energies to
        # get a total energy for each structure
        return scatter(property, graph.batch, dim=0, reduce="sum")
    else:
        # we only have one structure: sum over all the atoms
        return property.sum()


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
    Inherits from :class:`AtomicGraph`, and so contains all of the
    same properties, together with the following:

    Additional properties:
    ----------------------
    batch: torch.LongTensor
        The structure index of each atom, e.g. [0, 0, 0, 1, 1, 1, 1, 2, 2, 2].
        In this case, the batch contains 3 structures, with 3, 4 and 3 atoms
        Shape: (n_atoms,)
    ptr: torch.LongTensor
        A pointer to the start and end of each structure in the batch.
        In the above example, this would be [0, 3, 7, 10].
        Shape: (n_structures + 1,)
    n_structures: int
        The number of structures in the batch.
    """

    def __init__(
        self,
        Z: torch.ShortTensor,
        positions: torch.FloatTensor,
        neighbour_index: torch.LongTensor,
        batch: torch.LongTensor,
        cells: torch.FloatTensor,
        neighbour_offsets: torch.ShortTensor,
        ptr: torch.LongTensor,
        **labels: torch.Tensor,
    ):
        super().__init__(
            Z=Z,
            positions=positions,
            neighbour_index=neighbour_index,
            cell=cells,
            neighbour_offsets=neighbour_offsets,
            **labels,
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
        for batch, (start, end) in enumerate(zip(self.ptr[:-1], self.ptr[1:])):
            mask = (i >= start) & (i < end)
            actual_offsets[mask] = (
                self.neighbour_offsets[mask].float() @ self.cell[batch]
            )
        # for batch, (start, end) in enumerate(zip(self.ptr[:-1], self.ptr[1:])):
        #     actual_offsets[start:end] = (
        #         self.neighbour_offsets[start:end].float() @ self.cell[batch]
        #     )

        return self._positions[j] - self._positions[i] + actual_offsets

    @property
    def n_structures(self) -> int:
        return self.ptr.shape[0] - 1

    @property
    def structure_sizes(self) -> torch.Tensor:
        return self.ptr[1:] - self.ptr[:-1]

    def to(self, device: torch.device | str) -> AtomicGraphBatch:
        labels = {k: v.to(device) for k, v in self.labels.items()}

        return AtomicGraphBatch(
            Z=self.Z.to(device),
            positions=self._positions.to(device),
            neighbour_index=self.neighbour_index.to(device),
            batch=self.batch.to(device),
            cells=self.cells.to(device),
            neighbour_offsets=self.neighbour_offsets.to(device),
            ptr=self.ptr.to(device),
            **labels,
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

    # TODO: check that all graphs have the same labels
    labels = {
        k: torch.cat([g.labels[k] for g in graphs]) for k in graphs[0].labels
    }
    if "stress" in labels:
        labels["stress"] = labels["stress"].reshape(-1, 3, 3)

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
        Z=cast(torch.ShortTensor, Z),
        positions=cast(torch.FloatTensor, positions),
        neighbour_index=cast(torch.LongTensor, neighbour_index),
        batch=cast(torch.LongTensor, batch),
        cells=cast(torch.FloatTensor, cells),
        neighbour_offsets=cast(torch.ShortTensor, neighbour_offsets),
        ptr=cast(torch.LongTensor, ptr),
        **labels,
    )
