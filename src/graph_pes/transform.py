from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from graph_pes.data import AtomicGraph, AtomicGraphBatch, sum_per_structure
from graph_pes.nn import PerSpeciesParameter
from torch import nn


class Transform(nn.Module, ABC):
    r"""
    Transforms data from one space to another.

    .. math::
        \mathbf{x}^\prime = T(\mathbf{x})
    """

    def __init__(self, trainable: bool = True):
        super().__init__()
        self.trainable = trainable

    def _parameter(self, x) -> nn.Parameter:
        """get an optionally trainable parameter"""
        return nn.Parameter(x, requires_grad=self.trainable)

    def _per_species_parameter(
        self,
        zs: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        default: float = 0.0,
    ) -> PerSpeciesParameter:
        """get an optionally trainable per-species parameter"""
        return PerSpeciesParameter(zs, values, default, self.trainable)

    @abstractmethod
    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        """implements the forward transformation"""

    @abstractmethod
    def inverse(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        """implements the inverse transformation"""

    def fit_to_source(self, data: torch.Tensor, graphs: AtomicGraphBatch):
        """fits the transform to data in the source space"""
        pass

    def fit_to_target(self, data: torch.Tensor, graphs: AtomicGraphBatch):
        """fits the transform to data in the target space"""
        pass


class Identity(Transform):
    def forward(self, x, graph):
        return x

    def inverse(self, x, graph):
        return x


class Chain(Transform):
    def __init__(self, transforms: list[Transform], trainable: bool = True):
        super().__init__(trainable)
        for t in transforms:
            t.trainable = trainable
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, graph):
        for t in self.transforms:
            x = t(x, graph)
        return x

    def inverse(self, x, graph):
        for t in reversed(self.transforms):
            t: Transform
            x = t.inverse(x, graph)
        return x

    def fit_to_source(self, data: torch.Tensor, graphs: AtomicGraphBatch):
        for t in self.transforms:  # type: ignore
            t: Transform
            t.fit_to_source(data, graphs)
            data = t(data, graphs)

    def fit_to_target(self, data: torch.Tensor, graphs: AtomicGraphBatch):
        for t in reversed(self.transforms):  # type: ignore
            t: Transform
            t.fit_to_target(data, graphs)
            data = t.inverse(data, graphs)


def is_local_property(x, graph):
    return len(x.shape) and x.shape[0] == graph.n_atoms


# class PerSpeciesTransform(Transform, ABC):
#     r"""
#     Uses a per-species parameter to transform a per-structure/per-atom property.
#     """

#     def __init__(
#         self,
#         trainable: bool = True,
#         values: PerSpeciesParameter | None = None,
#         default: float = 0.0,
#     ):
#         super().__init__(trainable=trainable)

#         if values is not None:
#             values.values.requires_grad = trainable
#         else:
#             values = self._per_species_parameter(default=default)
#         self.values = values

#     @staticmethod
#     def is_local_property(x, graph):
#         return len(x.shape) and x.shape[0] == graph.n_atoms

#     def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
#         values = self.values[graph.Z]
#         if self.is_local_property(x, graph):
#             return self.per_species_op(x, values)
#         else:
#             return self.per_structure_op(x, values, graph)

#     @abstractmethod
#     def per_species_op(
#         self, x: torch.Tensor, values: torch.Tensor
#     ) -> torch.Tensor:
#         """implements the per-species forward transformation"""

#     @abstractmethod
#     def per_structure_op(
#         self, x: torch.Tensor, values: torch.Tensor, graph: AtomicGraph
#     ) -> torch.Tensor:
#         """implements the per-structure forward transformation"""


class PerSpeciesOffset(Transform):
    r"""
    adds a per-species offset to the input
    """

    def __init__(
        self, trainable: bool = True, offsets: PerSpeciesParameter | None = None
    ):
        super().__init__(trainable=trainable)
        if offsets is not None:
            offsets.values.requires_grad = trainable
        else:
            offsets = self._per_species_parameter(default=0.0)
        self.offsets = offsets
        self.op = torch.add

    def _perform_op(
        self, x: torch.Tensor, graph: AtomicGraph, op: Callable
    ) -> torch.Tensor:
        offsets = self.offsets[graph.Z]
        # if we have a total property, we need to sum offsets over the structure
        if not is_local_property(x, graph):
            offsets = sum_per_structure(offsets, graph)
        return op(x, offsets)

    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return self._perform_op(x, graph, self.op)

    def inverse(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return self._perform_op(x, graph, self.inverse_op)

    @property
    def inverse_op(self):
        if self.op == torch.add:
            return torch.sub
        elif self.op == torch.sub:
            return torch.add
        else:
            raise NotImplementedError

    def guess_offsets(
        self, x: torch.Tensor, graphs: AtomicGraphBatch
    ) -> PerSpeciesParameter:
        """guesses the offsets from the data"""
        if is_local_property(x, graphs):
            # fit to mean per species
            zs = torch.unique(graphs.Z)
            offsets = torch.zeros(len(zs))
            for idx, z in enumerate(zs):
                offsets[idx] = x[graphs.Z == z].mean()

        else:
            # do linear regression to get best guess at offsets
            zs = torch.unique(graphs.Z)
            N = torch.zeros(graphs.n_structures, len(zs))

            for idx, z in enumerate(zs):
                N[:, idx] = sum_per_structure((graphs.Z == z).float(), graphs)
            offsets = torch.linalg.lstsq(N, x).solution

        return self._per_species_parameter(zs, offsets)

    def fit_to_source(self, data: torch.Tensor, graphs: AtomicGraphBatch):
        self.offsets = self.guess_offsets(data, graphs)
        self.op = torch.sub

    def fit_to_target(self, data: torch.Tensor, graphs: AtomicGraphBatch):
        self.offsets = self.guess_offsets(data, graphs)
        self.op = torch.add

    def __repr__(self):
        return self.offsets.__repr__().replace(
            self.offsets.__class__.__name__, self.__class__.__name__
        )


def sum_scale_per_structure(
    scale: torch.Tensor, graph: AtomicGraph
) -> torch.Tensor:
    sum = sum_per_structure(scale**2, graph)
    return torch.sqrt(sum)


class PerSpeciesScale(Transform):
    r"""
    scales the input by a per-species factor
    """

    def __init__(
        self, trainable: bool = True, scales: PerSpeciesParameter | None = None
    ):
        super().__init__(trainable=trainable)
        if scales is not None:
            scales.values.requires_grad = trainable
        else:
            scales = self._per_species_parameter(default=1.0)
        self.scales = scales
        self.op = torch.mul

    def _perform_op(
        self, x: torch.Tensor, graph: AtomicGraph, op: Callable
    ) -> torch.Tensor:
        scales = self.scales[graph.Z]

        # if we have a total property, we need to sum scales over the structure
        # in accordance with the central limit theorem
        if not is_local_property(x, graph):
            # TODO implement this at some point
            scales = sum_per_structure(scales**2, graph) ** 0.5

        if len(x.shape) > 1:
            scales = scales.view(-1, 1)
        return op(x, scales)

    def forward(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return self._perform_op(x, graph, self.op)

    def inverse(self, x: torch.Tensor, graph: AtomicGraph) -> torch.Tensor:
        return self._perform_op(x, graph, self.inverse_op)

    @property
    def inverse_op(self):
        if self.op == torch.mul:
            return torch.div
        elif self.op == torch.div:
            return torch.mul
        else:
            raise NotImplementedError

    def guess_scales(self, x: torch.Tensor, graphs: AtomicGraphBatch):
        """guesses the scales from the data"""
        if is_local_property(x, graphs):
            # fit to mean per species
            zs = torch.unique(graphs.Z)
            scales = torch.zeros(len(zs))
            for idx, z in enumerate(zs):
                scales[idx] = x[graphs.Z == z].std()

        else:
            # this is very tricky, for now just get a single
            # scale for all species
            atoms_per_structure = graphs.ptr[1:] - graphs.ptr[:-1]
            scale = (x / atoms_per_structure**0.5).std()
            zs = torch.unique(graphs.Z)
            scales = torch.ones(len(zs)) * scale

        return self._per_species_parameter(zs, scales, default=1.0)

    def fit_to_source(self, data: torch.Tensor, graphs: AtomicGraphBatch):
        self.scales = self.guess_scales(data, graphs)
        self.op = torch.div

    def fit_to_target(self, data: torch.Tensor, graphs: AtomicGraphBatch):
        self.scales = self.guess_scales(data, graphs)
        self.op = torch.mul

    def __repr__(self):
        return self.scales.__repr__().replace(
            self.scales.__class__.__name__, self.__class__.__name__
        )
