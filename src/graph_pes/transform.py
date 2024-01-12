from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
from graph_pes.data import AtomicGraph, AtomicGraphBatch, sum_per_structure
from graph_pes.nn import PerSpeciesParameter
from torch import Tensor, nn


class Transform(nn.Module, ABC):
    r"""
    :math:`T: \mathbb{R}^n \rightarrow_{\mathcal{G}} \mathbb{R}^n`

    Abstract base class for shape-preserving transformations of
    data, conditioned on an :class:`AtomicGraph <graph_pes.data.AtomicGraph>`,
    :math:`\mathcal{G}`.

    Subclasses should implement :meth:`forward`, :meth:`inverse`,
    and optionally :meth:`fit_to_source` and :meth:`fit_to_target`.

    :meth:`_parameter` and :meth:`_per_species_parameter` are provided
    as convenience methods for creating parameters that respect the
    `trainable` flag.

    Parameters
    ----------
    trainable
        Whether the transform should be trainable.
    """

    def __init__(self, trainable: bool = True):
        super().__init__()
        self.trainable = trainable

    @abstractmethod
    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Implements the forward transformation, :math:`y = T(x, \mathcal{G})`.

        Parameters
        ----------
        x
            The input data.
        graph
            The graph to condition the transformation on.

        Returns
        -------
        y: Tensor
            The transformed data.
        """

    @abstractmethod
    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        r"""
        Implements the inverse transformation,
        :math:`x = T^{-1}(y, \mathcal{G})`.

        Parameters
        ----------
        x
            The input data.
        graph
            The graph to condition the inverse transformation on.

        Returns
        -------
        x: Tensor
            The inversely-transformed data.
        """

    def fit_to_source(self, data: Tensor, graphs: AtomicGraphBatch):
        """
        Fits the transform to data in the source space, :math:`x`.

        Parameters
        ----------
        data
            The data, :math:`x`, to fit to.
        graphs
            The graphs to condition the transformation on.
        """

    def fit_to_target(self, data: Tensor, graphs: AtomicGraphBatch):
        """
        Fits the transform to data in the target space, :math:`y`.

        Parameters
        ----------
        data
            The data, :math:`y`, to fit to.
        graphs
            The graphs to condition the inverse transformation on.
        """

    def _parameter(self, x: Tensor) -> nn.Parameter:
        """Wrap `x` in an optionally trainable parameter."""
        return nn.Parameter(x, requires_grad=self.trainable)

    def _per_species_parameter(
        self,
        generator: Callable[[tuple[int, int]], Tensor] | float = 0.0,
    ) -> PerSpeciesParameter:
        """Generate an (optionally trainable) per-species parameter."""
        return PerSpeciesParameter.of_dim(
            1, requires_grad=self.trainable, generator=generator
        )


class Identity(Transform):
    """The identity transform, provided for convenience: :math:`T(x) = x`."""

    def __init__(self):
        super().__init__(trainable=False)

    def forward(self, x, graph):
        return x

    def inverse(self, x, graph):
        return x


class Chain(Transform):
    r"""
    A chain of transformations, :math:`T_n \circ \dots \circ T_2 \circ T_1`.

    The forward transformation is applied sequentially from left to right,
    :math:`y = T_n \circ \dots \circ T_2 \circ T_1(x, \mathcal{G})`.

    The inverse transformation is applied sequentially from right to left,
    :math:`x = T_1^{-1} \circ T_2^{-1} \circ \dots
    \circ T_n^{-1}(y, \mathcal{G})`.

    Parameters
    ----------
    transforms
        The transformations to chain together.
    trainable
        Whether the chain should be trainable.
    """

    def __init__(self, transforms: list[Transform], trainable: bool = True):
        super().__init__(trainable)
        for t in transforms:
            t.trainable = trainable
        self.transforms: list[Transform] = nn.ModuleList(transforms)  # type: ignore

    def forward(self, x, graph):
        for t in self.transforms:
            x = t(x, graph)
        return x

    def inverse(self, x, graph):
        for t in reversed(self.transforms):
            x = t.inverse(x, graph)
        return x

    def fit_to_source(self, data: Tensor, graphs: AtomicGraphBatch):
        for t in self.transforms:
            t.fit_to_source(data, graphs)
            data = t(data, graphs)

    def fit_to_target(self, data: Tensor, graphs: AtomicGraphBatch):
        for t in reversed(self.transforms):
            t.fit_to_target(data, graphs)
            data = t.inverse(data, graphs)


def is_local_property(x, graph):
    return len(x.shape) and x.shape[0] == graph.n_atoms


class PerSpeciesOffset(Transform):
    r"""
    Maintains an (optionally trainable) per-species offset, :math:`o`.

    This transforms per-atom properties, :math:`x`, of a structure,
    :math:`\mathcal{G}`, according to :math:`y_i = x_i + o_{Z_i}`.

    This transforms per-structure properties, :math:`x`, of a structure,
    :math:`\mathcal{G}`, according to :math:`y = x + \sum_{i=1}^{N} o_{Z_i}`.

    Parameters
    ----------
    trainable
        Whether the offset should be trainable.
    offsets
        The offsets to use. If `None`, the offsets are initialized to zero.
    """

    def __init__(
        self, trainable: bool = True, offsets: PerSpeciesParameter | None = None
    ):
        super().__init__(trainable=trainable)
        if offsets is not None:
            offsets.requires_grad = trainable
        else:
            offsets = self._per_species_parameter(0.0)

        self.offsets = offsets
        self.op = torch.add

    def _perform_op(
        self, x: Tensor, graph: AtomicGraph, op: Callable
    ) -> Tensor:
        offsets = self.offsets[graph.Z]
        # if we have a total property, we need to sum offsets over the structure
        if not is_local_property(x, graph):
            offsets = sum_per_structure(offsets, graph)
        return op(x, offsets)

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return self._perform_op(x, graph, self.op)

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
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
        self, x: Tensor, graphs: AtomicGraphBatch
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

    def fit_to_source(self, data: Tensor, graphs: AtomicGraphBatch):
        """
        Calculates the offsets, :math:`o`, such that the mean of the
        transformed data is zero.

        Parameters
        ----------
        data
            The data to fit to.
        graphs
            The graphs to condition the transformation on.
        """
        self.offsets = self.guess_offsets(data, graphs)
        self.op = torch.sub

    def fit_to_target(self, data: Tensor, graphs: AtomicGraphBatch):
        self.offsets = self.guess_offsets(data, graphs)
        self.op = torch.add

    def __repr__(self):
        return self.offsets.__repr__().replace(
            self.offsets.__class__.__name__, self.__class__.__name__
        )


def sum_scale_per_structure(scale: Tensor, graph: AtomicGraph) -> Tensor:
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
        self, x: Tensor, graph: AtomicGraph, op: Callable
    ) -> Tensor:
        scales = self.scales[graph.Z]

        # if we have a total property, we need to sum scales over the structure
        # in accordance with the central limit theorem
        if not is_local_property(x, graph):
            # TODO implement this at some point
            scales = sum_per_structure(scales**2, graph) ** 0.5

        if len(x.shape) > 1:
            scales = scales.view(-1, 1)
        return op(x, scales)

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return self._perform_op(x, graph, self.op)

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return self._perform_op(x, graph, self.inverse_op)

    @property
    def inverse_op(self):
        if self.op == torch.mul:
            return torch.div
        elif self.op == torch.div:
            return torch.mul
        else:
            raise NotImplementedError

    def guess_scales(self, x: Tensor, graphs: AtomicGraphBatch):
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

    def fit_to_source(self, data: Tensor, graphs: AtomicGraphBatch):
        self.scales = self.guess_scales(data, graphs)
        self.op = torch.div

    def fit_to_target(self, data: Tensor, graphs: AtomicGraphBatch):
        self.scales = self.guess_scales(data, graphs)
        self.op = torch.mul

    def __repr__(self):
        return self.scales.__repr__().replace(
            self.scales.__class__.__name__, self.__class__.__name__
        )


class Scale(Transform):
    def __init__(self, trainable: bool = True, scale: float = 1.0):
        super().__init__(trainable=trainable)
        self.scale = self._parameter(Tensor(scale))

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x * self.scale

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x / self.scale

    def fit_to_source(self, data: Tensor, graphs: AtomicGraphBatch):
        self.scale = self._parameter(1 / data.std())

    def fit_to_target(self, data: Tensor, graphs: AtomicGraphBatch):
        self.scale = self._parameter(data.std())


class FixedScale(Transform):
    def __init__(self, scale: float):
        super().__init__(trainable=False)
        self.scale = scale

    def forward(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x * self.scale

    def inverse(self, x: Tensor, graph: AtomicGraph) -> Tensor:
        return x / self.scale
