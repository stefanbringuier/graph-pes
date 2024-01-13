from typing import Union

from jaxtyping import Shaped
from torch import Tensor

from .atomic_graph import (
    AtomicGraph,
    convert_to_atomic_graph,
    convert_to_atomic_graphs,
)
from .batching import AtomicDataLoader, AtomicGraphBatch, sum_per_structure
from .utils import random_split

LocalProperty = Shaped[Tensor, "N ..."]
GlobalProperty = Union[Tensor, Shaped[Tensor, "B ..."]]

__all__ = [
    "AtomicDataLoader",
    "AtomicGraph",
    "AtomicGraphBatch",
    "convert_to_atomic_graph",
    "convert_to_atomic_graphs",
    "random_split",
    "sum_per_structure",
    "GlobalProperty",
    "LocalProperty",
]
