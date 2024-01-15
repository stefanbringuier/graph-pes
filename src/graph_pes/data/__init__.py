from __future__ import annotations

from typing import Union

from jaxtyping import Shaped
from torch import Tensor
from typing_extensions import TypeAlias

from .atomic_graph import (
    AtomicGraph,
    convert_to_atomic_graph,
    convert_to_atomic_graphs,
)
from .batching import AtomicDataLoader, AtomicGraphBatch, sum_per_structure
from .utils import random_split

LocalProperty: TypeAlias = Shaped[Tensor, "N ..."]
"""
Type alias for a local property, i.e. a property that is defined for each atom.
"""
GlobalProperty: TypeAlias = Union[Tensor, Shaped[Tensor, "B ..."]]
"""
Type alias for a global property, i.e. a property that is defined 
for each structure.
"""

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
