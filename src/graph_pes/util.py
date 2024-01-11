from __future__ import annotations

from typing import Iterator, Sequence, TypeVar

import numpy as np
import torch
from ase.data import chemical_symbols

T = TypeVar("T")


MAX_Z = 118
"""The maximum atomic number in the periodic table."""


def pairs(a: Sequence[T]) -> Iterator[tuple[T, T]]:
    """
    Iterate over pairs of elements in `a`

    Parameters
    ----------
    a: Sequence
        The sequence to iterate over.

    Example
    -------
    >>> list(pairs([1, 2, 3]))
    [(1, 2), (2, 3)]
    """
    return zip(a, a[1:])


def shape_repr(
    dict_of_tensors: dict[str, torch.Tensor], sep: str = ", "
) -> str:
    """
    Generate a string representation of the shapes of the tensors in
    `dict_of_tensors`.

    Parameters
    ----------
    dict_of_tensors: dict[str, torch.Tensor]
        The dictionary of tensors to represent.
    sep: str
        The separator to use between each tensor.

    Returns
    -------
    str
        The string representation of the shapes of the tensors.

    Example
    -------
    >>> shape_repr({'a': torch.zeros(3), 'b': torch.zeros(4)})
    'a=[3], b=[4]'
    """

    def _get_shape(tensor: torch.Tensor) -> str:
        return "[" + ",".join(str(i) for i in tensor.shape) + "]"

    return sep.join(f"{k}={_get_shape(v)}" for k, v in dict_of_tensors.items())


def to_chem_symbol(z: int):
    """
    Convert an atomic number to a chemical symbol.

    Parameters
    ----------
    z: int
        The atomic number.

    Returns
    -------
    str
        The chemical symbol.

    Example
    -------
    >>> to_chem_symbol(1)
    'H'
    >>> to_chem_symbol(118)
    'Og'
    """
    return chemical_symbols[z]


def as_possible_tensor(value: object) -> torch.Tensor | None:
    """
    Convert a value to a tensor if possible.

    Parameters
    ----------
    value
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
