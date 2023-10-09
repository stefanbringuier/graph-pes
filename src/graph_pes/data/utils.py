from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np

__all__ = ["random_split"]

E = TypeVar("E")


def random_split(
    thing: Sequence[E], lengths: Sequence[int], seed: int = 0
) -> list[list[E]]:
    # TODO: raise error if not engouth things
    shuffle = np.random.RandomState(seed=seed).permutation(len(thing))
    ptr = [0, *np.cumsum(lengths)]

    return [
        [thing[i] for i in shuffle[ptr[n] : ptr[n + 1]]]
        for n in range(len(lengths))
    ]
