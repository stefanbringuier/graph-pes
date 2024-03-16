from __future__ import annotations

from graph_pes.core import GraphPESModel

from .painn import PaiNN
from .pairwise import LennardJones, LennardJonesMixture, Morse
from .schnet import SchNet
from .tensornet import TensorNet

__all__ = [
    "PaiNN",
    "LennardJones",
    "SchNet",
    "TensorNet",
    "Morse",
    "LennardJonesMixture",
]

ALL_MODELS: list[type[GraphPESModel]] = [globals()[model] for model in __all__]
