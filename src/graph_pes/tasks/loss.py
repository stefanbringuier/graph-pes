from __future__ import annotations
from dataclasses import dataclass

from typing import Callable

import torch

from graph_pes.data import AtomicGraph
from graph_pes.data.batching import AtomicGraphBatch


LossMetric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class RMSELoss(torch.nn.MSELoss):
    def __init__(self, *args, eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("eps", torch.scalar_tensor(eps))

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (super().forward(input, target) + self.eps).sqrt()


@dataclass
class Loss:
    target: str
    metric: LossMetric

    @property
    def name(self) -> str:
        # if metric is a class, we want the class name otherwise we want
        # the function name, all without the word "loss" in it
        return (
            getattr(
                self.metric,
                "__name__",
                self.metric.__class__.__name__,
            )
            .lower()
            .replace("loss", "")
        )

    def __call__(
        self, prediction: torch.Tensor, ground_truth: torch.Tensor
    ) -> torch.Tensor:
        return self.metric(prediction, ground_truth)


class WeightedLoss(Loss):
    weight_fn: Callable[[AtomicGraph | AtomicGraphBatch], float]

    def __init__(
        self,
        target: str,
        metric: LossMetric,
        weight_fn: Callable[[AtomicGraph | AtomicGraphBatch], float] | float,
    ):
        super().__init__(target, metric)

        if not callable(weight_fn):
            self.weight_fn = lambda _: weight_fn
        else:
            self.weight_fn = weight_fn

    def __call__(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        graph: AtomicGraph | AtomicGraphBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.metric(prediction, ground_truth)
        weight = self.weight_fn(graph)
        return value, torch.scalar_tensor(weight, device=value.device)


def default_losses() -> list[Loss]:
    return [
        Loss("energy", RMSELoss()),
        Loss("forces", RMSELoss()),
    ]
