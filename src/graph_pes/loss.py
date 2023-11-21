from __future__ import annotations

import torch
from graph_pes.data import AtomicGraphBatch
from graph_pes.transform import Chain, Identity, Transform
from torch import nn


class Loss(nn.Module):
    r"""
    Ultimately, a loss function compares a prediction to a ground truth label,
    and returns a scalar quantifying how good the prediction is.

    A :class:`graph_pes.Loss` optionally applies a series of pre-transforms
    to the predictions and labels before applying a metric:

    .. math::
        \begin{align*}
        P^\prime &= \text{transform}(P) \\
        L &= \text{metric}(P^\prime, P_{\text{true}})
        \end{align*}
    """

    def __init__(
        self,
        property_key: str,
        weight: float = 1.0,
        metric: nn.Module | None = None,
        transform: Transform | None = None,
        transforms: list[Transform] | None = None,
    ):
        super().__init__()
        self.property_key = property_key
        self.weight = weight
        self.metric = nn.MSELoss() if metric is None else metric

        if transform is not None and transforms is not None:
            raise ValueError("Cannot pass both transform and transforms")
        if transform is not None:
            self.transform = transform
        elif transforms is not None:
            self.transform = Chain(transforms)
        else:
            self.transform = Identity()
        self.transform.trainable = False

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        graphs: AtomicGraphBatch,
    ) -> torch.Tensor:
        P_hat = predictions[self.property_key]
        P_true = graphs.labels[self.property_key]

        # apply transforms
        P_hat_prime = self.transform(P_hat, graphs)
        P_true_prime = self.transform(P_true, graphs)

        # compute loss
        return self.metric(P_hat_prime, P_true_prime)

    def raw(
        self,
        predictions: dict[str, torch.Tensor],
        graphs: AtomicGraphBatch,
    ) -> torch.Tensor:
        P_hat = predictions[self.property_key]
        P_true = graphs.labels[self.property_key]

        # compute loss without transforms
        return self.metric(P_hat, P_true)

    def fit_transform(self, graphs: AtomicGraphBatch):
        self.transform.fit_to_source(graphs.labels[self.property_key], graphs)

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


class RMSE(torch.nn.MSELoss):
    def __init__(self, *args, eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("eps", torch.scalar_tensor(eps))

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (super().forward(input, target) + self.eps).sqrt()
