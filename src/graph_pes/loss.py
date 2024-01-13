from __future__ import annotations

from typing import Callable

import torch
from graph_pes.data import AtomicGraphBatch
from graph_pes.transform import Identity, Transform
from torch import Tensor, nn


class Loss(nn.Module):
    r"""
    Measure the discrepancy between predictions and labels.

    Often, it is convenient to apply some well known loss function,
    e.g. `MSELoss`,  to a transformed version of the predictions and labels,
    e.g. normalisation, such that the loss value takes on "nice" values,
    and that the resulting gradients and parameter updates are well-behaved.

    :class:`Loss`'s in `graph-pes` are thus lightweight wrappers around:
    * an (optional) pre-transform, :math:`T`
    * a loss metric, :math:`M`.

    Calculating the loss value, :math:`\mathcal{L}`, amounts to the following:

    .. math::
        \mathcal{L} = M\left( T(\hat{P}), T(P) \right)

    where :math:`\hat{P}` are the predictions, and :math:`P` are the labels.

    You can create a weighted, multi-component loss by using the `+` and `*`
    operators on :class:`Loss` instances. For example:

    .. code-block:: python

            Loss("energy") * 10 + Loss("forces")

    will create a loss function that is the sum of a 10x weighted energy loss
    and a force loss.

    Parameters
    ----------
    property_key
        The name of the property to apply transformations and loss metrics to.
    metric
        The loss metric to use.
    transform
        The transform to apply to the predictions and labels.
    """

    def __init__(
        self,
        property_key: str,
        metric: Callable[[Tensor, Tensor], Tensor] | None = None,
        transform: Transform | None = None,
    ):
        super().__init__()
        self.property_key = property_key
        self.metric = MAE() if metric is None else metric
        self.transform = transform or Identity()
        self.transform.trainable = False

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        graphs: AtomicGraphBatch,
    ) -> torch.Tensor:
        """
        Computes the loss value.

        Parameters
        ----------
        predictions
            The predictions from the model.
        graphs
            The graphs containing the labels.
        """
        P_hat = predictions[self.property_key]
        P_true = graphs.get_labels(self.property_key)

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
        """
        Compute the metric as applied directly to the predictions and labels.

        Parameters
        ----------
        predictions
            The predictions from the model.
        graphs
            The graphs containing the labels.
        """

        return self.metric(
            predictions[self.property_key],
            graphs.get_labels(self.property_key),
        )

    def fit_transform(self, graphs: AtomicGraphBatch):
        """
        Fit the transform to the labels.

        Parameters
        ----------
        graphs
            The graphs containing the labels.
        """

        self.transform.fit(graphs.get_labels(self.property_key), graphs)

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

    def __mul__(self, other: float) -> WeightedLoss:
        return WeightedLoss([self], [other])

    def __rmul__(self, other: float) -> WeightedLoss:
        return WeightedLoss([self], [other])

    def __add__(self, other: Loss | WeightedLoss) -> WeightedLoss:
        if isinstance(other, Loss):
            return WeightedLoss([self, other], [1, 1])
        elif isinstance(other, WeightedLoss):
            return WeightedLoss([self] + other.losses, [1] + other.weights)
        else:
            raise TypeError(f"Cannot add Loss and {type(other)}")

    def __radd__(self, other: Loss | WeightedLoss) -> WeightedLoss:
        return self.__add__(other)


class WeightedLoss(torch.nn.Module):
    r"""
    A lightweight wrapper around a collection of weighted losses.

    Creation can be done in two ways:

    1. Directly, by passing a list of losses and weights.
    2. Via `+` and `*` operators

    Hence:

    .. code-block:: python

        WeightedLoss([Loss("energy"), Loss("forces")], [10, 1])
        # is equivalent to
        10 * Loss("energy") + 1 * Loss("forces")

    Parameters
    ----------
    losses
        The collection of losses to aggregate.
    weights
        The weights to apply to each loss.
    """

    def __init__(
        self,
        losses: list[Loss],
        weights: list[float] | None = None,
    ):
        super().__init__()
        self.losses: list[Loss] = nn.ModuleList(losses)  # type: ignore
        self.weights = weights or [1.0] * len(losses)

    def __add__(self, other: WeightedLoss) -> WeightedLoss:
        return WeightedLoss(
            self.losses + other.losses, self.weights + other.weights
        )


class RMSE(torch.nn.MSELoss):
    r"""
    Root mean squared error metric:

    .. math::
        \sqrt{ \frac{1}{N} \sum_i^N \left( \hat{P}_i - P_i \right)^2 }
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return (super().forward(input, target)).sqrt()


class MAE(torch.nn.L1Loss):
    r"""
    Mean absolute error metric:

    .. math::
        \frac{1}{N} \sum_i^N \left| \hat{P}_i - P_i \right|
    """

    def __init__(self):
        super().__init__()
