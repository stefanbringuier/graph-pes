from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi as π
from typing import Any, Callable

import torch
from torch import nn


class DistanceExpansion(nn.Module, ABC):  # TODO- make protocol?
    r"""
    Base class for an expansion function, :math:`\phi(r)` such that:

    .. math::
        r \in \mathbb{R}^1 \quad \rightarrow \quad \phi(r) \in
        \mathbb{R}^{n_\text{features}}

    or, for a batch of distances:

    .. math::
        r \in \mathbb{R}^{n_\text{batch} \times 1} \quad \rightarrow \quad
        \phi(r) \in \mathbb{R}^{n_\text{batch} \times n_\text{features}}

    Parameters
    ----------
    n_features : int
        The number of features to expand into.
    cutoff : float
        The cutoff radius.
    """

    def __init__(self, n_features: int, cutoff: float):
        super().__init__()
        self.n_features = n_features
        self.cutoff = cutoff

    @abstractmethod
    def expand(self, r: torch.Tensor) -> torch.Tensor:
        r"""
        Perform the expansion.

        Parameters
        ----------
        r : torch.Tensor
            The distances to expand. Guaranteed to have shape :math:`(..., 1)`.
        """
        pass

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r"""
        Ensure that the input has the correct shape, :math:`(..., 1)`,
        and then perform the expansion.
        """
        if r.shape[-1] != 1:
            r = r.unsqueeze(-1)
        return self.expand(r)

    def properties(self) -> dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        _kwargs_rep = ", ".join(
            f"{k}={v}" for k, v in self.properties().items()
        )
        return (
            f"{self.__class__.__name__}(1 → {self.n_features}, {_kwargs_rep})"
        )


class Bessel(DistanceExpansion):
    r"""
    The Bessel expansion:

    .. math::
        \phi_{n}(r) = \frac{\sin(\pi n r / r_\text{cut})}{r} \quad n
        \in [1, n_\text{features}]

    where :math:`r_\text{cut}` is the cutoff radius and :math:`n` is the order
    of the Bessel function.

    Parameters
    ----------
    n_features : int
        The number of features to expand into.
    cutoff : float
        The cutoff radius.
    """

    def __init__(self, n_features: int, cutoff: float):
        super().__init__(n_features, cutoff)

        self.register_buffer(
            "frequencies", torch.arange(1, n_features + 1) * π / cutoff
        )

    def expand(self, r: torch.Tensor) -> torch.Tensor:
        numerator = torch.sin(r * self.frequencies)
        # we avoid dividing by zero by replacing any zero elements with 1
        denominator = torch.where(r != 0, torch.tensor(1.0), r)
        return numerator / denominator

    def properties(self) -> dict[str, Any]:
        return {"cutoff": self.cutoff}


class GaussianSmearing(DistanceExpansion):
    r"""
    A Gaussian smearing expansion:

    .. math::
        \phi_{n}(r) = \exp\left(-\frac{(r - \mu_n)^2}{2\sigma^2}\right)

    where :math:`\mu_n` is the center of the :math:`n`th Gaussian
    and :math:`\sigma` is the width of the Gaussians.

    Parameters
    ----------
    n_features : int
        The number of features to expand into.
    cutoff : float
        The cutoff radius.
    """

    def __init__(
        self,
        n_features: int,
        cutoff: float,
    ):
        super().__init__(n_features, cutoff)

        sigma = cutoff / n_features
        self.coef = -1 / (2 * sigma**2)
        self.register_buffer("centers", torch.linspace(0, cutoff, n_features))

    def expand(self, r: torch.Tensor) -> torch.Tensor:
        offsets = r - self.centers
        return torch.exp(self.coef * offsets**2)

    def properties(self) -> dict[str, Any]:
        return {"cutoff": self.cutoff}


Envelope = Callable[[torch.Tensor], torch.Tensor]


class PolynomialEnvelope(nn.Module):
    r"""
    A thrice differentiable envelope function.

    .. math::
        E_p(r) = 1 - \frac{(p+1)(p+2)}{2}\cdot r^p + p(p+2) \cdot r^{p+1}
        - \frac{p(p+1)}{2}\cdot d^{p+2}

    where :math:`r_\text{cut}` is the cutoff radius,
    and :math:`p \in \mathbb{N}`.

    Parameters
    ----------
    cutoff : float
        The cutoff radius.
    p: int
        The order of the envelope function.
    """

    def __init__(self, cutoff: float, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p
        self.register_buffer(
            "coefficients",
            torch.tensor(
                [
                    -(p + 1) * (p + 2) / 2,
                    p * (p + 2),
                    -(p * (p + 1)) / 2,
                ]
            ),
        )
        self.register_buffer("powers", torch.arange(p, p + 3))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        powers = (r.unsqueeze(-1) / self.cutoff) ** self.powers
        envelope = 1 + (powers * self.coefficients).sum(dim=-1)
        return torch.where(r <= self.cutoff, envelope, torch.tensor(0.0))

    def __repr__(self):
        return f"PolynomialEnvelope(cutoff={self.cutoff}, p={self.p})"


class ExtendedPolynomialEnvelope(PolynomialEnvelope):
    def __init__(self, cutoff: float, onset: float | None = None, p: int = 6):
        if onset is None:
            onset = cutoff / 3
        super().__init__(cutoff - onset, p)
        self.onset = onset

    def __call__(self, r):
        return torch.where(
            r < self.onset,
            torch.ones_like(r),
            super().__call__(r - self.onset),
        )

    def __repr__(self):
        return (
            "ExtendedPolynomialEnvelope("
            f"cutoff={self.cutoff}, onset={self.onset}, p={self.p})"
        )
