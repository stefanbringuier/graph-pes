from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi as π
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor, nn


class DistanceExpansion(nn.Module, ABC):
    r"""
    Abstract base class for an expansion function, :math:`\phi(r) :
    [0, r_{\text{cutoff}}] \rightarrow \mathbb{R}^{n_\text{features}}`.

    Subclasses should implement :meth:`expand`, which must also work over
    batches: :math:`\phi(r) : [0, r_{\text{cutoff}}]^{n_\text{batch} \times 1}
    \rightarrow \mathbb{R}^{n_\text{batch} \times n_\text{features}}`.

    Parameters
    ----------
    n_features
        The number of features to expand into.
    cutoff
        The cutoff radius.
    trainable
        Whether the expansion parameters are trainable.
    """

    def __init__(self, n_features: int, cutoff: float, trainable: bool = True):
        super().__init__()
        self.n_features = n_features
        self.cutoff = cutoff
        self.trainable = trainable

    @abstractmethod
    def expand(
        self, r: Float[Tensor, "... 1"]
    ) -> Float[Tensor, "... n_features"]:
        r"""
        Perform the expansion.

        Parameters
        ----------
        r : torch.Tensor
            The distances to expand. Guaranteed to have shape :math:`(..., 1)`.
        """

    def forward(
        self, r: Float[Tensor, "..."]
    ) -> Float[Tensor, "... n_features"]:
        if r.shape[-1] != 1:
            r = r.unsqueeze(-1)
        return self.expand(r)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(n_features={self.n_features}, "
            f"cutoff={self.cutoff}, trainable={self.trainable})"
        )


class Bessel(DistanceExpansion):
    r"""
    The Bessel expansion:

    .. math::
        \phi_{n}(r) = \sqrt{\frac{2}{r_{\text{cut}}}}
        \frac{\sin(n \pi \frac{r}{r_\text{cut}})}{r} \quad n
        \in [1, n_\text{features}]

    where :math:`r_\text{cut}` is the cutoff radius and :math:`n` is the order
    of the Bessel function, as introduced in `Directional Message Passing for
    Molecular Graphs <http://arxiv.org/abs/2003.03123>`_.

    Parameters
    ----------
    n_features
        The number of features to expand into.
    cutoff
        The cutoff radius.
    trainable
        Whether the expansion parameters are trainable.

    Attributes
    ----------
    frequencies
        :math:`n`, the frequencies of the Bessel functions.
    """

    def __init__(self, n_features: int, cutoff: float, trainable: bool = True):
        super().__init__(n_features, cutoff)
        self.frequencies = nn.Parameter(
            torch.arange(1, n_features + 1) * π / cutoff,
            requires_grad=trainable,
        )
        self.pre_factor = torch.sqrt(torch.tensor(2 / cutoff))

    def expand(self, r: torch.Tensor) -> torch.Tensor:
        numerator = self.pre_factor * torch.sin(r * self.frequencies)
        # we avoid dividing by zero by replacing any zero elements with 1
        denominator = torch.where(r != 0, torch.tensor(1.0), r)
        return numerator / denominator


class GaussianSmearing(DistanceExpansion):
    r"""
    A Gaussian smearing expansion:

    .. math::
        \phi_{n}(r) = \exp\left(-\frac{(r - \mu_n)^2}{2\sigma^2}\right)
        \quad n \in [1, n_\text{features}]

    where :math:`\mu_n` is the center of the :math:`n`'th Gaussian
    and :math:`\sigma` is a width shared across all the Gaussians.

    Parameters
    ----------
    n_features
        The number of features to expand into.
    cutoff
        The cutoff radius.
    trainable
        Whether the expansion parameters are trainable.

    Attributes
    ----------
    centers
        :math:`\mu_n`, the centers of the Gaussians.
    coef
        :math:`\frac{1}{2\sigma^2}`, the coefficient of the exponent.
    """

    def __init__(
        self,
        n_features: int,
        cutoff: float,
        trainable: bool = True,
    ):
        super().__init__(n_features, cutoff, trainable)

        sigma = cutoff / n_features
        self.coef = nn.Parameter(
            torch.tensor(-1 / (2 * sigma**2)),
            requires_grad=trainable,
        )
        self.centers = nn.Parameter(
            torch.linspace(0, cutoff, n_features),
            requires_grad=trainable,
        )

    def expand(self, r: torch.Tensor) -> torch.Tensor:
        offsets = r - self.centers
        return torch.exp(self.coef * offsets**2)


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
