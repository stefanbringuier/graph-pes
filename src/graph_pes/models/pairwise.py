from __future__ import annotations

import torch
from torch import scalar_tensor as scalar

from graph_pes.models.util import MLP
from graph_pes.util import MAX_Z

from .base import PairPotential, SimplePairPotential
from .distances import Bessel, PolynomialEnvelope


class LennardJones(SimplePairPotential):
    r"""
    A pair potential of the form:

    .. math::
        \varepsilon_{ij} = 4 \varepsilon \left[ \left( \frac{\sigma}{r_{ij}}
        \right)^{12} - \left( \frac{\sigma}{r_{ij}} \right)^{6} \right]

    where :math:`r_{ij}` is the distance between atoms :math:`i` and :math:`j`.

    Parameters
    ----------
    epsilon : Optional[float]
        The depth of the potential.
    sigma : Optional[float]
        The distance at which the potential is zero.

    Attributes
    ----------
    epsilon : torch.nn.Parameter
        The depth of the potential.
    sigma : torch.nn.Parameter
        The distance at which the potential is zero.
    """

    # Dev Note:
    # epsilon and sigma should both always be positive
    # we therefore actually store the log of these quantities
    # and then exponentiate them when we need them

    def __init__(self, epsilon: float = 1.0, sigma: float = 0.5):
        super().__init__()
        self._log_epsilon = torch.nn.Parameter(torch.log(scalar(epsilon)))
        self._log_sigma = torch.nn.Parameter(torch.log(scalar(sigma)))

    @property
    def epsilon(self) -> torch.Tensor:
        """The depth of the potential."""
        return torch.exp(self._log_epsilon)

    @property
    def sigma(self) -> torch.Tensor:
        """The distance at which the potential is zero."""
        return torch.exp(self._log_sigma)

    def pair_potential(self, r: torch.Tensor):
        """
        Evaluate the pair potential.

        Parameters
        ----------
        r : torch.Tensor
            The pair-wise distances between the atoms.
        """
        x = self.sigma / r
        return 4 * self.epsilon * (x**12 - x**6)


class MultiLennardJones(PairPotential):
    r"""
    A multi-component Lennard Jones potential:

    .. math::
        U_{ij}(r) = 4\epsilon_{ij} \left[ \left( \frac{\sigma_{ij}}{r} \right)^{12} - \left( \frac{\sigma_{ij}}{r} \right)^6 \right]

    where both :math:`\epsilon_{ij}` and :math:`\sigma_{ij}` are functions of
    the atomic numbers :math:`Z_i` and :math:`Z_j`:
    """

    def __init__(
        self,
        epsilon: torch.Tensor | float = 1.0,
        sigma: torch.Tensor | float = 0.5,
    ):
        super().__init__()

        if isinstance(epsilon, float):
            _log_epsilon = torch.log(torch.ones(MAX_Z, MAX_Z) * epsilon)
        else:
            if not len(epsilon.shape) == 2:
                raise ValueError(
                    f"epsilon must be a 2D tensor, got {epsilon.shape}"
                )
            if not (epsilon > 0).all():
                raise ValueError("epsilon must be positive for all pairs i,j")
            _log_epsilon = torch.log(epsilon)
        self._log_epsilon = torch.nn.Parameter(
            _log_epsilon, requires_grad=True
        )

        if isinstance(sigma, float):
            _log_sigma = torch.log(torch.ones(MAX_Z, MAX_Z) * sigma)
        else:
            if not len(sigma.shape) == 2:
                raise ValueError(
                    f"sigma must be a 2D tensor, got {sigma.shape}"
                )
            if not (sigma > 0).all():
                raise ValueError("sigma must be positive for all pairs i,j")
            _log_sigma = torch.log(sigma)
        self._log_sigma = torch.nn.Parameter(_log_sigma, requires_grad=True)

        self.offset = torch.nn.Parameter(torch.zeros(MAX_Z, MAX_Z))

    @property
    def epsilon(self):
        """The depth of the LJ well for each pair of elements"""
        return torch.exp(self._log_epsilon)

    @property
    def sigma(self):
        """The position of the LJ minimum for each pair of elements"""
        return torch.exp(self._log_sigma)

    def pair_potential(
        self, distances: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
    ) -> torch.Tensor:
        # to keep the potential symmetric, we use the average of the two
        # values for epsilon and sigma
        epsilon = self.epsilon[Z_i, Z_j] + self.epsilon[Z_j, Z_i]
        sigma = self.sigma[Z_i, Z_j] + self.sigma[Z_j, Z_i]

        x = sigma / distances
        lj = 4 * epsilon * (x**12 - x**6)

        # we add an aribtrary offset to the potential
        return lj + (self.offset[Z_i, Z_j] + self.offset[Z_j, Z_i]) / 2


class LearnablePairPotential(PairPotential):
    def __init__(
        self,
        hidden_layers: list[int],
        cutoff: float = 3.7,
        r_features: int = 6,
        z_features: int = 4,
        activation: str = "CELU",
    ):
        super().__init__()

        self.envelope = PolynomialEnvelope(cutoff)
        self.r_embedding = Bessel(r_features, cutoff)
        self.z_embedding = torch.nn.Embedding(MAX_Z, z_features)

        layers = [r_features + 2 * z_features] + hidden_layers + [1]
        self.mlp = MLP(layers, activation)

    def pair_potential(
        self, distances: torch.Tensor, Z_i: torch.Tensor, Z_j: torch.Tensor
    ) -> torch.Tensor:
        e = self.envelope(distances)

        # embed the pairwise distance and atomic numbers
        # application of the envelope here ensures smoothness wrt.
        # atoms entering and leaving the cutoff shell
        h_r = self.r_embedding(distances) * e
        h_i = self.z_embedding(Z_i).squeeze()
        h_j = self.z_embedding(Z_j).squeeze()

        # concatenate the features and predict the pair potential
        h = torch.cat([h_r, h_i, h_j], dim=-1)
        V = self.mlp(h)

        # the second envelope application ensures that the potential
        # has the correct asymptotic behaviour (V -> 0 as r -> cutoff)
        return V * e
