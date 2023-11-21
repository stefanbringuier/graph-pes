from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from ase.data import chemical_symbols
from graph_pes.util import pairs


class MLP(nn.Module):
    """
    A multi-layer perceptron model, alternating linear and activation layers.

    Parameters
    ----------
    layers: List[int]
        The number of nodes in each layer.
    activation: str or nn.Module
        The activation function to use.
    activate_last: bool
        Whether to apply the activation function to the last layer.
    bias: bool
        Whether to include a bias term in the linear layers.
    """

    def __init__(
        self,
        layers: list[int],
        activation: str | nn.Module = "CELU",
        activate_last: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.activation = (
            parse_activation(activation)
            if isinstance(activation, str)
            else activation
        )
        self.activate_last = activate_last

        self.linear_layers = nn.ModuleList(
            [nn.Linear(_in, _out, bias=bias) for _in, _out in pairs(layers)]
        )

    def forward(self, x):
        for i, linear in enumerate(self.linear_layers):
            x = linear(x)
            last_layer = i == len(self.linear_layers) - 1
            if not last_layer or self.activate_last:
                x = self.activation(x)

        return x

    @property
    def input_size(self):
        return self.linear_layers[0].in_features

    @property
    def output_size(self):
        return self.linear_layers[-1].out_features

    @property
    def layer_widths(self):
        inputs = [layer.in_features for layer in self.linear_layers]
        return inputs + [self.output_size]

    def __repr__(self):
        layers = " → ".join(map(str, self.layer_widths))
        return f"MLP({layers}, activation={self.activation})"


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(x) - self.shift


def parse_activation(act: str) -> torch.nn.Module:
    """
    Parse a string into a PyTorch activation function.

    Parameters
    ----------
    act: str
        The activation function to parse.

    Returns
    -------
    torch.nn.Module
        The parsed activation function.
    """
    activation = getattr(torch.nn, act, None)
    if activation is None:
        raise ValueError(f"Activation function {act} not found in `torch.nn`.")
    return activation()


class Product(nn.Module):
    def __init__(self, components: list[nn.Module]):
        super().__init__()
        self.components = nn.ModuleList(components)

    def forward(self, x):
        out = 1
        for component in self.components:
            out = out * component(x)
        return out


class PerSpeciesEmbedding(nn.Module):
    def __init__(
        self,
        Z_keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        default: float | None = None,
        dim: int = 16,
        requires_grad: bool = True,
    ):
        super().__init__()
        if Z_keys is not None and values is not None:
            assert len(Z_keys) == len(values)

        if values is not None:
            dim = values.shape[1]

        self.Z_keys = torch.tensor([]) if Z_keys is None else Z_keys
        values = torch.tensor([]) if values is None else values
        self.values = nn.Parameter(values.float(), requires_grad)
        self.default = default
        self.dim = dim
        self.requires_grad = requires_grad

    def _add_item(self, Z: int, value: torch.Tensor | None = None):
        if value is None:
            if self.default is None:
                # random normal weights
                value = torch.randn(1, self.dim)
            else:
                value = torch.full((1, self.dim), self.default)

        self.Z_keys = torch.cat([self.Z_keys, torch.tensor([Z])])
        self.values = nn.Parameter(
            torch.cat([self.values, value]), self.requires_grad
        )

    def __getitem__(self, Zs: torch.Tensor | int) -> torch.Tensor:
        if isinstance(Zs, int):
            Zs = torch.tensor([Zs])

        # if one or more of the Zs aren't in the keys, add them to the keys
        # and extend the values with the default value
        unique_Zs = torch.unique(Zs)
        new_Zs = unique_Zs[~torch.isin(unique_Zs, self.Z_keys)]
        for Z in new_Zs:
            self._add_item(Z)

        val_idxs = torch.nonzero(self.Z_keys == Zs[:, None], as_tuple=True)[1]
        return self.values[val_idxs]

    def __setitem__(self, Z: int, value: torch.Tensor):
        if Z not in self.Z_keys:
            self._add_item(Z, value)
        else:
            self.values[self.Z_keys == Z] = value

    @torch.no_grad()
    def __repr__(self):
        _dict = {}
        for Z, value in zip(self.Z_keys, self.values):
            _dict[chemical_symbols[int(Z.item())]] = value

        return f"{self.__class__.__name__}({_dict})"

    def forward(self, Zs: torch.Tensor) -> torch.Tensor:
        return self[Zs]


class PerSpeciesParameter(PerSpeciesEmbedding):
    def __init__(
        self,
        Z_keys: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        default: float = 0.0,
        requires_grad: bool = True,
    ):
        if values is not None:
            values = values.view(-1, 1)
        super().__init__(Z_keys, values, default, 1, requires_grad)

    def __getitem__(self, Zs: torch.Tensor | int) -> torch.Tensor:
        return super().__getitem__(Zs).squeeze(-1)

    def __setitem__(self, Z: int, value: float):
        super().__setitem__(Z, torch.tensor([value]))

    def __repr__(self):
        _dict = {}
        for Z, value in zip(self.Z_keys, self.values):
            _dict[chemical_symbols[int(Z.item())]] = value.item()

        return f"{self.__class__.__name__}({_dict})"


class ConstrainedParameter(nn.Module, ABC):
    def __init__(self, x: torch.Tensor, requires_grad: bool = True):
        super().__init__()
        self._parameter = nn.Parameter(x, requires_grad)

    @property
    @abstractmethod
    def value(self) -> torch.Tensor:
        """generate the constrained value"""

    def _do_math(self, other, function, rev=False):
        if isinstance(other, ConstrainedParameter):
            other_value = other.value
        else:
            other_value = other

        if rev:
            return function(other_value, self.value)
        else:
            return function(self.value, other_value)

    def __add__(self, other):
        return self._do_math(other, torch.add)

    def __radd__(self, other):
        return self._do_math(other, torch.add, rev=True)

    def __sub__(self, other):
        return self._do_math(other, torch.sub)

    def __rsub__(self, other):
        return self._do_math(other, torch.sub, rev=True)

    def __mul__(self, other):
        return self._do_math(other, torch.mul)

    def __rmul__(self, other):
        return self._do_math(other, torch.mul, rev=True)

    def __truediv__(self, other):
        return self._do_math(other, torch.true_divide)

    def __rtruediv__(self, other):
        return self._do_math(other, torch.true_divide, rev=True)

    def __pow__(self, other):
        return self._do_math(other, torch.pow)

    def __rpow__(self, other):
        return self._do_math(other, torch.pow, rev=True)

    def log(self):
        return torch.log(self.value)

    def sqrt(self):
        return torch.sqrt(self.value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.value})"


class PositiveParameter(ConstrainedParameter):
    def __init__(self, x: torch.Tensor | float, requires_grad: bool = True):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        super().__init__(torch.log(x), requires_grad)

    @property
    def value(self):
        return torch.exp(self._parameter)
