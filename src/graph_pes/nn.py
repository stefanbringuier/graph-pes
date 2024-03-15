from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .util import MAX_Z, pairs


class MLP(nn.Module):
    """
    A multi-layer perceptron model, alternating linear layers and activations.

    Parameters
    ----------
    layers: List[int]
        The number of nodes in each layer.
    activation: str or nn.Module
        The activation function to use: either a named activation function
        from `torch.nn`, or a `torch.nn.Module` instance.
    activate_last: bool
        Whether to apply the activation function after the last linear layer.
    bias: bool
        Whether to include bias terms in the linear layers.

    Examples
    --------
    >>> import torch
    >>> from graph_pes.nn import MLP
    >>> model = MLP([10, 5, 1])
    >>> model
    MLP(10 → 5 → 1, activation=CELU())
    >>> MLP([10, 5, 1], activation=torch.nn.ReLU())
    MLP(10 → 5 → 1, activation=ReLU())
    >>> MLP([10, 5, 1], activation="Tanh")
    MLP(10 → 5 → 1, activation=Tanh())
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x
            The input to the network.
        """
        for i, linear in enumerate(self.linear_layers):
            x = linear(x)
            last_layer = i == len(self.linear_layers) - 1
            if not last_layer or self.activate_last:
                x = self.activation(x)

        return x

    @property
    def input_size(self):
        """The size of the input to the network."""
        return self.linear_layers[0].in_features

    @property
    def output_size(self):
        """The size of the output of the network."""
        return self.linear_layers[-1].out_features

    @property
    def layer_widths(self):
        """The widths of the layers in the network."""
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
    act
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


from functools import reduce
from typing import Iterable


def prod(iterable):
    return reduce(lambda x, y: x * y, iterable, 1)


class PerElementParameter(torch.nn.Parameter):
    def __new__(
        cls, data: Tensor, requires_grad: bool = True
    ) -> PerElementParameter:
        return super().__new__(cls, data, requires_grad=requires_grad)  # type: ignore

    def __init__(self, data: Tensor, requires_grad: bool = True):
        super().__init__()
        # set extra state
        self._accessed_Zs = set()
        # set this to an arbitrary value: this gets updated post-init
        self._index_dims: int = 1

    def register_elements(self, Zs: Iterable[int]) -> None:
        self._accessed_Zs.update(sorted(Zs))

    @classmethod
    def of_shape(
        cls,
        shape: tuple[int, ...] = (),
        index_dims: int = 1,
        default_value: float | None = None,
        requires_grad: bool = True,
    ) -> PerElementParameter:
        actual_shape = tuple([MAX_Z] * index_dims) + shape
        if default_value is not None:
            data = torch.full(actual_shape, float(default_value))
        else:
            data = torch.randn(actual_shape)
        psp = PerElementParameter(data, requires_grad=requires_grad)
        psp._index_dims = index_dims
        return psp

    @classmethod
    def of_length(
        cls,
        length: int,
        index_dims: int = 1,
        default_value: float | None = None,
        requires_grad: bool = True,
    ) -> PerElementParameter:
        return PerElementParameter.of_shape(
            (length,), index_dims, default_value, requires_grad
        )

    def numel(self) -> int:
        n_elements = len(self._accessed_Zs)
        accessed_parameters = n_elements**self._index_dims
        parameter_length = prod(self.shape[self._index_dims :])
        return accessed_parameters * parameter_length

    # needed for de/serialization
    def __reduce_ex__(self, proto):
        return (
            _rebuild_per_element_parameter,
            (self.data, self.requires_grad, torch._utils._get_obj_state(self)),
        )

    def __repr__(self) -> str:
        # TODO implement custom repr for different shapes
        # 1 index dimension:
        #     table with header column for Z
        # 2 index dimensions with singleton further shape:
        #      2D table with Z as both header row and header column
        # print numel otherwise
        return (
            f"PSP(index_dims={self._index_dims}, "
            f"accessed_Zs={sorted(self._accessed_Zs)}, "
            f"shape={tuple(self.shape[self._index_dims:])})"
        )


def _rebuild_per_element_parameter(data, requires_grad, state):
    psp = PerElementParameter(data, requires_grad)
    psp._accessed_Zs = state["_accessed_Zs"]
    psp._index_dims = state["_index_dims"]
    return psp


# TODO update docs
class PerElementEmbedding(torch.nn.Module):
    """
    A per-species equivalent of `torch.nn.Embedding`.

    Parameters
    ----------
    length
        The length of each embedding vector.

    Examples
    --------
    >>> embedding = PerElementEmbedding(10)
    >>> len(graph["atomic_numbers"])  # number of atoms in the graph
    24
    >>> embedding(graph["atomic_numbers"])
    <tensor of shape (24, 10)>
    """

    def __init__(self, length: int):
        super().__init__()
        self._embeddings = PerElementParameter.of_length(length)

    def forward(self, Z: Tensor) -> Tensor:
        return self._embeddings[Z]

    def __repr__(self) -> str:
        return f"PerElementEmbedding(length={self._embeddings.shape[1]})"


class HaddamardProduct(nn.Module):
    def __init__(self, *components: nn.Module, left_aligned: bool = False):
        super().__init__()
        self.components: list[nn.Module] = nn.ModuleList(components)  # type: ignore
        self.left_aligned = left_aligned

    def forward(self, x):
        out = torch.scalar_tensor(1)
        for component in self.components:
            if self.left_aligned:
                out = left_aligned_mul(out, component(x))
            else:
                out = out * component(x)
        return out


# NB: we have to repeat code here somewhat because torchsript doesn't support
# typing for callable


def left_aligned_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Assume:
    x.shape: (n, ...)
    y.shape: (n, )

    We broadcast y to the left of x and add the two tensors elementwise.
    """
    if x.dim() == 1 or x.dim() == 0:
        return x + y
    # add a fake dimension to x to make it (n, 1, ...)
    x = x.unsqueeze(1)
    # transpose x to make it (1, ..., n)
    x = x.transpose(0, -1)
    # apply the operation
    result = x - y  # shape: (1, ..., n)
    # transpose back to the original shape
    result = result.transpose(0, -1)
    # remove the fake dimension
    return result.squeeze(1)


def left_aligned_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1 or x.dim() == 0:
        return x - y
    x = x.unsqueeze(1)
    x = x.transpose(0, -1)
    result = x - y  # shape: (1, ..., n)
    result = result.transpose(0, -1)
    return result.squeeze(1)


def left_aligned_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1 or x.dim() == 0:
        return x * y
    x = x.unsqueeze(1)
    x = x.transpose(0, -1)
    result = x * y  # shape: (1, ..., n)
    result = result.transpose(0, -1)
    return result.squeeze(1)


def left_aligned_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1 or x.dim() == 0:
        return x / y
    x = x.unsqueeze(1)
    x = x.transpose(0, -1)
    result = x / y  # shape: (1, ..., n)
    result = result.transpose(0, -1)
    return result.squeeze(1)
