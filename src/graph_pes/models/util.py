from __future__ import annotations

from torch import nn

from graph_pes.util import pairs

from .activations import parse_activation


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


class Product(nn.Module):
    def __init__(self, components: list[nn.Module]):
        super().__init__()
        self.components = nn.ModuleList(components)

    def forward(self, x):
        out = 1
        for component in self.components:
            out = out * component(x)
        return out
