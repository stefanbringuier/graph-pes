import torch


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
