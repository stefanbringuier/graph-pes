import numpy as np
import pytest
import torch
from graph_pes.util import as_possible_tensor, pairs, shape_repr, to_chem_symbol


def test_pairs():
    assert list(pairs([1, 2, 3])) == [(1, 2), (2, 3)]


def test_shape_repr():
    d = dict(
        a=torch.rand(1, 2),
        b=torch.rand(1),
    )
    assert shape_repr(d) == "a=[1,2], b=[1]"


def test_to_chem_symbol():
    assert to_chem_symbol(1) == "H"
    assert to_chem_symbol(6) == "C"
    assert to_chem_symbol(118) == "Og"


possible_tensors = [
    (1, True),
    (1.0, True),
    ([1, 2, 3], True),
    (torch.tensor([1, 2, 3]), True),
    (np.array([1, 2, 3]), True),
    (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64), True),
    ("hello", False),
]


@pytest.mark.parametrize("obj, can_be_converted", possible_tensors)
def test_as_possible_tensor(obj, can_be_converted):
    if can_be_converted:
        assert isinstance(as_possible_tensor(obj), torch.Tensor)
    else:
        assert as_possible_tensor(obj) is None
