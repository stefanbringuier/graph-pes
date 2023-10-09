import torch

from graph_pes.util import pairs, shape_repr, to_chem_symbol


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
