import torch
from ase import Atoms
from graph_pes.data import convert_to_atomic_graph
from graph_pes.transform import Identity

graph = convert_to_atomic_graph(
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)]),
    cutoff=1.5,
)


def test_identity():
    transform = Identity()
    x = torch.arange(10).float()

    assert transform.forward(x, graph).equal(x)
    assert transform.inverse(x, graph).equal(x)
