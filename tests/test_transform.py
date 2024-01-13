import torch
from ase import Atoms
from graph_pes.data import convert_to_atomic_graph
from graph_pes.transform import Identity

structure = Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)])
structure.info["energy"] = -1.0
structure.arrays["forces"] = torch.zeros((2, 3))
graph = convert_to_atomic_graph(structure, cutoff=1.5)


def test_identity():
    transform = Identity()
    x = torch.arange(10).float()

    assert transform.forward(x, graph).equal(x)
    assert transform.inverse(x, graph).equal(x)


def test_is_local_property():
    energy = graph.get_labels("energy")
    assert not graph.is_local_property(energy)

    forces = graph.get_labels("forces")
    assert graph.is_local_property(forces)
