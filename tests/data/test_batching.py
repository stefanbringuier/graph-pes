from ase import Atoms
from graph_pes.data import convert_to_atomic_graphs
from graph_pes.data.batching import AtomicGraphBatch

STRUCTURES = [
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False),
    Atoms("H3", positions=[(0, 0, 0), (0, 0, 1), (0, 0, 2)], pbc=False),
]
GRAPHS = convert_to_atomic_graphs(STRUCTURES, cutoff=1.5)


def test_batching():
    batch = AtomicGraphBatch.from_graphs(GRAPHS)
    assert isinstance(batch, AtomicGraphBatch)
    assert batch.n_atoms == 5
    assert batch.n_structures == 2
    assert list(batch.ptr) == [0, 2, 5]
    assert list(batch.batch) == [0, 0, 1, 1, 1]

    assert batch.n_edges == sum(g.n_edges for g in GRAPHS)
