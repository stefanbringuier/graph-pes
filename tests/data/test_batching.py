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
    assert batch.structure_sizes.tolist() == [2, 3]
    assert batch.neighbour_vectors.shape == (batch.n_edges, 3)


def test_label_batching():
    structures = [s.copy() for s in STRUCTURES]
    # per-structure labels:
    structures[0].info["label"] = 0
    structures[1].info["label"] = 1

    # per-atom labels:
    structures[0].arrays["atom_label"] = [0, 1]
    structures[1].arrays["atom_label"] = [2, 3, 4]

    graphs = convert_to_atomic_graphs(structures, cutoff=1.5)
    batch = AtomicGraphBatch.from_graphs(graphs)

    # per-structure labels are concatenated along a new batch axis (0)
    assert batch.structure_labels["label"].tolist() == [[0], [1]]

    # per-atom labels are concatenated along the first axis
    assert batch.atom_labels["atom_label"].tolist() == [0, 1, 2, 3, 4]


def test_pbcs():
    # structure 1: 1 atom in a (1x1x1) cell
    # structure 2: 1 atom in a (2x2x2) cell
    structures = [
        Atoms("H", positions=[(0, 0, 0)], pbc=True, cell=(1, 1, 1)),
        Atoms("H", positions=[(0, 0, 0)], pbc=True, cell=(2, 2, 2)),
    ]
    graphs = convert_to_atomic_graphs(structures, cutoff=1.2)

    assert graphs[0].n_edges == 6
    assert graphs[1].n_edges == 0

    batch = AtomicGraphBatch.from_graphs(graphs)
    assert batch.n_edges == 6

    vecs = batch.neighbour_vectors
    assert vecs.shape == (6, 3)

    assert batch.neighbour_distances.tolist() == [1.0] * 6
