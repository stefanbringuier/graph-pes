import numpy as np
import pytest
from ase import Atoms
from graph_pes.data import (
    AtomicDataLoader,
    convert_to_atomic_graphs,
    sum_per_structure,
)
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
    structures[0].info["stress"] = np.eye(3)
    structures[1].info["label"] = 1
    structures[1].info["stress"] = 2 * np.eye(3)

    # per-atom labels:
    structures[0].arrays["atom_label"] = [0, 1]
    structures[1].arrays["atom_label"] = [2, 3, 4]

    graphs = convert_to_atomic_graphs(structures, cutoff=1.5)
    batch = AtomicGraphBatch.from_graphs(graphs)

    # per-structure, array-type labels are concatenated along a new batch axis
    assert batch.structure_labels["stress"].shape == (2, 3, 3)
    # per-structure, scalar-type labels are concatenated
    assert batch.structure_labels["label"].tolist() == [0, 1]

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
    assert batch.cells.shape == (2, 3, 3)

    vecs = batch.neighbour_vectors
    assert vecs.shape == (6, 3)

    assert batch.neighbour_distances.tolist() == [1.0] * 6


def test_sum_per_structure():
    structures = [s.copy() for s in STRUCTURES]
    structures[0].arrays["atom_label"] = [0, 1]
    structures[1].arrays["atom_label"] = [2, 3, 4]
    graphs = convert_to_atomic_graphs(structures, cutoff=1.5)

    # sum per structure should work for:

    # 1. a single structure
    x = graphs[0].atom_labels["atom_label"]
    assert sum_per_structure(x, graphs[0]) == x.sum()

    # 2. a batch of structures
    batch = AtomicGraphBatch.from_graphs(graphs)
    x = batch.atom_labels["atom_label"]
    assert sum_per_structure(x, batch).tolist() == [1, 9]


def test_data_loader():
    loader = AtomicDataLoader(GRAPHS, batch_size=2)
    for batch in loader:
        assert isinstance(batch, AtomicGraphBatch)
        assert batch.n_atoms == 5
        assert batch.n_structures == 2
        assert batch.n_edges == 6
        assert batch.structure_sizes.tolist() == [2, 3]
        assert batch.neighbour_vectors.shape == (6, 3)
        break

    # test warning if try to pass own collate function
    with pytest.warns(UserWarning):
        AtomicDataLoader(GRAPHS, batch_size=2, collate_fn=lambda x: x)
