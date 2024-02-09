from ase import Atoms
from ase.io import read
from graph_pes.data import convert_to_atomic_graph, convert_to_atomic_graphs
from graph_pes.models.pairwise import LennardJones

structures = read("tests/test.xyz", ":")
graphs = convert_to_atomic_graphs(structures, cutoff=3)


def test_model():
    model = LennardJones()
    predictions = model.predict(graphs)
    assert "energy" in predictions
    assert "forces" in predictions
    assert "stress" in predictions and graphs[0].has_cell
    assert predictions["energy"].shape == (len(graphs),)
    assert predictions["stress"].shape == (len(graphs), 3, 3)


def test_isolated_atom():
    atom = Atoms("He", positions=[[0, 0, 0]])
    graph = convert_to_atomic_graph(atom, cutoff=3)
    assert graph.n_atoms == 1 and graph.n_edges == 0

    model = LennardJones()
    assert model(graph) == 0
