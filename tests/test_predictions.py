import pytest
import torch
from ase import Atoms
from graph_pes.core import get_predictions
from graph_pes.data import AtomicGraphBatch, convert_to_atomic_graph
from graph_pes.models.pairwise import LennardJones
from graph_pes.util import Keys

no_pbc = convert_to_atomic_graph(
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=False),
    cutoff=1.5,
)
pbc = convert_to_atomic_graph(
    Atoms("H2", positions=[(0, 0, 0), (0, 0, 1)], pbc=True, cell=(2, 2, 2)),
    cutoff=1.5,
)


def test_predictions():
    expected_shapes = {
        Keys.ENERGY: (),
        Keys.FORCES: (2, 3),
        Keys.STRESS: (3, 3),
    }

    model = LennardJones()

    # by default, get_predictions returns energy and forces on
    # structures with no cell:
    predictions = get_predictions(model, no_pbc)
    assert set(predictions.keys()) == {"energy", "forces"}

    for key in Keys.ENERGY, Keys.FORCES:
        assert predictions[key.value].shape == expected_shapes[key]

    # if we ask for stress, we get an error:
    with pytest.raises(ValueError):
        get_predictions(model, no_pbc, {Keys.STRESS: "stress"})

    # with pbc structures, we should get all three predictions
    predictions = get_predictions(model, pbc)
    assert set(predictions.keys()) == {"energy", "forces", "stress"}

    for key in Keys.ENERGY, Keys.FORCES, Keys.STRESS:
        assert predictions[key.value].shape == expected_shapes[key]

    # check that requesting a subset of predictions works, and that
    # the names are correctly mapped:
    predictions = get_predictions(model, no_pbc, {Keys.ENERGY: "total_energy"})
    assert set(predictions.keys()) == {"total_energy"}
    assert predictions["total_energy"].shape == expected_shapes[Keys.ENERGY]


def test_batched_prediction():
    batch = AtomicGraphBatch.from_graphs([pbc, pbc])

    expected_shapes = {
        Keys.ENERGY: (2,),  # two structures
        Keys.FORCES: (4, 3),  # four atoms
        Keys.STRESS: (2, 3, 3),  # two structures
    }

    predictions = get_predictions(LennardJones(), batch)

    for key in Keys.ENERGY, Keys.FORCES, Keys.STRESS:
        assert predictions[key.value].shape == expected_shapes[key]


def test_isolated_atom():
    atom = Atoms("H", positions=[(0, 0, 0)], pbc=False)
    graph = convert_to_atomic_graph(atom, cutoff=1.5)
    assert graph.n_edges == 0

    predictions = get_predictions(LennardJones(), graph)
    assert torch.allclose(predictions["forces"], torch.zeros(1, 3))
