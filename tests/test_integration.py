from ase.io import read
from graph_pes.core import get_predictions
from graph_pes.data.atomic_graph import convert_to_atomic_graphs
from graph_pes.data.batching import AtomicGraphBatch
from graph_pes.models.pairwise import LennardJones
from graph_pes.training import Loss, train_model


def test_integration():
    structures = read("tests/test.xyz", ":")
    graphs = convert_to_atomic_graphs(structures, cutoff=3)
    batch = AtomicGraphBatch.from_graphs(graphs)

    model = LennardJones()

    loss = Loss("energy")
    before = loss(get_predictions(model, batch, ["energy"]), batch)

    train_model(
        model,
        train_data=graphs[:8],
        val_data=graphs[8:],
        loss=loss,
        max_epochs=2,
        accelerator="cpu",
        callbacks=[],
    )

    after = loss(get_predictions(model, batch, ["energy"]), batch)

    assert after < before, "training did not improve the loss"
