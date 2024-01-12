import torch
from graph_pes.nn import MLP, PerSpeciesEmbedding, PerSpeciesParameter
from graph_pes.util import MAX_Z


def test_per_species_parameter():
    """Test the PerSpeciesParameter class."""

    psp = PerSpeciesParameter.of_dim(5)
    assert psp.data.shape == (MAX_Z, 5)

    # nothing has been accessed yet, so there should be no
    # trainable parameters
    assert psp.numel() == 0

    # access the parameter for hydrogen
    psp[1]
    assert psp.numel() == 5

    embedding = PerSpeciesEmbedding(10)
    Z = torch.tensor([1, 2, 3, 4, 5])
    assert embedding(Z).shape == (5, 10)
    assert embedding.parameters().__next__().numel() == 50


def test_mlp():
    mlp = MLP([10, 20, 1])

    # test behaviour
    assert mlp(torch.zeros(10)).shape == (1,)

    # test properties
    assert mlp.input_size == 10
    assert mlp.output_size == 1

    # test internals
    assert len(mlp.linear_layers) == 2

    # test nice repr
    assert "MLP(10 → 20 → 1" in str(mlp)
