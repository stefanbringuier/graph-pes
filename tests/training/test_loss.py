from __future__ import annotations

import pytest
import torch

from graph_pes.atomic_graph import AtomicGraph
from graph_pes.models.pairwise import LennardJones
from graph_pes.training.loss import MAE, RMSE, EquigradLoss, WeightedLoss


def test_metrics():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])

    assert torch.allclose(MAE()(a, b), torch.tensor(0.0))
    assert torch.allclose(RMSE()(a, b), torch.tensor(0.0))

    c = torch.tensor([0, 0, 0]).float()
    assert torch.allclose(MAE()(a, c), torch.tensor(2.0))
    assert torch.allclose(RMSE()(a, c), torch.tensor((1 + 4 + 9) / 3).sqrt())


def test_exception():
    with pytest.raises(ImportError):
        WeightedLoss()


@pytest.mark.parametrize("weight", [0.5, 1.0, 2.0])
def test_equigrad_loss(weight):

    model = LennardJones(cutoff=5.0)

    Z = torch.tensor([1, 1])
    R = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    cell = torch.zeros(3, 3)

    neighbour_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    neighbour_cell_offsets = torch.zeros(2, 3)

    graph = AtomicGraph.create_with_defaults(
        Z=Z,
        R=R,
        cell=cell,
        neighbour_list=neighbour_list,
        neighbour_cell_offsets=neighbour_cell_offsets,
        cutoff=5.0,
    )

    equigrad_loss = EquigradLoss(weight=weight)
    predictions = model.predict(graph, ["energy", "forces"])
    loss_value = equigrad_loss(model, graph, predictions)

    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.ndim == 0

    # Should be non-negative
    assert loss_value >= 0

    # Weight scale test
    if weight != 1.0:
        reference_loss = EquigradLoss(weight=1.0)(model, graph, predictions)
        assert torch.isclose(loss_value, reference_loss * weight)
