from __future__ import annotations

import pytest
import torch
import numpy as np

from graph_pes.atomic_graph import AtomicGraph, replace
from graph_pes.atomic_graph import to_batch
from graph_pes.models.pairwise import LennardJones
from graph_pes.training.loss import EquigradLoss

# Lennard-Jones model:
# Invariant energy w.r.t global rotation
# Equivariant forces w.r.t global rotation


def test_equigrad_mock_gradients():
    # Mock rotational gradients
    equivariant_rotgrad = torch.zeros(1, 3, 1)
    non_equivariant_rotgrad = torch.tensor([[[0.5]], [[0.3]], [[0.2]]])

    def equigrad_loss_fn(rotgrad):
        return torch.norm(rotgrad, dim=(1, 2)).mean()

    # Calculate losses
    equivariant_loss = equigrad_loss_fn(equivariant_rotgrad)
    non_equivariant_loss = equigrad_loss_fn(non_equivariant_rotgrad)

    # Assume Non-equivariant model should have higher loss
    assert non_equivariant_loss > equivariant_loss
    assert non_equivariant_loss > 0.1


class NonEquivariantModel(torch.nn.Module):
    """A simple model that violates rotational equivariance."""

    def __init__(self):
        super().__init__()
        self.implemented_properties = ["energy"]
        self.cutoff = torch.tensor(5.0)
        self.direction_weights = torch.nn.Parameter(torch.tensor([10.0, 0.5, 0.1]))

    def forward(self, graph):
        """Forward pass with non-equivariant behavior.

        This model's energy depends on absolute coordinates, breaking
        rotational equivariance.
        """
        positions = graph.R
        x_coords = positions[:, 0] * self.direction_weights[0]
        y_coords = positions[:, 1] * self.direction_weights[1]
        z_coords = positions[:, 2] * self.direction_weights[2]
        energy = torch.sum(x_coords) + torch.sum(y_coords) + torch.sum(z_coords)
        return {"energy": energy}


def create_test_graph():
    """Create a simple test graph for equivariance tests."""
    Z = torch.tensor([1, 1, 1, 1])
    R = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    cell = torch.zeros(3, 3)

    # All atoms connect to all other atoms
    neighbour_list = torch.tensor(
        [[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]],
        dtype=torch.long,
    )
    neighbour_cell_offsets = torch.zeros(12, 3)

    return AtomicGraph.create_with_defaults(
        Z=Z,
        R=R,
        cell=cell,
        neighbour_list=neighbour_list,
        neighbour_cell_offsets=neighbour_cell_offsets,
        cutoff=5.0,
    )


def test_equivariance_properties():
    equivariant_model = LennardJones(cutoff=5.0)
    non_equivariant_model = NonEquivariantModel()

    graph = create_test_graph()

    # Before rotation
    with torch.no_grad():
        equi_output = equivariant_model(graph)
        non_equi_output = non_equivariant_model(graph)

        equi_energy = (
            equi_output["energy"]
            if "energy" in equi_output
            else equi_output["local_energies"].sum()
        )

        non_equi_energy = (
            non_equi_output["energy"]
            if "energy" in non_equi_output
            else non_equi_output["local_energies"].sum()
        )

        # Rotated graph
        theta = np.pi / 2
        rotation_matrix = torch.tensor(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )

        rotated_R = graph.R @ rotation_matrix
        rotated_graph = replace(graph, R=rotated_R)

        # After rotation
        equi_output_rotated = equivariant_model(rotated_graph)
        non_equi_output_rotated = non_equivariant_model(rotated_graph)

        equi_energy_rotated = (
            equi_output_rotated["energy"]
            if "energy" in equi_output_rotated
            else equi_output_rotated["local_energies"].sum()
        )

        non_equi_energy_rotated = (
            non_equi_output_rotated["energy"]
            if "energy" in non_equi_output_rotated
            else non_equi_output_rotated["local_energies"].sum()
        )

        # Verify properties
        equi_diff = abs(equi_energy - equi_energy_rotated).item()
        non_equi_diff = abs(non_equi_energy - non_equi_energy_rotated).item()

        # Equivariant model: same energy after rotation
        assert equi_diff < 1e-5
        # Non-equivariant model: generally different energy after rotation
        assert non_equi_diff > 1e-5


def test_equigrad_loss_methods():
    equivariant_model = LennardJones(cutoff=5.0)
    non_equivariant_model = NonEquivariantModel()

    graph = create_test_graph()

    # Test finite-difference methods
    equigrad_loss_forward = EquigradLoss(weight=1.0, method_type="forward")
    equigrad_loss_central = EquigradLoss(weight=1.0, method_type="central")

    equivariant_rotgrad_forward = equigrad_loss_forward.compute_rotational_gradients(
        equivariant_model, graph
    )
    non_equivariant_rotgrad_forward = (
        equigrad_loss_forward.compute_rotational_gradients(non_equivariant_model, graph)
    )

    equivariant_loss_forward = torch.norm(
        equivariant_rotgrad_forward, dim=(1, 2)
    ).mean()
    non_equivariant_loss_forward = torch.norm(
        non_equivariant_rotgrad_forward, dim=(1, 2)
    ).mean()

    equivariant_rotgrad_central = equigrad_loss_central.compute_rotational_gradients(
        equivariant_model, graph
    )
    non_equivariant_rotgrad_central = (
        equigrad_loss_central.compute_rotational_gradients(non_equivariant_model, graph)
    )

    equivariant_loss_central = torch.norm(
        equivariant_rotgrad_central, dim=(1, 2)
    ).mean()
    non_equivariant_loss_central = torch.norm(
        non_equivariant_rotgrad_central, dim=(1, 2)
    ).mean()

    # Check that both methods
    assert equivariant_loss_forward < 0.1
    assert equivariant_loss_central < 0.1

    # Non-equivariant model should have non-zero loss
    assert non_equivariant_loss_forward > 1e-6
    assert non_equivariant_loss_central > 1e-6

    # Test that full loss function gives correct output
    with torch.no_grad():
        non_equi_output = non_equivariant_model(graph)
        non_equi_energy = non_equi_output["energy"]

        forward_loss = equigrad_loss_forward(
            non_equivariant_model, graph, {"energy": non_equi_energy}
        )
        central_loss = equigrad_loss_central(
            non_equivariant_model, graph, {"energy": non_equi_energy}
        )

        # Verify both methods produce positive values
        assert forward_loss > 0
        assert central_loss > 0


class ControlledNonEquivariantModel(torch.nn.Module):
    """A model with known non-equivariant behavior."""

    def __init__(self, magnitude=1.0):
        """Initialize with known magnitude of non-equivariance."""
        super().__init__()
        self.implemented_properties = ["energy"]
        self.cutoff = torch.tensor(5.0)
        self.magnitude = magnitude

    def forward(self, graph):
        """Forward pass with known rotation-dependent energy."""
        # Calculate center of mass
        positions = graph.R
        center_of_mass = positions.mean(dim=0)

        # Energy depends on x-coordinate
        energy = self.magnitude * center_of_mass[0]

        return {"energy": energy}


def test_equigrad_methods_comparison():
    """Test central difference accuracy vs forward difference."""
    magnitudes = [0.1, 1.0, 10.0]
    rotation_mag = 0.01

    for mag in magnitudes:
        # Create model with known behavior
        model = ControlledNonEquivariantModel(magnitude=mag)

        # Create test graph
        Z = torch.tensor([1, 1, 1, 1])
        R = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=torch.float32,
        )
        graph = AtomicGraph.create_with_defaults(Z=Z, R=R, cutoff=5.0)

        # Create loss instances for both methods
        loss_forward = EquigradLoss(
            weight=1.0, rotation_mag=rotation_mag, method_type="forward"
        )
        loss_central = EquigradLoss(
            weight=1.0, rotation_mag=rotation_mag, method_type="central"
        )

        forward_grads = loss_forward.compute_rotational_gradients(model, graph)
        central_grads = loss_central.compute_rotational_gradients(model, graph)

        expected_y_axis_grad = mag

        # Compare accuracy of both methods
        forward_y_grad = forward_grads[0, 1, 0].item()
        central_y_grad = central_grads[0, 1, 0].item()

        forward_error = (
            abs(forward_y_grad - expected_y_axis_grad) / expected_y_axis_grad
        )
        central_error = (
            abs(central_y_grad - expected_y_axis_grad) / expected_y_axis_grad
        )

        # Central difference should be more accurate
        assert central_error <= forward_error


def test_equigrad_batch_processing():
    # Test Equigrad with batch
    Z1 = torch.tensor([1, 1, 1, 1])
    R1 = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    Z2 = torch.tensor([1, 1, 1])
    R2 = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    )

    graph1 = AtomicGraph.create_with_defaults(Z=Z1, R=R1, cutoff=5.0)
    graph2 = AtomicGraph.create_with_defaults(Z=Z2, R=R2, cutoff=5.0)

    batch = to_batch([graph1, graph2])

    non_equivariant_model = NonEquivariantModel()

    equigrad_loss = EquigradLoss(weight=1.0, method_type="central")

    # Gradients for individual graphs and batched graph
    rotgrad_1 = equigrad_loss.compute_rotational_gradients(
        non_equivariant_model, graph1
    )
    rotgrad_2 = equigrad_loss.compute_rotational_gradients(
        non_equivariant_model, graph2
    )
    rotgrad_batch = equigrad_loss.compute_rotational_gradients(
        non_equivariant_model, batch
    )

    # Check dimensions
    assert rotgrad_batch.shape[0] == 2
    assert rotgrad_1.shape[0] == 1
    assert rotgrad_2.shape[0] == 1

    # Computation with batched input
    with torch.no_grad():
        output_batch = non_equivariant_model(batch)
        loss_batch = equigrad_loss(
            non_equivariant_model, batch, {"energy": output_batch["energy"]}
        )

        assert isinstance(loss_batch, torch.Tensor)
        assert loss_batch.ndim == 0  # scalar


def test_equigrad_small_rotations():
    # Test numerical stability with very small rotations

    model = ControlledNonEquivariantModel(magnitude=1.0)

    Z = torch.tensor([1, 1, 1, 1])
    R = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    graph = AtomicGraph.create_with_defaults(Z=Z, R=R, cutoff=5.0)

    # Progressively smaller rotations
    rotation_mags = [1e-2, 1e-3, 1e-4, 1e-5]
    expected_y_axis_grad = 1.0

    # Collect difference magnitudes
    forward_errors = []
    central_errors = []

    for rot_mag in rotation_mags:
        loss_forward = EquigradLoss(
            weight=1.0, rotation_mag=rot_mag, method_type="forward"
        )
        loss_central = EquigradLoss(
            weight=1.0, rotation_mag=rot_mag, method_type="central"
        )

        forward_grads = loss_forward.compute_rotational_gradients(model, graph)
        central_grads = loss_central.compute_rotational_gradients(model, graph)

        # Just do y rotation grad about x
        forward_y_grad = forward_grads[0, 1, 0].item()
        central_y_grad = central_grads[0, 1, 0].item()

        forward_error = (
            abs(forward_y_grad - expected_y_axis_grad) / expected_y_axis_grad
        )
        central_error = (
            abs(central_y_grad - expected_y_axis_grad) / expected_y_axis_grad
        )

        forward_errors.append(forward_error)
        central_errors.append(central_error)

        #  Check both are reasonable
        assert not torch.isnan(
            forward_grads
        ).any(), f"NaN in forward_grads at rot_mag={rot_mag}"
        assert not torch.isnan(
            central_grads
        ).any(), f"NaN in central_grads at rot_mag={rot_mag}"
        assert not torch.isinf(
            forward_grads
        ).any(), f"Inf in forward_grads at rot_mag={rot_mag}"
        assert not torch.isinf(
            central_grads
        ).any(), f"Inf in central_grads at rot_mag={rot_mag}"

    # Centeral diff better than forward diff
    for i in range(len(rotation_mags)):
        assert (
            central_errors[i] <= forward_errors[i]
        ), f"At rotation_mag={rotation_mags[i]}"

    # Check numerical stability
    for i in range(len(rotation_mags) - 1):
        # Check if reasonable
        error_growth_forward = (
            forward_errors[i + 1] / forward_errors[i] if forward_errors[i] > 0 else 1.0
        )
        error_growth_central = (
            central_errors[i + 1] / central_errors[i] if central_errors[i] > 0 else 1.0
        )
        # Limit numerical error growth rate
        assert (
            error_growth_forward < 100
        ), f"Forward error grew too fast: {error_growth_forward} at {rotation_mags[i]}->{rotation_mags[i+1]}"
        assert (
            error_growth_central < 100
        ), f"Central error grew too fast: {error_growth_central} at {rotation_mags[i]}->{rotation_mags[i+1]}"
