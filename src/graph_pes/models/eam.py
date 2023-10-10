"""Embedded atom models."""

from __future__ import annotations
from abc import ABC, abstractmethod

import torch
from graph_pes.data import AtomicGraph

from graph_pes.data.atomic_graph import AtomicGraph
from graph_pes.data.batching import AtomicGraphBatch
from graph_pes.models.base import (
    GraphPESModel,
    MessagePassingPESModel,
    PairPotential,
    SimplePairPotential,
)


def EmbeddedAtomModel(
    pair_potential: PairPotential,
    embedding_model: EmbeddingModel,
):
    """
    Create an embedded atom model.

    Parameters
    ----------
    pair_potential : PairPotential
        The pair potential.
    embedding_model : EmbeddingModel
        The embedding model.

    Returns
    -------
    GraphPESModel
        The embedded atom model.
    """
    return pair_potential + embedding_model


class EmbeddingModel(MessagePassingPESModel):
    def __init__(self, density_model, readout_model):
        super().__init__(aggr="add")
        self.density_model = density_model
        self.readout_model = readout_model

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        density_sums = self.propagate(
            graph.neighbour_index,
            distances=graph.neighbour_distances.unsqueeze(-1),
            Z=graph.Z.unsqueeze(-1),
        )
        return self.readout_model(density_sums)

    def message(self, distances, Z_i, Z_j):
        return self.density_model(distances, Z_i, Z_j)


class DensityModel(torch.nn.Module, ABC):
    r"""
    A density model takes as input the central atom identities, Z_i,
    the pair-wise distances, r_ij, and the atomic numbers of the
    neighbouring atoms, Z_j, and returns some "density", \rho_i, of the
    central atom.
    """

    @abstractmethod
    def forward(self, distances, Z_i, Z_j):
        r"""
        Evaluate the density model.

        Parameters
        ----------
        distances : torch.Tensor
            The pair-wise distances between the atoms.
        Z_i : torch.Tensor
            The atomic numbers of atom :math:`i`.
        Z_j : torch.Tensor
            The atomic numbers of atom :math:`j`.
        """
