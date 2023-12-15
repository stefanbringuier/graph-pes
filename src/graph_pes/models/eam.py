"""
Embedded Atom Models (EAM)
"""

from __future__ import annotations

from typing import Callable

import torch
from graph_pes.core import GraphPESModel
from graph_pes.data import AtomicGraph
from torch_geometric.utils import scatter

DistanceEmbedding = Callable[[torch.Tensor], torch.Tensor]
"""
embeds a distance tensor into a feature space
"""
Readout = Callable[[torch.Tensor], torch.Tensor]
"""
reads out a single value from a feature tensor
"""


class EAM(GraphPESModel):
    def __init__(
        self,
        embedding: DistanceEmbedding,
        readout: Readout,
        aggregation: str = "sum",
    ):
        super().__init__()
        self.embedding = embedding
        self.readout = readout
        self.aggregation = aggregation

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        # 1. embed the distances
        distance_embeddings = self.embedding(
            graph.neighbour_distances.view(-1, 1)
        )
        # 2. aggregate the embeddings over the neighbours
        central_atoms, neighbours = graph.neighbour_index
        agg_embeddings = scatter(
            distance_embeddings.squeeze(),
            central_atoms,
            dim=0,
            reduce=self.aggregation,
        ).view(graph.n_atoms, -1)
        # 3. read out the local energies
        return self.readout(agg_embeddings)
