# `graph-pes` - Potential Energy Surfaces on Graphs

`graph-pes` is a Python framework for training Potential Energy Surface (PES) 
models that operate on graph representations of atomic structures.
Under-the-hood, this relies on Pytorch/Geometric for efficient tensor operations.

Batteries are included:
- **easy data manipulations** : see docs relating to `AtomicGraph`, intuitive batching, easy conversion from ase etc.
- **easy construction of PES models** : implement `predict_local_energies` , easy to save, load and share
- **useful primitives** : PerSpeciesParameter
- **easy training** : forces, energies, well conditioned losses etc.

## Installation

```bash
pip install graph-pes
```

## Minimal example

```python
from graph_pes import GraphPESModel
from graph_pes.data import convert_to_graphs, AtomicGraph
from graph_pes.training import train_model
from ase.io import read
import torch

# 1. define a model
class LennardJones(GraphPES):
    def __init__(self):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(1.0))
        self.epsilon = nn.Parameter(torch.tensor(1.0))

    def predict_local_energies(self, graph: AtomicGraph) -> torch.Tensor:
        central_atoms, neighbours = graph.neighbour_index
        distances = graph.neighbour_distances
        
        # calculate pairwise interactions
        x = (self.sigma / distances)**6
        pairwise_interaction = 4 * self.epsilon * (x**2 - x)

        # sum over neighbours to get per-atom contributions
        return torch.scatter_add(
            torch.zeros(graph.n_atoms, device=graph.device),
            central_atoms,
            pairwise_interaction
        )

# 2. load some structures
structures = read('structures.xyz', index=':10')

# 3. convert to graphs (e.g. using a radius cutoff)
graphs = convert_to_graphs(structures, cutoff=5.0)

# 4. train the model
model = LennardJones()
train_model(model, graphs, max_epochs=100)
```