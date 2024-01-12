from ase.io import read
from graph_pes.data import (
    AtomicDataLoader,
    convert_to_atomic_graphs,
    random_split,
)
from graph_pes.loss import RMSE, Loss
from graph_pes.models.pairwise import LennardJones

# 1. Define and initialize the model
model = LennardJones(epsilon=1.0, sigma=1.0)

# 2. Load the training data
structures = read("data/training_data.xyz", ":100")
graphs = convert_to_atomic_graphs(structures, cutoff=5.0)
train, val = random_split(graphs, [90, 10], seed=42)

# 3. Define loss components
loss_components = [
    Loss("energy", weight=10),  # by default, uses MAE as the metric
    Loss("forces", metric=RMSE(), weight=1),
]

# 4. Fit summary statistics and transforms
model._energy_summation.fit_to_graphs(train)
for loss in loss_components:
    loss.fit_transform(train)

# 5. Train the model
train_loader, val_loader = AtomicDataLoader(train), AtomicDataLoader(val)

# typical PyTorch training loop
...
