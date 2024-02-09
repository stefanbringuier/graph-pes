from ase.io import read
from graph_pes.data import convert_to_atomic_graphs, random_split
from graph_pes.models.pairwise import LennardJones
from graph_pes.training import train_model

# load data using ASE
atoms = read("structures.xyz", index=":200")
graphs = convert_to_atomic_graphs(atoms, cutoff=3.0)
train, val, test = random_split(graphs, [160, 20, 20])

# train the model
best_model = train_model(LennardJones(), train, val)

# make predictions
test_predictions = best_model.predict(test)
# {'energy': ..., 'forces': ..., 'stress': ...}
