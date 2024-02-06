from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from graph_pes.core import GraphPESModel, get_predictions
from graph_pes.data import convert_to_atomic_graph


class GraphPESCalculator(Calculator):
    """
    ASE calculator wrapping any GraphPESModel.

    Parameters
    ----------
    model
        The model to use for the calculation.
    cutoff
        The cutoff radius for the atomic environment.
    device
        The device to use for the calculation, e.g. "cpu" or "cuda".
    **kwargs
        Properties passed to the :code:`ase` base class.
    """

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        model: GraphPESModel,
        cutoff: float,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model.to(device)
        self.cutoff = cutoff
        self.device = device

    def calculate(
        self,
        atoms: Atoms,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ):
        if properties is None:
            properties = ["energy"]

        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        graph = convert_to_atomic_graph(atoms, self.cutoff).to(self.device)
        predictions = get_predictions(self.model, graph)
        results = {
            "energy": predictions["energy"].detach().cpu().item(),
            "forces": predictions["forces"].detach().cpu().numpy(),
            "stress": predictions["stress"].detach().cpu().numpy(),
        }
        self.results = {key: results[key] for key in properties}
