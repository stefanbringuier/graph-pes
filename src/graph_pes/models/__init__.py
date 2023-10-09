from collections import namedtuple

import torch

from graph_pes.data.atomic_graph import AtomicGraph

from .base import GraphPESModel

EnergyAndForces = namedtuple("EnergyAndForces", ["energy", "forces"])


def energy_and_forces(pes: GraphPESModel, structure: AtomicGraph):
    """
    Evaluate the `pes`  on `structure` to obtain both the
    total energy and the forces on each atom.

    Parameters
    ----------
    pes : PES
        The PES to use.
    structure : AtomicStructure
        The atomic structure to evaluate.

    Returns
    -------
    EnergyAndForces
        The energy of the structure and forces on each atom.
    """

    # TODO handle the case where isolated atoms are present
    # such that the gradient of energy wrt their positions
    # is zero.

    # use the autograd machinery to auto-magically
    # calculate forces for (almost) free
    structure._positions.requires_grad_(True)
    energy = pes(structure)
    forces = -torch.autograd.grad(
        energy.sum(), structure._positions, create_graph=True
    )[0]
    structure._positions.requires_grad_(False)
    return EnergyAndForces(energy=energy.squeeze(), forces=forces)
