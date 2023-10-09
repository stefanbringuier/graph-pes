from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase import Atoms

from graph_pes.data.atomic_graph import AtomicGraph, convert_to_atomic_graph
from graph_pes.data.batching import AtomicGraphBatch
from graph_pes.models import GraphPESModel, energy_and_forces


def parity_plots(
    model: GraphPESModel, graphs: list[AtomicGraph], axs=None, **kwargs
):
    model.eval()

    energies, forces = [], []
    pred_energies, pred_forces = [], []
    n_atoms = []

    for graph in graphs:
        preds = energy_and_forces(model, graph)
        energies.append(graph.labels["energy"].item())
        forces.append(graph.labels["forces"].numpy())
        pred_energies.append(preds.energy.detach().item())
        pred_forces.append(preds.forces.detach().numpy())
        n_atoms.append(graph.n_atoms)

    energies = np.array(energies)
    forces = np.vstack(forces)
    pred_energies = np.array(pred_energies)
    pred_forces = np.vstack(pred_forces)
    n_atoms = np.array(n_atoms)
    energies /= n_atoms
    pred_energies /= n_atoms

    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    # energy parity plot
    e0 = energies.mean()
    axs[0].scatter(energies - e0, pred_energies - e0, s=10, lw=0, **kwargs)
    axs[0].axline((0, 0), slope=1, color="k", ls="--", lw=1)
    axs[0].set_aspect("equal", "datalim")
    axs[0].set_xlabel(r"$E$   (eV/atom)")
    axs[0].set_ylabel(r"$\hat{E}$   (eV/atom)")

    # force parity plot
    axs[1].scatter(
        forces.flatten(),
        pred_forces.flatten(),
        s=10,
        lw=0,
        alpha=0.2,
        **kwargs,
    )
    axs[1].axline((0, 0), slope=1, color="k", ls="--", lw=1)
    axs[1].set_aspect("equal", "datalim")
    axs[1].set_xlabel(r"$\mathbf{F}$   (eV/Å)")
    axs[1].set_ylabel(r"$\hat{\mathbf{F}}$   (eV/Å)")

    plt.tight_layout()
    _xticks = axs[0].get_xticks()
    axs[0].set_yticks(_xticks)
    _xticks = axs[1].get_xticks()
    axs[1].set_yticks(_xticks)


def dimer_curve(
    model: GraphPESModel,
    dimer: str,
    r_min: float = 0.25,
    r_max: float = 5,
    ax=None,
    **kwargs,
):
    distances = np.linspace(r_min, r_max, 200)

    def _dimer(r):
        return Atoms(dimer, positions=[[0, 0, 0], [0, 0, r]], pbc=False)

    graphs = [
        convert_to_atomic_graph(_dimer(r), cutoff=r_max + 1) for r in distances
    ]

    batch = AtomicGraphBatch.from_graphs(graphs)

    with torch.no_grad():
        energies = model(batch).detach().numpy() / 2

    if ax is None:
        ax = plt.gca()

    plot_kwargs = dict(label="$-$".join(dimer))
    plot_kwargs.update(kwargs)
    ax.plot(distances, energies - energies[-1], **plot_kwargs)
    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("Energy (eV/atom)")
