from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import torch
from graph_pes.core import GraphPESModel, energy_and_forces
from graph_pes.data import AtomicGraph, AtomicGraphBatch
from graph_pes.transform import (
    Chain,
    PerSpeciesOffset,
    PerSpeciesScale,
    Transform,
)


def parity_plots(
    model: GraphPESModel,
    graphs: list[AtomicGraph],
    E_transform: Transform | None = None,
    F_transform: Transform | None = None,
    axs: Sequence[plt.Axes] | None = None,
    E_kwargs: dict[str, Any] | None = None,
    F_kwargs: dict[str, Any] | None = None,
    **kwargs,
):
    if E_transform is None:
        E_transform = Chain([PerSpeciesScale(), PerSpeciesOffset()])
    if F_transform is None:
        F_transform = PerSpeciesScale()

    batch = AtomicGraphBatch.from_graphs(graphs)

    true_E, true_F = batch.labels["energy"], batch.labels["forces"]

    E_transform.fit_to_target(true_E, batch)
    F_transform.fit_to_target(true_F, batch)

    preds = energy_and_forces(model, batch)
    pred_E, pred_F = preds["energy"].detach(), preds["forces"].detach()

    with torch.no_grad():
        scaled_pred_E = E_transform.inverse(pred_E, batch)
        scaled_true_E = E_transform.inverse(true_E, batch)
        scaled_pred_F = F_transform.inverse(pred_F, batch)
        scaled_true_F = F_transform.inverse(true_F, batch)

    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(6, 3))  # type: ignore

    E_ax, F_ax = axs  # type: ignore

    E_defaults = dict(marker="+")
    E_kwargs = {**E_defaults, **kwargs, **(E_kwargs or {})}
    E_ax.scatter(scaled_true_E, scaled_pred_E, **E_kwargs)
    E_ax.axline((0, 0), slope=1, color="k", ls="--", lw=1)
    E_ax.set_aspect("equal", "datalim")
    E_ax.set_xlabel(r"$E$    (a.u.)")
    E_ax.set_ylabel(r"$\tilde{E}$    (a.u.)")

    F_defaults = dict(lw=0, s=3, alpha=0.2)
    F_kwargs = {**F_defaults, **kwargs, **(F_kwargs or {})}
    F_ax.scatter(scaled_true_F, scaled_pred_F, **F_kwargs)
    F_ax.axline((0, 0), slope=1, color="k", ls="--", lw=1)
    F_ax.set_aspect("equal", "datalim")
    F_ax.set_xlabel(r"$F$     (a.u.)")
    F_ax.set_ylabel(r"$\tilde{F}$     (a.u.)")

    return axs
