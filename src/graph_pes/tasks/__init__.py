from __future__ import annotations

import warnings
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import optim

from graph_pes.data import AtomicGraph
from graph_pes.data.batching import AtomicGraphBatch, sum_per_structure
from graph_pes.models import energy_and_forces
from graph_pes.models.base import GraphPESModel
from graph_pes.models.transforms import EnergyNormaliser, ForceNormaliser

from .loss import Loss, WeightedLoss, default_losses


class LearnThePES(pl.LightningModule):
    """
    A PyTorch Lightning module for learning the PES.

    The task this represents is of training a model to predict the total
    energy of a structure, given the positions of the atoms in that structure.
    """

    def __init__(
        self,
        model: GraphPESModel,
        loss: Loss | list[Loss] | None = None,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.lr = lr

        if loss is None:
            self.losses = default_losses()
        elif isinstance(loss, list):
            self.losses = loss
        else:
            self.losses = [loss]

        self._energy_normaliser = EnergyNormaliser()
        self._force_normaliser = ForceNormaliser()

    def fit_transforms_to(self, train_data: list[AtomicGraph]) -> None:
        """
        Fit the local energy transform to the provided data.

        Parameters
        ----------
        train_data : list[AtomicGraph]
            The training data.
        """
        self.model.fit_transform_to(train_data)
        self._energy_normaliser.fit_to(train_data)
        self._force_normaliser.fit_to(train_data)

    def forward(self, graph: AtomicGraphBatch) -> torch.Tensor:
        return self.model(graph)

    def training_step(self, structure: AtomicGraphBatch, _):
        return self._step(structure, "train")

    def validation_step(self, structure: AtomicGraphBatch, _):
        return self._step(structure, "val")

    def _step(self, graph: AtomicGraphBatch, prefix: str):
        # avoid long, repeated calls to self.log with broadly the same
        # arguments by defining:
        log = lambda name, value, verbose=True: self.log(
            f"{prefix}_{name}",
            value,
            prog_bar=verbose,
            on_step=False,
            on_epoch=True,
            batch_size=graph.n_structures,
        )

        # generate prediction:
        E_pred, F_pred = energy_and_forces(self.model, graph)
        E, F = graph.labels["energy"], graph.labels["forces"]

        # perform suitable normalisation:
        normed_E_pred, normed_E = [
            self._energy_normaliser(x, graph) for x in [E_pred, E]
        ]
        normed_F_pred, normed_F = [
            self._force_normaliser(x, graph) for x in [F_pred, F]
        ]

        # place in dict for easy access in what follows:
        raw_quantities = {"energy": (E_pred, E), "forces": (F_pred, F)}
        normed_quantities = {
            "energy": (normed_E_pred, normed_E),
            "forces": (normed_F_pred, normed_F),
        }

        # compute the losses
        total_loss = torch.scalar_tensor(0.0, device=self.device)

        for loss in self.losses:
            prediction, ground_truth = normed_quantities[loss.target]

            if isinstance(loss, WeightedLoss):
                value, weight = loss(prediction, ground_truth, graph)
            else:
                value = loss(prediction, ground_truth)
                weight = torch.scalar_tensor(1.0, device=self.device)

            log(f"{loss.target}_{loss.name}", value)
            total_loss = total_loss + weight * value

            # log the raw values only during validation (as extra, harmless info)
            if prefix == "val":
                raw_prediction, raw_ground_truth = raw_quantities[loss.target]
                raw_value = loss.metric(raw_prediction, raw_ground_truth)
                log(f"{loss.target}_raw_{loss.name}", raw_value, verbose=False)

        log("total_loss", total_loss)
        return total_loss

    def setup(self, stage):
        if not self.model._transform_has_been_fit:
            warnings.warn(
                "The model's local energy transform has not been fit to "
                "any data. This may cause problems when training. "
                "To rectify this, call `task.fit_transforms_to(train_data)`."
            )

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        # we override the defaults to turn on gradient tracking for the
        # validation step since we compute the forces using autograd
        torch.set_grad_enabled(True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    @classmethod
    def load_best_weights(
        cls,
        model: GraphPESModel,
        trainer: pl.Trainer,
    ) -> None:
        checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict)
