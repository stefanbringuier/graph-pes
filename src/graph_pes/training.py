from __future__ import annotations

from pathlib import Path
from typing import Callable, TypeVar

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .core import GraphPESModel, get_predictions
from .data import AtomicGraph
from .data.batching import AtomicDataLoader, AtomicGraphBatch
from .loss import RMSE, Loss, WeightedLoss
from .transform import Chain, PerAtomScale, Scale
from .util import Keys

Model = TypeVar("Model", bound=GraphPESModel)


def train_model(
    model: Model,
    train_data: list[AtomicGraph],
    val_data: list[AtomicGraph] | None = None,
    optimizer: Callable[[Model], torch.optim.Optimizer] | None = None,
    loss: WeightedLoss | Loss | None = None,
    property_labels: dict[Keys, str] | None = None,
    *,
    batch_size: int = 32,
    pre_fit_model: bool = True,
    # pytorch lightning
    **trainer_kwargs,
):
    # TODO check using a strict flag that all the data have the same keys
    batch = AtomicGraphBatch.from_graphs(train_data)

    # check that the property keys are valid
    if property_labels is None:
        property_labels = get_existing_keys(batch)
        if not property_labels:
            expected = [key.value for key in Keys.__members__.values()]
            raise ValueError(
                "No property_keys were provided, and none were found in "
                f"the data. Expected at least one of: {expected}"
            )
    else:
        missing = {
            label for label in property_labels.values() if label not in batch
        }
        if missing:
            raise ValueError(
                f"Input data is missing the following keys that were "
                f"requested: {missing}"
            )

    expected_shapes = {
        Keys.ENERGY: (batch.n_structures,),
        Keys.FORCES: (batch.n_atoms, 3),
        Keys.STRESS: (batch.n_structures, 3, 3),
    }
    for key, label in property_labels.items():
        if batch[label].shape != expected_shapes[key]:
            raise ValueError(
                f"Expected {label} to have shape {expected_shapes[key]}, "
                f"but found {batch[label].shape}"
            )
    if Keys.STRESS in property_labels and not batch.has_cell:
        raise ValueError("Can't train on stress without cell information.")

    # create the data loaders
    train_loader = AtomicDataLoader(train_data, batch_size, shuffle=True)
    val_loader = (
        AtomicDataLoader(val_data, batch_size, shuffle=False)
        if val_data is not None
        else None
    )

    # deal with fitting transforms
    # TODO: what if not training on energy?
    if pre_fit_model and Keys.ENERGY in property_labels:
        model.pre_fit(batch, property_labels[Keys.ENERGY])

    actual_loss = get_loss(loss, property_labels)
    actual_loss.fit_transform(batch)

    # deal with the optimizer
    if optimizer is None:
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    else:
        opt = optimizer(model)

    # create the task (a pytorch lightning module)
    task = LearnThePES(model, opt, actual_loss, property_labels)

    # create the trainer
    kwargs = default_trainer_kwargs()
    kwargs.update(trainer_kwargs)
    trainer = pl.Trainer(**kwargs)

    # train
    trainer.fit(task, train_loader, val_loader)

    # load the best weights
    return task.load_best_weights(model, trainer)


def get_existing_keys(batch: AtomicGraphBatch) -> dict[Keys, str]:
    return {
        key: key.value
        for key in Keys.__members__.values()
        if key.value in batch
    }


class LearnThePES(pl.LightningModule):
    def __init__(
        self,
        model: GraphPESModel,
        optimizer: torch.optim.Optimizer,
        loss: WeightedLoss,
        property_labels: dict[Keys, str],
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.property_labels = property_labels

    def forward(self, graphs: AtomicGraphBatch):
        return self.model(graphs)

    def _step(self, graph: AtomicGraphBatch, prefix: str):
        """
        Get (and log) the losses for a training/validation step.
        """

        def log(name, value, verbose=True):
            return self.log(
                f"{prefix}_{name}",
                value,
                prog_bar=verbose and prefix == "val",
                on_step=False,
                on_epoch=True,
                batch_size=graph.n_structures,
            )

        # generate prediction:
        predictions = get_predictions(self.model, graph, self.property_labels)

        # compute the losses
        total_loss = torch.scalar_tensor(0.0, device=self.device)

        for loss, weight in zip(self.loss.losses, self.loss.weights):
            value = loss(predictions, graph)
            # log the unweighted components of the loss
            log(f"{loss.property_key}_{loss.name}", value)
            # but weight them when computing the total loss
            total_loss = total_loss + weight * value

            # log the raw values only during validation
            if prefix == "val":
                raw_value = loss.raw(predictions, graph)
                log(f"{loss.property_key}_raw_{loss.name}", raw_value)

        log("total_loss", total_loss)
        return total_loss

    def training_step(self, structure: AtomicGraphBatch, _):
        return self._step(structure, "train")

    def validation_step(self, structure: AtomicGraphBatch, _):
        return self._step(structure, "val")

    def configure_optimizers(self):
        return self.optimizer

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        # we override the defaults to turn on gradient tracking for the
        # validation step since we (might) compute the forces using autograd
        torch.set_grad_enabled(True)

    @classmethod
    def load_best_weights(
        cls,
        model: GraphPESModel,
        trainer: pl.Trainer | None = None,
        checkpoint_path: Path | str | None = None,
    ) -> GraphPESModel:
        if checkpoint_path is None and trainer is None:
            raise ValueError(
                "Either trainer or checkpoint_path must be provided"
            )
        if checkpoint_path is None:
            path = trainer.checkpoint_callback.best_model_path  # type: ignore
        else:
            path = Path(checkpoint_path)
        checkpoint = torch.load(path)
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict)
        return model


def get_loss(
    loss: WeightedLoss | Loss | None, property_labels: dict[Keys, str]
) -> WeightedLoss:
    if loss is None:
        default_transforms = {
            Keys.ENERGY: Chain([PerAtomScale(), PerAtomScale()]),
            Keys.FORCES: PerAtomScale(),
            Keys.STRESS: Scale(),
        }
        default_weights = {
            Keys.ENERGY: 1.0,
            Keys.FORCES: 1.0,
            Keys.STRESS: 1.0,
        }
        return WeightedLoss(
            [
                Loss(
                    label,
                    metric=RMSE(),
                    transform=default_transforms[key],
                )
                for key, label in property_labels.items()
            ],
            [default_weights[key] for key in property_labels],
        )
    elif isinstance(loss, Loss):
        return WeightedLoss([loss], [1.0])
    else:
        return loss


def default_trainer_kwargs() -> dict:
    es_callback = EarlyStopping(
        monitor="val_total_loss",
        patience=75,
        mode="min",
        min_delta=1e-3,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch}-{val_total_loss:.4f}",
    )
    return {
        "accelerator": "auto",
        "max_epochs": 100,
        "callbacks": [es_callback, checkpoint_callback],
    }
