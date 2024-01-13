from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
from graph_pes.core import GraphPESModel, energy_and_forces, get_predictions
from graph_pes.data import AtomicGraph
from graph_pes.data.batching import AtomicDataLoader, AtomicGraphBatch
from graph_pes.loss import RMSE, Loss
from graph_pes.transform import PerSpeciesScale, PerSpeciesShift, Scale
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import optim


def train_model(
    model: GraphPESModel,
    train_data: list[AtomicGraph] | AtomicDataLoader,
    val_data: list[AtomicGraph] | AtomicDataLoader | None = None,
    optimizer: optim.Optimizer | None = None,
    lr: float | None = None,
    losses: list[Loss] | None = None,
    *,
    # optional data loader argument
    batch_size: int | None = None,
    # transform fitting
    fit_model_transform: bool = True,
    fit_loss_transforms: bool = True,
    # pytorch lightning
    dir: str | None = None,
    name: str | None = None,
    early_stopping: bool = True,
    lr_decay: float | None = None,
    **trainer_kwargs,
):
    # ensure that the data are converted to AtomicDataLoaders
    train_loader = to_data_loader(train_data, batch_size, shuffle=True)
    data_loaders = [train_loader]

    if val_data is not None:
        val_loader = to_data_loader(val_data, batch_size, shuffle=False)
        data_loaders.append(val_loader)

    # deal with fitting transforms
    # convert the train data into a single batch
    batch = AtomicGraphBatch.from_graphs(train_loader.dataset)  # type: ignore
    if fit_model_transform:
        for transform in model._energy_transforms.values():
            transform.fit_to_target(batch.labels["energy"], batch)  # type: ignore

    actual_losses: list[Loss] = losses or default_loss_fns()
    if fit_loss_transforms:
        for loss in actual_losses:
            loss.fit_transform(batch)

    # deal with the optimizer
    if optimizer is None:
        if lr is None:
            lr = 3e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if lr_decay is not None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
        optimizer = {"optimizer": optimizer, "lr_scheduler": scheduler}

    # create the task (a pytorch lightning module)
    task = LearnThePES(
        model=model,
        optimizer=optimizer,
        losses=actual_losses,
    )

    # create the trainer
    kwargs = default_trainer_kwargs(early_stopping, dir, name)
    kwargs.update(trainer_kwargs)
    trainer = pl.Trainer(**kwargs)

    # train
    trainer.fit(task, *data_loaders)  # type: ignore

    # load the best model
    # try:
    return task.load_best_weights(model, trainer)
    # except:
    # return model


class LearnThePES(pl.LightningModule):
    def __init__(
        self,
        model: GraphPESModel,
        optimizer: optim.Optimizer,
        losses: list[Loss],
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.losses = losses or default_loss_fns()

    def forward(self, graphs: AtomicGraphBatch):
        return self.model(graphs)

    def _step(self, graph: AtomicGraphBatch, prefix: str):
        """
        Get (and log) the losses for a training/validation step.
        """

        # avoid long, repeated calls to self.log with broadly the same
        # arguments by defining:
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
        predictions = get_predictions(self.model, graph)._asdict()

        # compute the losses
        total_loss = torch.scalar_tensor(0.0, device=self.device)

        for loss in self.losses:
            value = loss(predictions, graph)
            # log the unweighted components of the loss
            log(f"{loss.property_key}_{loss.name}", value)
            # but weight them when computing the total loss
            total_loss = total_loss + loss.weight * value

            # log the raw values only during validation (as extra, harmless info)
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
        # validation step since we compute the forces using autograd
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
            path = trainer.checkpoint_callback.best_model_path
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


def default_loss_fns() -> list[Loss]:
    # TODO should defailt to whatever is in the inputs
    return [
        Loss(
            "energy",
            metric=RMSE(),
            transforms=[PerSpeciesShift(), PerSpeciesScale()],
        ),
        Loss(
            "forces",
            metric=RMSE(),
            transform=PerSpeciesScale(),
        ),
        # Loss(
        #     "stress",
        #     metric=RMSE(),
        #     transform=Scale(),
        # ),
    ]


def default_trainer_kwargs(early_stopping, dir, name) -> dict:
    es_callback = EarlyStopping(
        monitor="val_total_loss",
        patience=75,
        mode="min",
        min_delta=1e-3,
    )
    chekcpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch}-{val_total_loss:.4f}",
    )
    callbacks = (
        [es_callback, chekcpoint_callback]
        if early_stopping
        else [chekcpoint_callback]
    )
    return {
        "accelerator": "auto",
        "max_epochs": 100,
        "callbacks": callbacks,
        "logger": TensorBoardLogger(".", name=dir, version=name),
    }


def to_data_loader(
    data: AtomicDataLoader | list[AtomicGraph],
    batch_size: int | None = None,
    **kwargs,
):
    if not isinstance(data, AtomicDataLoader):
        batch_size = batch_size or 32
        return AtomicDataLoader(data, batch_size=batch_size, **kwargs)

    if batch_size is not None:
        data.batch_size = batch_size

    return data
