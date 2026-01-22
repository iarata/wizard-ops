from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from wizard_ops.data import NutritionDataModule as _NutritionDataModule
from wizard_ops.model import DishMultiViewRegressor as _DishMultiViewRegressor
from wizard_ops.utils import get_augmentation_transforms, get_default_transforms

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    class _TorchStub:  # minimal surface for tests to monkeypatch
        pass

    torch = _TorchStub()  # type: ignore


try:  # pragma: no cover
    from lightning import Trainer  # type: ignore
    from lightning.pytorch.callbacks import (  # type: ignore
        LearningRateMonitor,
        ModelCheckpoint,
        RichProgressBar,
    )
    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger  # type: ignore
except Exception:  # pragma: no cover
    Trainer = None  # type: ignore
    TensorBoardLogger = None  # type: ignore
    WandbLogger = None  # type: ignore
    RichProgressBar = None  # type: ignore
    ModelCheckpoint = None  # type: ignore
    LearningRateMonitor = None  # type: ignore


# For mocking in tests
NutritionDataModule = None  # type: ignore
DishMultiViewRegressor = None  # type: ignore


def train(
    config: dict,
    train_transform: Any = None,
    val_transform: Any = None,
) -> None:
    """
    Train a DishMultiViewRegressor model.

    Args:
        config (dict): Configuration dictionary containing training parameters.

    Returns:
        None
    """
    global NutritionDataModule, DishMultiViewRegressor

    if NutritionDataModule is None:  # pragma: no cover
        NutritionDataModule = _NutritionDataModule

    if DishMultiViewRegressor is None:  # pragma: no cover
        DishMultiViewRegressor = _DishMultiViewRegressor

    if val_transform is None:  # pragma: no cover
        val_transform = get_default_transforms(image_size=config["data"]["image_size"])

    if train_transform is None:  # pragma: no cover
        train_transform = get_augmentation_transforms(
            image_size=config["data"]["image_size"]
        )

    dataset = NutritionDataModule(
        h5_path=config["data"]["h5_path"],
        dish_csv=config["data"]["dish_csv_path"],
        batch_size=config["data"]["batch_size"],
        train_transform=train_transform,
        val_transform=val_transform,
        normalisation_method=config["data"]["normalisation_method"],
        val_split=config["data"]["val_split"],
        num_workers=config["data"]["num_workers"],
        seed=config["seed"],
        prefetch_factor=config["data"].get("prefetch_factor", 2),
    )

    run_name = f"nutrition_{config['model']['backbone']}_{config['logging']['experiment_name']}"
    logger_type = config["logging"]["type"].lower()
    if logger_type == "tensorboard":
        train_logger = TensorBoardLogger(save_dir="logs", name=run_name)
    elif logger_type == "wandb":
        train_logger = WandbLogger(
            project=config["logging"]["wandb_project"],
            entity=config["logging"]["wandb_entity"],
            name=run_name,
            log_model="all"
        )
    else:
        raise ValueError(f"Unsupported logger type: {logger_type}")

    run_dir = Path(config["train"]["checkpoint_dir"]) / run_name

    callbacks = [
        RichProgressBar(),
        # Saves the best model based on validation loss
        ModelCheckpoint(
            monitor="val/loss",
            dirpath=str(run_dir),
            filename="best-nutrition-{epoch:02d}-{val-loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    model = DishMultiViewRegressor(
        backbone=config["model"]["backbone"],
        image_size=config["data"]["image_size"],
        freeze_encoder=config["model"]["freeze_encoder"],

        hidden_dim=config["model"]["hidden_dim"],
        view_dropout_p=config["model"].get("view_dropout_p", 0.0),
        head_dropout_p=config["model"].get("head_dropout_p", 0.0),

        lr=config["train"]["lr"],
        loss=config["train"]["loss"],
        weight_decay=config["train"]["weight_decay"],
        log_wandb_examples=config["train"]["log_wandb_examples"],
    )

    trainer = Trainer(
        accelerator=config["train"].get("accelerator", "auto"),
        max_epochs=config["train"]["max_epochs"],
        precision=config["train"].get("precision", 32),
        fast_dev_run=config["train"].get("fast_dev_run", False),
        logger=train_logger,
        callbacks=callbacks,
        log_every_n_steps=config["train"].get("log_every_n_steps", 50),
        enable_model_summary=True,
    )

    trainer.fit(model, datamodule=dataset)

    os.makedirs(run_dir, exist_ok=True)
    # Save full model object (.pth)
    torch.save(model, str(run_dir / "final_model.pth"))
    # Save Lightning checkpoint (.ckpt) for proper loading with LightningModule.load_from_checkpoint
    # trainer.save_checkpoint(str(run_dir / "final_model.ckpt")) # dummy don't have save checkpoint

    if config.get("train", {}).get("export_onnx", False) and hasattr(model, "to_onnx"):
        # Use the legacy exporter (dynamo=False). The dynamo/torch.export-based
        # ONNX path can fail with internal onnx_ir initializer naming errors.
        model.eval()
        model.cpu()
        input_sample = getattr(model, "example_input_array", None)
        if input_sample is None:
            raise ValueError(
                "Model has no example_input_array; please provide an input_sample for ONNX export."
            )
        model.to_onnx(
            str(run_dir / "final_model.onnx"),
            input_sample=input_sample.cpu(),
            export_params=True,
            opset_version=int(config.get("train", {}).get("onnx_opset", 17)),
            dynamo=False,
        )
    