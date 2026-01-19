from datetime import datetime

import albumentations as A
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from loguru import logger

from wizard_ops.data import NutritionDataModule, get_default_transforms
from wizard_ops.model import DishMultiViewRegressor


def train(
    config: dict,
    train_transform: A.Compose = None,
    val_transform: A.Compose = None,
) -> None:
    """
    Train a DishMultiViewRegressor model.

    Args:
        config (dict): Configuration dictionary containing training parameters.

    Returns:
        None
    """
    if val_transform is None:
        val_transform = get_default_transforms(image_size=224)

    if train_transform is None:
        train_transform = A.Compose(
            [
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.ToTensorV2(),
        ]
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

    run_name = f"nutrition_{config["model"]["backbone"]}_{datetime.now().strftime('%m%d_%H%M')}"
    logger_type = config["logging"]["type"].lower()
    if logger_type == "tensorboard":
        train_logger = TensorBoardLogger(save_dir="logs", name=run_name)
    elif logger_type == "wandb":
        train_logger = WandbLogger(
            project="nutrition-predictor", name=run_name, log_model="all"
        )
    else:
        raise typer.BadParameter(
            f"Invalid logger_type: {logger_type}. Must be 'tensorboard' or 'wandb'."
        )

    callbacks = [
        RichProgressBar(),
        # Saves the best model based on validation loss
        ModelCheckpoint(
            monitor="val/loss",
            dirpath=f"{config["train"]["checkpoint_dir"]}/{run_name}",
            filename="best-nutrition-{epoch:02d}-{val-loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    model = DishMultiViewRegressor(
        backbone=config["model"]["backbone"],
        image_size=config["model"]["image_size"],
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
        accelerator=config["train"]["accelerator"],
        devices=config["train"].get("devices", None),
        max_epochs=config["train"]["max_epochs"],
        precision=config["train"].get("precision", 32),
        fast_dev_run=config["train"].get("fast_dev_run", False),
        logger=train_logger,
        callbacks=callbacks,
        log_every_n_steps=config["train"].get("log_every_n_steps", 50),
        enable_model_summary=True,
    )
    
    logger.info(f"Starting run: {run_name}")
    trainer.fit(model, datamodule=dataset)
    
    logger.info("Training complete. Saving final model...")
    torch.save(model, 
        f"{config["train"]["checkpoint_dir"]}/{run_name}/final_model.pth"
    )