from datetime import datetime
from typing import Annotated

import albumentations as A
import hydra
import torch
import typer
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from loguru import logger
from torch.utils.data import DataLoader

from wizard_ops.data import NutritionDataModule, NutritionFastDataModule, get_default_transforms
from wizard_ops.model import DishMultiViewRegressor

app = typer.Typer(help="Commands to train nutrition predictor.")

ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

config = {
    "seed": 42,
    "batch_size": 32,
    "lr": 1e-2,
    "max_epochs": 10,
    "camera": "D",
    "num_outputs": 5,
    "train_val_split": 0.2,
}


@app.command()
def train(num_workers: int = 4, 
          fast_dev_run: bool = False,
          logger_type: Annotated[str, typer.Option(help="Logger to use for training", case_sensitive=False)] = "wandb") -> None:
    """
    Train a NutritionPredictor model using the provided dataset and configuration.

    Args:
        frame_idx (int, optional): Frame index for dataset processing. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 0.
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 10.
        fast_dev_run (bool, optional): If True, runs a quick validation run for debugging. Defaults to False.

    Returns:
        None
    """

    seed = config["seed"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    max_epochs = config["max_epochs"]
    train_val_split = config["train_val_split"]

    val_transform = get_default_transforms(image_size=224)

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),  # Data augmentation only for training
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        A.ToTensorV2(),
    ])

    val_transform = get_default_transforms(image_size=224)
    # dataset = NutritionDataModule(
    #     data_path="data.nosync",
    #     dish_csv="src/wizard_ops/metadata/data_stats.csv",
    #     batch_size=batch_size,
    #     image_size=224,
    #     train_transform=train_transform,
    #     val_transform=val_transform,
    #     normalise_dish_metadata=True,
    #     val_split=train_val_split,
    #     num_workers=num_workers,
    #     use_only_dishes_with_all_cameras=True,
    #     seed=seed
    # )
    dataset = NutritionFastDataModule(
        h5_path="data.nosync/images_224.h5",
        dish_csv="configs/metadata/data_stats.csv",
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        val_split=train_val_split,
        num_workers=num_workers,
        seed=seed
    )

    # Setup Logger based on user choice
    run_name = f"nutrition_resnet18_{datetime.now().strftime('%m%d_%H%M')}"
    logger_type = logger_type.lower()
    
    if logger_type == "tensorboard":
        train_logger = TensorBoardLogger(save_dir="logs", name=run_name)
    elif logger_type == "wandb":
        train_logger = WandbLogger(project="nutrition-predictor", name=run_name, log_model="all")
    else:
        raise typer.BadParameter(f"Invalid logger_type: {logger_type}. Must be 'tensorboard' or 'wandb'.")

    callbacks = [
        RichProgressBar(),
        # Saves the best model based on validation loss
        ModelCheckpoint(
            monitor="val/loss",
            dirpath=f"checkpoints/{run_name}",
            filename="best-nutrition-{epoch:02d}-{val-loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step")
    ]

    model = DishMultiViewRegressor(
        lr=lr,
        view_dropout_p=0.3,
        hidden_dim=512
    )

    trainer = Trainer(accelerator=ACCELERATOR, 
                      max_epochs=max_epochs, 
                      fast_dev_run=fast_dev_run,
                      logger=train_logger,
                      callbacks=callbacks,
                      log_every_n_steps=1,
                      enable_model_summary=True      
)
    logger.info(f"Starting run: {run_name}")
    trainer.fit(model, datamodule=dataset)
    # Save the final model
    torch.save(model.state_dict(), "models/model.pth")

if __name__ == "__main__":
    app()
