from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
import typer
from typing import Annotated
from loguru import logger
from datetime import datetime

from wizard_ops.data import NutritionDataset, get_default_transforms
from wizard_ops.model import NutritionPredictor

app = typer.Typer(help="Commands to train nutrition predictor.")

ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


config = {
    "seed": 42,
    "batch_size": 32,
    "lr": 1e-2,
    "max_epochs": 10,
    "camera": "D",
    "num_outputs": 5,
    "train_val_test_split": (0.7, 0.15, 0.15),
}


@app.command()
def train(frame_idx: int = 1, 
          num_workers: int = 4, 
          fast_dev_run: bool = False,
          logger_type: Annotated[str, typer.Option(help="Logger to use for training", case_sensitive=False)] = "tensorboard") -> None:
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
    camera = config["camera"]
    num_outputs = config["num_outputs"]
    train_val_test_split = config["train_val_test_split"]

    transform = get_default_transforms()
    dataset = NutritionDataset(
        data_path="src/wizard_ops/data.nosync",
        batch_size=batch_size,
        frame_idx=frame_idx, 
        num_workers=num_workers,
        train_val_test_split=train_val_test_split,
        camera=camera,
        transform=transform,
        seed=seed,
    )

    dataset.setup(stage="fit")

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
            monitor="val_loss",
            dirpath=f"checkpoints/{run_name}",
            filename="best-nutrition-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        ),
        # Logs the learning rate so you can see the scheduler working
        LearningRateMonitor(logging_interval="step")
    ]

    model = NutritionPredictor(num_outputs=num_outputs, lr=lr, seed=seed)
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


if __name__ == "__main__":
    app()
