import torch
from torch.utils.data import DataLoader
from lightning import Trainer
import typer
from loguru import logger
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(log_model='all', name='nutrition_logger')
app = typer.Typer(help="Commands to train nutrition predictor.")

ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from wizard_ops.data import NutritionDataset, get_default_transforms
from wizard_ops.model import NutritionPredictor

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
def train(
        frame_idx: int = 1, 
        num_workers: int = 4, 
        fast_dev_run: bool = False) -> None:
    """
    Train a NutritionPredictor model using the provided dataset and configuration.

    Args:
        frame_idx (int, optional): Frame index for dataset processing. Defaults to 1.
        num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        lr (float, optional): Learning rate. Defaults to 1e-2.
        max_epochs (int, optional): Maximum number of training epochs. Defaults to 10.
        train_val_test_split (tuple, optional): Train/val/test split ratios. Defaults to (0.7, 0.15, 0.15).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        camera (str, optional): Camera angle to use. Defaults to 'A'.
        num_outputs (int, optional): Number of output values. Defaults to 5.
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

    train_len = len(dataset.train_dataloader().dataset)
    val_len = len(dataset.val_dataloader().dataset) if hasattr(dataset, "val_dataloader") else None
    batch = next(iter(dataset.train_dataloader()))

    logger.info(f"Train samples: {train_len}, val samples: {val_len}, total samples: {train_len + val_len}")
    logger.debug(f"Image batch shape: {batch['image'].shape}")
    nutrition = torch.stack([batch[column] for column in ["calories", "mass", "fat", "carbs", "protein"]], dim=1)
    logger.debug(f"Nutrition batch shape: {nutrition.shape}")

    model = NutritionPredictor(num_outputs=num_outputs, lr=lr, seed=seed)
    trainer = Trainer(
        accelerator=ACCELERATOR,
        max_epochs=max_epochs,
        fast_dev_run=fast_dev_run,
        logger=wandb_logger,
        log_every_n_steps=100
    )

    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    app()
