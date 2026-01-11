import torch
from torch.utils.data import DataLoader
from lightning import Trainer
import typer
from loguru import logger

app = typer.Typer(help="Commands to train nutrition predictor.")

ACCELERATOR = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from wizard_ops.data import NutritionDataset, get_default_transforms
from wizard_ops.model import NutritionPredictor


@app.command()
def train(frame_idx: int = 1, num_workers: int = 4, max_epochs: int = 10, fast_dev_run: bool = False) -> None:
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
    transform = get_default_transforms()
    dataset = NutritionDataset(
        data_path="src/wizard_ops/data.nosync", frame_idx=frame_idx, num_workers=num_workers, transform=transform
    )

    dataset.setup(stage="fit")

    train_len = len(dataset.train_dataloader().dataset)
    val_len = len(dataset.val_dataloader().dataset) if hasattr(dataset, "val_dataloader") else None
    batch = next(iter(dataset.train_dataloader()))

    logger.info(f"Train samples: {train_len}, val samples: {val_len}, total samples: {train_len + val_len}")
    logger.debug(f"Image batch shape: {batch['image'].shape}")
    nutrition = torch.stack([batch[column] for column in ["calories", "mass", "fat", "carbs", "protein"]], dim=1)
    logger.debug(f"Nutrition batch shape: {nutrition.shape}")

    model = NutritionPredictor()
    trainer = Trainer(accelerator=ACCELERATOR, max_epochs=max_epochs, fast_dev_run=fast_dev_run)

    trainer.fit(model, datamodule=dataset)


if __name__ == "__main__":
    app()
