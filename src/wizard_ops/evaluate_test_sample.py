import torch
import numpy as np
from PIL import Image
from model import DishMultiViewRegressor
import lightning as L
import typer
from typing import Annotated

from data import get_default_transforms, NutritionDataModule, denormalize

app = typer.Typer()

@app.command()
@torch.inference_mode()
def evaluate_test_sample(
    data_path: Annotated[str, typer.Option(help="Path to the data directory")] = "data.nosync",
    dish_csv: Annotated[str, typer.Option(help="Path to dish metadata CSV")] = "src/wizard_ops/metadata/data_stats.csv",
    checkpoint_path: Annotated[str, typer.Option(help="Path to model checkpoint")] = "checkpoints/best-nutrition-0.ckpt",
    device: Annotated[str, typer.Option(help="Device to run on (cpu/mps/cuda)")] = "mps" if torch.backends.mps.is_available() else "cpu",
    sample_idx: Annotated[int, typer.Option(help="Index of validation sample to test")] = 0,
    print_stats: Annotated[bool, typer.Option(help="Print normalization statistics")] = False,
    print_normalized: Annotated[bool, typer.Option(help="Print normalized values (0-1 scale)")] = False,
):
    """Load a test sample from the validation set, make a prediction, and compare with ground truth."""
    
    # Load the model
    model = DishMultiViewRegressor.load_from_checkpoint(checkpoint_path)
    model.eval().to(device)
    
    typer.echo(f"Loaded model from {checkpoint_path}")
    typer.echo(f"Model hyperparameters: {model.hparams}")
    typer.echo()
    
    # Create data module
    val_transform = get_default_transforms(224)
    data_module = NutritionDataModule(
        data_path=data_path,
        dish_csv=dish_csv,
        batch_size=1,
        image_size=224,
        val_transform=val_transform,
        normalise_dish_metadata=True,
        val_split=0.2,
        num_workers=0,
        use_only_dishes_with_all_cameras=True,
        seed=42,
    )
    
    # Setup and get validation dataset
    data_module.setup()
    val_dataset = data_module.val_dataset
    
    typer.echo(f"Validation dataset size: {len(val_dataset)}")
    typer.echo(f"Loading sample at index {sample_idx}...")
    typer.echo()
    
    # Get a sample from the validation set
    sample = val_dataset[sample_idx]
    
    # Extract data
    dish_id = sample["dish_id"]
    images = sample["images"].unsqueeze(0).to(device)  # Add batch dimension: (1, 4, 5, C, H, W)
    
    # Get ground truth (already normalized by the dataset)
    ground_truth = {
        "total_calories": float(sample["total_calories"]),
        "total_mass": float(sample["total_mass"]),
        "total_fat": float(sample["total_fat"]),
        "total_carb": float(sample["total_carb"]),
        "total_protein": float(sample["total_protein"]),
    }
    
    # Make prediction
    preds, attn = model(images)
    preds = preds.squeeze(0).cpu()
    
    # Extract predictions
    predictions_normalized = {
        "total_calories": float(preds[0]),
        "total_mass": float(preds[1]),
        "total_fat": float(preds[2]),
        "total_carb": float(preds[3]),
        "total_protein": float(preds[4]),
    }
    
    # Denormalize predictions and ground truth
    norm_stats = val_dataset.normalization_stats
    predictions_denormalized = denormalize(predictions_normalized, norm_stats)
    ground_truth_denormalized = denormalize(ground_truth, norm_stats)
    
    # Print results
    typer.echo("=" * 80)
    typer.echo(f"DISH ID: {dish_id}")
    typer.echo("=" * 80)
    typer.echo()
    
    if print_normalized:
        typer.echo("NORMALIZED VALUES (0-1 scale):")
        typer.echo("-" * 80)
        typer.echo(f"{'Metric':<20} {'Predicted':<15} {'Actual':<15} {'Difference':<15}")
        typer.echo("-" * 80)
        for key in predictions_normalized.keys():
            pred = predictions_normalized[key]
            actual = ground_truth[key]
            diff = pred - actual
            typer.echo(f"{key:<20} {pred:<15.4f} {actual:<15.4f} {diff:<15.4f}")
        typer.echo()
    
    typer.echo("DENORMALIZED VALUES (original units):")
    typer.echo("-" * 80)
    typer.echo(f"{'Metric':<20} {'Predicted':<15} {'Actual':<15} {'Difference':<15}")
    typer.echo("-" * 80)
    for key in predictions_denormalized.keys():
        pred = predictions_denormalized[key]
        actual = ground_truth_denormalized[key]
        diff = pred - actual
        typer.echo(f"{key:<20} {pred:<15.2f} {actual:<15.2f} {diff:<15.2f}")
    typer.echo()
    
    if print_stats:
        typer.echo("NORMALIZATION STATS (max values):")
        typer.echo("-" * 80)
        if norm_stats:
            for key, max_val in norm_stats.items():
                typer.echo(f"{key:<20} {max_val:<15.2f}")
        else:
            typer.echo("No normalization stats available")
        typer.echo()
        
        return {
            "dish_id": dish_id,
            "predictions": predictions_denormalized,
            "ground_truth": ground_truth_denormalized,
            "predictions_normalized": predictions_normalized,
            "ground_truth_normalized": ground_truth,
        }


if __name__ == "__main__":
    app()