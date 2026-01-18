from typing import Annotated

import lightning as L
import numpy as np
import torch
import typer
from data import NutritionDataModule, denormalize, get_default_transforms
from model import DishMultiViewRegressor
from PIL import Image

app = typer.Typer()


@app.command("evaluate")
@torch.inference_mode()
def predict_from_pil_images(img_path: Annotated[str, typer.Option(help="Path to the image file")],
                            device: Annotated[str, typer.Option(help="Device to run on (cpu/mps/cuda)")] =
                            "mps" if torch.backends.mps.is_available() else "cpu",
                            normalized_output: Annotated[bool, typer.Option(help="Return normalized model output.")] = False,
                            uncertainty: Annotated[bool, typer.Option(help="Use monte carlo drop-out for predictions")] = False):
    """Predict from a single image path. The augmentation/transform is constructed
    inside the function to avoid Typer trying to serialize it to the CLI.
    """

    # build transform at runtime (Typer doesn't try to serialize this)
    transform = get_default_transforms(224)

    model = DishMultiViewRegressor.load_from_checkpoint("checkpoints/best-nutrition-0.ckpt")
    print(model.hparams)

    model.eval().to(device)

    tensors = []
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)

    out = transform(image=arr)
    # albumentations typically returns a dict {'image': ...}
    if isinstance(out, dict):
        image_obj = out.get("image")
    else:
        image_obj = out

    # normalize to a torch tensor with shape (C, H, W)
    if isinstance(image_obj, np.ndarray):
        image_tensor = torch.from_numpy(image_obj).permute(2, 0, 1).float() / 255.0
    elif isinstance(image_obj, torch.Tensor):
        image_tensor = image_obj
    else:
        raise TypeError(f"Unsupported transform output type: {type(image_obj)}")

    tensors.append(image_tensor)

    images = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
    # images: (1, V, C, H, W)


    if uncertainty:
        preds, std = model.predict_with_uncertainty(images,n_samples=20)
        preds = preds.squeeze(0).cpu()
        std = std.squeeze(0).cpu()

        std = {
            "total_calories": float(std[0]),
            "total_mass": float(std[1]),
            "total_fat": float(std[2]),
            "total_carb": float(std[3]),
            "total_protein": float(std[4]),
        }

    else:    
        preds, attn = model(images)
        preds = preds.squeeze(0).cpu()

    result = {
        "total_calories": float(preds[0]),
        "total_mass": float(preds[1]),
        "total_fat": float(preds[2]),
        "total_carb": float(preds[3]),
        "total_protein": float(preds[4]),
        "num_images_used": len(tensors),
    }

    if not normalized_output:
        # Load data to get norm_stats ###IS THIS BEST WAY TO DO IT?
        data_module = NutritionDataModule(
            data_path="data.nosync",
            dish_csv="src/wizard_ops/metadata/data_stats.csv",
            batch_size=1,
            image_size=224,
            normalise_dish_metadata=True,
            val_split=0.0,
            num_workers=0,
            use_only_dishes_with_all_cameras=True,
            seed=42,
        )
    
        # Setup and get validation dataset
        data_module.setup()
        dataset = data_module.train_dataset
        norm_stats = dataset.normalization_stats

        result = denormalize(result, norm_stats)

        if uncertainty:
            std = denormalize(std, norm_stats)
        

    if uncertainty:
        typer.echo("Result")
        typer.echo(result)
        typer.echo("Standard deviation")
        typer.echo(std)
        return result, std

    typer.echo("Result")
    typer.echo(result)
    return result

@app.command("evaluate_validation_sample")
@torch.inference_mode()
def evaluate_validation_sample(
    data_path: Annotated[str, typer.Option(help="Path to the data directory")] = "data.nosync",
    dish_csv: Annotated[str, typer.Option(help="Path to dish metadata CSV")] = "src/wizard_ops/metadata/data_stats.csv",
    checkpoint_path: Annotated[str, typer.Option(help="Path to model checkpoint")] = "checkpoints/best-nutrition-0.ckpt",
    device: Annotated[str, typer.Option(help="Device to run on (cpu/mps/cuda)")] = "mps" if torch.backends.mps.is_available() else "cpu",
    sample_idx: Annotated[int, typer.Option(help="Index of validation sample to test")] = 0,
    print_stats: Annotated[bool, typer.Option(help="Print normalization statistics")] = False,
    print_normalized: Annotated[bool, typer.Option(help="Print normalized values (0-1 scale)")] = False,
    uncertainty : Annotated[bool, typer.Option(help="Add uncertainty to predictions based on monte-carlo dropout.")] = False,
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
    norm_stats = val_dataset.normalization_stats

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
    
    # Print Dish ID
    typer.echo("=" * 80)
    typer.echo(f"DISH ID: {dish_id}")
    typer.echo("=" * 80)
    typer.echo()
    
    if uncertainty:
        mean_normalized, std_normalized = model.predict_with_uncertainty(images)
        mean_normalized = mean_normalized.cpu()
        std_normalized = std_normalized.cpu()

        # Convert to dict for denormalization
        mean_dict = {
            "total_calories": float(mean_normalized[0, 0]),
            "total_mass": float(mean_normalized[0, 1]),
            "total_fat": float(mean_normalized[0, 2]),
            "total_carb": float(mean_normalized[0, 3]),
            "total_protein": float(mean_normalized[0, 4]),
        }

        std_dict = {
            "total_calories": float(std_normalized[0, 0]),
            "total_mass": float(std_normalized[0, 1]),
            "total_fat": float(std_normalized[0, 2]),
            "total_carb": float(std_normalized[0, 3]),
            "total_protein": float(std_normalized[0, 4]),
        }

        # Denormalize mean: multiply by max_val
        mean_denormalized = denormalize(mean_dict, norm_stats)

        # Denormalize std: also multiply by max_val (std scales linearly)
        std_denormalized = denormalize(std_dict, norm_stats)
        
        # Denormalize ground truth
        ground_truth_denormalized = denormalize(ground_truth, norm_stats)

        # Print prediction and ground truth
        if print_normalized:
            typer.echo("NORMALIZED VALUES (0-1 scale):")
            typer.echo("-" * 80)
            typer.echo(f"{'Metric':<20} {'Predicted':<15} {'Actual':<15} {'Difference':<15}")
            typer.echo("-" * 80)
            for key in mean_dict.keys():
                pred = mean_dict[key]
                std = std_dict[key]
                actual = ground_truth[key]
                diff = pred - actual
                typer.echo(f"{key:<20} {pred:>7.4f} ± {std:<6.4f} {actual:<15.4f} {diff:<15.4f}")
            typer.echo()
        
        typer.echo("DENORMALIZED VALUES (original units):")
        typer.echo("-" * 80)
        typer.echo(f"{'Metric':<20} {'Predicted':<15} {'Actual':<15} {'Difference':<15}")
        typer.echo("-" * 80)
        for key in mean_denormalized.keys():
            pred = mean_denormalized[key]
            std = std_denormalized[key]
            actual = ground_truth_denormalized[key]
            diff = pred - actual
            typer.echo(f"{key:<20} {pred:>7.2f} ± {std:<6.2f} {actual:<15.2f} {diff:<15.2f}")
        typer.echo()


    else:
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
        predictions_denormalized = denormalize(predictions_normalized, norm_stats)
        ground_truth_denormalized = denormalize(ground_truth, norm_stats)

        # Print prediction and ground truth
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