import json
import logging
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Path to normalisation stats JSON file (relative to project root)
# Goes up from src/wizard_ops/data.py -> src/wizard_ops -> src -> project_root -> configs/metadata
NORMALIZATION_STATS_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "metadata" / "normalisation_stats.json"
# NORMALIZATION_STATS_PATH = Path("/Users/arata/Desktop/DTU HCAI/Kurser/Winter_2026/wizard_ops/configs/metadata/normalisation_stats.json")


# MARK: - Loads normalisation stats
def load_normalization_stats(path: Path | str | None = None) -> dict:
    """Load normalization statistics from JSON file.
    
    Args:
        path: Path to the normalization stats JSON file. Uses default if None.
        
    Returns:
        Dictionary containing normalization statistics for targets and images.
    """
    if path is None:
        path = NORMALIZATION_STATS_PATH
    path = Path(path)
    
    if not path.exists():
        logger.warning(f"Normalization stats file not found at {path}")
        return {}
    
    with open(path, "r") as f:
        return json.load(f)


# MARK: - Denormalise predictions
def denormalize_predictions(
    predictions: dict[str, float | torch.Tensor],
    stats: dict | None = None,
    method: str = "zscore",
) -> dict[str, float]:
    """Convert normalized model predictions back to real units.
    
    Args:
        predictions: Dictionary with normalized prediction values.
        stats: Normalization stats dict (loads from file if None).
        method: Normalization method used - 'zscore' or 'minmax'.
        
    Returns:
        Dictionary with denormalized values in real units.
    """
    if stats is None:
        stats = load_normalization_stats()
    
    target_stats = stats.get("targets", {})
    result = {}
    
    for key, value in predictions.items():
        if key not in target_stats:
            result[key] = float(value) if isinstance(value, torch.Tensor) else value
            continue
        
        stat = target_stats[key]
        val = float(value) if isinstance(value, torch.Tensor) else value
        
        if method == "zscore":
            # Denormalize: raw = normalized * std + mean
            result[key] = val * stat["std"] + stat["mean"]
        elif method == "minmax":
            # Denormalize: raw = normalized * (max - min) + min
            result[key] = val * (stat["max"] - stat["min"]) + stat["min"]
        else:
            result[key] = val
    
    return result


# MARK: - Default image transforms
def get_default_transforms(image_size: int = 224) -> A.Compose:
    """Default image transforms for the dataset.
    
    Args:
        image_size: Target image size (square).
        
    Returns:
        Albumentations Compose transform.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        A.ToTensorV2(),
    ])
  
  
# MARK: - Augmentation transforms for training
def get_augmentation_transforms(image_size: int = 224) -> A.Compose:
    """Training transforms with data augmentation.
    Can be more complex and include flips, rotations, color jitter, etc.
    
    Args:
        image_size: Target image size (square).
        
    Returns:
        Albumentations Compose transform with augmentation.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.GridDropout(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        A.ToTensorV2(),
    ])
    

# - MARK: Preprocessing dataset
def process_single_dish(
    dish_id: str,
    data_dir: Path,
    image_size: int,
) -> tuple[str, np.ndarray] | None:
    """Process all images for a single dish.
    
    Args:
        dish_id: The dish identifier
        data_dir: Root data directory
        image_size: Target image size
        
    Returns:
        Tuple of (dish_id, images_array) or None if failed
    """
    dish_dir = data_dir / dish_id
    
    # Pre-allocate array for this dish: (4 cameras, 5 frames, 3 channels, H, W)
    dish_images = np.zeros((4, 5, 3, image_size, image_size), dtype=np.uint8)
    
    try:
        for cam_idx, cam in enumerate(["A", "B", "C", "D"]):
            for frame_idx in range(5):
                frame_num = frame_idx + 1
                p = dish_dir / f"camera_{cam}_frame_{frame_num:03d}.jpeg"
                if not p.exists():
                    p = dish_dir / f"camera_{cam}_frame_{frame_num:03d}.jpg"
                
                # Load and resize image
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    img = img.resize((image_size, image_size), Image.BILINEAR)
                    arr = np.array(img)
                
                # Store as (C, H, W) - channels first for PyTorch
                dish_images[cam_idx, frame_idx] = arr.transpose(2, 0, 1)
        
        return (dish_id, dish_images)
    except Exception as e:
        logger.warning(f"Failed to process dish {dish_id}: {e}")
        return None