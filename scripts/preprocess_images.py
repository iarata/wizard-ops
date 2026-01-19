"""
Preprocess images into HDF5 format for fast training.

This script creates a single HDF5 file containing all pre-resized images,
which dramatically speeds up data loading during training by:
1. Eliminating repeated JPEG decoding
2. Storing images in a contiguous memory-mapped format
3. Pre-resizing images to training resolution

Usage:
    python preprocess_images.py --data-dir data.nosync --output data.nosync/images_224.h5 --size 224
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def preprocess_dataset(
    data_dir: Path,
    csv_path: Path,
    output_path: Path,
    image_size: int = 224,
    use_only_complete: bool = True,
    num_workers: int = None,
) -> None:
    """
    Preprocess all dish images into an HDF5 file using parallel processing.
    
    Args:
        data_dir: Root directory containing dish_* folders
        csv_path: Path to data_stats.csv
        output_path: Output HDF5 file path
        image_size: Target image size (square)
        use_only_complete: Only include dishes with all 20 images
        num_workers: Number of parallel workers (default: CPU count)
    """
    if num_workers is None:
        num_workers = os.cpu_count() or 4
    
    # Load metadata
    df = pd.read_csv(csv_path, dtype={"dish_id": str})
    logger.info(f"Loaded {len(df)} dishes from metadata")
    
    # Filter to complete dishes if requested
    if use_only_complete:
        mask = (
            (df["num_images_camera_A"] == 5) &
            (df["num_images_camera_B"] == 5) &
            (df["num_images_camera_C"] == 5) &
            (df["num_images_camera_D"] == 5)
        )
        df = df[mask].reset_index(drop=True)
        logger.info(f"Filtered to {len(df)} dishes with complete camera sets")
    
    # Collect valid dishes (quick filesystem check)
    valid_dishes = []
    for dish_id in tqdm(df["dish_id"], desc="Validating dishes"):
        dish_dir = data_dir / dish_id
        if not dish_dir.exists():
            continue
        
        # Check all 20 images exist
        all_exist = True
        for cam in ["A", "B", "C", "D"]:
            for frame in range(1, 6):
                p = dish_dir / f"camera_{cam}_frame_{frame:03d}.jpeg"
                if not p.exists():
                    p = dish_dir / f"camera_{cam}_frame_{frame:03d}.jpg"
                if not p.exists():
                    all_exist = False
                    break
            if not all_exist:
                break
        
        if all_exist:
            valid_dishes.append(dish_id)
    
    logger.info(f"Found {len(valid_dishes)} valid dishes with all images")
    logger.info(f"Processing with {num_workers} workers...")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process images in parallel and collect results
    process_fn = partial(process_single_dish, data_dir=data_dir, image_size=image_size)
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_fn, dish_id): dish_id for dish_id in valid_dishes}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Sort results by dish_id to ensure deterministic order
    results.sort(key=lambda x: x[0])
    
    logger.info(f"Successfully processed {len(results)} dishes")
    
    # Write to HDF5 file
    logger.info("Writing to HDF5 file...")
    with h5py.File(output_path, "w") as f:
        # Store metadata
        f.attrs["image_size"] = image_size
        f.attrs["num_dishes"] = len(results)
        f.attrs["cameras"] = ["A", "B", "C", "D"]
        f.attrs["frames_per_camera"] = 5
        
        # Create datasets
        # Shape: (num_dishes, 4 cameras, 5 frames, 3 channels, H, W)
        images = f.create_dataset(
            "images",
            shape=(len(results), 4, 5, 3, image_size, image_size),
            dtype=np.uint8,
            chunks=(1, 4, 5, 3, image_size, image_size),
            compression="lzf",
        )
        
        # Store dish IDs as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        dish_ids = f.create_dataset("dish_ids", (len(results),), dtype=dt)
        
        # Write all results
        for i, (dish_id, dish_images) in enumerate(tqdm(results, desc="Writing to HDF5")):
            dish_ids[i] = dish_id
            images[i] = dish_images
    
    logger.info(f"Saved preprocessed images to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Preprocess images for fast training")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data.nosync"),
        help="Root directory containing dish folders",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("configs/metadata/data_stats.csv"),
        help="Path to data_stats.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data.nosync/images_224.h5"),
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Target image size (square)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include dishes with missing images",
    )
    
    args = parser.parse_args()
    
    preprocess_dataset(
        data_dir=args.data_dir,
        csv_path=args.csv,
        output_path=args.output,
        image_size=args.size,
        use_only_complete=not args.include_incomplete,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
