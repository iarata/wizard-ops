import logging
import re
from pathlib import Path

import albumentations as A
import h5py
import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from wizard_ops.utils.helpers import get_augmentation_transforms, get_default_transforms, load_normalization_stats

_DISH_TOTAL_COLUMNS = [
    "dish_id",
    "total_calories",
    "total_mass",
    "total_fat",
    "total_carb",
    "total_protein",
    "num_ingrs",
    "num_images_camera_A",
    "num_images_camera_B",
    "num_images_camera_C",
    "num_images_camera_D",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Nutrition(Dataset):
    """Dataset that loads pre-processed images from HDF5 file.
    
    Usage:
        # I. Preprocess images using scripts/preprocess_images.py:
        # python scripts/preprocess_images.py --data-dir data.nosync --output data.nosync/images_224.h5
        
        # II. Create dataset
        dataset = Nutrition(
            h5_path="data.nosync/images_224.h5",
            dish_csv="configs/metadata/data_stats.csv",
            transform=get_augmentation_transforms(224),
        )
    """
    
    def __init__(
        self,
        h5_path: str | Path,
        dish_csv: str | Path,
        transform: A.Compose | None = None,
        normalization_method: str = "zscore",  # 'zscore', 'minmax', or 'max'
        normalization_stats: dict | None = None,
    ):
        """Initialize the HDF5 dataset.
        
        Args:
            h5_path: Path to the preprocessed HDF5 file.
            dish_csv: Path to the CSV file with dish metadata.
            transform: Albumentations transform to apply (for augmentation only,
                      resizing is already done in preprocessing).
            normalization_method: How to normalize targets ('zscore', 'minmax', 'max').
            normalization_stats: Pre-computed stats dict, or loads from default file.
        """
        self.h5_path = Path(h5_path)
        self.dish_csv = Path(dish_csv)
        self.transform = transform
        self.normalization_method = normalization_method
        
        # Load normalization stats
        if normalization_stats is None:
            self.norm_stats = load_normalization_stats()
        else:
            self.norm_stats = normalization_stats
        
        self.metadata = pd.read_csv(dish_csv, dtype={"dish_id": str})
        
        self._h5_file: h5py.File | None = None
        self._build_index()
        
    def _build_index(self) -> None:
        """Build mapping between metadata rows and HDF5 indices."""
        with h5py.File(self.h5_path, "r") as f:
            h5_dish_ids = [d.decode() if isinstance(d, bytes) else d for d in f["dish_ids"][:]]
        
        # Create mapping from dish_id to HDF5 index
        self._h5_id_to_idx = {dish_id: i for i, dish_id in enumerate(h5_dish_ids)}
        
        # Filter metadata to only dishes in HDF5 file
        valid_mask = self.metadata["dish_id"].astype(str).isin(self._h5_id_to_idx.keys())
        self.metadata = self.metadata[valid_mask].reset_index(drop=True)
        self._idx_to_h5_idx = [
            self._h5_id_to_idx[str(dish_id)] 
            for dish_id in self.metadata["dish_id"]
        ]
        
        logger.info(f"Nutrition: Loaded {len(self.metadata)} dishes from {self.h5_path}")
        
    def _get_h5_file(self) -> h5py.File:
        """Get or open the HDF5 file handle (for multiprocessing safety)."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file
    
    def _normalize_value(self, column: str, value: float) -> float:
        """Normalize a value using the configured method."""
        target_stats = self.norm_stats.get("targets", {})
        if column not in target_stats:
            return value
        
        stat = target_stats[column]
        
        if self.normalization_method == "zscore":
            return (value - stat["mean"]) / stat["std"]
        elif self.normalization_method == "minmax":
            range_val = stat["max"] - stat["min"]
            return (value - stat["min"]) / range_val if range_val > 0 else 0.0
        elif self.normalization_method == "max":
            return value / stat["max"] if stat["max"] > 0 else 0.0
        return value
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        dish_id = str(row["dish_id"])
        h5_idx = self._idx_to_h5_idx[idx]
        
        # Load images from HDF5 - shape: (4, 5, 3, H, W)
        h5 = self._get_h5_file()
        images = h5["images"][h5_idx]  # uint8 array
        
        if self.transform is not None:
            view_tensors = []
            for cam_idx in range(4):
                for frame_idx in range(5):
                    # Convert from (C, H, W) to (H, W, C) for albumentations
                    img = images[cam_idx, frame_idx].transpose(1, 2, 0)
                    transformed = self.transform(image=img)
                    view_tensors.append(transformed["image"])
            
            images_tensor = torch.stack(view_tensors, dim=0)
            images_tensor = images_tensor.view(4, 5, *images_tensor.shape[1:])
        else:
            # Just convert to tensor with ImageNet normalization
            images_tensor = torch.from_numpy(images.astype(np.float32)) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
            images_tensor = (images_tensor - mean) / std
        
        return {
            "dish_id": dish_id,
            "images": images_tensor,
            "total_calories": torch.tensor(
                self._normalize_value("total_calories", row["total_calories"]),
                dtype=torch.float
            ),
            "total_mass": torch.tensor(
                self._normalize_value("total_mass", row["total_mass"]),
                dtype=torch.float
            ),
            "total_fat": torch.tensor(
                self._normalize_value("total_fat", row["total_fat"]),
                dtype=torch.float
            ),
            "total_carb": torch.tensor(
                self._normalize_value("total_carb", row["total_carb"]),
                dtype=torch.float
            ),
            "total_protein": torch.tensor(
                self._normalize_value("total_protein", row["total_protein"]),
                dtype=torch.float
            ),
            "num_ingrs": torch.tensor(row["num_ingrs"], dtype=torch.long),
        }
    
    def __del__(self):
        """Close HDF5 file on deletion."""
        if self._h5_file is not None:
            self._h5_file.close()

class NutritionSubset(Dataset):
    """A subset of Nutrition dataset with optional different transform."""
    
    def __init__(
        self,
        dataset: "Nutrition",
        indices: list[int],
        transform: A.Compose | None = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]
        
        # Temporarily swap transform if different
        if self.transform is not None:
            original_transform = self.dataset.transform
            self.dataset.transform = self.transform
            result = self.dataset[real_idx]
            self.dataset.transform = original_transform
        else:
            result = self.dataset[real_idx]
        
        return result
    
class NutritionDataModule(L.LightningDataModule):
    """Lightning DataModule using pre-processed HDF5 images.
    
    This DataModule is optimized for:
    1. Using pre-resized images stored in HDF5 format
    2. Using z-score normalization with pre-computed stats
    3. Optimized data loading with prefetching
    
    Before using this module, preprocess images with:
        python scripts/preprocess_images.py --data-dir data.nosync --output data.nosync/images_224.h5
    
    Example:
        data_module = NutritionDataModule(
            h5_path="data.nosync/images_224.h5",
            dish_csv="configs/metadata/data_stats.csv",
            batch_size=32,
        )
    """
    
    def __init__(
        self,
        h5_path: str | Path,
        dish_csv: str | Path,
        batch_size: int = 32,
        train_transform: A.Compose | None = None,
        val_transform: A.Compose | None = None,
        normalization_method: str = "zscore",
        val_split: float = 0.2,
        num_workers: int = 4,
        seed: int = 42,
        prefetch_factor: int = 2,
    ):
        """Initialize the fast DataModule.
        
        Args:
            h5_path: Path to preprocessed HDF5 file.
            dish_csv: Path to data_stats.csv.
            batch_size: Training batch size.
            train_transform: Training augmentation transform.
            val_transform: Validation transform (usually no augmentation).
            normalization_method: How to normalize targets ('zscore', 'minmax', 'max').
            val_split: Fraction of data to use for validation.
            num_workers: Number of data loading workers.
            seed: Random seed for reproducibility.
            prefetch_factor: Number of batches to prefetch per worker.
        """
        super().__init__()
        self.h5_path = Path(h5_path)
        
        assert self.h5_path.exists(), f"HDF5 file not found: {self.h5_path}"
        
        self.dish_csv = Path(dish_csv)
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.normalization_method = normalization_method
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        
        self.train_dataset: NutritionHDF5Subset | None = None
        self.val_dataset: NutritionHDF5Subset | None = None
        self._full_dataset: NutritionHDF5 | None = None
        
    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets."""
        self._full_dataset = Nutrition(
            h5_path=self.h5_path,
            dish_csv=self.dish_csv,
            transform=None,
            normalization_method=self.normalization_method,
        )
        
        dataset_size = len(self._full_dataset)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size
        
        # Generate reproducible split
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_transform = self.train_transform or get_augmentation_transforms(224)
        val_transform = self.val_transform or get_default_transforms(224)
        
        self.train_dataset = NutritionSubset(
            dataset=self._full_dataset,
            indices=train_indices,
            transform=train_transform,
        )
        
        self.val_dataset = NutritionSubset(
            dataset=self._full_dataset,
            indices=val_indices,
            transform=val_transform,
        )
        
        logger.info(f"NutritionDataModule: {train_size} train, {val_size} val samples")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
    
    @property
    def normalization_stats(self) -> dict:
        """Get the normalization statistics used by this dataset."""
        if self._full_dataset is not None:
            return self._full_dataset.norm_stats
        return load_normalization_stats()


if __name__ == "__main__":
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        A.ToTensorV2(),
    ])
    
    val_transform = get_default_transforms(224)  # No augmentation for validation
    
    data_module = NutritionDataModule(
        h5_path="data.nosync/images_224.h5",
        dish_csv="configs/metadata/data_stats.csv",
        batch_size=32,
        train_transform=train_transform,
        val_transform=val_transform,
        normalization_method="zscore",
        val_split=0.2,
        num_workers=9,
        seed=42,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Size of first training batch: {next(iter(train_loader))['images'].shape}")
    print(f"Size of first validation batch: {next(iter(val_loader))['images'].shape}")