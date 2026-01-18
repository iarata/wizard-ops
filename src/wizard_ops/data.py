import logging
from pathlib import Path
from typing import Literal

import albumentations as A
import h5py
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from wizard_ops.utils import (
    get_augmentation_transforms,
    get_default_transforms,
    load_normalization_stats,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

NUM_CAMERAS = 4
NUM_FRAMES = 5

TARGET_COLUMNS = [
    "total_calories",
    "total_mass",
    "total_fat",
    "total_carb",
    "total_protein",
]

NormalisationMethod = Literal["zscore", "minmax", "max", "none"]


class TargetNormaliser:
    def __init__(
        self,
        stats: dict,
        method: NormalisationMethod = "zscore",
    ):
        self.method = method
        self.target_stats = (stats or {}).get("targets", {})

    def normalize(self, column: str, value: float) -> float:
        if self.method == "none":
            return float(value)

        stat = self.target_stats.get(column)
        if not stat:
            return float(value)

        v = float(value)

        if self.method == "zscore":
            std = float(stat.get("std", 0.0))
            mean = float(stat.get("mean", 0.0))
            return (v - mean) / std if std > 0 else 0.0

        if self.method == "minmax":
            mn = float(stat.get("min", 0.0))
            mx = float(stat.get("max", 0.0))
            rng = mx - mn
            return (v - mn) / rng if rng > 0 else 0.0

        if self.method == "max":
            mx = float(stat.get("max", 0.0))
            return v / mx if mx > 0 else 0.0

        return v


class Nutrition(Dataset):
    """
    Dataset that loads pre-processed images from an HDF5 file.

    Images are expected to be stored as uint8 in shape:
        (NUM_CAMERAS, NUM_FRAMES, 3, H, W)

    If `transform` is provided, it is applied per view (camera/frame). Otherwise,
    images are converted to float and ImageNet-normalized.
    """

    def __init__(
        self,
        h5_path: str | Path,
        dish_csv: str | Path,
        transform: A.Compose | None = None,
        normalization_method: NormalisationMethod = "zscore",
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

        stats = normalization_stats if normalization_stats is not None else load_normalization_stats()
        self.normaliser = TargetNormaliser(stats=stats, method=normalization_method)
        self.norm_stats = stats  # keep for backwards-compat / external access

        self.metadata = pd.read_csv(self.dish_csv, dtype={"dish_id": str})

        self._h5_file: h5py.File | None = None
        self._idx_to_h5_idx: list[int] = []
        self._build_index()

    def _build_index(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            dish_ids_ds = f["dish_ids"][:]
            h5_dish_ids = [
                d.decode() if isinstance(d, (bytes, np.bytes_)) else str(d)
                for d in dish_ids_ds
            ]

        h5_id_to_idx = {dish_id: i for i, dish_id in enumerate(h5_dish_ids)}
        valid_ids = set(h5_id_to_idx)

        keep = self.metadata["dish_id"].astype(str).isin(valid_ids)
        self.metadata = self.metadata.loc[keep].reset_index(drop=True)

        self._idx_to_h5_idx = [h5_id_to_idx[str(d)] for d in self.metadata["dish_id"]]

        logger.info("Nutrition: Loaded %d dishes from %s", len(self.metadata), self.h5_path)

    def _get_h5(self) -> h5py.File:
        # Lazily open per-process/per-worker
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __getstate__(self):
        # Ensure HDF5 handle is not pickled into DataLoader workers
        state = dict(self.__dict__)
        state["_h5_file"] = None
        return state

    @staticmethod
    def _imagenet_normalize(images_uint8: np.ndarray) -> torch.Tensor:
        # images_uint8: (CAMS, FRAMES, 3, H, W) uint8
        x = torch.from_numpy(images_uint8).to(torch.float32).div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(1, 1, 3, 1, 1)
        return (x - mean) / std

    @staticmethod
    def _apply_transform(
        images_uint8: np.ndarray,
        transform: A.Compose,
    ) -> torch.Tensor:
        # images_uint8: (CAMS, FRAMES, 3, H, W)
        cams, frames, c, h, w = images_uint8.shape
        assert cams == NUM_CAMERAS and frames == NUM_FRAMES and c == 3

        flat = images_uint8.reshape(cams * frames, 3, h, w)

        out_views: list[torch.Tensor] = []
        for view_chw in flat:
            view_hwc = np.transpose(view_chw, (1, 2, 0))  # (H, W, C) for Albumentations
            transformed = transform(image=view_hwc)["image"]
            if not torch.is_tensor(transformed):
                # In case the user forgot ToTensorV2
                transformed = torch.from_numpy(transformed)

            out_views.append(transformed)

        out = torch.stack(out_views, dim=0)
        return out.view(cams, frames, *out.shape[1:])

    def _targets_from_row(self, row: pd.Series) -> dict:
        targets = {
            name: torch.tensor(self.normaliser.normalize(name, row[name]), dtype=torch.float32)
            for name in TARGET_COLUMNS
        }
        targets["num_ingrs"] = torch.tensor(int(row["num_ingrs"]), dtype=torch.long)
        return targets

    def __len__(self) -> int:
        return len(self.metadata)

    def get_example(self, idx: int, transform: A.Compose | None = None) -> dict:
        row = self.metadata.iloc[idx]
        dish_id = str(row["dish_id"])
        h5_idx = self._idx_to_h5_idx[idx]

        images = self._get_h5()["images"][h5_idx]  # uint8, (4, 5, 3, H, W)

        if transform is not None:
            images_tensor = self._apply_transform(images, transform)
        elif self.transform is not None:
            images_tensor = self._apply_transform(images, self.transform)
        else:
            images_tensor = self._imagenet_normalize(images)

        out = {
            "dish_id": dish_id,
            "images": images_tensor,
            **self._targets_from_row(row),
        }

        # Optional convenience vector (keeps old keys too)
        out["targets"] = torch.stack([out[c] for c in TARGET_COLUMNS], dim=0)
        return out

    def __getitem__(self, idx: int) -> dict:
        return self.get_example(idx, transform=None)

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            finally:
                self._h5_file = None

    def __del__(self):
        self.close()


class NutritionSubset(Dataset):
    """Subset wrapper that can apply its own transform without mutating the base dataset."""

    def __init__(self, dataset: Nutrition, indices: list[int], transform: A.Compose | None = None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]
        return self.dataset.get_example(real_idx, transform=self.transform)


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
        normalization_method: NormalisationMethod = "zscore",
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
        self.dish_csv = Path(dish_csv)
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.normalization_method = normalization_method
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.prefetch_factor = prefetch_factor

        self._full_dataset: Nutrition | None = None
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

        assert self.h5_path.exists(), f"HDF5 file not found: {self.h5_path}"

    def setup(self, stage: str | None = None) -> None:
        self._full_dataset = Nutrition(
            h5_path=self.h5_path,
            dish_csv=self.dish_csv,
            transform=None,  # transforms are applied by subset wrappers
            normalization_method=self.normalization_method,
        )

        n = len(self._full_dataset)
        n_val = int(n * self.val_split)
        n_train = n - n_val

        g = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(n, generator=g).tolist()
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_tf = self.train_transform or get_augmentation_transforms(224)
        val_tf = self.val_transform or get_default_transforms(224)

        self.train_dataset = NutritionSubset(self._full_dataset, train_idx, transform=train_tf)
        self.val_dataset = NutritionSubset(self._full_dataset, val_idx, transform=val_tf)

        logger.info("NutritionDataModule: %d train, %d val samples", n_train, n_val)

    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None
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
        assert self.val_dataset is not None
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
        if self._full_dataset is not None:
            return self._full_dataset.norm_stats
        return load_normalization_stats()


if __name__ == "__main__":
    train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    val_transform = get_default_transforms(224)

    dm = NutritionDataModule(
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
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"First train batch images: {train_batch['images'].shape}")
    print(f"First val batch images: {val_batch['images'].shape}")
    
    # printing keys available in the batch
    print(f"Keys in train batch: {train_batch.keys()}")
    print(f"Keys in val batch: {val_batch.keys()}")