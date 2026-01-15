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

class NutritionHDF5(Dataset):
    """Fast dataset that loads pre-processed images from HDF5 file.
    
    This dataset is ~10-50x faster than loading individual JPEG files because:
    1. Images are pre-resized to target resolution
    2. No JPEG decoding overhead per batch
    3. HDF5 provides efficient memory-mapped access
    4. Data is contiguously stored for sequential access
    
    Usage:
        # First, preprocess images using scripts/preprocess_images.py:
        # python scripts/preprocess_images.py --data-dir data.nosync --output data.nosync/images_224.h5
        
        dataset = NutritionHDF5(
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
        
        # Load metadata
        self.metadata = pd.read_csv(dish_csv, dtype={"dish_id": str})
        
        # Open HDF5 file and build index
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
        
        # Create index mapping
        self._idx_to_h5_idx = [
            self._h5_id_to_idx[str(dish_id)] 
            for dish_id in self.metadata["dish_id"]
        ]
        
        logger.info(f"NutritionHDF5: Loaded {len(self.metadata)} dishes from {self.h5_path}")
        
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
        
        # Apply transforms if provided (for augmentation)
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
            # Normalize with ImageNet stats
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

class NutritionHDF5Subset(Dataset):
    """A subset of NutritionHDF5 dataset with optional different transform."""
    
    def __init__(
        self,
        dataset: "NutritionHDF5",
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
    
    
class Nutrition(Dataset):
    """Custom dataset for Nutrition data."""
    def __init__(
        self,
        data_path: str | Path,
        transform: A.Compose | None = None,
        dish_csv: str | Path | None = None,
        use_only_dishes_with_all_cameras: bool = False,
        images_per_camera: int = 5,
    ):
        
        self.data_path = Path(data_path)
        self.dish_csv = dish_csv
        self.transform = transform
        self.images_per_camera = images_per_camera
        self.dish_metadata = pd.read_csv(dish_csv, dtype={"dish_id": str})
        self._image_paths: list[list[Path]] = []
        
        # Normalization stats (to be set externally for per-split normalization)
        self._normalization_stats: dict[str, float] | None = None

        # Build index of image paths (and optionally filter to full-camera dishes).
        self.dish_metadata = self._build_index(
            self.dish_metadata,
            require_full_cameras=use_only_dishes_with_all_cameras,
        )
            
    def _expected_paths_for_dish(self, dish_id: str) -> list[Path] | None:
        """
        Return the 20 expected image paths for a dish (A-D x 5 frames) if all
        exist; otherwise return None.
        Expected naming: camera_{A..D}_frame_{001..005}.jpeg (or .jpg).
        """
        dish_dir = self.data_path / dish_id
        if not dish_dir.exists():
            return None

        paths: list[Path] = []
        for cam in ["A", "B", "C", "D"]:
            for frame_idx in range(1, self.images_per_camera + 1):
                stem = f"camera_{cam}_frame_{frame_idx:03d}"

                p_jpeg = dish_dir / f"{stem}.jpeg"
                p_jpg = dish_dir / f"{stem}.jpg"

                if p_jpeg.exists():
                    paths.append(p_jpeg)
                elif p_jpg.exists():
                    paths.append(p_jpg)
                else:
                    # Missing a required file
                    return None

        return paths

    def _build_index(
        self,
        df: pd.DataFrame,
        require_full_cameras: bool,
    ) -> pd.DataFrame:
        """
        Build self._image_paths aligned with dish_metadata rows.
        If require_full_cameras=True, drop dishes that don't have all 20 files.
        """
        kept_rows: list[int] = []
        image_paths: list[list[Path]] = []

        for i, dish_id in enumerate(df["dish_id"].astype(str).tolist()):
            paths = self._expected_paths_for_dish(dish_id)
            if paths is None:
                if require_full_cameras:
                    continue
                continue

            kept_rows.append(i)
            image_paths.append(paths)

        filtered = df.iloc[kept_rows].reset_index(drop=True)
        self._image_paths = image_paths

        logger.info(
            "Indexed dishes (require_full_cameras=%s): %d -> %d",
            require_full_cameras,
            len(df),
            len(filtered),
        )
        return filtered

    def _filter_only_full_camera_dishes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only dishes with exactly `images_per_camera` images for A,B,C,D."""
        required = self.images_per_camera

        count_cols = [
            "num_images_camera_A",
            "num_images_camera_B",
            "num_images_camera_C",
            "num_images_camera_D",
        ]

        # Prefer using the CSV counts if available (much faster than scanning dirs).
        if all(c in df.columns for c in count_cols):
            mask = (
                (df["num_images_camera_A"] == required)
                & (df["num_images_camera_B"] == required)
                & (df["num_images_camera_C"] == required)
                & (df["num_images_camera_D"] == required)
            )
            filtered = df.loc[mask].reset_index(drop=True)
            logger.info(
                "Filtered to full-camera dishes via CSV counts: %d -> %d",
                len(df),
                len(filtered),
            )
            return filtered

        # Fallback: scan filesystem
        valid_ids: list[str] = []
        for dish_id in df["dish_id"].astype(str).tolist():
            dish_dir = self.data_path / dish_id
            if not dish_dir.exists():
                continue
            ok = True
            for cam in ["A", "B", "C", "D"]:
                # camera_{A...D}_*.jpeg (and allow .jpg)
                cam_imgs = [
                    p
                    for p in dish_dir.iterdir()
                    if p.is_file()
                    and re.match(rf"camera_{cam}_.*\.(jpe?g)$", p.name, re.IGNORECASE)
                ]
                if len(cam_imgs) != required:
                    ok = False
                    break
            if ok:
                valid_ids.append(dish_id)

        filtered = df[df["dish_id"].isin(valid_ids)].reset_index(drop=True)
        logger.info(
            "Filtered to full-camera dishes via filesystem scan: %d -> %d",
            len(df),
            len(filtered),
        )
        return filtered

    def _normalise_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dish metadata DataFrame according to the values  of their corresponding columns."""
        df_normalised = df.copy()
        for column in _DISH_TOTAL_COLUMNS[1:]:
            max_value = df_normalised[column].max()
            if max_value > 0:
                df_normalised[column] = df_normalised[column] / max_value
        return df_normalised
    
    def set_normalization_stats(self, stats: dict[str, float]) -> None:
        """Set normalization statistics for this dataset.
        
        Args:
            stats: Dictionary mapping column names to their max values for normalization.
        """
        self._normalization_stats = stats
    
    def set_transform(self, transform: A.Compose) -> None:
        """Set the image transform for this dataset.
        
        Args:
            transform: Albumentations Compose transform.
        """
        self.transform = transform
    
    def _get_normalized_value(self, column: str, value: float) -> float:
        """Get normalized value using stored normalization stats.
        
        Args:
            column: Column name.
            value: Raw value.
            
        Returns:
            Normalized value if stats are set, otherwise raw value.
        """
        if self._normalization_stats is None:
            return value
        max_val = self._normalization_stats.get(column, 0)
        if max_val > 0:
            return value / max_val
        return value
        

    def _filter_only_complete_camera_sets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only dishes that have exactly `images_per_camera` images for each
        camera A, B, C, D.

        Prefer metadata columns if present, otherwise fall back to filesystem scan.
        """
        required = self.images_per_camera

        count_cols = [
            "num_images_camera_A",
            "num_images_camera_B",
            "num_images_camera_C",
            "num_images_camera_D",
        ]
        if all(col in df.columns for col in count_cols):
            mask = (
                (df["num_images_camera_A"] == required)
                & (df["num_images_camera_B"] == required)
                & (df["num_images_camera_C"] == required)
                & (df["num_images_camera_D"] == required)
            )
            filtered = df.loc[mask].reset_index(drop=True)
            if len(filtered) == 0:
                logger.warning(
                    "Filtering by metadata counts produced 0 dishes. "
                    "Falling back to filesystem scan."
                )
            else:
                return filtered

        valid_ids: list[str] = []
        for dish_id in df["dish_id"].astype(str).tolist():
            by_cam = self._list_images_by_camera(dish_id)
            if all(len(by_cam[c]) == required for c in ["A", "B", "C", "D"]):
                valid_ids.append(dish_id)
        return df[df["dish_id"].isin(valid_ids)].reset_index(drop=True)

    def _list_images_by_camera(self, dish_id: str) -> dict[str, list[Path]]:
        """
        Return dict: { 'A': [Path,...], 'B': [...], 'C': [...], 'D': [...] }
        Sorted deterministically within each camera.
        """
        dish_dir = self.data_path / str(dish_id)
        by_cam: dict[str, list[Path]] = {c: [] for c in ["A", "B", "C", "D"]}
        if not dish_dir.exists():
            return by_cam

        # Accept .jpeg/.jpg
        pat = re.compile(r"^camera_([A-D])_(.+)\.(jpe?g)$", re.IGNORECASE)

        for p in dish_dir.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if not m:
                continue
            cam = m.group(1).upper()
            by_cam[cam].append(p)

        def sort_key(path: Path) -> tuple[int, str]:
            # Try to extract a numeric index (natural sort); otherwise sort by name.
            m2 = re.search(r"(\d+)", path.stem)
            if m2:
                return (int(m2.group(1)), path.name)
            return (10**18, path.name)

        for cam in by_cam:
            by_cam[cam] = sorted(by_cam[cam], key=sort_key)

        return by_cam
    def __len__(self) -> int:
        return len(self.dish_metadata)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.dish_metadata.iloc[idx]
        dish_id = str(row["dish_id"])

        paths = self._image_paths[idx]  # length 20, fixed order

        view_tensors: list[torch.Tensor] = []
        for p in paths:
            with Image.open(p) as im:
                image_np = np.array(im.convert("RGB"))
            if self.transform is not None:
                transformed = self.transform(image=image_np)
                view_tensors.append(transformed["image"].float())
            else:
                # Convert to tensor without transform
                tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                view_tensors.append(tensor)

        # Reshape to (4, 5, C, H, W) in camera-major order.
        # - 4 = number of cameras / angles (A, B, C, D)
        # - 5 = number of frames per camera (frame_001 … frame_005)
        # - C = number of color channels (usually 3 for RGB)
        # - H, W = image height/width after transforms (e.g. 224×224)
        
        images = torch.stack(view_tensors, dim=0)
        images = images.view(4, self.images_per_camera, *images.shape[1:])
        
        return {
            "dish_id": dish_id,
            "images": images,
            "total_calories": torch.tensor(
                self._get_normalized_value("total_calories", row["total_calories"]),
                dtype=torch.float
            ),
            "total_mass": torch.tensor(
                self._get_normalized_value("total_mass", row["total_mass"]),
                dtype=torch.float
            ),
            "total_fat": torch.tensor(
                self._get_normalized_value("total_fat", row["total_fat"]),
                dtype=torch.float
            ),
            "total_carb": torch.tensor(
                self._get_normalized_value("total_carb", row["total_carb"]),
                dtype=torch.float
            ),
            "total_protein": torch.tensor(
                self._get_normalized_value("total_protein", row["total_protein"]),
                dtype=torch.float
            ),
            "num_ingrs": torch.tensor(row["num_ingrs"], dtype=torch.long),
        }
        
        
class NormalizedSubset(Dataset):
    """A subset of a dataset with its own normalization stats and transform."""
    
    def __init__(
        self,
        dataset: Nutrition,
        indices: list[int],
        transform: A.Compose | None = None,
        normalization_stats: dict[str, float] | None = None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.normalization_stats = normalization_stats
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> dict:
        # Get the actual index in the original dataset
        real_idx = self.indices[idx]
        row = self.dataset.dish_metadata.iloc[real_idx]
        dish_id = str(row["dish_id"])
        
        paths = self.dataset._image_paths[real_idx]
        
        view_tensors: list[torch.Tensor] = []
        for p in paths:
            with Image.open(p) as im:
                image_np = np.array(im.convert("RGB"))
            if self.transform is not None:
                transformed = self.transform(image=image_np)
                view_tensors.append(transformed["image"].float())
            else:
                tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                view_tensors.append(tensor)
        
        images = torch.stack(view_tensors, dim=0)
        images = images.view(4, self.dataset.images_per_camera, *images.shape[1:])
        
        def get_normalized(column: str, value: float) -> float:
            if self.normalization_stats is None:
                return value
            max_val = self.normalization_stats.get(column, 0)
            return value / max_val if max_val > 0 else value
        
        return {
            "dish_id": dish_id,
            "images": images,
            "total_calories": torch.tensor(
                get_normalized("total_calories", row["total_calories"]),
                dtype=torch.float
            ),
            "total_mass": torch.tensor(
                get_normalized("total_mass", row["total_mass"]),
                dtype=torch.float
            ),
            "total_fat": torch.tensor(
                get_normalized("total_fat", row["total_fat"]),
                dtype=torch.float
            ),
            "total_carb": torch.tensor(
                get_normalized("total_carb", row["total_carb"]),
                dtype=torch.float
            ),
            "total_protein": torch.tensor(
                get_normalized("total_protein", row["total_protein"]),
                dtype=torch.float
            ),
            "num_ingrs": torch.tensor(row["num_ingrs"], dtype=torch.long),
        }


class NutritionDataModule(L.LightningDataModule):
    """Lightning DataModule for Nutrition dataset."""
    def __init__(
        self,
        data_path: str | Path,
        dish_csv: str | Path,
        batch_size: int = 32,
        image_size: int = 224,
        train_transform: A.Compose | None = None,
        val_transform: A.Compose | None = None,
        normalise_dish_metadata: bool = False,
        val_split: float = 0.2,
        num_workers: int = 6,
        use_only_dishes_with_all_cameras: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.dish_csv = dish_csv
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.normalise_dish_metadata = normalise_dish_metadata
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.use_only_dishes_with_all_cameras = use_only_dishes_with_all_cameras
        
    def _compute_normalization_stats(
        self, 
        dataset: Nutrition, 
        indices: list[int]
    ) -> dict[str, float]:
        """Compute normalization statistics for a subset of the dataset.
        
        Args:
            dataset: The full Nutrition dataset.
            indices: List of indices for this split.
            
        Returns:
            Dictionary mapping column names to their max values.
        """
        stats = {}
        subset_df = dataset.dish_metadata.iloc[indices]
        
        for column in _DISH_TOTAL_COLUMNS[1:]:
            if column in subset_df.columns:
                stats[column] = subset_df[column].max()
            else:
                stats[column] = 1.0  # Default to 1 if column not found
                
        return stats
        
    def setup(self, stage: str | None = None):
        # Create base dataset without transforms (transforms applied per-split)
        full_dataset = Nutrition(
            data_path=self.data_path,
            dish_csv=self.dish_csv,
            transform=None,  # No transform at dataset level
            use_only_dishes_with_all_cameras=self.use_only_dishes_with_all_cameras,
        )
        
        # Split indices
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size
        
        # Generate reproducible random split
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Use default transforms if none provided
        train_transform = self.train_transform or get_default_transforms(self.image_size)
        val_transform = self.val_transform or get_default_transforms(self.image_size)
        
        # Compute normalization stats for each split independently
        train_stats = None
        val_stats = None
        if self.normalise_dish_metadata:
            train_stats = self._compute_normalization_stats(full_dataset, train_indices)
            val_stats = self._compute_normalization_stats(full_dataset, val_indices)
            logger.info("Train normalization stats: %s", train_stats)
            logger.info("Val normalization stats: %s", val_stats)
        
        # Create subsets with their own transforms and normalization
        self.train_dataset = NormalizedSubset(
            dataset=full_dataset,
            indices=train_indices,
            transform=train_transform,
            normalization_stats=train_stats,
        )
        
        self.val_dataset = NormalizedSubset(
            dataset=full_dataset,
            indices=val_indices,
            transform=val_transform,
            normalization_stats=val_stats,
        )
        
        # Store reference to full dataset for potential access
        self._full_dataset = full_dataset
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )


class NutritionFastDataModule(L.LightningDataModule):
    """Fast Lightning DataModule using pre-processed HDF5 images.
    
    This DataModule is optimized for fast training by:
    1. Using pre-resized images stored in HDF5 format
    2. Using z-score normalization with pre-computed stats
    3. Optimized data loading with prefetching
    
    Before using this module, preprocess images with:
        python scripts/preprocess_images.py --data-dir data.nosync --output data.nosync/images_224.h5
    
    Example:
        data_module = NutritionFastDataModule(
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
        self.dish_csv = Path(dish_csv)
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.normalization_method = normalization_method
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        
        # Will be set in setup()
        self.train_dataset: NutritionHDF5Subset | None = None
        self.val_dataset: NutritionHDF5Subset | None = None
        self._full_dataset: NutritionHDF5 | None = None
        
    def setup(self, stage: str | None = None) -> None:
        """Set up the datasets."""
        # Create base HDF5 dataset
        self._full_dataset = NutritionHDF5(
            h5_path=self.h5_path,
            dish_csv=self.dish_csv,
            transform=None,  # Transform applied per-subset
            normalization_method=self.normalization_method,
        )
        
        # Split indices
        dataset_size = len(self._full_dataset)
        val_size = int(dataset_size * self.val_split)
        train_size = dataset_size - val_size
        
        # Generate reproducible split
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Use default transforms with/without augmentation
        train_transform = self.train_transform or get_augmentation_transforms(224)
        val_transform = self.val_transform or get_default_transforms(224)
        
        # Create subsets
        self.train_dataset = NutritionHDF5Subset(
            dataset=self._full_dataset,
            indices=train_indices,
            transform=train_transform,
        )
        
        self.val_dataset = NutritionHDF5Subset(
            dataset=self._full_dataset,
            indices=val_indices,
            transform=val_transform,
        )
        
        logger.info(f"NutritionFastDataModule: {train_size} train, {val_size} val samples")
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,  # Avoid batch size variations
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
    # Test the DataModule
    
    
    # Custom transforms with augmentation for training only
    # train_transform = A.Compose([
    #     A.Resize(224, 224),
    #     A.HorizontalFlip(p=0.5),
    #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     A.ToTensorV2(),
    # ])

    # data_module = NutritionDataModule(
    #     data_path="data.nosync",
    #     dish_csv="path/to/csv",
    #     train_transform=train_transform,
    #     val_transform=get_default_transforms(224),
    #     normalise_dish_metadata=True,
    # )
    
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),  # Data augmentation only for training
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        A.ToTensorV2(),
    ])
    
    val_transform = get_default_transforms(224)  # No augmentation for validation
    
    data_module = NutritionDataModule(
        data_path="data.nosync",
        dish_csv="src/wizard_ops/metadata/data_stats.csv",
        batch_size=8,
        image_size=224,
        train_transform=train_transform,
        val_transform=val_transform,
        normalise_dish_metadata=True,
        val_split=0.2,
        num_workers=4,
        seed=42,
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Size of first training batch: {next(iter(train_loader))['images'].shape}")
    print(f"Size of first validation batch: {next(iter(val_loader))['images'].shape}")
    # for batch in train_loader:
    #     print(batch)
    #     break
