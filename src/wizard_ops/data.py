import logging
import os
import re
from pathlib import Path
from typing import Annotated

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

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
    
    
class Nutrition(Dataset):
    """Custom dataset for Nutrition data."""
    def __init__(
        self,
        data_path: str | Path,
        transform: A.Compose,
        dish_csv: str | Path | None = None,
        normalise_dish_metadata: bool = False,
        use_only_dishes_with_all_cameras: bool = False,
        images_per_camera: int = 5,
    ):
        
        self.data_path = Path(data_path)
        self.dish_csv = dish_csv
        self.transform = transform
        self.images_per_camera = images_per_camera
        self.dish_metadata = pd.read_csv(dish_csv, dtype={"dish_id": str})
        self._image_paths: list[list[Path]] = []

        # Build index of image paths (and optionally filter to full-camera dishes).
        self.dish_metadata = self._build_index(
            self.dish_metadata,
            require_full_cameras=use_only_dishes_with_all_cameras,
        )

        if normalise_dish_metadata:
            self.dish_metadata = self._normalise_metadata(self.dish_metadata)
            
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
                # If you ever want to allow incomplete dishes later, you'd
                # implement variable-length logic here. For now: skip only when
                # strict mode is on.
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
            transformed = self.transform(image=image_np)
            view_tensors.append(transformed["image"].float())

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
            "total_calories": torch.tensor(row["total_calories"], dtype=torch.float),
            "total_mass": torch.tensor(row["total_mass"], dtype=torch.float),
            "total_fat": torch.tensor(row["total_fat"], dtype=torch.float),
            "total_carb": torch.tensor(row["total_carb"], dtype=torch.float),
            "total_protein": torch.tensor(row["total_protein"], dtype=torch.float),
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
        self.normalise_dish_metadata = normalise_dish_metadata
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        self.use_only_dishes_with_all_cameras = use_only_dishes_with_all_cameras
        
    def setup(self, stage: str | None = None):
        full_dataset = Nutrition(
            data_path=self.data_path,
            dish_csv=self.dish_csv,
            transform=get_default_transforms(self.image_size),
            normalise_dish_metadata=self.normalise_dish_metadata,
            use_only_dishes_with_all_cameras=self.use_only_dishes_with_all_cameras,
        )
        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))
        
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



if __name__ == "__main__":
    # Test the DataModule
    data_module = NutritionDataModule(
        data_path="data.nosync",
        dish_csv="src/wizard_ops/metadata/data_stats.csv",
        batch_size=8,
        image_size=224,
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
