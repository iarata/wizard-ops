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
        camera: str | None = None,
    ):
        
        self.data_path = Path(data_path)
        self.dish_csv = dish_csv
        self.transform = transform
        self.dish_metadata = pd.read_csv(dish_csv)
        if normalise_dish_metadata:
            self.dish_metadata = self._normalise_metadata(self.dish_metadata)
        self.camera = camera
        
    def _normalise_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dish metadata DataFrame according to the values of their corresponding columns."""
        df_normalized = df.copy()
        for column in _DISH_TOTAL_COLUMNS[1:]:
            max_value = df_normalized[column].max()
            if max_value > 0:
                df_normalized[column] = df_normalized[column] / max_value
        return df_normalized
        
        
    def _get_dish_images_paths(self, dish_id: str) -> list[Path]:
        """Get image paths for a given dish ID."""
        dish_dir = self.data_path / dish_id / "frames_sampled30"
        if not dish_dir.exists():
            return []
        image_paths = []
        for img_file in dish_dir.iterdir():
            # camera_{camera}_*.jpeg
            if self.camera:
                pattern = re.compile(rf"camera_{self.camera}_.*\.jpeg$")
                if pattern.match(img_file.name):
                    image_paths.append(img_file)
            else:
                if img_file.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    image_paths.append(img_file)
        return image_paths
    
    def __len__(self) -> int:
        return len(self.dish_metadata)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.dish_metadata.iloc[idx]
        dish_id = row["dish_id"]
        images = self._get_dish_images_paths(dish_id)
        
        if not images:
            logger.error(f"Dish path ")
            raise FileNotFoundError(f"No images found for dish ID {dish_id}")
        
        # Load and transform all images for the dish
        transformed_images = []
        for img_path in images[:2]:  # Limit to max 2 images per dish
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            transformed = self.transform(image=image_np)
            transformed_images.append(transformed["image"].float())
            
            
        # Since different dishes may have different number of images,
        # we stack them into a tensor of shape (V, C, H, W) where V is the number of views (images)
        # v_images = torch.stack(transformed_images, dim=0)
        # Only when we have multiple images per dish and not fixed number of images per dish
        
        return {
            "dish_id": dish_id,
            "images": torch.stack(transformed_images, dim=0),
            "total_calories": torch.tensor(row["total_calories"], dtype=torch.float),
            "total_mass": torch.tensor(row["total_mass"], dtype=torch.float),
            "total_fat": torch.tensor(row["total_fat"], dtype=torch.float),
            "total_carb": torch.tensor(row["total_carb"], dtype=torch.float),
            "total_protein": torch.tensor(row["total_protein"], dtype=torch.float),
            "num_ingrs": torch.tensor(row["num_ingrs"], dtype=torch.int),
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
        camera: str | None = None,
        val_split: float = 0.2,
        num_workers: int = 6,
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.dish_csv = dish_csv
        self.batch_size = batch_size
        self.image_size = image_size
        self.normalise_dish_metadata = normalise_dish_metadata
        self.camera = camera
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        
    def setup(self, stage: str | None = None):
        full_dataset = Nutrition(
            data_path=self.data_path,
            dish_csv=self.dish_csv,
            transform=get_default_transforms(self.image_size),
            normalise_dish_metadata=self.normalise_dish_metadata,
            camera=self.camera,
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
        dish_csv="src/wizard_ops/metadata/dish_metadata_A.csv",
        batch_size=8,
        image_size=224,
        normalise_dish_metadata=True,
        camera="A",
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
