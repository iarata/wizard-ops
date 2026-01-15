from pathlib import Path
import os
import typer
import lightning as L
import kagglehub as kh
import albumentations as A
import pandas as pd
import torch
import numpy as np
from PIL import Image
from typing import Annotated
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

app = typer.Typer(help="Commands to manage the Nutrition dataset")

# Path to metadata files
METADATA_DIR = Path(__file__).parent / "metadata"
DISH_METADATA_CAFE1 = METADATA_DIR / "dish_metadata_cafe1.csv"
DISH_METADATA_CAFE2 = METADATA_DIR / "dish_metadata_cafe2.csv"
INGREDIENTS_METADATA = METADATA_DIR / "ingredients_metadata.csv"

# Columns for dish metadata CSV
DISH_COLUMNS = ["dish_id", "total_calories", "total_mass", "total_fat", "total_carbs", "total_protein"]


def load_dish_metadata() -> pd.DataFrame:
    """Load and combine dish metadata from both cafe CSV files."""
    dfs = []
    for metadata_file in [DISH_METADATA_CAFE1, DISH_METADATA_CAFE2]:
        if metadata_file.exists():
            # Read raw CSV lines and parse dish-level info
            with open(metadata_file, "r") as f:
                lines = f.readlines()
            
            records = []
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    dish_id = parts[0]
                    total_calories = float(parts[1])
                    total_mass = float(parts[2])
                    total_fat = float(parts[3])
                    total_carbs = float(parts[4])
                    total_protein = float(parts[5])
                    records.append({
                        "dish_id": dish_id,
                        "total_calories": total_calories,
                        "total_mass": total_mass,
                        "total_fat": total_fat,
                        "total_carbs": total_carbs,
                        "total_protein": total_protein,
                    })
            
            if records:
                dfs.append(pd.DataFrame(records))
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=DISH_COLUMNS)


def load_ingredients_metadata() -> pd.DataFrame:
    """Load ingredients metadata."""
    if INGREDIENTS_METADATA.exists():
        return pd.read_csv(INGREDIENTS_METADATA)
    return pd.DataFrame()


# This process the items in a given dataset
class Nutrition(Dataset):
    """Nutrition Dataset for food nutrition data with images.
    
    Each sample contains:
    - images: Dictionary of camera images (A, B, C, D)
    - nutrition: Dictionary with total_calories, total_mass, total_fat, total_carbs, total_protein
    - dish_id: The dish identifier
    """

    def __init__(
        self,
        data_path: Path,
        transform: A.Compose | None = None,
        camera: str = "A",
        frame_idx: int = 1,
    ) -> None:
        """Initialize the Nutrition dataset.
        
        Args:
            data_path: Path to the data directory containing dish folders.
            transform: Albumentations transform to apply to images.
            camera: Camera angle to use ('A', 'B', 'C', 'D', or 'all').
            frame_idx: Frame index to use (1-5).
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.camera = camera
        self.frame_idx = frame_idx
        
        # Load metadata
        self.dish_metadata = load_dish_metadata()
        
        # Find all valid dish folders that exist in both data and metadata
        self.dish_ids = self._find_valid_dishes()

    def _find_valid_dishes(self) -> list[str]:
        """Find all dish IDs that have both data folders and metadata."""
        valid_dishes = []
        
        if not self.data_path.exists():
            return valid_dishes
        
        # Get all dish folders from data directory
        dish_folders = [
            d.name for d in self.data_path.iterdir()
            if d.is_dir() and d.name.startswith("dish_")
        ]
        
        # Filter to only those with metadata
        metadata_dishes = set(self.dish_metadata["dish_id"].tolist())
        
        for dish_id in dish_folders:
            if dish_id in metadata_dishes:
                # Check if images exist for the specified camera
                frames_dir = self.data_path / dish_id / "frames_sampled30"
                if frames_dir.exists():
                    # Check if at least one image exists for this camera
                    matches = list(frames_dir.glob(f"camera_{self.camera}_frame_*.jp*g"))
                    if matches:
                        valid_dishes.append(dish_id)
        
        return sorted(valid_dishes)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dish_ids)

    def __getitem__(self, index: int) -> dict:
        """Return a given sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Dictionary containing:
                - 'image': Tensor of the image (C, H, W)
                - 'calories': Total calories
                - 'mass': Total mass in grams
                - 'fat': Total fat in grams
                - 'carbs': Total carbohydrates in grams
                - 'protein': Total protein in grams
                - 'dish_id': Dish identifier string
        """
        if index >= len(self.dish_ids):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dish_ids)}")
        
        dish_id = self.dish_ids[index]
        
        # Load image
        image = self._load_image(dish_id)
        
        # Get nutrition info
        nutrition_row = self.dish_metadata[self.dish_metadata["dish_id"] == dish_id].iloc[0]
        
        # Apply transforms if provided
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        # Convert image to tensor if not already
        if isinstance(image, np.ndarray):
            # Convert HWC to CHW format
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return {
            "image": image,
            "calories": torch.tensor(nutrition_row["total_calories"], dtype=torch.float32),
            "mass": torch.tensor(nutrition_row["total_mass"], dtype=torch.float32),
            "fat": torch.tensor(nutrition_row["total_fat"], dtype=torch.float32),
            "carbs": torch.tensor(nutrition_row["total_carbs"], dtype=torch.float32),
            "protein": torch.tensor(nutrition_row["total_protein"], dtype=torch.float32),
            "dish_id": dish_id,
        }

    def _load_image(self, dish_id: str) -> np.ndarray:
        """Load image for a given dish.
        
        Args:
            dish_id: The dish identifier.
            
        Returns:
            Image as numpy array (H, W, C).
        """
        frames_dir = self.data_path / dish_id / "frames_sampled30"
        
        # Construct image filename
        image_name = f"camera_{self.camera}_frame_{self.frame_idx:03d}.jpeg"
        image_path = frames_dir / image_name

        # Try the requested frame first (jpeg then jpg)
        candidates = [image_path, frames_dir / image_name.replace(".jpeg", ".jpg")]
        chosen = next((p for p in candidates if p.exists()), None)

        # Fallback: pick the first available frame for this camera
        if chosen is None:
            matches = sorted(frames_dir.glob(f"camera_{self.camera}_frame_*.jp*g"))
            if matches:
                chosen = matches[0]

        if chosen is None:
            raise FileNotFoundError(
                f"Image not found in {frames_dir} for camera={self.camera}, frame_idx={self.frame_idx}"
            )

        # Load image
        image = Image.open(chosen).convert("RGB")
        return np.array(image)

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder.
        
        This method can be used to resize images, normalize data, etc.
        
        Args:
            output_folder: Path to save preprocessed data.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save metadata as a single parquet file for faster loading
        if len(self.dish_metadata) > 0:
            self.dish_metadata.to_parquet(output_folder / "dish_metadata.parquet", index=False)
        
        # Create a mapping file for dish_id to index
        if self.dish_ids:
            mapping_df = pd.DataFrame({
                "index": range(len(self.dish_ids)),
                "dish_id": self.dish_ids,
            })
            mapping_df.to_csv(output_folder / "dish_mapping.csv", index=False)

# This generates a wrapper around the Nutrition dataset for use with Lightning
class NutritionDataset(L.LightningDataModule):
    """Lightning DataModule for food nutrition data.
    
    Handles train/val/test splits and data loading for the Nutrition5K dataset.
    """

    def __init__(
        self,
        data_path: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),
        camera: str = "A",
        frame_idx: int = 1,
        transform: A.Compose | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize the NutritionDataset DataModule.
        
        Args:
            data_path: Path to the data directory containing dish folders.
            batch_size: Batch size for dataloaders.
            num_workers: Number of worker processes for data loading.
            train_val_test_split: Tuple of (train, val, test) split ratios.
            camera: Camera angle to use ('A', 'B', 'C', 'D').
            frame_idx: Frame index to use (1-5).
            transform: Albumentations transform to apply to images.
            seed: Random seed for reproducible splits.
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.camera = camera
        self.frame_idx = frame_idx
        self.transform = transform
        self.seed = seed
        
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up the dataset splits for training, validation, and testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict').
        """
        # Create the full dataset
        full_dataset = Nutrition(
            data_path=self.data_path,
            transform=self.transform,
            camera=self.camera,
            frame_idx=self.frame_idx,
        )
        
        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(total_size * self.train_val_test_split[0])
        val_size = int(total_size * self.train_val_test_split[1])
        test_size = total_size - train_size - val_size
        
        # Create reproducible splits
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator,
        )

        

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not set. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset not set. Call setup() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset not set. Call setup() first.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


def get_default_transforms(image_size: int = 224) -> A.Compose:
    """Get default image transforms for the dataset.
    
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
    ])




# Typer CLI commands to download the dataset from Kaggle
@app.command("download")
def download(dir: Annotated[str, typer.Option("--dir", "-d", help="Directory to download the dataset to")]):
    """Download the Nutrition Dataset from Kaggle."""
    path = Path(dir)
    try:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            kh_path = kh.dataset_download(
                "zygmuntyt/nutrition5k-dataset-side-angle-images",                
            )
            os.rename(kh_path, path)
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
    return path
    typer.echo(f"Dataset downloaded to {dir}")

if __name__ == "__main__":
    app()
