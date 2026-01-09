from pathlib import Path
import os
import typer
import lightning as L
import kagglehub as kh
from typing import Annotated
from torch.utils.data import Dataset
from torch.utils.data import random_split

app = typer.Typer(help="Commands to manage the Nutrition dataset")

# This process the items in a given dataset
class Nutrition(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = Path(data_path)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        # placeholder implementation; replace with real logic
        return 0

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        # reference index to avoid unused-parameter hint
        raise IndexError(f"Dataset is empty (requested index {index})")

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        # placeholder: user should implement actual preprocessing logic here

# This generates a wrapper around the Nutrition dataset for use with Lightning
class NutritionDataset(L.LightningDataModule):
    """Dataset for food nutrition data."""

    def __init__(self, data_path: Path, batch_size: int = 32) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        """Download and set up the dataset for training, validation, and testing."""
        # reference stage to avoid unused-parameter hint
        if stage:
            # stage-specific behavior could be implemented here
            pass

        # ensure data is downloaded
        raw_path = self.data_path

        # minimal setup: instantiate dataset placeholders (user can replace with real logic)
        self.train_dataset = MyDataset(raw_path / "train")
        self.val_dataset = MyDataset(raw_path / "val")
        self.test_dataset = MyDataset(raw_path / "test")

    def train_dataloader(self):
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not set. Call setup() first.")
        from torch.utils.data import DataLoader

        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset not set. Call setup() first.")
        from torch.utils.data import DataLoader

        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return the test dataloader."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset not set. Call setup() first.")
        from torch.utils.data import DataLoader

        return DataLoader(self.test_dataset, batch_size=self.batch_size)




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
