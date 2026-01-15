
import pytest
import pandas as pd
import lightning as L
import albumentations as A
from torch.utils.data import Dataset
from wizard_ops.data import Nutrition
from wizard_ops.data import NutritionDataset
from wizard_ops.data import get_default_transforms, load_dish_metadata, load_ingredients_metadata


def test_nutrition_dataset_init(tmp_path):
    """Test the Nutrition dataset init."""
    dataset = Nutrition(data_path=tmp_path)
    assert isinstance(dataset, Dataset)
    assert dataset.camera == "A"
    assert dataset.frame_idx == 1
    assert len(dataset) == 0  # No valid dishes in empty tmp_path


def test_nutrition_dataset_len_empty(tmp_path):
    """Test the Nutrition dataset length with no data."""
    dataset = Nutrition(data_path=tmp_path)
    assert len(dataset) == 0


def test_nutrition_dataset_custom_camera(tmp_path):
    """Test the Nutrition dataset with custom camera setting."""
    dataset = Nutrition(data_path=tmp_path, camera="B", frame_idx=3)
    assert dataset.camera == "B"
    assert dataset.frame_idx == 3


def test_nutrition_dataset_index_error(tmp_path):
    """Test that IndexError is raised for invalid index."""
    dataset = Nutrition(data_path=tmp_path)
    with pytest.raises(IndexError):
        dataset[0]


def test_nutrition_datamodule_init(tmp_path):
    """Test the NutritionDataset DataModule init."""
    datamodule = NutritionDataset(data_path=tmp_path, batch_size=16)
    assert isinstance(datamodule, L.LightningDataModule)
    assert datamodule.batch_size == 16
    assert datamodule.num_workers == 4
    assert datamodule.train_val_test_split == (0.7, 0.15, 0.15)


def test_nutrition_datamodule_custom_split(tmp_path):
    """Test the NutritionDataset DataModule with custom split."""
    datamodule = NutritionDataset(
        data_path=tmp_path,
        train_val_test_split=(0.8, 0.1, 0.1),
        seed=123,
    )
    assert datamodule.train_val_test_split == (0.8, 0.1, 0.1)
    assert datamodule.seed == 123


def test_nutrition_datamodule_setup(tmp_path):
    """Test the NutritionDataset DataModule setup."""
    datamodule = NutritionDataset(data_path=tmp_path)
    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None


def test_nutrition_datamodule_dataloaders_error(tmp_path):
    """Test that dataloaders raise error before setup."""
    datamodule = NutritionDataset(data_path=tmp_path)
    with pytest.raises(RuntimeError):
        datamodule.train_dataloader()
    with pytest.raises(RuntimeError):
        datamodule.val_dataloader()
    with pytest.raises(RuntimeError):
        datamodule.test_dataloader()


def test_get_default_transforms():
    """Test the default transforms function."""
    transform = get_default_transforms(image_size=224)
    assert isinstance(transform, A.Compose)


def test_get_default_transforms_custom_size():
    """Test the default transforms with custom image size."""
    transform = get_default_transforms(image_size=512)
    assert transform is not None


def test_load_dish_metadata():
    """Test the load_dish_metadata function."""
    df = load_dish_metadata()
    assert isinstance(df, pd.DataFrame)


def test_load_ingredients_metadata():
    """Test the load_ingredients_metadata function."""
    df = load_ingredients_metadata()
    assert isinstance(df, pd.DataFrame)