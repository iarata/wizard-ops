import albumentations as A
import lightning as L
import pandas as pd
import pytest
from torch.utils.data import Dataset
from wizard_ops.data-cli import Nutrition
from wizard_ops.data-cli import NutritionDataset
from wizard_ops.data-cli import get_default_transforms, load_dish_metadata, load_ingredients_metadata


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
    import albumentations as A
    import lightning as L
    import pandas as pd
    import pytest
    from torch.utils.data import Dataset
    from wizard_ops.data import (
        NormalizedSubset,
        Nutrition,
        NutritionDataModule,
        get_default_transforms,
    )


    def _make_csv(tmp_path, names):
        path = tmp_path / "dishes.csv"
        df = pd.DataFrame({"dish_id": names})
        df.to_csv(path, index=False)
        return path


    def test_nutrition_dataset_init(tmp_path):
        """Test the Nutrition dataset init with an empty data dir and CSV."""
        csv_path = _make_csv(tmp_path, [])
        dataset = Nutrition(data_path=tmp_path, dish_csv=csv_path)
        assert isinstance(dataset, Dataset)
        assert len(dataset) == 0


    def test_nutrition_dataset_len_empty(tmp_path):
        csv_path = _make_csv(tmp_path, [])
        dataset = Nutrition(data_path=tmp_path, dish_csv=csv_path)
        assert len(dataset) == 0


    def test_nutrition_dataset_custom_images_per_camera(tmp_path):
        csv_path = _make_csv(tmp_path, [])
        dataset = Nutrition(data_path=tmp_path, dish_csv=csv_path, images_per_camera=3)
        assert dataset.images_per_camera == 3


    def test_nutrition_dataset_index_error(tmp_path):
        csv_path = _make_csv(tmp_path, [])
        dataset = Nutrition(data_path=tmp_path, dish_csv=csv_path)
        with pytest.raises(IndexError):
            _ = dataset[0]


    def test_nutrition_datamodule_init(tmp_path):
        csv_path = _make_csv(tmp_path, ["1", "2"])
        datamodule = NutritionDataModule(data_path=tmp_path, dish_csv=csv_path, batch_size=16)
        assert isinstance(datamodule, L.LightningDataModule)
        assert datamodule.batch_size == 16
        assert datamodule.num_workers == 6
        assert datamodule.val_split == 0.2


    def test_nutrition_datamodule_custom_split(tmp_path):
        csv_path = _make_csv(tmp_path, ["1"])
        datamodule = NutritionDataModule(data_path=tmp_path, dish_csv=csv_path, val_split=0.1, seed=123)
        assert datamodule.val_split == 0.1
        assert datamodule.seed == 123


    def test_nutrition_datamodule_setup(tmp_path):
        csv_path = _make_csv(tmp_path, ["1", "2", "3"])
        datamodule = NutritionDataModule(data_path=tmp_path, dish_csv=csv_path, batch_size=8)
        datamodule.setup()
        assert hasattr(datamodule, "train_dataset")
        assert hasattr(datamodule, "val_dataset")
        assert isinstance(datamodule.train_dataset, Dataset)
        assert isinstance(datamodule.val_dataset, Dataset)


    def test_nutrition_datamodule_dataloaders_error(tmp_path):
        csv_path = _make_csv(tmp_path, ["1"])
        datamodule = NutritionDataModule(data_path=tmp_path, dish_csv=csv_path)
        # dataloaders are only available after setup(); before that an AttributeError is expected
        with pytest.raises(AttributeError):
            datamodule.train_dataloader()
        with pytest.raises(AttributeError):
            datamodule.val_dataloader()


    def test_get_default_transforms():
        transform = get_default_transforms(image_size=224)
        assert isinstance(transform, A.Compose)


    def test_get_default_transforms_custom_size():
        transform = get_default_transforms(image_size=512)
        assert transform is not None