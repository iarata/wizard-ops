import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import wizard_ops.train as train_module
from PIL import Image


def _make_dummy_dataset(tmpdir: Path, dish_id: str = "dish_0", camera: str = "D", frame_idx: int = 1):
    data_root = tmpdir / "data.nosync"
    frames_dir = data_root / dish_id / "frames_sampled30"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # create one small RGB image
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    image_name = f"camera_{camera}_frame_{frame_idx:03d}.jpeg"
    img.save(frames_dir / image_name)

    # return the path to the data root and the dish id used
    return data_root, dish_id


def test_train_uses_provided_dataset(monkeypatch, tmp_path: Path):
    # Create a minimal dataset on disk
    data_root, dish_id = _make_dummy_dataset(tmp_path)

    # Prepare a fake metadata DataFrame to match the dataset
    import pandas as pd

    df = pd.DataFrame([
        {
            "dish_id": dish_id,
            "total_calories": 100.0,
            "total_mass": 200.0,
            "total_fat": 10.0,
            "total_carbs": 20.0,
            "total_protein": 5.0,
        }
    ])

    # Monkeypatch the loader to return our tiny DataFrame so Nutrition will accept the dish
    monkeypatch.setattr("wizard_ops.data.load_dish_metadata", lambda: df)

    # Monkeypatch the NutritionDataset constructor used in train_module so it points to our data_root
    RealDM = train_module.NutritionDataset

    def dm_factory(*args, **kwargs):
        # force the data_path to our temporary dataset
        kwargs["data_path"] = Path(data_root)
        return RealDM(*args, **kwargs)

    monkeypatch.setattr(train_module, "NutritionDataset", dm_factory)

    # Capture the datamodule passed to Trainer.fit
    captured = {}

    def fake_fit(self, model, datamodule=None, *a, **kw):
        captured["datamodule"] = datamodule
        return None

    monkeypatch.setattr(train_module.Trainer, "fit", fake_fit)

    # Run train with a fast_dev_run to keep it light (we monkeypatched fit anyway)
    train_module.train(frame_idx=1, num_workers=0, fast_dev_run=True, logger_type="tensorboard")

    # Ensure trainer.fit received a datamodule and it points to our temp data root
    assert "datamodule" in captured
    dm = captured["datamodule"]
    assert dm is not None
    # Check that the DataModule's data_path points to our temp data
    assert str(data_root) in str(dm.data_path)

    # Ensure the underlying dataset finds our dish_id after setup()
    dm.setup(stage="fit")
    # The underlying full dataset is created inside setup; check dish ids
    # Access via dm.train_dataset if available, or create Nutrition instance
    if hasattr(dm, "train_dataset") and dm.train_dataset is not None:
        # random_split returns Subset objects; get the underlying dataset if so
        from torch.utils.data import Subset

        base = dm.train_dataset
        if isinstance(base, Subset):
            base = base.dataset
        # base should be Nutrition dataset with dish_ids attribute
        assert dish_id in getattr(base, "dish_ids", [])
    else:
        # Fallback: instantiate a Nutrition dataset directly and inspect
        from wizard_ops.data import Nutrition

        nu = Nutrition(data_path=data_root, camera=train_module.config["camera"], frame_idx=1)
        assert dish_id in nu.dish_ids
