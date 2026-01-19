from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from wizard_ops.data import (
	NUM_CAMERAS,
	NUM_FRAMES,
	TARGET_COLUMNS,
	Nutrition,
	NutritionDataModule,
	NutritionSubset,
	TargetNormaliser,
)


@pytest.fixture()
def normalization_stats() -> dict:
	# Deterministic, non-degenerate stats for all target columns.
	return {
		"targets": {
			name: {"mean": 10.0, "std": 2.0, "min": 0.0, "max": 20.0}
			for name in TARGET_COLUMNS
		}
	}


@pytest.fixture()
def tiny_h5_and_csv(tmp_path: Path) -> tuple[Path, Path]:
	"""Create a minimal HDF5 + CSV pair compatible with Nutrition."""
	h5_path = tmp_path / "images.h5"
	csv_path = tmp_path / "dish.csv"

	# Small images for speed; dataset expects uint8 and shape (cams, frames, 3, H, W).
	n = 4
	h, w = 8, 8
	images = np.zeros((n, NUM_CAMERAS, NUM_FRAMES, 3, h, w), dtype=np.uint8)
	dish_ids = np.array([b"a", b"b", b"c", b"d"], dtype="S")

	with h5py.File(h5_path, "w") as f:
		f.create_dataset("images", data=images)
		f.create_dataset("dish_ids", data=dish_ids)

	# Include one extra row that should be filtered out (dish_id not in HDF5).
	rows = []
	for dish_id in ["a", "b", "c", "d", "not_in_h5"]:
		row: dict[str, object] = {
			"dish_id": dish_id,
			"num_ingrs": 3,
		}
		for col in TARGET_COLUMNS:
			row[col] = 12.0
		rows.append(row)
	pd.DataFrame(rows).to_csv(csv_path, index=False)

	return h5_path, csv_path


def test_target_normaliser_methods(normalization_stats: dict) -> None:
	tn_none = TargetNormaliser(stats=normalization_stats, method="none")
	assert tn_none.normalize("total_calories", 12.0) == 12.0

	tn_z = TargetNormaliser(stats=normalization_stats, method="zscore")
	# (12 - 10) / 2 = 1
	assert tn_z.normalize("total_calories", 12.0) == pytest.approx(1.0)

	tn_mm = TargetNormaliser(stats=normalization_stats, method="minmax")
	# (12 - 0) / (20 - 0) = 0.6
	assert tn_mm.normalize("total_calories", 12.0) == pytest.approx(0.6)

	tn_max = TargetNormaliser(stats=normalization_stats, method="max")
	# 12 / 20 = 0.6
	assert tn_max.normalize("total_calories", 12.0) == pytest.approx(0.6)


def test_target_normaliser_missing_stats_returns_value() -> None:
	tn = TargetNormaliser(stats={}, method="zscore")
	assert tn.normalize("total_calories", 12.0) == 12.0


def test_nutrition_builds_index_and_filters_metadata(
	tiny_h5_and_csv: tuple[Path, Path],
	normalization_stats: dict,
) -> None:
	h5_path, csv_path = tiny_h5_and_csv
	ds = Nutrition(
		h5_path=h5_path,
		dish_csv=csv_path,
		transform=None,
		normalisation_method="zscore",
		normalization_stats=normalization_stats,
	)

	# Filters out the CSV row not present in HDF5.
	assert len(ds) == 4
	assert set(ds.metadata["dish_id"].astype(str).tolist()) == {"a", "b", "c", "d"}

	# Index mapping points at the corresponding HDF5 row.
	assert ds._idx_to_h5_idx == [0, 1, 2, 3]


def test_nutrition_getstate_clears_h5_handle(
	tiny_h5_and_csv: tuple[Path, Path],
	normalization_stats: dict,
) -> None:
	h5_path, csv_path = tiny_h5_and_csv
	ds = Nutrition(
		h5_path=h5_path,
		dish_csv=csv_path,
		normalisation_method="zscore",
		normalization_stats=normalization_stats,
	)
	_ = ds._get_h5()
	assert ds._h5_file is not None

	state = ds.__getstate__()
	assert state["_h5_file"] is None


def test_nutrition_get_example_imagenet_normalize_default(
	tiny_h5_and_csv: tuple[Path, Path],
	normalization_stats: dict,
) -> None:
	h5_path, csv_path = tiny_h5_and_csv
	ds = Nutrition(
		h5_path=h5_path,
		dish_csv=csv_path,
		transform=None,
		normalisation_method="zscore",
		normalization_stats=normalization_stats,
	)

	ex = ds.get_example(0)
	assert ex["dish_id"] == "a"
	assert torch.is_tensor(ex["images"])
	assert ex["images"].shape == (NUM_CAMERAS, NUM_FRAMES, 3, 8, 8)
	assert ex["images"].dtype == torch.float32

	# Input images are zeros; ImageNet normalization should yield -mean/std.
	expected_r = -0.485 / 0.229
	assert float(ex["images"][0, 0, 0, 0, 0]) == pytest.approx(expected_r, rel=1e-5)

	# Targets are present and vector matches per-key targets.
	assert ex["targets"].shape == (len(TARGET_COLUMNS),)
	for i, name in enumerate(TARGET_COLUMNS):
		assert torch.allclose(ex["targets"][i], ex[name])


def test_nutrition_transform_path_uses_apply_transform(
	tiny_h5_and_csv: tuple[Path, Path],
	normalization_stats: dict,
) -> None:
	h5_path, csv_path = tiny_h5_and_csv
	ds = Nutrition(
		h5_path=h5_path,
		dish_csv=csv_path,
		transform=None,
		normalisation_method="zscore",
		normalization_stats=normalization_stats,
	)

	tf = Compose([ToTensorV2()])
	ex = ds.get_example(0, transform=tf)
	assert ex["images"].shape == (NUM_CAMERAS, NUM_FRAMES, 3, 8, 8)
	assert torch.is_tensor(ex["images"])


def test_nutrition_subset_applies_own_transform(
	tiny_h5_and_csv: tuple[Path, Path],
	normalization_stats: dict,
) -> None:
	h5_path, csv_path = tiny_h5_and_csv
	base = Nutrition(
		h5_path=h5_path,
		dish_csv=csv_path,
		transform=None,
		normalisation_method="zscore",
		normalization_stats=normalization_stats,
	)

	subset = NutritionSubset(base, indices=[0], transform=Compose([ToTensorV2()]))

	base_ex = base[0]
	subset_ex = subset[0]

	assert base.transform is None
	assert base_ex["dish_id"] == subset_ex["dish_id"]
	assert base_ex["images"].shape == subset_ex["images"].shape
	# Base path uses ImageNet normalization (float32), subset transform path may not.
	assert base_ex["images"].dtype == torch.float32


def test_datamodule_setup_and_dataloaders(
	monkeypatch: pytest.MonkeyPatch,
	tiny_h5_and_csv: tuple[Path, Path],
	normalization_stats: dict,
) -> None:
	# Avoid relying on repo-level JSON stats file.
	import wizard_ops.data as data_mod

	monkeypatch.setattr(data_mod, "load_normalization_stats", lambda: normalization_stats)

	h5_path, csv_path = tiny_h5_and_csv
	tf = Compose([ToTensorV2()])

	dm = NutritionDataModule(
		h5_path=h5_path,
		dish_csv=csv_path,
		batch_size=2,
		train_transform=tf,
		val_transform=tf,
		normalisation_method="zscore",
		val_split=0.5,
		num_workers=0,
		seed=0,
	)
	dm.setup()

	assert dm.train_dataset is not None
	assert dm.val_dataset is not None
	assert len(dm.train_dataset) == 2
	assert len(dm.val_dataset) == 2

	batch = next(iter(dm.train_dataloader()))
	assert "images" in batch
	assert "targets" in batch
	assert batch["images"].shape == (2, NUM_CAMERAS, NUM_FRAMES, 3, 8, 8)
	assert batch["targets"].shape == (2, len(TARGET_COLUMNS))