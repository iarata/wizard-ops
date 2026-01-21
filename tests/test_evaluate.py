from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import wizard_ops.evaluate as inf


class DummyTransform:
    """Albumentations-like callable: tf(image=HWC_uint8)->{'image': CHW_tensor}."""

    def __call__(self, *, image):
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3 and image.shape[2] == 3
        t = torch.from_numpy(np.transpose(image, (2, 0, 1))).to(torch.float32)
        return {"image": t}


def test_tile_or_trim_views_single_tiles_to_20():
    views = ["a"]
    out = inf._tile_or_trim_views(views)
    assert len(out) == inf.NUM_CAMERAS * inf.NUM_FRAMES
    assert out == ["a"] * 20


def test_tile_or_trim_views_4_cams_repeats_frames():
    views = ["c0", "c1", "c2", "c3"]
    out = inf._tile_or_trim_views(views)

    assert len(out) == 20
    # cam0 f0..f4, cam1 f0..f4, ...
    assert out[:5] == ["c0"] * 5
    assert out[5:10] == ["c1"] * 5
    assert out[10:15] == ["c2"] * 5
    assert out[15:20] == ["c3"] * 5


def test_tile_or_trim_views_5_frames_repeats_cameras():
    views = ["f0", "f1", "f2", "f3", "f4"]
    out = inf._tile_or_trim_views(views)

    assert len(out) == 20
    assert out[:5] == views
    assert out[5:10] == views
    assert out[10:15] == views
    assert out[15:20] == views


def test_as_rgb_numpy_from_path(tmp_path: Path):
    p = tmp_path / "x.jpg"
    Image.new("RGB", (11, 7), color=(1, 2, 3)).save(p)

    arr = inf._as_rgb_numpy(p)
    assert arr.dtype == np.uint8
    assert arr.shape == (7, 11, 3)


def test_build_model_input_tensor_from_single_numpy_image(monkeypatch):
    # Avoid Albumentations dependency in unit tests by patching transforms.
    monkeypatch.setattr(inf, "get_default_transforms", lambda **_: DummyTransform())

    img = np.zeros((12, 9, 3), dtype=np.uint8)
    x = inf._build_model_input_tensor(
        img,
        image_size=224,
        device=torch.device("cpu"),
    )

    assert torch.is_tensor(x)
    assert x.shape == (1, inf.NUM_CAMERAS, inf.NUM_FRAMES, 3, 12, 9)
    assert x.dtype == torch.float32


def test_build_model_input_tensor_from_tensor_passthrough():
    x_in = torch.zeros((inf.NUM_CAMERAS, inf.NUM_FRAMES, 3, 224, 224))
    x = inf._build_model_input_tensor(
        x_in,
        image_size=224,
        device=torch.device("cpu"),
    )
    assert x.shape == (1, inf.NUM_CAMERAS, inf.NUM_FRAMES, 3, 224, 224)


def test_build_model_input_tensor_rejects_bad_tensor_shape():
    bad = torch.zeros((3, 224, 224))
    with pytest.raises(ValueError):
        inf._build_model_input_tensor(
            bad,
            image_size=224,
            device=torch.device("cpu"),
        )


def test_extract_pred_tensor_from_tensor():
    t = torch.randn(2, 5)
    out = inf._extract_pred_tensor(t)
    assert out is t


def test_extract_pred_tensor_from_dict_preds():
    t = torch.randn(1, 5)
    out = inf._extract_pred_tensor({"preds": t})
    assert out is t


def test_predict_nutrition_outputs_5_metrics_and_denormalizes(monkeypatch):
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Ensure the inference pipeline built the expected shape.
            assert x.shape[1:3] == (inf.NUM_CAMERAS, inf.NUM_FRAMES)
            return torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], device=x.device)

    # Patch model loading to avoid real checkpoints.
    monkeypatch.setattr(
        inf,
        "load_model_for_inference",
        lambda checkpoint_path, device=None: (DummyModel(), torch.device("cpu")),
    )

    # Patch stats loader so denormalization is deterministic.
    stats = {
        "targets": {
            "total_calories": {"mean": 10.0, "std": 2.0},
            "total_mass": {"mean": 20.0, "std": 4.0},
            "total_fat": {"mean": 30.0, "std": 6.0},
            "total_carb": {"mean": 40.0, "std": 8.0},
            "total_protein": {"mean": 50.0, "std": 10.0},
        }
    }
    monkeypatch.setattr(inf, "load_normalization_stats", lambda *a, **k: stats)

    images = torch.zeros((inf.NUM_CAMERAS, inf.NUM_FRAMES, 3, 224, 224))

    pred = inf.predict_nutrition(
        checkpoint_path="dummy.pth",
        images=images,
        normalisation_method="zscore",
        image_size=224,
        device="cpu",
    )

    assert set(pred.keys()) == {"normalized", "denormalized"}
    assert list(pred["normalized"].keys()) == inf.TARGET_COLUMNS
    assert list(pred["denormalized"].keys()) == inf.TARGET_COLUMNS

    assert pred["normalized"]["total_calories"] == 1.0
    assert pred["denormalized"]["total_calories"] == 1.0 * 2.0 + 10.0
    assert pred["denormalized"]["total_mass"] == 2.0 * 4.0 + 20.0
    assert pred["denormalized"]["total_fat"] == 3.0 * 6.0 + 30.0
    assert pred["denormalized"]["total_carb"] == 4.0 * 8.0 + 40.0
    assert pred["denormalized"]["total_protein"] == 5.0 * 10.0 + 50.0


def test_predict_nutrition_supports_model_returning_dict(monkeypatch):
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return {"preds": torch.zeros((1, 5), device=x.device)}

    monkeypatch.setattr(
        inf,
        "load_model_for_inference",
        lambda checkpoint_path, device=None: (DummyModel(), torch.device("cpu")),
    )
    monkeypatch.setattr(inf, "load_normalization_stats", lambda *a, **k: {"targets": {}})

    images = torch.zeros((inf.NUM_CAMERAS, inf.NUM_FRAMES, 3, 224, 224))
    pred = inf.predict_nutrition(
        checkpoint_path="dummy.pth",
        images=images,
        normalisation_method="zscore",
        image_size=224,
        device="cpu",
    )

    assert pred["normalized"] == {k: 0.0 for k in inf.TARGET_COLUMNS}
    assert pred["denormalized"] == {k: 0.0 for k in inf.TARGET_COLUMNS}