from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

# IMPORT_PATH = os.getenv("TRAIN_IMPORT_PATH", "train")
from wizard_ops import train as train_mod

IMPORT_PATH = "wizard_ops.train"


def _base_config(tmp_path: Path, logger_type: str = "tensorboard") -> dict:
    return {
        "seed": 42,
        "logging": {"type": logger_type},
        "data": {
            "image_size": 224,
            "h5_path": str(tmp_path / "dummy.h5"),
            "dish_csv_path": str(tmp_path / "dummy.csv"),
            "batch_size": 2,
            "normalisation_method": "zscore",
            "val_split": 0.2,
            "num_workers": 0,
            "prefetch_factor": 2,
        },
        "model": {
            "backbone": "resnet18",
            "freeze_encoder": False,
            "hidden_dim": 64,
            "view_dropout_p": 0.0,
            "head_dropout_p": 0.0,
        },
        "train": {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "precision": 32,
            "fast_dev_run": False,
            "checkpoint_dir": str(tmp_path / "ckpts"),
            "lr": 3e-4,
            "loss": "smoothl1",
            "weight_decay": 1e-2,
            "log_wandb_examples": False,
            "log_every_n_steps": 1,
        },
    }


@pytest.fixture
def train_mod(monkeypatch):
    """
    Import the train module with minimal stubs for optional deps that often
    break imports in CI (wandb, typer).
    """
    # Provide a stub wandb module if it's not installed.
    if "wandb" not in sys.modules:
        sys.modules["wandb"] = types.SimpleNamespace()

    # Provide a stub typer module if train.py references typer.BadParameter
    # without importing it (as in your snippet).
    if "typer" not in sys.modules:

        class BadParameter(Exception):
            pass

        sys.modules["typer"] = types.SimpleNamespace(BadParameter=BadParameter)

    mod = importlib.import_module(IMPORT_PATH)
    # Ensure we test the current code (useful during iterative dev)
    mod = importlib.reload(mod)
    return mod


@dataclass
class Call:
    args: tuple
    kwargs: dict


class DummyDataModule:
    def __init__(self, *args, **kwargs):
        self.call = Call(args=args, kwargs=kwargs)


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.call = Call(args=args, kwargs=kwargs)


class DummyTrainer:
    def __init__(self, *args, **kwargs):
        self.call = Call(args=args, kwargs=kwargs)
        self.fit_calls: list[tuple] = []

    def fit(self, model, datamodule=None):
        self.fit_calls.append((model, datamodule))


class DummyTBLogger:
    def __init__(self, *args, **kwargs):
        self.call = Call(args=args, kwargs=kwargs)


class DummyWandbLogger:
    def __init__(self, *args, **kwargs):
        self.call = Call(args=args, kwargs=kwargs)


class DummyCallback:
    def __init__(self, *args, **kwargs):
        self.call = Call(args=args, kwargs=kwargs)


def test_train_uses_tensorboard_logger_and_calls_fit_and_save(
    train_mod, monkeypatch, tmp_path
):
    cfg = _base_config(tmp_path, logger_type="tensorboard")

    # Patch dependencies used inside train()
    monkeypatch.setattr(train_mod, "NutritionDataModule", DummyDataModule)
    monkeypatch.setattr(train_mod, "DishMultiViewRegressor", DummyModel)
    monkeypatch.setattr(train_mod, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_mod, "TensorBoardLogger", DummyTBLogger)
    monkeypatch.setattr(train_mod, "WandbLogger", DummyWandbLogger)

    monkeypatch.setattr(train_mod, "RichProgressBar", DummyCallback)
    monkeypatch.setattr(train_mod, "ModelCheckpoint", DummyCallback)
    monkeypatch.setattr(train_mod, "LearningRateMonitor", DummyCallback)

    # Make run_name deterministic by patching train_mod.datetime.now().strftime(...)
    class _FixedDT:
        @staticmethod
        def now():
            class _Now:
                @staticmethod
                def strftime(_fmt: str) -> str:
                    return "0101_0000"

            return _Now()

    monkeypatch.setattr(train_mod, "datetime", _FixedDT)

    # Capture torch.save
    saved = {}

    def _fake_save(obj, path):
        saved["obj"] = obj
        saved["path"] = path

    monkeypatch.setattr(train_mod.torch, "save", _fake_save)

    # Also avoid requiring albumentations defaults in this test by passing
    # explicit transforms through.
    train_tf = object()
    val_tf = object()

    train_mod.train(cfg, train_transform=train_tf, val_transform=val_tf)

    # Trainer was constructed and fit was called exactly once.
    # (We don't have direct access to the instance, but we can infer behavior
    # from torch.save being called with the DummyModel instance.)
    assert isinstance(saved.get("obj"), DummyModel)
    assert str(saved.get("path", "")).endswith("/final_model.pth")

    # Ensure the save path includes run_name + checkpoint_dir
    assert cfg["train"]["checkpoint_dir"] in str(saved["path"])
    assert "nutrition_resnet18_0101_0000" in str(saved["path"])


def test_train_uses_wandb_logger(train_mod, monkeypatch, tmp_path):
    cfg = _base_config(tmp_path, logger_type="wandb")

    monkeypatch.setattr(train_mod, "NutritionDataModule", DummyDataModule)
    monkeypatch.setattr(train_mod, "DishMultiViewRegressor", DummyModel)
    monkeypatch.setattr(train_mod, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_mod, "TensorBoardLogger", DummyTBLogger)
    monkeypatch.setattr(train_mod, "WandbLogger", DummyWandbLogger)

    monkeypatch.setattr(train_mod, "RichProgressBar", DummyCallback)
    monkeypatch.setattr(train_mod, "ModelCheckpoint", DummyCallback)
    monkeypatch.setattr(train_mod, "LearningRateMonitor", DummyCallback)

    # Deterministic name again
    class _FixedDT:
        @staticmethod
        def now():
            class _Now:
                @staticmethod
                def strftime(_fmt: str) -> str:
                    return "0101_0000"

            return _Now()

    monkeypatch.setattr(train_mod, "datetime", _FixedDT)

    # Capture the created logger by wrapping WandbLogger
    created = {}

    class _CapturingWandbLogger(DummyWandbLogger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created["kwargs"] = kwargs

    monkeypatch.setattr(train_mod, "WandbLogger", _CapturingWandbLogger)

    # avoid actual save
    monkeypatch.setattr(train_mod.torch, "save", lambda *_: None)

    train_mod.train(cfg, train_transform=object(), val_transform=object())

    assert created["kwargs"]["project"] == "nutrition-predictor"
    assert created["kwargs"]["name"].startswith("nutrition_resnet18_")
    assert created["kwargs"]["log_model"] == "all"


def test_train_raises_on_invalid_logger_type(train_mod, monkeypatch, tmp_path):
    cfg = _base_config(tmp_path, logger_type="nope")

    monkeypatch.setattr(train_mod, "NutritionDataModule", DummyDataModule)
    monkeypatch.setattr(train_mod, "DishMultiViewRegressor", DummyModel)
    monkeypatch.setattr(train_mod, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_mod, "TensorBoardLogger", DummyTBLogger)
    monkeypatch.setattr(train_mod, "WandbLogger", DummyWandbLogger)

    # avoid actual save
    monkeypatch.setattr(train_mod.torch, "save", lambda *_: None)

    with pytest.raises(ValueError) as e:
        train_mod.train(cfg, train_transform=object(), val_transform=object())

    assert "logger" in str(e.value).lower() or "invalid" in str(e.value).lower()


def test_train_constructs_datamodule_with_expected_config(train_mod, monkeypatch, tmp_path):
    cfg = _base_config(tmp_path, logger_type="tensorboard")

    created = {}

    class _CapturingDM(DummyDataModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created["kwargs"] = kwargs

    monkeypatch.setattr(train_mod, "NutritionDataModule", _CapturingDM)
    monkeypatch.setattr(train_mod, "DishMultiViewRegressor", DummyModel)
    monkeypatch.setattr(train_mod, "Trainer", DummyTrainer)
    monkeypatch.setattr(train_mod, "TensorBoardLogger", DummyTBLogger)

    monkeypatch.setattr(train_mod, "RichProgressBar", DummyCallback)
    monkeypatch.setattr(train_mod, "ModelCheckpoint", DummyCallback)
    monkeypatch.setattr(train_mod, "LearningRateMonitor", DummyCallback)

    monkeypatch.setattr(train_mod.torch, "save", lambda *_: None)

    train_tf = object()
    val_tf = object()
    train_mod.train(cfg, train_transform=train_tf, val_transform=val_tf)

    dm_kwargs = created["kwargs"]
    assert dm_kwargs["h5_path"] == cfg["data"]["h5_path"]
    assert dm_kwargs["dish_csv"] == cfg["data"]["dish_csv_path"]
    assert dm_kwargs["batch_size"] == cfg["data"]["batch_size"]
    assert dm_kwargs["normalisation_method"] == cfg["data"]["normalisation_method"]
    assert dm_kwargs["val_split"] == cfg["data"]["val_split"]
    assert dm_kwargs["num_workers"] == cfg["data"]["num_workers"]
    assert dm_kwargs["seed"] == cfg["seed"]
    assert dm_kwargs["prefetch_factor"] == cfg["data"]["prefetch_factor"]
    assert dm_kwargs["train_transform"] is train_tf
    assert dm_kwargs["val_transform"] is val_tf