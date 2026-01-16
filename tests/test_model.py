import pytest
import torch
from wizard_ops.model import NutritionPredictor


def test_forward_shape():
    model = NutritionPredictor(num_outputs=5, freeze_backbone=False, lr=1e-3)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 5)
    assert out.dtype == torch.float32


def test_freeze_backbone_flag():
    m1 = NutritionPredictor(freeze_backbone=True)
    # all backbone parameters should be frozen
    assert all(not p.requires_grad for p in m1.backbone.parameters())

    m2 = NutritionPredictor(freeze_backbone=False)
    # at least one parameter should be trainable when not frozen
    assert any(p.requires_grad for p in m2.backbone.parameters())


def test_configure_optimizers_structure():
    model = NutritionPredictor()
    opt_dict = model.configure_optimizers()
    assert isinstance(opt_dict, dict)
    assert "optimizer" in opt_dict
    assert "lr_scheduler" in opt_dict
    sched_info = opt_dict["lr_scheduler"]
    assert isinstance(sched_info, dict)
    assert "scheduler" in sched_info and "monitor" in sched_info
    assert sched_info["monitor"] == "val_loss"


def test_hparams_saved():
    model = NutritionPredictor(lr=0.007)
    assert hasattr(model, "hparams")
    # hparams may be mapping-like or an object with attributes
    if hasattr(model.hparams, "get"):
        assert float(model.hparams.get("lr", 0.0)) == pytest.approx(0.007)
    else:
        assert getattr(model.hparams, "lr") == pytest.approx(0.007)
