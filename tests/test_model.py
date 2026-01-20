import pytest
import torch
import torch.nn as nn

import wizard_ops.model as model_mod
from wizard_ops.model import DishMultiViewRegressor


def test_forward_shape():
    model = DishMultiViewRegressor(
        # backbone="dinov3_small",
        backbone="resnet18",
        image_size=64,
        freeze_encoder=False,
        lr=1e-3,
        hidden_dim=16,
    )
    model.eval()

    # (B, V, C, H, W)
    x = torch.randn(2, 3, 3, 64, 64)
    preds, attn_w = model(x)

    assert preds.shape == (2, 5)
    assert preds.dtype == torch.float32

    assert attn_w.shape == (2, 3)
    assert attn_w.dtype == torch.float32
    torch.testing.assert_close(attn_w.sum(dim=1), torch.ones(2), atol=1e-5, rtol=0)

def test_freeze_encoder_flag():
    m1 = DishMultiViewRegressor(freeze_encoder=True)
    assert all(not p.requires_grad for p in m1.encoder.parameters())

    m2 = DishMultiViewRegressor(freeze_encoder=False)
    assert any(p.requires_grad for p in m2.encoder.parameters())


def test_configure_optimizers_structure():
    model = DishMultiViewRegressor(lr=0.001, weight_decay=0.01)
    opt = model.configure_optimizers()

    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.001)
    assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)


def test_hparams_saved():
    model = DishMultiViewRegressor(lr=0.007)
    assert hasattr(model, "hparams")
    if hasattr(model.hparams, "get"):
        assert float(model.hparams.get("lr", 0.0)) == pytest.approx(0.007)
    else:
        assert getattr(model.hparams, "lr") == pytest.approx(0.007)
        
if __name__ == "__main__":
    pytest.main([__file__])