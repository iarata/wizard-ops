import pytest
import torch
import torch.nn as nn

import wizard_ops.model as model_mod
from wizard_ops.model import DishMultiViewRegressor


class DummyEncoder(nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(8, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x).flatten(1)  # (N, 8)
        return self.proj(x)  # (N, out_dim)


@pytest.fixture(autouse=True)
def _patch_encoder_builder(monkeypatch):
    # Avoid downloading any pretrained weights in unit tests
    def _build_img_encoder(*args, **kwargs):
        return DummyEncoder(out_dim=32)

    monkeypatch.setattr(model_mod, "build_img_encoder", _build_img_encoder)


def test_forward_shape():
    model = DishMultiViewRegressor(
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