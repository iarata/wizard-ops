from __future__ import annotations

from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from wizard_ops.utils import build_img_encoder


class AttentionPool(nn.Module):
    """
    Permutation-invariant pooling over views.
    Input:  x (B, V, D)
    Output: z (B, D)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(x).squeeze(-1)  # (B, V)

        if mask is not None:
            # mask: (B, V) with True for valid, False for ignored
            logits = logits.masked_fill(
                ~mask, torch.finfo(logits.dtype).min
            )

        w = torch.softmax(logits, dim=1)  # (B, V)
        z = torch.sum(x * w.unsqueeze(-1), dim=1)  # (B, D)
        return z, w


class DishMultiViewRegressor(L.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet18",
        image_size: int = 224,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        view_dropout_p: float = 0.3,
        hidden_dim: int = 512,
        head_dropout_p: float = 0.1,
        freeze_encoder: bool = False,
        loss: str = "smoothl1",  # "mse" or "smoothl1"
        log_wandb_examples: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Image encoder (shared across all views) ---
        self.encoder = build_img_encoder(backbone=backbone, image_size=image_size)
        feat_dim = int(getattr(self.encoder, "out_dim"))

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # --- View aggregation ---
        self.pool = AttentionPool(dim=feat_dim)

        # --- Regression head (5 outputs) ---
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_dropout_p),
            nn.Linear(hidden_dim, 5),
        )

        loss_l = loss.lower()
        if loss_l == "mse":
            self._loss_fn = nn.MSELoss()
        elif loss_l in {"smoothl1", "huber"}:
            self._loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss: {loss}")

    def _flatten_views(self, images: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          - (B, 4, 5, C, H, W) or
          - (B, V, C, H, W)
        Return:
          - (B, V, C, H, W)
        """
        if images.ndim == 6:
            b, a, f, c, h, w = images.shape
            return images.view(b, a * f, c, h, w)
        if images.ndim == 5:
            return images
        raise ValueError(f"Unexpected images shape: {tuple(images.shape)}")

    def _maybe_drop_views(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Randomly drop some views during training.
        Returns (images, mask) where mask is (B, V) bool.
        """
        if not self.training or float(self.hparams.view_dropout_p) <= 0:
            return images, None

        b, v, _, _, _ = images.shape
        keep = torch.rand(b, v, device=images.device) > float(
            self.hparams.view_dropout_p
        )

        # Ensure at least one view kept per sample
        for i in range(b):
            if not keep[i].any():
                keep[i, torch.randint(0, v, (1,), device=images.device)] = True

        return images, keep

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = self._flatten_views(images)  # (B, V, C, H, W)
        images, mask = self._maybe_drop_views(images)

        b, v, c, h, w = images.shape
        x = images.view(b * v, c, h, w)  # (B*V, C, H, W)

        feats = self.encoder(x)  # expected (B*V, D)
        if feats.ndim != 2:
            raise RuntimeError(
                f"Encoder must return (N, D); got {tuple(feats.shape)}"
            )

        feats = feats.view(b, v, -1)  # (B, V, D)
        dish_feat, attn_w = self.pool(feats, mask=mask)  # (B, D), (B, V)
        y = self.head(dish_feat)  # (B, 5)
        return y, attn_w

    def predict_with_uncertainty(
        self, images: torch.Tensor, n_samples: int = 20
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout (head-only) for uncertainty estimation.

        Keeps the module in eval mode (disables view dropout / BN updates), but
        enables Dropout layers inside the head for stochastic forward passes.
        """
        was_training = self.training
        self.eval()

        for m in self.head.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        preds = []
        with torch.no_grad():
            for _ in range(int(n_samples)):
                p, _ = self(images)
                preds.append(p)

        preds = torch.stack(preds, dim=0)  # (S, B, 5)
        mean_pred = preds.mean(dim=0)
        std_pred = preds.std(dim=0, unbiased=False)

        self.train(was_training)
        return mean_pred, std_pred

    def _log_wandb_examples(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        preds: torch.Tensor,
        stage: str,
        max_items: int = 4,
    ) -> None:
        if not bool(self.hparams.log_wandb_examples):
            return

        b = min(int(max_items), int(images.shape[0]))
        payload = {}

        for i in range(b):
            img = images[i, 0]  # first view, (C, H, W)
            denom = (img.max() - img.min()).clamp_min(1e-6)
            img01 = (img - img.min()) / denom

            gt = targets[i].detach().cpu().float().numpy()
            pr = preds[i].detach().cpu().float().numpy()

            caption = (
                f"GT - Calories: {gt[0]:.1f}, Mass: {gt[1]:.1f}, "
                f"Fat: {gt[2]:.1f}, Carb: {gt[3]:.1f}, Protein: {gt[4]:.1f}\n"
                f"Pred - Calories: {pr[0]:.1f}, Mass: {pr[1]:.1f}, "
                f"Fat: {pr[2]:.1f}, Carb: {pr[3]:.1f}, Protein: {pr[4]:.1f}"
            )
            payload[f"{stage}/example_{i}"] = wandb.Image(
                img01.cpu(), caption=caption
            )

        # Prefer Lightning's logger if present (correct step handling)
        exp = getattr(self.logger, "experiment", None)
        if exp is not None and hasattr(exp, "log"):
            exp.log(payload, step=int(self.global_step))
        else:
            wandb.log(payload)

    def common_step(self, batch, batch_idx: int, stage: str) -> torch.Tensor:
        images = batch["images"]

        targets = torch.stack(
            [
                batch["total_calories"],
                batch["total_mass"],
                batch["total_fat"],
                batch["total_carb"],
                batch["total_protein"],
            ],
            dim=1,
        ).to(dtype=torch.float32)

        preds, _ = self(images)
        loss = self._loss_fn(preds, targets)

        if stage == "val" and batch_idx == 0:
            self._log_wandb_examples(images, targets, preds, stage=stage)

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.common_step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.lr),
            weight_decay=float(self.hparams.weight_decay),
        )