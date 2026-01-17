from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision.models import resnet18, ResNet18_Weights

class AttentionPool(nn.Module):
    """
    Permutation-invariant pooling over views.
    Input:  x (B, V, D)
    Output: z (B, D)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: (B, V, D)
        logits = self.score(x).squeeze(-1)  # (B, V)

        if mask is not None:
            # mask: (B, V) with True for valid, False for padded
            logits = logits.masked_fill(~mask, float("-inf"))

        w = torch.softmax(logits, dim=1)  # (B, V)
        z = torch.sum(x * w.unsqueeze(-1), dim=1)  # (B, D)
        return z, w


class DishMultiViewRegressor(L.LightningModule):
    def __init__(
        self,
        lr: float = 3e-4,
        view_dropout_p: float = 0.3,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Image encoder (shared across all views) ---
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone  # outputs (B*V, feat_dim)

        # --- View aggregation ---
        self.pool = AttentionPool(dim=feat_dim)

        # --- Regression head (5 outputs) ---
        # Order: calories, mass, fat, carb, protein, num_ingrs(IGNORED)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 5),
        )

        self.lr = lr
        self.view_dropout_p = view_dropout_p

    def _flatten_views(self, images: torch.Tensor):
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
        raise ValueError(f"Unexpected images shape: {images.shape}")

    def _maybe_drop_views(self, images: torch.Tensor):
        """
        Randomly drop some views during training to make inference with fewer
        images robust. Ensures at least 1 view remains per sample.
        images: (B, V, C, H, W)
        """
        if not self.training or self.view_dropout_p <= 0:
            return images, None

        b, v, c, h, w = images.shape
        keep = torch.rand(b, v, device=images.device) > self.view_dropout_p
        # ensure at least one view kept per sample
        for i in range(b):
            if not keep[i].any():
                keep[i, torch.randint(0, v, (1,), device=images.device)] = True

        # Ragged selection per batch item is annoying; simplest is to keep padding
        # and pass a mask into attention pooling.
        return images, keep  # mask is (B, V)

    def forward(self, images: torch.Tensor):
        images = self._flatten_views(images)  # (B, V, C, H, W)
        images, mask = self._maybe_drop_views(images)

        b, v, c, h, w = images.shape
        x = images.view(b * v, c, h, w)
        feats = self.encoder(x)  # (B*V, D)
        feats = feats.view(b, v, -1)  # (B, V, D)

        dish_feat, attn_w = self.pool(feats, mask=mask)  # (B, D), (B, V)
        y = self.head(dish_feat)  # (B, 6)

        return y, attn_w
    
    def predict_with_uncertainty(self, images, n_samples=20):
        """Monte Carlo Dropout for uncertainty estimation.
        
        Enables only the dropout in the head for MC sampling, while keeping
        the model in eval mode to prevent view dropout and maintain BatchNorm behavior.
        
        Args:
            images: Input images (B, V, C, H, W)
            n_samples: Number of forward passes for MC sampling
            
        Returns:
            mean_pred: (B, 5) mean predictions across samples
            std_pred: (B, 5) prediction uncertainty (standard deviation)
        """
        # Keep model in eval mode to prevent view dropout
        self.eval()
        
        # Enable dropout only in the head for MC sampling
        for module in self.head.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred, _ = self(images)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # (n_samples, B, 5)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Restore eval mode for all dropout layers
        self.eval()
        
        return mean_pred, std_pred

    def training_step(self, batch, batch_idx):
        images = batch["images"]
        targets = torch.stack(
            [
                batch["total_calories"],
                batch["total_mass"],
                batch["total_fat"],
                batch["total_carb"],
                batch["total_protein"],
                # batch["num_ingrs"].float(),
            ],
            dim=1,
        )  # (B, 5)

        preds, _ = self(images)

        # Huber is robust for regression
        loss = F.smooth_l1_loss(preds, targets)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        targets = torch.stack(
            [
                batch["total_calories"],
                batch["total_mass"],
                batch["total_fat"],
                batch["total_carb"],
                batch["total_protein"],
                # batch["num_ingrs"].float(),
            ],
            dim=1,
        )
        preds, _ = self(images)
        loss = F.smooth_l1_loss(preds, targets)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)