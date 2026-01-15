import torch
from torch import nn, optim
import torchvision.models as models
from lightning import LightningModule
from lightning.pytorch import seed_everything
from torchmetrics import MeanAbsoluteError, MeanSquaredError

NUTRITION_COLUMNS = ["calories", "mass", "fat", "carbs", "protein"]


class NutritionPredictor(LightningModule):
    def __init__(self, num_outputs=5, lr=1e-3, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters()

        # Load ResNet18
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_filters = backbone.fc.in_features

        # Remove the final FC layer to use as a feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Adding an intermediate layer often helps with mapping complex image features to specific units
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)
        )

        # Loss & Metrics
        self.criterion = nn.MSELoss()
        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()


    def forward(self, x):
        features = self.backbone(x)
        return self.regressor(features)
    
    def _shared_step(self, batch):
        # Efficiently stack labels
        x = batch["image"]
        y = torch.stack([batch[col] for col in NUTRITION_COLUMNS], dim=1).float()
        
        preds = self(x)
        # It's better to scale labels during preprocessing
        loss = self.criterion(preds, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.train_mae(preds, y)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.val_mae.update(preds, y)
        
        # Store samples for the end-of-epoch summary
        if batch_idx == 0:
            self.sample_results = {"preds": preds[:3].detach(), "labels": y[:3].detach()}

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_mae", self.val_mae, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if not hasattr(self, 'sample_results'):
            return

        print(f"\n--- Validation Samples (Epoch {self.current_epoch}) ---")
        p, a = self.sample_results["preds"][0], self.sample_results["labels"][0]
        
        print(f"{'Nutrient':<10} | {'Pred':<10} | {'Actual':<10} | {'Diff':<10}")
        print("-" * 45)
        for i, name in enumerate(NUTRITION_COLUMNS):
            diff = p[i] - a[i]
            print(f"{name.capitalize():<10} | {p[i]:>10.1f} | {a[i]:>10.1f} | {diff:>10.1f}")
        print("-" * 45 + "\n") 

    def configure_optimizers(self):
        learning_rate = self.hparams.get("lr", 1e-3)
        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


if __name__ == "__main__":
    model = NutritionPredictor()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape} -> Output: {output.shape}")
    print("Example Output:", output)