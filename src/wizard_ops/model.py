import torch
from torch import nn, optim
import torchvision.models as models
from lightning import LightningModule
from lightning.pytorch import seed_everything

NUTRITION_COLUMNS = ["calories", "mass", "fat", "carbs", "protein"]


class NutritionPredictor(LightningModule):
    def __init__(self, num_outputs: int = 5, lr: float = 1e-2, seed = 42):
        super().__init__()
        seed_everything(seed=seed, workers=True)
        # RGB stream
        self.resnet_rgb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_rgb.fc = nn.Identity()

        # # Fusion Head
        # self.fc = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        # nn.Linear(512, num_outputs)
        # )
        self.fc = nn.Linear(512, num_outputs)
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, rgb_image):
        feat_rgb = self.resnet_rgb(rgb_image)
        return self.fc(feat_rgb)

    def training_step(self, batch, batch_idx):
        img = batch["image"]
        nutrition = torch.stack([batch[column] for column in NUTRITION_COLUMNS], dim=1)

        preds = self.forward(img)
        loss = self.criterion(preds, nutrition)
        self.log("train_loss", loss, prog_bar=True, batch_size=img.shape[0])
        return loss

    def validation_step(self, batch, batch_id):
        img = batch["image"]
        nutrition = torch.stack([ batch[column] for column in NUTRITION_COLUMNS ], dim=1)

        preds = self.forward(img)
        loss = self.criterion(preds, nutrition)
        self.log("val_loss", loss, prog_bar=True, batch_size=img.shape[0])
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = NutritionPredictor()
    x = torch.rand(1, 3, 224, 224)
    print(f"Output shape of model: {model(x).shape}")
