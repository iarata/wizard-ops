import torch
from torch import nn, optim
import torchvision.models as models
from lightning import LightningModule


class NutritionPredictor(LightningModule):
    def __init__(self, num_outputs=5):
        super().__init__()
        # RGB stream
        self.resnet_rgb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_rgb.fc = nn.Identity()
        
        # Fusion Head
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_outputs)
        )
    
        self.criterion = nn.MSELoss()

    def forward(self, rgb_image):
        feat_rgb = self.resnet_rgb(rgb_image)
        return self.fc(feat_rgb)
    
    def training_step(self, batch, batch_idx):
        img = batch['image']
        nutrition = torch.stack([batch['calories'], batch['mass'], batch['fat'], batch['carbs'], batch['protein'] ],axis=1)

        preds = self.forward(img)
        loss = self.criterion(preds,nutrition)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-2)
    
    
        

if __name__ == "__main__":
    model = NutritionPredictor()
    x = torch.rand(1, 3, 224, 224)
    depth_x = torch.rand(1, 1, 224, 224)
    print(f"Output shape of model: {model(x, depth_x).shape}")
