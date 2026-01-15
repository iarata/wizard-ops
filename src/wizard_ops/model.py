import torch
import torchvision.models as models
from torch import nn


class NutritionPredictor(nn.Module):
    def __init__(self, num_outputs=5):
        super().__init__()
        # RGB stream
        self.resnet_rgb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_rgb.fc = nn.Identity()
        
        # Depth stream
        self.resnet_depth = models.resnet18(weights=None)
        self.resnet_depth.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_depth.fc = nn.Identity()

        # Fusion Head
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_outputs)
        )
    
    def forward(self, rgb_image, depth_image):
        feat_rgb = self.resnet_rgb(rgb_image)
        feat_depth = self.resnet_depth(depth_image)
        fused = torch.cat([feat_rgb, feat_depth], dim=1)
        return self.fc(fused)

    
if __name__ == "__main__":
    model = NutritionPredictor()
    x = torch.rand(1, 3, 224, 224)
    depth_x = torch.rand(1, 1, 224, 224)
    print(f"Output shape of model: {model(x, depth_x).shape}")
