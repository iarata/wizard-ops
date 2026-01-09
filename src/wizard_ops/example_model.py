import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
import torch.nn.functional as F
import torchvision.models as models
from pytorch_lightning import LightningModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class BaseResNetModel(LightningModule):
    def __init__(self, num_classes=10):  # Adjust `num_classes` to your dataset
        super(BaseResNetModel, self).__init__()

        # Pre-trained ResNet-101 for RGB
        self.rgb_resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.rgb_resnet.fc = nn.Identity()  # type: ignore # Remove the classification head

        # Pre-trained ResNet-101 for Depth
        self.depth_resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.depth_resnet.fc = nn.Identity()  # type: ignore # Remove the classification head

        # Fully Connected Layer after merging
        self.fc = nn.Sequential(
            nn.Linear(2048 * 2, 512),  # Merge features (2048 from each ResNet)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb_input, depth_input):
        # Forward pass for RGB
        rgb_features = self.rgb_resnet(rgb_input)

        # Forward pass for Depth
        depth_features = self.depth_resnet(depth_input)

        # Concatenate RGB and Depth features
        merged_features = torch.cat((rgb_features, depth_features), dim=1)

        # Pass through FC layer
        output = self.fc(merged_features)
        return output

# Example Usage
if __name__ == "__main__":
    # Model Initialization
    # model = RGBDFusionNetwork()
    model = BaseResNetModel()
    model = model.to(DEVICE)

    # Dummy Data
    rgb_input = torch.rand((4, 3, 224, 224)).to(DEVICE)
    depth_input = torch.rand((4, 3, 224, 224)).to(DEVICE)

    # Forward Pass
    predictions = model(rgb_input, depth_input)
    print(predictions)