# -------------------------------- Wizard Ops --------------------------------- #
# This is a part of the Wizard Ops project (https://github.com/iarata/wizard-ops)
# Models licensed under their respective licenses and this builder generates
# models for use in Wizard Ops.
# ----------------------------------------------------------------------------- #
import lightning as L
import torch
import torch.nn as nn
import torchvision.models as tModels
from transformers import DINOv3ViTModel

# HF ids for ViT DINOv3 (/16)
_DINOV3_HF_IDS = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base":  "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

class _HFDinov3Wrapper(nn.Module):
    """Wrap HF DINOv3

    Args:
        nn (_type_): _description_
    """
    def __init__(self, vit: DINOv3ViTModel):
        super().__init__()
        self.vit = vit
        self.out_dim = int(vit.config.hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # hf models need pixel_values=
        # some versions support interpolant_pos_embeddings, fallback if not
        try: 
            out = self.vit(pixel_values=x, interpolant_pos_embeddings=True)
        except TypeError:
            out = self.vit(pixel_values=x)
        
        pool = getattr(out, "pooler_output", None)
        if pool is not None:
            return pool
        return out.last_hidden_state[:, 0]  # CLS token

def build_img_encoder(
    backbone: str = "resnet18",
    image_size: int = 224,
) -> L.LightningModule:
    match backbone:
        case "resnet18":
            m = tModels.resnet18(weights=tModels.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = int(m.fc.in_features)
            m.fc = nn.Identity()
            
        case "resnet50":
            m = tModels.resnet50(weights=tModels.ResNet50_Weights.IMAGENET1K_V1)
            feat_dim = int(m.fc.in_features)
            m.fc = nn.Identity()
            
        case "efficientnet_b0":
            m = tModels.efficientnet_b0(weights=tModels.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feat_dim = int(m.classifier[-1].in_features)
            m.classifier = nn.Identity()
            
        case "efficientnet_b3":
            m = tModels.efficientnet_b3(weights=tModels.EfficientNet_B3_Weights.IMAGENET1K_V1)
            feat_dim = int(m.classifier[-1].in_features)
            m.classifier = nn.Identity()
            
        case "convnext_tiny":
            m = tModels.convnext_tiny(weights=tModels.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            feat_dim = int(m.classifier[-1].in_features)
            m.classifier = nn.Sequential(m.classifier[0], m.classifier[1])  # classifier = layernorm + flatten + linear
            
        case "convnext_small":
            m = tModels.convnext_small(weights=tModels.ConvNeXt_Small_Weights.IMAGENET1K_V1)
            feat_dim = int(m.classifier[-1].in_features)
            m.classifier = nn.Sequential(m.classifier[0], m.classifier[1])  # classifier = layernorm + flatten + linear

        case "dinov3_small":
            if image_size % 16 != 0:
                raise ValueError(f"DINOv3 small only supports image sizes multiple of 16. Got {image_size}")
            vit = DINOv3ViTModel.from_pretrained(_DINOV3_HF_IDS["small"])
            m = _HFDinov3Wrapper(vit)
            feat_dim = int(m.out_dim)
        
        case "dinov3_large":
            if image_size % 16 != 0:
                raise ValueError(f"DINOv3 large only supports image sizes multiple of 16. Got {image_size}")
            vit = DINOv3ViTModel.from_pretrained(_DINOV3_HF_IDS["large"])
            m = _HFDinov3Wrapper(vit)
            feat_dim = int(m.out_dim)
            
        case "dinov3_base":
            if image_size % 16 != 0:
                raise ValueError(f"DINOv3 base only supports image sizes multiple of 16. Got {image_size}")
            vit = DINOv3ViTModel.from_pretrained(_DINOV3_HF_IDS["base"])
            m = _HFDinov3Wrapper(vit)
            feat_dim = int(m.out_dim)
            
        case _:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    m.out_dim = int(feat_dim)
    m.image_size = image_size
    return m
        
        