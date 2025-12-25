#! /usr/bin/env python3
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
    EfficientNet_B4_Weights,
    EfficientNet_B5_Weights,
    EfficientNet_B6_Weights,
    EfficientNet_B7_Weights,
)

BatchNorm2d = nn.BatchNorm2d

__all__ = [
    "get_efficientnet_b0",
    "get_efficientnet_b1",
    "get_efficientnet_b2",
    "get_efficientnet_b3",
    "get_efficientnet_b4",
    "get_efficientnet_b5",
    "get_efficientnet_b6",
    "get_efficientnet_b7",
]

 #strides over channels:
 # [stem, stage1, stage2, stage3, stage4, stage5, stage6, stage7, head]
 # strides per channels:
 # [2, 1, 2, 2, 2, 1, 2, 1, 1]
EFFICIENTNET_CHANNELS = {
    "efficientnet_b0": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
    "efficientnet_b1": [32, 16, 24, 40, 80, 112, 192, 320, 1280],
    "efficientnet_b2": [32, 16, 24, 48, 88, 120, 208, 352, 1408],
    "efficientnet_b3": [32, 24, 32, 48, 96, 136, 232, 384, 1536],
    "efficientnet_b4": [32, 24, 32, 56, 112, 160, 272, 448, 1792],
    "efficientnet_b5": [32, 24, 40, 64, 128, 176, 304, 512, 2048],
    "efficientnet_b6": [32, 32, 40, 72, 144, 200, 344, 576, 2304],
    "efficientnet_b7": [32, 32, 48, 80, 160, 224, 384, 640, 2560],
}


# ------------------------------------------------------------
# EfficientNet backbone (features-only, torchvision)
# ------------------------------------------------------------

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name, pretrained=True, out_indices=[7]):
        super().__init__()

        self.model_name = model_name
        self.out_indices = out_indices

        model = models.__dict__[model_name](weights="IMAGENET1K_V1" if pretrained else None)
        self.features = model.features

        stage_channels = EFFICIENTNET_CHANNELS[model_name]

        # ---- metadata ----
        self.out_channels = [stage_channels[i] for i in out_indices]
        self.out_channels_last = self.out_channels[-1]

        #remove the last stages which are not utilized
        max_index = max(out_indices)
        self.features = self.features[: max_index + 1]


    def forward(self, x):
        feats = []

        for i, block in enumerate(self.features):
            x = block(x)
            if i in self.out_indices:
                feats.append(x)

        return feats  # list of feature maps


# ------------------------------------------------------------
# Factory functions (API-compatible)
# ------------------------------------------------------------
def get_efficientnet_b0(pretrained=True, out_indices=[7]):
    return EfficientNetBackbone("efficientnet_b0", pretrained, out_indices=out_indices)

def get_efficientnet_b1(pretrained=True, out_indices=[7]):
    return EfficientNetBackbone("efficientnet_b1", pretrained, out_indices=out_indices)

def get_efficientnet_b2(pretrained=True, out_indices=[7]):
    return EfficientNetBackbone("efficientnet_b2", pretrained, out_indices=out_indices)

def get_efficientnet_b3(pretrained=True, out_indices=[7]):
    return EfficientNetBackbone("efficientnet_b3", pretrained, out_indices=out_indices)

def get_efficientnet_b4(pretrained=True, out_indices=[7]):
    return EfficientNetBackbone("efficientnet_b4", pretrained, out_indices=out_indices)

def get_efficientnet_b5(pretrained=True, out_indices=[7]):
    return EfficientNetBackbone("efficientnet_b5", pretrained, out_indices=out_indices)

def get_efficientnet_b6(pretrained=True, out_indices=[7]):
    return EfficientNetBackbone("efficientnet_b6", pretrained, out_indices=out_indices)

def get_efficientnet_b7(pretrained=True, out_indices=[7]):
    return EfficientNetBackbone("efficientnet_b7", pretrained, out_indices=out_indices)

# ------------------------------------------------------------
# Debug / sanity check
# ------------------------------------------------------------
if __name__ == "__main__":
    net = get_efficientnet_b0(pretrained=True).cuda()
    x = torch.randn(1, 3, 512, 1024).cuda()
    feats = net(x)

    for i, f in enumerate(feats):
        print(f"feat[{i}] shape = {f.shape}")
