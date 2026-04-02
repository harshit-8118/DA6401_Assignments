"""Localization modules
"""

import torch
import torch.nn as nn
from vgg11 import VGG11Encoder

class RegressionHead(nn.Module):
    """Regression head for bounding box prediction."""

    def __init__(self):
        super(RegressionHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(    
            nn.Linear(512 * 7 * 7, 1536, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1536, 4, bias=True),
            nn.ReLU(inplace=True), 
        )

    def forward(self, f5: torch.Tensor) -> torch.Tensor:
        """Forward pass for regression head.
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format.
        """
        x = self.pool(f5)
        x = self.flatten(x)
        return self.fc(x)


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, num_classes: int = 4, dropout_p: float = 0.5, freeze_backbone: bool = True):
        super(VGG11Localizer, self).__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)
        
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.regression_head = RegressionHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format.
        """
        # TODO: Implement forward pass.
        out, features = self.encoder(x, return_features=True)
        f5 = features['f5']
        return self.regression_head(f5)


if __name__ == "__main__":
    model = VGG11Localizer()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Localizer output:", out.shape)   # [2, 4]
    print("Values in [0,1]:", out.min().item(), out.max().item())