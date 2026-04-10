"""Localization modules
"""

import torch
import torch.nn as nn
from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout

IMG_SIZE = 224
class RegressionHead(nn.Module):
    """Regression head for bounding box prediction."""
    def __init__(self, dropout_p: float = 0.5):
        super(RegressionHead, self).__init__()
        kern = 7
        self.pool = nn.AdaptiveAvgPool2d((kern, kern))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(    
            nn.Linear(512 * kern * kern, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),

            nn.Linear(2048, 1024, bias=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.1),

            nn.Linear(1024, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 4, bias=True),
        )
 
    def forward(self, f5: torch.Tensor) -> torch.Tensor:
        """Forward pass for regression head.
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format.
        """
        return torch.sigmoid(self.fc(self.flatten(self.pool(f5))))


class VGG11Localizer(nn.Module):
    """VGG11-based localizer."""

    def __init__(self, in_channels: int = 3, num_classes: int = 4, dropout_p: float = 0.5, freeze_backbone: bool = True):
        super(VGG11Localizer, self).__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)
        
        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.regression_head = RegressionHead(dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height) format.
        """
        # TODO: Implement forward pass.
        _, features = self.encoder(x, return_features=True)
        f5 = features['f5']
        return self.regression_head(f5)


if __name__ == "__main__":
    model = VGG11Localizer()
    dummy = torch.randn(1, 3, 224, 224)
    out = model(dummy)
    print("Localizer output:", out.shape)   # [2, 4]
    print("Values in [0,1]:", out.min().item(), out.max().item())