"""
VGG11 encoder
"""

from typing import Dict, Tuple, Union
from models.layers import CustomDropout

import torch
import torch.nn as nn


class VGG11EncoderNoBn(nn.Module):
    """
    VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 37, dropout_p: float = 0.5):
        super(VGG11EncoderNoBn, self).__init__()
        self.enc1 = nn.Sequential(
            # -- conv_1 --
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.enc2 = nn.Sequential(
            # -- conv_2 --
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.enc3 = nn.Sequential(
            # -- conv_3 --
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # -- conv_4 --
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.enc4 = nn.Sequential(
            # -- conv_5 --
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.enc5 = nn.Sequential(
            # -- conv_7 --
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            # -- conv_8 --
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.head = nn.Sequential(
            # avg pool layer
            nn.Flatten(),
            # classifier
            nn.Linear(in_features=7*7*512, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True), 
            CustomDropout(p=dropout_p),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )


    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        f3 = self.enc3(f2)
        f4 = self.enc4(f3)
        f5 = self.enc5(f4)
        
        out = self.head(f5)

        if return_features:
            features = {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}
            return out, features
        
        return out
        # raise NotImplementedError("Implement VGG11EncoderNoBn.forward")


class VGG11ClassifierNoBn(nn.Module):
    """Full classifier = VGG11EncoderNoBn + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11ClassifierNoBn, self).__init__()
        self.classifier = VGG11EncoderNoBn(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        # TODO: Implement forward pass.
        return self.classifier(x)