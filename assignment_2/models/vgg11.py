"""
VGG11 encoder
"""

from typing import Dict, Tuple, Union
from layers import CustomDropout

import torch
import torch.nn as nn


class VGG11Encoder(nn.Module):
    """
    VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 37, dropout_p: float = 0.5):
        super(VGG11Encoder, self).__init__()
        self.enc1 = nn.Sequential(
            # -- conv_1 --
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.enc2 = nn.Sequential(
            # -- conv_2 --
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        
        self.enc3 = nn.Sequential(
            # -- conv_3 --
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # -- conv_4 --
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.enc4 = nn.Sequential(
            # -- conv_5 --
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.enc5 = nn.Sequential(
            # -- conv_7 --
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # -- conv_8 --
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        self.head = nn.Sequential(
            # avg pool layer
            # nn.AdaptiveAvgPool2d(output_size=(7, 7)),
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
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        # TODO: Implement forward pass.

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
        # raise NotImplementedError("Implement VGG11Encoder.forward")
    

if __name__ == '__main__':
    model = VGG11Encoder()
    dummy_input = torch.randn(1, 3, 224, 224)
    print(model)
    # 1. Test Training Mode (Dropout Active)
    model.train()
    out_train = model(dummy_input)
    print(out_train.shape)  # Should be [1, 37] 
    out_train, features = model(dummy_input, return_features=True)
    for f, v in features.items():
        print(f"{f}: {v.shape}")  # Should show the shapes of the feature maps

    print(model.head[3].p)  # Should show dropout probability (0.5)
    print("Train forward successful.")
    
    # 2. Test Eval Mode (Dropout Inactive)
    model.eval()
    out_eval = model(dummy_input)
    print("Eval forward successful.")