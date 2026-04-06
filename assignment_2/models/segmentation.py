"""Segmentation model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg11 import VGG11Encoder

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_convs: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        layers = []
        in_conv = out_ch + skip_ch  # after concat
        layers.append(nn.Conv2d(in_conv, out_ch, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        if num_convs == 2:
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:

                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class VGG11UNet(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)

        # center bottleneck (same spatial size as f5)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder mirrors VGG11 conv counts per block
        self.dec5 = UpBlock(512, 512, 512, num_convs=2)  # center + enc5
        self.dec4 = UpBlock(512, 512, 256, num_convs=2)  # dec5 + enc4
        self.dec3 = UpBlock(256, 256, 128, num_convs=2)  # dec4 + enc3
        self.dec2 = UpBlock(128, 128, 64,  num_convs=1)  # dec3 + enc2
        self.dec1 = UpBlock(64,  64,  64,  num_convs=1)  # dec2 + enc1  
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self.encoder(x, return_features=True)
        f1, f2, f3, f4, f5 = features["f1"], features["f2"], features["f3"], features["f4"], features["f5"]

        center = self.center(f5)
        d5 = self.dec5(center, f5)
        d4 = self.dec4(d5, f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        d1 = self.final_up(d1)
        return self.final_conv(d1)


if __name__ == "__main__":
    model = VGG11UNet(num_classes=1, in_channels=3, dropout_p=0.5)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)