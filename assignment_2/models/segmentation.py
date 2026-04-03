"""Segmentation model
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Standard U-Net decoder block: Conv -> Conv -> Upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 1, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11UNet, self).__init__()
        # Encoder backbone
        self.encoder = VGG11Encoder(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)
        
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder layers
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec5 = DecoderBlock(512 + 512, 512)
        self.dec4 = DecoderBlock(512 + 512, 256)
        self.dec3 = DecoderBlock(256 + 256, 128)
        self.dec2 = DecoderBlock(128 + 128, 64)
        self.dec1 = DecoderBlock(64 + 64, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # TODO: Implement forward pass.
        _, features = self.encoder(x, return_features=True) 

        enc1, enc2, enc3, enc4, enc5 = features['f1'], features['f2'], features['f3'], features['f4'], features['f5']

        center = self.center(enc5)
        print("enc1 shape:", enc1.shape)
        print("enc2 shape:", enc2.shape)
        print("enc3 shape:", enc3.shape)
        print("enc4 shape:", enc4.shape)
        print("enc5 shape:", enc5.shape)
        print("center shape:", center.shape)
        dec5 = self.dec5(torch.cat([center, enc5], dim=1))
        dec4 = self.dec4(torch.cat([dec5, enc4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))
        print("dec5 shape:", dec5.shape)
        print("dec4 shape:", dec4.shape)    
        print("dec3 shape:", dec3.shape)
        print("dec2 shape:", dec2.shape)
        print("dec1 shape:", dec1.shape)
        print("final shape:", self.final_conv(dec1).shape)
        return self.final_conv(dec1)
    

if __name__ == "__main__":
    model = VGG11UNet(num_classes=1, in_channels=3, dropout_p=0.5)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)