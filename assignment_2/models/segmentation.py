"""Segmentation model
"""

import torch
import torch.nn as nn

from vgg11 import VGG11Encoder

class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 1, in_channels: int = 3, num_filters: int = 32):
        super(VGG11UNet, self).__init__()
        # Encoder backbone
        self.encoder = VGG11Encoder(in_channels=in_channels, num_classes=num_classes)
        
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder layers
        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8
        )
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8
        )
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2
        )
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(num_filters * (2 + 1), num_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_classes, kernel_size=1)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # TODO: Implement forward pass.
        _, features = self.encoder(x, return_features=True) 

        enc1, enc2, enc3, enc4, enc5 = features['f1'], features['f2'], features['f3'], features['f4'], features['f5']

        center = self.center(self.pool(enc5))
        print("enc1 shape:", enc1.shape)
        print("enc2 shape:", enc2.shape)
        print("enc3 shape:", enc3.shape)
        print("enc4 shape:", enc4.shape)
        print("enc5 shape:", enc5.shape)
        print("center shape:", center.shape)
        dec5 = self.dec5(torch.cat([center, enc5], dim=1))
        print("dec5 shape:", dec5.shape)
        dec4 = self.dec4(torch.cat([dec5, enc4], dim=1))
        print("dec4 shape:", dec4.shape)    
        dec3 = self.dec3(torch.cat([dec4, enc3], dim=1))
        print("dec3 shape:", dec3.shape)
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))
        print("dec2 shape:", dec2.shape)
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))
        print("dec1 shape:", dec1.shape)
        final = self.final(dec1)
        print("final shape:", final.shape)
        return final
    

if __name__ == "__main__":
    model = VGG11UNet(num_classes=1, in_channels=3, num_filters=32)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(output.shape)