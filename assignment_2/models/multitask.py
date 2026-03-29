"""Unified multi-task model
"""

import torch
import torch.nn as nn
from vgg11 import VGG11Encoder
from localization import VGG11Localizer
from segmentation import VGG11UNet

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super(MultiTaskPerceptionModel, self).__init__()
        # Shared VGG11 backbone
        self.vgg11 = VGG11Encoder(in_channels=in_channels, num_classes=num_breeds, dropout_p=dropout_p)
        # Task-specific heads
        self.localizer = VGG11Localizer(in_channels=in_channels, num_classes=4, freeze_backbone=True)
        self.vgg11_unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels, num_filters=32)


    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # TODO: Implement forward pass.

        cls_logits = self.vgg11(x)
        bbox = self.localizer(x)
        seg_logits = self.vgg11_unet(x)

        return {
            "classification": cls_logits,
            "localization":   bbox,
            "segmentation":   seg_logits,
        }


if __name__ == "__main__":
    model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3, in_channels=3, dropout_p=0.5)
    x = torch.randn(1, 3, 256, 256)
    outputs = model(x)
    print("Classification output shape:", outputs["classification"].shape)
    print("Localization output shape:", outputs["localization"].shape)
    print("Segmentation output shape:", outputs["segmentation"].shape)