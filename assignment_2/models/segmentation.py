"""Segmentation model
"""

import torch
import torch.nn as nn

class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # TODO: Implement forward pass.
        raise NotImplementedError("Implement VGG11UNet.forward")
