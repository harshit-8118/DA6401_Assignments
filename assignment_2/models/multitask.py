"""Unified multi-task model
"""

import torch
import torch.nn as nn

class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3):
        pass

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        # TODO: Implement forward pass.
        raise NotImplementedError("Implement MultiTaskPerceptionModel.forward")
