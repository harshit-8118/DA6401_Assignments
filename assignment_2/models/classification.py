"""Classification components
"""

import torch
import torch.nn as nn


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        # TODO: Implement forward pass.
        raise NotImplementedError("Implement VGG11Classifier.forward")
