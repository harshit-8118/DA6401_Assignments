"""Classification components
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        super(VGG11Classifier, self).__init__()
        self.classifier = VGG11Encoder(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p)
        

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass for classification model.
        Returns:
            Classification logits [B, num_classes].
        """
        # TODO: Implement forward pass.
        if return_features:
            return self.classifier(x, return_features)
        return self.classifier(x)

if __name__ == "__main__":
    model = VGG11Classifier(num_classes=37, in_channels=3, dropout_p=0.5)
    dummy_input = torch.randn(1, 3, 224, 224)  # [B, C, H, W]
    output = model(dummy_input)
    print(output.shape)  # Should be [1, 37]