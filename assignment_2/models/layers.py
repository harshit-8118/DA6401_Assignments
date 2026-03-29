"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer.
    """

    def __init__(self, p: float = 0.5):
        super(CustomDropout, self).__init__()
        if p < 0 or p >= 1: 
            raise ValueError(f"Value should be in between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement dropout.
        if not self.training or self.p == 0: 
            return x 
        
        mask = (torch.rand_like(x) > self.p).float()
        return (x * mask) / (1 - self.p)
        # raise NotImplementedError("Implement CustomDropout.forward")

if __name__ == "__main__":
    dropout = CustomDropout(p=0.5)
    sample_input = torch.ones(5, 10)
    
    print("--- Training Mode ---")
    dropout.train()
    train_out = dropout(sample_input)
    print(train_out) 
    
    print("\n--- Evaluation Mode ---")
    dropout.eval()
    eval_out = dropout(sample_input)
    print(eval_out)
    # Notice the values are 1.0 (identity mapping)