"""
viz_and_fixes.py
================
1. visualize_augmentations() — saves grid to file + logs to wandb
2. mixup_data()              — Mixup augmentation (pure torch, no extra libs)
3. Updated train_classifier  — early stopping + mixup + stronger dropout hint
4. TTA (Test-Time Augmentation) for inference boost
"""
 
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # works on servers with no display
import matplotlib.pyplot as plt
 
def visualize_augmentations(dataset, num_samples=16,
                             save_path="aug_preview.png",
                             use_wandb=False, wandb_key="aug/preview"):
    MEAN = np.array([0.485, 0.456, 0.406])
    STD  = np.array([0.229, 0.224, 0.225])
 
    cols = min(8, num_samples)
    rows = max(1, (num_samples + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).ravel()
 
    rng = np.random.default_rng(0)   # fixed seed for reproducible preview
    indices = rng.choice(len(dataset), size=num_samples, replace=False)
 
    for ax, idx in zip(axes, indices):
        img, label, _, _ = dataset[int(idx)]
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(STD * img_np + MEAN, 0, 1)   # un-normalise
        ax.imshow(img_np)
        ax.set_title(f"cls {int(label)}", fontsize=8)
        ax.axis("off")
 
    # hide unused axes if num_samples < rows*cols
    for ax in axes[num_samples:]:
        ax.axis("off")
 
    plt.suptitle("Training augmentations (sample)", fontsize=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] saved augmentation preview → {save_path}")
 
    if use_wandb:
        import wandb
        wandb.log({wandb_key: wandb.Image(save_path)})
        print(f"  [viz] logged to W&B under '{wandb_key}'")
 
 
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    if alpha <= 0:
        return x, y, y, 1.0
 
    lam = float(np.random.beta(alpha, alpha))
    B   = x.size(0)
    idx = torch.randperm(B, device=x.device)
 
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a   = y
    y_b   = y[idx]
    return x_mix, y_a, y_b, lam
 
 
def mixup_criterion(criterion, logits, y_a, y_b, lam):
    """Compute Mixup loss."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
