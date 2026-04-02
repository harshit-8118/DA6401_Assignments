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
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 1.  AUGMENTATION VISUALIZER
#     Saves a grid image to disk. Optionally logs to wandb.
# ─────────────────────────────────────────────────────────────────────────────
 
def visualize_augmentations(dataset, num_samples=16,
                             save_path="aug_preview.png",
                             use_wandb=False, wandb_key="aug/preview"):
    """Visualize augmented samples from a dataset.
 
    Args:
        dataset    : any torch Dataset returning (img_tensor, label, bbox, mask)
        num_samples: number of images to show (must be even, laid out in 2 rows)
        save_path  : where to save the PNG on disk
        use_wandb  : also log the image to W&B
        wandb_key  : W&B log key
 
    Usage in make_loaders (replace the broken version):
        visualize_augmentations(train_ds, save_path="aug_preview.png",
                                use_wandb=args.use_wandb)
        # Then check aug_preview.png — no plt.show() needed
    """
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
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2.  MIXUP
#     Pure torch — no extra libraries.
#     Mixes two images and their labels. Forces smooth decision boundaries.
#     Highly effective against overfitting on small datasets.
# ─────────────────────────────────────────────────────────────────────────────
 
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Apply Mixup to a batch.
 
    Args:
        x    : [B, C, H, W] image batch
        y    : [B] integer label batch
        alpha: Beta distribution parameter. 0.4 is a good default.
               Higher alpha = more mixing. Set to 0 to disable.
 
    Returns:
        x_mix     : mixed images
        y_a, y_b  : original and shuffled labels
        lam       : mixing coefficient in [0, 1]
 
    Loss usage:
        loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
    """
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

# def cutmix(x, y, alpha=1.0):
#     lam = np.random.beta(alpha, alpha)
#     rand_index = torch.randperm(x.size(0))

#     bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

#     x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
#     y_a, y_b = y, y[rand_index]

#     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

#     return x, y_a, y_b, lam