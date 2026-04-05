import os, argparse, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torch.utils.data import DataLoader
import pathlib
from data.pets_dataset import OxfordIIITPetDataset, get_val_transforms
from models.localization import VGG11Localizer
from data.stratified_split import get_stratified_split

CKPT_LOC   = os.path.join("checkpoints", "localizer.pth")
IMG_SIZE   = 224
NUM_BREEDS = 37
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def make_loaders(args, use_aug=True):
    from data.pets_dataset import (OxfordIIITPetDataset,
                                    get_train_transforms,
                                    get_val_transforms)
    root     = pathlib.Path(args.data_root)
    ann_file = root / "annotations" / "trainval.txt"   
    
    _, val_records = get_stratified_split(
        ann_file, val_frac=0.1, seed=args.seed)
    
 
    val_ds = OxfordIIITPetDataset(
        root        = args.data_root,
        records     = val_records,
        images_dir  = root / "images",
        masks_dir   = root / "annotations" / "trimaps",
        transform   = get_val_transforms(IMG_SIZE),
    )
 
    kw = dict(num_workers=args.num_workers,
              pin_memory=(args.device == "cuda"))
 
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False, **kw)
 
    print(f" Val: {len(val_ds)}")
    return val_loader
 

def unnorm(tensor):
    """FloatTensor [C,H,W] → numpy [H,W,C] in [0,1]."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * STD + MEAN, 0, 1)


def draw_box(ax, cx, cy, w, h, H, W, color, label="", lw=2):
    """Draw a single (cx,cy,w,h) normalised bbox on ax."""
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    bw = w * W
    bh = h * H
    rect = patches.Rectangle((x1, y1), bw, bh,
                               linewidth=lw, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    if label:
        ax.text(x1 + 3, y1 + 12, label,
                color='white', fontsize=6,
                bbox=dict(facecolor=color, alpha=0.7, pad=1, linewidth=0))


def iou_single(pred, gt):
    """IoU between two (cx,cy,w,h) normalised boxes."""
    px1, py1 = pred[0] - pred[2]/2, pred[1] - pred[3]/2
    px2, py2 = pred[0] + pred[2]/2, pred[1] + pred[3]/2
    gx1, gy1 = gt[0] - gt[2]/2,   gt[1] - gt[3]/2
    gx2, gy2 = gt[0] + gt[2]/2,   gt[1] + gt[3]/2
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter + 1e-6
    return inter / union


def run_inference(args):
    device = torch.device(args.device)
    # Dataset & loader 
    loader = make_loaders(args)

    # Model 
    model = VGG11Localizer(freeze_backbone=False).to(device)
    if os.path.exists(CKPT_LOC):
        ckpt = torch.load(CKPT_LOC, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded {CKPT_LOC}")
    else:
        print(f"Warning: {CKPT_LOC} not found — using random weights")
    model.eval()

    # Collect predictions
    all_imgs, all_preds, all_gt_bboxes, all_labels = [], [], [], []

    with torch.no_grad():
        for imgs, labels, bboxes_norm, _ in loader:
            pred_px   = model(imgs.to(device))                       # pixel output
            pred_norm = torch.clamp(pred_px / IMG_SIZE, 0.0, 1.0)   # [0,1]
            all_imgs.append(imgs.cpu())
            all_preds.append(pred_norm.cpu())
            all_gt_bboxes.append(bboxes_norm.cpu())   # normalised gt (fallback for test)
            all_labels.append(labels.cpu())

    all_imgs      = torch.cat(all_imgs)
    all_preds     = torch.cat(all_preds)
    all_gt_bboxes = torch.cat(all_gt_bboxes)
    all_labels    = torch.cat(all_labels)

    print(f"Total test samples: {len(all_imgs)}")

    # Sample n random images
    n = min(args.n, len(all_imgs))
    indices = random.sample(range(len(all_imgs)), n)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).ravel()
    avg_iou = 0
    for plot_i, idx in enumerate(indices):
        ax  = axes[plot_i]
        img = unnorm(all_imgs[idx])
        H, W = img.shape[:2]

        pred = all_preds[idx].numpy()       # (cx,cy,w,h) normalised
        gt   = all_gt_bboxes[idx].numpy()   # (cx,cy,w,h) normalised (fallback for test)
        iou  = iou_single(pred, gt)

        avg_iou += iou

        ax.imshow(img)

        # Predicted box — RED
        draw_box(ax, pred[0], pred[1], pred[2], pred[3], H, W,
                 color='red', label=f"pred IoU={iou:.2f}")

        ax.set_title(f"cls={all_labels[idx].item()}  IoU={iou:.2f}", fontsize=8)
        ax.axis('off')

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis('off')

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color='red',  label='Predicted bbox'),
                 Patch(color='lime', label='GT bbox (full-image fallback for test)')],
        loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.0)
    )
    plt.suptitle(f"Localization inference on test set (n={n})", fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(args.save, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved → {args.save}")

    # Summary stats 
    all_ious = [iou_single(all_preds[i].numpy(), all_gt_bboxes[i].numpy())
                for i in range(len(all_preds))]
    print(f"Mean IoU of samples: {avg_iou / n}")
    print(f"Mean IoU (vs fallback gt): {np.mean(all_ious):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",   default="./data/oxford-iiit-pet/")
    ap.add_argument("--n",           type=int, default=16, help="images to show")
    ap.add_argument("--batch_size",  type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save",        default="bbox_results.png") 
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run_inference(args)