import os, argparse, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from torch.utils.data import DataLoader
import pathlib

from data.pets_dataset import OxfordIIITPetDataset, get_val_transforms
from models.segmentation import VGG11UNet
from data.stratified_split import get_stratified_split

CKPT_SEG    = os.path.join("checkpoints", "unet")
IMG_SIZE    = 224
MEAN        = np.array([0.485, 0.456, 0.406])
STD         = np.array([0.229, 0.224, 0.225])
MASK_COLORS = np.array([[0,200,0],[200,0,0],[0,0,200]], dtype=np.uint8)  # fg,bg,boundary


def unnorm(tensor):
    img = tensor.permute(1,2,0).cpu().numpy()
    return np.clip(img * STD + MEAN, 0, 1)


def overlay(img_float, mask_np, alpha=0.5):
    """Blend colour-coded mask onto float image. Returns float [0,1]."""
    nc = MASK_COLORS.shape[0]
    out = (img_float * 255).astype(np.uint8).copy()
    for c in range(nc):
        m = mask_np == c
        if m.any():
            out[m] = ((1-alpha)*out[m] + alpha*MASK_COLORS[c]).astype(np.uint8)
    return out.astype(np.float32) / 255.0


def nc1_pred_to_display(binary_pred):
    out = np.ones_like(binary_pred)    # default: bg -> colour index 1 (red)
    out[binary_pred == 1] = 0          # fg detected -> colour index 0 (green)
    return out


def load_model(device, nc):
    model = VGG11UNet(num_classes=nc).to(device)
    path  = f"{CKPT_SEG}_{nc}.pth"
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Loaded {path}")
    else:
        print(f"Warning: {path} not found — random weights")
    return model


def predict(model, imgs_batch, device, nc):
    with torch.no_grad():
        logits = model(imgs_batch.to(device))
        if nc == 1:
            return (logits.squeeze(1) > 0).long().cpu()
        return logits.argmax(1).cpu()


def make_val_loader(args):
    root     = pathlib.Path(args.data_root)
    ann_file = root / "annotations" / "trainval.txt"
    _, val_records = get_stratified_split(ann_file, val_frac=0.1, seed=args.seed)
    val_ds = OxfordIIITPetDataset(
        root       = args.data_root,
        records    = val_records,
        images_dir = root / "images",
        masks_dir  = root / "annotations" / "trimaps",
        transform  = get_val_transforms(IMG_SIZE),
    )
    return DataLoader(val_ds, batch_size=args.batch_size,
                      shuffle=False, num_workers=args.num_workers)


def draw_grid(all_imgs, all_gt, all_pred, indices, n_rows, n_cols, nc, save_path):
    """3 columns: Original | GT overlay | Pred overlay."""
    fig, axes = plt.subplots(n_rows, n_cols * 3,
                             figsize=(n_cols * 9, n_rows * 3.2))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    axes = np.array(axes).reshape(n_rows, n_cols * 3)

    for row_i in range(n_rows):
        for col_i in range(n_cols):
            flat_i = row_i * n_cols + col_i
            ax_orig  = axes[row_i, col_i * 3]
            ax_gt    = axes[row_i, col_i * 3 + 1]
            ax_pred  = axes[row_i, col_i * 3 + 2]

            if flat_i >= len(indices):
                ax_orig.axis('off'); ax_gt.axis('off'); ax_pred.axis('off')
                continue

            idx    = indices[flat_i]
            img    = unnorm(all_imgs[idx])
            gt_np  = all_gt[idx].numpy()
            pr_np  = all_pred[idx].numpy()

            gt_display = gt_np
            pr_display = pr_np if nc == 3 else nc1_pred_to_display(pr_np)

            ax_orig.imshow(img);                       ax_orig.axis('off')
            ax_gt.imshow(overlay(img, gt_display));    ax_gt.axis('off')
            ax_pred.imshow(overlay(img, pr_display));  ax_pred.axis('off')

            if row_i == 0:
                ax_orig.set_title("Original",      fontsize=9, fontweight='bold')
                ax_gt.set_title("GT Overlay",      fontsize=9, fontweight='bold')
                ax_pred.set_title("Pred Overlay",  fontsize=9, fontweight='bold')

    legend_handles = [
        Patch(color=np.array([0,200,0])/255,   label='Foreground'),
        Patch(color=np.array([200,0,0])/255,   label='Background'),
        Patch(color=np.array([0,0,200])/255,   label='Boundary'),
    ]
    if nc == 1:
        legend_handles = legend_handles[:2]   # no boundary class in binary mode

    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(legend_handles), fontsize=9)
    plt.suptitle("Segmentation | Original — GT Overlay — Predicted Overlay",
                 fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved → {save_path}")

def run_val_grid(args):
    device = torch.device(args.device)
    nc     = args.seg_classes
    model  = load_model(device, nc)
    model.eval()
    loader = make_val_loader(args)

    all_imgs, all_gt, all_pred = [], [], []
    with torch.no_grad():
        for imgs, _, _, masks in loader:
            all_imgs.append(imgs)
            all_gt.append(masks)
            all_pred.append(predict(model, imgs, device, nc))

    all_imgs = torch.cat(all_imgs)
    all_gt   = torch.cat(all_gt)
    all_pred = torch.cat(all_pred)

    n        = min(args.rows * args.cols, len(all_imgs))
    indices  = random.sample(range(len(all_imgs)), n)
    draw_grid(all_imgs, all_gt, all_pred, indices, args.rows, args.cols, nc, args.save)


def run_single(args):
    if not args.image_path or not os.path.exists(args.image_path):
        print("Error: --image_path not found"); return

    device = torch.device(args.device)
    nc     = args.seg_classes
    model  = load_model(device, nc)
    model.eval()

    pil_img = Image.open(args.image_path).convert('RGB')
    tfm     = get_val_transforms(IMG_SIZE)
    out     = tfm(image=np.array(pil_img),
                  mask=np.zeros((pil_img.height, pil_img.width), dtype=np.uint8),
                  bboxes=[], bbox_labels=[])
    img_t   = out['image']
    img_np  = unnorm(img_t)

    with torch.no_grad():
        logits = model(img_t.unsqueeze(0).to(device))
        pr_np  = ((logits.squeeze(1) > 0).long() if nc == 1
                  else logits.argmax(1).squeeze(0)).cpu().numpy()

    pr_display = pr_np if nc == 3 else nc1_pred_to_display(pr_np)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(img_np);                      axes[0].set_title("Original");       axes[0].axis('off')
    axes[1].imshow(overlay(img_np, pr_display)); axes[1].set_title("Pred Overlay");   axes[1].axis('off')

    legend_handles = [
        Patch(color=np.array([0,200,0])/255,   label='Foreground'),
        Patch(color=np.array([200,0,0])/255,   label='Background'),
        Patch(color=np.array([0,0,200])/255,   label='Boundary'),
    ]
    fig.legend(handles=legend_handles[:2 if nc==1 else 3],
               loc='lower center', ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(args.save, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved → {args.save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",        choices=["val_grid","single"], default="val_grid")
    ap.add_argument("--image_path",  default=None)
    ap.add_argument("--data_root",   default="./data/oxford-iiit-pet/")
    ap.add_argument("--rows",        type=int, default=4)
    ap.add_argument("--cols",        type=int, default=2)
    ap.add_argument("--batch_size",  type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--seg_classes", type=int, default=3, choices=[1,3])
    ap.add_argument("--save",        default="seg_results.png")
    ap.add_argument("--device",      default="cuda:1" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.mode == "single":
        run_single(args)
    else:
        run_val_grid(args)