"""
Usage:
    python wandb_report_tasks.py --task 2.1 --data_root ./data/oxford-iiit-pet
    python wandb_report_tasks.py --task 2.2 --data_root ./data/oxford-iiit-pet
    python wandb_report_tasks.py --task 2.3 --data_root ./data/oxford-iiit-pet
    python wandb_report_tasks.py --task 2.4 --data_root ./data/oxford-iiit-pet --image_path ./sample.jpg
    python wandb_report_tasks.py --task 2.5 --data_root ./data/oxford-iiit-pet
    python wandb_report_tasks.py --task 2.6 --data_root ./data/oxford-iiit-pet
    python wandb_report_tasks.py --task 2.7 --image_path ./wild1.jpg ./wild2.jpg ./wild3.jpg
    python wandb_report_tasks.py --task 2.8

REQUIRED FILES BEFORE RUNNING:
    checkpoints/classifier.pth   — trained VGG11Classifier
    checkpoints/localizer.pth    — trained VGG11Localizer
    checkpoints/unet_3.pth       — trained VGG11UNet (nc=3)

REQUIRED CODE CHANGES (in your models/ folder):
    1. models/vgg11.py     — compact head (enc1..enc5 + small FC head)
    2. models/localization.py — RegressionHead with NO Sigmoid, bias=112
    3. losses/iou_loss.py  — IoULoss with reduction={"mean","sum","none"}

"""

import argparse
import os
import sys
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
from PIL import Image
from torch.utils.data import DataLoader, Subset
import wandb

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from models.classification import VGG11Classifier
from models.vgg11          import VGG11Encoder
from models.localization   import VGG11Localizer
from models.segmentation   import VGG11UNet
from losses.iou_loss       import IoULoss
from data.pets_dataset     import OxfordIIITPetDataset, get_train_transforms, get_val_transforms
from data.stratified_split import get_stratified_split

from wandb_utils import (
    make_val_loader, 
    make_train_val_loaders,
    quick_train_clf, 
    epoch_to_reach, 
    log_loss_curve, 
    log_combined_curves,
    lr_sweep_probe, 
    accuracy_loss_comparision_plots
)

# ── constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
NUM_BREEDS = 37
MEAN       = np.array([0.485, 0.456, 0.406])
STD        = np.array([0.229, 0.224, 0.225])
CKPT_CLF   = os.path.join("checkpoints", "classifier.pth")
CKPT_LOC   = os.path.join("checkpoints", "localizer.pth")
CKPT_SEG   = os.path.join("checkpoints", "unet_3.pth")

MASK_COLORS = np.array([[0,200,0],[200,0,0],[0,0,200]], dtype=np.uint8)


# ── helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ckpt(path, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    return ckpt.get("state_dict", ckpt)


def unnorm(tensor):
    img = tensor.permute(1,2,0).cpu().numpy()
    return np.clip(img * STD + MEAN, 0, 1)


def mask_to_rgb(mask_np):
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for c, color in enumerate(MASK_COLORS):
        rgb[mask_np == c] = color
    return rgb


def overlay_mask(img_float, mask_np, alpha=0.5):
    out = (img_float * 255).astype(np.uint8).copy()
    for c, color in enumerate(MASK_COLORS):
        m = mask_np == c
        if m.any():
            out[m] = ((1-alpha)*out[m] + alpha*color).astype(np.uint8)
    return out.astype(np.float32) / 255.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2.1 — Regularization Effect of Dropout (BatchNorm activation distributions)
# ═══════════════════════════════════════════════════════════════════════════════

def task_2_1(args):
    device = get_device()
    set_seed(42)
    run = wandb.init(project=args.wandb_project, name="2.1-batchnorm-activations_exp2",
                     tags=["report", "2.1"])

    train_loader, val_loader = make_train_val_loaders(args, batch_size=64)


    from vgg11_nobn import VGG11ClassifierNoBn
    model_bn  = VGG11Classifier(num_classes=NUM_BREEDS, dropout_p=0.5).to(device)
    model_nobn = VGG11ClassifierNoBn(num_classes=NUM_BREEDS, dropout_p=0.5).to(device)

    # ── Hook to capture 3rd conv layer activations ─────────────────────────
    def get_hook(storage):
        def hook(module, inp, out): storage.append(out.detach().cpu())
        return hook

    acts_bn, acts_nobn = [], []
    # 3rd conv = enc3[0] (first Conv2d in enc3 block)
    h1 = model_bn.classifier.enc3[0].register_forward_hook(get_hook(acts_bn))
    h2 = model_nobn.classifier.enc3[0].register_forward_hook(get_hook(acts_nobn))

    # Run 1 forward pass on a fixed batch
    imgs_fixed = next(iter(val_loader))[0][:16].to(device)
    model_bn.eval(); model_nobn.eval()
    with torch.no_grad():
        model_bn(imgs_fixed); model_nobn(imgs_fixed)
    h1.remove(); h2.remove()

    # ── Plot distributions ─────────────────────────────────────────────────
    bn_vals  = acts_bn[0].numpy().ravel()
    nbn_vals = acts_nobn[0].numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(nbn_vals, bins=100, color='red',  alpha=0.7, density=True)
    axes[0].set_title("3rd Conv Activations — WITHOUT BatchNorm")
    axes[0].set_xlabel("Activation value"); axes[0].set_ylabel("Density")
    axes[0].axvline(nbn_vals.mean(), color='k', linestyle='--', label=f"mean={nbn_vals.mean():.2f}")
    axes[0].legend()

    axes[1].hist(bn_vals,  bins=100, color='blue', alpha=0.7, density=True)
    axes[1].set_title("3rd Conv Activations — WITH BatchNorm")
    axes[1].set_xlabel("Activation value")
    axes[1].axvline(bn_vals.mean(), color='k', linestyle='--', label=f"mean={bn_vals.mean():.2f}")
    axes[1].legend()

    plt.tight_layout()
    img_path = "2.1_activation_distributions.png"
    plt.savefig(img_path, dpi=120); plt.close()

    run.log({
        "2.1/activation_distributions": wandb.Image(img_path),
        "2.1/nobn_mean":  float(nbn_vals.mean()),
        "2.1/nobn_std":   float(nbn_vals.std()),
        "2.1/bn_mean":    float(bn_vals.mean()),
        "2.1/bn_std":     float(bn_vals.std()),
    })

    # --- LR sweep BEFORE training ---
    import copy

    lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

    for lr in lrs:
        out_bn = lr_sweep_probe(copy.deepcopy(model_bn), train_loader, device, lr)
        out_nb = lr_sweep_probe(copy.deepcopy(model_nobn), train_loader, device, lr)

        run.log({
            "lr_sweep/lr": lr,
            "lr_sweep/bn_stable": int(out_bn["stable"]),
            "lr_sweep/bn_start_loss": out_bn.get("start", None),
            "lr_sweep/bn_end_loss": out_bn.get("end", None),
            "lr_sweep/bn_max_loss": out_bn.get("max", None),

            "lr_sweep/nobn_stable": int(out_nb["stable"]),
            "lr_sweep/nobn_start_loss": out_nb.get("start", None),
            "lr_sweep/nobn_end_loss": out_nb.get("end", None),
            "lr_sweep/nobn_max_loss": out_nb.get("max", None),
        })


    # ── Train both for 20 epochs and compare convergence ───────────────────
    print("Training WITH BN for 20 epochs...")
    t_bn, v_bn, _, _ = quick_train_clf(model_bn, train_loader, val_loader, device, 20, "bn", run)
    print("Training WITHOUT BN for 20 epochs...")
    t_nbn, v_nbn, _, _ = quick_train_clf(model_nobn, train_loader, val_loader, device, 20, "nobn", run)

    log_loss_curve("BatchNorm", t_bn, v_bn, run)
    log_loss_curve("NoBatchNorm", t_nbn, v_nbn, run)

    log_combined_curves(t_bn, v_bn, t_nbn, v_nbn, run)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(v_bn,  'b-o', label="Val loss WITH BN")
    ax.plot(v_nbn, 'r-o', label="Val loss WITHOUT BN")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Convergence: BN vs No-BN"); ax.legend(); ax.grid(True)
    conv_path = "2.1_convergence.png"
    plt.savefig(conv_path, dpi=120); plt.close()
    run.log({"2.1/convergence_comparison": wandb.Image(conv_path)})

    print(f"  BN final val loss: {v_bn[-1]:.4f}")
    print(f"  No-BN final val loss: {v_nbn[-1]:.4f}")
    run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2 — Internal Dynamics: Dropout p=0, 0.2, 0.5
# ═══════════════════════════════════════════════════════════════════════════════
def task_2_2(args):
    """
    Train classifier under 3 dropout conditions for N epochs.
    Overlay train vs val loss curves in W&B.
    One wandb run per condition, plus a summary run with all overlaid.
    """
    device = get_device()
    set_seed(42)
    train_loader, val_loader = make_train_val_loaders(args, batch_size=32)
    epochs = args.epochs_2_2

    conditions = [
        ("no_dropout",   0.0),
        ("dropout_0.2",  0.2),
        ("dropout_0.5",  0.5),
    ]

    all_results = {}
    for name, dp in conditions:
        print(f"\n=== Training: {name} ===")
        run = wandb.init(project=args.wandb_project, name=f"2.2-{name}",
                         tags=["report","2.2"], config={"dropout_p": dp})
        model = VGG11Classifier(num_classes=NUM_BREEDS, dropout_p=dp).to(device)
        t_losses, v_losses, t_accs, v_accs = quick_train_clf(
            model, train_loader, val_loader, device, epochs, name, run, task='2.2')
        loss_gaps = t_losses - v_losses
        acc_gaps = t_accs - v_accs
        all_results[name] = {
            "train_loss": t_losses,
            "val_loss": v_losses,
            "train_acc": t_accs,
            "val_acc": v_accs,
            "loss_gap": loss_gaps,
            "acc_gap": acc_gaps,
        }
        run.finish()

    # Summary plot showing all 3 overlaid
    accuracy_loss_comparision_plots(args, all_results, epochs)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.3 — Transfer Learning Showdown (3 strategies on segmentation)
# ═══════════════════════════════════════════════════════════════════════════════
import time
def task_2_3(args):
    device = get_device()
    set_seed(42)
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader = make_train_val_loaders(args, batch_size=16)
    epochs = args.epochs_2_3
    nc = 3

    class_weights = torch.tensor([1.0, 0.7, 4.5], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    def dice_score(pred, tgt, eps=1e-6):
        scores = []
        for c in range(nc):
            p = (pred == c).float()
            t = (tgt == c).float()
            scores.append(((2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)).item())
        return float(np.mean(scores))

    def px_accuracy(pred, tgt):
        return float((pred == tgt).float().mean().item())

    def run_strategy(name, freeze_fn, tag, lr):
        run = wandb.init(
            project=args.wandb_project,
            name=f"2.3-{name}",
            tags=["report", "2.3"],
            config={"strategy": name, "lr": lr, "epochs": epochs},
            reinit=True
        )

        model = VGG11UNet(num_classes=nc).to(device)
        # Load classifier backbone
        if os.path.exists(CKPT_CLF):
            sd = load_ckpt(CKPT_CLF, device)
            enc_sd = {k[len("classifier."):]: v for k, v in sd.items()
                    if k.startswith("classifier.enc")}
            model.encoder.load_state_dict(enc_sd, strict=False)
            print(f"  [{name}] loaded backbone from classifier.pth")

        freeze_fn(model)

        trainable = [p for p in model.parameters() if p.requires_grad]
        run.log({"2.3/trainable_params": sum(p.numel() for p in trainable)})

        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        for epoch in range(1, epochs + 1):
            start_t = time.perf_counter()

            # ---- Train ----
            model.train()
            t_loss = 0.0
            t_dice = 0.0
            t_acc = 0.0

            for imgs, _, _, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        logits = model(imgs)
                        loss = criterion(logits, masks)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                t_loss += loss.item()
                preds = logits.argmax(1)
                t_dice += dice_score(preds.detach().cpu(), masks.detach().cpu())
                t_acc  += px_accuracy(preds.detach().cpu(), masks.detach().cpu())

            t_loss /= len(train_loader)
            t_dice /= len(train_loader)
            t_acc  /= len(train_loader)

            # ---- Val ----
            model.eval()
            v_loss = 0.0
            v_dice = 0.0
            v_acc  = 0.0

            with torch.no_grad():
                for imgs, _, _, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            logits = model(imgs)
                            loss = criterion(logits, masks)
                    else:
                        logits = model(imgs)
                        loss = criterion(logits, masks)

                    v_loss += loss.item()
                    preds = logits.argmax(1)
                    v_dice += dice_score(preds.cpu(), masks.cpu())
                    v_acc  += px_accuracy(preds.cpu(), masks.cpu())

            v_loss /= len(val_loader)
            v_dice /= len(val_loader)
            v_acc  /= len(val_loader)
            sched.step()

            epoch_time = time.perf_counter() - start_t

            print(f"  [{name}] ep {epoch}/{epochs} "
                  f"train_loss={t_loss:.4f} val_loss={v_loss:.4f} "
                  f"val_dice={v_dice:.4f} val_acc={v_acc:.4f}")

            run.log({
                "epoch": epoch,
                "time/epoch_sec": epoch_time,
                "lr": sched.get_last_lr()[0],

                f"2.3/{tag}/train_loss": t_loss,
                f"2.3/{tag}/train_dice": t_dice,
                f"2.3/{tag}/train_acc":  t_acc,

                f"2.3/{tag}/val_loss":   v_loss,
                f"2.3/{tag}/val_dice":   v_dice,
                f"2.3/{tag}/val_acc":    v_acc,
            })

        run.finish()

        return {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_acc": [],
        }

    # ---- Freezing strategies ----
    def freeze_all(m):
        for p in m.encoder.parameters():
            p.requires_grad = False

    def freeze_early(m):
        # freeze early low-level blocks, unfreeze later blocks
        for p in m.encoder.enc1.parameters(): p.requires_grad = False
        for p in m.encoder.enc2.parameters(): p.requires_grad = False
        for p in m.encoder.enc3.parameters(): p.requires_grad = False

    def freeze_none(m):
        pass  # all trainable

    # ---- LR choices (helps show clearer differences) ----
    lr_strict  = 1e-3   # decoder only can handle higher LR
    lr_partial = 5e-4
    lr_full    = 1e-4   # lower to avoid destabilizing full backbone

    run_strategy("strict_feature_extractor", freeze_all,  "strict",  lr_strict)
    run_strategy("partial_fine_tuning",      freeze_early, "partial", lr_partial)
    run_strategy("full_fine_tuning",         freeze_none,  "full",    lr_full)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.4 — Feature Map Visualization
# ═══════════════════════════════════════════════════════════════════════════════
def task_2_4(args):
    device = get_device()
    run = wandb.init(project=args.wandb_project, name="2.4-feature-maps",
                     tags=["report", "2.4"])

    model = VGG11Classifier(num_classes=NUM_BREEDS).to(device)
    model.load_state_dict(load_ckpt(CKPT_CLF, device))
    model.eval()

    # --- Input image ---
    if args.image_path and os.path.exists(args.image_path[0]):
        pil_img = Image.open(args.image_path[0]).convert("RGB")
        tfm = get_val_transforms(IMG_SIZE)
        out = tfm(image=np.array(pil_img), mask=np.zeros((pil_img.height, pil_img.width), dtype=np.uint8),
                  bboxes=[], bbox_labels=[])
        img_tensor = out["image"].unsqueeze(0).to(device)
        img_display = unnorm(out["image"])
    else:
        val_loader = make_val_loader(args, batch_size=1)
        imgs, _, _, _ = next(iter(val_loader))
        img_tensor = imgs.to(device)
        img_display = unnorm(imgs[0])

    # --- Hooks ---
    feat_maps = {}
    def make_hook(name):
        def hook(module, inp, out):
            feat_maps[name] = out.detach().cpu()
        return hook

    # first conv (enc1[0]) and last conv before pooling (enc5[3])
    # if you want post-ReLU maps, use enc1[2] and enc5[5]
    h1 = model.classifier.enc1[0].register_forward_hook(make_hook("enc1_conv1"))
    h2 = model.classifier.enc2[0].register_forward_hook(make_hook("enc2_conv1"))
    h3 = model.classifier.enc3[3].register_forward_hook(make_hook("enc3_conv2"))
    h4 = model.classifier.enc4[3].register_forward_hook(make_hook("enc4_conv2"))
    h5 = model.classifier.enc5[3].register_forward_hook(make_hook("enc5_conv2"))

    with torch.no_grad():
        model(img_tensor)

    h1.remove(); h2.remove()

    def plot_feature_maps(feat, title, n_maps=16, save_path=None):
        feat = feat[0]  # [C, H, W]
        n = min(n_maps, feat.shape[0])
        cols = 8
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.array(axes).ravel()

        for i in range(n):
            fm = feat[i].numpy()
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-6)  # normalize per map
            axes[i].imshow(fm, cmap="viridis")
            axes[i].axis("off")
        for i in range(n, len(axes)):
            axes[i].axis("off")

        plt.suptitle(title, fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=120)
        plt.close()
        return save_path

    # Original image
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_display); ax.axis("off"); ax.set_title("Input Image")
    plt.savefig("2.4_input.png", dpi=120); plt.close()

    p1 = plot_feature_maps(feat_maps["enc1_conv1"], "Layer 1 (enc1) — Edges & Textures",
                           save_path="2.4_enc1_fmaps.png")
    p2 = plot_feature_maps(feat_maps["enc2_conv1"], "Layer 2 (enc2) — Mid level Textures",
                           save_path="2.4_enc2_fmaps.png")

    p3 = plot_feature_maps(feat_maps["enc3_conv2"], "Layer 3 (enc3) — High‑level Shapes",
                           save_path="2.4_enc3_fmaps.png")
    p4 = plot_feature_maps(feat_maps["enc4_conv2"], "Layer 4 (enc5) — High‑level Shapes",
                           save_path="2.4_enc4_fmaps.png")
    p5 = plot_feature_maps(feat_maps["enc5_conv2"], "Layer 5 (enc5) — High‑level Shapes",
                           save_path="2.4_enc5_fmaps.png")
    run.log({
        "2.4/input_image":       wandb.Image("2.4_input.png"),
        "2.4/enc1_feature_maps": wandb.Image(p1),
        "2.4/enc2_feature_maps": wandb.Image(p2),
        "2.4/enc3_feature_maps": wandb.Image(p3),
        "2.4/enc4_feature_maps": wandb.Image(p4),
        "2.4/enc5_feature_maps": wandb.Image(p5),
    })

    print("Feature maps logged to W&B ✓")
    run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.5 — Object Detection: Confidence & IoU Table
# ═══════════════════════════════════════════════════════════════════════════════
def task_2_5(args):
    device = get_device()
    run    = wandb.init(project=args.wandb_project, name="2.5-detection-table",
                        tags=["report","2.5"])

    model = VGG11Localizer(freeze_backbone=False).to(device)
    model.load_state_dict(load_ckpt(CKPT_LOC, device))
    model.eval()

    val_loader = make_val_loader(args, batch_size=1)

    def iou_single(p, g, eps=1e-6):
        px1,py1,px2,py2 = p[0]-p[2]/2, p[1]-p[3]/2, p[0]+p[2]/2, p[1]+p[3]/2
        gx1,gy1,gx2,gy2 = g[0]-g[2]/2, g[1]-g[3]/2, g[0]+g[2]/2, g[1]+g[3]/2
        ix1,iy1 = max(px1,gx1), max(py1,gy1)
        ix2,iy2 = min(px2,gx2), min(py2,gy2)
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        union = (px2-px1)*(py2-py1)+(gx2-gx1)*(gy2-gy1)-inter+eps
        return inter/union

    def draw_bbox_img(img_np, pred_norm, gt_norm, iou):
        H, W = img_np.shape[:2]
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(img_np)
        for box, color, label in [(gt_norm,'lime','GT'), (pred_norm,'red',f'Pred IoU={iou:.2f}')]:
            cx,cy,bw,bh = box
            x1=(cx-bw/2)*W; y1=(cy-bh/2)*H
            ax.add_patch(patches.Rectangle((x1,y1),bw*W,bh*H,
                         linewidth=2,edgecolor=color,facecolor='none'))
            ax.text(x1+2,y1+10,label,color='white',fontsize=7,
                    bbox=dict(facecolor=color,alpha=0.6,pad=1))
        ax.axis('off'); plt.tight_layout()
        path = f"/tmp/det_{random.randint(0,9999)}.png"
        plt.savefig(path,dpi=80); plt.close()
        return path

    table = wandb.Table(columns=["image","gt_box","pred_box","iou","confidence","note"])
    rows_collected = 0

    with torch.no_grad():
        for imgs, labels, bboxes_norm, _ in val_loader:
            if rows_collected >= 16: break
            pred_px   = model(imgs.to(device))
            pred_norm = torch.clamp(pred_px / IMG_SIZE, 0.0, 1.0).cpu()
            gt_n   = bboxes_norm[0].numpy()
            pr_n   = pred_norm[0].numpy()
            iou    = iou_single(pr_n, gt_n)
            conf   = float(1.0 - abs(pr_n - gt_n).mean())  # proximity as confidence proxy
            img_np = unnorm(imgs[0])
            img_path = draw_bbox_img(img_np, pr_n, gt_n, iou)
            note = "HIGH CONF LOW IOU (failure)" if conf > 0.7 and iou < 0.3 else ""
            table.add_data(wandb.Image(img_path),
                           str(np.round(gt_n,3).tolist()),
                           str(np.round(pr_n,3).tolist()),
                           round(iou,4), round(conf,4), note)
            rows_collected += 1

    run.log({"2.5/detection_table": table})
    print(f"Logged {rows_collected} rows to W&B detection table ✓")
    run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.6 — Segmentation: Dice vs Pixel Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def task_2_6(args):
    """
    Log 5 sample images: Original | GT Trimap | Predicted Trimap Mask.
    Track Pixel Accuracy and Dice Score on val loader.
    Create scatter plot showing Dice vs PixelAcc to illustrate the gap.
    """
    device = get_device()
    run    = wandb.init(project=args.wandb_project, name="2.6-segmentation-eval",
                        tags=["report","2.6"])

    nc    = 3
    model = VGG11UNet(num_classes=nc).to(device)
    model.load_state_dict(load_ckpt(CKPT_SEG, device))
    model.eval()

    val_loader = make_val_loader(args, batch_size=8)

    all_px_acc, all_dice = [], []
    sample_imgs = []

    with torch.no_grad():
        for imgs, _, _, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs).argmax(1)

            for b in range(imgs.shape[0]):
                px_acc = (preds[b] == masks[b]).float().mean().item()
                dice_vals = []
                for c in range(nc):
                    p_c = (preds[b]==c).float(); t_c = (masks[b]==c).float()
                    d   = (2*(p_c*t_c).sum()+1e-6)/(p_c.sum()+t_c.sum()+1e-6)
                    dice_vals.append(d.item())
                mean_dice = float(np.mean(dice_vals))
                all_px_acc.append(px_acc); all_dice.append(mean_dice)

                if len(sample_imgs) < 5:
                    sample_imgs.append((
                        unnorm(imgs[b].cpu()),
                        masks[b].cpu().numpy(),
                        preds[b].cpu().numpy(),
                        px_acc, mean_dice
                    ))

    # ── Log 5 sample images ───────────────────────────────────────────────
    fig, axes = plt.subplots(5, 3, figsize=(10, 16))
    for row, (img, gt, pr, pxa, dc) in enumerate(sample_imgs):
        axes[row,0].imshow(img);               axes[row,0].axis('off')
        axes[row,1].imshow(mask_to_rgb(gt));   axes[row,1].axis('off')
        axes[row,2].imshow(mask_to_rgb(pr));   axes[row,2].axis('off')
        if row == 0:
            axes[row,0].set_title("Original")
            axes[row,1].set_title("GT Trimap")
            axes[row,2].set_title("Predicted Trimap")
        axes[row,2].set_xlabel(f"PixAcc={pxa:.3f}  Dice={dc:.3f}", fontsize=8)

    fig.legend(handles=[Patch(color=np.array([0,200,0])/255, label='fg'),
                         Patch(color=np.array([200,0,0])/255, label='bg'),
                         Patch(color=np.array([0,0,200])/255, label='boundary')],
               loc='lower center', ncol=3)
    plt.suptitle("Segmentation Samples: Original | GT | Prediction")
    plt.tight_layout(rect=[0,0.03,1,0.97])
    plt.savefig("2.6_seg_samples.png", dpi=100); plt.close()

    # ── Dice vs Pixel Accuracy scatter ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(all_px_acc, all_dice, alpha=0.4, s=10)
    ax.plot([0,1],[0,1],'r--', label="y=x (equal)")
    ax.set_xlabel("Pixel Accuracy"); ax.set_ylabel("Mean Dice Score")
    ax.set_title("Pixel Accuracy vs Dice Score\n"
                 "(Points above diagonal = PixAcc artificially high)")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig("2.6_dice_vs_pixacc.png", dpi=100); plt.close()

    run.log({
        "2.6/seg_samples":        wandb.Image("2.6_seg_samples.png"),
        "2.6/dice_vs_pixacc":     wandb.Image("2.6_dice_vs_pixacc.png"),
        "2.6/mean_pixel_accuracy": float(np.mean(all_px_acc)),
        "2.6/mean_dice":           float(np.mean(all_dice)),
    })

    # Log per-image to W&B table for interactive view
    table = wandb.Table(columns=["pixel_accuracy","mean_dice"])
    for pa, dc in zip(all_px_acc, all_dice):
        table.add_data(round(pa,4), round(dc,4))
    run.log({"2.6/metrics_table": table})
    run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.7 — Final Pipeline Showcase on Wild Images
# ═══════════════════════════════════════════════════════════════════════════════

def task_2_7(args):
    """
    Run full pipeline (clf + loc + seg) on 3 novel internet pet images.
    Visualize: bbox prediction, segmentation overlay, breed label + confidence.

    REQUIRED: --image_path img1.jpg img2.jpg img3.jpg
    """
    if not args.image_path or len(args.image_path) < 1:
        print("Error: --image_path requires at least 1 image path"); return

    device = get_device()
    run    = wandb.init(project=args.wandb_project, name="2.7-wild-inference",
                        tags=["report","2.7"])

    clf_model = VGG11Classifier(num_classes=NUM_BREEDS).to(device)
    clf_model.load_state_dict(load_ckpt(CKPT_CLF, device)); clf_model.eval()

    loc_model = VGG11Localizer(freeze_backbone=False).to(device)
    loc_model.load_state_dict(load_ckpt(CKPT_LOC, device)); loc_model.eval()

    seg_model = VGG11UNet(num_classes=3).to(device)
    seg_model.load_state_dict(load_ckpt(CKPT_SEG, device)); seg_model.eval()

    # Load breed map from list.txt
    list_path = os.path.join(args.data_root, "annotations", "list.txt")
    breed_map = {}
    if os.path.exists(list_path):
        with open(list_path) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    cid = int(parts[1]) - 1
                    bname = "_".join(parts[0].split("_")[:-1])
                    if cid not in breed_map: breed_map[cid] = bname
    else:
        breed_map = {i: f"Breed_{i}" for i in range(NUM_BREEDS)}

    tfm = get_val_transforms(IMG_SIZE)

    images_to_show = args.image_path[:3]
    result_paths   = []

    for img_path in images_to_show:
        if not os.path.exists(img_path):
            print(f"  Skipping {img_path} — not found"); continue

        pil_img = Image.open(img_path).convert("RGB")
        out = tfm(image=np.array(pil_img),
                  mask=np.zeros((pil_img.height,pil_img.width),dtype=np.uint8),
                  bboxes=[], bbox_labels=[])
        img_t  = out["image"].unsqueeze(0).to(device)
        img_np = unnorm(out["image"])
        H, W   = img_np.shape[:2]

        with torch.no_grad():
            # Classification
            logits = clf_model(img_t)
            probs  = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(1).item()
            conf   = probs[0, pred_idx].item()

            # Localisation
            pred_px   = loc_model(img_t)
            pred_norm = torch.clamp(pred_px / IMG_SIZE, 0.0, 1.0).cpu().numpy()[0]

            # Segmentation
            seg_logits = seg_model(img_t)
            seg_pred   = seg_logits.argmax(1).squeeze(0).cpu().numpy()

        breed = breed_map.get(pred_idx, f"Breed_{pred_idx}")
        cx, cy, bw, bh = pred_norm
        x1 = (cx-bw/2)*W; y1 = (cy-bh/2)*H

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        # Col 1: image + bbox
        axes[0].imshow(img_np)
        axes[0].add_patch(patches.Rectangle((x1,y1),bw*W,bh*H,
                           linewidth=2,edgecolor='red',facecolor='none'))
        axes[0].set_title(f"Breed: {breed}\nConf: {conf:.2%}", fontsize=9)
        axes[0].axis('off')

        # Col 2: segmentation mask
        axes[1].imshow(mask_to_rgb(seg_pred))
        axes[1].set_title("Seg Prediction"); axes[1].axis('off')

        # Col 3: overlay
        axes[2].imshow(overlay_mask(img_np, seg_pred))
        axes[2].set_title("Seg Overlay"); axes[2].axis('off')

        fig.legend(handles=[Patch(color=np.array([0,200,0])/255,label='fg'),
                             Patch(color=np.array([200,0,0])/255,label='bg'),
                             Patch(color=np.array([0,0,200])/255,label='boundary')],
                   loc='lower center', ncol=3, fontsize=8)
        plt.suptitle(f"Wild Image: {os.path.basename(img_path)}", fontsize=11)
        plt.tight_layout(rect=[0,0.05,1,1])
        save_p = f"2.7_{os.path.splitext(os.path.basename(img_path))[0]}.png"
        plt.savefig(save_p, dpi=120); plt.close()
        result_paths.append(save_p)

        run.log({
            f"2.7/{os.path.basename(img_path)}": wandb.Image(save_p),
            f"2.7/{os.path.basename(img_path)}_breed": breed,
            f"2.7/{os.path.basename(img_path)}_confidence": round(conf, 4),
        })
        print(f"  {img_path}: {breed} ({conf:.2%})")

    run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.8 — Meta-Analysis: Combined metric plots
# ═══════════════════════════════════════════════════════════════════════════════

def task_2_8(args):
    """
    Final evaluation run: evaluate all three trained models on val split.
    Log comprehensive metrics to W&B for the meta-analysis section.
    Produces overlaid train/val curves (reads from checkpoint history if available,
    otherwise does a fresh eval pass).
    """
    device = get_device()
    run    = wandb.init(project=args.wandb_project, name="2.8-meta-analysis",
                        tags=["report","2.8"])

    val_loader = make_val_loader(args, batch_size=16)

    # ── Classification ────────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, f1_score
    clf_model = VGG11Classifier(num_classes=NUM_BREEDS).to(device)
    clf_model.load_state_dict(load_ckpt(CKPT_CLF, device)); clf_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels, _, _ in val_loader:
            all_preds.extend(clf_model(imgs.to(device)).argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
    clf_acc    = accuracy_score(all_labels, all_preds)
    clf_f1_mac = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    print(f"  Classification — Acc={clf_acc:.4f}  F1_macro={clf_f1_mac:.4f}")

    # ── Localisation ──────────────────────────────────────────────────────
    loc_model = VGG11Localizer(freeze_backbone=False).to(device)
    loc_model.load_state_dict(load_ckpt(CKPT_LOC, device)); loc_model.eval()
    iou_fn = IoULoss(reduction="none")
    all_ious = []
    with torch.no_grad():
        for imgs, _, bboxes_norm, _ in val_loader:
            pred_px   = loc_model(imgs.to(device))
            pred_norm = torch.clamp(pred_px / IMG_SIZE, 0.0, 1.0)
            ious      = 1.0 - iou_fn(pred_norm.cpu(), bboxes_norm)
            all_ious.extend(ious.tolist())
    mean_iou  = float(np.mean(all_ious))
    acc_iou50 = float(np.mean([i >= 0.5 for i in all_ious]))
    print(f"  Localisation   — mIoU={mean_iou:.4f}  Acc@IoU0.5={acc_iou50:.4f}")

    # ── Segmentation ──────────────────────────────────────────────────────
    nc = 3
    seg_model = VGG11UNet(num_classes=nc).to(device)
    seg_model.load_state_dict(load_ckpt(CKPT_SEG, device)); seg_model.eval()
    all_dice_seg, all_px_seg = [], []
    with torch.no_grad():
        for imgs, _, _, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = seg_model(imgs).argmax(1)
            for b in range(imgs.shape[0]):
                px_acc = (preds[b]==masks[b]).float().mean().item()
                dc_vals = []
                for c in range(nc):
                    p_c=(preds[b]==c).float(); t_c=(masks[b]==c).float()
                    dc_vals.append(((2*(p_c*t_c).sum()+1e-6)/(p_c.sum()+t_c.sum()+1e-6)).item())
                all_dice_seg.append(float(np.mean(dc_vals)))
                all_px_seg.append(px_acc)
    mean_dice_seg = float(np.mean(all_dice_seg))
    mean_pxacc    = float(np.mean(all_px_seg))
    print(f"  Segmentation   — Dice={mean_dice_seg:.4f}  PixAcc={mean_pxacc:.4f}")

    # ── Summary bar chart ─────────────────────────────────────────────────
    metrics = {
        "Clf Accuracy": clf_acc, "Clf F1-macro": clf_f1_mac,
        "Loc mIoU": mean_iou, "Loc Acc@0.5": acc_iou50,
        "Seg Dice": mean_dice_seg, "Seg PixAcc": mean_pxacc,
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue','steelblue','darkorange','darkorange','green','green']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor='white')
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Score")
    ax.set_title("Meta-Analysis: Final Val Metrics Across All Tasks")
    ax.grid(axis='y', alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='steelblue',label='Classification'),
                       Patch(color='darkorange', label='Localisation'),
                       Patch(color='green',      label='Segmentation')],
              loc='upper right')
    plt.xticks(rotation=20, ha='right'); plt.tight_layout()
    plt.savefig("2.8_meta_summary.png", dpi=120); plt.close()

    run.log({
        "2.8/clf_accuracy":      clf_acc,
        "2.8/clf_f1_macro":      clf_f1_mac,
        "2.8/loc_mean_iou":      mean_iou,
        "2.8/loc_acc_iou50":     acc_iou50,
        "2.8/seg_mean_dice":     mean_dice_seg,
        "2.8/seg_pixel_accuracy":mean_pxacc,
        "2.8/summary_chart":     wandb.Image("2.8_meta_summary.png"),
    })
    run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="W&B Report Tasks 2.1-2.8")
    p.add_argument("--task",           type=str, required=True,
                   choices=["2.1","2.2","2.3","2.4","2.5","2.6","2.7","2.8"],
                   help="Which report task to run")
    p.add_argument("--data_root",      type=str, default="data/oxford-iiit-pet")
    p.add_argument("--wandb_project",  type=str, default="DA6402-Assignment-2_v1")
    p.add_argument("--image_path",     type=str, nargs="+", default='inference/clf2.png',
                   help="Image path(s) for tasks 2.4 and 2.7")
    p.add_argument("--epochs_2_2",     type=int, default=15,
                   help="Epochs for 2.2 dropout comparison (default 15)")
    p.add_argument("--epochs_2_3",     type=int, default=10,
                   help="Epochs per strategy for 2.3 (default 10)")
    p.add_argument("--device",         type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(42)

    TASK_MAP = {
        "2.1": task_2_1, "2.2": task_2_2, "2.3": task_2_3,
        "2.4": task_2_4, "2.5": task_2_5, "2.6": task_2_6,
        "2.7": task_2_7, "2.8": task_2_8,
    }
    print(f"\n{'='*60}\nRunning W&B Report Task {args.task}\n{'='*60}")
    TASK_MAP[args.task](args)
    print(f"\nTask {args.task} complete ✓")