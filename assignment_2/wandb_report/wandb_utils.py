
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import wandb


# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data.pets_dataset     import OxfordIIITPetDataset, get_train_transforms, get_val_transforms
from data.stratified_split import get_stratified_split

# ── constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
NUM_BREEDS = 37
MEAN       = np.array([0.485, 0.456, 0.406])
STD        = np.array([0.229, 0.224, 0.225])
CKPT_CLF   = os.path.join("checkpoints", "classifier.pth")
CKPT_LOC   = os.path.join("checkpoints", "localizer.pth")
CKPT_SEG   = os.path.join("checkpoints", "unet_3.pth")
ENTITY_NAME = 'da25s003-indian-institute-of-technology-madras'

MASK_COLORS = np.array([[0,200,0],[200,0,0],[0,0,200]], dtype=np.uint8)


def make_val_loader(args, batch_size=16):
    root     = pathlib.Path(args.data_root)
    ann_file = root / "annotations" / "trainval.txt"
    _, val_records = get_stratified_split(ann_file, val_frac=0.1, seed=42)
    val_ds = OxfordIIITPetDataset(
        root=args.data_root, records=val_records,
        images_dir=root/"images", masks_dir=root/"annotations"/"trimaps",
        transform=get_val_transforms(IMG_SIZE),
    )
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)


def make_train_val_loaders(args, batch_size=32):
    root     = pathlib.Path(args.data_root)
    ann_file = root / "annotations" / "trainval.txt"
    train_recs, val_recs = get_stratified_split(ann_file, val_frac=0.1, seed=42)
    train_ds = OxfordIIITPetDataset(
        root=args.data_root, records=train_recs,
        images_dir=root/"images", masks_dir=root/"annotations"/"trimaps",
        transform=get_train_transforms(IMG_SIZE),
    )
    val_ds = OxfordIIITPetDataset(
        root=args.data_root, records=val_recs,
        images_dir=root/"images", masks_dir=root/"annotations"/"trimaps",
        transform=get_val_transforms(IMG_SIZE),
    )
    kw = dict(num_workers=2, pin_memory=torch.cuda.is_available())
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    print("-"*30)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw))


# Task 1 utils

def quick_train_clf(model, train_loader, val_loader, device, epochs, tag, run, task='2.1'):
    """Train classifier for N epochs, log to wandb run, return train/val loss lists."""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-5)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, 15))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])

    use_amp   = device.type == "cuda"
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    t_losses, v_losses, t_accs, v_accs = [], [], [], []
    for epoch in range(1, epochs+1):
        model.train(); t_loss = 0.; t_correct=0; t_total=0
        for imgs, labels, _, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(imgs); loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            preds = logits.argmax(dim=1)
            t_correct += (preds == labels).sum().item()
            t_total += labels.numel()
            t_loss += loss.item()
           
        t_loss /= len(train_loader)
        t_acc = t_correct / max(1, t_total)
        scheduler.step()

        model.eval(); v_loss = 0.; v_correct = 0; v_total = 0
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                v_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.numel()
        v_loss /= len(val_loader)
        v_acc = v_correct / max(1, v_total)
        run.log({
            f"{tag}/train_loss": t_loss,
            f"{tag}/val_loss": v_loss,
            f"{tag}/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch
        })

        t_losses.append(t_loss); v_losses.append(v_loss)
        t_accs.append(t_acc); v_accs.append(v_acc)
        if task == '2.1':
            run.log({f"{tag}/train_loss": t_loss, f"{tag}/val_loss": v_loss, "epoch": epoch})
            print(f"  [{tag}] ep {epoch}/{epochs} train={t_loss:.4f} val={v_loss:.4f}")
        if task == '2.2':
            run.log({
                f"{tag}/train_loss": t_loss,
                f"{tag}/val_loss": v_loss,
                f"{tag}/train_acc": t_acc,
                f"{tag}/val_acc": v_acc,
                f"{tag}/loss_gap": t_loss - v_loss,
                f"{tag}/acc_gap": t_acc - v_acc,
                "epoch": epoch
            })
            print(f"  [{tag}] ep {epoch}/{epochs} train={t_loss:.4f} val={v_loss:.4f} train_acc={t_acc:.4f} val_acc={v_acc}, loss_gap={(t_loss - v_loss):.4f} acc_gap={(t_acc - v_acc):.4f}")

    return np.asarray(t_losses), np.asarray(v_losses), np.asarray(t_accs),np.asarray(v_accs)

def epoch_to_reach(v_losses, threshold):
    for i, v in enumerate(v_losses, 1):
        if v <= threshold:
            return i
    return None

def log_loss_curve(tag, t_losses, v_losses, run):
    epochs = list(range(1, len(t_losses)+1))
    run.log({
        f"{tag}/loss_curve": wandb.plot.line_series(
            xs=epochs,
            ys=[t_losses, v_losses],
            keys=["train", "val"],
            title=f"{tag} loss curve",
            xname="epoch"
        )
    })

def log_combined_curves(t_bn, v_bn, t_nbn, v_nbn, run):
    epochs = list(range(1, len(t_bn)+1))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(epochs, t_bn,  'b--', label="BN train")
    ax.plot(epochs, v_bn,  'b-',  label="BN val")
    ax.plot(epochs, t_nbn, 'r--', label="NoBN train")
    ax.plot(epochs, v_nbn, 'r-',  label="NoBN val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Curves: BN vs No-BN")
    ax.grid(True); ax.legend()
    path = "2.1_training_curves.png"
    plt.savefig(path, dpi=120); plt.close()
    run.log({"2.1/training_curves": wandb.Image(path)})

def lr_sweep_probe(model, train_loader, device, lr=1e-3, steps=200):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    losses = []

    for i, (imgs, labels, _, _) in enumerate(train_loader):
        if i >= steps:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            return {"stable": False, "reason": "nan", "losses": losses}
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Simple stability rule
    if len(losses) < 5:
        return {"stable": False, "reason": "too_short", "losses": losses}

    start = sum(losses[:5]) / 5
    end   = sum(losses[-5:]) / 5
    max_l = max(losses)

    stable = (max_l < 5 * start) and (end < 2 * start)
    return {
        "stable": stable,
        "reason": "ok" if stable else "loss_increase",
        "losses": losses,
        "start": start,
        "end": end,
        "max": max_l
    }


# Task 2 utils

def accuracy_loss_comparision_plots(args, all_results, epochs):
    conditions = [
        ("no_dropout",   0.0),
        ("dropout_0.2",  0.2),
        ("dropout_0.5",  0.5),
    ]

    run_summary = wandb.init(project=args.wandb_project, entity=ENTITY_NAME,
                              name="2.2-dropout-comparison", tags=["report","2.2"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['green', 'orange', 'red']
    for (name, dp), color in zip(conditions, colors):
        t_l = all_results[name]["train_loss"]
        v_l = all_results[name]["val_loss"]
        axes[0].plot(t_l, color=color, linestyle='-',  label=f"{name} train")
        axes[1].plot(v_l, color=color, linestyle='--', label=f"{name} val")

    for ax, title in zip(axes, ["Training Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(title); ax.legend(); ax.grid(True)

    plt.suptitle("Dropout Comparison: Train vs Validation Loss", fontsize=12)
    plt.tight_layout()
    path = "2.2_dropout_comparison.png"
    plt.savefig(path, dpi=120); plt.close()
    run_summary.log({"2.2/dropout_comparison": wandb.Image(path)})

    # --- Accuracy comparison plot (train vs val) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['green', 'orange', 'red']
    for (name, dp), color in zip(conditions, colors):
        t_a = all_results[name]["train_acc"]
        v_a = all_results[name]["val_acc"]
        axes[0].plot(t_a, color=color, linestyle='-',  label=f"{name} train")
        axes[1].plot(v_a, color=color, linestyle='--', label=f"{name} val")

    for ax, title in zip(axes, ["Training Accuracy", "Validation Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_title(title); ax.legend(); ax.grid(True)

    plt.suptitle("Dropout Comparison: Train vs Validation Accuracy", fontsize=12)
    plt.tight_layout()
    acc_path = "2.2_dropout_accuracy.png"
    plt.savefig(acc_path, dpi=120); plt.close()
    run_summary.log({"2.2/dropout_accuracy_comparison": wandb.Image(acc_path)})

    # --- Gap plots (loss_gap + acc_gap) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (name, dp), color in zip(conditions, colors):
        loss_gap = all_results[name]["loss_gap"]
        acc_gap  = all_results[name]["acc_gap"]
        axes[0].plot(loss_gap, color=color, linestyle='-',  label=f"{name}")
        axes[1].plot(acc_gap,  color=color, linestyle='--', label=f"{name}")

    for ax, title in zip(axes, ["Loss Gap (Train - Val)", "Accuracy Gap (Train - Val)"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Gap")
        ax.set_title(title); ax.legend(); ax.grid(True)

    plt.suptitle("Dropout Comparison: Generalization Gaps", fontsize=12)
    plt.tight_layout()
    gap_path = "2.2_dropout_gaps.png"
    plt.savefig(gap_path, dpi=120); plt.close()
    run_summary.log({"2.2/dropout_gap_comparison": wandb.Image(gap_path)})

    # Also log individual curves for W&B interactive overlay
    ep = list(range(1, epochs+1))
    for i, ep_i in enumerate(ep):
        log = {"epoch": ep_i}
        for name, _ in conditions:
            log[f"2.2/{name}/train_loss"] = all_results[name]["train_loss"][i]
            log[f"2.2/{name}/val_loss"]   = all_results[name]["val_loss"][i]
            log[f"2.2/{name}/train_acc"]  = all_results[name]["train_acc"][i]
            log[f"2.2/{name}/val_acc"]    = all_results[name]["val_acc"][i]
            log[f"2.2/{name}/loss_gap"]   = all_results[name]["loss_gap"][i]
            log[f"2.2/{name}/acc_gap"]    = all_results[name]["acc_gap"][i]
        run_summary.log(log)

    run_summary.finish()


# Task 6 utils
def make_confusion_map(pred, gt, fg_class=0):
    # pred, gt: [H, W] numpy or torch arrays with class ids
    pred = torch.as_tensor(pred)
    gt   = torch.as_tensor(gt)

    pred_fg = (pred == fg_class)
    gt_fg   = (gt == fg_class)

    tp = pred_fg & gt_fg
    fp = pred_fg & (~gt_fg)
    fn = (~pred_fg) & gt_fg
    tn = (~pred_fg) & (~gt_fg)

    # RGB map: TP=green, FP=red, FN=blue, TN=black
    h, w = pred.shape
    vis = torch.zeros((h, w, 3), dtype=torch.uint8)
    vis[tp] = torch.tensor([0, 255, 0], dtype=torch.uint8)   # TP
    vis[fp] = torch.tensor([255, 0, 0], dtype=torch.uint8)   # FP
    vis[fn] = torch.tensor([0, 0, 255], dtype=torch.uint8)   # FN
    vis[tn] = torch.tensor([0, 0, 0], dtype=torch.uint8)     # TN

    return vis.numpy()


# Task 7 utils

def extract_label_from_filename(path):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    # take text before first underscore
    return name

# Task 8 utils
def _series(hist, key):
    # hist can be list of dicts or a DataFrame
    if isinstance(hist, list):
        vals = [row.get(key, np.nan) for row in hist]
        return np.array([v if v is not None else np.nan for v in vals], dtype=float)
    return hist[key].to_numpy()

def _x_axis(hist, candidates):
    for k in candidates:
        if isinstance(hist, list):
            vals = [row.get(k) for row in hist]
            if any(v is not None for v in vals):
                return np.array([v if v is not None else np.nan for v in vals], dtype=float), k
        else:
            if k in hist.columns and hist[k].notna().any():
                return hist[k].to_numpy(), k
    # fallback to step index
    length = len(hist) if isinstance(hist, list) else len(hist.index)
    return np.arange(length), "Step"

def plot_overlay_clf(df, out_dir=".", prefix="clf"):
    x, xlab = _x_axis(df, ["clf_epochs", "epoch"])

    # Loss
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "clf/train/loss"), label="train loss")
    ax.plot(x, _series(df, "clf/val/loss"),   label="val loss")
    ax.set_title("Classification Loss (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True)
    loss_path = f"{out_dir}/{prefix}_loss_overlay.png"
    plt.tight_layout(); plt.savefig(loss_path, dpi=120); plt.close()

    # Accuracy
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "clf/train/accuracy"), label="train acc")
    ax.plot(x, _series(df, "clf/val/accuracy"),   label="val acc")
    ax.set_title("Classification Accuracy (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True)
    acc_path = f"{out_dir}/{prefix}_acc_overlay.png"
    plt.tight_layout(); plt.savefig(acc_path, dpi=120); plt.close()

    # F1 macro
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "clf/train/f1_macro"), label="train F1-macro")
    ax.plot(x, _series(df, "clf/val/f1_macro"),   label="val F1-macro")
    ax.set_title("Classification F1-macro (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("F1-macro")
    ax.legend(); ax.grid(True)
    f1_path = f"{out_dir}/{prefix}_f1_overlay.png"
    plt.tight_layout(); plt.savefig(f1_path, dpi=120); plt.close()

    return loss_path, acc_path, f1_path

def plot_overlay_loc(df, out_dir=".", prefix="loc"):
    x, xlab = _x_axis(df, ["loc_epochs", "epoch"])

    # Total Loss
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "loc/train/total_loss"), label="train total")
    ax.plot(x, _series(df, "loc/val/total_loss"),   label="val total")
    ax.set_title("Localization Total Loss (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True)
    total_path = f"{out_dir}/{prefix}_total_loss_overlay.png"
    plt.tight_layout(); plt.savefig(total_path, dpi=120); plt.close()

    # IoU Loss
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "loc/train/iou_loss"), label="train IoU loss")
    ax.plot(x, _series(df, "loc/val/iou_loss"),   label="val IoU loss")
    ax.set_title("Localization IoU Loss (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("IoU Loss")
    ax.legend(); ax.grid(True)
    iou_path = f"{out_dir}/{prefix}_iou_loss_overlay.png"
    plt.tight_layout(); plt.savefig(iou_path, dpi=120); plt.close()

    # MSE
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "loc/train/mse"), label="train MSE")
    ax.plot(x, _series(df, "loc/val/mse"),   label="val MSE")
    ax.set_title("Localization MSE (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("MSE")
    ax.legend(); ax.grid(True)
    mse_path = f"{out_dir}/{prefix}_mse_overlay.png"
    plt.tight_layout(); plt.savefig(mse_path, dpi=120); plt.close()

    # Mean IoU
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "loc/train/mean_iou"), label="train mIoU")
    ax.plot(x, _series(df, "loc/val/mean_iou"),   label="val mIoU")
    ax.set_title("Localization Mean IoU (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("Mean IoU")
    ax.legend(); ax.grid(True)
    miou_path = f"{out_dir}/{prefix}_miou_overlay.png"
    plt.tight_layout(); plt.savefig(miou_path, dpi=120); plt.close()

    return total_path, iou_path, mse_path, miou_path

def plot_overlay_seg(df, out_dir=".", prefix="seg"):
    x, xlab = _x_axis(df, ["seg_epochs", "epoch"])

    # Loss
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "seg/train/loss"), label="train loss")
    ax.plot(x, _series(df, "seg/val/loss"),   label="val loss")
    ax.set_title("Segmentation Loss (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True)
    loss_path = f"{out_dir}/{prefix}_loss_overlay.png"
    plt.tight_layout(); plt.savefig(loss_path, dpi=120); plt.close()

    # Dice
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "seg/train/mean_dice"), label="train dice")
    ax.plot(x, _series(df, "seg/val/mean_dice"),   label="val dice")
    ax.set_title("Segmentation Mean Dice (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("Dice")
    ax.legend(); ax.grid(True)
    dice_path = f"{out_dir}/{prefix}_dice_overlay.png"
    plt.tight_layout(); plt.savefig(dice_path, dpi=120); plt.close()

    # Pixel Acc
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, _series(df, "seg/train/px_accuracy"), label="train px-acc")
    ax.plot(x, _series(df, "seg/val/px_accuracy"),   label="val px-acc")
    ax.set_title("Segmentation Pixel Accuracy (Train vs Val)")
    ax.set_xlabel(xlab); ax.set_ylabel("Pixel Accuracy")
    ax.legend(); ax.grid(True)
    px_path = f"{out_dir}/{prefix}_pxacc_overlay.png"
    plt.tight_layout(); plt.savefig(px_path, dpi=120); plt.close()

    return loss_path, dice_path, px_path
