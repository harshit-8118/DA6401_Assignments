"""train.py — Sequential training for DA6401 Assignment-2.

Tasks:
  Task 1: VGG11Classifier  -> classifier.pth
  Task 2: VGG11Localizer   -> localizer.pth  (backbone from classifier.pth)
  Task 3: VGG11UNet        -> unet.pth       (encoder  from classifier.pth)

Data splits:
  trainval.txt -> 90% train / 10% val  (random split, seeded)
  test.txt     -> held-out test set    (inference / final evaluation only)

Segmentation decision (--seg_classes):
  3  -> CrossEntropyLoss on all trimap classes {0=fg, 1=bg, 2=boundary}
        Richer supervision; boundary class explicitly learned.
  1  -> BCEWithLogitsLoss, binary fg vs rest
        Simpler; matches VGG11UNet(num_classes=1) default.
  Default: 3  (recommended — full trimap supervision)

Localizer bbox convention:
  Dataset returns normalised (cx,cy,w,h) in [0,1].
  Localizer outputs pixel-space (cx,cy,w,h) — no sigmoid in RegressionHead.
  MSE loss computed on pixel-space values.
  IoU  loss computed on normalised values (divide pred by IMG_SIZE, clamp [0,1]).
  IoULoss always receives values in [0,1] -> range guaranteed.

Usage:
  python train.py --data_root ./data --task all --use_wandb
  python train.py --data_root ./data --task clf --clf_epochs 30 --use_wandb
  python train.py --data_root ./data --task loc --use_wandb
  python train.py --data_root ./data --task seg --seg_classes 3 --use_wandb
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from models.classification import VGG11Classifier
from models.localization   import VGG11Localizer
from models.segmentation   import VGG11UNet
from losses.iou_loss import IoULoss
from data.pets_dataset import OxfordIIITPetDataset, get_train_transforms, get_val_transforms

# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

IMG_SIZE   = 224
NUM_BREEDS = 37
os.makedirs("checkpoints", exist_ok=True)

CKPT_CLF   = os.path.join("checkpoints", "classifier.pth")
CKPT_LOC   = os.path.join("checkpoints", "localizer.pth")
CKPT_SEG   = os.path.join("checkpoints", "unet.pth")


# --------------------------------------------------------------------------- #
#  Reproducibility                                                             #
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    """Fix seed at hardware level for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic cuDNN (slight speed cost, full reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  Checkpoint helpers                                                          #
# --------------------------------------------------------------------------- #

def save_checkpoint(path: str, model: nn.Module, epoch: int, best_metric: float):
    torch.save({
        "state_dict":  model.state_dict(),
        "epoch":       epoch,
        "best_metric": best_metric,
    }, path)
    print(f"  [ckpt] saved {path}  epoch={epoch}  metric={best_metric:.4f}")


def load_backbone_from_classifier(model, clf_path: str, encoder_attr: str = "encoder"):
    """Transfer enc1..enc5 weights from saved VGG11Classifier into model.encoder.

    VGG11Classifier keys: 'classifier.enc*'  ->  target: 'enc*'
    """
    if not os.path.exists(clf_path):
        print(f"  [backbone] {clf_path} not found — random init")
        return
    ckpt = torch.load(clf_path, map_location="cpu", weights_only=True)
    sd   = ckpt.get("state_dict", ckpt)
    enc  = getattr(model, encoder_attr)
    remapped = {
        k[len("classifier."):]: v
        for k, v in sd.items()
        if k.startswith("classifier.enc")
    }
    missing, unexpected = enc.load_state_dict(remapped, strict=False)
    print(f"  [backbone] loaded {len(remapped)} tensors | "
          f"missing={len(missing)} unexpected={len(unexpected)}")


# --------------------------------------------------------------------------- #
#  DataLoaders                                                                 #
# --------------------------------------------------------------------------- #

def make_loaders(args):
    """
    trainval.txt -> 90% train / 10% val  (random split, seeded)
    test.txt     -> test  (held-out, used only for final evaluation)
    """
    full_train = OxfordIIITPetDataset(
        root=args.data_root, split="trainval",
        transform=get_train_transforms(IMG_SIZE),
    )
    full_train_no_aug = OxfordIIITPetDataset(
        root=args.data_root, split="trainval",
        transform=get_val_transforms(IMG_SIZE),   # no aug for val subset
    )
    test_ds = OxfordIIITPetDataset(
        root=args.data_root, split="test",
        transform=get_val_transforms(IMG_SIZE),
    )

    n_total = len(full_train)
    n_val   = max(1, int(0.1 * n_total))
    n_train = n_total - n_val

    # Use a fixed generator so the split is reproducible
    gen = torch.Generator().manual_seed(args.seed)
    train_idx, val_idx = random_split(
        range(n_total), [n_train, n_val], generator=gen)

    # Subset with augmentation for train, no augmentation for val
    from torch.utils.data import Subset
    train_ds = Subset(full_train,        list(train_idx))
    val_ds   = Subset(full_train_no_aug, list(val_idx))

    kw = dict(num_workers=args.num_workers, pin_memory=(args.device.type == "cuda"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, drop_last=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, drop_last=False, **kw)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
#  Metric helpers                                                              #
# --------------------------------------------------------------------------- #

def clf_metrics(all_preds: list, all_labels: list) -> dict:
    """Full classification metrics from collected batch lists."""
    p = np.array(all_preds)
    t = np.array(all_labels)
    return {
        "accuracy":   accuracy_score(t, p),
        "f1_macro":   f1_score(t, p, average="macro",    zero_division=0),
        "f1_micro":   f1_score(t, p, average="micro",    zero_division=0),
        "f1_weighted":f1_score(t, p, average="weighted", zero_division=0),
        "prec_macro": precision_score(t, p, average="macro",    zero_division=0),
        "prec_micro": precision_score(t, p, average="micro",    zero_division=0),
        "rec_macro":  recall_score(t, p, average="macro",    zero_division=0),
        "rec_micro":  recall_score(t, p, average="micro",    zero_division=0),
    }


def iou_metric(pred_norm: torch.Tensor, tgt_norm: torch.Tensor,
               eps: float = 1e-6) -> float:
    """Mean IoU for a batch of normalised (cx,cy,w,h) boxes."""
    px1 = pred_norm[:, 0] - pred_norm[:, 2] / 2
    py1 = pred_norm[:, 1] - pred_norm[:, 3] / 2
    px2 = pred_norm[:, 0] + pred_norm[:, 2] / 2
    py2 = pred_norm[:, 1] + pred_norm[:, 3] / 2
    tx1 = tgt_norm[:, 0]  - tgt_norm[:, 2]  / 2
    ty1 = tgt_norm[:, 1]  - tgt_norm[:, 3]  / 2
    tx2 = tgt_norm[:, 0]  + tgt_norm[:, 2]  / 2
    ty2 = tgt_norm[:, 1]  + tgt_norm[:, 3]  / 2
    iw  = torch.clamp(torch.min(px2, tx2) - torch.max(px1, tx1), 0)
    ih  = torch.clamp(torch.min(py2, ty2) - torch.max(py1, ty1), 0)
    inter = iw * ih
    union = (px2-px1)*(py2-py1) + (tx2-tx1)*(ty2-ty1) - inter + eps
    return (inter / union).mean().item()


def seg_metrics(pred_mask: torch.Tensor, tgt_mask: torch.Tensor,
                num_classes: int, eps: float = 1e-6) -> dict:
    """Pixel accuracy, per-class Dice, mean Dice, sklearn classification metrics."""
    p_flat = pred_mask.cpu().numpy().ravel()
    t_flat = tgt_mask.cpu().numpy().ravel()

    px_acc = (pred_mask == tgt_mask).float().mean().item()

    # Dice per class
    dice_per_class = []
    for c in range(num_classes):
        p_c = (pred_mask == c).float()
        t_c = (tgt_mask  == c).float()
        d   = (2 * (p_c * t_c).sum() + eps) / (p_c.sum() + t_c.sum() + eps)
        dice_per_class.append(d.item())
    mean_dice = float(np.mean(dice_per_class))

    labels = list(range(num_classes))
    return {
        "px_accuracy":  px_acc,
        "mean_dice":    mean_dice,
        "dice_fg":      dice_per_class[0],
        "dice_bg":      dice_per_class[1] if num_classes > 1 else 0.0,
        "dice_boundary":dice_per_class[2] if num_classes > 2 else 0.0,
        "f1_macro":     f1_score(t_flat, p_flat, average="macro",    labels=labels, zero_division=0),
        "f1_micro":     f1_score(t_flat, p_flat, average="micro",    labels=labels, zero_division=0),
        "f1_weighted":  f1_score(t_flat, p_flat, average="weighted", labels=labels, zero_division=0),
        "prec_macro":   precision_score(t_flat, p_flat, average="macro",    labels=labels, zero_division=0),
        "rec_macro":    recall_score(t_flat, p_flat, average="macro",    labels=labels, zero_division=0),
    }


def wandb_log(metrics: dict, use_wandb: bool):
    if use_wandb:
        wandb.log(metrics)


# --------------------------------------------------------------------------- #
#  Task 1: Classification                                                      #
# --------------------------------------------------------------------------- #
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def train_classifier(args):
    print(f"\n{'='*60}\nTASK 1: Classification\n{'='*60}")
    device = args.device

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name="task1-classifier",
                   config=vars(args), reinit=True)

    train_loader, val_loader, test_loader = make_loaders(args)

    model     = VGG11Classifier(num_classes=NUM_BREEDS, dropout_p=args.dropout_p).to(device)
    # applying kaiming init to the classifier head only (encoder will be loaded from checkpoint)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.clf_lr, weight_decay=0)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.clf_epochs - 5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.clf_epochs, eta_min=1e-6)

    best_f1 = 0.0
    for epoch in range(1, args.clf_epochs + 1):
        # ---- Train ----
        model.train()
        t_loss = 0.0
        t_preds, t_labels = [], []

        scaler = torch.amp.GradScaler('cuda')  # for mixed precision training (optional)
        for imgs, labels, _, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            t_loss += loss.item()
            t_preds.extend(logits.argmax(1).cpu().tolist())
            t_labels.extend(labels.cpu().tolist())

        t_loss /= len(train_loader)
        t_m     = clf_metrics(t_preds, t_labels)
        scheduler.step()

        # ---- Val ----
        model.eval()
        v_loss = 0.0
        v_preds, v_labels = [], []
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                v_loss += criterion(logits, labels).item()
                v_preds.extend(logits.argmax(1).cpu().tolist())
                v_labels.extend(labels.cpu().tolist())
        v_loss /= len(val_loader)
        v_m     = clf_metrics(v_preds, v_labels)

        print(f"  Epoch {epoch:3d}/{args.clf_epochs} | "
              f"train loss={t_loss:.4f} acc={t_m['accuracy']:.4f} f1={t_m['f1_macro']:.4f} | "
              f"val   loss={v_loss:.4f} acc={v_m['accuracy']:.4f} f1={v_m['f1_macro']:.4f}")

        wandb_log({
            "epoch": epoch,
            "clf/lr":              scheduler.get_last_lr()[0],
            "clf/train/loss":      t_loss,
            "clf/train/accuracy":  t_m["accuracy"],
            "clf/train/f1_macro":  t_m["f1_macro"],
            "clf/train/f1_micro":  t_m["f1_micro"],
            "clf/train/f1_weighted": t_m["f1_weighted"],
            "clf/train/prec_macro":  t_m["prec_macro"],
            "clf/train/prec_micro":  t_m["prec_micro"],
            "clf/train/rec_macro":   t_m["rec_macro"],
            "clf/train/rec_micro":   t_m["rec_micro"],
            "clf/val/loss":        v_loss,
            "clf/val/accuracy":    v_m["accuracy"],
            "clf/val/f1_macro":    v_m["f1_macro"],
            "clf/val/f1_micro":    v_m["f1_micro"],
            "clf/val/f1_weighted": v_m["f1_weighted"],
            "clf/val/prec_macro":  v_m["prec_macro"],
            "clf/val/prec_micro":  v_m["prec_micro"],
            "clf/val/rec_macro":   v_m["rec_macro"],
            "clf/val/rec_micro":   v_m["rec_micro"],
        }, args.use_wandb)

        if v_m["f1_macro"] > best_f1:
            best_f1 = v_m["f1_macro"]
            save_checkpoint(CKPT_CLF, model, epoch, best_f1)

    # ---- Test (final evaluation only) ----
    ckpt  = torch.load(CKPT_CLF, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    te_loss = 0.0
    te_preds, te_labels = [], []
    with torch.no_grad():
        for imgs, labels, _, _ in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            te_loss += criterion(logits, labels).item()
            te_preds.extend(logits.argmax(1).cpu().tolist())
            te_labels.extend(labels.cpu().tolist())
    te_loss /= len(test_loader)
    te_m     = clf_metrics(te_preds, te_labels)
    print(f"  [TEST] loss={te_loss:.4f} acc={te_m['accuracy']:.4f} "
          f"f1_macro={te_m['f1_macro']:.4f}")
    wandb_log({
        "clf/test/loss":        te_loss,
        "clf/test/accuracy":    te_m["accuracy"],
        "clf/test/f1_macro":    te_m["f1_macro"],
        "clf/test/f1_micro":    te_m["f1_micro"],
        "clf/test/f1_weighted": te_m["f1_weighted"],
        "clf/test/prec_macro":  te_m["prec_macro"],
        "clf/test/prec_micro":  te_m["prec_micro"],
        "clf/test/rec_macro":   te_m["rec_macro"],
        "clf/test/rec_micro":   te_m["rec_micro"],
    }, args.use_wandb)

    if args.use_wandb:
        wandb.finish()
    return best_f1


# --------------------------------------------------------------------------- #
#  Task 2: Localisation                                                        #
# --------------------------------------------------------------------------- #

def train_localizer(args):
    print(f"\n{'='*60}\nTASK 2: Localisation\n{'='*60}")
    device = args.device

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name="task2-localizer",
                   config=vars(args), reinit=True)

    train_loader, val_loader, test_loader = make_loaders(args)

    model    = VGG11Localizer(dropout_p=args.dropout_p, freeze_backbone=True).to(device)
    load_backbone_from_classifier(model, CKPT_CLF, encoder_attr="encoder")

    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")   # requires inputs in [0,1]

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.loc_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.loc_epochs, eta_min=1e-6)

    unfreeze_at = max(1, args.loc_epochs // 3)
    best_iou    = 0.0

    for epoch in range(1, args.loc_epochs + 1):

        # Unfreeze backbone after warmup
        if epoch == unfreeze_at:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.loc_lr * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.loc_epochs - epoch + 1, eta_min=1e-6)
            print(f"  [Epoch {epoch}] backbone unfrozen (lr={args.loc_lr*0.1:.2e})")

        # ---- Train ----
        model.train()
        t_mse, t_iou_loss, t_total, t_iou_metric = 0., 0., 0., 0.
        for imgs, _, bboxes_norm, _ in train_loader:
            imgs        = imgs.to(device)
            bboxes_norm = bboxes_norm.to(device)        # [0,1] normalised
            bboxes_px   = bboxes_norm * IMG_SIZE        # pixel-space targets

            optimizer.zero_grad()
            pred_px   = model(imgs)                     # pixel-space output (no sigmoid)

            # MSE in pixel space (both pred and target in pixels)
            loss_mse  = mse_loss(pred_px, bboxes_px)

            # IoU loss requires [0,1] — normalise pred, clamp for safety
            pred_norm = torch.clamp(pred_px / IMG_SIZE, 0.0, 1.0)
            loss_iou  = iou_loss(pred_norm, bboxes_norm)   # both in [0,1] ✓

            loss = loss_mse + loss_iou * (IMG_SIZE ** 2)   # scale IoU to match MSE magnitude
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            t_mse        += loss_mse.item()
            t_iou_loss   += loss_iou.item()
            t_total      += loss.item()
            t_iou_metric += iou_metric(pred_norm.detach(), bboxes_norm)

        n = len(train_loader)
        t_mse /= n; t_iou_loss /= n; t_total /= n; t_iou_metric /= n
        scheduler.step()

        # ---- Val ----
        model.eval()
        v_mse, v_iou_loss, v_total, v_iou_metric = 0., 0., 0., 0.
        with torch.no_grad():
            for imgs, _, bboxes_norm, _ in val_loader:
                imgs        = imgs.to(device)
                bboxes_norm = bboxes_norm.to(device)
                bboxes_px   = bboxes_norm * IMG_SIZE
                pred_px     = model(imgs)
                pred_norm   = torch.clamp(pred_px / IMG_SIZE, 0.0, 1.0)
                l_mse       = mse_loss(pred_px, bboxes_px)
                l_iou       = iou_loss(pred_norm, bboxes_norm)
                v_mse        += l_mse.item()
                v_iou_loss   += l_iou.item()
                v_total      += (l_mse + l_iou * IMG_SIZE**2).item()
                v_iou_metric += iou_metric(pred_norm, bboxes_norm)

        n = len(val_loader)
        v_mse /= n; v_iou_loss /= n; v_total /= n; v_iou_metric /= n

        print(f"  Epoch {epoch:3d}/{args.loc_epochs} | "
              f"train mse={t_mse:.2f} iou_l={t_iou_loss:.4f} miou={t_iou_metric:.4f} | "
              f"val   mse={v_mse:.2f} iou_l={v_iou_loss:.4f} miou={v_iou_metric:.4f}")

        wandb_log({
            "epoch": epoch,
            "loc/lr":                scheduler.get_last_lr()[0],
            "loc/train/mse_loss":    t_mse,
            "loc/train/iou_loss":    t_iou_loss,
            "loc/train/total_loss":  t_total,
            "loc/train/mean_iou":    t_iou_metric,
            "loc/val/mse_loss":      v_mse,
            "loc/val/iou_loss":      v_iou_loss,
            "loc/val/total_loss":    v_total,
            "loc/val/mean_iou":      v_iou_metric,
        }, args.use_wandb)

        if v_iou_metric > best_iou:
            best_iou = v_iou_metric
            save_checkpoint(CKPT_LOC, model, epoch, best_iou)

    # ---- Test ----
    ckpt = torch.load(CKPT_LOC, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    te_mse, te_iou_loss, te_total, te_iou_metric = 0., 0., 0., 0.
    with torch.no_grad():
        for imgs, _, bboxes_norm, _ in test_loader:
            imgs        = imgs.to(device)
            bboxes_norm = bboxes_norm.to(device)
            bboxes_px   = bboxes_norm * IMG_SIZE
            pred_px     = model(imgs)
            pred_norm   = torch.clamp(pred_px / IMG_SIZE, 0.0, 1.0)
            l_mse       = mse_loss(pred_px, bboxes_px)
            l_iou       = iou_loss(pred_norm, bboxes_norm)
            te_mse        += l_mse.item()
            te_iou_loss   += l_iou.item()
            te_total      += (l_mse + l_iou * IMG_SIZE**2).item()
            te_iou_metric += iou_metric(pred_norm, bboxes_norm)

    n = len(test_loader)
    te_mse /= n; te_iou_loss /= n; te_total /= n; te_iou_metric /= n
    print(f"  [TEST] mse={te_mse:.2f} iou_loss={te_iou_loss:.4f} mIoU={te_iou_metric:.4f}")
    wandb_log({
        "loc/test/mse_loss":   te_mse,
        "loc/test/iou_loss":   te_iou_loss,
        "loc/test/total_loss": te_total,
        "loc/test/mean_iou":   te_iou_metric,
    }, args.use_wandb)

    if args.use_wandb:
        wandb.finish()
    return best_iou


# --------------------------------------------------------------------------- #
#  Task 3: Segmentation                                                        #
# --------------------------------------------------------------------------- #

def train_segmentation(args):
    print(f"\n{'='*60}\nTASK 3: Segmentation  (seg_classes={args.seg_classes})\n{'='*60}")
    device = args.device
    nc     = args.seg_classes   # 1 or 3

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=f"task3-unet-nc{nc}",
                   config=vars(args), reinit=True)

    train_loader, val_loader, test_loader = make_loaders(args)

    model = VGG11UNet(num_classes=nc, in_channels=3,
                      dropout_p=args.dropout_p).to(device)
    load_backbone_from_classifier(model, CKPT_CLF, encoder_attr="encoder")

    # Loss function depends on num_classes
    if nc == 1:
        # Binary: foreground (trimap=0) vs rest
        criterion = nn.BCEWithLogitsLoss()
    else:
        # 3-class: {0=fg, 1=bg, 2=boundary}
        # Upweight boundary (rare) and foreground to prevent background dominance
        class_weights = torch.tensor([1.5, 1.0, 2.0], device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Freeze encoder for warmup
    for p in model.encoder.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.seg_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.seg_epochs, eta_min=1e-6)

    unfreeze_at = max(1, args.seg_epochs // 3)
    best_dice   = 0.0

    for epoch in range(1, args.seg_epochs + 1):

        if epoch == unfreeze_at:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.seg_lr * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.seg_epochs - epoch + 1, eta_min=1e-6)
            print(f"  [Epoch {epoch}] encoder unfrozen")

        # ---- Train ----
        model.train()
        t_loss = 0.0
        all_pred, all_tgt = [], []

        for imgs, _, _, masks in train_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)    # [B,H,W] int64, values {0,1,2}

            optimizer.zero_grad()
            logits = model(imgs)        # [B, nc, H, W]

            if nc == 1:
                target = (masks == 0).float()   # fg=1, rest=0
                loss   = criterion(logits.squeeze(1), target)
                preds  = (logits.squeeze(1) > 0).long()  # >0 means sigmoid>0.5
            else:
                loss  = criterion(logits, masks)
                preds = logits.argmax(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            t_loss += loss.item()
            all_pred.append(preds.detach().cpu())
            all_tgt.append(masks.cpu())

        t_loss /= len(train_loader)
        t_pred_all = torch.cat(all_pred)
        t_tgt_all  = torch.cat(all_tgt)
        t_m = seg_metrics(t_pred_all, t_tgt_all, nc if nc > 1 else 2)
        scheduler.step()

        # ---- Val ----
        model.eval()
        v_loss = 0.0
        all_pred, all_tgt = [], []
        with torch.no_grad():
            for imgs, _, _, masks in val_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                if nc == 1:
                    target = (masks == 0).float()
                    loss   = criterion(logits.squeeze(1), target)
                    preds  = (logits.squeeze(1) > 0).long()
                else:
                    loss  = criterion(logits, masks)
                    preds = logits.argmax(1)
                v_loss += loss.item()
                all_pred.append(preds.cpu())
                all_tgt.append(masks.cpu())

        v_loss /= len(val_loader)
        v_pred_all = torch.cat(all_pred)
        v_tgt_all  = torch.cat(all_tgt)
        v_m = seg_metrics(v_pred_all, v_tgt_all, nc if nc > 1 else 2)

        print(f"  Epoch {epoch:3d}/{args.seg_epochs} | "
              f"train loss={t_loss:.4f} dice={t_m['mean_dice']:.4f} px_acc={t_m['px_accuracy']:.4f} | "
              f"val   loss={v_loss:.4f} dice={v_m['mean_dice']:.4f} px_acc={v_m['px_accuracy']:.4f}")

        wandb_log({
            "epoch": epoch,
            "seg/lr":                    scheduler.get_last_lr()[0],
            "seg/train/loss":            t_loss,
            "seg/train/mean_dice":       t_m["mean_dice"],
            "seg/train/dice_fg":         t_m["dice_fg"],
            "seg/train/dice_bg":         t_m["dice_bg"],
            "seg/train/dice_boundary":   t_m["dice_boundary"],
            "seg/train/px_accuracy":     t_m["px_accuracy"],
            "seg/train/f1_macro":        t_m["f1_macro"],
            "seg/train/f1_micro":        t_m["f1_micro"],
            "seg/train/f1_weighted":     t_m["f1_weighted"],
            "seg/train/prec_macro":      t_m["prec_macro"],
            "seg/train/rec_macro":       t_m["rec_macro"],
            "seg/val/loss":              v_loss,
            "seg/val/mean_dice":         v_m["mean_dice"],
            "seg/val/dice_fg":           v_m["dice_fg"],
            "seg/val/dice_bg":           v_m["dice_bg"],
            "seg/val/dice_boundary":     v_m["dice_boundary"],
            "seg/val/px_accuracy":       v_m["px_accuracy"],
            "seg/val/f1_macro":          v_m["f1_macro"],
            "seg/val/f1_micro":          v_m["f1_micro"],
            "seg/val/f1_weighted":       v_m["f1_weighted"],
            "seg/val/prec_macro":        v_m["prec_macro"],
            "seg/val/rec_macro":         v_m["rec_macro"],
        }, args.use_wandb)

        if v_m["mean_dice"] > best_dice:
            best_dice = v_m["mean_dice"]
            save_checkpoint(CKPT_SEG, model, epoch, best_dice)

    # ---- Test ----
    ckpt = torch.load(CKPT_SEG, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    te_loss = 0.0
    all_pred, all_tgt = [], []
    with torch.no_grad():
        for imgs, _, _, masks in test_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            if nc == 1:
                target = (masks == 0).float()
                loss   = criterion(logits.squeeze(1), target)
                preds  = (logits.squeeze(1) > 0).long()
            else:
                loss  = criterion(logits, masks)
                preds = logits.argmax(1)
            te_loss += loss.item()
            all_pred.append(preds.cpu())
            all_tgt.append(masks.cpu())

    te_loss /= len(test_loader)
    te_pred  = torch.cat(all_pred)
    te_tgt   = torch.cat(all_tgt)
    te_m     = seg_metrics(te_pred, te_tgt, nc if nc > 1 else 2)
    print(f"  [TEST] loss={te_loss:.4f} dice={te_m['mean_dice']:.4f} "
          f"px_acc={te_m['px_accuracy']:.4f} f1_macro={te_m['f1_macro']:.4f}")
    wandb_log({
        "seg/test/loss":          te_loss,
        "seg/test/mean_dice":     te_m["mean_dice"],
        "seg/test/dice_fg":       te_m["dice_fg"],
        "seg/test/dice_bg":       te_m["dice_bg"],
        "seg/test/dice_boundary": te_m["dice_boundary"],
        "seg/test/px_accuracy":   te_m["px_accuracy"],
        "seg/test/f1_macro":      te_m["f1_macro"],
        "seg/test/f1_micro":      te_m["f1_mi" "cro"],
        "seg/test/f1_weighted":   te_m["f1_weighted"],
        "seg/test/prec_macro":    te_m["prec_macro"],
        "seg/test/rec_macro":     te_m["rec_macro"],
    }, args.use_wandb)

    if args.use_wandb:
        wandb.finish()
    return best_dice


# --------------------------------------------------------------------------- #
#  Args                                                                        #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment-2 Training")

    p.add_argument("--data_root",     type=str,   default="./data/oxford-iiit-pet/")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--task",          type=str,   default="clf",
                   choices=["all", "clf", "loc", "seg"])
    p.add_argument("-b", "--batch_size",    type=int,   default=64)
    p.add_argument("-dp", "--dropout_p",     type=float, default=0.5)
    p.add_argument("--seed",          type=int,   default=42)

    p.add_argument("--clf_lr",        type=float, default=1e-4)
    p.add_argument("--clf_epochs",    type=int,   default=70)

    p.add_argument("--loc_lr",        type=float, default=1e-3)
    p.add_argument("--loc_epochs",    type=int,   default=30)

    p.add_argument("--seg_lr",        type=float, default=1e-3)
    p.add_argument("--seg_epochs",    type=int,   default=30)
    p.add_argument("--seg_classes",   type=int,   default=3, choices=[1, 3],
                   help="1=binary fg/rest, 3=full trimap {fg,bg,boundary}")

    p.add_argument("--wandb_project", type=str,   default="DA6402-Assignment-2_v1")
    p.add_argument("--use_wandb",     action="store_true",
                   help="Enable Weights & Biases logging")

    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    args        = parse_args()
    set_seed(args.seed)
    args.device = get_device()

    print(f"Device : {args.device}")
    print(f"Task   : {args.task}")
    print(f"WandB  : {args.use_wandb}")
    print(f"Seed   : {args.seed}")
    print(f"seg_classes: {args.seg_classes}")

    if args.task in ("all", "clf"):
        train_classifier(args)
    if args.task in ("all", "loc"):
        train_localizer(args)
    if args.task in ("all", "seg"):
        train_segmentation(args)

    print("\nDone. Checkpoints saved:")
    for ck in [CKPT_CLF, CKPT_LOC, CKPT_SEG]:
        print(f"  {ck}  {'✓' if os.path.exists(ck) else '(not trained)'}")