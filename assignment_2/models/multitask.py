"""Unified multi-task model
"""
import os
import torch
import torch.nn as nn

from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet

def _load_sd(path, device="cpu"):
    """Load checkpoint; handles both raw state_dict and {'state_dict': ...} wrappers."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _maybe_download(path, gdrive_id):
    if os.path.exists(path):
        return
    try:
        import gdown
        print(f"  Downloading {path} ...")
        gdown.download(id=gdrive_id, output=path, quiet=False)
    except Exception as e:
        print(f"  gdown failed for {path}: {e}")


class MultiTaskPerceptionModel(nn.Module):
    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet_3.pth",
    ):
        super().__init__()
        
        self.image_size = 224
        _IDS = {
            "classifier": "1UbIKWiy7j9M9FgV4MhG_KgePEJNO7B3n",
            "localizer":  "1K_Jy5WR4XxR2PF-idCY3QjHcPR1qqz41",
            "unet3":      "1QCukqGob1aKBS5qPiaCuYXPmgaHJxUza",
        }
        _maybe_download(classifier_path, _IDS["classifier"])
        _maybe_download(localizer_path,  _IDS["localizer"])
        _maybe_download(unet_path,       _IDS["unet3"])

        # --- Separate models (no shared backbone) ---
        self.clf_model = VGG11Classifier(num_classes=num_breeds, dropout_p=0.5)
        self.loc_model = VGG11Localizer(in_channels=in_channels, dropout_p=0.5, freeze_backbone=False)
        self.seg_model = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        # --- Load full checkpoints ---
        self._load_model(self.clf_model, classifier_path, prefix="classifier")
        self._load_model(self.loc_model, localizer_path,  prefix="localizer")
        self._load_model(self.seg_model, unet_path,       prefix="unet")

    def _load_model(self, model, path, prefix=""):
        if not os.path.exists(path):
            print(f"  [{prefix}] not found: {path} — random init")
            return
        sd = _load_sd(path)

        # handle wrapped checkpoints
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        # strip common prefixes
        cleaned = {}
        for k, v in sd.items():
            k = k.replace("module.", "")
            cleaned[k] = v

        miss, unexp = model.load_state_dict(cleaned, strict=False)
        print(f"  [{prefix}] loaded | missing={len(miss)} unexpected={len(unexp)}")

    def forward(self, x: torch.Tensor):
        # Classification logits
        cls_logits = self.clf_model(x)

        # Localization boxes (normalized [0,1] if your reg head uses sigmoid)
        bbox = self.loc_model(x) * float(self.image_size)

        # Segmentation logits (B,C,H,W) — required by autograder
        seg = self.seg_model(x)
        seg += 1
        return {
            "classification": cls_logits,
            "localization":   bbox,
            "segmentation":   seg,
        }
