"""Unified multi-task model
"""
import os
import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.localization import RegressionHead
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
        # after loading backbone
        
        self.image_size = 224
        _IDS = {
            "classifier": "1UbIKWiy7j9M9FgV4MhG_KgePEJNO7B3n",
            "localizer":  "1CSOBMojczbiXYHcZP5Of3p3L5Dy2LYOb",
            "unet3":      "1QCukqGob1aKBS5qPiaCuYXPmgaHJxUza",
        }
        _maybe_download(classifier_path, _IDS["classifier"])
        _maybe_download(localizer_path,  _IDS["localizer"])
        _maybe_download(unet_path,       _IDS["unet3"])

        # models (shared backbone) 
        self.backbone = VGG11Encoder(
            in_channels=in_channels,
            num_classes=num_breeds,
            dropout_p=0.5,
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

        self.regression_head = RegressionHead()

        _unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        self.center = _unet.center
        self.dec5   = _unet.dec5
        self.dec4   = _unet.dec4
        self.dec3   = _unet.dec3
        self.dec2   = _unet.dec2
        self.dec1   = _unet.dec1
        self.final_up = _unet.final_up
        self.final_conv  = _unet.final_conv   
        del _unet

        # Load full checkpoints
        self._load_backbone(classifier_path)
        self._load_regression_head(localizer_path)
        self._load_seg_decoder(unet_path)


    # Private weight loaders 
    def _load_backbone(self, path):
        if not os.path.exists(path):
            print(f"  [backbone]  not found: {path} — random init"); return
        sd = _load_sd(path)
        remapped = {
            k[len("classifier."):]: v
            for k, v in sd.items()
            if k.startswith("classifier.")
        }
        miss, unexp = self.backbone.load_state_dict(remapped, strict=False)
        print(f"  [backbone]  loaded {len(remapped)} tensors | "
              f"missing={len(miss)} unexpected={len(unexp)}")


    def _load_regression_head(self, path):
        if not os.path.exists(path):
            print(f"  [reg_head]  not found: {path} — random init")
            return

        sd = torch.load(path, map_location="cpu", weights_only=True)

        if "regression_head" in sd:
            state_dict = sd["regression_head"]
        else:
            state_dict = sd  # fallback

        miss, unexp = self.regression_head.load_state_dict(state_dict, strict=False)

        print(f"  [reg_head]  loaded {len(state_dict)} tensors | "
            f"missing={len(miss)} unexpected={len(unexp)}")


    def _load_seg_decoder(self, path):
        if not os.path.exists(path):
            print(f"  [seg_dec]   not found: {path} — random init"); return
        sd = _load_sd(path)
        decoder_prefixes = ("center.", "dec5.", "dec4.", "dec3.", "dec2.", "dec1.", "final_up.", "final_conv.")
        remapped = {
            k: v for k, v in sd.items()
            if any(k.startswith(p) for p in decoder_prefixes)
        }
        miss, unexp = self.load_state_dict(remapped, strict=False)
        decoder_miss = [m for m in miss if m.startswith(("center.", "dec", "final_"))]
        print(f"[seg_dec] missing decoder keys: {len(decoder_miss)}")

        print(f"  [seg_dec]   loaded {len(remapped)} tensors | "
              f"missing={len(miss)} unexpected={len(unexp)}")


    #  Forward 
    def forward(self, x: torch.Tensor):
        # One encoder pass — produces classification logits + skip features
        cls_logits, features = self.backbone(x, return_features=True) 
        f1 = features['f1'] 
        f2 = features['f2'] 
        f3 = features['f3'] 
        f4 = features['f4'] 
        f5 = features['f5'] 

        # Task 2: localisation
        bbox = self.regression_head(f5) * float(self.image_size)   

        # Task 3: segmentation decoder
        center = self.center(f5)
        dec5   = self.dec5(center, f5)
        dec4   = self.dec4(dec5, f4)
        dec3   = self.dec3(dec4, f3)
        dec2   = self.dec2(dec3, f2)
        dec1   = self.dec1(dec2, f1)
        dec1   = self.final_up(dec1)
        seg    = self.final_conv(dec1)                
        seg += 1

        return {
            "classification": cls_logits,
            "localization":   bbox,
            "segmentation":   seg,
        }
