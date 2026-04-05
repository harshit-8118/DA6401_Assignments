"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .localization import RegressionHead
from .segmentation import VGG11UNet


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

        #  Download weights if not present 
        _IDS = {
            "classifier": "1UbIKWiy7j9M9FgV4MhG_KgePEJNO7B3n",
            "localizer":  "1uKtLBJ92pmoeYu1G9je86gPfMETj7WtX",
            "unet1":      "1XZznmSzs_0S5H3UwFHU5tEsR7Ay3twhv",
            "unet3":      "1GYbIyk7nxF_XAZHDJK14yf43uG_xE9O6",
        }
        unet_id = _IDS["unet3"] if seg_classes == 3 else _IDS["unet1"]
        _maybe_download(classifier_path, _IDS["classifier"])
        _maybe_download(localizer_path,  _IDS["localizer"])
        _maybe_download(unet_path,       unet_id)

        #  Shared VGG11 backbone 
        self.backbone = VGG11Encoder(
            in_channels=in_channels,
            num_classes=num_breeds,
            dropout_p=0.5,
        )

        #  Localizer head  (operates on f5 from backbone) 
        self.regression_head = RegressionHead()

        #  Segmentation decoder  (shares backbone skip features) 
        _unet = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)
        self.center = _unet.center
        self.dec5   = _unet.dec5
        self.dec4   = _unet.dec4
        self.dec3   = _unet.dec3
        self.dec2   = _unet.dec2
        self.dec1   = _unet.dec1
        self.final  = _unet.final_conv   
        del _unet

        #  Load pretrained weights 
        self._load_backbone(classifier_path)
        self._load_regression_head(localizer_path)
        self._load_seg_decoder(unet_path)

    #  Private weight loaders 

    def _load_backbone(self, path):
        if not os.path.exists(path):
            print(f"  [backbone]  not found: {path} — random init"); return
        sd = _load_sd(path)
        # classifier.pth keys: "classifier.enc1.0.0.weight" etc.
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
            print(f"  [reg_head]  not found: {path} — random init"); return
        sd = _load_sd(path)
        # localizer.pth keys: "regression_head.pool.*", "regression_head.fc.*"
        remapped = {
            k[len("regression_head."):]: v
            for k, v in sd.items()
            if k.startswith("regression_head.")
        }
        miss, unexp = self.regression_head.load_state_dict(remapped, strict=False)
        print(f"  [reg_head]  loaded {len(remapped)} tensors | "
              f"missing={len(miss)} unexpected={len(unexp)}")

    def _load_seg_decoder(self, path):
        if not os.path.exists(path):
            print(f"  [seg_dec]   not found: {path} — random init"); return
        sd = _load_sd(path)
        # unet.pth keys include "encoder.*" (skip) and decoder keys
        decoder_prefixes = ("center.", "dec4.", "dec3.", "dec2.", "dec1.", "final.")
        remapped = {
            k: v for k, v in sd.items()
            if any(k.startswith(p) for p in decoder_prefixes)
        }
        miss, unexp = self.load_state_dict(remapped, strict=False)
        print(f"  [seg_dec]   loaded {len(remapped)} tensors | "
              f"missing={len(miss)} unexpected={len(unexp)}")

    #  Forward 

    def forward(self, x: torch.Tensor):
        # One encoder pass — produces classification logits + skip features
        cls_logits, features = self.backbone(x, return_features=True)
        f1 = features['f1']   # [B,  64, H/2,  W/2 ]
        f2 = features['f2']   # [B, 128, H/4,  W/4 ]
        f3 = features['f3']   # [B, 256, H/8,  W/8 ]
        f4 = features['f4']   # [B, 512, H/16, W/16]
        f5 = features['f5']   # [B, 512, H/32, W/32]

        # Task 2: localisation
        bbox = self.regression_head(f5)          # [B, 4]

        # Task 3: segmentation decoder
        center = self.center(f5)
        dec5   = self.dec5(torch.cat([center, f5], dim=1))
        dec4   = self.dec4(torch.cat([dec5, f4], dim=1))
        dec3   = self.dec3(torch.cat([dec4,   f3], dim=1))
        dec2   = self.dec2(torch.cat([dec3,   f2], dim=1))
        dec1   = self.dec1(torch.cat([dec2,   f1], dim=1))
        seg    = self.final(dec1)                # [B, seg_classes, H, W]

        return {
            "classification": cls_logits,
            "localization":   bbox,
            "segmentation":   seg,
        }