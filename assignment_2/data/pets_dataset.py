"""Dataset for Oxford-IIIT Pet — multi-task loader.

Dependencies: torch, numpy, pillow, albumentations, wandb, scikit-learn only.
No opencv required — image loading uses PIL, border modes passed as plain integers.

Provides per-sample:
  image   : torch.Tensor [3, H, W]  float32, ImageNet-normalised
  label   : torch.Tensor []         int64,   breed index [0, 36]
  bbox    : torch.Tensor [4]        float32, (cx, cy, w, h) in [0, 1]
  mask    : torch.Tensor [H, W]     int64,   {0=fg, 1=bg, 2=boundary}

Directory layout (standard torchvision download):
  root/oxford-iiit-pet/
    images/          *.jpg
    annotations/
      trainval.txt / test.txt
      trimaps/       *.png
      xmls/          *.xml   (Pascal VOC bounding boxes)

Trimap remapping:  1->0 (foreground), 2->1 (background), 3->2 (boundary)
Bbox format:       XML (xmin,ymin,xmax,ymax) -> normalised (cx,cy,w,h)
"""

import pathlib
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
import torch


# --------------------------------------------------------------------------- #
#  Border mode constants (plain integers — opencv-compatible values)            #
# --------------------------------------------------------------------------- #

BORDER_CONSTANT     = 0   # equivalent to BORDER_CONSTANT
BORDER_REFLECT_101  = 4   # equivalent to BORDER_REFLECT_101


# --------------------------------------------------------------------------- #
#  ImageNet stats                                                               #
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
#  Transforms                                                                  #
# --------------------------------------------------------------------------- #

def get_train_transforms(img_size: int = 224) -> A.Compose:
    """Rigorous augmentation pipeline for training.

    Group 1 — Spatial/geometric (applied jointly to image + mask + bbox):
      RandomResizedCrop : partial crops and zoom variation (scale 50-100%)
      HorizontalFlip    : mirror views
      Affine            : rotation ±35°, shear ±20°, scale 0.8-1.2,
                          translation ±12% — covers shear + elongation tests
      Perspective       : off-axis viewpoint warp
      ElasticTransform  : non-rigid deformation
      GridDistortion    : additional non-linear spatial warp

    Group 2 — Photometric (image only):
      OneOf[BrightnessContrast / ColorJitter / HueSaturationValue]
      RandomGamma, CLAHE, RGBShift, Equalize

    Group 3 — Degradation (image only):
      OneOf[GaussianBlur / MotionBlur / MedianBlur]
      GaussNoise, ISONoise, ImageCompression, CoarseDropout (occlusion)
    """
    bbox_p = A.BboxParams(
        format="yolo",             # (cx, cy, w, h) normalised
        label_fields=["bbox_labels"],
        clip=True,
        min_visibility=0.3,
    )

    return A.Compose([

        # ================================================================
        # GROUP 1: Spatial / geometric
        # ================================================================

        # Simulate cropped / zoomed inputs; random aspect ratio 3:4 to 4:3
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.5, 1.0),
            ratio=(0.75, 1.33),
            p=1.0,
        ),

        A.HorizontalFlip(p=0.5),

        # Rotation + shear + scale + translation in one transform.
        # shear=±20° directly handles "shear" and "elongation" test cases.
        # border_mode=4 (REFLECT_101) avoids black border artifacts.
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent={"x": (-0.12, 0.12), "y": (-0.12, 0.12)},
            rotate=(-35, 35),
            shear=(-20, 20),
            border_mode=BORDER_REFLECT_101,
            p=0.75,
        ),

        # Perspective warp: simulates camera angle / viewpoint change
        A.Perspective(scale=(0.04, 0.14), p=0.35),

        # Elastic deformation: non-rigid warp (fur, body shape variation)
        A.ElasticTransform(
            alpha=50,
            sigma=6,
            border_mode=BORDER_REFLECT_101,
            p=0.25,
        ),

        # Grid distortion: another non-linear spatial warp
        A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.2),

        # ================================================================
        # GROUP 2: Photometric (image only)
        # ================================================================

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35),
            A.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.1),
            A.HueSaturationValue(
                hue_shift_limit=25, sat_shift_limit=45, val_shift_limit=35),
        ], p=0.85),

        A.RandomGamma(gamma_limit=(65, 140), p=0.35),

        # Local contrast enhancement — helps on low-contrast test images
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),

        # Histogram equalisation — handles unusual global exposure
        A.Equalize(p=0.1),

        # ================================================================
        # GROUP 3: Degradation / noise (image only)
        # ================================================================

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9)),    # defocus blur
            A.MotionBlur(blur_limit=(3, 11)),     # camera shake
            A.MedianBlur(blur_limit=7),           # impulse noise smoothing
        ], p=0.45),

        A.GaussNoise(p=0.35),

        # ISO noise: camera sensor noise with a colour component
        A.ISONoise(color_shift=(0.01, 0.06), intensity=(0.1, 0.5), p=0.25),

        # JPEG compression artefacts (common in internet-scraped images)
        A.ImageCompression(quality_range=(50, 95), p=0.25),

        # Coarse Dropout (Cutout): rectangular occlusion patches.
        # fill=0 keeps dropped region black; fill_mask leaves mask labels intact.
        A.CoarseDropout(
            num_holes_range=(4, 10),
            hole_height_range=(12, 36),
            hole_width_range=(12, 36),
            fill=0,
            p=0.35,
        ),

        # ================================================================
        # Normalise + convert to tensor
        # ================================================================
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),

    ], bbox_params=bbox_p)


def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Deterministic resize + normalise for validation / test."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["bbox_labels"],
        clip=True,
        min_visibility=0.3,
    ))


# --------------------------------------------------------------------------- #
#  Helper: parse Pascal VOC XML bounding box                                   #
# --------------------------------------------------------------------------- #

def _parse_bbox_xml(
    xml_path: pathlib.Path, img_w: int, img_h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Parse first <bndbox> and return (cx, cy, w, h) normalised to [0,1].

    Returns None if the file does not exist or has no bndbox element.
    """
    if not xml_path.exists():
        return None

    root = ET.parse(xml_path).getroot()
    bb   = root.find(".//bndbox")
    if bb is None:
        return None

    xmin = max(0.0, min(float(bb.find("xmin").text), img_w))
    ymin = max(0.0, min(float(bb.find("ymin").text), img_h))
    xmax = max(0.0, min(float(bb.find("xmax").text), img_w))
    ymax = max(0.0, min(float(bb.find("ymax").text), img_h))

    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h

    return (cx, cy, w, h)


# --------------------------------------------------------------------------- #
#  Dataset                                                                     #
# --------------------------------------------------------------------------- #

class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset.

    Args:
        root      : Root directory containing ``oxford-iiit-pet/``.
        split     : ``"trainval"`` or ``"test"``.
        transform : Albumentations Compose pipeline. Defaults to
                    get_train_transforms for trainval, get_val_transforms for test.
        img_size  : Square output size (default 224).
        download  : Download via torchvision if dataset not present.
    """

    TRIMAP_REMAP    = {1: 0, 2: 1, 3: 2}   # raw pixel value -> class index
    NUM_SEG_CLASSES = 3
    NUM_BREEDS      = 37

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform: Optional[A.Compose] = None,
        img_size: int = 224,
    ) -> None:
        assert split in ("trainval", "test"), \
            f"split must be 'trainval' or 'test', got '{split}'"

        self.root     = pathlib.Path(root)
        self.split    = split
        self.img_size = img_size

        self.images_dir = self.root / "images"
        self.masks_dir  = self.root / "annotations" / "trimaps"
        self.xmls_dir   = self.root / "annotations" / "xmls"
        ann_file        = self.root / "annotations" / f"{split}.txt"

        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                "Pass download=True or set up the dataset manually."
            )

        self._image_ids: List[str] = []
        self._labels:    List[int] = []

        with open(ann_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                self._image_ids.append(parts[0])
                self._labels.append(int(parts[1]) - 1)   # 0-indexed [0,36]

        if transform is not None:
            self.transform = transform
        elif split == "trainval":
            self.transform = get_train_transforms(img_size)
        else:
            self.transform = get_val_transforms(img_size)

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int):
        """Return (image, label, bbox, mask).

        image : FloatTensor [3, H, W]
        label : LongTensor  []         breed in [0, 36]
        bbox  : FloatTensor [4]        (cx, cy, w, h) in [0, 1]
        mask  : LongTensor  [H, W]     pixel class in {0, 1, 2}
        """
        image_id = self._image_ids[idx]
        label    = self._labels[idx]

        # ---- Load image via PIL -> HxWx3 uint8 numpy ----
        pil_img = Image.open(self.images_dir / f"{image_id}.jpg").convert("RGB")
        image   = np.array(pil_img)          # [H, W, 3] uint8
        img_h, img_w = image.shape[:2]

        # ---- Load trimap mask via PIL -> HxW uint8 numpy ----
        mask_path = self.masks_dir / f"{image_id}.png"
        if mask_path.exists():
            mask_raw = np.array(Image.open(mask_path).convert("L"))  # [H, W] uint8
        else:
            mask_raw = np.ones((img_h, img_w), dtype=np.uint8)       # fallback: all fg

        # Remap {1,2,3} -> {0,1,2} for contiguous class indices
        mask = np.zeros_like(mask_raw, dtype=np.uint8)
        for raw_val, cls_idx in self.TRIMAP_REMAP.items():
            mask[mask_raw == raw_val] = cls_idx

        # ---- Load bounding box from Pascal VOC XML ----
        bbox_cxcywh = _parse_bbox_xml(
            self.xmls_dir / f"{image_id}.xml", img_w, img_h)
        if bbox_cxcywh is None:
            print(f"xml missing: {image_id}.xml")
            bbox_cxcywh = (0.5, 0.5, 1.0, 1.0)   # full-image fallback

        # ---- Apply albumentations (image + mask + bbox jointly) ----
        transformed = self.transform(
            image=image,
            mask=mask,
            bboxes=[bbox_cxcywh],
            bbox_labels=[label],
        )

        image_t = transformed["image"]                   # FloatTensor [3, H, W]
        mask_t  = transformed["mask"].long()             # LongTensor  [H, W]
        bboxes  = transformed["bboxes"]

        # If bbox was fully clipped out of frame, use full-image fallback
        bbox_t = (
            torch.tensor(bboxes[0], dtype=torch.float32)
            if len(bboxes) > 0
            else torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
        )

        return image_t, torch.tensor(label, dtype=torch.long), bbox_t, mask_t


# --------------------------------------------------------------------------- #
#  Smoke-test                                                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys
    import os 
    from torch.utils.data import DataLoader

    # Get the directory where pets_dataset.py actually lives
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default to the folder sitting right next to this script
    default_path = os.path.join(current_file_dir, "oxford-iiit-pet")

    root = sys.argv[1] if len(sys.argv) > 1 else default_path

    try:
        ds = OxfordIIITPetDataset(root=root, split="trainval")
    except FileNotFoundError:
        print(f"Dataset not found at '{root}'.")
        print("Usage: python dataset.py /path/to/data")
        sys.exit(0)

    print(f"Dataset size: {len(ds)}")
    img, label, bbox, mask = ds[0]
    print(f"  image : {img.shape}  {img.dtype}")
    print(f"  label : {label}  {label.dtype}")
    print(f"  bbox  : {bbox}  {bbox.dtype}")
    print(f"  mask  : {mask.shape}  {mask.dtype}  unique={mask.unique().tolist()}")

    assert img.shape  == (3, 224, 224)
    assert bbox.shape == (4,)
    assert 0.0 <= bbox.min() and bbox.max() <= 1.0
    assert mask.shape == (224, 224)
    assert set(mask.unique().tolist()).issubset({0, 1, 2})

    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    imgs, labels, bboxes, masks = next(iter(loader))
    print(f"\nBatch: imgs={imgs.shape} labels={labels.shape} "
          f"bboxes={bboxes.shape} masks={masks.shape}")
    print("All checks passed ✓")