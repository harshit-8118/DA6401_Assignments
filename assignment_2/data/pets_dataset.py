import pathlib
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset
import torch


BORDER_CONSTANT     = 0   # equivalent to BORDER_CONSTANT
BORDER_REFLECT_101  = 4   # equivalent to BORDER_REFLECT_101

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)



#  Transforms                                                                  #
def get_train_transforms(img_size: int = 224) -> A.Compose:
    bbox_p = A.BboxParams(
        format="yolo", label_fields=["bbox_labels"],
        clip=True, min_visibility=0.3, min_area=100
    )
    return A.Compose([
 
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.7, 1.0),       # was (0.5, 1.0) — too aggressive
            ratio=(0.85, 1.18),     # near-square crops only
            p=1.0,
        ),
 
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=BORDER_REFLECT_101, p=0.3),
 
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05,
            p=0.5,
        ),
 
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
 
    ], bbox_params=bbox_p)
 
 
def get_val_transforms(img_size: int = 224) -> A.Compose:
    """Deterministic: resize + normalize only."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="yolo", label_fields=["bbox_labels"],
        clip=True, min_visibility=0.3,
    ))


#  Helper: parse Pascal VOC XML bounding box                                   #


def _parse_bbox_xml(xml_path: pathlib.Path, img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
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

    if w < 0.01 or h < 0.01:
        return None   
    
    return (cx, cy, w, h)



#  Dataset                                                                     #
class OxfordIIITPetDataset(Dataset):
    TRIMAP_REMAP    = {1: 0, 2: 1, 3: 2}
    NUM_SEG_CLASSES = 3
    NUM_BREEDS      = 37
 
    def __init__(
        self,
        root: str,
        split: Optional[str] = None,
        records: Optional[List[Tuple]] = None,
        images_dir: Optional[pathlib.Path] = None,
        masks_dir:  Optional[pathlib.Path] = None,
        transform:  Optional[A.Compose] = None,
        img_size:   int = 224,
    ):
        self.root     = pathlib.Path(root)
        self.img_size = img_size
        self.xmls_dir = self.root / "annotations" / "xmls"
 
        if records is not None:
            # Way 2: explicit records (stratified split path)
            assert images_dir is not None and masks_dir is not None, \
                "images_dir and masks_dir required when passing records"
            self.images_dir  = pathlib.Path(images_dir)
            self.masks_dir   = pathlib.Path(masks_dir)
            self._image_ids  = [r[0] for r in records]
            self._labels     = [int(r[1]) - 1 for r in records]  # 0-indexed
            self.transform   = transform if transform is not None \
                               else get_val_transforms(img_size)
 
        elif split is not None:
            # Way 1: load from annotation file (for test set)
            assert split in ("trainval", "trainval_aug", "test"), \
                f"Unknown split: {split!r}"
            is_aug = split == "trainval_aug"
            self.images_dir = self.root / ("images_aug" if is_aug else "images")
            self.masks_dir  = self.root / "annotations" / \
                              ("trimaps_aug" if is_aug else "trimaps")
            ann_file = self.root / "annotations" / f"{split}.txt"
            if not ann_file.exists():
                raise FileNotFoundError(f"Annotation file not found: {ann_file}")
            self._image_ids, self._labels = [], []
            with open(ann_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    self._image_ids.append(parts[0])
                    self._labels.append(int(parts[1]) - 1)
            self.transform = transform if transform is not None \
                             else (get_train_transforms(img_size)
                                   if "trainval" in split
                                   else get_val_transforms(img_size))
        else:
            raise ValueError("Provide either split= or records=")
 
    def __len__(self):
        return len(self._image_ids)
 
    def __getitem__(self, idx):
        image_id = self._image_ids[idx]
        label    = self._labels[idx]
 
        pil_img  = Image.open(self.images_dir / f"{image_id}.jpg").convert("RGB")
        image    = np.array(pil_img)
        img_h, img_w = image.shape[:2]
 
        mask_path = self.masks_dir / f"{image_id}.png"
        if mask_path.exists():
            mask_raw = np.array(Image.open(mask_path).convert("L"))
        else:
            mask_raw = np.ones((img_h, img_w), dtype=np.uint8)
 
        mask = np.zeros_like(mask_raw, dtype=np.uint8)
        for raw_val, cls_idx in self.TRIMAP_REMAP.items():
            mask[mask_raw == raw_val] = cls_idx
 
        # XMLs only exist for original images, not aug copies
        xml_id   = image_id.replace("_aug1","").replace("_aug2","") \
                            .replace("_aug3","").replace("_aug4","")
        bbox     = _parse_bbox_xml(self.xmls_dir / f"{xml_id}.xml", img_w, img_h)
        if bbox is None:
            bbox = (0.5, 0.5, 1.0, 1.0)
 
        out = self.transform(
            image=image, mask=mask,
            bboxes=[bbox], bbox_labels=[label],
        )
 
        bbox_t = (torch.tensor(out["bboxes"][0], dtype=torch.float32)
                  if len(out["bboxes"]) > 0
                  else torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32))
 
        return (out["image"],
                torch.tensor(label, dtype=torch.long),
                bbox_t,
                out["mask"].long())



#  Smoke-test                                                                  #
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