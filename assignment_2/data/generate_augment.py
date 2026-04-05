import argparse
import os
import pathlib
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import albumentations as A


BORDER_REFLECT = 4   # BORDER_REFLECT_101 integer value
def policy_geometric(size=224):
    """Flip + mild crop + rotation. Preserves colour exactly."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(size=(size,size), scale=(0.60,1.0),
                            ratio=(0.85,1.18), p=1.0),
        A.Rotate(limit=35, border_mode=BORDER_REFLECT, p=0.8),
    ])

def policy_colour(size=224):
    """Colour/exposure variation. No spatial change."""
    return A.Compose([
        A.Resize(size, size),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                      hue=0.06, p=0.9),
        A.RandomGamma(gamma_limit=(70,130), p=0.5),
        A.CLAHE(clip_limit=3.0, p=0.4),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                   b_shift_limit=15, p=0.3),
    ])

def policy_combined(size=224):
    return A.Compose([
        A.RandomResizedCrop(size=(size,size),
                            scale=(0.6, 1.0),
                            ratio=(0.9, 1.1), p=1.0),

        A.HorizontalFlip(p=0.5),

        A.Affine(scale=(0.9,1.1),
                 translate_percent=(0.05,0.1),
                 rotate=(-15,15), p=0.6),

        A.OneOf([
            A.GaussianBlur(blur_limit=7),
            A.MotionBlur(blur_limit=7),
        ], p=0.4),

        A.CoarseDropout(max_holes=6,
                        max_height=32,
                        max_width=32,
                        p=0.5),

        A.ColorJitter(brightness=0.25,
                      contrast=0.25,
                      saturation=0.25,
                      hue=0.05, p=0.6),
    ])

def policy_degradation(size=224):
    """Blur / noise / compression — simulates real-world photo quality."""
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(5,9)),
            A.MotionBlur(blur_limit=(5,9)),
            A.MedianBlur(blur_limit=(5,7))
        ], p=0.7),
        A.GaussNoise(p=0.5),
        A.ImageCompression(quality_range=(80,95), p=0.4),
    ])

POLICIES = [
    policy_geometric,
    policy_colour,
    policy_combined,
    policy_degradation,
]

def parse_ann_file(path: pathlib.Path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            records.append(line.split())   # [image_id, cls, species, breed]
    return records


def parse_xml_bbox(xml_path: pathlib.Path, img_w: int, img_h: int
                   ) -> Optional[Tuple[float,float,float,float]]:
    if not xml_path.exists():
        return None
    try:
        bb = ET.parse(xml_path).getroot().find('.//bndbox')
        if bb is None: return None
        xmin = max(0., min(float(bb.find('xmin').text), img_w))
        ymin = max(0., min(float(bb.find('ymin').text), img_h))
        xmax = max(0., min(float(bb.find('xmax').text), img_w))
        ymax = max(0., min(float(bb.find('ymax').text), img_h))
        cx = ((xmin+xmax)/2) / img_w
        cy = ((ymin+ymax)/2) / img_h
        w  = (xmax-xmin) / img_w
        h  = (ymax-ymin) / img_h
        return (cx, cy, w, h)
    except Exception:
        return None

def generate(data_root: str, copies: int = 4, img_size: int = 224,
             seed: int = 42):
    np.random.seed(seed)
    root      = pathlib.Path(data_root)
    imgs_dir  = root / 'images'
    masks_dir = root / 'annotations' / 'trimaps'
    xmls_dir  = root / 'annotations' / 'xmls'
    ann_file  = root / 'annotations' / 'trainval.txt'

    out_imgs  = root / 'images_aug';          out_imgs.mkdir(exist_ok=True)
    out_masks = root / 'annotations' / 'trimaps_aug'
    out_masks.mkdir(exist_ok=True)

    records   = parse_ann_file(ann_file)
    aug_lines = []

    # Keep originals in the combined file (resized to img_size)
    resize_tfm = A.Compose([A.Resize(img_size, img_size)])
    for rec in records:
        image_id, cls_id, species, breed = rec
        src_img  = imgs_dir  / f'{image_id}.jpg'
        src_mask = masks_dir / f'{image_id}.png'
        if not src_img.exists():
            continue

        img_np  = np.array(Image.open(src_img).convert('RGB'))
        H0, W0  = img_np.shape[:2]
        mask_np = (np.array(Image.open(src_mask).convert('L'))
                   if src_mask.exists()
                   else np.ones((H0,W0), dtype=np.uint8))

        # Save resized original into images_aug
        out_img_path  = out_imgs  / f'{image_id}.jpg'
        out_mask_path = out_masks / f'{image_id}.png'
        if not out_img_path.exists():
            r = resize_tfm(image=img_np, mask=mask_np)
            Image.fromarray(r['image']).save(out_img_path, quality=95)
            Image.fromarray(r['mask']).save(out_mask_path)

        aug_lines.append(f'{image_id} {cls_id} {species} {breed}')

    print(f'Originals: {len(records)} records')

    # Generate augmented copies
    policies_used = POLICIES[:copies]
    total_aug = 0

    for rec in records:
        image_id, cls_id, species, breed = rec
        src_img  = imgs_dir  / f'{image_id}.jpg'
        src_mask = masks_dir / f'{image_id}.png'
        if not src_img.exists():
            continue

        img_np  = np.array(Image.open(src_img).convert('RGB'))
        H0, W0  = img_np.shape[:2]
        mask_np = (np.array(Image.open(src_mask).convert('L'))
                   if src_mask.exists()
                   else np.ones((H0,W0), dtype=np.uint8))

        for aug_idx, policy_fn in enumerate(policies_used, start=1):
            aug_id        = f'{image_id}_aug{aug_idx}'
            out_img_path  = out_imgs  / f'{aug_id}.jpg'
            out_mask_path = out_masks / f'{aug_id}.png'

            if out_img_path.exists() and out_mask_path.exists():
                aug_lines.append(f'{aug_id} {cls_id} {species} {breed}')
                total_aug += 1
                continue

            tfm = policy_fn(size=img_size)
            out = tfm(image=img_np, mask=mask_np)

            aug_img  = out['image']
            aug_mask = out['mask']

            # Resize to target if not already done by crop
            if aug_img.shape[:2] != (img_size, img_size):
                r = A.Compose([A.Resize(img_size, img_size)])(
                    image=aug_img, mask=aug_mask)
                aug_img, aug_mask = r['image'], r['mask']

            Image.fromarray(aug_img).save(out_img_path, quality=95)
            Image.fromarray(aug_mask.astype(np.uint8)).save(out_mask_path)
            aug_lines.append(f'{aug_id} {cls_id} {species} {breed}')
            total_aug += 1

        if (records.index(rec)+1) % 500 == 0:
            print(f'  {records.index(rec)+1}/{len(records)} done...')

    # Write combined annotation file
    out_ann = root / 'annotations' / 'trainval_aug.txt'
    with open(out_ann, 'w') as f:
        f.write('\n'.join(aug_lines) + '\n')

    print(f'\nDone.')
    print(f'  Original images  : {len(records)}')
    print(f'  Augmented copies : {total_aug}')
    print(f'  Total in trainval_aug.txt: {len(aug_lines)}')
    print(f'  Images saved to  : {out_imgs}')
    print(f'  Masks  saved to  : {out_masks}')
    print(f'  Ann file         : {out_ann}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='data/oxford-iiit-pet')
    ap.add_argument('--copies',    type=int, default=4,
                    help='augmented copies per image (1-4)')
    ap.add_argument('--img_size',  type=int, default=224)
    ap.add_argument('--seed',      type=int, default=42)
    args = ap.parse_args()
    generate(args.data_root, args.copies, args.img_size, args.seed)