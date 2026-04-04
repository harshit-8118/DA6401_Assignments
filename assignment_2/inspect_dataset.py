import argparse
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def parse_annotation_file(path: Path):
    """Parse a trainval.txt / test.txt annotation file.

    Each line: <image_id>  <class_id(1-37)>  <species(1=cat,2=dog)>  <breed_id>

    Returns list of dicts.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            records.append({
                "image_id":  parts[0],
                "class_id":  int(parts[1]),          # 1-37
                "species":   int(parts[2]),           # 1=cat, 2=dog
                "breed_id":  int(parts[3]),
                "breed_name": " ".join(
                    w.title() for w in parts[0].rsplit("_", 1)[0].split("_")
                ),
            })
    return records


def image_size(path: Path):
    """Return (H, W) of an image/trimap, or None if file missing."""
    if not path.exists():
        return None
    try:
        with Image.open(path) as img:
            w, h = img.size
        return h, w
    except Exception:
        return None


def parse_xml_bbox(path: Path):
    """Return (xmin, ymin, xmax, ymax) from XML, or None."""
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
        bb = root.find(".//bndbox")
        if bb is None:
            return None
        return (
            float(bb.find("xmin").text),
            float(bb.find("ymin").text),
            float(bb.find("xmax").text),
            float(bb.find("ymax").text),
        )
    except Exception:
        return None


def fmt(val, unit="px"):
    return f"{val:.1f} {unit}" if val is not None else "N/A"


# --------------------------------------------------------------------------- #
#  Main audit                                                                  #
# --------------------------------------------------------------------------- #

def audit(data_root: str):
    root       = Path(data_root) / "oxford-iiit-pet"
    imgs_dir   = root / "images"
    masks_dir  = root / "annotations" / "trimaps"
    xmls_dir   = root / "annotations" / "xmls"
    tv_txt     = root / "annotations" / "trainval.txt"
    te_txt     = root / "annotations" / "test.txt"

    splits = {}
    if tv_txt.exists():
        splits["trainval"] = parse_annotation_file(tv_txt)
    else:
        print(f"WARNING: {tv_txt} not found")
    if te_txt.exists():
        splits["test"] = parse_annotation_file(te_txt)
    else:
        print(f"WARNING: {te_txt} not found")

    if not splits:
        print("No annotation files found. Check --data_root.")
        return

    all_records = [r for recs in splits.values() for r in recs]

    # ------------------------------------------------------------------ #
    # 1. Breed / species / class table                                    #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("BREED / SPECIES / CLASS TABLE")
    print("="*70)
    print(f"{'Class':>5}  {'Species':>8}  {'Breed Name':<35}  {'Train+Val':>9}  {'Test':>6}")
    print("-"*70)

    # collect per-class counts per split
    class_info = {}
    for split_name, records in splits.items():
        for r in records:
            cid = r["class_id"]
            if cid not in class_info:
                class_info[cid] = {
                    "breed_name": r["breed_name"],
                    "species":    "cat" if r["species"] == 1 else "dog",
                    "trainval":   0,
                    "test":       0,
                }
            class_info[cid][split_name] = class_info[cid].get(split_name, 0) + 1

    for cid in sorted(class_info):
        info = class_info[cid]
        sp   = info["species"]
        print(f"  {cid:3d}  {sp:>8}  {info['breed_name']:<35}  "
              f"{info.get('trainval', 0):>9}  {info.get('test', 0):>6}")

    print(f"\n  Total classes : {len(class_info)}")
    print(f"  Cats          : {sum(1 for i in class_info.values() if i['species']=='cat')}")
    print(f"  Dogs          : {sum(1 for i in class_info.values() if i['species']=='dog')}")

    # ------------------------------------------------------------------ #
    # 2. Missing file counts per split                                    #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("MISSING FILE REPORT (per split)")
    print("="*70)

    for split_name, records in splits.items():
        missing_img  = []
        missing_mask = []
        missing_xml  = []

        for r in records:
            iid = r["image_id"]
            if not (imgs_dir  / f"{iid}.jpg").exists(): missing_img.append(iid)
            if not (masks_dir / f"{iid}.png").exists(): missing_mask.append(iid)
            if not (xmls_dir  / f"{iid}.xml").exists(): missing_xml.append(iid)

        print(f"\n  [{split_name}]  total={len(records)}")
        print(f"    Missing images  : {len(missing_img)}")
        print(f"    Missing trimaps : {len(missing_mask)}")
        print(f"    Missing XMLs    : {len(missing_xml)}")

        if missing_img:
            print(f"    -> missing image examples : {missing_img[:10]}")
        if missing_mask:
            print(f"    -> missing trimap examples: {missing_mask[:10]}")
        if missing_xml:
            print(f"    -> missing XML examples   : {missing_xml[:10]}")
            if split_name == "test":
                print("    *** TEST XMLs are INTENTIONALLY absent in the official")
                print("        Oxford-IIIT Pet release. Use full-image bbox fallback")
                print("        (0.5, 0.5, 1.0, 1.0) for test samples in training.")

    # ------------------------------------------------------------------ #
    # 3. Image size statistics                                            #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("IMAGE SIZE STATISTICS")
    print("="*70)

    for split_name, records in splits.items():
        heights, widths = [], []
        for r in records:
            sz = image_size(imgs_dir / f"{r['image_id']}.jpg")
            if sz:
                heights.append(sz[0])
                widths.append(sz[1])

        if heights:
            print(f"\n  [{split_name}]  measured={len(heights)}/{len(records)}")
            print(f"    Height — min={min(heights)}  max={max(heights)}  "
                  f"mean={np.mean(heights):.1f}  std={np.std(heights):.1f}")
            print(f"    Width  — min={min(widths)}   max={max(widths)}   "
                  f"mean={np.mean(widths):.1f}  std={np.std(widths):.1f}")
            # aspect ratio
            ratios = [h/w for h,w in zip(heights,widths)]
            print(f"    Aspect H/W — min={min(ratios):.2f}  max={max(ratios):.2f}  "
                  f"mean={np.mean(ratios):.2f}")
            # non-square count
            non_sq = sum(1 for h,w in zip(heights,widths) if h != w)
            print(f"    Non-square  : {non_sq} ({100*non_sq/len(heights):.1f}%)")
        else:
            print(f"\n  [{split_name}]  no images found — check imgs_dir={imgs_dir}")

    # ------------------------------------------------------------------ #
    # 4. Trimap size statistics                                           #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("TRIMAP SIZE STATISTICS")
    print("="*70)

    for split_name, records in splits.items():
        heights, widths = [], []
        unique_vals_all = set()
        for r in records:
            path = masks_dir / f"{r['image_id']}.png"
            sz   = image_size(path)
            if sz:
                heights.append(sz[0])
                widths.append(sz[1])
            # check pixel values
            if path.exists():
                try:
                    arr = np.array(Image.open(path))
                    unique_vals_all.update(arr.ravel().tolist())
                except Exception:
                    pass

        if heights:
            print(f"\n  [{split_name}]  measured={len(heights)}/{len(records)}")
            print(f"    Height — min={min(heights)}  max={max(heights)}  "
                  f"mean={np.mean(heights):.1f}")
            print(f"    Width  — min={min(widths)}   max={max(widths)}   "
                  f"mean={np.mean(widths):.1f}")
            print(f"    Unique pixel values in trimaps : {sorted(unique_vals_all)}")
            meaning = {1: "foreground", 2: "background", 3: "boundary/not-classified"}
            for v in sorted(unique_vals_all):
                print(f"      {v} -> {meaning.get(v, 'unknown')}")
        else:
            print(f"\n  [{split_name}]  no trimaps found")

    # ------------------------------------------------------------------ #
    # 5. XML / bbox statistics (trainval only — test has no XMLs)        #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("XML BOUNDING BOX STATISTICS (trainval only)")
    print("="*70)

    tv_records = splits.get("trainval", [])
    bbox_ws, bbox_hs, bbox_cxs, bbox_cys = [], [], [], []
    xml_found = 0

    for r in tv_records:
        img_sz = image_size(imgs_dir / f"{r['image_id']}.jpg")
        bb     = parse_xml_bbox(xmls_dir / f"{r['image_id']}.xml")
        if bb and img_sz:
            xml_found += 1
            xmin, ymin, xmax, ymax = bb
            img_h, img_w = img_sz
            bbox_ws.append((xmax - xmin) / img_w)
            bbox_hs.append((ymax - ymin) / img_h)
            bbox_cxs.append(((xmin + xmax) / 2) / img_w)
            bbox_cys.append(((ymin + ymax) / 2) / img_h)

    if bbox_ws:
        print(f"\n  XMLs found: {xml_found}/{len(tv_records)}")
        print(f"  Normalised bbox width  — min={min(bbox_ws):.3f}  max={max(bbox_ws):.3f}  "
              f"mean={np.mean(bbox_ws):.3f}")
        print(f"  Normalised bbox height — min={min(bbox_hs):.3f}  max={max(bbox_hs):.3f}  "
              f"mean={np.mean(bbox_hs):.3f}")
        print(f"  cx (normalised)        — min={min(bbox_cxs):.3f}  max={max(bbox_cxs):.3f}  "
              f"mean={np.mean(bbox_cxs):.3f}")
        print(f"  cy (normalised)        — min={min(bbox_cys):.3f}  max={max(bbox_cys):.3f}  "
              f"mean={np.mean(bbox_cys):.3f}")
    else:
        print("  No XMLs found or no trainval split available.")

    # ------------------------------------------------------------------ #
    # 6. Summary + training recommendations                              #
    # ------------------------------------------------------------------ #
    print("\n" + "="*70)
    print("TRAINING RECOMMENDATIONS")
    print("="*70)

    tv_xml_missing = sum(
        1 for r in splits.get("trainval", [])
        if not (xmls_dir / f"{r['image_id']}.xml").exists()
    )
    te_xml_missing = sum(
        1 for r in splits.get("test", [])
        if not (xmls_dir / f"{r['image_id']}.xml").exists()
    )

    print(f"""
  XMLs missing in trainval : {tv_xml_missing}
  XMLs missing in test     : {te_xml_missing}
""")


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data",
                    help="Root directory containing oxford-iiit-pet/")
    args = ap.parse_args()
    audit(args.data_root)