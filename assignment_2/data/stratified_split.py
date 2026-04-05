import pathlib
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def get_stratified_split(ann_file: str, val_frac: float = 0.1, seed: int = 42,) -> Tuple[List[Tuple[str,int,int,int]], List[Tuple[str,int,int,int]]]:
    path = pathlib.Path(ann_file)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    records, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            image_id = parts[0]
            cls_id   = int(parts[1])
            species  = int(parts[2])
            breed_id = int(parts[3])
            records.append((image_id, cls_id, species, breed_id))
            labels.append(cls_id)

    labels = np.array(labels)
    sss    = StratifiedShuffleSplit(n_splits=1, test_size=val_frac,
                                    random_state=seed)
    train_idx, val_idx = next(sss.split(labels, labels))

    train_records = [records[i] for i in train_idx]
    val_records   = [records[i] for i in val_idx]

    # Sanity checks
    train_ids_set = {r[0] for r in train_records}
    val_ids_set   = {r[0] for r in val_records}
    assert len(train_ids_set & val_ids_set) == 0, \
        "BUG: train/val image ID overlap detected"

    train_cls = np.array([r[1] for r in train_records])
    val_cls   = np.array([r[1] for r in val_records])
    n_classes  = labels.max()
    assert np.all(np.bincount(train_cls-1, minlength=n_classes) > 0), \
        "Some classes missing from train split"
    assert np.all(np.bincount(val_cls-1, minlength=n_classes) > 0), \
        "Some classes missing from val split"

    return train_records, val_records


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann_file', default='data/oxford-iiit-pet/annotations/trainval.txt')
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--seed',     type=int,   default=42)
    args = ap.parse_args()

    train_recs, val_recs = get_stratified_split(
        args.ann_file, args.val_frac, args.seed)

    import numpy as np
    t_cls = np.array([r[1] for r in train_recs])
    v_cls = np.array([r[1] for r in val_recs])
    print(f"Train: {len(train_recs)} images | "
          f"per-class min={np.bincount(t_cls-1).min()} "
          f"max={np.bincount(t_cls-1).max()}")
    print(f"Val:   {len(val_recs)}  images | "
          f"per-class min={np.bincount(v_cls-1).min()} "
          f"max={np.bincount(v_cls-1).max()}")
    print("No train/val ID overlap ✓")