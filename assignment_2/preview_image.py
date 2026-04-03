import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.insert(0, '.')
from data.pets_dataset import OxfordIIITPetDataset, get_val_transforms, get_train_transforms
from torch.utils.data import DataLoader

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])
MASK_COLORS = np.array([[0,255,0],[255,0,0],[0,0,255]], dtype=np.uint8)  # fg=green, bg=red, boundary=blue

ds     = OxfordIIITPetDataset(root='data/oxford-iiit-pet/', split='trainval', transform=get_train_transforms())
loader = DataLoader(ds, batch_size=1, shuffle=True)

n = 8
fig, axes = plt.subplots(3, n, figsize=(n*3, 9))

for i, (imgs, labels, bboxes, masks) in enumerate(loader):
    if i == n: break

    img = np.clip(imgs[0].permute(1,2,0).numpy() * STD + MEAN, 0, 1)
    H, W = img.shape[:2]
    cx, cy, bw, bh = bboxes[0].numpy()
    x1, y1 = (cx - bw/2) * W, (cy - bh/2) * H

    # row 0: image + bbox
    axes[0, i].imshow(img)
    axes[0, i].add_patch(patches.Rectangle((x1, y1), bw*W, bh*H, linewidth=2, edgecolor='red', facecolor='none'))
    axes[0, i].set_title(f"cls={labels[0].item()}", fontsize=8)
    axes[0, i].axis('off')

    # row 1: trimap mask (colour coded)
    mask_rgb = MASK_COLORS[masks[0].numpy()]
    axes[1, i].imshow(mask_rgb)
    axes[1, i].set_title("mask", fontsize=8)
    axes[1, i].axis('off')

    # row 2: image overlaid with mask (alpha blend)
    overlay = (img * 255).astype(np.uint8).copy()
    alpha = 0.45
    for cls_idx, color in enumerate(MASK_COLORS):
        overlay[masks[0].numpy() == cls_idx] = (
            (1 - alpha) * overlay[masks[0].numpy() == cls_idx] + alpha * color
        ).astype(np.uint8)
    axes[2, i].imshow(overlay)
    axes[2, i].add_patch(patches.Rectangle((x1, y1), bw*W, bh*H, linewidth=1.5, edgecolor='yellow', facecolor='none'))
    axes[2, i].set_title("overlay", fontsize=8)
    axes[2, i].axis('off')

from matplotlib.patches import Patch
fig.legend(handles=[Patch(color='green',label='fg'), Patch(color='red',label='bg'), Patch(color='blue',label='boundary')],
           loc='lower center', ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig("bbox_preview.png", dpi=100, bbox_inches='tight')
plt.close()
print("Saved bbox_preview.png")