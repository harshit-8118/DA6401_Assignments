import torch 
import os 
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data.pets_dataset import OxfordIIITPetDataset, get_val_transforms
from torch.utils.data import DataLoader
from models.classification import VGG11Classifier
from sklearn.metrics import accuracy_score, f1_score

# Constants
CKPT_CLF   = os.path.join("checkpoints", "classifier.pth")
IMG_SIZE   = 224
NUM_BREEDS = 37

def get_breed_map(data_root):
    """Parses list.txt to create a mapping of Class ID to Breed Name."""
    list_path = os.path.join(data_root, "annotations", "list.txt")
    breed_map = {}
    
    if not os.path.exists(list_path):
        print(f"Warning: {list_path} not found. Using generic names.")
        return {i: f"Breed_{i}" for i in range(NUM_BREEDS)}

    with open(list_path, "r") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split()
            if len(parts) < 4: continue
            
            # The 'class_id' in this dataset is typically 1-indexed.
            image_name = parts[0]
            class_id = int(parts[1]) - 1 
            
            # Extract breed name from image_name (e.g., 'Abyssinian_100' -> 'Abyssinian')
            breed_name = "_".join(image_name.split("_")[:-1])
            
            if class_id not in breed_map:
                breed_map[class_id] = breed_name
                
    return breed_map

def run_test_loader(args, model, device, breed_map):
    test_ds = OxfordIIITPetDataset(
        root      = args.data_root,
        split     = "test",
        transform = get_val_transforms(IMG_SIZE),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model.eval()
    te_preds, te_labels = [], []
    
    print(f"Evaluating on {len(test_ds)} images...")
    with torch.no_grad():
        for imgs, labels, _, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            te_preds.extend(logits.argmax(1).cpu().tolist())
            te_labels.extend(labels.tolist())
    
    acc = accuracy_score(te_labels, te_preds)
    f1 = f1_score(te_labels, te_preds, average="macro")
    print(f"\n[TEST RESULTS] Acc: {acc:.4f} | F1: {f1:.4f}")

def run_single_inference(args, model, device, breed_map):
    if not args.image_path or not os.path.exists(args.image_path):
        print("Error: Provide valid --image_path")
        return

    raw_img = Image.open(args.image_path).convert('RGB')
    transform = get_val_transforms(IMG_SIZE)
    
    # FIX: Pass empty bboxes and labels to satisfy Albumentations requirements
    transformed = transform(
        image=np.array(raw_img),
        bboxes=[],
        bbox_labels=[]
    )
    # Albumentations expects numpy
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        conf = probs[0][pred_idx].item()

    breed_name = breed_map.get(pred_idx, "Unknown")
    print(f"\n[INFERENCE]\nPredicted: {breed_name} (ID: {pred_idx})\nConfidence: {conf:.2%}")
    
    return breed_name, conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["test", "single"], default="test")
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--data_root", type=str, default="./data/oxford-iiit-pet/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)

    # Load Breed Names
    breed_map = get_breed_map(args.data_root)

    # Initialize Model
    model = VGG11Classifier(num_classes=NUM_BREEDS).to(device)
    if os.path.exists(CKPT_CLF):
        ckpt = torch.load(CKPT_CLF, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
    
    if args.mode == "test":
        run_test_loader(args, model, device, breed_map)
    else:
        run_single_inference(args, model, device, breed_map)