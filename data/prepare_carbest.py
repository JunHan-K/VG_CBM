#!/usr/bin/env python3
"""
prepare_carbest.py
------------------
Creates the Car-Best dataset from Stanford Cars 196.

What this script does:
  1. Downloads Stanford Cars 196 via Hugging Face datasets (train + test splits)
  2. Filters to 31 visually distinct classes where a frozen ResNet-50 linear probe
     achieves >= 40% per-class accuracy (removes year-variant classes that are
     visually indistinguishable)
  3. Saves the result as an ImageFolder-compatible directory:
       data/stanford_cars_hf_carbest/
         train/  <class_name>/  *.jpg
         test/   <class_name>/  *.jpg
  4. Writes data/carbest_classes.json with the 31 class names

Requirements:
  pip install datasets huggingface_hub

Usage:
  python data/prepare_carbest.py --data_root ./data
  python data/prepare_carbest.py --data_root ./data --hf_cache ./data/hf_cache

The script will print progress and takes ~5-10 minutes on first run
(downloads ~1.8 GB).
"""

import os
import json
import shutil
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as tvm


# The 31 selected class names (filtered from Stanford Cars 196)
CARBEST_CLASSES = [
    "AM General Hummer SUV 2000",
    "Aston Martin V8 Vantage Convertible 2012",
    "Audi S5 Convertible 2012",
    "Bentley Continental Supersports Convertible 2012",
    "Buick Rainier SUV 2007",
    "Chevrolet Corvette ZR1 2012",
    "Chrysler PT Cruiser Convertible 2008",
    "Dodge Caravan Minivan 1997",
    "Dodge Ram Pickup 3500 Quad Cab 2009",
    "Dodge Sprinter Cargo Van 2009",
    "Ferrari 458 Italia Convertible 2012",
    "Ford Freestar Minivan 2007",
    "Ford F-150 Regular Cab 2007",
    "Ford E-Series Wagon Van 2012",
    "GMC Savana Van 2012",
    "Geo Metro Convertible 1993",
    "HUMMER H2 SUT Crew Cab 2009",
    "Honda Odyssey Minivan 2012",
    "Hyundai Tucson SUV 2012",
    "Jaguar XK XKR 2012",
    "Jeep Liberty SUV 2012",
    "Lamborghini Gallardo LP 570-4 Superleggera 2012",
    "Land Rover Range Rover SUV 2012",
    "MINI Cooper Roadster Convertible 2012",
    "Mercedes-Benz 300-Class Convertible 1993",
    "Mercedes-Benz Sprinter Van 2012",
    "Nissan NV Passenger Van 2012",
    "Ram C/V Cargo Van Minivan 2012",
    "Spyker C8 Convertible 2009",
    "Toyota 4Runner SUV 2012",
    "smart fortwo Convertible 2012",
]


def download_from_hf(hf_cache, split):
    """Download Stanford Cars split from Hugging Face."""
    from datasets import load_dataset
    print(f"Downloading Stanford Cars 196 ({split}) from Hugging Face...")
    ds = load_dataset("tanganke/stanford_cars", split=split, cache_dir=hf_cache)
    return ds


def get_class_names_from_hf(ds):
    """Extract class name list from the HF dataset."""
    return ds.features["label"].names


def filter_and_save(ds, class_names, target_classes, out_dir, split):
    """Copy images for target_classes into ImageFolder structure."""
    # Map target class name -> folder index in target_classes list
    target_set = set(target_classes)
    target_idx = {name: i for i, name in enumerate(target_classes)}

    out_path = Path(out_dir) / split
    out_path.mkdir(parents=True, exist_ok=True)

    # Create class folders
    for name in target_classes:
        safe = name.replace("/", "_").replace(" ", "_")
        (out_path / safe).mkdir(exist_ok=True)

    count = 0
    for i, sample in enumerate(ds):
        cls_name = class_names[sample["label"]]
        if cls_name not in target_set:
            continue
        safe = cls_name.replace("/", "_").replace(" ", "_")
        img = sample["image"]
        img.save(out_path / safe / f"{i:06d}.jpg")
        count += 1
        if (count) % 200 == 0:
            print(f"  {split}: {count} images saved...")

    print(f"  {split}: {count} images total for {len(target_classes)} classes")
    return count


def verify_with_linear_probe(out_dir, min_per_class_acc=0.40):
    """
    Optional verification: trains a frozen ResNet-50 linear probe on the saved
    data and checks that all classes exceed min_per_class_acc.
    Prints a warning if any class falls below the threshold.
    """
    val_tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    train_ds = ImageFolder(os.path.join(out_dir, 'train'), transform=val_tf)
    test_ds  = ImageFolder(os.path.join(out_dir, 'test'),  transform=val_tf)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nVerifying with linear probe on {device}...")

    # Backbone
    backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    extractor = nn.Sequential(*list(backbone.children())[:-1])
    for p in extractor.parameters(): p.requires_grad_(False)

    def extract(ds):
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                f = extractor(x.to(device)).squeeze(-1).squeeze(-1)
                feats.append(f.cpu())
                labels.append(y)
        return torch.cat(feats), torch.cat(labels)

    X_tr, y_tr = extract(train_ds)
    X_te, y_te = extract(test_ds)
    n_cls = len(train_ds.classes)

    head = nn.Linear(2048, n_cls).to(device)
    opt  = torch.optim.Adam(head.parameters(), lr=1e-3)
    for ep in range(30):
        head.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 256):
            idx = perm[i:i+256]
            logits = head(X_tr[idx].to(device))
            loss = F.cross_entropy(logits, y_tr[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()

    head.eval()
    with torch.no_grad():
        preds = head(X_te.to(device)).argmax(1).cpu()

    per_class_acc = {}
    for c in range(n_cls):
        mask = (y_te == c)
        if mask.sum() > 0:
            per_class_acc[train_ds.classes[c]] = (preds[mask] == c).float().mean().item()

    below = [(k, v) for k, v in per_class_acc.items() if v < min_per_class_acc]
    if below:
        print(f"\nWARNING: {len(below)} classes below {min_per_class_acc:.0%}:")
        for k, v in sorted(below, key=lambda x: x[1]):
            print(f"  {v:.1%}  {k}")
    else:
        overall = (preds == y_te).float().mean().item()
        print(f"\nAll classes above {min_per_class_acc:.0%}. Overall acc: {overall:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data',
                        help='Directory where the dataset will be saved')
    parser.add_argument('--hf_cache', default=None,
                        help='Hugging Face cache directory (default: HF default)')
    parser.add_argument('--skip_verify', action='store_true',
                        help='Skip linear probe verification step')
    args = parser.parse_args()

    out_dir = os.path.join(args.data_root, 'stanford_cars_hf_carbest')

    if os.path.isdir(out_dir):
        print(f"Dataset already exists at: {out_dir}")
        print("Delete the directory to re-download.")
        return

    # Download
    train_ds = download_from_hf(args.hf_cache, 'train')
    test_ds  = download_from_hf(args.hf_cache, 'test')
    class_names = get_class_names_from_hf(train_ds)

    print(f"\nStanford Cars 196: {len(class_names)} total classes")
    print(f"Filtering to {len(CARBEST_CLASSES)} Car-Best classes...\n")

    # Verify all target classes exist in the dataset
    missing = [c for c in CARBEST_CLASSES if c not in class_names]
    if missing:
        print("WARNING: The following target classes were not found in the dataset:")
        for m in missing:
            print(f"  {m}")
        print("\nClass names in dataset (first 20):")
        for c in class_names[:20]:
            print(f"  {c}")
        return

    # Save filtered dataset
    filter_and_save(train_ds, class_names, CARBEST_CLASSES, out_dir, 'train')
    filter_and_save(test_ds,  class_names, CARBEST_CLASSES, out_dir, 'test')

    # Save class definitions
    classes_json = os.path.join(args.data_root, 'carbest_classes.json')
    with open(classes_json, 'w') as f:
        json.dump({'class_names': CARBEST_CLASSES}, f, indent=2)
    print(f"\nClass definitions saved to: {classes_json}")
    print(f"Dataset saved to: {out_dir}")

    # Verify
    if not args.skip_verify:
        verify_with_linear_probe(out_dir)

    print("\nDone. You can now run:")
    print(f"  python train.py --dataset car_best --data_root {args.data_root} --run_dir ./runs/car")


if __name__ == '__main__':
    main()
