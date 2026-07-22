"""
YOLO Dataset Formatter & Zipper
===============================
Copies raw images and auto-labels into a strict YOLOv8/v11 directory structure,
then zips it for easy upload to Google Colab.

Output structure:
datasets/annotated/v1.0/yolo_format/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── dataset.yaml
"""

import os
import yaml
import random
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
RAW_DIR = REPO_ROOT / "datasets" / "raw"
LABELS_DIR = REPO_ROOT / "datasets" / "annotated" / "v1.0" / "labels"
YOLO_DIR = REPO_ROOT / "datasets" / "annotated" / "v1.0" / "yolo_format"
ZIP_OUT = REPO_ROOT / "vision_yolo_dataset.zip"

ALL_CLASSES = ["bubble", "narration", "text", "sfx", "panel"]
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
SEED = 42

def main():
    if not LABELS_DIR.exists():
        print("Labels not found. Run auto_labeling and extract first.")
        return

    # Gather all images that have a corresponding label file
    print("Finding matched image-label pairs...")
    pairs = []
    
    for label_path in LABELS_DIR.rglob("*.txt"):
        stem = label_path.stem
        chapter = label_path.parent.name
        title = label_path.parent.parent.name
        
        # Reconstruct raw image path
        img_path_jpg = RAW_DIR / title / chapter / f"{stem}.jpg"
        img_path_png = RAW_DIR / title / chapter / f"{stem}.png"
        
        if img_path_jpg.exists():
            pairs.append((img_path_jpg, label_path))
        elif img_path_png.exists():
            pairs.append((img_path_png, label_path))

    print(f"Found {len(pairs)} matched pairs.")
    
    # Shuffle and split
    random.seed(SEED)
    random.shuffle(pairs)
    
    n_train = int(len(pairs) * SPLIT_RATIOS[0])
    n_val = int(len(pairs) * SPLIT_RATIOS[1])
    
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train+n_val],
        "test": pairs[n_train+n_val:]
    }
    
    # Create output directories
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
        
    for split in ["train", "val", "test"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    # Copy files
    print("\nCopying files into YOLO structure...")
    for split, split_pairs in splits.items():
        for img_path, label_path in tqdm(split_pairs, desc=f"{split}"):
            # We use a unique name by prepending title and chapter to avoid clashes
            title = img_path.parent.parent.name
            chapter = img_path.parent.name
            unique_stem = f"{title}_{chapter}_{img_path.stem}".replace(" ", "_")
            
            new_img = YOLO_DIR / "images" / split / f"{unique_stem}{img_path.suffix}"
            new_lbl = YOLO_DIR / "labels" / split / f"{unique_stem}.txt"
            
            shutil.copy2(img_path, new_img)
            shutil.copy2(label_path, new_lbl)

    # Write dataset.yaml
    yaml_content = {
        "path": "/content/yolo_format",  # Relative to Colab structure
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(ALL_CLASSES),
        "names": ALL_CLASSES
    }
    
    with open(YOLO_DIR / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print("\nZipping dataset for Colab...")
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
        
    # Zip the yolo_format directory
    with zipfile.ZipFile(ZIP_OUT, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(YOLO_DIR):
            for file in files:
                file_path = Path(root) / file
                # Store relative to yolo_format
                arcname = file_path.relative_to(YOLO_DIR.parent)
                zipf.write(file_path, arcname)

    size_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print(f"\nDone! Dataset ready for Colab.")
    print(f"Zip file: {ZIP_OUT} ({size_mb:.1f} MB)")
    print("Upload this zip to your Google Drive root.")

if __name__ == "__main__":
    main()
