"""
Dataset Split Generator — Vision Engine Phase 2
================================================
Splits annotated images into train / val / test sets.

Splits are stratified by manga title to ensure all titles appear in each split.

Output:
    datasets/splits/train.txt
    datasets/splits/val.txt
    datasets/splits/test.txt
    datasets/annotated/v1.0/dataset.yaml

Usage:
    python scripts/generate_splits.py
    python scripts/generate_splits.py --train 0.70 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = REPO_ROOT / "datasets" / "registry.yaml"
RAW_ROOT = REPO_ROOT / "datasets" / "raw"
ANNOTATED_ROOT = REPO_ROOT / "datasets" / "annotated" / "v1.0"
SPLITS_DIR = REPO_ROOT / "datasets" / "splits"

ALL_CLASSES = ["bubble", "narration", "text", "sfx", "panel"]

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_labeled_images(registry: dict) -> dict[str, list[Path]]:
    """Return images that have a corresponding non-empty label file, grouped by title."""
    labels_dir = ANNOTATED_ROOT / "labels"
    by_title: dict[str, list[Path]] = defaultdict(list)

    for source in registry["sources"]:
        title_id = source["id"]
        for chapter in source["chapters"]:
            chapter_dir = REPO_ROOT / chapter["path"]
            if not chapter_dir.exists():
                continue
            for img_path in sorted(chapter_dir.glob("*.jpg")):
                label_file = labels_dir / (img_path.stem + ".txt")
                if label_file.exists() and label_file.stat().st_size > 0:
                    by_title[title_id].append(img_path)

    return by_title


def split_list(items: list, train: float, val: float, seed: int) -> tuple[list, list, list]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_title = collect_labeled_images(registry)

    if not by_title:
        print("[ERROR] No labeled images found. Run auto_label.py first.")
        return

    total = sum(len(v) for v in by_title.values())
    print(f"\nLabeled images found: {total}")
    for title, imgs in by_title.items():
        print(f"  {title}: {len(imgs)}")

    all_train, all_val, all_test = [], [], []

    # Stratify by title
    for title, images in by_title.items():
        tr, va, te = split_list(images, train_ratio, val_ratio, seed=SEED)
        all_train.extend(tr)
        all_val.extend(va)
        all_test.extend(te)

    # Shuffle final lists
    rng = random.Random(SEED)
    rng.shuffle(all_train)
    rng.shuffle(all_val)
    rng.shuffle(all_test)

    # Write split files
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    def write_split(name: str, paths: list[Path]) -> None:
        out = SPLITS_DIR / f"{name}.txt"
        lines = [str(p.resolve()) for p in paths]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"  {name}.txt: {len(paths)} images → {out}")

    print("\nWriting split manifests...")
    write_split("train", all_train)
    write_split("val", all_val)
    write_split("test", all_test)

    # Write dataset.yaml for YOLO training
    dataset_yaml = {
        "path": str((ANNOTATED_ROOT).resolve()),
        "train": str((SPLITS_DIR / "train.txt").resolve()),
        "val":   str((SPLITS_DIR / "val.txt").resolve()),
        "test":  str((SPLITS_DIR / "test.txt").resolve()),
        "nc": len(ALL_CLASSES),
        "names": ALL_CLASSES,
        "version": "1.0",
        "description": "Vision Engine manga/manhwa annotation dataset v1.0",
    }
    yaml_out = ANNOTATED_ROOT / "dataset.yaml"
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    yaml_out.write_text(yaml.dump(dataset_yaml, default_flow_style=False), encoding="utf-8")
    print(f"\n  dataset.yaml written → {yaml_out}")

    print(f"\nSplit summary:")
    print(f"  Train: {len(all_train)} ({len(all_train)/total*100:.1f}%)")
    print(f"  Val:   {len(all_val)} ({len(all_val)/total*100:.1f}%)")
    print(f"  Test:  {len(all_test)} ({len(all_test)/total*100:.1f}%)")
    print(f"  Total: {total}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/val/test splits")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    parser.add_argument("--test",  type=float, default=0.15)
    args = parser.parse_args()
    run(args.train, args.val, args.test)


if __name__ == "__main__":
    main()
