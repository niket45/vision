"""
Annotation Stats Reporter — Vision Engine Phase 2
==================================================
Prints a summary of the current annotation state:
  - How many images are labeled
  - Class distribution
  - Per-title breakdown

Usage:
    python scripts/annotation_stats.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = REPO_ROOT / "datasets" / "registry.yaml"
LABELS_DIR = REPO_ROOT / "datasets" / "annotated" / "v1.0" / "labels"

ALL_CLASSES = ["bubble", "narration", "text", "sfx", "panel"]


def main() -> None:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))

    total_images = registry["summary"]["total_pages"]
    label_files = list(LABELS_DIR.glob("*.txt")) if LABELS_DIR.exists() else []
    labeled = len(label_files)
    unlabeled = total_images - labeled

    print(f"\n{'='*55}")
    print(f" Vision Engine — Annotation Status")
    print(f"{'='*55}")
    print(f"  Total pages:   {total_images}")
    print(f"  Labeled:       {labeled}  ({labeled/total_images*100:.1f}%)")
    print(f"  Unlabeled:     {unlabeled}  ({unlabeled/total_images*100:.1f}%)")
    print()

    # Class distribution
    class_counts: dict[str, int] = defaultdict(int)
    empty_label_files = 0
    for lf in label_files:
        lines = lf.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            empty_label_files += 1
            continue
        for line in lines:
            parts = line.strip().split()
            if parts:
                idx = int(parts[0])
                if idx < len(ALL_CLASSES):
                    class_counts[ALL_CLASSES[idx]] += 1

    if class_counts:
        print(f"  Class distribution:")
        total_boxes = sum(class_counts.values())
        for cls in ALL_CLASSES:
            count = class_counts.get(cls, 0)
            bar = "█" * min(int(count / max(total_boxes, 1) * 40), 40)
            print(f"    {cls:<12} {count:>6}  {bar}")
        print(f"    {'TOTAL':<12} {total_boxes:>6}")
        print(f"\n  Empty label files (no detections): {empty_label_files}")

    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
