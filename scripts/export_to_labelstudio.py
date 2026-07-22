"""
Label Studio Import Script — Vision Engine Phase 2
====================================================
Converts YOLO auto-labels + images into Label Studio JSON format
for human review.

Creates one Label Studio task per image with pre-annotations from
the auto-labeling step.

Usage:
    python scripts/export_to_labelstudio.py
    python scripts/export_to_labelstudio.py --output ls_tasks.json
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
ANNOTATED_ROOT = REPO_ROOT / "datasets" / "annotated" / "v1.0"

ALL_CLASSES = ["bubble", "narration", "text", "sfx", "panel"]

# Label Studio label colors
CLASS_COLORS = {
    "bubble":   "#FF6B6B",
    "narration": "#4ECDC4",
    "text":     "#45B7D1",
    "sfx":      "#96CEB4",
    "panel":    "#FFEAA7",
}


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

def yolo_to_ls(
    cx: float, cy: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """Convert YOLO (cx,cy,w,h normalised) to Label Studio (x,y,w,h in %)."""
    x = (cx - w / 2) * 100
    y = (cy - h / 2) * 100
    w_pct = w * 100
    h_pct = h * 100
    return x, y, w_pct, h_pct


def convert(output_path: Path) -> None:
    labels_dir = ANNOTATED_ROOT / "labels"
    images_dir = ANNOTATED_ROOT / "images"

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        print("[ERROR] No label files found. Run auto_label.py first.")
        return

    tasks = []
    skipped = 0

    for label_file in tqdm(label_files, desc="Converting to Label Studio format"):
        img_path = images_dir / (label_file.stem + ".jpg")
        if not img_path.exists():
            img_path = images_dir / (label_file.stem + ".png")
        if not img_path.exists():
            skipped += 1
            continue

        # Build pre-annotations from YOLO labels
        annotations = []
        lines = label_file.read_text(encoding="utf-8").strip().splitlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_idx = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            class_name = ALL_CLASSES[class_idx] if class_idx < len(ALL_CLASSES) else "unknown"

            x_pct, y_pct, w_pct, h_pct = yolo_to_ls(cx, cy, w, h)

            annotations.append({
                "id": f"{label_file.stem}_{len(annotations)}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "original_width": 0,   # filled by Label Studio on load
                "original_height": 0,
                "value": {
                    "x": round(x_pct, 4),
                    "y": round(y_pct, 4),
                    "width": round(w_pct, 4),
                    "height": round(h_pct, 4),
                    "rotation": 0,
                    "rectanglelabels": [class_name],
                },
            })

        task = {
            "data": {
                "image": f"/data/local-files/?d={img_path.resolve().as_posix()}",
            },
            "annotations": [
                {
                    "result": annotations,
                    "was_cancelled": False,
                    "ground_truth": False,
                }
            ] if annotations else [],
            "predictions": [],
        }
        tasks.append(task)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nLabel Studio export complete:")
    print(f"  Tasks:   {len(tasks)}")
    print(f"  Skipped: {skipped}")
    print(f"  Output:  {output_path}")
    print(f"\nNext: Import this file into Label Studio:")
    print(f"  1. Open Label Studio → Create Project")
    print(f"  2. Go to Settings → Labeling Interface → use template below")
    print(f"  3. Import → Upload JSON → select {output_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export auto-labels to Label Studio JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "datasets" / "annotated" / "v1.0" / "labelstudio_tasks.json",
    )
    args = parser.parse_args()
    convert(args.output)


if __name__ == "__main__":
    main()
