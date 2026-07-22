"""
Auto-Labeling Pipeline — Vision Engine Phase 2
===============================================
Uses GroundingDINO-Tiny (via HuggingFace Transformers) to auto-label
manga/manhwa pages for:
  - Priority 1: speech_bubble, narration_box
  - Priority 2: text_region
  - Priority 3: sfx

CPU-safe: runs in batches with progress resumption.
Output: YOLO format labels in datasets/annotated/v1.0/labels/

Usage:
    python scripts/auto_label.py --classes bubble narration --conf 0.30
    python scripts/auto_label.py --classes bubble narration text --conf 0.25
    python scripts/auto_label.py --dry-run  # test on 5 images only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = REPO_ROOT / "datasets" / "registry.yaml"
RAW_ROOT = REPO_ROOT / "datasets" / "raw"
OUT_ROOT = REPO_ROOT / "datasets" / "annotated" / "v1.0"
PROGRESS_FILE = REPO_ROOT / "datasets" / "annotated" / "v1.0" / ".autolabel_progress.json"

# GroundingDINO model — tiny is fastest on CPU
MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Map our class names → natural language prompts for GroundingDINO
CLASS_PROMPTS: dict[str, str] = {
    "bubble":   "speech bubble . dialogue bubble . thought bubble",
    "narration": "narration box . caption box . text box",
    "text":     "text region . written text",
    "sfx":      "sound effect text . onomatopoeia",
    "panel":    "comic panel . panel border",
}

# YOLO class index order (must be consistent with dataset.yaml)
ALL_CLASSES = ["bubble", "narration", "text", "sfx", "panel"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def collect_images(registry: dict) -> list[Path]:
    """Return all raw image paths from the registry."""
    images: list[Path] = []
    for source in registry["sources"]:
        for chapter in source["chapters"]:
            chapter_dir = REPO_ROOT / chapter["path"]
            if chapter_dir.exists():
                imgs = sorted(chapter_dir.glob("*.jpg")) + sorted(chapter_dir.glob("*.png"))
                images.extend(imgs)
    return images


def bbox_to_yolo(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert [x_min, y_min, x_max, y_max] (absolute) to YOLO cx,cy,w,h (normalised)."""
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    return cx, cy, w, h


def load_progress() -> set[str]:
    """Return set of already-processed image paths (for resumption)."""
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        return set(data.get("done", []))
    return set()


def save_progress(done: set[str]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({"done": sorted(done)}, indent=2))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    classes: list[str],
    conf_threshold: float,
    dry_run: bool,
    batch_size: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"Vision Engine — Auto-Labeling Pipeline")
    print(f"{'='*60}")
    print(f"  Model:     {MODEL_ID}")
    print(f"  Classes:   {classes}")
    print(f"  Threshold: {conf_threshold}")
    print(f"  Device:    {'cuda' if torch.cuda.is_available() else 'cpu (slow — be patient)'}")
    print(f"  Dry run:   {dry_run}")
    print()

    # Build text prompt for GroundingDINO
    text_prompt = " . ".join(CLASS_PROMPTS[c] for c in classes) + " ."

    # Load model
    print("Loading GroundingDINO-Tiny model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}.\n")

    # Collect images
    registry = load_registry()
    all_images = collect_images(registry)
    if dry_run:
        all_images = all_images[:5]
        print(f"DRY RUN: processing first 5 images only.\n")

    # Resume support
    done = load_progress()
    images_to_process = [p for p in all_images if str(p) not in done]
    print(f"Total images: {len(all_images)} | Already done: {len(done)} | To process: {len(images_to_process)}\n")

    # Prepare output dirs
    images_out = OUT_ROOT / "images"
    labels_out = OUT_ROOT / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Class index mapping
    class_to_idx = {c: i for i, c in enumerate(ALL_CLASSES)}

    # --- Map GroundingDINO label strings back to our class names ---
    # GroundingDINO returns the matched phrase from the prompt.
    # We map each phrase back to its class.
    label_map: dict[str, str] = {}
    for cls, prompt in CLASS_PROMPTS.items():
        if cls in classes:
            for phrase in prompt.split(" . "):
                label_map[phrase.strip()] = cls

    total_detections = 0
    errors = 0
    t_start = time.time()

    for img_path in tqdm(images_to_process, desc="Auto-labeling", unit="img"):
        try:
            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size

            # Run GroundingDINO
            inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=conf_threshold,
                text_threshold=conf_threshold,
                target_sizes=[(img_h, img_w)],
            )[0]

            # Build YOLO label lines
            label_lines: list[str] = []
            for box, score, label_text in zip(
                results["boxes"].tolist(),
                results["scores"].tolist(),
                results["labels"],
            ):
                # Map label text to class
                matched_class = None
                for phrase, cls in label_map.items():
                    if phrase.lower() in label_text.lower() or label_text.lower() in phrase.lower():
                        matched_class = cls
                        break
                if matched_class is None:
                    continue  # unknown label — skip

                class_idx = class_to_idx[matched_class]
                cx, cy, w, h = bbox_to_yolo(box, img_w, img_h)
                label_lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Write label file
            label_file = labels_out / (img_path.stem + ".txt")
            label_file.write_text("\n".join(label_lines), encoding="utf-8")

            # Symlink / copy image path to annotated/images (store relative path only)
            img_out_path = images_out / img_path.name
            if not img_out_path.exists():
                import shutil
                shutil.copy2(img_path, img_out_path)

            total_detections += len(label_lines)
            done.add(str(img_path))

            # Save progress every 50 images
            if len(done) % 50 == 0:
                save_progress(done)

        except Exception as exc:
            print(f"\n[ERROR] {img_path.name}: {exc}")
            errors += 1
            continue

    save_progress(done)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Auto-labeling complete!")
    print(f"  Processed:    {len(images_to_process)} images")
    print(f"  Detections:   {total_detections}")
    print(f"  Errors:       {errors}")
    print(f"  Time elapsed: {elapsed/60:.1f} min")
    print(f"  Labels saved: {labels_out}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vision Engine Auto-Labeling Pipeline")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["bubble", "narration"],
        choices=list(CLASS_PROMPTS.keys()),
        help="Classes to auto-label (default: bubble narration)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Detection confidence threshold (default: 0.30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1 for CPU safety)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only process first 5 images (for testing)",
    )
    args = parser.parse_args()

    run(
        classes=args.classes,
        conf_threshold=args.conf,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
