# Phase 2 — Dataset Engineering & Annotation Pipeline

## Overview

This document describes the end-to-end annotation workflow for the Vision Engine.

**No GPU required.** The auto-labeling pipeline runs on CPU using GroundingDINO-Tiny.

---

## Annotation Classes (Priority Order)

| Priority | Class | Label Studio Hotkey | Color |
|---|---|---|---|
| P1 | `bubble` | `1` | Red |
| P1 | `narration` | `2` | Teal |
| P2 | `text` | `3` | Blue |
| P3 | `sfx` | `4` | Green |
| P4 | `panel` | `5` | Yellow |

---

## Step-by-Step Workflow

### Step 1 — Auto-Label (GroundingDINO)

Run the auto-labeling pipeline on all 1,237 pages.  
On CPU this takes roughly **3-5 seconds/image → ~2-3 hours total**.

```bash
# Activate venv
.venv\Scripts\activate

# Dry run first (5 images, sanity check)
python scripts/auto_label.py --dry-run

# Full run — P1 classes (bubble + narration)
python scripts/auto_label.py --classes bubble narration --conf 0.30

# Later — add text regions
python scripts/auto_label.py --classes bubble narration text --conf 0.25
```

> **Resumable:** If interrupted, re-run the same command — already-processed images are skipped automatically via `.autolabel_progress.json`.

---

### Step 2 — Check Stats

```bash
python scripts/annotation_stats.py
```

---

### Step 3 — Start Label Studio

```bash
label-studio start
# Opens at http://localhost:8080
```

1. Create a new **Project**
2. Go to **Settings → Labeling Interface → Code**
3. Paste the contents of `configs/labelstudio_interface.xml`
4. Click **Save**

---

### Step 4 — Import Auto-Labels for Human Review

```bash
python scripts/export_to_labelstudio.py
```

In Label Studio:
1. **Import** → Upload JSON → select `datasets/annotated/v1.0/labelstudio_tasks.json`
2. Review each image — correct bounding boxes, add missed detections, delete false positives

**Review focus areas** (known hard cases per spec):
- Connected / overlapping bubbles
- Small narration boxes
- Cross-page bubbles at top/bottom of page
- SFX blending into background art

---

### Step 5 — Export Final Labels

In Label Studio:
1. **Export** → Choose **YOLO** format
2. Save to `datasets/annotated/v1.0/labels/`

---

### Step 6 — Generate Train/Val/Test Splits

```bash
python scripts/generate_splits.py
```

Output:
- `datasets/splits/train.txt`
- `datasets/splits/val.txt`
- `datasets/splits/test.txt`
- `datasets/annotated/v1.0/dataset.yaml`

---

### Step 7 — Tag Dataset Release

Update `datasets/registry.yaml`:
```yaml
annotation_status: "v1.0-complete"
```

Commit:
```bash
git add datasets/
git commit -m "data: annotated dataset v1.0 — N images, N boxes"
git push origin main
```

---

## Dataset Target

| Metric | Target |
|---|---|
| Minimum labeled pages | 500 (before training) |
| Ideal labeled pages | 1,000+ |
| Min boxes per class | 300 |
| Train/Val/Test split | 70 / 15 / 15 |
