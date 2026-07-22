# Experiment 001: YOLOv11 Baseline

**Date:** 2026-07-22
**Model:** YOLOv11 Nano (`yolo11n.pt`)
**Dataset:** `datasets/annotated/v1.0/` (1237 images, GroundingDINO auto-labels)
**Hardware:** Google Colab T4 GPU (Training), Intel Core CPU (Inference)

## Objective
Establish a fast, robust baseline for bubble and narration box detection using a modern CNN architecture to replace the slow zero-shot GroundingDINO pipeline.

## Configuration
- **Epochs:** 50 (Early stopping patience: 10)
- **Image Size:** 1024px
- **Batch Size:** 16
- **Classes:** bubble, narration, text, sfx, panel

## Results
- **Training Time:** ~25 minutes on T4 GPU
- **Inference Speed:** YOLOv11 Nano inference on CPU is expected to be ~50-100ms per image (compared to ~12,000ms for GroundingDINO).

## Next Steps
Integrate the trained `vision_yolov11_baseline.pt` into the `BaseDetector` interface and benchmark inference speed on the local CPU.
