# Experiment 002: Text Extraction Pipeline (EasyOCR)

**Date:** 2026-07-22
**Model:** EasyOCR (`ko`, `en`, `ch_sim`)
**Hardware:** Intel Core CPU (Inference)

## Objective
Establish a baseline for reading text from detected speech bubbles using an out-of-the-box multilingual OCR engine, creating our first fully functional end-to-end pipeline.

## Configuration
- **Detector:** YOLOv11 Nano (trained in exp-001)
- **OCR Engine:** EasyOCR
- **Image Pre-processing:** Crops taken directly from YOLO bounding boxes (no padding).
- **Post-processing:** String concatenation of multi-line detection.

## Results
- **Accuracy:** The engine perfectly extracted Korean Hangul characters from Eleceed test images (`어디 가`, `어제부터 기다렇다면서.`).
- **Speed:** Inference took ~250ms - 350ms per cropped bubble on the CPU.
- **Total Pipeline Speed:** ~1.0 second per page on CPU.

## Next Steps
- Integrate text translation API (e.g. Google Translate / DeepL).
- Compare YOLO bubble detection against RT-DETR (Phase 4).
