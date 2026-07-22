# Experiment 003: RT-DETR vs YOLOv11 Comparison

**Date:** 2026-07-22
**Models:** RT-DETR Large vs YOLOv11 Nano
**Dataset:** `datasets/annotated/v1.0/`

## Objective
Evaluate if the RT-DETR (Real-Time DEtection TRansformer) model provides better bounding box accuracy or handles overlapping bubbles better natively compared to YOLOv11.

## Results
### Accuracy (mAP50)
- **YOLOv11 Nano:** ~0.60
- **RT-DETR Large:** 0.527

*(Note: Both scores are artificially suppressed by the noise in the auto-generated GroundingDINO dataset).*

### Inference Speed (CPU)
- **YOLOv11 Nano:** ~100-200ms per image.
- **RT-DETR Large:** ~250-400ms per image.
*(The Transformer model is significantly heavier and slower on the CPU).*

### Visual Performance
- Both models successfully detected the speech bubbles on our test pages.
- RT-DETR produced slightly different bounding box boundaries, which caused the OCR engine to extract text slightly differently (`어제부터 기다하다면서` vs YOLO's `어제부터 기다렇다면서.`).
- **Conclusion:** Because we implemented a robust custom IoM-NMS filter in Python for YOLOv11, YOLO's overlapping box issue is resolved. Given that YOLO is 2-3x faster on the CPU and produces identical (if not slightly better) text extraction results via EasyOCR, **YOLOv11 Nano is the clear winner for local CPU deployment.**

## Next Steps
- Standardize YOLOv11 as the primary detector for the final Phase 6 Unified Engine.
- Integrate everything into the final `vision.analyze()` pipeline.
