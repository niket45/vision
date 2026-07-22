# Vision Engine

A modular, benchmark-driven computer vision framework for structured manga/manhwa page analysis.

## Scope

Computer vision only. No translation, rendering, or GUI.

- Object Detection (panels, bubbles, narration boxes, SFX)
- Instance Segmentation (bubble masks, text masks)
- OCR (text detection + recognition)
- Reading Order Inference
- Model Training & Benchmarking

## Architecture

```
src/
├── schema/          # Canonical VisionResult data model
├── detector/        # Object detection (YOLO, RT-DETR, D-FINE, GroundingDINO)
├── segmentation/    # Instance segmentation (YOLO-Seg, SAM2)
├── ocr/             # Text detection + recognition (PaddleOCR, Surya, RapidOCR)
├── reading_order/   # Reading order inference (rule-based, graph, transformer)
├── evaluation/      # Metric computation (mAP, IoU, CER, WER)
└── inference/       # Unified vision.analyze() public API
```

## Scroll-Format Support

This engine targets **webtoon/manhwa scroll-format** pages:
- Pages are tall vertical strips processed as-is (no tiling)
- Speech bubbles may span two consecutive pages (`cross_page=True` in `BubbleRegion`)
- Cross-page bubbles are linked via `linked_bubble_id`

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pytest
```

## Design Principles

- **Model Agnostic** — no public interface depends on a specific model
- **Deterministic Interfaces** — all outputs conform to the stable `VisionResult` schema
- **Benchmark Driven** — every technology decision backed by empirical results
- **Reproducible** — every experiment fully logged (dataset version, git hash, seed, hardware)

## Versioning

Follows semantic versioning. The canonical `VisionResult` schema only changes on major versions.

## Docs

Full specification: [`docs/vision_engine_specification.md`](docs/vision_engine_specification.md)
