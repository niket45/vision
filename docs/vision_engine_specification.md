# Vision Engine Research & Development Specification

**Version:** 1.0

## 1. Objective

**Develop a modular, benchmark-driven computer vision framework for structured manga page analysis. The framework shall detect, segment, recognize, classify, and organize visual elements present on manga pages through a stable, implementation-independent API. The system must remain model-agnostic, allowing underlying algorithms and machine learning models to evolve without changing the external contract.**

## 2. Design Principles

The Vision Engine shall adhere to the following architectural principles:

*   **Modular:** Every subsystem shall be independently replaceable. Changing the detector must not require changes to OCR, segmentation, or inference.
*   **Model Agnostic:** No public interface shall depend on a specific machine learning model.
*   **Deterministic Interfaces:** All outputs shall conform to the stable data schema regardless of implementation.
*   **Benchmark Driven:** Technology decisions must be supported by empirical benchmark results rather than assumptions.
*   **Reproducible:** Every experiment must be reproducible from source code, configuration, dataset version, and random seed.
*   **Extensible:** New detectors, OCR engines, or segmentation algorithms shall be integrated without modifying existing public APIs.

## 3. Scope

The project is responsible only for computer vision.

**Included:**
*   Object Detection
*   Instance Segmentation
*   Text Detection
*   OCR
*   Reading Order
*   Region Classification
*   Confidence Estimation
*   Model Training & Benchmarking

**Not Included:**
*   Translation
*   Text Rendering
*   Image Editing
*   GUI or Web Development
*   Database or Application Logic

## 4. Vision Engine Modules

The Vision Engine must be separated into independent, decoupled subsystems. 

```text
Vision Engine
│
├── Detection
│   ├── Object Detector
│   ├── Bubble Detector
│   ├── Panel Detector
│   └── Region Classifier
│
├── Segmentation
│   ├── Bubble Segmentation
│   ├── Text Mask Segmentation
│   └── Region Masks
│
├── OCR
│   ├── Text Detection
│   ├── Text Recognition
│   └── Language Identification
│
├── Reading Order
│
├── Dataset
│
├── Training
│
├── Evaluation
│
└── Inference
```

*Note: The internal execution strategy is implementation-defined. Modules may execute sequentially, jointly, or in parallel, provided the public output contract remains unchanged.*

## 5. Canonical Vision Data Model

The Vision Engine shall produce a canonical `VisionResult`.

This canonical data model is the source of truth for all downstream integrations. Individual APIs, SDKs, or applications may serialize this model differently, but they shall preserve its semantics.

**VisionResult**
```text
image:
    width: int
    height: int

metadata:
    model: string
    version: string
    inference_time: float

regions:
    Region[]

diagnostics:
    warnings: array[string]
    confidence_summary: dict
```

**Base Region**
```text
id: string
type: string
bbox: [x_min, y_min, x_max, y_max]
polygon: [[x1, y1], [x2, y2], ...]
mask: array (binary)
confidence: float
```

**TextRegion** (Inherits Base)
```text
text: string
language: string
orientation: string (horizontal, vertical)
```

**BubbleRegion** (Inherits Base)
```text
shape: string (oval, rectangular, cloud, jagged)
tail: boolean
parent_panel: string (id)
```

**PanelRegion** (Inherits Base)
```text
children: array[string] (ids of regions inside panel)
reading_order: int
```

## 6. Repository Architecture

The codebase must follow a structured, scalable architecture.

```text
vision-engine/
├── src/
│   ├── detector/
│   ├── segmentation/
│   ├── ocr/
│   ├── reading_order/
│   ├── evaluation/
│   └── inference/
├── datasets/
├── experiments/
├── benchmarks/
├── configs/
├── weights/
├── docs/
└── tests/
```

## 7. Coding Standards & Rules

To ensure a high-quality, reusable library:
*   **Python 3.11**
*   **Type Hints** enforced everywhere.
*   **Black Formatter** for consistent styling.
*   **Every module must:**
    *   Have dedicated unit tests.
    *   Have its own benchmark scripts.
    *   Include a module-specific `README.md`.
    *   Expose a stable, versioned interface.
    *   Strictly avoid application-specific logic or downstream assumptions.

## 8. Non-Functional Requirements

The Vision Engine shall:
*   Be deterministic given identical inputs.
*   Support CPU and GPU inference.
*   Support batch inference.
*   Be thread-safe.
*   Avoid global mutable state.
*   Expose structured logging.
*   Fail gracefully.

## 9. Versioning Policy

The Vision Engine shall follow semantic versioning.

*   **Major Version:** Breaking changes to public interfaces.
*   **Minor Version:** Backward-compatible functionality.
*   **Patch Version:** Bug fixes and performance improvements.

*The Canonical Vision Data Model shall only change in a major version.*

## 10. Continuous Integration

Every Pull Request must:
*   [x] Run unit tests.
*   [x] Run linting.
*   [x] Run type checking.
*   [x] Run benchmark smoke tests.
*   [x] Validate dataset format.
*   [x] Validate documentation examples.
*   [x] Validate configuration files.

## 11. Dataset Standards

Datasets evolve and must be strictly managed. Every dataset release shall include:
*   Version
*   Annotation Schema Version
*   Changelog
*   Annotation Guidelines
*   Split Definition (Train/Val/Test)
*   Label Distribution
*   License

## 12. Technology Evaluation Matrix

Do not assume a specific model is best. Define experiments and evaluate candidates according to this matrix:

| Category | Initial Candidates |
| :--- | :--- |
| **Detection** | YOLOv11, RT-DETR, D-FINE, GroundingDINO |
| **Segmentation** | YOLO-Seg, SAM2 |
| **OCR** | PaddleOCR, Surya OCR, RapidOCR, EasyOCR |
| **Reading Order** | Rule-based, Graph-based, Transformer |

## 13. Experiment Tracking

Every experiment must record the following parameters to ensure reproducibility:
*   Dataset version
*   Git commit
*   Random seed
*   Hardware specifications
*   Training configuration (Epochs, LR, Batch Size)
*   Weights
*   Evaluation Metrics
*   Execution Date

## 14. Success Criteria (Subsystem Metrics)

Success is measured independently per subsystem.

### Detector
*   **Precision:** >95%
*   **Recall:** >90%
*   **mAP50-95:** Tracked for regression

### OCR
*   **CER (Character Error Rate):** <2%
*   **WER (Word Error Rate):** <5%
*   **Latency:** <50ms per region

### Segmentation
*   **IoU (Intersection over Union):** >90%
*   **Dice Score:** >0.90
*   **Boundary F1 Score:** Tracked for edge precision

### Reading Order
*   **Ordering Accuracy:** >95% correct sequence reconstruction

### System Level
*   **GPU Inference:** <100 ms/page
*   **CPU Inference:** Benchmarked and optimized where practical

## 15. Milestones

Project delivery is phased to ensure incremental validation without relying on transient implementation names.

*   **Phase 1: Environment & Pipeline Initialization**
    *   *Deliverable:* Repository scaffolded, CI/CD setup, mock training pipeline runs end-to-end without errors.
*   **Phase 2: Dataset Engineering**
    *   *Deliverable:* A sufficiently large, representative, and versioned annotated dataset covering all supported region classes.
*   **Phase 3: Baseline CNN Detector Evaluation**
    *   *Deliverable:* Benchmark completed and documented for the baseline CNN detector against the dataset.
*   **Phase 4: Transformer Detector Evaluation**
    *   *Deliverable:* Benchmark completed for the transformer detector. Model selected for the detection subsystem using objective results.
*   **Phase 5: OCR Benchmarking**
    *   *Deliverable:* Empirical comparison of OCR engines; integration of the selected engine.
*   **Phase 6: Unified Inference Library**
    *   *Deliverable:* Complete, versioned Python library exposing the stable `vision.analyze()` API contract.

## 16. Risk Factors

The following edge cases represent the highest risk to precision and must be specifically accounted for during dataset curation and evaluation:

*   **Small SFX detection:** Distinguishing small sound effects from background art.
*   **Rotated text:** Non-axis-aligned text in dynamic scenes.
*   **Vertical Japanese text:** Ensuring OCR and orientation metrics handle verticality natively.
*   **Connected speech bubbles:** Accurately segmenting bubbles that overlap or share borders.
*   **Tiny narration boxes:** Preventing false negatives on small rectangular boxes.
*   **Double-page spreads:** Managing layout and reading order across stitched pages.

## 17. Future Extensions

The architecture should support future additions including:
*   Additional object categories
*   Additional languages
*   New detector architectures
*   Vision-language models
*   Video page sequences

*without modifying the canonical data model.*

## 18. Definition of Done

The Vision Engine is considered complete when:

*   [x] Dataset is finalized.
*   [x] Benchmarks are reproducible.
*   [x] Best-performing models are selected using objective metrics.
*   [x] Stable inference API is implemented.
*   [x] Documentation is complete.
*   [x] Unit tests pass.
*   [x] Benchmark tests pass.
*   [x] Public API remains stable.
