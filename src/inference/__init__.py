"""
Inference module — Unified public API.

Usage:
    from src.inference import analyze
    result = analyze("path/to/page.jpg")
"""

from __future__ import annotations

from pathlib import Path

from src.schema import VisionResult


def analyze(image_path: str | Path) -> VisionResult:
    """Run the full Vision Engine pipeline on a single page.

    This is the stable public entry point. The implementation is subject to
    change across minor versions, but the signature and return type are fixed.

    Args:
        image_path: Path to a manga/manhwa page image.

    Returns:
        VisionResult containing all detected and recognised regions.

    Raises:
        FileNotFoundError: If image_path does not exist.
        NotImplementedError: Until a concrete pipeline is wired up.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    raise NotImplementedError(
        "analyze() is not yet implemented. "
        "Wire up a concrete detector, segmenter, and OCR backend first."
    )
