"""
OCR module — Text detection and recognition subsystem.

Supports: PaddleOCR, Surya OCR, RapidOCR, EasyOCR (benchmark candidates).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from src.schema import BBox, TextRegion


@dataclass
class OCRResult:
    """Raw output from an OCR engine for a single text region."""

    bbox: BBox
    text: str
    confidence: float
    language: str = "unknown"
    orientation: str = "horizontal"


class BaseOCR(ABC):
    """Abstract base for all OCR backends."""

    @abstractmethod
    def recognize(self, image_path: Path) -> list[OCRResult]:
        """Run text detection + recognition on a full page image.

        Args:
            image_path: Path to the image (full page or cropped region).

        Returns:
            List of OCRResult, one per detected text line or block.
        """
        ...

    @abstractmethod
    def recognize_region(self, image_path: Path, bbox: BBox) -> OCRResult:
        """Run OCR on a specific cropped region.

        Args:
            image_path: Source image path.
            bbox: Region to crop and recognise.

        Returns:
            Single OCRResult for the cropped region.
        """
        ...

    def to_text_regions(self, results: list[OCRResult]) -> list[TextRegion]:
        """Convert raw OCRResults to canonical TextRegion objects."""
        import uuid
        return [
            TextRegion(
                id=str(uuid.uuid4()),
                type="text",
                bbox=r.bbox,
                confidence=r.confidence,
                text=r.text,
                language=r.language,
                orientation=r.orientation,  # type: ignore[arg-type]
            )
            for r in results
        ]
