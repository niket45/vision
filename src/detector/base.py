"""
Detector module — Object detection subsystem.

Supports: YOLOv11, RT-DETR, D-FINE, GroundingDINO (benchmark candidates).
Public interface is model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.schema import VisionResult


class BaseDetector(ABC):
    """Abstract base for all detection backends.

    Implementations must not leak model-specific types into return values.
    All outputs must conform to VisionResult.
    """

    @abstractmethod
    def detect(self, image_path: Path, confidence_threshold: float = 0.5) -> VisionResult:
        """Run detection on a single image.

        Args:
            image_path: Absolute path to the input image.
            confidence_threshold: Minimum confidence to include a detection.

        Returns:
            VisionResult containing all detected regions.
        """
        ...

    @abstractmethod
    def detect_batch(
        self, image_paths: list[Path], confidence_threshold: float = 0.5
    ) -> list[VisionResult]:
        """Run detection on a batch of images.

        Args:
            image_paths: List of absolute paths to input images.
            confidence_threshold: Minimum confidence to include detections.

        Returns:
            List of VisionResult, one per image, in the same order.
        """
        ...
