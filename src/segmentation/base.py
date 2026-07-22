"""
Segmentation module — Instance segmentation subsystem.

Supports: YOLO-Seg, SAM2 (benchmark candidates).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.schema import Region, VisionResult


class BaseSegmenter(ABC):
    """Abstract base for all segmentation backends."""

    @abstractmethod
    def segment(self, image_path: Path, regions: list[Region]) -> list[Region]:
        """Add polygon masks to pre-detected regions.

        Args:
            image_path: Absolute path to the input image.
            regions: Regions from a prior detection pass (bboxes populated).

        Returns:
            The same regions with `polygon` and `mask` fields populated.
        """
        ...

    @abstractmethod
    def segment_from_scratch(self, image_path: Path) -> VisionResult:
        """Detect and segment in a single pass (for models like YOLO-Seg).

        Returns:
            VisionResult with all regions fully populated including masks.
        """
        ...
