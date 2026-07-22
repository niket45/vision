"""
Reading Order module — Determines the correct read sequence of regions.

Scroll-format note: reading order for manhwa is generally top-to-bottom,
left-to-right within a row, but cross-page bubble awareness is required.

Candidates: Rule-based (geometric), Graph-based, Transformer-based.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.schema import BubbleRegion, PanelRegion, Region


class BaseReadingOrderSolver(ABC):
    """Abstract base for reading order solvers."""

    @abstractmethod
    def sort(self, regions: list[Region]) -> list[Region]:
        """Return regions sorted in reading order.

        Args:
            regions: Unsorted regions from a VisionResult.

        Returns:
            Same regions with `reading_order` index assigned where applicable,
            sorted by that index.
        """
        ...
