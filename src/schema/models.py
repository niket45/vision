"""
Canonical Vision Data Model — Vision Engine v0.1
=================================================
This is the source of truth for all downstream integrations.
The schema only changes on MAJOR version bumps.

Scroll-format extensions (non-breaking):
    BubbleRegion.cross_page       — bubble continues across a page boundary
    BubbleRegion.linked_bubble_id — ID of the counterpart region on the adjacent page
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

BBox = tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
Polygon = list[tuple[int, int]]   # [[x1, y1], [x2, y2], ...]
Mask = list[list[int]]            # binary 2-D array (row-major)


# ---------------------------------------------------------------------------
# Base region
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """Base class for every detected region.

    All positional (non-default) fields come first so subclasses can freely
    add keyword-only fields with defaults without violating Python dataclass ordering.
    """

    id: str
    bbox: BBox
    confidence: float
    type: str = "region"
    polygon: Polygon = field(default_factory=list)
    mask: Mask = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if len(self.bbox) != 4:
            raise ValueError("bbox must be a 4-tuple (x_min, y_min, x_max, y_max)")


# ---------------------------------------------------------------------------
# Specialised region types
# ---------------------------------------------------------------------------

@dataclass
class TextRegion(Region):
    """A region containing recognised text (inside a bubble or standalone)."""

    text: str = ""
    language: str = "unknown"
    orientation: Literal["horizontal", "vertical"] = "horizontal"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.type = "text"


@dataclass
class BubbleRegion(Region):
    """A speech or thought bubble.

    Scroll-format extensions:
        cross_page       — True when the bubble visually continues on the next/previous page.
        linked_bubble_id — ID of the corresponding BubbleRegion on the adjacent page,
                           allowing downstream consumers to merge partial bubbles.
    """

    shape: Literal["oval", "rectangular", "cloud", "jagged", "unknown"] = "unknown"
    tail: bool = False
    parent_panel: str | None = None

    # Scroll-format: cross-page bubble support
    cross_page: bool = False
    linked_bubble_id: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.type = "bubble"


@dataclass
class NarrationRegion(Region):
    """A narration / caption box (rectangular text box, not a speech bubble)."""

    text: str = ""
    language: str = "unknown"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.type = "narration"


@dataclass
class SFXRegion(Region):
    """A sound-effect text region rendered as stylised art."""

    text: str = ""
    language: str = "unknown"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.type = "sfx"


@dataclass
class PanelRegion(Region):
    """A single panel within a page.

    Note: In scroll-format manhwa, panel borders are often implicit.
    """

    children: list[str] = field(default_factory=list)  # IDs of child regions
    reading_order: int = -1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.type = "panel"


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------

@dataclass
class ImageInfo:
    width: int
    height: int
    source_path: str = ""


@dataclass
class InferenceMetadata:
    model: str
    version: str
    inference_time_ms: float
    device: str = "unknown"  # "cpu" | "cuda" | "mps"


@dataclass
class Diagnostics:
    warnings: list[str] = field(default_factory=list)
    confidence_summary: dict[str, float] = field(default_factory=dict)


@dataclass
class VisionResult:
    """The canonical output of the Vision Engine.

    This is the stable public contract. All modules must produce and consume
    this structure. It must not change except on a major version bump.
    """

    image: ImageInfo
    metadata: InferenceMetadata
    regions: list[Region]
    diagnostics: Diagnostics = field(default_factory=Diagnostics)

    # Convenience accessors
    def bubbles(self) -> list[BubbleRegion]:
        return [r for r in self.regions if isinstance(r, BubbleRegion)]

    def panels(self) -> list[PanelRegion]:
        return [r for r in self.regions if isinstance(r, PanelRegion)]

    def text_regions(self) -> list[TextRegion]:
        return [r for r in self.regions if isinstance(r, TextRegion)]

    def narrations(self) -> list[NarrationRegion]:
        return [r for r in self.regions if isinstance(r, NarrationRegion)]

    def sfx(self) -> list[SFXRegion]:
        return [r for r in self.regions if isinstance(r, SFXRegion)]

    def cross_page_bubbles(self) -> list[BubbleRegion]:
        """Return all bubbles that span a page boundary."""
        return [r for r in self.bubbles() if r.cross_page]
