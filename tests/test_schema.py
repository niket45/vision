"""Tests for the canonical VisionResult schema."""

import pytest
from src.schema.models import (
    BubbleRegion,
    Diagnostics,
    ImageInfo,
    InferenceMetadata,
    NarrationRegion,
    PanelRegion,
    Region,
    SFXRegion,
    TextRegion,
    VisionResult,
)


# ---------------------------------------------------------------------------
# Region base
# ---------------------------------------------------------------------------

class TestRegion:
    def test_valid_region(self):
        r = Region(id="r1", type="test", bbox=(0, 0, 100, 200), confidence=0.9)
        assert r.id == "r1"
        assert r.confidence == 0.9

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence"):
            Region(id="r2", type="test", bbox=(0, 0, 100, 200), confidence=1.5)

    def test_invalid_bbox(self):
        with pytest.raises(ValueError, match="bbox"):
            Region(id="r3", type="test", bbox=(0, 0, 100), confidence=0.5)  # type: ignore


# ---------------------------------------------------------------------------
# BubbleRegion — scroll-format extensions
# ---------------------------------------------------------------------------

class TestBubbleRegion:
    def test_default_not_cross_page(self):
        b = BubbleRegion(id="b1", bbox=(10, 10, 50, 80), confidence=0.95)
        assert b.cross_page is False
        assert b.linked_bubble_id is None

    def test_cross_page_bubble(self):
        b = BubbleRegion(
            id="b1",
            bbox=(10, 900, 50, 1000),
            confidence=0.88,
            cross_page=True,
            linked_bubble_id="b2",
        )
        assert b.cross_page is True
        assert b.linked_bubble_id == "b2"

    def test_tail_field(self):
        b = BubbleRegion(id="b3", bbox=(0, 0, 100, 100), confidence=0.7, tail=True)
        assert b.tail is True


# ---------------------------------------------------------------------------
# VisionResult convenience accessors
# ---------------------------------------------------------------------------

class TestVisionResult:
    def _make_result(self) -> VisionResult:
        return VisionResult(
            image=ImageInfo(width=800, height=3000, source_path="test.jpg"),
            metadata=InferenceMetadata(
                model="test-model", version="0.1.0", inference_time_ms=42.0
            ),
            regions=[
                BubbleRegion(id="b1", bbox=(10, 10, 50, 80), confidence=0.9),
                BubbleRegion(
                    id="b2",
                    bbox=(10, 2900, 50, 3000),
                    confidence=0.8,
                    cross_page=True,
                    linked_bubble_id="b3",
                ),
                TextRegion(id="t1", bbox=(12, 15, 48, 75), confidence=0.85, text="Hello"),
                NarrationRegion(id="n1", bbox=(0, 0, 200, 50), confidence=0.92),
                SFXRegion(id="s1", bbox=(300, 100, 500, 200), confidence=0.7),
                PanelRegion(id="p1", bbox=(0, 0, 800, 500), confidence=0.99, reading_order=0),
            ],
        )

    def test_bubbles(self):
        r = self._make_result()
        assert len(r.bubbles()) == 2

    def test_cross_page_bubbles(self):
        r = self._make_result()
        assert len(r.cross_page_bubbles()) == 1
        assert r.cross_page_bubbles()[0].id == "b2"

    def test_text_regions(self):
        r = self._make_result()
        assert len(r.text_regions()) == 1
        assert r.text_regions()[0].text == "Hello"

    def test_panels(self):
        r = self._make_result()
        assert len(r.panels()) == 1
        assert r.panels()[0].reading_order == 0

    def test_narrations(self):
        r = self._make_result()
        assert len(r.narrations()) == 1

    def test_sfx(self):
        r = self._make_result()
        assert len(r.sfx()) == 1
