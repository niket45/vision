"""Shared pytest fixtures for the Vision Engine test suite."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_image_dir() -> Path:
    """Return path to a directory containing test images."""
    return Path(__file__).parent / "fixtures" / "images"
