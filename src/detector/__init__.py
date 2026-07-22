"""Detector module."""
from .base import BaseDetector
from .yolo import YoloDetector
from .rtdetr import RtDetrDetector

__all__ = ["BaseDetector", "YoloDetector", "RtDetrDetector"]
