"""OCR module."""
from .base import BaseOCR, OCRResult
from .easyocr_backend import EasyOcrBackend

__all__ = ["BaseOCR", "OCRResult", "EasyOcrBackend"]
