from pathlib import Path
from typing import Literal

from src.ocr.base import BaseOCR, OCRResult
from src.schema.models import BBox


class EasyOcrBackend(BaseOCR):
    """EasyOCR implementation of the BaseOCR interface."""

    def __init__(self, langs: list[str] = None, gpu: bool = False):
        """Initialize EasyOCR.
        
        Args:
            langs: List of language codes (e.g., ['en', 'ko'], ['ch_sim', 'en']).
                   Defaults to ['en'] if none provided.
            gpu: Whether to run inference on GPU.
        """
        try:
            import easyocr
        except ImportError:
            raise ImportError(
                "easyocr is required to use EasyOcrBackend. "
                "Install it with: pip install easyocr"
            )

        if langs is None:
            langs = ['en']
            
        self.langs = langs
        # The first time this runs, it will download the language models to ~/.EasyOCR/model
        self.reader = easyocr.Reader(self.langs, gpu=gpu)

    def recognize(self, image_path: Path) -> list[OCRResult]:
        """Run text detection + recognition on a full page image."""
        
        # EasyOCR returns a list of tuples: (bbox, text, prob)
        # bbox is a list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        results = self.reader.readtext(str(image_path))
        
        ocr_results = []
        for bbox_points, text, confidence in results:
            # Convert 4-point polygon to a 2-point bounding box (xmin, ymin, xmax, ymax)
            x_coords = [int(pt[0]) for pt in bbox_points]
            y_coords = [int(pt[1]) for pt in bbox_points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Determine orientation (simple heuristic based on aspect ratio)
            width = x_max - x_min
            height = y_max - y_min
            orientation = "vertical" if height > width * 1.5 else "horizontal"
            
            ocr_results.append(
                OCRResult(
                    bbox=(x_min, y_min, x_max, y_max),
                    text=text.strip(),
                    confidence=float(confidence),
                    language=",".join(self.langs),
                    orientation=orientation
                )
            )
            
        return ocr_results

    def recognize_region(self, image_path: Path, bbox: BBox) -> OCRResult:
        """Run OCR on a specific cropped region."""
        from PIL import Image
        import numpy as np
        
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure coordinates are positive
        x_min, y_min = max(0, x_min), max(0, y_min)
        
        # Load and crop the image
        img = Image.open(image_path).convert("RGB")
        img_crop = img.crop((x_min, y_min, x_max, y_max))
        
        # Convert to numpy array for EasyOCR
        img_array = np.array(img_crop)
        
        # Pass the cropped image array to readtext
        results = self.reader.readtext(img_array)
        
        if not results:
            return OCRResult(
                bbox=bbox, 
                text="", 
                confidence=0.0, 
                language=",".join(self.langs)
            )
            
        # If there are multiple detected text lines inside the crop, we concatenate them
        full_text = " ".join([res[1] for res in results])
        # Average confidence
        avg_conf = sum([res[2] for res in results]) / len(results)
        
        # The orientation of the crop itself
        width = x_max - x_min
        height = y_max - y_min
        orientation = "vertical" if height > width * 1.5 else "horizontal"
        
        return OCRResult(
            bbox=bbox,
            text=full_text.strip(),
            confidence=float(avg_conf),
            language=",".join(self.langs),
            orientation=orientation
        )
