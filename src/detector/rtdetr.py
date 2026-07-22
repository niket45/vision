import time
from pathlib import Path
from typing import Any

from src.detector.base import BaseDetector
from src.schema.models import (
    VisionResult,
    ImageInfo,
    InferenceMetadata,
    Region,
    BubbleRegion,
    NarrationRegion,
    TextRegion,
    SFXRegion,
    PanelRegion,
)


class RtDetrDetector(BaseDetector):
    """RT-DETR implementation of the BaseDetector interface."""

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        """Initialize the RT-DETR model.
        
        Args:
            model_path: Path to the trained .pt weights file.
            device: 'cpu' or 'cuda'.
        """
        try:
            from ultralytics import RTDETR
        except ImportError:
            raise ImportError(
                "ultralytics is required to use RtDetrDetector. "
                "Install it with: pip install ultralytics"
            )

        self.model_path = str(model_path)
        self.device = device
        self.model = RTDETR(self.model_path)
        self.model.to(self.device)

        # Mapping of class index to Region subclass based on our dataset.yaml
        self._class_map = {
            0: BubbleRegion,
            1: NarrationRegion,
            2: TextRegion,
            3: SFXRegion,
            4: PanelRegion,
        }
        self._class_names = {
            0: "bubble",
            1: "narration",
            2: "text",
            3: "sfx",
            4: "panel",
        }

    def _convert_result(
        self, yolo_result: Any, image_path: Path, inference_time_ms: float
    ) -> VisionResult:
        """Convert a single ultralytics Result into a canonical VisionResult."""
        
        # Extract original image dimensions
        img_h, img_w = yolo_result.orig_shape
        image_info = ImageInfo(
            width=img_w, height=img_h, source_path=str(image_path)
        )

        metadata = InferenceMetadata(
            model="RT-DETR",
            version="Large",
            inference_time_ms=inference_time_ms,
            device=self.device,
        )

        regions: list[Region] = []
        if yolo_result.boxes is not None:
            boxes = yolo_result.boxes.xyxy.cpu().numpy()
            confidences = yolo_result.boxes.conf.cpu().numpy()
            classes = yolo_result.boxes.cls.cpu().numpy().astype(int)
            
            # Note: RT-DETR doesn't strictly need custom NMS since Transformers have it built-in,
            # but we leave it here just in case the dataset noise forces overlapping predictions.
            def get_iom(box_a, box_b):
                ax1, ay1, ax2, ay2 = box_a
                bx1, by1, bx2, by2 = box_b
                
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                
                i_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
                
                min_area = min(a_area, b_area)
                if min_area == 0: return 0.0
                return i_area / min_area

            # IoM-based NMS (suppress heavily overlapping boxes regardless of which is larger)
            indices = list(range(len(boxes)))
            indices.sort(key=lambda x: confidences[x], reverse=True)
            
            keep_indices = []
            for i in indices:
                keep = True
                for j in keep_indices:
                    if get_iom(boxes[i], boxes[j]) > 0.85:
                        keep = False
                        break
                if keep:
                    keep_indices.append(i)

            for i in keep_indices:
                box = boxes[i]
                conf = confidences[i]
                cls_idx = classes[i]
                
                x_min, y_min, x_max, y_max = map(int, box)
                bbox = (x_min, y_min, x_max, y_max)
                
                region_id = f"r{i}_{self._class_names.get(cls_idx, 'unknown')}"
                region_cls = self._class_map.get(cls_idx, Region)
                
                if region_cls is Region:
                    region = Region(id=region_id, bbox=bbox, confidence=float(conf))
                else:
                    region = region_cls(id=region_id, bbox=bbox, confidence=float(conf))
                
                regions.append(region)

        return VisionResult(image=image_info, metadata=metadata, regions=regions)

    def detect(self, image_path: Path, confidence_threshold: float = 0.5) -> VisionResult:
        """Run detection on a single image."""
        start_time = time.time()
        
        results = self.model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            imgsz=1024,
            device=self.device,
            agnostic_nms=True,
            verbose=False,
        )
        
        inference_time_ms = (time.time() - start_time) * 1000.0
        return self._convert_result(results[0], image_path, inference_time_ms)

    def detect_batch(
        self, image_paths: list[Path], confidence_threshold: float = 0.5
    ) -> list[VisionResult]:
        """Run detection on a batch of images."""
        if not image_paths:
            return []

        start_time = time.time()
        
        results = self.model.predict(
            source=[str(p) for p in image_paths],
            conf=confidence_threshold,
            imgsz=1024,
            device=self.device,
            agnostic_nms=True,
            verbose=False,
        )
        
        total_time_ms = (time.time() - start_time) * 1000.0
        time_per_image = total_time_ms / len(image_paths)

        vision_results = []
        for path, res in zip(image_paths, results):
            vision_results.append(self._convert_result(res, path, time_per_image))
            
        return vision_results
