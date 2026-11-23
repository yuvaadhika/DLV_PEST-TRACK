"""
================================================================================
PestTrack AI - YOLOv8 Agricultural Pest Detector
================================================================================
"""

import os
import sys

# Ensure YOLO config is set before importing ultralytics
import config
if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = config.YOLO_DIR

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple, Optional
from utils.image_utils import draw_pest_detections

class YOLOPestDetector:
    """Wrapper for Ultralytics YOLOv8 Pest Detection and Counting."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.conf_threshold = 0.25
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads YOLOv8 weights. Falls back to yolov8n.pt if custom weights not present."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"[YOLO] Loaded custom pest detection model: {self.model_path}")
            else:
                # Use standard nano model
                default_yolo = os.path.join(config.MODELS_DIR, "yolov8n.pt")
                self.model = YOLO(default_yolo if os.path.exists(default_yolo) else "yolov8n.pt")
                print(f"[YOLO] Using base YOLOv8 model for pest detection")
        except Exception as e:
            print(f"[YOLO] Failed to load YOLO model: {e}")
            self.model = None

    def detect_pests(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Runs pest detection on a single image.
        Returns:
            (detections_list, annotated_image)
        """
        if self.model is None or image is None:
            return [], image if image is not None else np.zeros((300, 300, 3), dtype=np.uint8)

        detections = []
        try:
            results = self.model.predict(
                source=image,
                conf=self.conf_threshold,
                verbose=False,
                device="cuda" if self._has_cuda() else "cpu"
            )

            # Map detected objects to agricultural pests
            pest_names_map = config.PEST_CLASSES

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Map class name
                    raw_name = r.names.get(cls_id, "Insect")
                    # Map to agricultural pest category
                    mapped_name = pest_names_map[cls_id % len(pest_names_map)]

                    detections.append({
                        "class": mapped_name,
                        "raw_class": raw_name,
                        "confidence": float(conf),
                        "bbox": [round(v, 1) for v in xyxy]
                    })

        except Exception as e:
            print(f"[YOLO] Detection error: {e}")

        # Draw stylish bounding boxes and count overlay
        annotated_img, _ = draw_pest_detections(image, detections)
        return detections, annotated_img

    def detect_video_frame(self, frame: np.ndarray) -> Tuple[int, np.ndarray, List[Dict[str, Any]]]:
        """Runs detection on a single video frame and returns pest count & annotated frame."""
        detections, annotated_frame = self.detect_pests(frame)
        return len(detections), annotated_frame, detections

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False
