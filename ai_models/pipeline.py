"""
================================================================================
PestTrack AI - Integrated Multi-Model AI Pipeline
Combines:
  1. YOLOv8 Pest Detection & Counting
  2. PyTorch CNN Pest Species Classification
  3. PyTorch U-Net Leaf Damage Semantic Segmentation
  4. Rule-Based Infestation Severity Engine
================================================================================
"""

import os
import cv2
import datetime
from PIL import Image
from typing import Dict, Any, Optional
import config
from ai_models.yolo_detector import YOLOPestDetector
from ai_models.cnn_classifier import PestClassifier
from ai_models.unet_segmentor import LeafDamageSegmentor
from database.db_manager import DatabaseManager
from utils.image_utils import load_image_cv, save_image_cv, cv2_to_pil, create_side_by_side_comparison

class PestTrackPipeline:
    """Master AI Orchestrator integrating all 3 deep learning vision models."""

    def __init__(self):
        print("[Pipeline] Initializing PestTrack AI Vision Models...")
        self.detector = YOLOPestDetector()
        self.classifier = PestClassifier()
        self.segmentor = LeafDamageSegmentor()
        self.db = DatabaseManager()
        print("[Pipeline] All models initialized successfully.")

    def evaluate_infestation_severity(self, pest_count: int, damage_percentage: float) -> str:
        """
        Evaluates Infestation Severity based on documented agricultural thresholds:
        - High: Pest Count >= 8 OR Leaf Damage > 30%
        - Medium: Pest Count between 3 and 7 OR Leaf Damage between 10% and 30%
        - Low: Pest Count <= 2 AND Leaf Damage < 10%
        """
        if pest_count >= 8 or damage_percentage > 30.0:
            return "High"
        elif (3 <= pest_count <= 7) or (10.0 <= damage_percentage <= 30.0):
            return "Medium"
        else:
            return "Low"

    def analyze_full_image(self, image_path: str) -> Dict[str, Any]:
        """
        Executes end-to-end multi-model inference on a crop image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        cv_img = load_image_cv(image_path)
        if cv_img is None:
            raise ValueError(f"Failed to read image at: {image_path}")

        image_filename = os.path.basename(image_path)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # ----------------------------------------------------------------------
        # Model 1: YOLO Pest Detection & Counting
        # ----------------------------------------------------------------------
        detections, det_annotated_img = self.detector.detect_pests(cv_img)
        pest_count = len(detections)

        # ----------------------------------------------------------------------
        # Model 2: PyTorch CNN Pest Species Classification
        # ----------------------------------------------------------------------
        pil_img = cv2_to_pil(cv_img)
        # If pests were detected, optionally crop around the highest confidence pest
        if detections and len(detections) > 0:
            top_box = sorted(detections, key=lambda x: x["confidence"], reverse=True)[0]["bbox"]
            x1, y1, x2, y2 = [int(v) for v in top_box]
            h, w = cv_img.shape[:2]
            # Add small margin
            x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
            x2, y2 = min(w, x2 + 10), min(h, y2 + 10)
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                crop_pil = pil_img.crop((x1, y1, x2, y2))
                cls_result = self.classifier.predict(crop_pil)
            else:
                cls_result = self.classifier.predict(pil_img)
        else:
            cls_result = self.classifier.predict(pil_img)

        predicted_species = cls_result["predicted_species"]
        classification_conf = cls_result["confidence"]
        top_predictions = cls_result["top_predictions"]

        # ----------------------------------------------------------------------
        # Model 3: PyTorch U-Net Leaf Damage Segmentation
        # ----------------------------------------------------------------------
        seg_result = self.segmentor.segment_leaf(cv_img)
        damage_mask = seg_result["mask"]
        seg_overlay = seg_result["overlay"]
        damage_pct = seg_result["damage_percentage"]

        # ----------------------------------------------------------------------
        # Infestation Severity Analysis
        # ----------------------------------------------------------------------
        severity_level = self.evaluate_infestation_severity(pest_count, damage_pct)

        # ----------------------------------------------------------------------
        # Save Visual Artifacts
        # ----------------------------------------------------------------------
        det_save_path = os.path.join(config.DETECTIONS_DIR, f"det_{timestamp_str}_{image_filename}")
        seg_save_path = os.path.join(config.SEGMENTATIONS_DIR, f"seg_{timestamp_str}_{image_filename}")
        comp_save_path = os.path.join(config.SEGMENTATIONS_DIR, f"comp_{timestamp_str}_{image_filename}")

        # Create 3-panel comparison
        comparison_img = create_side_by_side_comparison(cv_img, damage_mask, seg_overlay, damage_pct)

        save_image_cv(det_annotated_img, det_save_path)
        save_image_cv(seg_overlay, seg_save_path)
        save_image_cv(comparison_img, comp_save_path)

        # ----------------------------------------------------------------------
        # Record into SQLite Database
        # ----------------------------------------------------------------------
        analysis_id = self.db.insert_analysis(
            analysis_type="full",
            image_name=image_filename,
            original_image_path=image_path,
            result_image_path=comp_save_path,
            pest_count=pest_count,
            primary_pest_species=predicted_species,
            confidence=classification_conf,
            damage_percentage=damage_pct,
            severity_level=severity_level,
            notes=f"Detected {pest_count} pests. Leaf damage: {damage_pct:.1f}%.",
            detections=detections
        )

        return {
            "id": analysis_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": image_filename,
            "original_image_path": image_path,
            "detection_image_path": det_save_path,
            "segmentation_image_path": seg_save_path,
            "comparison_image_path": comp_save_path,
            "pest_count": pest_count,
            "detections": detections,
            "predicted_species": predicted_species,
            "confidence": classification_conf,
            "top_predictions": top_predictions,
            "damage_percentage": damage_pct,
            "severity_level": severity_level,
            "disclaimer": config.DISCLAIMER_TEXT
        }
