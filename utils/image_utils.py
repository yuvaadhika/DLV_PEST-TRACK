"""
================================================================================
PestTrack AI - Image Processing Utilities
================================================================================
"""

import os
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from typing import Tuple, List, Dict, Any, Optional

def load_image_cv(image_path: str) -> Optional[np.ndarray]:
    """Loads an image using OpenCV with Unicode path support on Windows."""
    if not os.path.exists(image_path):
        return None
    try:
        # np.fromfile handles Unicode/spaces in Windows paths
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image_cv(image: np.ndarray, save_path: str) -> bool:
    """Saves an OpenCV image handling Windows Unicode paths."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1]
        if not ext:
            ext = ".jpg"
            save_path += ext
        is_success, buf = cv2.imencode(ext, image)
        if is_success:
            with open(save_path, "wb") as f:
                buf.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"Error saving image to {save_path}: {e}")
        return False

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Converts OpenCV BGR image to PIL RGB Image."""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def pil_to_ctk(pil_img: Image.Image, target_size: Tuple[int, int]) -> ctk.CTkImage:
    """Converts a PIL image into a responsive CTkImage maintaining aspect ratio."""
    # Compute proportional size
    orig_w, orig_h = pil_img.size
    target_w, target_h = target_size

    ratio = min(target_w / max(1, orig_w), target_h / max(1, orig_h))
    new_w = max(1, int(orig_w * ratio))
    new_h = max(1, int(orig_h * ratio))

    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))

def draw_pest_detections(image: np.ndarray,
                         detections: List[Dict[str, Any]],
                         show_confidence: bool = True) -> Tuple[np.ndarray, int]:
    """
    Draws professional bounding boxes and labels around detected pests.
    Returns: (annotated_image, total_pest_count)
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    pest_count = len(detections)

    # Color palette for pest detection bounding boxes
    box_color = (46, 125, 50)     # Dark Green BGR: (50, 125, 46)
    text_bg_color = (27, 94, 32)  # Deep Green BGR: (32, 94, 27)
    text_color = (255, 255, 255)  # White

    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cls_name = det.get("class", "Pest")
        conf = det.get("confidence", 0.0)

        # Draw thick stylish box corners
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 125, 46), 2, cv2.LINE_AA)

        # Label Text
        if show_confidence:
            label = f"{cls_name} {conf * 100:.1f}%"
        else:
            label = f"{cls_name}"

        # Font & Badge calculation
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        (t_w, t_h), baseline = cv2.getTextSize(label, font, scale, thickness)

        # Top label badge background
        badge_y1 = max(0, y1 - t_h - 8)
        badge_y2 = y1
        badge_x2 = min(w, x1 + t_w + 10)

        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), (32, 94, 27), -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5), font, scale, text_color, thickness, cv2.LINE_AA)

    # Top overlay banner showing count
    banner_text = f"PestTrack AI | Total Pests Detected: {pest_count}"
    banner_w = min(w, 420)
    cv2.rectangle(annotated, (10, 10), (banner_w, 45), (20, 30, 25), -1)
    cv2.putText(annotated, banner_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 230, 120), 2, cv2.LINE_AA)

    return annotated, pest_count

def generate_segmentation_overlay(original_img: np.ndarray,
                                  mask: np.ndarray,
                                  highlight_color: Tuple[int, int, int] = (0, 0, 230),
                                  alpha: float = 0.45) -> np.ndarray:
    """
    Overlays a translucent mask over visible leaf damaged areas.
    mask: 2D uint8 array (0=healthy, 255=damaged)
    """
    h, w = original_img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = original_img.copy()

    # Create colored damage mask
    colored_damage = np.zeros_like(original_img)
    # Red-orange highlight for damage
    colored_damage[mask > 127] = highlight_color

    # Blend original and colored damage
    mask_indices = mask > 127
    overlay[mask_indices] = cv2.addWeighted(
        original_img[mask_indices], 1.0 - alpha,
        colored_damage[mask_indices], alpha, 0
    )

    # Draw smooth contours around damaged regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 255), 1, cv2.LINE_AA)

    return overlay

def calculate_damage_metrics(original_img: np.ndarray,
                             damage_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculates accurate damaged-area percentage against total leaf surface.
    Formula: Damage % = (Damaged Pixels / Total Leaf Pixels) * 100
    """
    h, w = original_img.shape[:2]
    if damage_mask.shape[:2] != (h, w):
        damage_mask = cv2.resize(damage_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Detect leaf boundary by segmenting non-background in HSV
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    # Agricultural leaves typically fall in green/yellow/brown hues
    # Exclude white or black background
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    _, leaf_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    leaf_mask_inv = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)[1]
    leaf_area_mask = cv2.bitwise_and(leaf_mask, leaf_mask_inv)

    total_leaf_pixels = int(np.count_nonzero(leaf_area_mask))
    damaged_pixels = int(np.count_nonzero(damage_mask > 127))

    # Fallback if whole frame is crop/leaf
    if total_leaf_pixels < (h * w * 0.05):
        total_leaf_pixels = h * w

    damage_pct = (damaged_pixels / max(1, total_leaf_pixels)) * 100.0
    damage_pct = min(100.0, max(0.0, damage_pct))

    return {
        "damage_percentage": round(damage_pct, 2),
        "damaged_pixels": damaged_pixels,
        "leaf_pixels": total_leaf_pixels
    }

def create_side_by_side_comparison(original_img: np.ndarray,
                                   mask: np.ndarray,
                                   overlay_img: np.ndarray,
                                   damage_pct: float) -> np.ndarray:
    """
    Creates a clean 3-panel comparison image:
    [Original Leaf] | [Binary Damage Mask] | [Segmented Overlay with Metrics]
    """
    h, w = original_img.shape[:2]
    target_h = 360
    target_w = int(w * (target_h / h))

    orig_resized = cv2.resize(original_img, (target_w, target_h))
    over_resized = cv2.resize(overlay_img, (target_w, target_h))

    # Mask in 3 channels
    if len(mask.shape) == 2:
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_3ch = mask
    mask_resized = cv2.resize(mask_3ch, (target_w, target_h))

    # Stack horizontally
    combined = np.hstack([orig_resized, mask_resized, over_resized])

    # Add header bar
    header_h = 45
    header = np.zeros((header_h, combined.shape[1], 3), dtype=np.uint8)
    header[:] = (27, 40, 32)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(header, "1. ORIGINAL LEAF", (20, 30), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "2. DAMAGE MASK", (target_w + 20, 30), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, f"3. SEGMENTED OVERLAY (Damage: {damage_pct:.1f}%)", (2 * target_w + 20, 30), font, 0.6, (80, 240, 140), 2, cv2.LINE_AA)

    final_comparison = np.vstack([header, combined])
    return final_comparison
