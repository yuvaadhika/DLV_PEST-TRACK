"""
================================================================================
PestTrack AI - PyTorch U-Net Leaf Damage Segmentation Architecture
Architecture: Encoder-Decoder with Skip Connections for Semantic Masking
================================================================================
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional
import config
from utils.image_utils import calculate_damage_metrics, generate_segmentation_overlay

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    """
    Standard U-Net Architecture optimized for Crop Leaf Damage Segmentation.
    Encoder extracts hierarchical spatial context; Decoder reconstructs damaged pixel mask.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Decoder (Upsampling with Skip Connections)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)

        # 1x1 Convolution Head with Sigmoid for binary probability mask
        self.outc = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d1 = self.up1(x5)
        d1 = torch.cat([x4, d1], dim=1)
        d1 = self.conv_up1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.conv_up2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.conv_up3(d3)

        d4 = self.up4(d3)
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.conv_up4(d4)

        logits = self.outc(d4)
        return logits

class LeafDamageSegmentor:
    """Inference wrapper for PyTorch U-Net Leaf Damage Mask Generation."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.UNET_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        self.target_size = (256, 256)
        self._load_weights()

    def _load_weights(self):
        """Loads trained weights if available."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"[U-Net] Successfully loaded weights from: {self.model_path}")
            except Exception as e:
                print(f"[U-Net] Error loading weights: {e}. Model initialized.")
                self.model.eval()
        else:
            print(f"[U-Net] Weights file not found at {self.model_path}. Initialized.")
            self.model.eval()

    def segment_leaf(self, cv_image: np.ndarray) -> Dict[str, Any]:
        """
        Generates visible damage mask and computes damage metrics.
        Returns:
            dict containing 'mask', 'overlay', 'damage_percentage', 'severity'
        """
        orig_h, orig_w = cv_image.shape[:2]
        self.model.eval()

        # Preprocess: RGB conversion, resize to 256x256, normalize to [0, 1]
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.target_size)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            prob_map = output[0, 0].cpu().numpy()

        # Threshold probability at 0.45
        binary_mask_small = (prob_map > 0.45).astype(np.uint8) * 255

        # Resize mask back to original image dimensions
        binary_mask = cv2.resize(binary_mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Refine mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        # Compute damage metrics
        metrics = calculate_damage_metrics(cv_image, binary_mask)
        damage_pct = metrics["damage_percentage"]

        # Generate overlay
        overlay = generate_segmentation_overlay(cv_image, binary_mask)

        # Determine severity level from damage percentage
        if damage_pct < 10.0:
            severity = "Low"
        elif damage_pct <= 30.0:
            severity = "Medium"
        else:
            severity = "High"

        return {
            "mask": binary_mask,
            "overlay": overlay,
            "damage_percentage": damage_pct,
            "severity_level": severity,
            "damaged_pixels": metrics["damaged_pixels"],
            "total_pixels": metrics["leaf_pixels"]
        }
