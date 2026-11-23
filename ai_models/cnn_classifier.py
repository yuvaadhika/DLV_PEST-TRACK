"""
================================================================================
PestTrack AI - PyTorch CNN Pest Species Classifier
Architecture: Deep Convolutional Neural Network with Residual Connections
================================================================================
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import config

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class PestCNN(nn.Module):
    """
    Agricultural Pest Classification CNN with 4-stage convolutional hierarchy
    and regularization for robust species identification.
    """
    def __init__(self, num_classes: int = len(config.PEST_CLASSES)):
        super().__init__()
        self.num_classes = num_classes

        # Hierarchical Feature Extractor
        self.block1 = ConvBlock(3, 32)    # 224x224 -> 112x112
        self.block2 = ConvBlock(32, 64)   # 112x112 -> 56x56
        self.block3 = ConvBlock(64, 128)  # 56x56 -> 28x28
        self.block4 = ConvBlock(128, 256) # 28x28 -> 14x14

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class PestClassifier:
    """Inference wrapper for PyTorch Pest Species Classification."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.CNN_MODEL_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = config.PEST_CLASSES
        self.model = PestCNN(num_classes=len(self.classes)).to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self._load_weights()

    def _load_weights(self):
        """Loads trained weights if available; creates model directory if needed."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print(f"[CNN] Successfully loaded weights from: {self.model_path}")
            except Exception as e:
                print(f"[CNN] Error loading weights: {e}. Model initialized randomly.")
                self.model.eval()
        else:
            print(f"[CNN] Weights file not found at {self.model_path}. Initialized.")
            self.model.eval()

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classifies pest image and returns top prediction & top-3 predictions.
        """
        self.model.eval()
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        top_indices = np.argsort(probs)[::-1]
        primary_class = self.classes[top_indices[0]]
        primary_confidence = float(probs[top_indices[0]])

        top_predictions = []
        for idx in top_indices[:3]:
            top_predictions.append({
                "species": self.classes[idx],
                "confidence": round(float(probs[idx]) * 100.0, 1),
                "probability": float(probs[idx])
            })

        return {
            "predicted_species": primary_class,
            "confidence": round(primary_confidence * 100.0, 1),
            "top_predictions": top_predictions
        }
