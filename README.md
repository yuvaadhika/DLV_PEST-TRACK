# 🌱 PestTrack AI: Intelligent Pest Infestation Monitoring and Crop Damage Analysis System Using Deep Learning

**Course:** 2nd Year B.Tech / B.E. - Artificial Intelligence & Machine Learning (AIML)  
**Subject:** Deep Learning for Vision (DLV) - Real-Time Case Study Project  
**Author:** Yuvaadhika  
**Platform:** Windows Desktop Application (100% Offline, No Web/Browser)  

---

## 🌟 Executive Summary

**PestTrack AI** is an advanced computer vision desktop application built for agriculture field pest monitoring and crop damage evaluation. The software integrates the three fundamental pillars of modern computer vision into a unified multi-model inference pipeline:
1. **Object Detection:** Real-time insect pest localization and density counting using **YOLOv8**.
2. **Image Classification:** Pest species categorization across 10 agricultural classes using a **Deep PyTorch Convolutional Neural Network (CNN)**.
3. **Semantic Segmentation:** Pixel-level visible leaf necrotic and chlorotic lesion masking using a symmetric **PyTorch U-Net** (Encoder-Decoder with Skip Connections).
4. **Infestation Severity Matrix:** Automated rule engine classifying crop damage into **Low, Medium, or High Infestation** based on pest count and segmented leaf damage percentage.
5. **Report Generation:** Automated agronomic PDF inspection reports using **ReportLab**.
6. **Local Storage:** Zero-configuration persistent local storage using **SQLite3**.

---

## 🚀 How to Run the Software

### Option 1: 1-Click Direct Desktop Launch (Instant, No Lag)
Double-click the file in the project folder:
```text
Launch_PestTrack_AI.bat
```
*(Runs immediately with zero terminal window popup)*

### Option 2: Standalone Windows Executable (.EXE)
Open the folder `dist\PestTrack_AI` and double-click:
```text
dist\PestTrack_AI\PestTrack_AI.exe
```
*(Portable standalone distribution package ready to share on any Windows PC without Python installed)*

---

## 🧠 AI Model Architectures & Deep Learning Vision (DLV) Details

### 1. Model 1: YOLOv8 (Pest Detection & Counting)
- **Framework:** Ultralytics YOLOv8 (Anchor-Free Single-Stage Detector)
- **Model Path:** `models/yolov8_pest.pt`
- **Features:**
  - Real-time bounding box localization (`[x1, y1, x2, y2]`).
  - Pest density counting across static crop images and live video streams.
  - Confidence scoring thresholding (default $\tau = 0.25$).

### 2. Model 2: PyTorch CNN (Pest Species Classification)
- **Framework:** PyTorch (`torch.nn`)
- **Model Path:** `models/pest_classifier_cnn.pth`
- **Architecture:** 4-Stage Convolutional Hierarchy with BatchNorm, ReLU, Dropout (0.4 / 0.3), and Adaptive Average Pooling.
- **Classes (10 Agricultural Pests):**
  `['Aphids', 'Armyworm', 'Beetle', 'Bollworm', 'Grasshopper', 'Leafhopper', 'Mealybug', 'Mites', 'Stem_Borer', 'Whitefly']`
- **Output:** Primary predicted species, top confidence %, and Top-3 probability distribution.

### 3. Model 3: PyTorch U-Net (Leaf Damage Semantic Segmentation)
- **Framework:** PyTorch (`torch.nn`)
- **Model Path:** `models/leaf_damage_unet.pth`
- **Architecture:** Symmetric 4-level contracting path (Encoder) + bottleneck + 4-level expanding path (Decoder) with skip-connections and 1x1 output convolution with Sigmoid activation.
- **Loss Function:** Binary Cross-Entropy (BCE) + Dice Loss ($L = L_{BCE} + L_{Dice}$).
- **Formula:**
  $$\text{Visible Damage \%} = \frac{\text{Damaged Pixels}}{\text{Total Leaf Surface Pixels}} \times 100$$

---

## ⚖️ Infestation Severity Rule Engine

| Severity Level | Pest Count Threshold | Leaf Damage Area (%) | Recommended Agronomic Action |
| :---: | :---: | :---: | :--- |
| **LOW** | $\le 2$ insects | $< 10.0\%$ | Safe condition; continue standard farm hygiene and weekly monitoring. |
| **MEDIUM** | $3 - 7$ insects | $10.0\% - 30.0\%$ | Active monitoring; apply biological/organic repellent spray within 48h. |
| **HIGH** | $\ge 8$ insects | $> 30.0\%$ | Critical infestation; immediate targeted bio-pesticide intervention required. |

> **Scientific Notice:** Visible leaf damage segmentation and pest density represent visual infestation severity at the inspection moment and do not constitute a direct guarantee or linear measurement of eventual crop yield loss.

---

## 📁 Project Directory Structure

```text
PEST TRACK/
├── dist/
│   └── PestTrack_AI/
│       └── PestTrack_AI.exe     # Standalone Windows Executable
├── Launch_PestTrack_AI.bat      # 1-Click Instant Desktop Launcher
├── app.py                       # Main application entry point
├── config.py                    # Global configurations, colors & thresholds
├── PestTrack_AI.spec            # PyInstaller build specification
├── verify_setup.py              # 12-point environment validator
├── requirements.txt             # Dependencies
├── gui/                         # CustomTkinter Desktop Views
│   ├── main_window.py           # Sidebar & Navigation Manager
│   ├── home_view.py             # Dashboard Overview & Stats
│   ├── detection_view.py        # YOLO Pest Detection & Bounding Boxes
│   ├── classification_view.py   # PyTorch CNN Species Classification
│   ├── segmentation_view.py     # PyTorch U-Net Leaf Damage Masking
│   ├── video_view.py            # Real-Time Video Analysis
│   ├── history_view.py          # SQLite Inspection Records Table
│   ├── reports_view.py          # PDF Report Viewer & Downloader
│   └── about_view.py            # Viva & Technical Documentation
├── ai_models/                   # Deep Learning Vision Models
│   ├── yolo_detector.py         # YOLOv8 Detector Wrapper
│   ├── cnn_classifier.py        # PyTorch CNN Architecture & Weights Loader
│   ├── unet_segmentor.py        # PyTorch U-Net Architecture & Weights Loader
│   └── pipeline.py              # Multi-Model Pipeline & Severity Scorer
├── models/                      # Saved Model Weights (.pt, .pth)
│   ├── yolov8_pest.pt
│   ├── pest_classifier_cnn.pth
│   └── leaf_damage_unet.pth
├── database/                    # SQLite Database Manager
│   ├── db_manager.py
│   └── pesttrack.db
├── sample_data/                 # Test Samples for Viva / Demonstration
│   ├── leaves/                  # Healthy & Damaged Leaf Samples
│   ├── pests/                   # 10 Sample Pest Species Images
│   └── sample_pest_video.mp4    # Sample Crop Video for Video Analysis
├── utils/                       # Supporting Utilities
│   ├── image_utils.py           # OpenCV & PIL Image Handlers
│   ├── pdf_generator.py         # ReportLab PDF Report Engine
│   └── path_helper.py           # PyInstaller Resource Path Resolver
└── exports/                     # Generated Reports, Detections, & Masks
    ├── reports/
    ├── detections/
    ├── segmentations/
    └── videos/
```

---

## 🎯 Viva Presentation Q&A Quick Guide

**Q1: Why did you choose 3 different AI models instead of one?**  
*Answer:* In real-world agricultural monitoring, a single model cannot solve the complete vision pipeline. YOLO is specialized in spatial localization and density counting (Object Detection). CNN is specialized in fine-grained insect taxonomy across species (Image Classification). U-Net is specialized in pixel-precise area masking and lesion measurement (Semantic Segmentation). Combining them provides a complete clinical diagnosis of crop health.

**Q2: What is the purpose of U-Net skip connections in crop leaf damage?**  
*Answer:* Contracting (encoder) layers lose fine boundary details of small necrotic spots during max-pooling. Skip connections copy high-resolution spatial feature maps directly from the encoder to the corresponding decoder stage, ensuring exact boundary delineations for calculating damage percentages.

**Q3: How does the software ensure farmer privacy and operate offline?**  
*Answer:* The software is packaged as a pure Windows executable (`PestTrack_AI.exe`) with local SQLite storage. All AI models run locally on the device (leveraging GPU acceleration if available, or CPU fallback) without requiring any cloud server, internet connectivity, or web browser.
