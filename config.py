"""
================================================================================
PestTrack AI - Configuration & Constants
================================================================================
"""

import os
import sys

# Project Metadata
APP_NAME = "PestTrack AI"
APP_TITLE = "Intelligent Pest Infestation Monitoring & Crop Damage Analysis System"
APP_VERSION = "2.0.0"
AUTHOR = "2nd Year AIML - Deep Learning for Vision"

# Paths
def get_base_dir():
    """Returns application base directory, handling PyInstaller 6 _internal and _MEIPASS."""
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        internal_dir = os.path.join(exe_dir, "_internal")
        if os.path.exists(os.path.join(internal_dir, "models")):
            return internal_dir
        if os.path.exists(os.path.join(exe_dir, "models")):
            return exe_dir
        return getattr(sys, '_MEIPASS', exe_dir)
    return os.path.dirname(os.path.abspath(__file__))

def get_data_dir():
    """Returns persistent writable directory for user data, database, and exports."""
    # Store user data next to executable or in project root, not in temp _MEIPASS
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        data_dir = os.path.join(exe_dir, "PestTrack_Data")
    else:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

BASE_DIR = get_base_dir()
DATA_DIR = get_data_dir()

# Set Ultralytics config directory locally to prevent Windows AppData permission issues
YOLO_DIR = os.path.join(DATA_DIR, ".ultralytics")
os.makedirs(YOLO_DIR, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = YOLO_DIR
os.environ["MPLCONFIGDIR"] = os.path.join(DATA_DIR, ".matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Model Weights Paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8_pest.pt")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "pest_classifier_cnn.pth")
UNET_MODEL_PATH = os.path.join(MODELS_DIR, "leaf_damage_unet.pth")

# Database Path
DB_DIR = os.path.join(DATA_DIR, "database")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "pesttrack.db")

# Exports & Samples Paths
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
REPORTS_DIR = os.path.join(EXPORTS_DIR, "reports")
DETECTIONS_DIR = os.path.join(EXPORTS_DIR, "detections")
SEGMENTATIONS_DIR = os.path.join(EXPORTS_DIR, "segmentations")
VIDEOS_DIR = os.path.join(EXPORTS_DIR, "videos")
SAMPLES_DIR = os.path.join(BASE_DIR, "sample_data")

for d in [EXPORTS_DIR, REPORTS_DIR, DETECTIONS_DIR, SEGMENTATIONS_DIR, VIDEOS_DIR]:
    os.makedirs(d, exist_ok=True)

# Theme Colors (Modern Agriculture Green & White Theme)
COLOR_THEME = {
    "primary": "#2E7D32",         # Forest Green
    "primary_hover": "#1B5E20",   # Deep Green
    "secondary": "#4CAF50",       # Emerald Green
    "accent": "#81C784",          # Light Green
    "background_light": "#F4F7F5",# Soft Light Grey-Green
    "background_dark": "#1A221E", # Deep Slate Green
    "card_light": "#FFFFFF",      # Pure White
    "card_dark": "#232D28",       # Dark Slate Card
    "sidebar_light": "#E8F5E9",   # Mint Soft Sidebar
    "sidebar_dark": "#161D19",    # Night Sidebar
    "text_dark": "#1C2826",       # Dark Charcoal
    "text_light": "#ECEFF1",      # Off White
    "text_muted": "#607D8B",      # Slate Grey Muted
    "severity_low": "#2E7D32",    # Green
    "severity_medium": "#F57C00", # Orange / Amber
    "severity_high": "#C62828",   # Crimson Red
    "border_light": "#DCE7DE",
    "border_dark": "#334139"
}

# Pest Classification & Detection Classes (10 Agricultural Pest Species)
PEST_CLASSES = [
    "Aphids",
    "Armyworm",
    "Beetle",
    "Bollworm",
    "Grasshopper",
    "Leafhopper",
    "Mealybug",
    "Mites",
    "Stem_Borer",
    "Whitefly"
]

# Infestation Severity Rule Engine Thresholds
SEVERITY_RULES = {
    "low": {"max_pests": 2, "max_damage_pct": 10.0},
    "medium": {"min_pests": 3, "max_pests": 7, "min_damage_pct": 10.0, "max_damage_pct": 30.0},
    "high": {"min_pests": 8, "min_damage_pct": 30.0}
}

# Yield Loss Scientific Disclaimer
DISCLAIMER_TEXT = (
    "Scientific Notice: Visible leaf damage segmentation and pest density represent visual "
    "infestation severity at the inspection moment and do not constitute a direct guarantee "
    "or linear measurement of eventual crop yield loss."
)
