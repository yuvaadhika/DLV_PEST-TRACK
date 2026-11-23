"""
================================================================================
PestTrack AI - Path Helper for Development and PyInstaller Packaging
================================================================================
"""

import sys
import os

def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and for PyInstaller bundle.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return os.path.join(base_path, relative_path)

def ensure_dir(path: str) -> str:
    """Ensures directory exists and returns it."""
    os.makedirs(path, exist_ok=True)
    return path
