"""
================================================================================
PestTrack AI - Stage 1 Environment Setup Verification Script
Author: PestTrack AI Project Team
Description: Validates Python, PyTorch, CUDA GPU, Ultralytics YOLO, OpenCV,
             CustomTkinter, SQLite3, and supporting libraries.
================================================================================
"""

import sys
import os
import platform

def print_separator(title=""):
    if title:
        print(f"\n{'=' * 20} {title} {'=' * 20}")
    else:
        print("=" * 60)

def test_environment():
    print_separator("PESTTRACK AI: STAGE 1 VERIFICATION")
    print(f"[*] Operating System : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"[*] Working Directory: {os.getcwd()}")
    print(f"[*] Python Executable: {sys.executable}")
    print(f"[*] Python Version   : {sys.version.split()[0]}")

    results = []

    # 1. Check Python Version (Prefer 3.10 or 3.11)
    if sys.version_info >= (3, 10):
        results.append(("Python Version (>= 3.10)", True, f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))
    else:
        results.append(("Python Version (>= 3.10)", False, f"Detected older Python {sys.version_info.major}.{sys.version_info.minor}"))

    # 2. Check PyTorch & CUDA / GPU
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_info = "CPU Only"
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            gpu_info = f"CUDA Enabled ({gpu_name}, {vram_gb} GB VRAM)"
        results.append(("PyTorch Core", True, f"v{torch.__version__} - {gpu_info}"))
    except ImportError as e:
        results.append(("PyTorch Core", False, f"Import error: {e}"))

    # 3. Check Torchvision
    try:
        import torchvision
        results.append(("Torchvision", True, f"v{torchvision.__version__}"))
    except ImportError as e:
        results.append(("Torchvision", False, f"Import error: {e}"))

    # 4. Check Ultralytics (YOLO)
    try:
        import ultralytics
        results.append(("Ultralytics (YOLO)", True, f"v{ultralytics.__version__}"))
    except ImportError as e:
        results.append(("Ultralytics (YOLO)", False, f"Import error: {e}"))

    # 5. Check OpenCV
    try:
        import cv2
        results.append(("OpenCV (Computer Vision)", True, f"v{cv2.__version__}"))
    except ImportError as e:
        results.append(("OpenCV (Computer Vision)", False, f"Import error: {e}"))

    # 6. Check NumPy
    try:
        import numpy as np
        results.append(("NumPy", True, f"v{np.__version__}"))
    except ImportError as e:
        results.append(("NumPy", False, f"Import error: {e}"))

    # 7. Check CustomTkinter & Tkinter GUI
    try:
        import customtkinter as ctk
        import tkinter as tk
        results.append(("CustomTkinter (GUI)", True, f"v{ctk.__version__} (Tk v{tk.TkVersion})"))
    except ImportError as e:
        results.append(("CustomTkinter (GUI)", False, f"Import error: {e}"))

    # 8. Check Pillow (PIL)
    try:
        import PIL
        results.append(("Pillow (Image Processing)", True, f"v{PIL.__version__}"))
    except ImportError as e:
        results.append(("Pillow (Image Processing)", False, f"Import error: {e}"))

    # 9. Check Matplotlib
    try:
        import matplotlib
        results.append(("Matplotlib (Charts & Stats)", True, f"v{matplotlib.__version__}"))
    except ImportError as e:
        results.append(("Matplotlib (Charts & Stats)", False, f"Import error: {e}"))

    # 10. Check ReportLab (PDF Reports)
    try:
        import reportlab
        results.append(("ReportLab (PDF Generator)", True, f"v{reportlab.__version__}"))
    except ImportError as e:
        results.append(("ReportLab (PDF Generator)", False, f"Import error: {e}"))

    # 11. Check SQLite3 (Database)
    try:
        import sqlite3
        con = sqlite3.connect(":memory:")
        cur = con.cursor()
        cur.execute("CREATE TABLE test (id INT);")
        cur.execute("INSERT INTO test VALUES (1);")
        con.commit()
        con.close()
        results.append(("SQLite3 (Database Engine)", True, f"v{sqlite3.sqlite_version} (In-memory test passed)"))
    except Exception as e:
        results.append(("SQLite3 (Database Engine)", False, f"SQLite error: {e}"))

    # 12. Check PyInstaller
    try:
        import PyInstaller
        results.append(("PyInstaller (EXE Packager)", True, f"v{PyInstaller.__version__}"))
    except ImportError as e:
        results.append(("PyInstaller (EXE Packager)", False, f"Import error: {e}"))

    # Display Report
    print_separator("VERIFICATION RESULTS")
    all_passed = True
    for component, passed, detail in results:
        status_tag = "[PASS]" if passed else "[FAIL]"
        print(f" {status_tag:<8} | {component:<26} | {detail}")
        if not passed:
            all_passed = False

    print_separator()
    if all_passed:
        print("[SUCCESS] All dependencies verified successfully!")
        print("[READY]   Your environment is 100% prepared for Stage 2 (Folder Structure & GUI)!")
    else:
        print("[WARNING] Some packages are missing or failed verification.")
        print("[ACTION]  Install the missing packages using the instructions provided in Stage 1.")
    print_separator()
    return all_passed

if __name__ == "__main__":
    success = test_environment()
    sys.exit(0 if success else 1)
