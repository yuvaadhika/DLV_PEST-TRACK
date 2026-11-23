"""
================================================================================
PestTrack AI - Intelligent Pest Infestation Monitoring System
Academic Case Study Project: 2nd Year AIML - Deep Learning for Vision (DLV)
================================================================================
Entry Point for Windows Desktop Application & Executable
"""

import sys
import os
import multiprocessing

# Add base directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from gui.main_window import MainWindow

def main():
    # Required for PyInstaller Windows multiprocessing / PyTorch freeze support
    multiprocessing.freeze_support()

    print("=====================================================================")
    print("           PestTrack AI - Intelligent Crop Monitoring System          ")
    print("      Deep Learning for Vision (DLV) - Windows Desktop Software      ")
    print("=====================================================================")

    try:
        app = MainWindow()
        app.mainloop()
    except Exception as e:
        import traceback
        print(f"\n[CRITICAL ERROR] Application encountered an error: {e}")
        traceback.print_exc()
        try:
            from tkinter import messagebox
            messagebox.showerror("PestTrack AI - Runtime Error", f"Application error:\n{e}")
        except Exception:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
