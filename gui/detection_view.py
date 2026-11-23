"""
================================================================================
PestTrack AI - Model 1: YOLO Pest Detection & Counting View
================================================================================
"""

import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import config
from utils.image_utils import load_image_cv, cv2_to_pil, pil_to_ctk

class DetectionView(ctk.CTkScrollableFrame):
    def __init__(self, parent, controller, preloaded_image=None):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.pipeline = controller.pipeline
        self.db = controller.db

        self.current_image_path = preloaded_image
        self.analysis_result = None

        self._build_ui()

        if self.current_image_path:
            self._load_and_display_original(self.current_image_path)
            self._run_detection()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        header.pack(fill="x", pady=(0, 15), padx=5, ipady=10)

        ctk.CTkLabel(
            header,
            text="🔍 Model 1: YOLO Pest Detection & Counting",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 2))

        ctk.CTkLabel(
            header,
            text="Detect insects in agricultural crop images, locate bounding boxes, identify species, and count density.",
            font=ctk.CTkFont(size=12),
            text_color=config.COLOR_THEME["text_muted"]
        ).pack(anchor="w", padx=20, pady=(0, 6))

        # Controls Toolbar
        ctrl_frame = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        ctrl_frame.pack(fill="x", pady=(0, 15), padx=5, ipady=8)

        self.upload_btn = ctk.CTkButton(
            ctrl_frame,
            text="📁 Select Image",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=config.COLOR_THEME["primary"],
            hover_color=config.COLOR_THEME["primary_hover"],
            command=self._choose_image
        )
        self.upload_btn.pack(side="left", padx=15)

        self.sample_btn = ctk.CTkButton(
            ctrl_frame,
            text="🧪 Use High Infestation Sample",
            font=ctk.CTkFont(size=13),
            fg_color=config.COLOR_THEME["secondary"],
            command=self._use_sample
        )
        self.sample_btn.pack(side="left", padx=5)

        self.run_btn = ctk.CTkButton(
            ctrl_frame,
            text="🚀 Run YOLO Detection",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=config.COLOR_THEME["primary_hover"],
            state="disabled",
            command=self._run_detection
        )
        self.run_btn.pack(side="left", padx=15)

        self.report_btn = ctk.CTkButton(
            ctrl_frame,
            text="📄 Generate PDF Report",
            font=ctk.CTkFont(size=13),
            fg_color="#1976D2",
            hover_color="#1565C0",
            state="disabled",
            command=self._export_report
        )
        self.report_btn.pack(side="right", padx=15)

        # Image Previews Section (Side-by-Side: Original vs YOLO Detection)
        preview_frame = ctk.CTkFrame(self, fg_color="transparent")
        preview_frame.pack(fill="x", pady=(0, 15), padx=5)
        preview_frame.grid_columnconfigure((0, 1), weight=1, uniform="prev")

        # Left: Original Image
        self.orig_card = ctk.CTkFrame(preview_frame, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.orig_card.grid(row=0, column=0, padx=(0, 8), sticky="nsew", ipady=10)

        ctk.CTkLabel(self.orig_card, text="1. Original Crop Image", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=(8, 6))
        self.orig_img_lbl = ctk.CTkLabel(self.orig_card, text="No Image Selected\nClick 'Select Image' above", font=ctk.CTkFont(size=13), text_color="#888888")
        self.orig_img_lbl.pack(pady=40, expand=True)

        # Right: Detection Result
        self.det_card = ctk.CTkFrame(preview_frame, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.det_card.grid(row=0, column=1, padx=(8, 0), sticky="nsew", ipady=10)

        ctk.CTkLabel(self.det_card, text="2. YOLOv8 Pest Detection Result", font=ctk.CTkFont(size=14, weight="bold"), text_color=config.COLOR_THEME["primary"]).pack(anchor="w", padx=15, pady=(8, 6))
        self.det_img_lbl = ctk.CTkLabel(self.det_card, text="Detection pending...\nClick 'Run YOLO Detection'", font=ctk.CTkFont(size=13), text_color="#888888")
        self.det_img_lbl.pack(pady=40, expand=True)

        # Detection Metrics & Summary Bar
        self.metrics_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.metrics_card.pack(fill="x", pady=(0, 15), padx=5, ipady=12)

        self.metrics_lbl = ctk.CTkLabel(
            self.metrics_card,
            text="Detection Summary: Ready to analyze",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=config.COLOR_THEME["text_dark"]
        )
        self.metrics_lbl.pack(anchor="w", padx=20, pady=4)

        self.details_lbl = ctk.CTkLabel(
            self.metrics_card,
            text="Pest Count: 0  |  Primary Species: -  |  Detection Confidence: 0.0%",
            font=ctk.CTkFont(size=13),
            text_color=config.COLOR_THEME["text_muted"]
        )
        self.details_lbl.pack(anchor="w", padx=20, pady=2)

    def _choose_image(self):
        f = filedialog.askopenfilename(
            title="Select Crop Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if f:
            self.current_image_path = f
            self._load_and_display_original(f)
            self.run_btn.configure(state="normal")

    def _use_sample(self):
        sample = os.path.join(config.SAMPLES_DIR, "leaves", "high_infestation_crop.jpg")
        if os.path.exists(sample):
            self.current_image_path = sample
            self._load_and_display_original(sample)
            self.run_btn.configure(state="normal")
            self._run_detection()

    def _load_and_display_original(self, path):
        try:
            pil_img = Image.open(path)
            ctk_img = pil_to_ctk(pil_img, (460, 320))
            self.orig_img_lbl.configure(image=ctk_img, text="")
            self.orig_img_lbl.image = ctk_img
            self.run_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _run_detection(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            return

        try:
            # Run integrated multi-model pipeline
            res = self.pipeline.analyze_full_image(self.current_image_path)
            self.analysis_result = res

            # Update Detection Preview Image
            det_img_path = res["detection_image_path"]
            if os.path.exists(det_img_path):
                pil_det = Image.open(det_img_path)
                ctk_det = pil_to_ctk(pil_det, (460, 320))
                self.det_img_lbl.configure(image=ctk_det, text="")
                self.det_img_lbl.image = ctk_det

            # Update Metrics
            count = res["pest_count"]
            species = res["predicted_species"]
            conf = res["confidence"]
            sev = res["severity_level"]
            dmg = res["damage_percentage"]

            sev_color = config.COLOR_THEME["severity_high"] if sev == "High" else (
                config.COLOR_THEME["severity_medium"] if sev == "Medium" else config.COLOR_THEME["severity_low"]
            )

            self.metrics_lbl.configure(
                text=f"Analysis Complete: Infestation Severity is {sev.upper()}",
                text_color=sev_color
            )
            self.details_lbl.configure(
                text=f"🐛 Total Pests: {count}  |  🔬 Primary Species: {species} ({conf:.1f}%)  |  🍃 Visible Leaf Damage: {dmg:.1f}%"
            )

            self.report_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Detection Error", f"An error occurred during YOLO detection: {e}")

    def _export_report(self):
        if self.analysis_result:
            self.controller.show_view("reports", record_id=self.analysis_result.get("id"))
