"""
================================================================================
PestTrack AI - Model 2: PyTorch CNN Pest Species Classification View
================================================================================
"""

import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import config
from utils.image_utils import pil_to_ctk

class ClassificationView(ctk.CTkScrollableFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.classifier = controller.pipeline.classifier

        self.current_image_path = None
        self._build_ui()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        header.pack(fill="x", pady=(0, 15), padx=5, ipady=10)

        ctk.CTkLabel(
            header,
            text="🐛 Model 2: PyTorch CNN Pest Species Classifier",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 2))

        ctk.CTkLabel(
            header,
            text="Classify agricultural insect species using deep convolutional feature hierarchy with top-3 probability distribution.",
            font=ctk.CTkFont(size=12),
            text_color=config.COLOR_THEME["text_muted"]
        ).pack(anchor="w", padx=20, pady=(0, 6))

        # Controls & Sample Selection
        ctrl_frame = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        ctrl_frame.pack(fill="x", pady=(0, 15), padx=5, ipady=8)

        ctk.CTkButton(
            ctrl_frame,
            text="📁 Upload Pest Image",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=config.COLOR_THEME["primary"],
            hover_color=config.COLOR_THEME["primary_hover"],
            command=self._choose_image
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            ctrl_frame,
            text="📋 Paste (Ctrl+V)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0288D1",
            hover_color="#0277BD",
            command=self._paste_image
        ).pack(side="left", padx=5)

        ctk.CTkLabel(ctrl_frame, text="Quick Samples:", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=(10, 5))

        sample_pests = ["aphids", "beetle", "armyworm", "grasshopper", "whitefly"]
        for sp in sample_pests:
            ctk.CTkButton(
                ctrl_frame,
                text=sp.title(),
                width=80,
                height=32,
                font=ctk.CTkFont(size=12),
                fg_color=config.COLOR_THEME["sidebar_light"],
                text_color=config.COLOR_THEME["primary"],
                hover_color=config.COLOR_THEME["accent"],
                command=lambda name=sp: self._load_sample_pest(name)
            ).pack(side="left", padx=4)

        # Content Area: Left Image Preview, Right Predictions
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="x", pady=(0, 15), padx=5)
        content.grid_columnconfigure(0, weight=4, uniform="c")
        content.grid_columnconfigure(1, weight=5, uniform="c")

        # Left Image Preview
        img_card = ctk.CTkFrame(content, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        img_card.grid(row=0, column=0, padx=(0, 8), sticky="nsew", ipady=15)

        ctk.CTkLabel(img_card, text="Pest Image Preview", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=(8, 10))
        self.pest_img_lbl = ctk.CTkLabel(img_card, text="Select a pest image or sample above", font=ctk.CTkFont(size=13), text_color="#888888")
        self.pest_img_lbl.pack(pady=40, expand=True)

        # Right Classification Results
        self.result_card = ctk.CTkFrame(content, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.result_card.grid(row=0, column=1, padx=(8, 0), sticky="nsew", ipady=15)

        ctk.CTkLabel(self.result_card, text="Species Prediction Results", font=ctk.CTkFont(size=14, weight="bold"), text_color=config.COLOR_THEME["primary"]).pack(anchor="w", padx=15, pady=(8, 10))

        self.pred_badge = ctk.CTkLabel(
            self.result_card,
            text="Predicted Species: None",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=config.COLOR_THEME["text_dark"]
        )
        self.pred_badge.pack(anchor="w", padx=15, pady=(5, 2))

        self.conf_lbl = ctk.CTkLabel(
            self.result_card,
            text="Classification Confidence: 0.0%",
            font=ctk.CTkFont(size=14),
            text_color=config.COLOR_THEME["text_muted"]
        )
        self.conf_lbl.pack(anchor="w", padx=15, pady=(0, 15))

        # Top 3 Predictions Bars
        ctk.CTkLabel(self.result_card, text="Top Predictions Distribution:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=15, pady=(10, 4))
        self.bar_frames = []
        for _ in range(3):
            b_frame = ctk.CTkFrame(self.result_card, fg_color="transparent")
            b_frame.pack(fill="x", padx=15, pady=4)
            name_lbl = ctk.CTkLabel(b_frame, text="-", width=110, anchor="w", font=ctk.CTkFont(size=12))
            name_lbl.pack(side="left")
            p_bar = ctk.CTkProgressBar(b_frame, height=14, corner_radius=4, progress_color=config.COLOR_THEME["primary"])
            p_bar.set(0)
            p_bar.pack(side="left", fill="x", expand=True, padx=8)
            pct_lbl = ctk.CTkLabel(b_frame, text="0.0%", width=50, anchor="e", font=ctk.CTkFont(size=12))
            pct_lbl.pack(side="right")
            self.bar_frames.append((name_lbl, p_bar, pct_lbl))

    def _choose_image(self):
        f = filedialog.askopenfilename(
            title="Select Pest Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if f:
            self._analyze_file(f)

    def _paste_image(self):
        """Pastes image directly from clipboard."""
        from utils.image_utils import get_clipboard_image
        clip = get_clipboard_image()
        if clip is not None:
            pil_img, name = clip
            self.on_paste_image(pil_img, name)
        else:
            messagebox.showinfo("Clipboard Empty", "No image found on clipboard! Press Ctrl+C on any image, then click Paste.")

    def on_paste_image(self, pil_img: Image.Image, name: str):
        """Handler for Ctrl+V paste."""
        import datetime
        temp_dir = os.path.join(config.DATA_DIR, "exports", "detections")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"paste_cls_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        pil_img.save(temp_path)
        self._analyze_file(temp_path)

    def _load_sample_pest(self, pest_name):
        sample_path = os.path.join(config.SAMPLES_DIR, "pests", f"{pest_name}_sample.jpg")
        if os.path.exists(sample_path):
            self._analyze_file(sample_path)
        else:
            self._choose_image()

    def _analyze_file(self, path):
        self.current_image_path = path
        try:
            pil_img = Image.open(path)
            ctk_img = pil_to_ctk(pil_img, (360, 260))
            self.pest_img_lbl.configure(image=ctk_img, text="")
            self.pest_img_lbl.image = ctk_img

            # Run PyTorch CNN Inference
            res = self.classifier.predict(pil_img)
            species = res["predicted_species"]
            conf = res["confidence"]
            top_preds = res["top_predictions"]

            self.pred_badge.configure(
                text=f"Predicted Species: {species}",
                text_color=config.COLOR_THEME["primary"]
            )
            self.conf_lbl.configure(text=f"Classification Confidence: {conf:.1f}%")

            # Update top predictions bars
            for i in range(min(3, len(top_preds))):
                item = top_preds[i]
                n_lbl, p_bar, pct_lbl = self.bar_frames[i]
                n_lbl.configure(text=item["species"])
                p_bar.set(item["probability"])
                pct_lbl.configure(text=f"{item['confidence']:.1f}%")

        except Exception as e:
            messagebox.showerror("Classification Error", f"Error classifying pest: {e}")
