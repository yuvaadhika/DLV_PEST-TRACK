"""
================================================================================
PestTrack AI - Model 3: PyTorch U-Net Leaf Damage Segmentation View
================================================================================
"""

import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import config
from utils.image_utils import load_image_cv, cv2_to_pil, pil_to_ctk

class SegmentationView(ctk.CTkScrollableFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.segmentor = controller.pipeline.segmentor

        self.current_image_path = None
        self._build_ui()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        header.pack(fill="x", pady=(0, 15), padx=5, ipady=10)

        ctk.CTkLabel(
            header,
            text="🍃 Model 3: PyTorch U-Net Leaf Damage Segmentation",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 2))

        ctk.CTkLabel(
            header,
            text="Pixel-level semantic segmentation of visible necrotic damage, chlorosis, and insect chewing holes.",
            font=ctk.CTkFont(size=12),
            text_color=config.COLOR_THEME["text_muted"]
        ).pack(anchor="w", padx=20, pady=(0, 6))

        # Controls Toolbar
        ctrl_frame = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        ctrl_frame.pack(fill="x", pady=(0, 15), padx=5, ipady=8)

        ctk.CTkButton(
            ctrl_frame,
            text="📁 Select Leaf Image",
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

        ctk.CTkButton(
            ctrl_frame,
            text="🌿 Low Damage Sample",
            font=ctk.CTkFont(size=13),
            fg_color=config.COLOR_THEME["secondary"],
            command=lambda: self._load_sample("low_damage_leaf.jpg")
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            ctrl_frame,
            text="🍂 High Damage Sample",
            font=ctk.CTkFont(size=13),
            fg_color="#D32F2F",
            hover_color="#B71C1C",
            command=lambda: self._load_sample("high_infestation_crop.jpg")
        ).pack(side="left", padx=5)

        # Metrics Card (Damage % and Severity)
        self.metrics_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.metrics_card.pack(fill="x", pady=(0, 15), padx=5, ipady=10)

        self.dmg_pct_lbl = ctk.CTkLabel(
            self.metrics_card,
            text="Visible Damaged-Area Percentage: 0.0 %",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        )
        self.dmg_pct_lbl.pack(anchor="w", padx=20, pady=2)

        self.sev_lbl = ctk.CTkLabel(
            self.metrics_card,
            text="Infestation Severity Level: Low",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=config.COLOR_THEME["severity_low"]
        )
        self.sev_lbl.pack(anchor="w", padx=20, pady=2)

        # 3-Panel Side-by-Side Comparison Previews
        previews = ctk.CTkFrame(self, fg_color="transparent")
        previews.pack(fill="x", pady=(0, 15), padx=5)
        previews.grid_columnconfigure((0, 1, 2), weight=1, uniform="prev")

        # 1. Original
        card1 = ctk.CTkFrame(previews, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        card1.grid(row=0, column=0, padx=4, sticky="nsew", ipady=8)
        ctk.CTkLabel(card1, text="1. Original Crop Leaf", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=4)
        self.img1_lbl = ctk.CTkLabel(card1, text="No Leaf Loaded", font=ctk.CTkFont(size=12), text_color="#888888")
        self.img1_lbl.pack(pady=30, expand=True)

        # 2. Damage Mask
        card2 = ctk.CTkFrame(previews, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        card2.grid(row=0, column=1, padx=4, sticky="nsew", ipady=8)
        ctk.CTkLabel(card2, text="2. U-Net Damage Mask", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=4)
        self.img2_lbl = ctk.CTkLabel(card2, text="Awaiting Analysis", font=ctk.CTkFont(size=12), text_color="#888888")
        self.img2_lbl.pack(pady=30, expand=True)

        # 3. Highlighted Overlay
        card3 = ctk.CTkFrame(previews, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        card3.grid(row=0, column=2, padx=4, sticky="nsew", ipady=8)
        ctk.CTkLabel(card3, text="3. Segmented Damage Overlay", font=ctk.CTkFont(size=13, weight="bold"), text_color=config.COLOR_THEME["primary"]).pack(pady=4)
        self.img3_lbl = ctk.CTkLabel(card3, text="Awaiting Analysis", font=ctk.CTkFont(size=12), text_color="#888888")
        self.img3_lbl.pack(pady=30, expand=True)

        # Scientific Disclaimer Box
        disc_frame = ctk.CTkFrame(self, fg_color="#FFFDE7", border_color="#FFF59D", border_width=1, corner_radius=8)
        disc_frame.pack(fill="x", padx=5, pady=(5, 10), ipady=8)
        ctk.CTkLabel(
            disc_frame,
            text=f"⚠️ {config.DISCLAIMER_TEXT}",
            font=ctk.CTkFont(size=11),
            text_color="#616161",
            wraplength=850
        ).pack(padx=15, pady=4)

    def _choose_image(self):
        f = filedialog.askopenfilename(
            title="Select Crop Leaf Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if f:
            self._run_segmentation(f)

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
        temp_dir = os.path.join(config.DATA_DIR, "exports", "segmentations")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"paste_seg_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        pil_img.save(temp_path)
        self._run_segmentation(temp_path)

    def _load_sample(self, fname):
        sample_path = os.path.join(config.SAMPLES_DIR, "leaves", fname)
        if os.path.exists(sample_path):
            self._run_segmentation(sample_path)
        else:
            self._choose_image()

    def _run_segmentation(self, img_path):
        self.current_image_path = img_path
        try:
            cv_img = load_image_cv(img_path)
            if cv_img is None:
                return

            # Display Original
            pil_orig = cv2_to_pil(cv_img)
            ctk_orig = pil_to_ctk(pil_orig, (300, 220))
            self.img1_lbl.configure(image=ctk_orig, text="")
            self.img1_lbl.image = ctk_orig

            # Run PyTorch U-Net Segmentation
            res = self.segmentor.segment_leaf(cv_img)
            mask = res["mask"]
            overlay = res["overlay"]
            dmg_pct = res["damage_percentage"]
            sev = res["severity_level"]

            # Display Mask & Overlay
            pil_mask = Image.fromarray(mask)
            ctk_mask = pil_to_ctk(pil_mask, (300, 220))
            self.img2_lbl.configure(image=ctk_mask, text="")
            self.img2_lbl.image = ctk_mask

            pil_overlay = cv2_to_pil(overlay)
            ctk_overlay = pil_to_ctk(pil_overlay, (300, 220))
            self.img3_lbl.configure(image=ctk_overlay, text="")
            self.img3_lbl.image = ctk_overlay

            # Update Metrics
            sev_color = config.COLOR_THEME["severity_high"] if sev == "High" else (
                config.COLOR_THEME["severity_medium"] if sev == "Medium" else config.COLOR_THEME["severity_low"]
            )
            self.dmg_pct_lbl.configure(text=f"Visible Damaged-Area Percentage: {dmg_pct:.2f} %")
            self.sev_lbl.configure(
                text=f"Infestation Severity Level: {sev.upper()}",
                text_color=sev_color
            )

        except Exception as e:
            messagebox.showerror("Segmentation Error", f"Failed to segment leaf damage: {e}")
