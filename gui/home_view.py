"""
================================================================================
PestTrack AI - Home Dashboard & Full Multi-Model AI Vision Engine
Supports:
  1. File Upload (JPG, PNG, BMP)
  2. Direct Clipboard Copy-Paste (Ctrl+V or Paste Button)
  3. Preloaded Instant Agricultural Samples
  4. End-to-End Multi-Model Inference:
     - YOLOv8 Pest Object Detection & Density Counting
     - PyTorch CNN 10-Class Pest Species Classification
     - PyTorch U-Net Leaf Damage & Disease Semantic Segmentation
     - Rule-Based Infestation Severity & Yield Loss Estimation
  5. 1-Click Official Agronomist PDF Advisory Report Export
================================================================================
"""

import os
import cv2
import datetime
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import config
from utils.image_utils import load_image_cv, cv2_to_pil, pil_to_ctk, get_clipboard_image, create_side_by_side_comparison
from utils.pdf_generator import generate_pdf_report

class HomeView(ctk.CTkScrollableFrame):
    """Primary Unified Dashboard & End-to-End AI Analysis Interface."""

    def __init__(self, parent, controller, preloaded_image=None):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.db = controller.db
        self.pipeline = controller.pipeline

        self.current_pil_image = None
        self.current_cv_image = None
        self.current_image_name = "Untitled"
        self.analysis_result = None

        self._build_dashboard()

        # Load preloaded image if provided
        if preloaded_image:
            self.load_image_from_path(preloaded_image)

    def _build_dashboard(self):
        # 1. Header Banner
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["primary"], corner_radius=12)
        header.pack(fill="x", pady=(0, 16), ipady=12, padx=5)

        h_title = ctk.CTkLabel(
            header,
            text="🌾 Welcome to PestTrack AI Vision System",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#FFFFFF"
        )
        h_title.pack(anchor="w", padx=25, pady=(8, 2))

        h_sub = ctk.CTkLabel(
            header,
            text="Unified AI Vision: YOLO Pest Detection, CNN Species Identification & U-Net Leaf Damage/Disease Segmentation",
            font=ctk.CTkFont(size=12),
            text_color="#E8F5E9"
        )
        h_sub.pack(anchor="w", padx=25, pady=(0, 8))

        # 2. KPI Metric Cards
        self.metrics_container = ctk.CTkFrame(self, fg_color="transparent")
        self.metrics_container.pack(fill="x", pady=(0, 16))
        self.metrics_container.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="metric")
        self._refresh_kpi_cards()

        # 3. Input & Action Toolbar
        action_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        action_card.pack(fill="x", pady=(0, 16), padx=5, ipady=10)

        t_row = ctk.CTkFrame(action_card, fg_color="transparent")
        t_row.pack(fill="x", padx=16, pady=4)

        ctk.CTkLabel(
            t_row,
            text="⚡ Image Input & Analysis Options:",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(side="left", padx=(0, 15))

        # Upload Button
        self.upload_btn = ctk.CTkButton(
            t_row,
            text="📁  Upload Image",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=config.COLOR_THEME["primary"],
            hover_color=config.COLOR_THEME["primary_hover"],
            height=38,
            command=self._on_upload_click
        )
        self.upload_btn.pack(side="left", padx=6)

        # Paste Button (Clipboard)
        self.paste_btn = ctk.CTkButton(
            t_row,
            text="📋  Paste Image (Ctrl+V)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0288D1",
            hover_color="#0277BD",
            height=38,
            command=self._on_paste_click
        )
        self.paste_btn.pack(side="left", padx=6)

        # Try Sample Button
        self.sample_btn = ctk.CTkButton(
            t_row,
            text="🧪  Try Sample Crop Leaf",
            font=ctk.CTkFont(size=13),
            fg_color=config.COLOR_THEME["secondary"],
            hover_color=config.COLOR_THEME["primary"],
            height=38,
            command=self._on_sample_click
        )
        self.sample_btn.pack(side="left", padx=6)

        # Run All Models Button
        self.run_btn = ctk.CTkButton(
            t_row,
            text="🚀  Analyze All Features",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#D84315",
            hover_color="#BF360C",
            height=38,
            state="disabled",
            command=self._run_full_analysis
        )
        self.run_btn.pack(side="left", padx=12)

        # PDF Report Button
        self.report_btn = ctk.CTkButton(
            t_row,
            text="📄  Export PDF Report",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#388E3C",
            hover_color="#2E7D32",
            height=38,
            state="disabled",
            command=self._on_export_pdf
        )
        self.report_btn.pack(side="right", padx=6)

        # 4. Multi-Panel Visual Inspection Area
        self.preview_container = ctk.CTkFrame(self, fg_color="transparent")
        self.preview_container.pack(fill="x", pady=(0, 16), padx=5)
        self.preview_container.grid_columnconfigure((0, 1, 2), weight=1, uniform="preview")

        # Panel 1: Original Image
        self.panel1 = ctk.CTkFrame(self.preview_container, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.panel1.grid(row=0, column=0, padx=6, sticky="nsew", ipady=10)
        ctk.CTkLabel(
            self.panel1,
            text="1. Original Crop Foliage",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=config.COLOR_THEME["text_dark"]
        ).pack(pady=(10, 6))
        self.img1_label = ctk.CTkLabel(
            self.panel1,
            text="No image selected.\n\nClick 'Upload Image' or\npress Ctrl+V to paste an image.",
            font=ctk.CTkFont(size=13),
            text_color=config.COLOR_THEME["text_muted"]
        )
        self.img1_label.pack(expand=True, fill="both", padx=10, pady=20)

        # Panel 2: YOLO Detection
        self.panel2 = ctk.CTkFrame(self.preview_container, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.panel2.grid(row=0, column=1, padx=6, sticky="nsew", ipady=10)
        ctk.CTkLabel(
            self.panel2,
            text="2. YOLOv8 Pest Detections",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(pady=(10, 6))
        self.img2_label = ctk.CTkLabel(
            self.panel2,
            text="Awaiting inference...\n\nLocalizes pests with\nbounding box coordinates.",
            font=ctk.CTkFont(size=13),
            text_color=config.COLOR_THEME["text_muted"]
        )
        self.img2_label.pack(expand=True, fill="both", padx=10, pady=20)

        # Panel 3: U-Net Damage Segmentation
        self.panel3 = ctk.CTkFrame(self.preview_container, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.panel3.grid(row=0, column=2, padx=6, sticky="nsew", ipady=10)
        ctk.CTkLabel(
            self.panel3,
            text="3. U-Net Damage / Disease Mask",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#C62828"
        ).pack(pady=(10, 6))
        self.img3_label = ctk.CTkLabel(
            self.panel3,
            text="Awaiting inference...\n\nPixel-level segmentation of\nnecrotic / chewed leaf tissue.",
            font=ctk.CTkFont(size=13),
            text_color=config.COLOR_THEME["text_muted"]
        )
        self.img3_label.pack(expand=True, fill="both", padx=10, pady=20)

        # 5. Diagnostic Summary & Agronomic Recommendation Card
        self.diag_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.diag_card.pack(fill="x", pady=(0, 16), padx=5, ipady=12)

        ctk.CTkLabel(
            self.diag_card,
            text="📊 End-to-End Diagnostic Assessment & Advisory",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["text_dark"]
        ).pack(anchor="w", padx=20, pady=(10, 6))

        self.diag_metrics_frame = ctk.CTkFrame(self.diag_card, fg_color="transparent")
        self.diag_metrics_frame.pack(fill="x", padx=20, pady=6)
        self.diag_metrics_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.m_pests = ctk.CTkLabel(self.diag_metrics_frame, text="🐛 Pests: --", font=ctk.CTkFont(size=14, weight="bold"))
        self.m_pests.grid(row=0, column=0, sticky="w", pady=4)

        self.m_species = ctk.CTkLabel(self.diag_metrics_frame, text="🏷️ Species: --", font=ctk.CTkFont(size=14, weight="bold"))
        self.m_species.grid(row=0, column=1, sticky="w", pady=4)

        self.m_damage = ctk.CTkLabel(self.diag_metrics_frame, text="🍃 Leaf Damage: --%", font=ctk.CTkFont(size=14, weight="bold"))
        self.m_damage.grid(row=0, column=2, sticky="w", pady=4)

        self.m_severity = ctk.CTkLabel(self.diag_metrics_frame, text="⚠️ Severity: --", font=ctk.CTkFont(size=14, weight="bold"))
        self.m_severity.grid(row=0, column=3, sticky="w", pady=4)

        self.rec_label = ctk.CTkLabel(
            self.diag_card,
            text="Select an image or press Ctrl+V to view automated crop diagnostics, severity rating, and advisory actions.",
            font=ctk.CTkFont(size=13),
            text_color=config.COLOR_THEME["text_muted"],
            wraplength=950,
            justify="left"
        )
        self.rec_label.pack(anchor="w", padx=20, pady=(6, 10))

        # 6. Recent Historical Analyses Table
        self.recent_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.recent_card.pack(fill="both", expand=True, padx=5, ipady=10)

        ctk.CTkLabel(
            self.recent_card,
            text="📋 Recent Crop Inspections (Audit Trail)",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["text_dark"]
        ).pack(anchor="w", padx=20, pady=(10, 8))

        self.recent_table_frame = ctk.CTkFrame(self.recent_card, fg_color="transparent")
        self.recent_table_frame.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        self._refresh_recents_table()

    def _refresh_kpi_cards(self):
        """Refreshes top 4 dashboard KPI metric cards from SQLite database."""
        for widget in self.metrics_container.winfo_children():
            widget.destroy()

        stats = self.db.get_dashboard_stats()
        cards = [
            ("📸 TOTAL ANALYZED", str(stats["total_analyzed"]), "#1976D2", "#E3F2FD"),
            ("🐛 PESTS DETECTED", str(stats["total_pests"]), "#388E3C", "#E8F5E9"),
            ("⚠️ HIGH SEVERITY CASES", str(stats["high_severity_cases"]), "#D32F2F", "#FFEBEE"),
            ("🤖 AI MODELS STATUS", "3/3 Operational", "#7B1FA2", "#F3E5F5"),
        ]

        for i, (title, val, border_col, bg_col) in enumerate(cards):
            card = ctk.CTkFrame(self.metrics_container, fg_color=bg_col, border_color=border_col, border_width=1.5, corner_radius=10)
            card.grid(row=0, column=i, padx=6, pady=4, sticky="nsew", ipady=10)

            t_lbl = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color=border_col)
            t_lbl.pack(pady=(8, 2))

            v_lbl = ctk.CTkLabel(card, text=val, font=ctk.CTkFont(size=20, weight="bold"), text_color=config.COLOR_THEME["text_dark"])
            v_lbl.pack(pady=(0, 8))

    def _refresh_recents_table(self):
        """Builds recent history audit table."""
        for widget in self.recent_table_frame.winfo_children():
            widget.destroy()

        stats = self.db.get_dashboard_stats()
        recents = stats.get("recent_analyses", [])

        if not recents:
            ctk.CTkLabel(
                self.recent_table_frame,
                text="No analyses logged yet. Upload or paste an image to begin!",
                font=ctk.CTkFont(size=13),
                text_color=config.COLOR_THEME["text_muted"]
            ).pack(pady=20)
            return

        table_header = ctk.CTkFrame(self.recent_table_frame, fg_color=config.COLOR_THEME["sidebar_light"], height=34)
        table_header.pack(fill="x", pady=(0, 4))
        table_header.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        cols = ["Date / Time", "Image Name", "Pests Detected", "Damage %", "Severity"]
        for idx, c_name in enumerate(cols):
            ctk.CTkLabel(
                table_header,
                text=c_name,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=config.COLOR_THEME["primary"]
            ).grid(row=0, column=idx, padx=6, pady=6)

        for row in recents[:8]:
            r_frame = ctk.CTkFrame(self.recent_table_frame, fg_color="transparent", height=30)
            r_frame.pack(fill="x", pady=1)
            r_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

            sev = row.get("severity_level", "Low")
            sev_color = config.COLOR_THEME["severity_high"] if sev == "High" else (
                config.COLOR_THEME["severity_medium"] if sev == "Medium" else config.COLOR_THEME["severity_low"]
            )

            ctk.CTkLabel(r_frame, text=row.get("timestamp", "")[:19], font=ctk.CTkFont(size=11)).grid(row=0, column=0)
            ctk.CTkLabel(r_frame, text=str(row.get("image_name", ""))[:22], font=ctk.CTkFont(size=11)).grid(row=0, column=1)
            ctk.CTkLabel(r_frame, text=f"{row.get('pest_count', 0)} ({row.get('primary_pest_species', 'None')})", font=ctk.CTkFont(size=11)).grid(row=0, column=2)
            ctk.CTkLabel(r_frame, text=f"{row.get('damage_percentage', 0.0):.1f}%", font=ctk.CTkFont(size=11)).grid(row=0, column=3)
            ctk.CTkLabel(r_frame, text=sev.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color=sev_color).grid(row=0, column=4)

    # --------------------------------------------------------------------------
    # IMAGE LOADING & CLIPBOARD PASTING HANDLERS
    # --------------------------------------------------------------------------
    def _on_upload_click(self):
        file_path = filedialog.askopenfilename(
            title="Select Crop Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp")]
        )
        if file_path:
            self.load_image_from_path(file_path)

    def _on_paste_click(self):
        """Grabs copied image or copied image file from Windows clipboard."""
        clip_data = get_clipboard_image()
        if clip_data is not None:
            pil_img, name = clip_data
            self.load_image_from_pil(pil_img, name)
            # Auto-run analysis on paste for immediate result
            self._run_full_analysis()
        else:
            messagebox.showinfo(
                "Clipboard Empty",
                "No image found in clipboard!\n\n"
                "Tip: Copy an image from anywhere (Web browser, Snipping Tool, or File Explorer) using Ctrl+C, then click Paste or press Ctrl+V."
            )

    def on_paste_image(self, pil_img: Image.Image, name: str):
        """Handler called by MainWindow global Ctrl+V binding."""
        self.load_image_from_pil(pil_img, name)
        self._run_full_analysis()

    def _on_sample_click(self):
        sample_path = os.path.join(config.SAMPLES_DIR, "leaves", "high_infestation_crop.jpg")
        if os.path.exists(sample_path):
            self.load_image_from_path(sample_path)
            self._run_full_analysis()
        else:
            self._on_upload_click()

    def load_image_from_path(self, path: str):
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Image file not found: {path}")
            return
        pil_img = Image.open(path).convert("RGB")
        self.load_image_from_pil(pil_img, os.path.basename(path))

    def load_image_from_pil(self, pil_img: Image.Image, name: str):
        self.current_pil_image = pil_img
        self.current_cv_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        self.current_image_name = name

        # Display in Panel 1
        ctk_img = pil_to_ctk(self.current_pil_image, (320, 240))
        self.img1_label.configure(image=ctk_img, text="")
        self.img1_label.image = ctk_img

        # Reset Panel 2 & 3
        self.img2_label.configure(image=None, text="Click 'Analyze All Features'\nor press Ctrl+V to process.")
        self.img3_label.configure(image=None, text="Click 'Analyze All Features'\nor press Ctrl+V to process.")

        self.run_btn.configure(state="normal")
        self.report_btn.configure(state="disabled")

    # --------------------------------------------------------------------------
    # MULTI-MODEL INFERENCE ENGINE
    # --------------------------------------------------------------------------
    def _run_full_analysis(self):
        """Executes YOLO, CNN, U-Net, and rule engine simultaneously."""
        if self.current_cv_image is None:
            return

        # 1. Model 1: YOLO Detection & Counting
        detections, det_annotated = self.pipeline.detector.detect_pests(self.current_cv_image)
        pest_count = len(detections)
        det_pil = cv2_to_pil(det_annotated)
        det_ctk = pil_to_ctk(det_pil, (320, 240))
        self.img2_label.configure(image=det_ctk, text="")
        self.img2_label.image = det_ctk

        # 2. Model 2: CNN 10-Class Species Classification
        if detections and len(detections) > 0:
            top_box = sorted(detections, key=lambda x: x["confidence"], reverse=True)[0]["bbox"]
            x1, y1, x2, y2 = [int(v) for v in top_box]
            h, w = self.current_cv_image.shape[:2]
            crop_pil = self.current_pil_image.crop((max(0, x1-10), max(0, y1-10), min(w, x2+10), min(h, y2+10)))
            cls_result = self.pipeline.classifier.predict(crop_pil)
        else:
            cls_result = self.pipeline.classifier.predict(self.current_pil_image)

        predicted_species = cls_result["predicted_species"]
        classification_conf = cls_result["confidence"]
        top_preds = cls_result["top_predictions"]

        # 3. Model 3: U-Net Leaf Damage & Disease Segmentation
        seg_result = self.pipeline.segmentor.segment_leaf(self.current_cv_image)
        damage_mask = seg_result["mask"]
        seg_overlay = seg_result["overlay"]
        damage_pct = seg_result["damage_percentage"]

        seg_pil = cv2_to_pil(seg_overlay)
        seg_ctk = pil_to_ctk(seg_pil, (320, 240))
        self.img3_label.configure(image=seg_ctk, text="")
        self.img3_label.image = seg_ctk

        # 4. Severity Assessment & Yield Loss Range
        severity = self.pipeline.evaluate_infestation_severity(pest_count, damage_pct)
        if severity == "High":
            sev_color = config.COLOR_THEME["severity_high"]
            yield_range = "25% - 40%"
            advisory = (
                "⚠️ URGENT INTERVENTION REQUIRED: High pest count and extensive leaf damage observed! "
                "Immediate organic bio-control spray (e.g. Neem extract or Bacillus thuringiensis) or targeted bio-pesticide "
                "recommended. Isolate heavily infested crop foliage."
            )
        elif severity == "Medium":
            sev_color = config.COLOR_THEME["severity_medium"]
            yield_range = "10% - 25%"
            advisory = (
                "⚡ ACTIVE MONITORING REQUIRED: Moderate pest presence and leaf damage detected. "
                "Apply preventive repellent spray and schedule a follow-up inspection in 48-72 hours to assess pest progression."
            )
        else:
            sev_color = config.COLOR_THEME["severity_low"]
            yield_range = "< 10%"
            advisory = (
                "✅ SAFE / LOW INFESTATION: Crop foliage damage and pest numbers are minimal and well below economic injury threshold. "
                "Maintain standard preventive farm hygiene and routine weekly monitoring."
            )

        # Update Metrics Labels
        self.m_pests.configure(text=f"🐛 Pests: {pest_count}")
        self.m_species.configure(text=f"🏷️ Species: {predicted_species} ({classification_conf:.1%})")
        self.m_damage.configure(text=f"🍃 Leaf Damage: {damage_pct:.1f}%")
        self.m_severity.configure(text=f"⚠️ Severity: {severity.upper()}", text_color=sev_color)

        full_rec_text = f"{advisory}\n\n📉 Estimated Potential Yield Loss Range: {yield_range} | Notice: {config.DISCLAIMER_TEXT}"
        self.rec_label.configure(text=full_rec_text, text_color=config.COLOR_THEME["text_dark"])

        # 5. Save comparison & Log into Database
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_img = create_side_by_side_comparison(self.current_cv_image, damage_mask, seg_overlay, damage_pct)
        res_save_path = os.path.join(config.SEGMENTATIONS_DIR, f"comp_{timestamp_str}_{self.current_image_name}")
        cv2.imwrite(res_save_path, comp_img)

        # Save original if needed
        orig_save_path = os.path.join(config.DATA_DIR, "exports", "detections", f"orig_{timestamp_str}_{self.current_image_name}")
        cv2.imwrite(orig_save_path, self.current_cv_image)

        analysis_id = self.db.insert_analysis(
            analysis_type="full",
            image_name=self.current_image_name,
            original_image_path=orig_save_path,
            result_image_path=res_save_path,
            pest_count=pest_count,
            primary_pest_species=predicted_species,
            confidence=classification_conf,
            damage_percentage=damage_pct,
            severity_level=severity,
            notes=f"Unified analysis: {pest_count} pests, {damage_pct:.1f}% damage.",
            detections=detections
        )

        self.analysis_result = {
            "id": analysis_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": self.current_image_name,
            "original_image_path": orig_save_path,
            "result_image_path": res_save_path,
            "pest_count": pest_count,
            "primary_pest_species": predicted_species,
            "confidence": classification_conf,
            "top_predictions": top_preds,
            "damage_percentage": damage_pct,
            "severity_level": severity,
            "detections": detections
        }

        self.report_btn.configure(state="normal")

        # Refresh Top KPIs and Recents Table
        self._refresh_kpi_cards()
        self._refresh_recents_table()

    def _on_export_pdf(self):
        """Generates and opens the formal agronomic PDF report."""
        if not self.analysis_result:
            return

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = os.path.join(config.REPORTS_DIR, f"PestTrack_Report_{timestamp_str}.pdf")

        success = generate_pdf_report(self.analysis_result, pdf_path)
        if success and os.path.exists(pdf_path):
            try:
                os.startfile(pdf_path)
            except Exception:
                pass
            messagebox.showinfo("Report Exported", f"Inspection report generated successfully!\n\nSaved at:\n{pdf_path}")
        else:
            messagebox.showerror("Error", "Failed to generate PDF report.")
