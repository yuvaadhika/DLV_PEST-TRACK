"""
================================================================================
PestTrack AI - Home Dashboard View
================================================================================
"""

import os
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import config
from utils.image_utils import load_image_cv, cv2_to_pil, pil_to_ctk

class HomeView(ctk.CTkScrollableFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.db = controller.db
        self.pipeline = controller.pipeline

        self._build_dashboard()

    def _build_dashboard(self):
        # Header Card
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["primary"], corner_radius=12)
        header.pack(fill="x", pady=(0, 20), ipady=15, padx=5)

        h_title = ctk.CTkLabel(
            header,
            text="🌾 Welcome to PestTrack AI Dashboard",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#FFFFFF"
        )
        h_title.pack(anchor="w", padx=25, pady=(10, 4))

        h_sub = ctk.CTkLabel(
            header,
            text="Real-Time Pest Detection (YOLO), Species Classification (CNN), & Leaf Damage Segmentation (U-Net)",
            font=ctk.CTkFont(size=13),
            text_color="#E8F5E9"
        )
        h_sub.pack(anchor="w", padx=25, pady=(0, 10))

        # Metrics Statistics Row
        stats = self.db.get_dashboard_stats()
        metrics_frame = ctk.CTkFrame(self, fg_color="transparent")
        metrics_frame.pack(fill="x", pady=(0, 20))
        metrics_frame.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="metric")

        cards = [
            ("📸 Total Images Analyzed", str(stats["total_analyzed"]), "#1976D2", "#E3F2FD"),
            ("🐛 Total Pests Detected", str(stats["total_pests"]), "#388E3C", "#E8F5E9"),
            ("⚠️ High Severity Cases", str(stats["high_severity_cases"]), "#D32F2F", "#FFEBEE"),
            ("🤖 AI Models Status", "3/3 Operational", "#7B1FA2", "#F3E5F5"),
        ]

        for i, (title, val, border_col, bg_col) in enumerate(cards):
            card = ctk.CTkFrame(metrics_frame, fg_color=bg_col, border_color=border_col, border_width=1.5, corner_radius=10)
            card.grid(row=0, column=i, padx=8, pady=5, sticky="nsew", ipady=12)

            t_lbl = ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12, weight="bold"), text_color=border_col)
            t_lbl.pack(pady=(10, 4))

            v_lbl = ctk.CTkLabel(card, text=val, font=ctk.CTkFont(size=22, weight="bold"), text_color=config.COLOR_THEME["text_dark"])
            v_lbl.pack(pady=(0, 10))

        # Quick Actions Card
        action_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        action_card.pack(fill="x", pady=(0, 20), padx=5, ipady=12)

        ctk.CTkLabel(
            action_card,
            text="⚡ Quick Analysis Actions",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(10, 10))

        btn_row = ctk.CTkFrame(action_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 10))

        upload_btn = ctk.CTkButton(
            btn_row,
            text="📁  Upload & Analyze Image",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=config.COLOR_THEME["primary"],
            hover_color=config.COLOR_THEME["primary_hover"],
            height=42,
            command=self._quick_upload
        )
        upload_btn.pack(side="left", padx=(0, 15))

        sample_btn = ctk.CTkButton(
            btn_row,
            text="🧪  Analyze Sample Crop Leaf",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=config.COLOR_THEME["secondary"],
            hover_color=config.COLOR_THEME["primary"],
            height=42,
            command=self._quick_sample
        )
        sample_btn.pack(side="left", padx=(0, 15))

        # Recent Analyses Table
        recent_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        recent_card.pack(fill="both", expand=True, padx=5, ipady=10)

        ctk.CTkLabel(
            recent_card,
            text="📋 Recent Crop Inspections",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["text_dark"]
        ).pack(anchor="w", padx=20, pady=(12, 10))

        recents = stats["recent_analyses"]
        if not recents:
            ctk.CTkLabel(
                recent_card,
                text="No analyses performed yet. Click 'Upload & Analyze Image' to begin!",
                font=ctk.CTkFont(size=13),
                text_color=config.COLOR_THEME["text_muted"]
            ).pack(pady=30)
        else:
            table_header = ctk.CTkFrame(recent_card, fg_color=config.COLOR_THEME["sidebar_light"], height=36)
            table_header.pack(fill="x", padx=15, pady=(0, 6))
            table_header.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

            cols = ["Date / Time", "Image Name", "Pests Detected", "Damage %", "Severity"]
            for idx, c_name in enumerate(cols):
                ctk.CTkLabel(
                    table_header,
                    text=c_name,
                    font=ctk.CTkFont(size=12, weight="bold"),
                    text_color=config.COLOR_THEME["primary"]
                ).grid(row=0, column=idx, padx=6, pady=8)

            for row in recents:
                r_frame = ctk.CTkFrame(recent_card, fg_color="transparent", height=32)
                r_frame.pack(fill="x", padx=15, pady=2)
                r_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

                sev = row.get("severity_level", "Low")
                sev_color = config.COLOR_THEME["severity_high"] if sev == "High" else (
                    config.COLOR_THEME["severity_medium"] if sev == "Medium" else config.COLOR_THEME["severity_low"]
                )

                ctk.CTkLabel(r_frame, text=row.get("timestamp", ""), font=ctk.CTkFont(size=12)).grid(row=0, column=0)
                ctk.CTkLabel(r_frame, text=row.get("image_name", "")[:24], font=ctk.CTkFont(size=12)).grid(row=0, column=1)
                ctk.CTkLabel(r_frame, text=f"{row.get('pest_count', 0)} ({row.get('primary_pest_species', 'None')})", font=ctk.CTkFont(size=12)).grid(row=0, column=2)
                ctk.CTkLabel(r_frame, text=f"{row.get('damage_percentage', 0.0):.1f}%", font=ctk.CTkFont(size=12)).grid(row=0, column=3)
                ctk.CTkLabel(r_frame, text=sev.upper(), font=ctk.CTkFont(size=12, weight="bold"), text_color=sev_color).grid(row=0, column=4)

    def _quick_upload(self):
        file_path = filedialog.askopenfilename(
            title="Select Crop Image",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if file_path:
            self.controller.show_view("detection", preloaded_image=file_path)

    def _quick_sample(self):
        sample_path = os.path.join(config.SAMPLES_DIR, "leaves", "high_infestation_crop.jpg")
        if os.path.exists(sample_path):
            self.controller.show_view("detection", preloaded_image=sample_path)
        else:
            self._quick_upload()
