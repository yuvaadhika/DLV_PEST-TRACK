"""
================================================================================
PestTrack AI - Modern Desktop GUI Main Window
Theme: Modern Agriculture Green & White Interface
================================================================================
"""

import os
import customtkinter as ctk
from PIL import Image
import config
from ai_models.pipeline import PestTrackPipeline
from database.db_manager import DatabaseManager

# Import Views
from gui.home_view import HomeView
from gui.detection_view import DetectionView
from gui.classification_view import ClassificationView
from gui.segmentation_view import SegmentationView
from gui.video_view import VideoView
from gui.history_view import HistoryView
from gui.reports_view import ReportsView
from gui.about_view import AboutView

class MainWindow(ctk.CTk):
    """Primary Application Window for PestTrack AI."""

    def __init__(self):
        super().__init__()

        # Window Setup
        self.title(f"{config.APP_NAME} - {config.APP_TITLE}")
        self.geometry("1240x820")
        self.minsize(1050, 700)

        # Set Agriculture Color Theme
        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("green")

        # Initialize AI Core & Database
        self.db = DatabaseManager()
        self.pipeline = PestTrackPipeline()

        # Grid layout (1 row, 2 columns: Sidebar & Main Content)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Navigation State
        self.nav_buttons = {}
        self.active_view = None

        self._build_sidebar()
        self._build_content_area()

        # Show Home View by default
        self.show_view("home")

    def _build_sidebar(self):
        """Constructs left navigation menu."""
        self.sidebar_frame = ctk.CTkFrame(
            self,
            width=230,
            corner_radius=0,
            fg_color=config.COLOR_THEME["sidebar_light"]
        )
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(11, weight=1)

        # Brand / Logo Header
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="🌱 PestTrack AI",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(24, 4), sticky="w")

        self.tagline_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Intelligent Crop Vision",
            font=ctk.CTkFont(size=12),
            text_color=config.COLOR_THEME["text_muted"]
        )
        self.tagline_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        # Navigation Menu Items
        menu_items = [
            ("home", "🏠  Home"),
            ("detection", "🔍  Pest Detection"),
            ("classification", "🐛  Pest Species"),
            ("segmentation", "🍃  Damage Segmentation"),
            ("video", "📹  Video Analysis"),
            ("history", "📊  Analysis History"),
            ("reports", "📄  PDF Reports"),
            ("about", "ℹ️  About Project"),
        ]

        for idx, (view_id, label) in enumerate(menu_items, start=2):
            btn = ctk.CTkButton(
                self.sidebar_frame,
                text=label,
                anchor="w",
                font=ctk.CTkFont(size=14, weight="normal"),
                fg_color="transparent",
                text_color=config.COLOR_THEME["text_dark"],
                hover_color=config.COLOR_THEME["accent"],
                height=42,
                corner_radius=8,
                command=lambda v=view_id: self.show_view(v)
            )
            btn.grid(row=idx, column=0, padx=14, pady=4, sticky="ew")
            self.nav_buttons[view_id] = btn

        # Theme toggle & Exit button at bottom
        self.theme_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="Dark Mode",
            font=ctk.CTkFont(size=12),
            text_color=config.COLOR_THEME["text_dark"],
            command=self._toggle_theme
        )
        self.theme_switch.grid(row=12, column=0, padx=20, pady=(10, 10), sticky="w")

        self.exit_btn = ctk.CTkButton(
            self.sidebar_frame,
            text="🚪  Exit",
            anchor="w",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#FFEBEE",
            text_color="#C62828",
            hover_color="#FFCDD2",
            height=38,
            corner_radius=8,
            command=self.quit
        )
        self.exit_btn.grid(row=13, column=0, padx=14, pady=(0, 20), sticky="ew")

    def _build_content_area(self):
        """Constructs right dynamic view container."""
        self.content_frame = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color=config.COLOR_THEME["background_light"]
        )
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

    def show_view(self, view_name: str, **kwargs):
        """Switches active view and highlights sidebar selection."""
        # Update sidebar button styling
        for v_id, btn in self.nav_buttons.items():
            if v_id == view_name:
                btn.configure(
                    fg_color=config.COLOR_THEME["primary"],
                    text_color="#FFFFFF"
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=config.COLOR_THEME["text_dark"] if ctk.get_appearance_mode() == "Light" else "#FFFFFF"
                )

        # Clear existing view
        if self.active_view is not None:
            self.active_view.destroy()

        # Instantiate requested view
        if view_name == "home":
            self.active_view = HomeView(self.content_frame, self)
        elif view_name == "detection":
            self.active_view = DetectionView(self.content_frame, self)
        elif view_name == "classification":
            self.active_view = ClassificationView(self.content_frame, self)
        elif view_name == "segmentation":
            self.active_view = SegmentationView(self.content_frame, self)
        elif view_name == "video":
            self.active_view = VideoView(self.content_frame, self)
        elif view_name == "history":
            self.active_view = HistoryView(self.content_frame, self)
        elif view_name == "reports":
            self.active_view = ReportsView(self.content_frame, self, **kwargs)
        elif view_name == "about":
            self.active_view = AboutView(self.content_frame, self)

        self.active_view.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    def _toggle_theme(self):
        """Toggles between Light and Dark appearance mode."""
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("Dark")
            self.sidebar_frame.configure(fg_color=config.COLOR_THEME["sidebar_dark"])
            self.content_frame.configure(fg_color=config.COLOR_THEME["background_dark"])
        else:
            ctk.set_appearance_mode("Light")
            self.sidebar_frame.configure(fg_color=config.COLOR_THEME["sidebar_light"])
            self.content_frame.configure(fg_color=config.COLOR_THEME["background_light"])
