"""
================================================================================
PestTrack AI - System Architecture & Technical Specifications
================================================================================
"""

import customtkinter as ctk
import config

class AboutView(ctk.CTkScrollableFrame):
    """System Architecture, AI Models, and Technical Specifications View."""

    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self._build_ui()

    def _build_ui(self):
        # 1. Header Banner
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["primary"], corner_radius=10)
        header.pack(fill="x", pady=(0, 15), padx=5, ipady=15)

        ctk.CTkLabel(
            header,
            text=f"🌱 {config.APP_NAME} - Version {config.APP_VERSION}",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#FFFFFF"
        ).pack(anchor="w", padx=20, pady=(6, 2))

        ctk.CTkLabel(
            header,
            text=config.APP_TITLE,
            font=ctk.CTkFont(size=13),
            text_color="#E8F5E9"
        ).pack(anchor="w", padx=20, pady=(0, 6))

        # 2. System Overview & Core Capabilities Card
        sys_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        sys_card.pack(fill="x", pady=(0, 15), padx=5, ipady=12)

        ctk.CTkLabel(
            sys_card,
            text="⚡ System Capabilities & Highlights",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 6))

        highlights = [
            ("Multi-Model Vision", "Simultaneously executes YOLOv8 object detection, 4-stage CNN classification, and U-Net semantic segmentation."),
            ("100% Offline Ready", "Fully self-contained Windows application requiring zero cloud connectivity, external servers, or API subscriptions."),
            ("Smart Duplicate Prevention", "Integrated SQLite database engine automatically updates records for existing samples, avoiding redundant data."),
            ("Direct Clipboard Paste", "Instant image input via Ctrl+V or one-click clipboard pasting from Web, Snipping Tool, or File Explorer."),
            ("Automated PDF Reporting", "Generates publication-quality agronomist advisory inspection documents with visual evidence and action recommendations.")
        ]

        for k, v in highlights:
            row = ctk.CTkFrame(sys_card, fg_color="transparent")
            row.pack(fill="x", padx=20, pady=3)
            ctk.CTkLabel(row, text=f"• {k}:", width=190, anchor="w", font=ctk.CTkFont(size=12, weight="bold"), text_color=config.COLOR_THEME["text_dark"]).pack(side="left")
            ctk.CTkLabel(row, text=v, anchor="w", font=ctk.CTkFont(size=12), text_color="#444444").pack(side="left")

        # 3. 3 AI Models Technical Architecture Card
        models_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        models_card.pack(fill="x", pady=(0, 15), padx=5, ipady=12)

        ctk.CTkLabel(
            models_card,
            text="🤖 Integrated 3-Pillar Computer Vision Engine",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 10))

        model_specs = [
            (
                "Model 1: YOLOv8 (Pest Detection & Density Counting)",
                "Architecture: Single-stage Anchor-Free Object Detector\n"
                "Function: Identifies insect positions with bounding boxes, computes pest density, and provides real-time detection in images and video streams.\n"
                "Saved Weights: models/yolov8_pest.pt"
            ),
            (
                "Model 2: PyTorch CNN (Pest Species Classification)",
                "Architecture: 4-Stage Hierarchical Convolutional Network with BatchNorm & Dropout\n"
                "Function: Categorizes detected pests into 10 agricultural classes (Aphids, Armyworm, Beetle, Bollworm, Grasshopper, Leafhopper, Mealybug, Mites, Stem Borer, Whitefly) with Top-3 confidence distributions.\n"
                "Saved Weights: models/pest_classifier_cnn.pth"
            ),
            (
                "Model 3: PyTorch U-Net (Leaf Damage & Disease Semantic Segmentation)",
                "Architecture: Symmetric Encoder-Decoder with Skip Connections & Sigmoid Masking\n"
                "Function: Segments visible damaged regions (chlorosis, necrosis, chewing holes), generates binary damage masks, and calculates damaged-area percentage.\n"
                "Saved Weights: models/leaf_damage_unet.pth"
            )
        ]

        for m_title, m_desc in model_specs:
            box = ctk.CTkFrame(models_card, fg_color=config.COLOR_THEME["sidebar_light"], corner_radius=8)
            box.pack(fill="x", padx=20, pady=5, ipady=6)
            ctk.CTkLabel(box, text=m_title, font=ctk.CTkFont(size=13, weight="bold"), text_color=config.COLOR_THEME["primary"]).pack(anchor="w", padx=12, pady=(4, 2))
            ctk.CTkLabel(box, text=m_desc, font=ctk.CTkFont(size=11), text_color="#444444", justify="left").pack(anchor="w", padx=12, pady=(0, 4))

        # 4. Severity Rule Engine & Scientific Disclaimer Card
        rule_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        rule_card.pack(fill="x", pady=(0, 15), padx=5, ipady=12)

        ctk.CTkLabel(
            rule_card,
            text="⚖️ Infestation Severity Matrix & Scientific Disclaimer",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 6))

        rules_text = (
            "• Low Infestation    : Pest count <= 2 AND Leaf Damage < 10%\n"
            "• Medium Infestation : Pest count 3 to 7 OR Leaf Damage 10% to 30%\n"
            "• High Infestation   : Pest count >= 8 OR Leaf Damage > 30%"
        )
        ctk.CTkLabel(rule_card, text=rules_text, font=ctk.CTkFont(family="Consolas", size=12), justify="left").pack(anchor="w", padx=20, pady=4)

        disc_box = ctk.CTkFrame(rule_card, fg_color="#FFFDE7", border_color="#FFF59D", border_width=1, corner_radius=6)
        disc_box.pack(fill="x", padx=20, pady=(8, 4), ipady=6)
        ctk.CTkLabel(
            disc_box,
            text=f"Notice: {config.DISCLAIMER_TEXT}",
            font=ctk.CTkFont(size=11),
            text_color="#616161",
            wraplength=850
        ).pack(padx=12, pady=4)
