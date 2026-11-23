"""
================================================================================
PestTrack AI - PDF Reports Generator View
================================================================================
"""

import os
import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
import config
from utils.pdf_generator import generate_pdf_report
from utils.image_utils import pil_to_ctk

class ReportsView(ctk.CTkScrollableFrame):
    def __init__(self, parent, controller, record_id=None):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.db = controller.db
        self.record_id = record_id

        self.current_record = None
        self._build_ui()
        self._populate_records_dropdown()

        if self.record_id:
            self._load_record(self.record_id)

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        header.pack(fill="x", pady=(0, 15), padx=5, ipady=10)

        ctk.CTkLabel(
            header,
            text="📄 Automated Agronomic PDF Reports",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 2))

        ctk.CTkLabel(
            header,
            text="Generate printable, professional crop inspection reports containing AI detection boxes, U-Net masks, and severity analysis.",
            font=ctk.CTkFont(size=12),
            text_color=config.COLOR_THEME["text_muted"]
        ).pack(anchor="w", padx=20, pady=(0, 6))

        # Selection Toolbar
        ctrl_frame = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        ctrl_frame.pack(fill="x", pady=(0, 15), padx=5, ipady=8)

        ctk.CTkLabel(ctrl_frame, text="Select Inspection Record:", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=(15, 8))

        self.record_dropdown = ctk.CTkOptionMenu(
            ctrl_frame,
            values=["No records found"],
            width=280,
            command=self._on_dropdown_select
        )
        self.record_dropdown.pack(side="left", padx=5)

        self.gen_pdf_btn = ctk.CTkButton(
            ctrl_frame,
            text="📥 Generate & Open PDF Report",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=config.COLOR_THEME["primary"],
            hover_color=config.COLOR_THEME["primary_hover"],
            state="disabled",
            command=self._generate_and_open_pdf
        )
        self.gen_pdf_btn.pack(side="right", padx=15)

        # Report Details Preview Container
        self.preview_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.preview_card.pack(fill="both", expand=True, padx=5, ipady=15)

        ctk.CTkLabel(self.preview_card, text="Inspection Report Preview", font=ctk.CTkFont(size=16, weight="bold"), text_color=config.COLOR_THEME["primary"]).pack(anchor="w", padx=20, pady=(10, 8))

        self.summary_box = ctk.CTkTextbox(self.preview_card, height=180, font=ctk.CTkFont(family="Consolas", size=12))
        self.summary_box.pack(fill="x", padx=20, pady=(0, 15))
        self.summary_box.insert("1.0", "Select a record above to preview the report details...")
        self.summary_box.configure(state="disabled")

        # Visual Artifacts Display
        img_frame = ctk.CTkFrame(self.preview_card, fg_color="transparent")
        img_frame.pack(fill="x", padx=20, pady=5)
        img_frame.grid_columnconfigure((0, 1), weight=1, uniform="rep_img")

        # Left Original Preview
        c_orig = ctk.CTkFrame(img_frame, fg_color=config.COLOR_THEME["sidebar_light"], corner_radius=8)
        c_orig.grid(row=0, column=0, padx=(0, 8), sticky="nsew", ipady=10)
        ctk.CTkLabel(c_orig, text="Original Crop Image", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=4)
        self.rep_orig_lbl = ctk.CTkLabel(c_orig, text="No Image", text_color="#888888")
        self.rep_orig_lbl.pack(pady=20, expand=True)

        # Right Result Preview
        c_res = ctk.CTkFrame(img_frame, fg_color=config.COLOR_THEME["sidebar_light"], corner_radius=8)
        c_res.grid(row=0, column=1, padx=(8, 0), sticky="nsew", ipady=10)
        ctk.CTkLabel(c_res, text="AI Processed Result", font=ctk.CTkFont(size=12, weight="bold"), text_color=config.COLOR_THEME["primary"]).pack(pady=4)
        self.rep_res_lbl = ctk.CTkLabel(c_res, text="No Result", text_color="#888888")
        self.rep_res_lbl.pack(pady=20, expand=True)

    def _populate_records_dropdown(self):
        records = self.db.get_all_analyses(limit=50)
        if records:
            vals = [f"#{r['id']} - {r.get('timestamp','')} ({r.get('primary_pest_species','-')})" for r in records]
            self.record_dropdown.configure(values=vals)
            self.record_dropdown.set(vals[0])
            self._load_record(records[0]["id"])
        else:
            self.record_dropdown.configure(values=["No records available"])

    def _on_dropdown_select(self, choice):
        if choice and choice.startswith("#"):
            try:
                r_id = int(choice.split(" - ")[0].replace("#", ""))
                self._load_record(r_id)
            except Exception:
                pass

    def _load_record(self, record_id):
        record = self.db.get_analysis_by_id(record_id)
        if not record:
            return
        self.current_record = record
        self.gen_pdf_btn.configure(state="normal")

        # Format Summary
        summary_text = (
            f"========================================================================\n"
            f" PESTTRACK AI INSPECTION SUMMARY REPORT\n"
            f"========================================================================\n"
            f" Report ID               : PT-{record['id']:04d}\n"
            f" Timestamp               : {record['timestamp']}\n"
            f" Crop Image Name         : {record['image_name']}\n"
            f"------------------------------------------------------------------------\n"
            f" AI Model 1 (YOLO)       : Total Pests Detected = {record['pest_count']}\n"
            f" AI Model 2 (CNN)        : Primary Species = {record['primary_pest_species']} ({record['confidence']} %)\n"
            f" AI Model 3 (U-Net)      : Visible Leaf Damage = {record['damage_percentage']} %\n"
            f" Overall Infestation     : {record['severity_level'].upper()} SEVERITY\n"
            f"------------------------------------------------------------------------\n"
            f" Yield Impact Disclaimer : {config.DISCLAIMER_TEXT}\n"
            f"========================================================================\n"
        )

        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", "end")
        self.summary_box.insert("1.0", summary_text)
        self.summary_box.configure(state="disabled")

        # Update Images
        orig_p = record.get("original_image_path", "")
        res_p = record.get("result_image_path", "")

        if orig_p and os.path.exists(orig_p):
            try:
                ctk_orig = pil_to_ctk(Image.open(orig_p), (360, 220))
                self.rep_orig_lbl.configure(image=ctk_orig, text="")
                self.rep_orig_lbl.image = ctk_orig
            except Exception:
                pass

        if res_p and os.path.exists(res_p):
            try:
                ctk_res = pil_to_ctk(Image.open(res_p), (360, 220))
                self.rep_res_lbl.configure(image=ctk_res, text="")
                self.rep_res_lbl.image = ctk_res
            except Exception:
                pass

    def _generate_and_open_pdf(self):
        if not self.current_record:
            return

        r_id = self.current_record["id"]
        pdf_filename = f"PestTrack_Report_PT{r_id:04d}.pdf"
        output_pdf_path = os.path.join(config.REPORTS_DIR, pdf_filename)

        success = generate_pdf_report(self.current_record, output_pdf_path)
        if success and os.path.exists(output_pdf_path):
            messagebox.showinfo("PDF Generated", f"Inspection report successfully generated:\n{output_pdf_path}")
            try:
                # Open PDF with default Windows viewer
                os.startfile(output_pdf_path)
            except Exception:
                pass
        else:
            messagebox.showerror("Error", "Failed to generate PDF report.")
