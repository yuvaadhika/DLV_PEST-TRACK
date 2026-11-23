"""
================================================================================
PestTrack AI - Analysis History & Full Visual Pipeline Audit View
Displays Complete End-to-End Pipeline for Every Prior Analysis:
  1. Original Crop Foliage (Input Leaf)
  2. YOLOv8 Pest Detections (Bounding Boxes & Count)
  3. U-Net Leaf Damage & Disease Mask (Segmented Foliage & Damage %)
  4. Diagnostics, Yield Impact & Treatment Advisory
  5. 1-Click PDF Report Export (with User Confirmation)
================================================================================
"""

import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import config
from utils.image_utils import pil_to_ctk
from utils.pdf_generator import generate_pdf_report

class HistoryView(ctk.CTkScrollableFrame):
    """Exhibits complete historical records from original leaf to end-process visual outputs."""

    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.db = controller.db

        self.records = []
        self.selected_record = None

        self._build_ui()
        self._load_records()

    def _build_ui(self):
        # 1. Header Banner
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        header.pack(fill="x", pady=(0, 15), padx=5, ipady=10)

        ctk.CTkLabel(
            header,
            text="📊 Comprehensive Analysis History & Audit Trail",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 2))

        ctk.CTkLabel(
            header,
            text="Review all historical crop inspections from original input leaf to end multi-model AI visual outputs.",
            font=ctk.CTkFont(size=12),
            text_color=config.COLOR_THEME["text_muted"]
        ).pack(anchor="w", padx=20, pady=(0, 6))

        # 2. Controls Toolbar
        toolbar = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        toolbar.pack(fill="x", pady=(0, 15), padx=5, ipady=8)

        ctk.CTkButton(
            toolbar,
            text="🔄 Refresh Records",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=config.COLOR_THEME["primary"],
            hover_color=config.COLOR_THEME["primary_hover"],
            command=self._load_records
        ).pack(side="left", padx=15)

        ctk.CTkLabel(toolbar, text="Select Record to Inspect:", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=(10, 6))

        self.dropdown_var = ctk.StringVar(value="Select an inspection record...")
        self.record_dropdown = ctk.CTkOptionMenu(
            toolbar,
            variable=self.dropdown_var,
            values=["No records available"],
            width=360,
            command=self._on_dropdown_selected
        )
        self.record_dropdown.pack(side="left", padx=6)

        ctk.CTkButton(
            toolbar,
            text="💾 Backup DB",
            font=ctk.CTkFont(size=12),
            fg_color=config.COLOR_THEME["secondary"],
            command=self._backup_db
        ).pack(side="right", padx=10)

        ctk.CTkButton(
            toolbar,
            text="📥 Restore DB",
            font=ctk.CTkFont(size=12),
            fg_color="#455A64",
            hover_color="#37474F",
            command=self._restore_db
        ).pack(side="right", padx=5)

        # 3. Full End-to-End Visual Exhibition Showcase Card
        self.showcase_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.showcase_card.pack(fill="x", pady=(0, 15), padx=5, ipady=12)

        self.showcase_title = ctk.CTkLabel(
            self.showcase_card,
            text="🖼️ End-to-End Visual Pipeline (Original Leaf ➔ YOLO Detections ➔ U-Net Damage Mask)",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        )
        self.showcase_title.pack(anchor="w", padx=20, pady=(8, 10))

        # 3 Side-by-Side Visual Panels
        self.panels_frame = ctk.CTkFrame(self.showcase_card, fg_color="transparent")
        self.panels_frame.pack(fill="x", padx=15, pady=(0, 10))
        self.panels_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="hist_panels")

        # Panel 1: Original Leaf
        p1 = ctk.CTkFrame(self.panels_frame, fg_color=config.COLOR_THEME["sidebar_light"], corner_radius=8)
        p1.grid(row=0, column=0, padx=5, sticky="nsew", ipady=8)
        ctk.CTkLabel(p1, text="1. Original Crop Foliage", font=ctk.CTkFont(size=13, weight="bold"), text_color=config.COLOR_THEME["text_dark"]).pack(pady=(6, 4))
        self.orig_lbl = ctk.CTkLabel(p1, text="No Image", text_color="#777777")
        self.orig_lbl.pack(pady=20, expand=True)

        # Panel 2: YOLO Detection
        p2 = ctk.CTkFrame(self.panels_frame, fg_color=config.COLOR_THEME["sidebar_light"], corner_radius=8)
        p2.grid(row=0, column=1, padx=5, sticky="nsew", ipady=8)
        ctk.CTkLabel(p2, text="2. YOLOv8 Pest Detections", font=ctk.CTkFont(size=13, weight="bold"), text_color=config.COLOR_THEME["primary"]).pack(pady=(6, 4))
        self.det_lbl = ctk.CTkLabel(p2, text="No Detection", text_color="#777777")
        self.det_lbl.pack(pady=20, expand=True)

        # Panel 3: U-Net Damage Mask
        p3 = ctk.CTkFrame(self.panels_frame, fg_color=config.COLOR_THEME["sidebar_light"], corner_radius=8)
        p3.grid(row=0, column=2, padx=5, sticky="nsew", ipady=8)
        ctk.CTkLabel(p3, text="3. U-Net Damage Mask Overlay", font=ctk.CTkFont(size=13, weight="bold"), text_color="#C62828").pack(pady=(6, 4))
        self.seg_lbl = ctk.CTkLabel(p3, text="No Mask", text_color="#777777")
        self.seg_lbl.pack(pady=20, expand=True)

        # 4. Detailed Diagnostics Summary
        self.diag_frame = ctk.CTkFrame(self.showcase_card, fg_color="transparent")
        self.diag_frame.pack(fill="x", padx=20, pady=(6, 6))
        self.diag_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.d_pests = ctk.CTkLabel(self.diag_frame, text="🐛 Pests: --", font=ctk.CTkFont(size=13, weight="bold"))
        self.d_pests.grid(row=0, column=0, sticky="w", pady=2)

        self.d_species = ctk.CTkLabel(self.diag_frame, text="🏷️ Species: --", font=ctk.CTkFont(size=13, weight="bold"))
        self.d_species.grid(row=0, column=1, sticky="w", pady=2)

        self.d_damage = ctk.CTkLabel(self.diag_frame, text="🍃 Leaf Damage: --%", font=ctk.CTkFont(size=13, weight="bold"))
        self.d_damage.grid(row=0, column=2, sticky="w", pady=2)

        self.d_severity = ctk.CTkLabel(self.diag_frame, text="⚠️ Severity: --", font=ctk.CTkFont(size=13, weight="bold"))
        self.d_severity.grid(row=0, column=3, sticky="w", pady=2)

        self.d_advisory = ctk.CTkLabel(
            self.showcase_card,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#444444",
            wraplength=950,
            justify="left"
        )
        self.d_advisory.pack(anchor="w", padx=20, pady=(4, 8))

        # Action Buttons Row for Selected Record
        act_row = ctk.CTkFrame(self.showcase_card, fg_color="transparent")
        act_row.pack(fill="x", padx=20, pady=(4, 8))

        self.export_pdf_btn = ctk.CTkButton(
            act_row,
            text="📄  Export PDF Inspection Report",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#1976D2",
            hover_color="#1565C0",
            state="disabled",
            command=self._export_selected_pdf
        )
        self.export_pdf_btn.pack(side="left", padx=(0, 10))

        self.del_btn = ctk.CTkButton(
            act_row,
            text="🗑️  Delete This Record",
            font=ctk.CTkFont(size=13),
            fg_color="#D32F2F",
            hover_color="#B71C1C",
            state="disabled",
            command=self._delete_selected_record
        )
        self.del_btn.pack(side="left")

        # 5. Historical Audit Trail Table Container
        self.table_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.table_card.pack(fill="both", expand=True, padx=5, ipady=10)

        ctk.CTkLabel(
            self.table_card,
            text="📋 Historical Audit Log Table",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=config.COLOR_THEME["text_dark"]
        ).pack(anchor="w", padx=20, pady=(10, 6))

        # Table Header
        self.t_header = ctk.CTkFrame(self.table_card, fg_color=config.COLOR_THEME["sidebar_light"], height=36)
        self.t_header.pack(fill="x", padx=15, pady=(4, 6))
        self.t_header.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        cols = ["ID", "Timestamp", "Image Name", "Species / Count", "Damage %", "Severity", "Action"]
        for idx, col in enumerate(cols):
            ctk.CTkLabel(
                self.t_header,
                text=col,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=config.COLOR_THEME["primary"]
            ).grid(row=0, column=idx, padx=4, pady=6)

        self.rows_container = ctk.CTkFrame(self.table_card, fg_color="transparent")
        self.rows_container.pack(fill="both", expand=True, padx=15)

    def _load_records(self):
        """Loads all records from database and populates selector and table."""
        for widget in self.rows_container.winfo_children():
            widget.destroy()

        self.records = self.db.get_all_analyses(limit=150)
        if not self.records:
            self.record_dropdown.configure(values=["No records available"])
            self.dropdown_var.set("No records available")
            self.orig_lbl.configure(image=None, text="No records logged yet.")
            self.det_lbl.configure(image=None, text="No records logged yet.")
            self.seg_lbl.configure(image=None, text="No records logged yet.")
            self.export_pdf_btn.configure(state="disabled")
            self.del_btn.configure(state="disabled")
            ctk.CTkLabel(
                self.rows_container,
                text="No analysis history logged yet. Analyze an image from Home to view records here!",
                font=ctk.CTkFont(size=13),
                text_color=config.COLOR_THEME["text_muted"]
            ).pack(pady=30)
            return

        # Populate dropdown
        options = []
        for r in self.records:
            options.append(f"#{r['id']} - {r.get('image_name', 'Crop')[:22]} | {r.get('primary_pest_species', 'Pest')} | {r.get('severity_level', 'Low')} ({r.get('timestamp', '')[:16]})")

        self.record_dropdown.configure(values=options)
        self.dropdown_var.set(options[0])

        # Automatically display the latest record in the showcase
        self._display_record_details(self.records[0])

        # Build table rows
        for r in self.records:
            r_id = r["id"]
            row_frame = ctk.CTkFrame(self.rows_container, fg_color="transparent", height=34)
            row_frame.pack(fill="x", pady=2)
            row_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

            sev = r.get("severity_level", "Low")
            sev_color = config.COLOR_THEME["severity_high"] if sev == "High" else (
                config.COLOR_THEME["severity_medium"] if sev == "Medium" else config.COLOR_THEME["severity_low"]
            )

            ctk.CTkLabel(row_frame, text=f"#{r_id}", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0)
            ctk.CTkLabel(row_frame, text=r.get("timestamp", "")[:16], font=ctk.CTkFont(size=11)).grid(row=0, column=1)
            ctk.CTkLabel(row_frame, text=str(r.get("image_name", ""))[:20], font=ctk.CTkFont(size=11)).grid(row=0, column=2)
            ctk.CTkLabel(row_frame, text=f"{r.get('primary_pest_species', '-')}: {r.get('pest_count', 0)}", font=ctk.CTkFont(size=11)).grid(row=0, column=3)
            ctk.CTkLabel(row_frame, text=f"{r.get('damage_percentage', 0.0):.1f}%", font=ctk.CTkFont(size=11)).grid(row=0, column=4)
            ctk.CTkLabel(row_frame, text=sev.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color=sev_color).grid(row=0, column=5)

            # View Details button
            ctk.CTkButton(
                row_frame,
                text="👁️ View",
                width=65,
                height=26,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=config.COLOR_THEME["primary"],
                hover_color=config.COLOR_THEME["primary_hover"],
                command=lambda rec=r: self._display_record_details(rec)
            ).grid(row=0, column=6)

    def _on_dropdown_selected(self, choice):
        try:
            r_id = int(choice.split(" - ")[0].replace("#", ""))
            matched = next((r for r in self.records if r["id"] == r_id), None)
            if matched:
                self._display_record_details(matched)
        except Exception:
            pass

    def _display_record_details(self, record: dict):
        """Populates the 3-panel showcase and diagnostic metrics for the chosen record."""
        self.selected_record = record
        r_id = record["id"]
        img_name = record.get("image_name", "Crop_Leaf")
        t_stamp = record.get("timestamp", "")

        self.showcase_title.configure(
            text=f"🖼️ Record #{r_id}: {img_name} ({t_stamp}) - Full End-to-End Visual Pipeline"
        )

        # 1. Panel 1: Original Image
        orig_p = record.get("original_image_path", "")
        if orig_p and os.path.exists(orig_p):
            try:
                ctk_orig = pil_to_ctk(Image.open(orig_p), (300, 210))
                self.orig_lbl.configure(image=ctk_orig, text="")
                self.orig_lbl.image = ctk_orig
            except Exception:
                self.orig_lbl.configure(image=None, text="Error loading original")
        else:
            self.orig_lbl.configure(image=None, text="Original image not found")

        # 2. Panel 2 & 3: Visual Detection and Damage Mask
        # Parse notes to extract det and seg paths if available
        notes = record.get("notes", "")
        det_path = ""
        seg_path = ""
        if "det:" in notes and "|seg:" in notes:
            try:
                parts = notes.split("|")
                det_path = parts[0].replace("det:", "").strip()
                seg_path = parts[1].replace("seg:", "").strip()
            except Exception:
                pass

        # If det_path exists, load it
        if det_path and os.path.exists(det_path):
            try:
                ctk_det = pil_to_ctk(Image.open(det_path), (300, 210))
                self.det_lbl.configure(image=ctk_det, text="")
                self.det_lbl.image = ctk_det
            except Exception:
                self.det_lbl.configure(image=None, text="Error loading detection")
        else:
            # Fallback to result_image_path if available
            res_p = record.get("result_image_path", "")
            if res_p and os.path.exists(res_p):
                try:
                    ctk_det = pil_to_ctk(Image.open(res_p), (300, 210))
                    self.det_lbl.configure(image=ctk_det, text="")
                    self.det_lbl.image = ctk_det
                except Exception:
                    self.det_lbl.configure(image=None, text="No detection preview")
            else:
                self.det_lbl.configure(image=None, text="No detection preview")

        # If seg_path exists, load it
        if seg_path and os.path.exists(seg_path):
            try:
                ctk_seg = pil_to_ctk(Image.open(seg_path), (300, 210))
                self.seg_lbl.configure(image=ctk_seg, text="")
                self.seg_lbl.image = ctk_seg
            except Exception:
                self.seg_lbl.configure(image=None, text="Error loading segmentation")
        else:
            res_p = record.get("result_image_path", "")
            if res_p and os.path.exists(res_p):
                try:
                    ctk_seg = pil_to_ctk(Image.open(res_p), (300, 210))
                    self.seg_lbl.configure(image=ctk_seg, text="")
                    self.seg_lbl.image = ctk_seg
                except Exception:
                    self.seg_lbl.configure(image=None, text="No segmentation preview")
            else:
                self.seg_lbl.configure(image=None, text="No segmentation preview")

        # 3. Update Diagnostics Labels
        p_count = record.get("pest_count", 0)
        species = record.get("primary_pest_species", "Unknown")
        conf = record.get("confidence", 0.0)
        damage_pct = record.get("damage_percentage", 0.0)
        sev = record.get("severity_level", "Low")

        sev_color = config.COLOR_THEME["severity_high"] if sev == "High" else (
            config.COLOR_THEME["severity_medium"] if sev == "Medium" else config.COLOR_THEME["severity_low"]
        )

        yield_range = "25% - 40%" if sev == "High" else ("10% - 25%" if sev == "Medium" else "< 10%")

        self.d_pests.configure(text=f"🐛 Pests: {p_count} detected")
        self.d_species.configure(text=f"🏷️ Species: {species} ({conf:.1%})")
        self.d_damage.configure(text=f"🍃 Leaf Damage: {damage_pct:.1f}%")
        self.d_severity.configure(text=f"⚠️ Severity: {sev.upper()}", text_color=sev_color)

        advisory_msg = f"📉 Estimated Yield Impact: {yield_range} | Notice: {config.DISCLAIMER_TEXT}"
        self.d_advisory.configure(text=advisory_msg)

        self.export_pdf_btn.configure(state="normal")
        self.del_btn.configure(state="normal")

    def _export_selected_pdf(self):
        """Generates PDF report for currently showcased record after confirmation."""
        if not self.selected_record:
            return

        confirm = messagebox.askyesno(
            "Confirm PDF Download",
            f"Are you sure you want to generate and download the official PDF inspection report for Record #{self.selected_record['id']}?\n\n"
            f"• Image: {self.selected_record.get('image_name', 'Crop')}\n"
            f"• Severity: {self.selected_record.get('severity_level', 'Low').upper()}\n\n"
            "Click 'Yes' to confirm and open report."
        )
        if not confirm:
            return

        r_id = self.selected_record["id"]
        pdf_path = os.path.join(config.REPORTS_DIR, f"PestTrack_Report_PT{r_id:04d}.pdf")

        success = generate_pdf_report(self.selected_record, pdf_path)
        if success and os.path.exists(pdf_path):
            try:
                os.startfile(pdf_path)
            except Exception:
                pass
            messagebox.showinfo("PDF Generated", f"Inspection report generated successfully:\n{pdf_path}")
        else:
            messagebox.showerror("Error", "Failed to generate PDF report.")

    def _delete_selected_record(self):
        """Deletes the currently showcased record after confirmation."""
        if not self.selected_record:
            return

        r_id = self.selected_record["id"]
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete inspection record #{r_id}?")
        if confirm:
            self.db.delete_analysis(r_id)
            self._load_records()

    def _backup_db(self):
        f = filedialog.asksaveasfilename(
            defaultextension=".db",
            filetypes=[("SQLite Database", "*.db")],
            initialfile="pesttrack_backup.db"
        )
        if f:
            if self.db.backup_database(f):
                messagebox.showinfo("Backup Successful", f"Database successfully backed up to:\n{f}")
            else:
                messagebox.showerror("Backup Failed", "Failed to backup database.")

    def _restore_db(self):
        f = filedialog.askopenfilename(
            title="Select Database Backup to Restore",
            filetypes=[("SQLite Database", "*.db")]
        )
        if f:
            confirm = messagebox.askyesno("Confirm Restore", "Restoring will replace the current database. Proceed?")
            if confirm:
                if self.db.restore_database(f):
                    messagebox.showinfo("Restore Successful", "Database restored successfully!")
                    self._load_records()
                else:
                    messagebox.showerror("Restore Failed", "Failed to restore database.")
