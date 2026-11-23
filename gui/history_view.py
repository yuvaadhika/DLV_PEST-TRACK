"""
================================================================================
PestTrack AI - Analysis History & SQLite Database Management View
================================================================================
"""

import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import config
from utils.image_utils import pil_to_ctk

class HistoryView(ctk.CTkScrollableFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.db = controller.db

        self._build_ui()
        self._load_records()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        header.pack(fill="x", pady=(0, 15), padx=5, ipady=10)

        ctk.CTkLabel(
            header,
            text="📊 Analysis History & Local SQLite Database",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 2))

        ctk.CTkLabel(
            header,
            text="Review previous field inspections, generate retrospective reports, and manage database backups.",
            font=ctk.CTkFont(size=12),
            text_color=config.COLOR_THEME["text_muted"]
        ).pack(anchor="w", padx=20, pady=(0, 6))

        # Toolbar (Refresh, Backup, Restore)
        toolbar = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        toolbar.pack(fill="x", pady=(0, 15), padx=5, ipady=8)

        ctk.CTkButton(
            toolbar,
            text="🔄 Refresh Records",
            font=ctk.CTkFont(size=13),
            fg_color=config.COLOR_THEME["primary"],
            hover_color=config.COLOR_THEME["primary_hover"],
            command=self._load_records
        ).pack(side="left", padx=15)

        ctk.CTkButton(
            toolbar,
            text="💾 Backup Database",
            font=ctk.CTkFont(size=13),
            fg_color=config.COLOR_THEME["secondary"],
            command=self._backup_db
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            toolbar,
            text="📥 Restore Database",
            font=ctk.CTkFont(size=13),
            fg_color="#455A64",
            hover_color="#37474F",
            command=self._restore_db
        ).pack(side="left", padx=5)

        # Records Table Container
        self.table_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        self.table_card.pack(fill="both", expand=True, padx=5, ipady=10)

        # Table Header
        self.t_header = ctk.CTkFrame(self.table_card, fg_color=config.COLOR_THEME["sidebar_light"], height=38)
        self.t_header.pack(fill="x", padx=15, pady=(10, 6))
        self.t_header.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        cols = ["ID", "Timestamp", "Image Name", "Species / Count", "Damage %", "Severity", "Actions"]
        for idx, col in enumerate(cols):
            ctk.CTkLabel(
                self.t_header,
                text=col,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=config.COLOR_THEME["primary"]
            ).grid(row=0, column=idx, padx=4, pady=8)

        self.rows_container = ctk.CTkFrame(self.table_card, fg_color="transparent")
        self.rows_container.pack(fill="both", expand=True, padx=15)

    def _load_records(self):
        # Clear existing rows
        for widget in self.rows_container.winfo_children():
            widget.destroy()

        records = self.db.get_all_analyses(limit=100)
        if not records:
            ctk.CTkLabel(
                self.rows_container,
                text="No analysis history found in SQLite database.",
                font=ctk.CTkFont(size=13),
                text_color=config.COLOR_THEME["text_muted"]
            ).pack(pady=30)
            return

        for r in records:
            r_id = r["id"]
            row_frame = ctk.CTkFrame(self.rows_container, fg_color="transparent", height=38)
            row_frame.pack(fill="x", pady=2)
            row_frame.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

            sev = r.get("severity_level", "Low")
            sev_color = config.COLOR_THEME["severity_high"] if sev == "High" else (
                config.COLOR_THEME["severity_medium"] if sev == "Medium" else config.COLOR_THEME["severity_low"]
            )

            ctk.CTkLabel(row_frame, text=f"#{r_id}", font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0)
            ctk.CTkLabel(row_frame, text=r.get("timestamp", "")[:16], font=ctk.CTkFont(size=11)).grid(row=0, column=1)
            ctk.CTkLabel(row_frame, text=r.get("image_name", "")[:18], font=ctk.CTkFont(size=11)).grid(row=0, column=2)
            ctk.CTkLabel(row_frame, text=f"{r.get('primary_pest_species', '-')}: {r.get('pest_count', 0)}", font=ctk.CTkFont(size=11)).grid(row=0, column=3)
            ctk.CTkLabel(row_frame, text=f"{r.get('damage_percentage', 0.0):.1f}%", font=ctk.CTkFont(size=11)).grid(row=0, column=4)
            ctk.CTkLabel(row_frame, text=sev.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color=sev_color).grid(row=0, column=5)

            # Action buttons frame
            act_f = ctk.CTkFrame(row_frame, fg_color="transparent")
            act_f.grid(row=0, column=6)

            ctk.CTkButton(
                act_f,
                text="📄",
                width=30,
                height=26,
                font=ctk.CTkFont(size=12),
                fg_color="#1976D2",
                hover_color="#1565C0",
                command=lambda rec_id=r_id: self.controller.show_view("reports", record_id=rec_id)
            ).pack(side="left", padx=2)

            ctk.CTkButton(
                act_f,
                text="🗑️",
                width=30,
                height=26,
                font=ctk.CTkFont(size=12),
                fg_color="#D32F2F",
                hover_color="#B71C1C",
                command=lambda rec_id=r_id: self._delete_record(rec_id)
            ).pack(side="left", padx=2)

    def _delete_record(self, record_id):
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete inspection record #{record_id}?")
        if confirm:
            self.db.delete_analysis(record_id)
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
