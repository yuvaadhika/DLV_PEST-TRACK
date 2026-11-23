"""
================================================================================
PestTrack AI - Video-Based Real-Time Pest Detection View
================================================================================
"""

import os
import cv2
import threading
import time
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import config
from utils.image_utils import cv2_to_pil, pil_to_ctk

class VideoView(ctk.CTkScrollableFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color="transparent")
        self.controller = controller
        self.detector = controller.pipeline.detector

        self.video_path = None
        self.is_playing = False
        self.cap = None
        self.current_count = 0

        self._build_ui()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        header.pack(fill="x", pady=(0, 15), padx=5, ipady=10)

        ctk.CTkLabel(
            header,
            text="📹 Video-Based Real-Time Pest Detection",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        ).pack(anchor="w", padx=20, pady=(6, 2))

        ctk.CTkLabel(
            header,
            text="Analyze live video streams or recorded crop field footage frame-by-frame using YOLOv8 object tracking.",
            font=ctk.CTkFont(size=12),
            text_color=config.COLOR_THEME["text_muted"]
        ).pack(anchor="w", padx=20, pady=(0, 6))

        # Controls Toolbar
        ctrl_frame = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        ctrl_frame.pack(fill="x", pady=(0, 15), padx=5, ipady=8)

        self.select_btn = ctk.CTkButton(
            ctrl_frame,
            text="📁 Select Video File",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=config.COLOR_THEME["primary"],
            hover_color=config.COLOR_THEME["primary_hover"],
            command=self._select_video
        )
        self.select_btn.pack(side="left", padx=15)

        self.sample_btn = ctk.CTkButton(
            ctrl_frame,
            text="🧪 Use Sample Pest Video",
            font=ctk.CTkFont(size=13),
            fg_color=config.COLOR_THEME["secondary"],
            command=self._load_sample_video
        )
        self.sample_btn.pack(side="left", padx=5)

        self.play_btn = ctk.CTkButton(
            ctrl_frame,
            text="▶️ Start Video Analysis",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2E7D32",
            hover_color="#1B5E20",
            state="disabled",
            command=self._toggle_playback
        )
        self.play_btn.pack(side="left", padx=15)

        # Video Player Frame
        player_card = ctk.CTkFrame(self, fg_color=config.COLOR_THEME["card_light"], corner_radius=10)
        player_card.pack(fill="x", pady=(0, 15), padx=5, ipady=10)

        # Video Display Canvas
        self.video_lbl = ctk.CTkLabel(
            player_card,
            text="Select a video file to begin real-time analysis",
            font=ctk.CTkFont(size=13),
            text_color="#888888",
            width=640,
            height=420
        )
        self.video_lbl.pack(pady=10)

        # Video Metrics Bar
        self.metrics_bar = ctk.CTkFrame(player_card, fg_color=config.COLOR_THEME["sidebar_light"], height=40, corner_radius=6)
        self.metrics_bar.pack(fill="x", padx=20, pady=(5, 10))

        self.counter_lbl = ctk.CTkLabel(
            self.metrics_bar,
            text="Real-Time Detected Pests: 0",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=config.COLOR_THEME["primary"]
        )
        self.counter_lbl.pack(side="left", padx=20, pady=8)

        self.fps_lbl = ctk.CTkLabel(
            self.metrics_bar,
            text="Processing Speed: 0 FPS",
            font=ctk.CTkFont(size=13),
            text_color=config.COLOR_THEME["text_muted"]
        )
        self.fps_lbl.pack(side="right", padx=20, pady=8)

    def _select_video(self):
        f = filedialog.askopenfilename(
            title="Select Crop Video File",
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv")]
        )
        if f:
            self._prepare_video(f)

    def _load_sample_video(self):
        sample_path = os.path.join(config.SAMPLES_DIR, "sample_pest_video.mp4")
        if os.path.exists(sample_path):
            self._prepare_video(sample_path)
        else:
            self._select_video()

    def _prepare_video(self, path):
        self._stop_playback()
        self.video_path = path
        self.play_btn.configure(state="normal", text="▶️ Start Video Analysis", fg_color="#2E7D32")
        # Load first frame
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            pil_img = cv2_to_pil(frame)
            ctk_img = pil_to_ctk(pil_img, (640, 420))
            self.video_lbl.configure(image=ctk_img, text="")
            self.video_lbl.image = ctk_img

    def _toggle_playback(self):
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return
        self.is_playing = True
        self.play_btn.configure(text="⏸️ Pause Video", fg_color="#C62828")
        threading.Thread(target=self._video_loop, daemon=True).start()

    def _stop_playback(self):
        self.is_playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.configure(text="▶️ Start Video Analysis", fg_color="#2E7D32")

    def _video_loop(self):
        self.cap = cv2.VideoCapture(self.video_path)
        prev_time = time.time()

        while self.is_playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Run YOLO frame detection
            count, annotated_frame, _ = self.detector.detect_video_frame(frame)
            self.current_count = count

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / max(1e-5, (curr_time - prev_time))
            prev_time = curr_time

            # Update UI in main thread
            pil_img = cv2_to_pil(annotated_frame)
            ctk_img = pil_to_ctk(pil_img, (640, 420))

            try:
                self.after(0, self._update_frame, ctk_img, count, fps)
            except Exception:
                break

            time.sleep(0.02) # ~30-40 fps cap

        if self.cap:
            self.cap.release()

    def _update_frame(self, ctk_img, count, fps):
        if not self.is_playing:
            return
        self.video_lbl.configure(image=ctk_img, text="")
        self.video_lbl.image = ctk_img
        self.counter_lbl.configure(text=f"Real-Time Detected Pests: {count}")
        self.fps_lbl.configure(text=f"Processing Speed: {fps:.1f} FPS")

    def destroy(self):
        self._stop_playback()
        super().destroy()
