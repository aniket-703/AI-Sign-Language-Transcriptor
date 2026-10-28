import os
import sys
import time
import pickle
import threading
import tkinter as tk
import customtkinter as ctk
from PIL import Image
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import enchant

# Setup directories dynamically relative to script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'Model')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'output.txt')

# Hand skeleton connections for landmark drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17), (5, 9), (9, 13), (13, 17)     # Palm
]

LABELS_DICT = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 
    19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 
    28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'Next'
}

CUSTOM_WORDS = {
    'ANIKET', 'SHARMA', 'VISHAL', 'DIVYANSH', 'GUPTA', 'MANYA', 'AYUSH', 'SONIKA'
}


def draw_clean_landmarks(frame, hand_landmarks_list):
    """Draws refined, subtle hand skeleton lines and joints on an RGB frame."""
    h, w, _ = frame.shape
    for landmarks in hand_landmarks_list:
        # Subtle glowing connections
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (56, 189, 248), 2, cv2.LINE_AA)
        # Clean small nodes
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (14, 165, 233), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)


def probe_available_cameras(max_tested=4):
    """Scans camera devices and automatically selects the first active camera."""
    found = []
    best_idx = 1  # Default to 1 based on hardware scan
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                mean_val = float(np.mean(frame))
                is_active = mean_val > 5.0
                label = f"Camera {idx} ({frame.shape[1]}x{frame.shape[0]}{' • Active' if is_active else ' • Idle'})"
                found.append((idx, label, is_active))
                if is_active:
                    best_idx = idx
            cap.release()
    if not found:
        found.append((0, "Camera 0 (Default)", True))
    return found, best_idx


class CleanSignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Appearance configuration: Sleek modern dark mode
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("SignScript • AI Sign Transcriptor")
        self.geometry("1240x780")
        self.minsize(1050, 700)
        self.configure(fg_color="#0d0f12")

        # Window icon
        logo_path = os.path.join(IMAGES_DIR, 'logo.ico')
        if os.path.exists(logo_path):
            try:
                self.iconbitmap(logo_path)
            except Exception:
                pass

        # Load AI Model & MediaPipe
        self.load_model()
        self.init_mediapipe()

        # Spell Checker
        try:
            self.english_dict = enchant.Dict("en_US")
        except Exception:
            self.english_dict = None

        # State Variables
        self.is_active = True
        self.current_symbol = "-"
        self.confidence = 0.0
        self.word = ""
        self.sentence = ""
        self.prev_symbol = ""
        self.last_symbol_time = time.time()
        self.hold_threshold = 1.8  # seconds
        self.fps = 0.0
        self.prev_time = time.time()
        self.ref_window = None

        # Initialize Camera
        self.camera_list, default_idx = probe_available_cameras()
        self.active_camera_idx = default_idx
        self.vid = None
        self.open_camera(self.active_camera_idx)

        # Build Clean UI Layout
        self.build_ui()

        # Start Video Rendering Loop
        self.after(20, self.update_frame)

    def load_model(self):
        """Loads classifier with scikit-learn 1.8 compatibility."""
        model_path = os.path.join(MODEL_DIR, 'model_final.p')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, 'model_Final.p')
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        for est in getattr(self.model, 'estimators_', []):
            if not hasattr(est, 'monotonic_cst'):
                est.monotonic_cst = None

    def init_mediapipe(self):
        """Loads MediaPipe Tasks HandLandmarker."""
        task_path = os.path.join(MODEL_DIR, 'hand_landmarker.task')
        if not os.path.exists(task_path):
            import urllib.request
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                task_path
            )
        base_options = python.BaseOptions(model_asset_path=task_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)

    def open_camera(self, idx):
        """Opens camera safely with DirectShow backend."""
        if self.vid is not None and self.vid.isOpened():
            self.vid.release()
        self.vid = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.vid.isOpened():
            self.vid = cv2.VideoCapture(idx)
        self.active_camera_idx = idx

    def build_ui(self):
        """Constructs a clean, minimalist UI layout."""
        # 1. Top Minimal Navbar
        navbar = ctk.CTkFrame(self, height=54, corner_radius=0, fg_color="#121418")
        navbar.pack(fill="x", padx=0, pady=0)

        # Brand / Title
        brand_frame = ctk.CTkFrame(navbar, fg_color="transparent")
        brand_frame.pack(side="left", padx=20, pady=8)

        logo_lbl = ctk.CTkLabel(
            brand_frame,
            text="🤟 SignScript",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color="#38bdf8"
        )
        logo_lbl.pack(side="left", padx=(0, 8))

        tag_lbl = ctk.CTkLabel(
            brand_frame,
            text="AI Real-Time Transcriptor",
            font=ctk.CTkFont(size=11),
            text_color="#64748b"
        )
        tag_lbl.pack(side="left")

        # Right Bar Controls
        nav_right = ctk.CTkFrame(navbar, fg_color="transparent")
        nav_right.pack(side="right", padx=20, pady=8)

        # Status Pill
        self.status_pill = ctk.CTkLabel(
            nav_right,
            text="● Live",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#34d399",
            fg_color="#064e3b",
            corner_radius=12,
            padx=10,
            pady=2
        )
        self.status_pill.pack(side="left", padx=6)

        # Camera Dropdown
        cam_names = [label for _, label, _ in self.camera_list]
        selected_cam_name = cam_names[0]
        for idx, lbl, _ in self.camera_list:
            if idx == self.active_camera_idx:
                selected_cam_name = lbl
                break

        self.cam_select = ctk.CTkOptionMenu(
            nav_right,
            values=cam_names,
            command=self.on_camera_selected,
            width=200,
            height=28,
            corner_radius=8,
            fg_color="#1e222b",
            button_color="#2b313d",
            button_hover_color="#38404f",
            font=ctk.CTkFont(size=11)
        )
        self.cam_select.set(selected_cam_name)
        self.cam_select.pack(side="left", padx=6)

        # Reference Chart Modal Button
        self.btn_ref_modal = ctk.CTkButton(
            nav_right,
            text="📖 Signs Chart",
            command=self.open_reference_chart,
            height=28,
            corner_radius=8,
            fg_color="#1e222b",
            hover_color="#2b313d",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8"
        )
        self.btn_ref_modal.pack(side="left", padx=6)

        # 2. Main Workspace Split
        workspace = ctk.CTkFrame(self, fg_color="transparent")
        workspace.pack(fill="both", expand=True, padx=20, pady=16)
        workspace.grid_columnconfigure(0, weight=6)
        workspace.grid_columnconfigure(1, weight=5)
        workspace.grid_rowconfigure(0, weight=1)

        # LEFT CARD: Video Viewport
        video_card = ctk.CTkFrame(
            workspace,
            corner_radius=16,
            fg_color="#14171d",
            border_width=1,
            border_color="#212631"
        )
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        # Camera Feed Container
        video_box = ctk.CTkFrame(video_card, corner_radius=12, fg_color="#0a0c0f")
        video_box.pack(fill="both", expand=True, padx=14, pady=14)

        # Placeholder / Live Image Label
        self.video_display = ctk.CTkLabel(video_box, text="", corner_radius=12)
        self.video_display.pack(fill="both", expand=True)

        # Bottom Video Controls Bar
        video_footer = ctk.CTkFrame(video_card, fg_color="transparent")
        video_footer.pack(fill="x", padx=14, pady=(0, 14))

        # Play/Pause Button
        self.btn_pause = ctk.CTkButton(
            video_footer,
            text="⏸ Pause",
            command=self.toggle_capture,
            width=90,
            height=32,
            corner_radius=8,
            fg_color="#ef4444",
            hover_color="#dc2626",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.btn_pause.pack(side="left", padx=(0, 10))

        # Hold Threshold Slider
        self.slider_label = ctk.CTkLabel(
            video_footer,
            text=f"Lock delay: {self.hold_threshold:.1f}s",
            font=ctk.CTkFont(size=11),
            text_color="#94a3b8"
        )
        self.slider_label.pack(side="left", padx=(0, 8))

        self.hold_slider = ctk.CTkSlider(
            video_footer,
            from_=0.8,
            to=3.2,
            number_of_steps=24,
            command=self.on_slider_changed,
            height=14,
            button_color="#38bdf8",
            progress_color="#0284c7"
        )
        self.hold_slider.set(self.hold_threshold)
        self.hold_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # FPS & Res Tag
        self.fps_tag = ctk.CTkLabel(
            video_footer,
            text="-- FPS",
            font=ctk.CTkFont(size=11),
            text_color="#64748b"
        )
        self.fps_tag.pack(side="right")

        # RIGHT CARD: Transcription Workspace
        trans_card = ctk.CTkFrame(
            workspace,
            corner_radius=16,
            fg_color="#14171d",
            border_width=1,
            border_color="#212631"
        )
        trans_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)

        # Zone 1: Hero Prediction Tile
        hero_tile = ctk.CTkFrame(trans_card, corner_radius=14, fg_color="#1a1e27", border_width=1, border_color="#2a303e")
        hero_tile.pack(fill="x", padx=16, pady=16)

        hero_top = ctk.CTkFrame(hero_tile, fg_color="transparent")
        hero_top.pack(fill="x", padx=16, pady=(12, 4))

        ctk.CTkLabel(
            hero_top,
            text="DETECTED SIGN",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#64748b"
        ).pack(side="left")

        self.conf_badge = ctk.CTkLabel(
            hero_top,
            text="0% Match",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#38bdf8"
        )
        self.conf_badge.pack(side="right")

        self.hero_char = ctk.CTkLabel(
            hero_tile,
            text="-",
            font=ctk.CTkFont(size=44, weight="bold"),
            text_color="#f8fafc"
        )
        self.hero_char.pack(pady=(0, 4))

        # Slim Hold Progress Line
        self.lock_progress = ctk.CTkProgressBar(
            hero_tile,
            height=4,
            corner_radius=2,
            progress_color="#38bdf8",
            fg_color="#262d3a"
        )
        self.lock_progress.set(0.0)
        self.lock_progress.pack(fill="x", padx=16, pady=(0, 12))

        # Zone 2: Word Buffer Tile
        word_tile = ctk.CTkFrame(trans_card, corner_radius=12, fg_color="#1a1e27", border_width=1, border_color="#2a303e")
        word_tile.pack(fill="x", padx=16, pady=(0, 14))

        word_inner = ctk.CTkFrame(word_tile, fg_color="transparent")
        word_inner.pack(fill="x", padx=14, pady=10)

        ctk.CTkLabel(
            word_inner,
            text="WORD",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#64748b"
        ).pack(side="left", padx=(0, 10))

        self.word_display = ctk.CTkLabel(
            word_inner,
            text="",
            font=ctk.CTkFont(size=18, weight="bold", family="Consolas"),
            text_color="#34d399",
            anchor="w"
        )
        self.word_display.pack(side="left", fill="x", expand=True)

        self.btn_back = ctk.CTkButton(
            word_inner,
            text="⌫",
            width=36,
            height=28,
            command=self.backspace,
            corner_radius=6,
            fg_color="#2a303e",
            hover_color="#38404f",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.btn_back.pack(side="right")

        # Zone 3: Full Sentence Reading Area
        sent_header = ctk.CTkFrame(trans_card, fg_color="transparent")
        sent_header.pack(fill="x", padx=18, pady=(4, 6))

        ctk.CTkLabel(
            sent_header,
            text="TRANSCRIBED TEXT",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="#64748b"
        ).pack(side="left")

        self.counter_lbl = ctk.CTkLabel(
            sent_header,
            text="0 words • 0 chars",
            font=ctk.CTkFont(size=10),
            text_color="#64748b"
        )
        self.counter_lbl.pack(side="right")

        self.sentence_box = ctk.CTkTextbox(
            trans_card,
            font=ctk.CTkFont(size=14),
            fg_color="#0d0f12",
            border_width=1,
            border_color="#212631",
            corner_radius=10,
            wrap="word"
        )
        self.sentence_box.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        # Zone 4: Clean Action Toolbar
        action_bar = ctk.CTkFrame(trans_card, fg_color="transparent")
        action_bar.pack(fill="x", padx=16, pady=(0, 16))

        self.btn_speak = ctk.CTkButton(
            action_bar,
            text="🔊 Speak",
            command=self.speak_text,
            width=110,
            height=34,
            corner_radius=8,
            fg_color="#0284c7",
            hover_color="#0369a1",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.btn_speak.pack(side="left", padx=(0, 8))

        self.btn_copy = ctk.CTkButton(
            action_bar,
            text="📋 Copy",
            command=self.copy_text,
            width=80,
            height=34,
            corner_radius=8,
            fg_color="#1e222b",
            hover_color="#2b313d",
            font=ctk.CTkFont(size=12)
        )
        self.btn_copy.pack(side="left", padx=(0, 8))

        self.btn_clear = ctk.CTkButton(
            action_bar,
            text="🗑️ Clear",
            command=self.clear_all,
            width=80,
            height=34,
            corner_radius=8,
            fg_color="#1e222b",
            hover_color="#7f1d1d",
            font=ctk.CTkFont(size=12)
        )
        self.btn_clear.pack(side="right")

    def on_camera_selected(self, choice):
        """Switches camera device smoothly."""
        for idx, lbl, _ in self.camera_list:
            if lbl == choice:
                self.open_camera(idx)
                break

    def on_slider_changed(self, val):
        self.hold_threshold = float(val)
        self.slider_label.configure(text=f"Lock delay: {self.hold_threshold:.1f}s")

    def toggle_capture(self):
        self.is_active = not self.is_active
        if self.is_active:
            self.btn_pause.configure(text="⏸ Pause", fg_color="#ef4444", hover_color="#dc2626")
            self.status_pill.configure(text="● Live", text_color="#34d399", fg_color="#064e3b")
            self.last_symbol_time = time.time()
        else:
            self.btn_pause.configure(text="▶ Resume", fg_color="#10b981", hover_color="#059669")
            self.status_pill.configure(text="⏸ Paused", text_color="#fbbf24", fg_color="#451a03")
            self.hero_char.configure(text="-")
            self.lock_progress.set(0.0)

    def backspace(self):
        if self.word:
            self.word = self.word[:-1]
            self.word_display.configure(text=self.word)

    def clear_all(self):
        self.word = ""
        self.sentence = ""
        self.word_display.configure(text="")
        self.sentence_box.delete("1.0", "end")
        self.hero_char.configure(text="-")
        self.conf_badge.configure(text="0% Match")
        self.lock_progress.set(0.0)
        self.counter_lbl.configure(text="0 words • 0 chars")

    def copy_text(self):
        text = self.sentence_box.get("1.0", "end-1c").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.btn_copy.configure(text="✓ Copied")
            self.after(1500, lambda: self.btn_copy.configure(text="📋 Copy"))

    def speak_text(self):
        text = self.sentence_box.get("1.0", "end-1c").strip() or self.word.strip()
        if not text:
            return
        def _tts():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 145)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"TTS error: {e}")
        threading.Thread(target=_tts, daemon=True).start()

    def spell_correct(self, user_input):
        """Auto-corrects typos using PyEnchant and custom vocabulary."""
        if not user_input.strip() or self.english_dict is None:
            return user_input.strip()

        corrected_words = []
        for word in user_input.split():
            upper = word.upper()
            if upper in CUSTOM_WORDS:
                corrected_words.append(upper)
            elif self.english_dict.check(word):
                corrected_words.append(word)
            else:
                suggestions = self.english_dict.suggest(word)
                if suggestions:
                    corrected_words.append(suggestions[0])
                else:
                    corrected_words.append(word)
        return ' '.join(corrected_words)

    def open_reference_chart(self):
        """Opens clean modal window displaying the 37 sign gestures."""
        if self.ref_window is not None and self.ref_window.winfo_exists():
            self.ref_window.lift()
            return

        ref_path = os.path.join(IMAGES_DIR, 'symbol.png')
        if not os.path.exists(ref_path):
            return

        self.ref_window = ctk.CTkToplevel(self)
        self.ref_window.title("Sign Language Reference Chart")
        self.ref_window.geometry("540x560")
        self.ref_window.configure(fg_color="#0d0f12")

        chart_pil = Image.open(ref_path)
        chart_img = ctk.CTkImage(light_image=chart_pil, dark_image=chart_pil, size=(500, 480))
        lbl = ctk.CTkLabel(self.ref_window, image=chart_img, text="", corner_radius=12)
        lbl.pack(padx=20, pady=20, fill="both", expand=True)

    def update_frame(self):
        """Main real-time computer vision and rendering pipeline."""
        ret = False
        frame = None
        if self.vid is not None and self.vid.isOpened():
            ret, frame = self.vid.read()

        curr_time = time.time()
        self.fps = 1.0 / max(0.001, (curr_time - self.prev_time))
        self.prev_time = curr_time

        if ret and frame is not None:
            # Horizontal flip for natural interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.is_active:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = self.hand_detector.detect(mp_img)

                if results.hand_landmarks:
                    draw_clean_landmarks(rgb_frame, results.hand_landmarks)
                    hand = results.hand_landmarks[0]
                    x_ = [lm.x for lm in hand]
                    y_ = [lm.y for lm in hand]

                    data_aux = []
                    for lm in hand:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))
                    data_aux.extend([x**2 for x in data_aux])

                    try:
                        prediction = self.model.predict([np.asarray(data_aux)])
                        raw_pred = prediction[0]
                        if isinstance(raw_pred, str) and raw_pred.isdigit():
                            self.current_symbol = LABELS_DICT.get(int(raw_pred), raw_pred)
                        elif isinstance(raw_pred, (int, np.integer)):
                            self.current_symbol = LABELS_DICT.get(int(raw_pred), str(raw_pred))
                        else:
                            self.current_symbol = str(raw_pred)

                        probs = self.model.predict_proba([np.asarray(data_aux)])[0]
                        self.confidence = float(np.max(probs)) * 100
                    except Exception:
                        self.current_symbol = "?"
                        self.confidence = 0.0

                    # Draw clean minimal bounding box and tag
                    x1 = max(0, int(min(x_) * w) - 15)
                    y1 = max(0, int(min(y_) * h) - 15)
                    x2 = min(w, int(max(x_) * w) + 15)
                    y2 = min(h, int(max(y_) * h) + 15)
                    cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (56, 189, 248), 2)

                    # Update Hero Labels
                    self.hero_char.configure(text=self.current_symbol)
                    self.conf_badge.configure(text=f"{self.confidence:.0f}% Match")

                    # Handle Character Lock Logic
                    if self.current_symbol not in ("?", "Blank", ""):
                        if self.current_symbol == "Next":
                            if self.word:
                                corrected = self.spell_correct(self.word)
                                self.sentence += (" " if self.sentence else "") + corrected
                                self.word = ""
                                self.word_display.configure(text="")
                                self.sentence_box.delete("1.0", "end")
                                self.sentence_box.insert("1.0", self.sentence)

                                words_cnt = len(self.sentence.split())
                                chars_cnt = len(self.sentence)
                                self.counter_lbl.configure(text=f"{words_cnt} words • {chars_cnt} chars")
                            self.last_symbol_time = time.time()
                            self.lock_progress.set(0.0)
                        else:
                            if self.prev_symbol == self.current_symbol:
                                elapsed = time.time() - self.last_symbol_time
                                progress = min(1.0, elapsed / self.hold_threshold)
                                self.lock_progress.set(progress)

                                if elapsed >= self.hold_threshold:
                                    self.word += self.current_symbol
                                    self.word_display.configure(text=self.word)
                                    self.last_symbol_time = time.time()
                                    self.lock_progress.set(0.0)
                            else:
                                self.last_symbol_time = time.time()
                                self.lock_progress.set(0.0)
                    self.prev_symbol = self.current_symbol
                else:
                    self.hero_char.configure(text="-")
                    self.conf_badge.configure(text="Waiting...")
                    self.lock_progress.set(0.0)
                    self.prev_symbol = ""

            self.fps_tag.configure(text=f"{int(self.fps)} FPS • {w}x{h}")

            # Resize nicely for display
            target_w = 640
            target_h = int(640 * (h / w))
            resized = cv2.resize(rgb_frame, (target_w, target_h))

            pil_img = Image.fromarray(resized)
            self.current_ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(target_w, target_h))
            self.video_display.configure(image=self.current_ctk_img)

        self.after(20, self.update_frame)

    def destroy(self):
        if self.vid is not None and self.vid.isOpened():
            self.vid.release()
        super().destroy()


if __name__ == '__main__':
    app = CleanSignLanguageApp()
    app.mainloop()
