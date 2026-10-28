import os
import time
import pickle
import threading
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyttsx3
import enchant

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'Model')
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hand skeleton connections for visualization
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

class SignTranscriptorEngine:
    def __init__(self):
        # Locate model
        model_path = os.path.join(MODEL_DIR, 'model_final.p')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, 'model_Final.p')
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']

        # Ensure compatibility
        for est in getattr(self.model, 'estimators_', []):
            if not hasattr(est, 'monotonic_cst'):
                est.monotonic_cst = None

        # Ensure hand landmarker task file
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
        self.detector = vision.HandLandmarker.create_from_options(options)

        try:
            self.english_dict = enchant.Dict("en_US")
        except Exception:
            self.english_dict = None

    def draw_landmarks(self, frame_bgr, landmarks_list):
        """Draws skeleton lines and landmarks on OpenCV BGR frame."""
        h, w, _ = frame_bgr.shape
        for landmarks in landmarks_list:
            # Draw connections
            for p1_idx, p2_idx in HAND_CONNECTIONS:
                p1 = landmarks[p1_idx]
                p2 = landmarks[p2_idx]
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 230, 115), 2, cv2.LINE_AA)
            # Draw joints
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (cx, cy), 5, (255, 120, 0), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, (cx, cy), 3, (0, 240, 255), -1, cv2.LINE_AA)

    def predict_hand(self, hand_landmarks):
        """Extracts normalized coordinates and squared terms, runs prediction."""
        x_ = [lm.x for lm in hand_landmarks]
        y_ = [lm.y for lm in hand_landmarks]

        data_aux = []
        for lm in hand_landmarks:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        # 42 raw + 42 squared = 84 features
        data_aux.extend([x**2 for x in data_aux])

        try:
            pred = self.model.predict([np.asarray(data_aux)])[0]
            if isinstance(pred, str) and pred.isdigit():
                symbol = LABELS_DICT.get(int(pred), pred)
            elif isinstance(pred, (int, np.integer)):
                symbol = LABELS_DICT.get(int(pred), str(pred))
            else:
                symbol = str(pred)

            # Get confidence
            probs = self.model.predict_proba([np.asarray(data_aux)])[0]
            confidence = float(np.max(probs))
            return symbol, confidence, (min(x_), min(y_), max(x_), max(y_))
        except Exception:
            return "?", 0.0, None

    def spell_correct(self, user_input):
        """Spell-checks words using custom names and English dictionary."""
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

    def speak_text(self, text):
        """Speaks text in a background thread."""
        def _speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 145)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"TTS Error: {e}")
        threading.Thread(target=_speak, daemon=True).start()

    def transcribe_video_file(self, video_path, sample_rate_fps=4):
        """
        Transcribes a video file containing sign language gestures.
        Returns a list of timestamped segments and full text.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(fps / sample_rate_fps))
        
        current_word = ""
        sentence = ""
        segments = []
        last_symbol = ""
        symbol_count = 0
        current_sec = 0.0

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                current_sec = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = self.detector.detect(mp_img)

                if results.hand_landmarks:
                    symbol, conf, _ = self.predict_hand(results.hand_landmarks[0])
                    if symbol not in ("?", "Blank", ""):
                        if symbol == last_symbol:
                            symbol_count += 1
                        else:
                            last_symbol = symbol
                            symbol_count = 1

                        # Require symbol persistence across consecutive samples (~1s)
                        if symbol_count == sample_rate_fps:
                            if symbol == "Next":
                                if current_word:
                                    corrected = self.spell_correct(current_word)
                                    sentence += (" " if sentence else "") + corrected
                                    segments.append({
                                        "start": max(0.0, current_sec - 1.5),
                                        "end": current_sec,
                                        "word": corrected,
                                        "sentence": sentence
                                    })
                                    current_word = ""
                            else:
                                current_word += symbol
                                segments.append({
                                    "start": max(0.0, current_sec - 1.0),
                                    "end": current_sec,
                                    "char": symbol,
                                    "current_word": current_word
                                })
                else:
                    last_symbol = ""
                    symbol_count = 0

            frame_idx += 1

        cap.release()

        # Final word flush
        if current_word:
            corrected = self.spell_correct(current_word)
            sentence += (" " if sentence else "") + corrected
            segments.append({
                "start": max(0.0, current_sec - 1.0),
                "end": current_sec,
                "word": corrected,
                "sentence": sentence
            })

        return {
            "transcript": sentence.strip(),
            "segments": segments,
            "duration": current_sec
        }
