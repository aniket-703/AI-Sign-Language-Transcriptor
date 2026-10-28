import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MODEL_DIR = os.path.join(BASE_DIR, 'Model')
os.makedirs(MODEL_DIR, exist_ok=True)
TASK_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')

# Ensure MediaPipe HandLandmarker model file is present
if not os.path.exists(TASK_PATH):
    import urllib.request
    print("Downloading hand_landmarker.task model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        TASK_PATH
    )

base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
detector = vision.HandLandmarker.create_from_options(options)

data = []
labels = []

if not os.path.exists(DATA_DIR):
    print(f"Data directory '{DATA_DIR}' not found. Please run collect_imgs.py first.")
    exit(1)

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = detector.detect(mp_image)

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                data_aux = []
                x_ = [lm.x for lm in hand_landmarks]
                y_ = [lm.y for lm in hand_landmarks]

                for lm in hand_landmarks:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

output_pickle = os.path.join(MODEL_DIR, 'data_final.pickle')
try:
    with open(output_pickle, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Dataset successfully created with {len(data)} samples saved to: {output_pickle}")
except Exception as e:
    print(f"Error occurred while saving dataset: {e}")
