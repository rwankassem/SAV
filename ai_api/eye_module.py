import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

# =========================
# Config
# =========================
WEIGHTS_PATH = r"C:\Users\rwank\OneDrive\Desktop\SAV\model\eyes_best.weights.h5"
IMG = 128

THRESH_OPEN = 0.45
THRESH_CLOSE = 0.33
MARGIN = 0.12
INVERT_OPEN_PROB = False

# =========================
# Build model architecture
# =========================
aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.25),
], name="augmentation")

noise = layers.GaussianNoise(0.02)

base = tf.keras.applications.MobileNetV3Small(
    input_shape=(IMG, IMG, 3),
    include_top=False,
    weights="imagenet"
)

inp = keras.Input(shape=(IMG, IMG, 3))
x = aug(inp)
x = noise(x)
x = tf.keras.applications.mobilenet_v3.preprocess_input(x)

base.trainable = False
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.20)(x)
out = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inp, out)

# Load trained weights
model.load_weights(WEIGHTS_PATH)

# =========================
# MediaPipe FaceMesh
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

LEFT_EYE_IDS = set()
RIGHT_EYE_IDS = set()

for a, b in mp_face_mesh.FACEMESH_LEFT_EYE:
    LEFT_EYE_IDS.add(a)
    LEFT_EYE_IDS.add(b)

for a, b in mp_face_mesh.FACEMESH_RIGHT_EYE:
    RIGHT_EYE_IDS.add(a)
    RIGHT_EYE_IDS.add(b)

EYE_IDS = LEFT_EYE_IDS.union(RIGHT_EYE_IDS)

# =========================
# Helper functions
# =========================
def resize_rgb_image(img_rgb: np.ndarray, size=(IMG, IMG)) -> np.ndarray:
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    pil_img = pil_img.resize(size)
    return np.array(pil_img)

def predict_open_prob(eye_rgb: np.ndarray) -> float:
    eye_rgb = resize_rgb_image(eye_rgb, (IMG, IMG))
    x = eye_rgb.astype(np.float32)
    x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    p = model.predict(x, verbose=0)[0][0]
    if INVERT_OPEN_PROB:
        p = 1.0 - p

    return float(np.clip(p, 0.0, 1.0))

def eye_bbox_from_landmarks(lms, w: int, h: int):
    xs, ys = [], []
    for idx in EYE_IDS:
        xs.append(int(lms[idx].x * w))
        ys.append(int(lms[idx].y * h))

    if not xs:
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    bw = x2 - x1
    bh = y2 - y1

    x1 = int(x1 - MARGIN * bw)
    x2 = int(x2 + MARGIN * bw)
    y1 = int(y1 - MARGIN * bh)
    y2 = int(y2 + MARGIN * bh)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2

# =========================
# Main API function
# =========================
def run_eye_inference(frame_rgb: np.ndarray):
    if frame_rgb is None:
        return {
            "eye_state": "unknown",
            "confidence": None
        }

    h, w = frame_rgb.shape[:2]
    res = face_mesh.process(frame_rgb)

    if not res.multi_face_landmarks:
        return {
            "eye_state": "unknown",
            "confidence": None
        }

    lms = res.multi_face_landmarks[0].landmark
    box = eye_bbox_from_landmarks(lms, w, h)

    if box is None:
        return {
            "eye_state": "unknown",
            "confidence": None
        }

    x1, y1, x2, y2 = box
    eye_crop = frame_rgb[y1:y2, x1:x2]

    if eye_crop.size == 0:
        return {
            "eye_state": "unknown",
            "confidence": None
        }

    p_open = predict_open_prob(eye_crop)

    if p_open >= THRESH_OPEN:
        eye_state = "open"
    elif p_open <= THRESH_CLOSE:
        eye_state = "closed"
    else:
        eye_state = "uncertain"

    return {
        "eye_state": eye_state,
        "confidence": round(float(p_open), 4)
    }