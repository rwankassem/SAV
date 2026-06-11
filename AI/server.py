from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import time
import math
from collections import deque

app = FastAPI()

# =========================
# MODEL
# =========================
MODEL_PATH = "eyes_int8.tflite"
IMG = 128

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

in_scale, in_zero = inp["quantization"]
out_scale, out_zero = out["quantization"]

def predict_open_prob(eye_bgr):
    eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
    eye_rgb = cv2.resize(eye_rgb, (IMG, IMG))
    x = eye_rgb.astype(np.float32)

    xq = (x / in_scale + in_zero).astype(np.int8)[None, ...]
    interpreter.set_tensor(inp["index"], xq)
    interpreter.invoke()

    yq = interpreter.get_tensor(out["index"])[0][0]
    y = (yq - out_zero) * out_scale
    return float(y)

# =========================
# MEDIAPIPE
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_IDS = set()
RIGHT_EYE_IDS = set()

for a, b in mp_face_mesh.FACEMESH_LEFT_EYE:
    LEFT_EYE_IDS.update([a, b])
for a, b in mp_face_mesh.FACEMESH_RIGHT_EYE:
    RIGHT_EYE_IDS.update([a, b])

EYE_IDS = LEFT_EYE_IDS | RIGHT_EYE_IDS

def eye_bbox(lms, w, h):
    xs, ys = [], []
    for i in EYE_IDS:
        xs.append(int(lms[i].x * w))
        ys.append(int(lms[i].y * h))

    if not xs:
        return None

    return min(xs), min(ys), max(xs), max(ys)

# =========================
# STATE MEMORY (important)
# =========================
hist_close = deque(maxlen=11)
state = "OPEN"
eye_close_start = None
eye_alert = False
eye_danger = False

# =========================
# HEAD (simplified)
# =========================
yawn_count = 0
yawn_active = False

MAR_THRESHOLD = 0.6

def euclidean(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def MAR(lm):
    up = lm[13]
    down = lm[14]
    l = lm[61]
    r = lm[291]

    return euclidean(up, down) / euclidean(l, r)

# =========================
# API
# =========================
@app.post("/esp")
async def predict(file: UploadFile = File(...)):

    img_bytes = await file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    response = {
        "eyes": "UNKNOWN",
        "yawn": yawn_count,
        "head": "OK"
    }

    if not res.multi_face_landmarks:
        return response

    lm = res.multi_face_landmarks[0].landmark

    # =========================
    # EYES
    # =========================
    box = eye_bbox(lm, w, h)

    if box:
        x1, y1, x2, y2 = box
        eye_crop = frame[y1:y2, x1:x2]

        if eye_crop.size != 0:
            p_open = predict_open_prob(eye_crop)
            p_open = 1 - p_open
            p_close = 1 - p_open

            hist_close.append(p_close)
            avg = float(sum(hist_close) / len(hist_close))

            if state == "OPEN" and avg >= 0.67:
                state = "CLOSED"
            elif state == "CLOSED" and avg <= 0.55:
                state = "OPEN"

            # timing logic
            if state == "CLOSED":
                global eye_close_start
                if eye_close_start is None:
                    eye_close_start = time.time()

                t = time.time() - eye_close_start

                if t >= 2:
                    eye_alert = True
                if t >= 4:
                    eye_danger = True
            else:
                eye_close_start = None
                eye_alert = False
                eye_danger = False

            response["eyes"] = state

    # =========================
    # YAWN (simplified MAR)
    # =========================
    mar = MAR(lm)

    if mar > MAR_THRESHOLD and not yawn_active:
        yawn_count += 1
        yawn_active = True

    if mar < MAR_THRESHOLD:
        yawn_active = False

    response["yawn"] = yawn_count

    return response