from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from collections import deque
import math

app = FastAPI()

# =========================
# MODEL
# =========================
MODEL_PATH = r"C:\Users\rwank\OneDrive\Desktop\SAV\model\eyes_int8.tflite"
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
    return (yq - out_zero) * out_scale


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

# =========================
# STATE
# =========================
yawn_count = 0
yawn_active = False
head_down_start = None

MAR_THRESHOLD = 0.6



def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def calculate_mar(landmarks):
    upper = landmarks[13]
    lower = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]
    vertical = euclidean_distance(upper, lower)
    horizontal = euclidean_distance(left, right)
    return vertical / horizontal if horizontal != 0 else 0.0


# =========================
# CORE LOGIC
# =========================
def process_frame(frame):

    global yawn_count, yawn_active, head_down_start

    state = {
        "eyes": "UNKNOWN",
        "yawn": "NORMAL",
        "head": "OK"
    }

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        # =========================
        # YAWN
        # =========================
        mar = calculate_mar(lm)

        if mar > MAR_THRESHOLD and not yawn_active:
            yawn_count += 1
            yawn_active = True

        if mar < MAR_THRESHOLD:
            yawn_active = False

        if yawn_count >= 7:
            state["yawn"] = "DANGER"
        elif yawn_count >= 5:
            state["yawn"] = "DROWSY"
        else:
            state["yawn"] = "NORMAL"

        # =========================
        # HEAD
        # =========================
        pitch_sim = lm[1].y

        if pitch_sim > 0.6:
            if head_down_start is None:
                head_down_start = time.time()

            elapsed = time.time() - head_down_start

            if elapsed > 4:
                state["head"] = "DANGER"
            elif elapsed > 2:
                state["head"] = "ALERT"
            else:
                state["head"] = "OK"
        else:
            head_down_start = None
            state["head"] = "OK"

        # =========================
        # EYES (MODEL)
        # =========================
        eye_sample = frame[0:100, 0:100]
        p_open = predict_open_prob(eye_sample)

        state["eyes"] = "CLOSED" if p_open < 0.5 else "OPEN"

    return state


# =========================
# API
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    state = process_frame(frame)

    return state


@app.get("/")
def home():
    return {"status": "AI Server Running"}