import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "../model/eyes_int8.tflite"
IMG = 96
INVERT_OPEN_PROB = True

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

in_scale, in_zero = inp["quantization"]
out_scale, out_zero = out["quantization"]

def predict_open_prob_raw(eye_bgr: np.ndarray) -> float:
    eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
    eye_rgb = cv2.resize(eye_rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    x = eye_rgb.astype(np.float32)

    xq = (x / in_scale + in_zero).astype(np.int8)[None, ...]
    interpreter.set_tensor(inp["index"], xq)
    interpreter.invoke()

    yq = interpreter.get_tensor(out["index"])[0][0]
    y = (yq - out_zero) * out_scale
    return float(y)

def predict_open_prob(eye_bgr: np.ndarray) -> float:
    p = predict_open_prob_raw(eye_bgr)
    if INVERT_OPEN_PROB:
        p = 1.0 - p
    return float(np.clip(p, 0.0, 1.0))

# Alias
predict_eye = predict_open_prob

# Thresholds
THRESH_LOW = 0.35
THRESH_HIGH = 0.65

# Eye landmarks
LEFT_EYE_IDS = {33, 133, 145, 153, 158, 159, 160}
RIGHT_EYE_IDS = {362, 263, 373, 380, 381, 382, 384}
EYE_IDS = LEFT_EYE_IDS.union(RIGHT_EYE_IDS)

def eye_bbox_from_landmarks(lms, w: int, h: int):
    xs, ys = [], []
    for idx in EYE_IDS:
        x = int(lms[idx].x * w)
        y = int(lms[idx].y * h)
        xs.append(x)
        ys.append(y)

    if not xs:
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    bw = x2 - x1
    bh = y2 - y1

    # Expand
    margin = 0.1
    x1 = max(0, int(x1 - margin * bw))
    y1 = max(0, int(y1 - margin * bh))
    x2 = min(w, int(x2 + margin * bw))
    y2 = min(h, int(y2 + margin * bh))

    return x1, y1, x2, y2

def get_best_eye(frame, lm, w, h):
    box = eye_bbox_from_landmarks(lm, w, h)
    if box is None:
        return None, None
    x1, y1, x2, y2 = box
    eye_img = frame[y1:y2, x1:x2]
    return eye_img, box