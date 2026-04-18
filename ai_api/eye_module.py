import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "eyes_int8.tflite"
IMG_SIZE = 128
MARGIN = 0.6

THRESH_LOW = 0.30
THRESH_HIGH = 0.45

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

in_scale, in_zero = inp["quantization"]
out_scale, out_zero = out["quantization"]

LEFT_EYE = [33, 133, 160, 144, 159, 145]
RIGHT_EYE = [362, 263, 387, 373, 386, 374]


def predict_eye(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    x = img.astype(np.float32)
    xq = (x / in_scale + in_zero).astype(np.int8)[None, ...]

    interpreter.set_tensor(inp["index"], xq)
    interpreter.invoke()

    yq = interpreter.get_tensor(out["index"])[0][0]
    score = (yq - out_zero) * out_scale

    return float(score)


def crop_eye(frame, lm, idxs, w, h):
    xs = [lm[i].x * w for i in idxs]
    ys = [lm[i].y * h for i in idxs]

    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))

    bw = x2 - x1
    bh = y2 - y1

    x1 -= int(bw * MARGIN)
    x2 += int(bw * MARGIN)
    y1 -= int(bh * MARGIN)
    y2 += int(bh * MARGIN)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if (x2 - x1) < 40 or (y2 - y1) < 30:
        return None, None

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None, None

    return crop, (x1, y1, x2, y2)


def get_best_eye(frame, lm, w, h):

    left, lbox = crop_eye(frame, lm, LEFT_EYE, w, h)
    right, rbox = crop_eye(frame, lm, RIGHT_EYE, w, h)

    candidates = []

    if left is not None:
        candidates.append((left, lbox))
    if right is not None:
        candidates.append((right, rbox))

    if not candidates:
        return frame, (0, 0, w, h)

    return max(candidates, key=lambda x: x[0].shape[0] * x[0].shape[1])