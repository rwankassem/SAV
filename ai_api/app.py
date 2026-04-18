from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
import mediapipe as mp
import cv2

from eye_module import predict_eye, get_best_eye, THRESH_LOW, THRESH_HIGH
from yawn_module import detect_yawn
from head_pose_module import detect_head_pose

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

eye_history = []


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")
    frame = np.array(img)

    rgb = frame.copy()
    h, w = frame.shape[:2]

    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        return {"eye_state": "no_face"}

    lm = res.multi_face_landmarks[0].landmark

    # ================= EYE =================
    eye_img, box = get_best_eye(frame, lm, w, h)

    score = predict_eye(eye_img)

    eye_history.append(score)
    if len(eye_history) > 3:
        eye_history.pop(0)

    smooth_score = sum(eye_history) / len(eye_history)

    # ================= HYSTERESIS LOGIC =================
    if smooth_score < THRESH_LOW:
        eye_state = "closed"
    elif smooth_score > THRESH_HIGH:
        eye_state = "open"
    else:
        eye_state = "open"  # keep stable default

    print("\n========== EYE DEBUG ==========")
    print("RAW SCORE:", score)
    print("SMOOTH SCORE:", smooth_score)
    print("STATE:", eye_state)
    print("THRESH LOW:", THRESH_LOW)
    print("THRESH HIGH:", THRESH_HIGH)
    print("BOX:", box)

    # ================= YAWN =================
    yawn = detect_yawn(lm)
    yawn_flag = bool(yawn["yawn_detected"])

    # ================= HEAD =================
    head = detect_head_pose(lm, w, h)
    head_flag = bool(head["head_alert"])

    # ================= FINAL =================
    is_drowsy = (eye_state == "closed" or yawn_flag or head_flag)

    print("\n========== FINAL ==========")
    print("eye:", eye_state)
    print("yawn:", yawn_flag)
    print("head:", head_flag)

    return {
        "eye_state": eye_state,
        "eye_score": float(smooth_score),
        "yawn_detected": yawn_flag,
        "head_alert": head_flag,
        "driver_status": "drowsy" if is_drowsy else "normal"
    }