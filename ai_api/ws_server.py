from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import base64
import cv2
import numpy as np
import time
import mediapipe as mp

from ws_eye_service import predict_eye
from ws_yawn_service import WSYawnDetector

app = FastAPI()

# ================= EYE =================
eye_state = "open"
eye_start = None

CLOSE_THR = 0.35
DROWSY_TIME = 2.0
DANGER_TIME = 5.0

eye_level = "normal"

# ================= YAWN =================
yawn_model = WSYawnDetector()
last_yawn_time = 0
yawn_level = "normal"


def decode(data):
    try:
        img = base64.b64decode(data)
        arr = np.frombuffer(img, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None


mp_face_mesh = mp.solutions.face_mesh


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    print("CONNECTED")

    global eye_state, eye_start, eye_level
    global yawn_level, last_yawn_time

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        try:
            while True:
                data = await websocket.receive_text()
                frame = decode(data)

                if frame is None:
                    await websocket.send_json({"status": "bad"})
                    continue

                now = time.time()

                # ================= EYE =================
                score = predict_eye(frame)
                eye = "closed" if score < CLOSE_THR else "open"

                if eye == "closed":
                    if eye_state != "closed":
                        eye_start = now
                    eye_state = "closed"
                    dur = now - eye_start if eye_start else 0
                else:
                    eye_state = "open"
                    eye_start = None
                    dur = 0
                    eye_level = "normal"   # RESET مهم جدًا

                # update eye level ONLY when closed
                if eye_state == "closed":
                    if dur >= DANGER_TIME:
                        eye_level = "danger"
                    elif dur >= DROWSY_TIME:
                        eye_level = "drowsy"
                    else:
                        eye_level = "normal"

                # ================= YAWN =================
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)

                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    yawn = yawn_model.predict(lm)

                    if yawn["yawn"]:
                        last_yawn_time = now
                else:
                    yawn = {"yawn": False, "count": 0, "mar": 0.0}

                # YAWN LEVEL LOGIC (stable)
                if yawn["count"] >= 6:
                    yawn_level = "danger"
                elif yawn["count"] >= 3:
                    yawn_level = "drowsy"
                else:
                    # decay system
                    if now - last_yawn_time > 4:
                        yawn_level = "normal"

                # ================= FINAL DECISION =================
                if eye_level == "danger" or yawn_level == "danger":
                    final = "danger"
                elif eye_level == "drowsy" or yawn_level == "drowsy":
                    final = "drowsy"
                else:
                    final = "normal"

                await websocket.send_json({
                    "eye": eye_state,
                    "eye_duration": round(dur, 2),
                    "eye_level": eye_level,

                    "yawn": yawn["yawn"],
                    "yawn_count": yawn["count"],
                    "yawn_level": yawn_level,

                    "drowsy": final
                })

        except WebSocketDisconnect:
            print("DISCONNECTED")