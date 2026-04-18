from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
import mediapipe as mp

from eye_module import predict_eye, get_best_eye
from yawn_module import detect_yawn
from head_pose_module import detect_head_pose

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def decode_frame(data):
    try:
        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print("decode error:", e)
        return None


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WS CONNECTED")

    try:
        while True:
            data = await websocket.receive_text()

            frame = decode_frame(data)
            if frame is None:
                await websocket.send_json({"error": "bad_frame"})
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = face_mesh.process(rgb)

            if not res.multi_face_landmarks:
                await websocket.send_json({"status": "no_face"})
                continue

            lm = res.multi_face_landmarks[0].landmark

            # ================= EYE =================
            eye_img, box = get_best_eye(frame, lm, w, h)

            if eye_img is None:
                eye_state = "bad_crop"
                eye_score = 0.0
            else:
                eye_score = predict_eye(eye_img)
                eye_state = "closed" if eye_score < 0.30 else "open"

            # ================= YAWN =================
            yawn = detect_yawn(lm)

            # ================= HEAD =================
            head = detect_head_pose(lm, w, h)

            # ================= FINAL =================
            is_drowsy = (
                eye_state == "closed"
                or yawn["yawn_detected"]
                or head["head_alert"]
            )

            response = {
                "eye_state": eye_state,
                "eye_score": float(eye_score),
                "yawn_detected": bool(yawn["yawn_detected"]),
                "head_alert": bool(head["head_alert"]),
                "driver_status": "drowsy" if is_drowsy else "normal"
            }

            print(response)

            await websocket.send_json(response)

    except Exception as e:
        print("WS CRASH:", e)

    finally:
        print("WS DISCONNECTED")