from fastapi import FastAPI, WebSocket
import base64
import cv2
import numpy as np

app = FastAPI()

def decode_frame(data):
    try:
        img_bytes = base64.b64decode(data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except:
        return None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WS CONNECTED")

    while True:
        data = await websocket.receive_text()

        frame = decode_frame(data)
        if frame is None:
            await websocket.send_json({"error": "bad_frame"})
            continue

        h, w = frame.shape[:2]

        await websocket.send_json({
            "status": "ok",
            "width": w,
            "height": h
        })