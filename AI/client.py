import cv2
import requests
import time

ESP_URL = "http://172.20.10.3/stream"
SERVER_URL = "http://127.0.0.1:8002/predict"

cap = cv2.VideoCapture(ESP_URL)

# =========================
# CONTROL FPS
# =========================
SEND_EVERY = 0.2   
last_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("CLIENT VIEW", frame)

    now = time.time()

    if now - last_time < SEND_EVERY:
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    last_time = now

    _, img = cv2.imencode(".jpg", frame)

    try:
        requests.post(
            SERVER_URL,
            files={"file": img.tobytes()},
            timeout=2
        )
    except:
        pass

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()