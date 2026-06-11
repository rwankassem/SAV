import cv2
import requests
import time

ESP_URL = "http://172.20.10.3/stream"
SERVER_URL = "http://165.232.78.206:8000/esp"

cap = cv2.VideoCapture(ESP_URL)

SEND_EVERY = 0.75
last_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    now = time.time()

   
    frame = cv2.resize(frame, (320, 240))

    # =========================
    # SEND FRAME
    # =========================
    if now - last_time >= SEND_EVERY:
        last_time = now

        _, img = cv2.imencode(".jpg", frame)

        try:
            response = requests.post(
                SERVER_URL,
                files={"file": ("frame.jpg", img.tobytes(), "image/jpeg")},
                timeout=2
            )

            data = response.json()
            print(data)

            text = f"{data.get('state')} | {data.get('score')}"

            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except:
            pass

    # =========================
    # DISPLAY (AFTER ALL CHANGES)
    # =========================
    cv2.imshow("CLIENT VIEW", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()