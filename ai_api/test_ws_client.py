import cv2
import base64
import json
import time
from websocket import create_connection, WebSocketConnectionClosedException

print("STARTED")

ws = create_connection("ws://127.0.0.1:8002/ws")

cap = cv2.VideoCapture(0)

last_print = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("cam", frame)

    # encode frame
    _, buffer = cv2.imencode('.jpg', frame)
    jpg = base64.b64encode(buffer).decode('utf-8')

    try:
        ws.send(jpg)

        result = ws.recv()
        data = json.loads(result)

        now = time.time()

        if now - last_print > 0.2:
            print(
                "EYE:", data.get("eye"),
                "| DUR:", data.get("eye_duration"),
                "| YAWN:", data.get("yawn"),
                "| DROWSY:", data.get("drowsy")
            )
            last_print = now

    except WebSocketConnectionClosedException:
        print("WS CLOSED → reconnecting...")
        try:
            ws = create_connection("ws://127.0.0.1:8002/ws")
        except:
            time.sleep(1)

    except Exception as e:
        print("CLIENT ERROR:", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
ws.close()