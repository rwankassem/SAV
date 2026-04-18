import math
import numpy as np

MAR_THRESHOLD = 0.55

# منع التذبذب
YAWN_FRAMES_THRESHOLD = 3

yawn_buffer = []


def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def calculate_mar(lm):
    try:
        upper = lm[13]
        lower = lm[14]
        left = lm[78]
        right = lm[308]

        vertical = euclidean(upper, lower)
        horizontal = euclidean(left, right)

        if horizontal == 0:
            return 0.0

        return vertical / horizontal

    except:
        return 0.0


def detect_yawn(lm):
    global yawn_buffer

    mar = calculate_mar(lm)

    is_yawn = mar > MAR_THRESHOLD

    # smoothing buffer
    yawn_buffer.append(is_yawn)
    if len(yawn_buffer) > YAWN_FRAMES_THRESHOLD:
        yawn_buffer.pop(0)

    stable_yawn = sum(yawn_buffer) >= 2 

    print("\n---- YAWN DEBUG ----")
    print("MAR:", mar)
    print("THRESH:", MAR_THRESHOLD)
    print("BUFFER:", yawn_buffer)
    print("STABLE YAWN:", stable_yawn)

    return {
        "mar": float(mar),
        "yawn_detected": bool(stable_yawn)
    }