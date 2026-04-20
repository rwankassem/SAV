import math

MAR_THRESHOLD = 0.55
YAWN_FRAMES = 3

buffer = []

def euclidean(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def detect_yawn(lm):
    global buffer

    try:
        upper = lm[13]
        lower = lm[14]
        left = lm[78]
        right = lm[308]

        vertical = euclidean(upper, lower)
        horizontal = euclidean(left, right)

        if horizontal == 0:
            return {"mar": 0.0, "yawn": False}

        mar = vertical / horizontal
        is_yawn = mar > MAR_THRESHOLD

        buffer.append(is_yawn)
        if len(buffer) > YAWN_FRAMES:
            buffer.pop(0)

        stable = sum(buffer) >= 2

        return {
            "mar": float(mar),
            "yawn": stable
        }

    except:
        return {"mar": 0.0, "yawn": False}