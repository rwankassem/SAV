import math
import time

MAR_THRESHOLD = 0.6

YAWN_BUFFER_SIZE = 3
YAWN_RESET_TIME = 5  # seconds without yawn → reset

class WSYawnDetector:

    def __init__(self):
        self.count = 0
        self.last_state = False
        self.last_yawn_time = time.time()

    def _dist(self, a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

    def predict(self, lm):

        try:
            upper = lm[13]
            lower = lm[14]
            left = lm[61]
            right = lm[291]

            vertical = self._dist(upper, lower)
            horizontal = self._dist(left, right)

            if horizontal == 0:
                return {"yawn": False, "mar": 0.0, "count": self.count}

            mar = vertical / horizontal
            is_yawn = mar > MAR_THRESHOLD

            now = time.time()

            # reset if long time no yawning
            if now - self.last_yawn_time > YAWN_RESET_TIME:
                self.count = 0

            # rising edge only
            if is_yawn and not self.last_state:
                self.count += 1
                self.last_yawn_time = now

            self.last_state = is_yawn

            return {
                "yawn": is_yawn,
                "mar": float(mar),
                "count": self.count
            }

        except:
            return {"yawn": False, "mar": 0.0, "count": self.count}