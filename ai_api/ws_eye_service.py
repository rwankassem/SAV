from eye_module import predict_eye

class WSEyeDetector:

    def __init__(self):
        self.history = []

    def predict(self, frame):
        try:
            if frame is None:
                return "unknown"

            score = predict_eye(frame)

            if score is None:
                return "unknown"

            # smoothing (important for stability)
            self.history.append(score)

            if len(self.history) > 7:
                self.history.pop(0)

            avg = sum(self.history) / len(self.history)

            return "closed" if avg < 0.35 else "open"

        except Exception as e:
            print("WS ERROR:", e)
            return "unknown"