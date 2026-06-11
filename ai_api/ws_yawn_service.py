import numpy as np

class WSYawnDetector:
    def __init__(self):
        self.yawn_count = 0
        self.yawn_active = False
        self.MAR_THRESHOLD = 0.6

    def calculate_mar(self, lm):
        # Mouth points
        mouth_points = [61, 291, 39, 181, 0, 17, 269, 405]
        
        # Calculate distances
        A = np.linalg.norm(np.array([lm[mouth_points[2]].x, lm[mouth_points[2]].y]) - np.array([lm[mouth_points[3]].x, lm[mouth_points[3]].y]))
        B = np.linalg.norm(np.array([lm[mouth_points[4]].x, lm[mouth_points[4]].y]) - np.array([lm[mouth_points[5]].x, lm[mouth_points[5]].y]))
        C = np.linalg.norm(np.array([lm[mouth_points[0]].x, lm[mouth_points[0]].y]) - np.array([lm[mouth_points[1]].x, lm[mouth_points[1]].y]))
        
        mar = (A + B) / (2.0 * C)
        return mar

    def predict(self, lm):
        mar = self.calculate_mar(lm)
        yawn = mar > self.MAR_THRESHOLD
        
        if yawn and not self.yawn_active:
            self.yawn_count += 1
            self.yawn_active = True
        if not yawn:
            self.yawn_active = False
        
        return {"yawn": yawn, "count": self.yawn_count, "mar": mar}