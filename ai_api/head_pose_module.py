import cv2
import numpy as np
import math

# landmarks used (same idea as your original model)
IDX_NOSE = 1
IDX_CHIN = 152
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_LEFT_MOUTH = 61
IDX_RIGHT_MOUTH = 291

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose
    (0.0, -63.6, -12.5),      # chin
    (-43.3, 32.7, -26.0),     # left eye
    (43.3, 32.7, -26.0),      # right eye
    (-28.9, -28.9, -24.1),    # left mouth
    (28.9, -28.9, -24.1)      # right mouth
], dtype=np.float32)

# smoothing buffers
pitch_hist = []
roll_hist = []

SMOOTH_N = 5

# thresholds (adjusted like production)
PITCH_DOWN = 15      # head down
ROLL_SIDE = 20       # head tilt


def clamp_angle(a):
    # remove extreme jumps
    if a > 90:
        a = 90
    if a < -90:
        a = -90
    return a


def smooth(val, hist):
    hist.append(val)
    if len(hist) > SMOOTH_N:
        hist.pop(0)
    return sum(hist) / len(hist)


def detect_head_pose(lm, w, h):

    image_points = np.array([
        (lm[IDX_NOSE].x * w, lm[IDX_NOSE].y * h),
        (lm[IDX_CHIN].x * w, lm[IDX_CHIN].y * h),
        (lm[IDX_LEFT_EYE].x * w, lm[IDX_LEFT_EYE].y * h),
        (lm[IDX_RIGHT_EYE].x * w, lm[IDX_RIGHT_EYE].y * h),
        (lm[IDX_LEFT_MOUTH].x * w, lm[IDX_LEFT_MOUTH].y * h),
        (lm[IDX_RIGHT_MOUTH].x * w, lm[IDX_RIGHT_MOUTH].y * h),
    ], dtype=np.float32)

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    ok, rvec, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs
    )

    if not ok:
        return {"head_alert": False, "pitch": 0.0, "roll": 0.0}

    R, _ = cv2.Rodrigues(rvec)

    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw   = math.degrees(math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2)))
    roll  = math.degrees(math.atan2(R[1, 0], R[0, 0]))

    # clamp + smooth
    pitch = clamp_angle(pitch)
    roll = clamp_angle(roll)

    pitch_s = smooth(pitch, pitch_hist)
    roll_s = smooth(roll, roll_hist)

    # alert logic (stable)
    head_alert = (abs(pitch_s) > PITCH_DOWN) or (abs(roll_s) > ROLL_SIDE)

    print("\n---- HEAD DEBUG ----")
    print("RAW PITCH:", pitch, "RAW ROLL:", roll)
    print("SMOOTH PITCH:", pitch_s, "SMOOTH ROLL:", roll_s)
    print("ALERT:", head_alert)

    return {
        "pitch": float(pitch_s),
        "roll": float(roll_s),
        "head_alert": bool(head_alert)
    }