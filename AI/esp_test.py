import cv2
import time
import math
import numpy as np
from collections import deque
import mediapipe as mp

def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def calculate_ear(landmarks, eye_indices):
    top = landmarks[eye_indices[0]]
    bottom = landmarks[eye_indices[1]]
    left = landmarks[eye_indices[2]]
    right = landmarks[eye_indices[3]]
    v1 = landmarks[eye_indices[4]]
    v2 = landmarks[eye_indices[5]]

    vertical_1 = euclidean_distance(top, bottom)
    vertical_2 = euclidean_distance(v1, v2)
    horizontal = euclidean_distance(left, right)
    return (vertical_1 + vertical_2) / (2.0 * horizontal) if horizontal != 0 else 0.0

def calculate_mar(landmarks):
    upper = landmarks[13]
    lower = landmarks[14]
    left = landmarks[78]
    right = landmarks[308]
    vertical = euclidean_distance(upper, lower)
    horizontal = euclidean_distance(left, right)
    return vertical / horizontal if horizontal != 0 else 0.0

IDX_NOSE_TIP = 1
IDX_CHIN = 152
IDX_LEFT_EYE_OUTER = 33
IDX_RIGHT_EYE_OUTER = 263
IDX_LEFT_MOUTH = 61
IDX_RIGHT_MOUTH = 291

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float32)

def wrap180(a):
    return float((a + 180.0) % 360.0 - 180.0)

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(R[2,1], R[2,2])
        yaw = math.atan2(-R[2,0], sy)
        roll = math.atan2(R[1,0], R[0,0])
    else:
        pitch = math.atan2(-R[1,2], R[1,1])
        yaw = math.atan2(-R[2,0], sy)
        roll = 0.0
    pitch, yaw, roll = np.degrees([pitch, yaw, roll])
    return wrap180(pitch), wrap180(yaw), wrap180(roll)

CAM_INDEX = 0
SMOOTH_N = 15
PITCH_DELTA_ON = 10.0
PITCH_DELTA_OFF = 7.0
PITCH_ALERT_TIME = 2.5
ROLL_ABS_ON = 20.0
ROLL_ALERT_TIME = 1.5
EAR_THRESHOLD = 0.21
EYE_CLOSED_THRESHOLD = 0.18
MAR_THRESHOLD = 0.6
YAWN_LIMIT = 3

hist_pitch = deque(maxlen=SMOOTH_N)
hist_roll = deque(maxlen=SMOOTH_N)
pitch0 = None
calib_pitch = []
pitch_frames = 0
roll_frames = 0
pitch_alert_on = False
roll_alert_on = False
pitch_hold_frames = 0
blink_count = 0
yawn_count = 0
eye_closed_frames = 0
yawn_active = False

LEFT_EYE = [159, 145, 33, 133, 158, 153]
RIGHT_EYE = [386, 374, 362, 263, 385, 380]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# cap = cv2.VideoCapture("http://10.204.189.15:80/stream")
cap = cv2.VideoCapture("https://velda-undefinable-remington.ngrok-free.dev/stream")
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()
# cap = cv2.VideoCapture(CAM_INDEX)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
if fps <= 0:
    fps = 30.0
calib_frames_needed = int(2.0 * fps)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    alert_now = False

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        # Head pose
        image_points = np.array([
            (lm[IDX_NOSE_TIP].x * w, lm[IDX_NOSE_TIP].y * h),
            (lm[IDX_CHIN].x * w, lm[IDX_CHIN].y * h),
            (lm[IDX_LEFT_EYE_OUTER].x * w, lm[IDX_LEFT_EYE_OUTER].y * h),
            (lm[IDX_RIGHT_EYE_OUTER].x * w, lm[IDX_RIGHT_EYE_OUTER].y * h),
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
        ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs)
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = rotationMatrixToEulerAngles(R)
            hist_pitch.append(pitch)
            hist_roll.append(roll)
            pitch_s = float(np.mean(hist_pitch))
            roll_s = float(np.mean(hist_roll))

            if pitch0 is None:
                calib_pitch.append(pitch_s)
                if len(calib_pitch) >= calib_frames_needed:
                    pitch0 = float(np.mean(calib_pitch))
                    calib_pitch.clear()
                cv2.putText(frame, f"Calibrating... ({max(0, calib_frames_needed-len(calib_pitch))})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                pitch_delta = pitch_s - pitch0
                is_down_on = (pitch_delta <= -PITCH_DELTA_ON)
                is_down_off = (pitch_delta >= -PITCH_DELTA_OFF)

                if is_down_on:
                    pitch_frames += 1
                else:
                    if not pitch_alert_on:
                        pitch_frames = 0
                pitch_time = pitch_frames / fps
                if (not pitch_alert_on) and (pitch_time >= PITCH_ALERT_TIME):
                    pitch_alert_on = True
                    pitch_hold_frames = int(0.7 * fps)
                if pitch_alert_on and pitch_hold_frames > 0:
                    pitch_hold_frames -= 1
                if pitch_alert_on and pitch_hold_frames <= 0 and is_down_off:
                    pitch_alert_on = False
                    pitch_frames = 0
                if abs(roll_s) >= ROLL_ABS_ON:
                    roll_frames += 1
                else:
                    roll_frames = 0
                    roll_alert_on = False
                if (roll_frames / fps) >= ROLL_ALERT_TIME:
                    roll_alert_on = True
                if pitch_alert_on or roll_alert_on:
                    alert_now = True
                    cv2.putText(frame, "HEAD ALERT", (int(w * 0.1), int(h * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.putText(frame, f"Pitch: {pitch_s:.1f}  Roll: {roll_s:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Blink and yawn
        left_ear = calculate_ear(lm, LEFT_EYE)
        right_ear = calculate_ear(lm, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(lm)

        if ear < EAR_THRESHOLD:
            blink_count += 1
        if ear < EYE_CLOSED_THRESHOLD:
            eye_closed_frames += 1
        else:
            eye_closed_frames = 0
        if mar > MAR_THRESHOLD and not yawn_active:
            yawn_count += 1
            yawn_active = True
        if mar < MAR_THRESHOLD:
            yawn_active = False

        cv2.putText(frame, f'EAR: {ear:.2f}  MAR: {mar:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Blinks: {blink_count}  Yawns: {yawn_count}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if eye_closed_frames > 20:
            cv2.putText(frame, "EYES CLOSED!", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if yawn_count >= YAWN_LIMIT:
            cv2.putText(frame, "DROWSY DETECTED!", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Drowsiness + Head Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()