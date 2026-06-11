import cv2
import mediapipe as mp
from eye_module import predict_open_prob, get_best_eye

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def predict_eye(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    
    if not res.multi_face_landmarks:
        return 0.5  # neutral
    
    lm = res.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]
    eye_img, box = get_best_eye(frame, lm, w, h)
    
    if eye_img is None:
        return 0.5
    
    return predict_open_prob(eye_img)