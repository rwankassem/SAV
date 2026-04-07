from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
import numpy as np

from eye_module import run_eye_inference

app = FastAPI(title="Driver Drowsiness AI API")


class PredictionResponse(BaseModel):
    eye_state: str
    yawn_detected: bool
    head_alert: bool
    driver_status: str
    confidence: Optional[float] = None


@app.get("/")
def home():
    return {"message": "AI API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
        frame = np.array(image)  # RGB image
    except Exception:
        return {
            "eye_state": "unknown",
            "yawn_detected": False,
            "head_alert": False,
            "driver_status": "invalid_frame",
            "confidence": None
        }

    eye_result = run_eye_inference(frame)

    eye_state = eye_result["eye_state"]
    confidence = eye_result["confidence"]

    # placeholders
    yawn_detected = False
    head_alert = False

    driver_status = "drowsy" if (
        eye_state == "closed" or yawn_detected or head_alert
    ) else "normal"

    return {
        "eye_state": eye_state,
        "yawn_detected": yawn_detected,
        "head_alert": head_alert,
        "driver_status": driver_status,
        "confidence": confidence
    }