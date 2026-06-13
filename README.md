# AI & ESP32-CAM Drowsiness Detection

A real-time AI-powered driver drowsiness detection system that combines an ESP32-CAM, Computer Vision, and Deep Learning to monitor driver fatigue. The system detects eye closure and yawning signs and can provide real-time drowsiness warnings.

## Project Structure

```
SAV/
│
├── pio-esp32cam/
│   ├── src/main.cpp          # ESP32-CAM firmware
│   ├── src/app_httpd.cpp     # Alternative HTTP server implementation
│   └── camera_pins.h         # AI Thinker pin configuration
│
├── AI/
│   ├── esp_test.py           # Real-time AI inference demo
│   └── training notebooks    # Model training and experiments
│
├── ai_api/
│   ├── AI processing modules
│   ├── Eye state detection
│   └── Yawn detection
│
└── model/
    ├── eyes_int8.tflite      # Quantized eye state model
    └── Training artifacts
```

---

# ESP32-CAM Features

The ESP32-CAM firmware is responsible for:

* Capturing real-time camera frames.
* Streaming video using MJPEG through the `/stream` endpoint.
* Recording camera frames to a MicroSD card.
* Controlling recording through HTTP commands.

Endpoints:

* Stream: `http://<esp_ip>/stream`
* Start recording: `http://<esp_ip>/start`
* Stop recording: `http://<esp_ip>/stop`

Recorded data is stored in a custom binary format containing JPEG frames.

---

# AI Pipeline

The AI system processes camera frames to detect driver drowsiness using a combination of Computer Vision and Deep Learning techniques.

## 1. Face Landmark Detection

MediaPipe FaceMesh is used to detect facial landmarks with high precision.

It extracts important regions such as:

* Eye landmarks for eye state analysis.
* Mouth landmarks for yawning detection.

---

## 2. Eye State Detection

The eye regions are extracted and passed to a TensorFlow Lite quantized model (`eyes_int8.tflite`) that classifies the eye state as:

* Open
* Closed

The model was trained using eye state datasets from Kaggle:

* MRL Eye Open/Close Dataset:
  https://www.kaggle.com/datasets/rameezakther/mrl-eye-open-or-close-dataset

* Eyes Open or Closed Dataset:
  https://www.kaggle.com/datasets/akshitmadan/eyes-open-or-closed

---

## 3. Yawn Detection

Yawning is detected using mouth landmarks and the Mouth Aspect Ratio (MAR) technique.

The system monitors mouth opening over consecutive frames to distinguish normal speech from a real yawn.

---

## 4. Drowsiness Decision

The final drowsiness status is determined by combining multiple indicators:

* Eye closure duration.
* Eye open probability from the AI model.
* Yawning frequency and duration.

This multi-factor approach improves reliability and reduces false alarms.

---

# Model

The project uses a lightweight TensorFlow Lite model:

* `eyes_int8.tflite`: Quantized eye state classification model optimized for fast inference.

The original training experiments and evaluation notebooks are available inside the AI directory.

---

# Setup

## ESP32-CAM

Requirements:

* ESP32-CAM (AI Thinker)
* MicroSD card
* USB-to-Serial adapter
* PlatformIO

Steps:

1. Open the firmware project in PlatformIO.
2. Update Wi-Fi credentials in `main.cpp`.
3. Update the upload COM port if required.
4. Build and upload the firmware.

---

## Python Environment

Install the required dependencies:

```bash
cd ai_api
pip install -r requirements.txt
```

For the AI demo:

```bash
pip install opencv-python mediapipe numpy requests
```


---

# Future Improvements

* Add a complete dashboard for monitoring and alerts.
* Improve drowsiness scoring using temporal smoothing.
* Add support for analyzing recorded ESP32-CAM data.


---

# License

This project currently does not include a license file. Add a LICENSE file before distributing or publishing the repository.
