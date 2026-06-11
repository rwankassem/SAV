#!/usr/bin/env python3
"""
esp_runner.py
Robust reader for ESP32-CAM MJPEG stream URL.
- tries OpenCV VideoCapture (FFMPEG) first
- falls back to MJPEG parsing via requests
- reconnects automatically on errors
"""

import cv2
import time
import argparse
import requests
import numpy as np
import sys


def open_with_opencv(url, prefer_ffmpeg=True, timeout=5.0):
    backends = []
    try:
        if prefer_ffmpeg and hasattr(cv2, 'CAP_FFMPEG'):
            backends.append(cv2.CAP_FFMPEG)
    except Exception:
        pass
    backends.append(cv2.CAP_ANY)

    for b in backends:
        try:
            cap = cv2.VideoCapture(url, b)
        except Exception:
            try:
                cap = cv2.VideoCapture(url)
            except Exception:
                cap = None
        if cap is None:
            continue
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cap.isOpened():
                return cap
            time.sleep(0.1)
        try:
            cap.release()
        except Exception:
            pass
    return None


def mjpeg_stream_generator(url, timeout=10):
    sess = requests.Session()
    resp = sess.get(url, stream=True, timeout=timeout)
    if resp.status_code != 200:
        resp.close()
        raise IOError(f"HTTP {resp.status_code}")
    bytes_buf = b''
    for chunk in resp.iter_content(chunk_size=1024):
        if not chunk:
            continue
        bytes_buf += chunk
        start = bytes_buf.find(b'\xff\xd8')
        end = bytes_buf.find(b'\xff\xd9')
        if start != -1 and end != -1 and end > start:
            jpg = bytes_buf[start:end+2]
            bytes_buf = bytes_buf[end+2:]
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield img
    resp.close()


class StreamReader:
    def __init__(self, url):
        self.url = url
        self.cap = None
        self.mjpeg_gen = None

    def connect(self):
        self.close()
        self.cap = open_with_opencv(self.url)
        if self.cap:
            print("Opened stream via OpenCV VideoCapture")
            return True
        try:
            self.mjpeg_gen = mjpeg_stream_generator(self.url)
            print("Using MJPEG fallback (requests)")
            return True
        except Exception as e:
            print("Failed to open stream:", e)
            self.mjpeg_gen = None
            return False

    def read(self):
        if self.cap:
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if ok and frame is not None:
                return frame
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.mjpeg_gen:
            try:
                frame = next(self.mjpeg_gen)
                return frame
            except Exception:
                self.mjpeg_gen = None
        return None

    def close(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.mjpeg_gen = None


def try_import_mediapipe():
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        return mp, mp_face_mesh
    except Exception:
        return None, None


def main():
    parser = argparse.ArgumentParser(description="ESP stream runner")
    parser.add_argument('--url', '-u', default='http://172.20.10.3/stream', help='ESP MJPEG stream URL')
    args = parser.parse_args()

    reader = StreamReader(args.url)
    if not reader.connect():
        print("Initial connect failed — will retry in loop")

    mp, mp_face_mesh = try_import_mediapipe()
    face_mesh = None
    if mp:
        try:
            face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print("MediaPipe FaceMesh ready")
        except Exception as e:
            print("FaceMesh init failed:", e)
            face_mesh = None

    last_ok = 0
    reconnect_delay = 2.0
    while True:
        frame = reader.read()
        if frame is None:
            if time.time() - last_ok > reconnect_delay:
                print("No frame — attempting reconnect...")
                if reader.connect():
                    last_ok = time.time()
                    print("Reconnected")
                else:
                    print("Reconnect failed; sleeping")
                    time.sleep(reconnect_delay)
            else:
                time.sleep(0.1)
            continue

        last_ok = time.time()
        if face_mesh:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if res and res.multi_face_landmarks:
                    cv2.putText(frame, "Face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            except Exception as e:
                print("FaceMesh processing error:", e)

        cv2.imshow("ESP32-CAM", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break

    reader.close()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == '__main__':
    main()
