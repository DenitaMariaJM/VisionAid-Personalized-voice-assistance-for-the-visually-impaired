import cv2
import os
from datetime import datetime

IMAGE_DIR = "captured_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def capture_image():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    if not ret:
        return None

    path = f"{IMAGE_DIR}/img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(path, frame)
    return path
