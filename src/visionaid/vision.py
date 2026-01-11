"""Image capture and vision analysis utilities."""

import base64
from datetime import datetime  # For timestamp-based filenames
import os                # For directory and file handling
import logging
import time

import cv2               # OpenCV library for camera access
from openai import OpenAI

from .config import (
    CAMERA_AUTO_PROBE,
    CAMERA_BACKEND,
    CAMERA_FRAME_HEIGHT,
    CAMERA_FRAME_WIDTH,
    CAMERA_INDEX,
    CAMERA_PROBE_MAX,
    CAMERA_WARMUP_FRAMES,
    VISION_JPEG_QUALITY,
    VISION_MAX_DIM,
    VISION_MAX_TOKENS,
)

client = OpenAI()
logger = logging.getLogger(__name__)


# ==============================
# IMAGE STORAGE CONFIGURATION
# ==============================

# Directory where captured images will be stored
# Images are saved so they can be:
# - Sent to the LLM
# - Referenced later (DB / debugging)
IMAGE_DIR = "captured_images"

# Create the directory if it does not already exist
# This prevents runtime errors when saving images
os.makedirs(IMAGE_DIR, exist_ok=True)


# ==============================
# IMAGE CAPTURE FUNCTION
# ==============================

def try_capture_frame(index):
    backend = None
    if CAMERA_BACKEND == "v4l2":
        backend = cv2.CAP_V4L2
    elif CAMERA_BACKEND == "dshow":
        backend = cv2.CAP_DSHOW
    elif CAMERA_BACKEND == "avfoundation":
        backend = cv2.CAP_AVFOUNDATION

    cam = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not cam.isOpened():
        cam.release()
        return None, None

    if CAMERA_FRAME_WIDTH > 0:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    if CAMERA_FRAME_HEIGHT > 0:
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)

    for _ in range(CAMERA_WARMUP_FRAMES):
        cam.read()
        time.sleep(0.02)

    ret, frame = cam.read()
    cam.release()
    if not ret:
        return None, None
    return index, frame


def capture_image():
    """
    Captures a single image from the default camera (camera index 0).

    Returns:
        str: File path of the saved image if capture succeeds
        None: If the camera fails to capture an image
    """

    # Open the default camera (0 = built-in webcam / USB camera)
    used_index, frame = try_capture_frame(CAMERA_INDEX)
    if frame is None and CAMERA_AUTO_PROBE:
        for idx in range(CAMERA_PROBE_MAX):
            if idx == CAMERA_INDEX:
                continue
            used_index, frame = try_capture_frame(idx)
            if frame is not None:
                break

    if frame is None:
        logger.warning(
            "camera_capture_failed index=%s auto_probe=%s",
            CAMERA_INDEX,
            CAMERA_AUTO_PROBE,
        )
        return None

    # Generate a unique filename using the current date and time
    # This avoids overwriting old images
    filename = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    # Full path where the image will be saved
    path = os.path.join(IMAGE_DIR, filename)

    # Save the captured frame as a JPEG image
    cv2.imwrite(path, frame)

    # Return the image path so it can be:
    # - Passed to the LLM
    # - Stored in the database
    if used_index is not None and used_index != CAMERA_INDEX:
        logger.info("camera_auto_probe_used index=%s", used_index)
    return path


def analyze_image(image_path, prompt):
    """
    Sends an image to a vision-capable model for analysis.
    """
    if not image_path:
        return ""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return ""
        height, width = image.shape[:2]
        max_dim = max(height, width)
        if max_dim > VISION_MAX_DIM:
            scale = VISION_MAX_DIM / float(max_dim)
            new_size = (int(width * scale), int(height * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), VISION_JPEG_QUALITY],
        )
        if not success:
            return ""
        b64_data = base64.b64encode(encoded.tobytes()).decode("utf-8")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze images for accessibility. Always respond "
                        "in English. Focus on obstacles and navigation-"
                        "relevant details."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_data}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=VISION_MAX_TOKENS,
        )
        message = response.choices[0].message
        return (message.content or "").strip()
    except Exception as exc:
        logger.warning("vision_analysis_failed error=%s", exc)
        return ""
