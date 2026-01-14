"""Image capture and vision analysis utilities."""

import base64
from datetime import datetime  # For timestamp-based filenames
import os                # For directory and file handling
from pathlib import Path
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
    VISION_MODEL,
    VISION_JPEG_QUALITY,
    VISION_MAX_DIM,
    VISION_MAX_TOKENS,
)

client = OpenAI()
logger = logging.getLogger(__name__)


# ==============================
# IMAGE STORAGE CONFIGURATION
# ==============================

# Directory where captured images will be stored (repo-root `captured_images/`).
# Images are saved so they can be:
# - Sent to the LLM
# - Referenced later (DB / debugging)
_DEFAULT_IMAGE_DIR = Path(__file__).resolve().parents[2] / "captured_images"
IMAGE_DIR = Path(os.getenv("VISIONAID_IMAGE_DIR", str(_DEFAULT_IMAGE_DIR))).expanduser()

# Create the directory if it does not already exist
# This prevents runtime errors when saving images
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


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
        # Fallback: some systems fail with a forced backend but work with CAP_ANY.
        if backend is not None:
            cam = cv2.VideoCapture(index)
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

    # Generate a unique filename using the current date and time (with ms)
    # to avoid overwriting images captured within the same second.
    now = datetime.now()
    filename = (
        f"img_{now.strftime('%Y%m%d_%H%M%S')}_{int(now.microsecond / 1000):03d}.jpg"
    )

    # Full path where the image will be saved
    path = IMAGE_DIR / filename

    # Save the captured frame as a JPEG image
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        logger.warning(
            "camera_capture_write_failed path=%s cwd=%s",
            str(path),
            os.getcwd(),
        )
        return None
    logger.info("camera_capture_saved path=%s cwd=%s", str(path), os.getcwd())

    # Return the image path so it can be:
    # - Passed to the LLM
    # - Stored in the database
    if used_index is not None and used_index != CAMERA_INDEX:
        logger.info("camera_auto_probe_used index=%s", used_index)
    return str(path)


def analyze_image(image_path, prompt):
    """Analyze an image using a vision-capable model and return text."""
    if not image_path:
        return ""
    if not os.path.exists(image_path):
        logger.warning("vision_image_missing path=%s", image_path)
        return ""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("vision_imread_failed path=%s", image_path)
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
            logger.warning("vision_imencode_failed path=%s", image_path)
            return ""
        b64_data = base64.b64encode(encoded.tobytes()).decode("utf-8")
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are looking at a real photo from the user's camera. "
                        "Always respond in English. "
                        "Never claim you cannot see the image; you can. "
                        "If the image is too dark/blurred/blocked, say that and ask "
                        "the user to reposition the camera. "
                        "Answer the user's question using what you see. "
                        "Be concise and practical. If there are safety hazards "
                        "or obstacles, mention them."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or "Describe this image."},
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


def image_path_to_data_url(image_path: str) -> str:
    """Convert an image file to a resized/encoded data URL for realtime vision input."""
    if not image_path:
        return ""
    try:
        if not os.path.exists(image_path):
            return ""
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
        return f"data:image/jpeg;base64,{b64_data}"
    except Exception as exc:
        logger.warning("vision_encode_failed error=%s", exc)
        return ""
