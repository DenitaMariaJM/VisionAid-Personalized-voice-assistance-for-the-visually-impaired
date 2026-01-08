import cv2               # OpenCV library for camera access
import os                # For directory and file handling
from datetime import datetime  # For timestamp-based filenames


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

def capture_image():
    """
    Captures a single image from the default camera (camera index 0).

    Returns:
        str: File path of the saved image if capture succeeds
        None: If the camera fails to capture an image
    """

    # Open the default camera (0 = built-in webcam / USB camera)
    cam = cv2.VideoCapture(0)

    # Read a single frame from the camera
    ret, frame = cam.read()

    # Release the camera immediately to free resources
    cam.release()

    # If the frame was not captured successfully
    # ret == False means camera access failed
    if not ret:
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
    return path
