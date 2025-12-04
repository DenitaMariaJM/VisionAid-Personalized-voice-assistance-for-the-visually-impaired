import cv2
import time
import base64
import os
import numpy as np
import pyttsx3
from openai import OpenAI
import threading

MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def describe_scene(frame):
    # Encode frame to JPEG
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    b64_img = base64.b64encode(buffer).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"

    response = client.responses.create(
        model=MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": """
                Describe this scene for a blind user, focusing on guidance and spatial awareness.
                Tell the user:
                - Where people, objects, and obstacles are relative to them (left, right, front)
                - How they can navigate safely
                - Important interactions in the scene
                Keep it short, clear, and suitable for spoken instructions.
                """},
                {"type": "input_image", "image_url": data_url}
            ]
        }]
    )
    return response.output_text.strip()

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def describe_and_speak(frame):
    """Thread target for non-blocking GPT + TTS"""
    try:
        desc = describe_scene(frame)
        print("\n=== Scene Update ===\n", desc, "\n====================\n")
        speak_text(desc)
        return desc
    except Exception as e:
        print("[ERROR in GPT/TTS thread]:", e)
        return None

def live_scene_description(interval=2, change_threshold=15, resize_width=320, resize_height=240):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    print("🎥 Live scene description started. Press 'q' to quit.")

    prev_gray = None
    last_desc = ""
    last_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        small_frame = cv2.resize(frame, (resize_width, resize_height))

        # Convert to grayscale for scene-change detection
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        scene_changed = False
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            diff_score = np.mean(diff)
            if diff_score > change_threshold:
                scene_changed = True

        # If scene changed and interval passed, call GPT in a separate thread
        if scene_changed and (time.time() - last_time > interval):
            thread = threading.Thread(target=lambda: describe_and_speak(small_frame))
            thread.start()
            last_time = time.time()

        prev_gray = gray

        # Show live preview (optional for testing)
        cv2.imshow("Live Feed - Press 'q' to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Optimized: resize frame, detect scene change, interval 2s
    live_scene_description(interval=2, change_threshold=15, resize_width=320, resize_height=240)
