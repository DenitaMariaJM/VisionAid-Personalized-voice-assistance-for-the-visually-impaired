import os
import cv2
import time
import base64
import sqlite3
import pyttsx3
import speech_recognition as sr
from datetime import datetime
from openai import OpenAI

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
DB_NAME = "assistant.db"
IMAGE_SAVE_DIR = "captured_images"
WAKE_WORD = "alexa"
CAMERA_INDEX = 0

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# SETUP
# -----------------------------
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Initialize TTS engine
tts = pyttsx3.init()
tts.setProperty("rate", 170)

def speak(text):
    tts.say(text)
    tts.runAndWait()

# -----------------------------
# DATABASE SETUP
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            time TEXT,
            image_path TEXT,
            description TEXT,
            user_interaction TEXT,
            tags TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_record(image_path, description, user_interaction, tags):
    now = datetime.now()
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        INSERT INTO interactions (date, time, image_path, description, user_interaction, tags)
        VALUES (?, ?, ?, ?, ?, ?)
    """,
    (now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
     image_path, description, user_interaction, tags))
    conn.commit()
    conn.close()

# -----------------------------
# IMAGE CAPTURE + COMPRESSION
# -----------------------------
def capture_and_compress():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise Exception("Camera not found")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise Exception("Image capture failed")

    max_dim = 720
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    img_name = f"{int(time.time())}.jpg"
    img_path = os.path.join(IMAGE_SAVE_DIR, img_name)

    cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    return img_path

def img_to_data_uri(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# -----------------------------
# SPEECH RECOGNITION
# -----------------------------
def listen_for_wake_word():
    r = sr.Recognizer()
    mic = sr.Microphone()

    print("Listening for wake word 'Alexa'...")

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.4)

        while True:
            audio = r.listen(source, phrase_time_limit=4)

            try:
                text = r.recognize_google(audio).lower()
                print("Heard:", text)
                if text.startswith(WAKE_WORD):
                    remaining = text[len(WAKE_WORD):].strip()
                    if remaining:
                        return remaining
                    else:
                        speak("Yes, what can I help you with?")
                        return listen_for_command()
            except:
                pass

def listen_for_command():
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

# -----------------------------
# EXTRACT TAGS FROM GPT RESPONSE
# -----------------------------
def extract_tags(result_text):
    """
    Find the line starting with 'Tags:' and return everything after it.
    """
    for line in result_text.split("\n"):
        if line.lower().startswith("tags:"):
            return line.split(":", 1)[1].strip()
    return ""

# -----------------------------
# GPT MULTIMODAL CALL
# -----------------------------
def analyze_image_and_query(image_uri, query):
    prompt = (
    "You are part of a personalized voice-assistance system for a blind user. "
    "Your goal is to describe the scene briefly using only essential information and also generate helpful context tags "
    "that will be stored in our system. These tags will help the assistant remember "
    "the user's surroundings, habits, and activities over time. "
    "Keep sentences short and clear. Mention important objects, obstacles, "
    "people, or safety-related details. Avoid storytelling or unnecessary details.\n\n"

    f"User asked: '{query}'. "
    "Answer in simple, calm, and easy-to-understand language suitable for a blind person.\n\n"

    "After your description, generate a set of short tags (not restricted to one) "
    "that summarize important aspects of the scene — such as environment, objects, "
    "activities, risks, emotions, or anything meaningful for personalized assistance. "
    "Choose tags that will be helpful for long-term context understanding and memory.\n\n"

    "Return the answer in this format:\n"
    "Description: <your description>\n"
    "Tags: <tag1>, <tag2>, <tag3>, ..."
)


    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_uri}
                ]
            }
        ]
    )

    try:
        return response.output_text
    except:
        parts = []
        for item in response.output:
            if "content" in item:
                for c in item["content"]:
                    if c.get("type") in ("output_text", "summary_text"):
                        parts.append(c.get("text", ""))
        return "\n".join(parts)

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    init_db()

    while True:
        try:
            query = listen_for_wake_word()
            print("User:", query)
            if not query.strip():
                speak("I didn't catch that. Please ask your question after saying Alexa.")
                continue
            speak("Capturing image...")
            image_path = capture_and_compress()
            image_uri = img_to_data_uri(image_path)

            speak("Analyzing, please wait...")
            result = analyze_image_and_query(image_uri, query)
            print("GPT Response:\n", result)

            # Extract all tags
            tags = extract_tags(result)

            # Store everything
            insert_record(image_path, result, query, tags)

            # Extract description only
            if "Tags:" in result:
                description_only = result.split("Tags:")[0]
            else:
                description_only = result

            description_only = description_only.replace("Description:", "").strip()

            speak(description_only)


        except KeyboardInterrupt:
            print("Exiting.")
            break
        except Exception as e:
            print("Error:", e)
            speak("I faced an error, but I am still running.")

if __name__ == "__main__":
    main()
