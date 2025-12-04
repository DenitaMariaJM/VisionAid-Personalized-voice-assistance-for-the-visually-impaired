import cv2
import time
import base64
import os
import pyttsx3
from openai import OpenAI
import sqlite3
import datetime
import spacy
import re


# Load local NLP model for tag extraction
nlp = spacy.load("en_core_web_sm")

def init_db():
    """Initialize local SQLite memory database."""
    conn = sqlite3.connect("assistant_memory.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    description TEXT,
                    tags TEXT,
                    image_path TEXT,
                    image_blob BLOB
                )''')
    conn.commit()
    conn.close()
    print("✅ Database initialized: assistant_memory.db")

def extract_tags(description):
    """Extract key nouns and verbs locally using spaCy NLP."""
    doc = nlp(description)
    tags = set()

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "VERB"]:
            tags.add(token.lemma_.lower())

    return ", ".join(sorted(tags))

def store_memory(description, tags, frame):
    """Save scene description, tags, and image into local SQLite memory."""
    conn = sqlite3.connect("assistant_memory.db")
    c = conn.cursor()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("memories", exist_ok=True)

    # Save image to file
    image_path = f"memories/frame_{timestamp}.jpg"
    cv2.imwrite(image_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

    # Convert image to binary blob for DB
    _, buffer = cv2.imencode(".jpg", frame)
    image_blob = buffer.tobytes()

    # Insert into DB
    c.execute('''INSERT INTO memory (timestamp, description, tags, image_path, image_blob)
                 VALUES (?, ?, ?, ?, ?)''',
              (timestamp, description, tags, image_path, image_blob))
    conn.commit()
    conn.close()
    print(f"🧠 Memory stored: {image_path}")


MODEL = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def describe_scene(frame):
    # Encode frame to JPEG
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
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
    return response.output_text
def clean_text_for_speech(text):
    """Remove Markdown and symbols so voice sounds natural."""
    # Remove markdown symbols like **, *, _, #
    text = re.sub(r'[*_#>`~]', '', text)
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def speak_text(text):
    engine = pyttsx3.init()
    clean_text = clean_text_for_speech(text)
    engine.say(clean_text)
    engine.runAndWait()

def live_scene_description(interval=5):
    init_db()  # <-- initialize the database at start

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    last_time = 0

    print("🎥 Live scene description started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_time > interval:  # sample frame every N seconds
            try:
                desc = describe_scene(frame)
                tags = extract_tags(desc)  # <-- new
                store_memory(desc, tags, frame)  # <-- new

                print("\n=== Scene Update ===\n", desc, "\n====================\n")
                speak_text(desc)
            except Exception as e:
                print("[ERROR]", str(e))
            last_time = now

        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_scene_description(interval=5)  # update every 5 seconds


