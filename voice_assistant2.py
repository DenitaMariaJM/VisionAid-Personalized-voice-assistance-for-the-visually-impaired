import os
import cv2
import time
import base64
import sqlite3
import websocket
import json
import pyaudio
import speech_recognition as sr
from datetime import datetime
from openai import OpenAI
import threading
import asyncio
from queue import Queue
from tts_server_internal import start_tts_server
import io
import wave

audio_lock = threading.Lock()


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
is_speaking = False

# -----------------------------
# SETUP
# -----------------------------
class TTSClient:
    def __init__(self, url):
        print("DEBUG: Initializing persistent TTS client")
        self.url = url
        self.ws = websocket.create_connection(self.url)
        self.ws.settimeout(30)

        self.audio = pyaudio.PyAudio()

    def speak(self, text):
        if not text.strip():
            return

        print("TTS:", text)

        try:
            self.ws.send(json.dumps({
                "text": text,
                "voice": "neutral",
                "rate": 1.0
            }))

            audio_data = b""

            while True:
                data = self.ws.recv()

                if isinstance(data, str) and data == "END":
                    break

                if isinstance(data, bytes):
                    audio_data += data

            if not audio_data:
                print("⚠️ No audio received")
                return

            # ✅ Thread-safe WAV playback
            with audio_lock:
                with wave.open(io.BytesIO(audio_data), "rb") as wf:
                    stream = self.audio.open(
                        format=self.audio.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True
                    )

                    stream.write(wf.readframes(wf.getnframes()))
                    stream.stop_stream()
                    stream.close()

        except Exception as e:
            print("TTS error:", e)
            try:
                self.ws.close()
            except:
                pass

            print("🔁 Reconnecting TTS WebSocket...")
            self.ws = websocket.create_connection(self.url)
            self.ws.settimeout(30)

    def close(self):
        try:
            self.ws.close()
        except:
            pass

        try:
            self.audio.terminate()
        except:
            pass


tts_queue = Queue()
tts_client = None

def tts_worker():
    global tts_client, is_speaking

    while True:
        text = tts_queue.get()

        if text is None:
            break

        try:
            is_speaking = True
            tts_client.speak(text)

        except Exception as e:
            print("🔴 TTS worker error:", e)

            # Attempt clean recovery
            try:
                tts_client.close()
            except:
                pass

            time.sleep(0.5)

            try:
                print("🔁 Reinitializing TTS client...")
                tts_client = TTSClient(TTS_WS_URL)
                tts_client.speak(text)
            except Exception as e:
                print("❌ TTS recovery failed:", e)

        finally:
            is_speaking = False



os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

TTS_WS_URL = "ws://localhost:8765"  # your TTS server



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
def init_daily_summary_db():
    conn = sqlite3.connect("daily_summary.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            summary TEXT,
            key_tags TEXT
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
def generate_daily_summary_for_date(target_date):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT description, tags FROM interactions WHERE date=?",
        (target_date,)
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        return None, None

    all_entries = ""
    for desc, tag in rows:
        all_entries += f"Description: {desc}\nTags: {tag}\n\n"

    prompt = f"""
You are generating long-term memory for a blind user's voice assistant.

Below are the user's interactions for the day:
{all_entries}

Write a compact, factual memory summary (4–6 sentences) that includes:
- Common environments
- Likely place type ONLY if confident
- Navigation difficulty
- Important landmarks
- Safety concerns
- User movement behavior

Do NOT mention the date.
Do NOT write a story.
Do NOT use phrases like "the individual".

Return format:
Summary: <memory-oriented summary>
Key_Tags: <comma-separated tags>
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
    )

    output = response.output_text

    summary, key_tags = "", ""
    for line in output.split("\n"):
        if line.lower().startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
        elif line.lower().startswith("key_tags:"):
            key_tags = line.split(":", 1)[1].strip()

    return summary, key_tags


# -----------------------------
# IMAGE CAPTURE + COMPRESSION
# -----------------------------
def get_unsummarized_dates():
    today = datetime.now().strftime("%Y-%m-%d")

    # Dates with interactions
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT date FROM interactions")
    interaction_dates = {row[0] for row in c.fetchall()}
    conn.close()

    # Dates already summarized
    conn = sqlite3.connect("daily_summary.db")
    c = conn.cursor()
    c.execute("SELECT date FROM summaries")
    summarized_dates = {row[0] for row in c.fetchall()}
    conn.close()

    # 🔹 Exclude today
    interaction_dates.discard(today)

    return interaction_dates - summarized_dates
def run_pending_summaries():
    pending_dates = get_unsummarized_dates()

    for date in sorted(pending_dates):
        print(f"Generating summary for {date}...")
        summary, key_tags = generate_daily_summary_for_date(date)

        if summary:
            conn = sqlite3.connect("daily_summary.db")
            c = conn.cursor()
            c.execute(
                "INSERT INTO summaries (date, summary, key_tags) VALUES (?, ?, ?)",
                (date, summary, key_tags)
            )
            conn.commit()
            conn.close()

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
    global is_speaking

    r = sr.Recognizer()
    mic = sr.Microphone()

    print("🎤 Listening for wake word 'Alexa'...")

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.4)

        while True:

            # 🔇 Do NOT listen while speaking
            if is_speaking:
                time.sleep(0.2)
                continue

            try:
                audio = r.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=4
                )

                text = r.recognize_google(audio).lower()
                print("Heard:", text)

                if text.startswith(WAKE_WORD):
                    return text[len(WAKE_WORD):].strip()

            except sr.WaitTimeoutError:
                print("⏳ No speech detected")

            except sr.UnknownValueError:
                print("❓ Could not understand audio")

            except Exception as e:
                print("🎤 Mic error:", e)



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
def get_last_n_summaries(n=30):
    conn = sqlite3.connect("daily_summary.db")
    c = conn.cursor()

    c.execute("""
        SELECT date, summary, key_tags
        FROM summaries
        ORDER BY date DESC
        LIMIT ?
    """, (n,))

    rows = c.fetchall()
    conn.close()

    memory_text = ""
    for date, summary, tags in rows:
        memory_text += f"Date: {date}\nSummary: {summary}\nTags: {tags}\n\n"

    return memory_text.strip()
def generate_memory_aware_response(
    user_query,
    current_description,
    current_tags,
    past_memory
):
    prompt = f"""
You are a personalized voice assistant for a blind user.

USER QUESTION:
{user_query}

CURRENT SCENE:
{current_description}

CURRENT TAGS:
{current_tags}

PAST MEMORY (last few days):
{past_memory}

Answer the user's question clearly and calmly.
Use past memory ONLY if it is relevant.
If the place feels familiar, mention it.
If there are known risks, warn the user.
Avoid repeating unnecessary details.

Respond as if speaking directly to the user.
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]
    )

    return response.output_text.strip()

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    global tts_client

    # Start TTS server
    threading.Thread(
        target=start_tts_server,
        daemon=True
    ).start()

    time.sleep(1)

    # Create persistent TTS client
    tts_client = TTSClient(TTS_WS_URL)

    # Start TTS worker thread
    threading.Thread(
        target=tts_worker,
        daemon=True
    ).start()


    print("DEBUG: Entered main()")
    init_db()
    init_daily_summary_db()

    run_pending_summaries()

    print("DEBUG: About to speak system ready")
    tts_queue.put("System is ready.")


    while True:
        try:
            query = listen_for_wake_word()

            if not query.strip():
                tts_queue.put("Yes, what can I help you with?")
                query = listen_for_command()

            if not query.strip():
                tts_queue.put("I did not hear a command.")
                continue

            # 2. Capture image
            tts_queue.put("Capturing image...")


            image_path = capture_and_compress()
            image_uri = img_to_data_uri(image_path)

            # 3. Analyze current scene
            tts_queue.put("Analyzing, please wait...")
            result = analyze_image_and_query(image_uri, query)
            print("GPT Scene Analysis:\n", result)

            # 4. Extract tags
            tags = extract_tags(result)

            # 5. Clean description (IMPORTANT)
            if "Tags:" in result:
                description_only = result.split("Tags:")[0]
            else:
                description_only = result

            description_only = description_only.replace("Description:", "").strip()

            # 6. Store clean interaction
            insert_record(image_path, description_only, query, tags)

            # 7. Load past 4 days memory
            past_memory = get_last_n_summaries(4)

            # 8. Generate memory-aware response
            final_response = generate_memory_aware_response(
                user_query=query,
                current_description=description_only,
                current_tags=tags,
                past_memory=past_memory
            )

            print("Final Personalized Response:\n", final_response)
            time.sleep(0.5)

            # 9. Speak final response
            tts_queue.put(final_response)


        except KeyboardInterrupt:
            print("Exiting.")
            break

        except Exception as e:
            print("ERROR:", e)
            tts_queue.put("I faced an error, but I am still running.")


if __name__ == "__main__":
    main()
