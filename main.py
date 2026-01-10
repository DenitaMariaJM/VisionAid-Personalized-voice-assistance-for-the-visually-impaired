# ==============================
# IMPORTS
# ==============================

import sqlite3          # For saving interactions to SQLite DB
import base64            # For encoding images before sending to LLM
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from openai import OpenAI

# Project-specific modules
from config import (
    WAKE_WORD,
    MODEL,
    REQUIRE_WAKE_WORD,
    DEBUG_SPEECH,
    MEMORY_ENABLED,
    MEMORY_TOP_K,
    MEMORY_TIMEOUT,
    VISION_ENABLED,
    VISION_TIMEOUT,
)
from voice import listen, speak
from vision import capture_image
from memory import store_memory, search_memory
from db import init_db, DB_NAME


# ==============================
# INITIAL SETUP
# ==============================

# Initialize OpenAI client
client = OpenAI()

# Initialize database and tables (runs only once safely)
init_db()


# ==============================
# HELPER FUNCTIONS
# ==============================

def encode_image(image_path):
    """
    Reads an image file from disk and converts it to a base64 string.
    This is REQUIRED because LLMs cannot access local files directly.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def is_vision_query(query: str) -> bool:
    """
    Determines whether the user query needs visual understanding.
    If True, the system MUST capture and send an image to the LLM.
    """
    keywords = [
        "see", "front", "ahead", "in front",
        "left", "right", "around", "near me"
    ]
    return any(k in query.lower() for k in keywords)


def save_interaction(query, answer, image_path):
    """
    Saves the interaction details into the SQLite database.
    This function is isolated to ensure DB writes are reliable.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute(
        """
        INSERT INTO interactions (query, response, image_path)
        VALUES (?, ?, ?)
        """,
        (query, answer, image_path)
    )

    conn.commit()
    conn.close()


def format_memories(memories, max_chars=200):
    if not memories:
        return "None."
    trimmed = []
    for item in memories:
        text = str(item).strip()
        trimmed.append(text[:max_chars])
    return "\n".join(trimmed)


# ==============================
# MAIN ASSISTANT LOOP
# ==============================

def run():
    """
    Main loop of the voice assistant.
    Continuously listens for the wake word and processes user queries.
    """

    # Greet the user once when assistant starts
    speak("Assistant is ready")

    executor = ThreadPoolExecutor(max_workers=2)

    while True:
        try:
            # ------------------------------
            # 1. LISTEN TO USER
            # ------------------------------
            text = listen()
            if not text:
                continue
            raw_text = text
            text = text.lower()
            if DEBUG_SPEECH:
                print(f"Heard: {raw_text}")

            # Ignore speech unless wake word is detected
            if REQUIRE_WAKE_WORD and WAKE_WORD:
                if WAKE_WORD not in text:
                    if DEBUG_SPEECH:
                        print(f'Wake word "{WAKE_WORD}" not detected.')
                    continue

            # Remove wake word from spoken text
            if REQUIRE_WAKE_WORD and WAKE_WORD:
                query = text.replace(WAKE_WORD, "").strip()
            else:
                query = text.strip()

            # If user said only the wake word (e.g., "Alexa")
            if not query:
                speak("Yes? Please tell me how I can help.")
                continue

            # ------------------------------
            # 2. IMAGE CAPTURE (IF NEEDED)
            # ------------------------------
            image_path = None
            image_future = None

            # Force image capture for vision-based questions
            if VISION_ENABLED and is_vision_query(query):
                image_future = executor.submit(capture_image)

            # ------------------------------
            # 3. MEMORY RETRIEVAL
            # ------------------------------
            memories = []
            memory_future = None
            if MEMORY_ENABLED:
                memory_future = executor.submit(search_memory, query, MEMORY_TOP_K)

            if memory_future:
                try:
                    memories = memory_future.result(timeout=MEMORY_TIMEOUT)
                except TimeoutError:
                    if DEBUG_SPEECH:
                        print("Memory retrieval timed out.")
                except Exception as exc:
                    if DEBUG_SPEECH:
                        print(f"Memory retrieval failed: {exc}")

            if image_future:
                try:
                    image_path = image_future.result(timeout=VISION_TIMEOUT)
                except TimeoutError:
                    if DEBUG_SPEECH:
                        print("Image capture timed out.")
                except Exception as exc:
                    if DEBUG_SPEECH:
                        print(f"Image capture failed: {exc}")

            # ------------------------------
            # 4. PROMPT CONSTRUCTION
            # ------------------------------
            prompt = f"""
You are a voice assistant designed to help a visually impaired user navigate safely.

IMPORTANT RULES:
- Do NOT describe decorative details (artwork, colors, ceiling, textures).
- Focus ONLY on objects or obstacles that affect movement.
- Prioritize what is directly in front of the user.
- Mention left or right ONLY if relevant for navigation.
- Keep responses short, calm, and action-oriented.
- Avoid uncertainty words like "possibly", "might be", or guesses.
- If the path is clear, explicitly say it is safe to move.

Response format:
- Start with what is in front.
- Then mention left or right if needed.
- End with a safety or movement suggestion.

Previous context:
{format_memories(memories)}

User question:
{query}
"""

            # ------------------------------
            # 5. LLM MESSAGE FORMAT
            # ------------------------------
            # If image exists, send BOTH text + image (multimodal)
            if image_path:
                image_base64 = encode_image(image_path)

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }]
            else:
                # Text-only query
                messages = [{
                    "role": "user",
                    "content": prompt
                }]

            # ------------------------------
            # 6. CALL THE LLM
            # ------------------------------
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )

            # Extract assistant's reply
            answer = response.choices[0].message.content

            # ------------------------------
            # 7. SPEAK RESPONSE
            # ------------------------------
            speak(answer)

            # ------------------------------
            # 8. STORE MEMORY & DB
            # ------------------------------
            # Store interaction in vector memory (for context)
            store_memory(f"{query} {answer}")

            # Persist interaction in SQLite database
            save_interaction(query, answer, image_path)

        except Exception as e:
            # Catch-all to prevent assistant from crashing
            print("ERROR:", e)
            speak("Sorry, I did not understand.")


# ==============================
# START ASSISTANT
# ==============================

run()
