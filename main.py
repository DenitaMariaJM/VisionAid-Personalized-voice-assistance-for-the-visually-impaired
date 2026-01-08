# ==============================
# IMPORTS
# ==============================

import sqlite3          # For saving interactions to SQLite DB
import base64            # For encoding images before sending to LLM
from openai import OpenAI

# Project-specific modules
from config import WAKE_WORD
from voice import listen, speak
from vision import capture_image
from memory import store_memory, search_memory
from db import init_db


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
    conn = sqlite3.connect("assistant.db")
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

    while True:
        try:
            # ------------------------------
            # 1. LISTEN TO USER
            # ------------------------------
            text = listen().lower()

            # Ignore speech unless wake word is detected
            if WAKE_WORD not in text:
                continue

            # Remove wake word from spoken text
            query = text.replace(WAKE_WORD, "").strip()

            # If user said only the wake word (e.g., "Alexa")
            if not query:
                speak("Yes? Please tell me how I can help.")
                continue


            # ------------------------------
            # 2. IMAGE CAPTURE (IF NEEDED)
            # ------------------------------
            image_path = None

            # Force image capture for vision-based questions
            if is_vision_query(query):
                image_path = capture_image()


            # ------------------------------
            # 3. MEMORY RETRIEVAL
            # ------------------------------
            # Fetch semantically similar past interactions
            memories = search_memory(query)


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
{memories}

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
                model="gpt-4o-mini",   # Vision-capable model
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
