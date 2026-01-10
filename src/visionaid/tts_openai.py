from openai import OpenAI
from playsound import playsound
import uuid
import os
import time

client = OpenAI()

def speak(text):
    """
    Bulletproof OpenAI TTS for Windows.
    Uses unique filenames to avoid file locking.
    """
    if not text or not text.strip():
        return
    audio_file = f"assistant_{uuid.uuid4().hex}.mp3"
    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=text
        )
        with open(audio_file, "wb") as f:
            f.write(response.read())
        playsound(audio_file)
        # Allow Windows to release file lock
        time.sleep(0.2)
    except Exception as exc:
        print(f"TTS failed: {exc}")
    finally:
        try:
            os.remove(audio_file)
        except FileNotFoundError:
            pass
        except PermissionError:
            pass
