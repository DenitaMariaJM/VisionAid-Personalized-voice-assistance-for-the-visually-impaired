from openai import OpenAI
from playsound import playsound
import uuid
import os
import time

print("✅ USING UPDATED tts_openai.py")

client = OpenAI()

def speak(text):
    """
    Bulletproof OpenAI TTS for Windows.
    Uses unique filenames to avoid file locking.
    """
    audio_file = f"assistant_{uuid.uuid4().hex}.mp3"

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

    try:
        os.remove(audio_file)
    except PermissionError:
        pass
