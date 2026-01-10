import re

from .config import MEMORY_ENABLED, REALTIME_ENABLED
from .context import build_context
from .db import init_db, log_interaction
from .llm_response import get_llm_response
from .memory import store_memory
from .stt_whisper import transcribe_audio
from .tts_openai import speak
from .voice_io import listen_for_wake_word, play_beep, record_command
from .utils.language_guard import is_english

WAKE_WORDS = ["al", "alexa"]
FILLER_PHRASES = ["and the", "uh", "um", "hmm"]

def clean_and_validate_command(text):
    """
    Removes wake words and validates meaningful user intent.
    """
    if not text:
        return None

    text = text.lower()

    # Remove wake words
    for wake in WAKE_WORDS:
        text = re.sub(rf"\b{wake}\b", "", text)

    text = text.strip()

    # Reject if empty
    if not text:
        return None

    # Reject filler-only phrases
    if text in FILLER_PHRASES:
        return None

    # Allow short but meaningful questions
    meaningful_words = [w for w in text.split() if len(w) > 2]
    if len(meaningful_words) < 1:
        return None

    return text


def main():
    if REALTIME_ENABLED:
        from .realtime_client import run_realtime

        run_realtime()
        return

    print("🔊 Voice Assistant Started")
    init_db()

    while True:
        listen_for_wake_word()

        # 🔔 ALWAYS notify user
        play_beep()
        print("🟢 Listening for command...")

        audio_file = record_command()
        raw_text = transcribe_audio(audio_file)
        clean_text = clean_and_validate_command(raw_text)

        if not clean_text:
            print("🤖 ASSISTANT: Sorry, I didn’t understand that. Please repeat.")
            speak("Sorry, I didn’t understand that. Please repeat.")
            continue

        if not is_english(clean_text):
            print("🤖 ASSISTANT: I can only communicate in English. Please repeat in English.")
            speak("I can only communicate in English. Please repeat in English.")
            continue

        print(f"\n👤 USER: {clean_text}")

        extra_context, image_path = build_context(clean_text)
        assistant_text = get_llm_response(clean_text, extra_context=extra_context)
        if not assistant_text:
            print("🤖 ASSISTANT: I'm having trouble right now. Please try again.")
            speak("I'm having trouble right now. Please try again.")
            continue

        print(f"🤖 ASSISTANT: {assistant_text}\n")
        speak(assistant_text)

        if MEMORY_ENABLED:
            store_memory(f"User: {clean_text}\nAssistant: {assistant_text}")
        log_interaction(clean_text, assistant_text, image_path)


if __name__ == "__main__":
    main()
