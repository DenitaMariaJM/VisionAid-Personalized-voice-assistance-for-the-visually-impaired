from voice_io import listen_for_wake_word, record_command, play_beep
from stt_whisper import transcribe_audio
from llm_response import get_llm_response
from tts_openai import speak
import re
from utils.command_validation import is_confident_command


listen_for_wake_word()
play_beep()          # 🔔 User hears confirmation

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
    print("🔊 Voice Assistant Started")

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

        print(f"\n👤 USER: {clean_text}")

        assistant_text = get_llm_response(clean_text)

        print(f"🤖 ASSISTANT: {assistant_text}\n")
        speak(assistant_text)


if __name__ == "__main__":
    main()
