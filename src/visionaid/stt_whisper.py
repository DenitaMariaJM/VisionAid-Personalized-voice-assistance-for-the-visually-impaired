from openai import OpenAI

client = OpenAI()

def transcribe_audio(audio_file):
    """
    Transcribes speech using OpenAI Whisper
    (Forced to English)
    """
    if not audio_file:
        return ""
    try:
        with open(audio_file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f,
                language="en"   # ✅ FORCE ENGLISH
            )
        text = getattr(transcript, "text", "")
        return (text or "").strip()
    except Exception as exc:
        print(f"Transcription failed: {exc}")
        return ""
