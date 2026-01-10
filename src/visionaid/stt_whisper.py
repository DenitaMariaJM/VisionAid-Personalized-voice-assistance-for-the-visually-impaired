"""Speech-to-text helper for OpenAI Whisper transcription."""

import logging

from openai import OpenAI

from .config import STT_MODEL

client = OpenAI()
logger = logging.getLogger(__name__)

def transcribe_audio(audio_file):
    """
    Transcribes speech using OpenAI Whisper
    (Forced to English)
    """
    if not audio_file:
        return ""
    try:
        if isinstance(audio_file, (bytes, bytearray)):
            file_payload = ("speech.wav", audio_file, "audio/wav")
            transcript = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=file_payload,
                language="en"   # ✅ FORCE ENGLISH
            )
        else:
            with open(audio_file, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model=STT_MODEL,
                    file=f,
                    language="en"   # ✅ FORCE ENGLISH
                )
        text = getattr(transcript, "text", "")
        return (text or "").strip()
    except Exception as exc:
        logger.warning("transcription_failed error=%s", exc)
        return ""
