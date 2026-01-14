"""Text-to-speech output helpers."""

import io
import logging
import wave

import sounddevice as sd
from openai import OpenAI

from .config import AUDIO_OUTPUT_DEVICE, TTS_FORMAT, TTS_MODEL, TTS_VOICE

logger = logging.getLogger(__name__)
client = OpenAI()


def synthesize_wav_bytes(text: str) -> bytes:
    if not text:
        return b""
    try:
        # Prefer WAV to avoid external decoders.
        fmt = TTS_FORMAT
        if fmt != "wav":
            fmt = "wav"
        resp = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format=fmt,
        )
        # `resp` is an HttpxBinaryResponseContent wrapper.
        return resp.read()
    except Exception as exc:
        logger.warning("tts_failed error=%s", exc)
        return b""


def play_wav_bytes(wav_bytes: bytes):
    if not wav_bytes:
        return
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if not frames:
            return
        stream = sd.RawOutputStream(
            samplerate=sample_rate,
            dtype={1: "int8", 2: "int16", 4: "int32"}.get(sampwidth, "int16"),
            channels=channels,
            device=AUDIO_OUTPUT_DEVICE,
        )
        try:
            stream.start()
            stream.write(frames)
        finally:
            stream.stop()
            stream.close()
    except Exception as exc:
        logger.warning("tts_playback_failed error=%s", exc)


def speak(text: str):
    wav_bytes = synthesize_wav_bytes(text)
    play_wav_bytes(wav_bytes)
