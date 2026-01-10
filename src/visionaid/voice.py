import io
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

try:
    from .config import (
        STT_MODEL,
        TTS_MODEL,
        TTS_VOICE,
        TTS_SPEED,
        STT_SAMPLE_RATE,
        STT_CHANNELS,
        STT_PHRASE_TIME_LIMIT,
        STT_SILENCE_THRESHOLD,
        STT_SILENCE_DURATION,
        STT_MIN_SPEECH_DURATION,
        STT_CHUNK_DURATION,
        AUDIO_INPUT_DEVICE,
        AUDIO_OUTPUT_DEVICE,
    )
except Exception:
    STT_MODEL = "gpt-4o-mini-transcribe"
    TTS_MODEL = "gpt-4o-mini-tts"
    TTS_VOICE = "alloy"
    TTS_SPEED = 1.0
    STT_SAMPLE_RATE = 16000
    STT_CHANNELS = 1
    STT_PHRASE_TIME_LIMIT = 6
    STT_SILENCE_THRESHOLD = 0.01
    STT_SILENCE_DURATION = 0.8
    STT_MIN_SPEECH_DURATION = 0.3
    STT_CHUNK_DURATION = 0.2
    AUDIO_INPUT_DEVICE = None
    AUDIO_OUTPUT_DEVICE = None


client = OpenAI()


def _read_response_bytes(response):
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)
    for attr in ("read", "content"):
        value = getattr(response, attr, None)
        if callable(value):
            try:
                return value()
            except Exception:
                pass
        if value:
            return value
    iter_bytes = getattr(response, "iter_bytes", None)
    if callable(iter_bytes):
        return b"".join(iter_bytes())
    return None


def _record_audio(max_duration):
    chunk_duration = max(float(STT_CHUNK_DURATION), 0.05)
    max_duration = max(float(max_duration), chunk_duration)
    audio_chunks = []
    total_duration = 0.0
    speech_duration = 0.0
    silence_duration = 0.0
    speech_detected = False

    while total_duration < max_duration:
        frames = int(chunk_duration * STT_SAMPLE_RATE)
        audio = sd.rec(
            frames,
            samplerate=STT_SAMPLE_RATE,
            channels=STT_CHANNELS,
            dtype="float32",
            device=AUDIO_INPUT_DEVICE,
        )
        sd.wait()
        audio_chunks.append(audio)
        total_duration += chunk_duration

        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak >= STT_SILENCE_THRESHOLD:
            speech_detected = True
            silence_duration = 0.0
            speech_duration += chunk_duration
        else:
            if speech_detected:
                silence_duration += chunk_duration
                if (
                    silence_duration >= STT_SILENCE_DURATION
                    and speech_duration >= STT_MIN_SPEECH_DURATION
                ):
                    break

    if not speech_detected:
        return None
    return np.concatenate(audio_chunks, axis=0)


def _audio_is_silent(audio):
    if audio is None or audio.size == 0:
        return True
    peak = float(np.max(np.abs(audio)))
    return peak < STT_SILENCE_THRESHOLD


def _wav_bytes(audio):
    buffer = io.BytesIO()
    sf.write(buffer, audio, STT_SAMPLE_RATE, format="WAV")
    return buffer.getvalue()


def speak(text):
    if not text.strip():
        return
    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="wav",
            speed=TTS_SPEED,
        )
        audio_bytes = _read_response_bytes(response)
        if not audio_bytes:
            print("TTS failed: empty audio response.")
            return
        data, rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        sd.play(data, rate, device=AUDIO_OUTPUT_DEVICE)
        sd.wait()
    except Exception as exc:
        print(f"TTS failed: {exc}")


def listen():
    duration = STT_PHRASE_TIME_LIMIT or 6
    print("Listening...")
    try:
        audio = _record_audio(duration)
    except Exception as exc:
        print(f"Audio capture failed: {exc}")
        time.sleep(0.1)
        return ""

    if _audio_is_silent(audio):
        print("No speech detected.")
        time.sleep(0.1)
        return ""

    try:
        audio_bytes = _wav_bytes(audio)
        transcript = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=("speech.wav", audio_bytes, "audio/wav"),
        )
        text = getattr(transcript, "text", None)
        return (text or "").strip()
    except Exception as exc:
        print(f"Speech recognition failed: {exc}")
        time.sleep(0.1)
        return ""
