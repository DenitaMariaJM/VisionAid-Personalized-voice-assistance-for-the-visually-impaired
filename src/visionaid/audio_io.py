"""Microphone capture utilities with simple VAD."""

import io
import logging
import time
import wave

import numpy as np
import sounddevice as sd

from .config import (
    AUDIO_INPUT_DEVICE,
    REALTIME_CHUNK_MS,
    REALTIME_MAX_BUFFER_SECONDS,
    REALTIME_SAMPLE_RATE,
    REALTIME_SILENCE_DURATION,
    REALTIME_SILENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _audio_peak(pcm_bytes: bytes) -> float:
    if not pcm_bytes:
        return 0.0
    data = np.frombuffer(pcm_bytes, dtype=np.int16)
    if data.size == 0:
        return 0.0
    return float(np.max(np.abs(data))) / 32768.0


def _wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()


def record_utterance_wav_bytes() -> bytes:
    """
    Record a single utterance from the microphone using a simple energy VAD.

    Returns WAV bytes (16-bit mono) or b"" if no speech detected.
    """
    rate = REALTIME_SAMPLE_RATE
    chunk_ms = REALTIME_CHUNK_MS
    frames_per_chunk = int(rate * (chunk_ms / 1000.0))
    max_bytes = int(rate * REALTIME_MAX_BUFFER_SECONDS * 2)

    buffer = bytearray()
    speech_detected = False
    silence_seconds = 0.0

    stream = sd.RawInputStream(
        samplerate=rate,
        blocksize=frames_per_chunk,
        dtype="int16",
        channels=1,
        device=AUDIO_INPUT_DEVICE,
    )

    try:
        stream.start()
        started_at = time.time()
        while True:
            pcm, overflowed = stream.read(frames_per_chunk)
            if overflowed:
                logger.warning("audio_input_overflowed")

            peak = _audio_peak(pcm)
            if peak >= REALTIME_SILENCE_THRESHOLD:
                if not speech_detected:
                    speech_detected = True
                    buffer.clear()
                    silence_seconds = 0.0
                buffer.extend(pcm)
            else:
                if speech_detected:
                    buffer.extend(pcm)
                    silence_seconds += chunk_ms / 1000.0
                    if silence_seconds >= REALTIME_SILENCE_DURATION:
                        break
                else:
                    if time.time() - started_at > REALTIME_MAX_BUFFER_SECONDS:
                        return b""

            if len(buffer) > max_bytes:
                # Truncate rather than continuing to grow buffers.
                logger.info("audio_capture_truncated max_seconds=%s", REALTIME_MAX_BUFFER_SECONDS)
                break
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass

    if not speech_detected or not buffer:
        return b""
    return _wav_bytes(bytes(buffer), rate)
