"""Realtime audio assistant over WebSocket with local VAD and hooks."""

import base64
import io
import json
import logging
import os
import threading
import time
import wave

import numpy as np
import sounddevice as sd
import websocket

from .config import (
    AUDIO_INPUT_DEVICE,
    AUDIO_OUTPUT_DEVICE,
    MEMORY_ENABLED,
    REALTIME_CHUNK_MS,
    REALTIME_MODEL,
    REALTIME_MIN_SPEECH_DURATION,
    REALTIME_MAX_OUTPUT_TOKENS,
    REALTIME_OUTPUT_SUPPRESS_SECONDS,
    REALTIME_MAX_BUFFER_SECONDS,
    REALTIME_RESPONSE_STYLE,
    REALTIME_SAMPLE_RATE,
    REALTIME_SILENCE_DURATION,
    REALTIME_SILENCE_THRESHOLD,
    REALTIME_TRANSCRIPTION_MODEL,
    REALTIME_TRANSCRIPT_TIMEOUT,
    REALTIME_USE_LOCAL_FALLBACK,
    REALTIME_VOICE,
    REALTIME_WAKE_WORDS,
    validate_config,
)
from .context import build_context
from .db import init_db, log_interaction
from .logging_utils import configure_logging
from .memory import build_memory_entry, load_memory, store_memory
from .stt_whisper import transcribe_audio
from .utils.command_validation import is_confident_command
from .utils.language_guard import is_english

logger = logging.getLogger(__name__)


def _b64encode_audio(pcm_bytes):
    return base64.b64encode(pcm_bytes).decode("utf-8")


def _b64decode_audio(b64_str):
    return base64.b64decode(b64_str)


def _build_realtime_url():
    return f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"


def _audio_peak(pcm_bytes):
    if not pcm_bytes:
        return 0.0
    data = np.frombuffer(pcm_bytes, dtype=np.int16)
    if data.size == 0:
        return 0.0
    return float(np.max(np.abs(data))) / 32768.0


def _extract_transcript(data):
    event_type = str(data.get("type", ""))
    if "transcription" not in event_type:
        return ""
    if not any(key in event_type for key in ("completed", "done", "final")):
        return ""
    for key in ("transcript", "text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    item = data.get("item") or {}
    content = item.get("content") or []
    for part in content:
        if not isinstance(part, dict):
            continue
        for key in ("text", "transcript"):
            value = part.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _suppress_alsa_errors():
    try:
        import ctypes

        def _no_alsa_errors(filename, line, function, err, fmt):
            return

        error_handler = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
        )
        c_handler = error_handler(_no_alsa_errors)
        asound = ctypes.cdll.LoadLibrary("libasound.so")
        asound.snd_lib_error_set_handler(c_handler)
    except Exception:
        pass


class RealtimeAssistant:
    def __init__(self):
        configure_logging()
        validate_config()
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set OPENAI_API_KEY env var first.")

        self.rate = REALTIME_SAMPLE_RATE
        self.chunk_ms = REALTIME_CHUNK_MS
        self.frames_per_chunk = int(self.rate * (self.chunk_ms / 1000.0))
        self.sample_width = 2
        self._audio_in = None
        self._audio_out = None
        self._ws = None
        self._stop_event = threading.Event()

        self._buffer = bytearray()
        self._speech_detected = False
        self._speech_duration = 0.0
        self._silence_duration = 0.0
        self._response_in_flight = False

        self._last_user_text = ""
        self._last_image_path = None
        self._assistant_text = ""
        self._last_output_time = 0.0
        self._output_suppress_seconds = REALTIME_OUTPUT_SUPPRESS_SECONDS
        self._max_buffer_bytes = int(
            self.rate * REALTIME_MAX_BUFFER_SECONDS * self.sample_width
        )
        self._awaiting_transcript = False
        self._pending_audio = None
        self._pending_since = None

        init_db()
        load_memory()

    def _send_event(self, event):
        try:
            if self._ws:
                self._ws.send(json.dumps(event))
        except Exception as exc:
            logger.warning("websocket_send_failed error=%s", exc)

    def _reset_vad(self):
        self._buffer = bytearray()
        self._speech_detected = False
        self._speech_duration = 0.0
        self._silence_duration = 0.0

    def _wav_bytes(self, pcm_bytes):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.rate)
            wf.writeframes(pcm_bytes)
        return buffer.getvalue()

    def _build_context(self, user_text):
        extra_context, image_path = build_context(user_text)
        self._last_image_path = image_path
        return extra_context

    def _handle_user_text(self, user_text):
        if not user_text or self._response_in_flight:
            return

        if not is_english(user_text):
            self._assistant_text = ""
            self._response_in_flight = True
            event = {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio", "text"],
                    "instructions": (
                        "Respond with: I can only communicate in English. "
                        "Please repeat in English."
                    ),
                    "max_output_tokens": 40,
                },
            }
            self._send_event(event)
            return

        if not is_confident_command(user_text):
            return

        if REALTIME_WAKE_WORDS:
            lowered = user_text.lower()
            if not any(word in lowered for word in REALTIME_WAKE_WORDS):
                return
            for word in REALTIME_WAKE_WORDS:
                lowered = lowered.replace(word, "").strip()
            user_text = lowered.strip()
            if not user_text:
                return

        self._last_user_text = user_text
        extra_context = self._build_context(user_text)

        if extra_context:
            input_text = f"Context:\n{extra_context}\n\nUser: {user_text}"
        else:
            input_text = user_text

        event = {
            "type": "response.create",
            "response": {
                "output_modalities": ["audio", "text"],
                "instructions": REALTIME_RESPONSE_STYLE,
                "input": [{"type": "input_text", "text": input_text}],
                "max_output_tokens": REALTIME_MAX_OUTPUT_TOKENS,
                "metadata": {"user_text": user_text},
            },
        }
        self._assistant_text = ""
        self._response_in_flight = True
        self._send_event(event)

    def _queue_pending_audio(self, audio_bytes):
        self._pending_audio = audio_bytes
        self._pending_since = time.time()
        self._awaiting_transcript = True

    def _maybe_fallback_transcription(self):
        if not REALTIME_USE_LOCAL_FALLBACK:
            return
        if not self._awaiting_transcript or not self._pending_audio:
            return
        if REALTIME_TRANSCRIPT_TIMEOUT <= 0:
            return
        if time.time() - self._pending_since < REALTIME_TRANSCRIPT_TIMEOUT:
            return
        wav_bytes = self._wav_bytes(self._pending_audio)
        self._pending_audio = None
        self._awaiting_transcript = False
        transcript = transcribe_audio(wav_bytes)
        if transcript:
            self._handle_user_text(transcript)

    def _mic_stream_loop(self):
        while not self._stop_event.is_set():
            self._maybe_fallback_transcription()
            try:
                pcm, overflowed = self._audio_in.read(self.frames_per_chunk)
                if overflowed:
                    logger.warning("audio_input_overflowed")
            except Exception as exc:
                logger.warning("mic_read_failed error=%s", exc)
                time.sleep(0.05)
                continue

            if self._response_in_flight:
                continue

            now = time.time()
            if now - self._last_output_time < self._output_suppress_seconds:
                self._reset_vad()
                continue

            peak = _audio_peak(pcm)
            if peak >= REALTIME_SILENCE_THRESHOLD:
                if not self._speech_detected:
                    self._reset_vad()
                    self._speech_detected = True
                self._speech_duration += self.chunk_ms / 1000.0
                self._buffer.extend(pcm)
                self._send_event(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": _b64encode_audio(pcm),
                    }
                )
                continue

            if self._speech_detected:
                self._silence_duration += self.chunk_ms / 1000.0
                self._buffer.extend(pcm)
                if len(self._buffer) > self._max_buffer_bytes:
                    self._send_event({"type": "input_audio_buffer.clear"})
                    self._reset_vad()
                    continue
                self._send_event(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": _b64encode_audio(pcm),
                    }
                )
                if (
                    self._silence_duration >= REALTIME_SILENCE_DURATION
                    and self._speech_duration >= REALTIME_MIN_SPEECH_DURATION
                ):
                    audio_bytes = bytes(self._buffer)
                    self._send_event({"type": "input_audio_buffer.commit"})
                    self._send_event({"type": "input_audio_buffer.clear"})
                    self._reset_vad()
                    self._queue_pending_audio(audio_bytes)
                elif self._silence_duration >= REALTIME_SILENCE_DURATION:
                    self._send_event({"type": "input_audio_buffer.clear"})
                    self._reset_vad()
            else:
                time.sleep(0.005)

    def _on_open(self, ws):
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio", "text"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": self.rate},
                    },
                    "output": {
                        "format": {"type": "audio/pcm"},
                        "voice": REALTIME_VOICE,
                    },
                },
                "input_audio_transcription": {
                    "model": REALTIME_TRANSCRIPTION_MODEL,
                    "language": "en",
                },
                "instructions": REALTIME_RESPONSE_STYLE,
            },
        }
        self._send_event(session_update)

        threading.Thread(target=self._mic_stream_loop, daemon=True).start()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        event_type = data.get("type")
        transcript = _extract_transcript(data)
        if transcript:
            self._pending_audio = None
            self._awaiting_transcript = False
            self._handle_user_text(transcript)
            return

        if event_type == "response.output_text.delta":
            delta = data.get("delta", "")
            if delta:
                self._assistant_text += delta
                print(delta, end="", flush=True)
            return

        if event_type == "response.audio.delta":
            delta_b64 = data.get("delta")
            if delta_b64:
                pcm = _b64decode_audio(delta_b64)
                try:
                    self._audio_out.write(pcm)
                    self._last_output_time = time.time()
                except Exception as exc:
                    logger.warning("audio_output_failed error=%s", exc)
            return

        if event_type == "response.done":
            if self._assistant_text:
                print("\n", flush=True)
            self._response_in_flight = False
            self._last_output_time = time.time()
            assistant_text = self._assistant_text.strip()
            if self._last_user_text and assistant_text:
                if MEMORY_ENABLED:
                    store_memory(
                        build_memory_entry(self._last_user_text, assistant_text)
                    )
                log_interaction(
                    self._last_user_text,
                    assistant_text,
                    self._last_image_path,
                )
            self._last_user_text = ""
            self._last_image_path = None
            return

    def _on_error(self, ws, error):
        logger.warning("websocket_error error=%s", error)

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("websocket_closed status=%s message=%s",
                    close_status_code, close_msg)
        self._stop_event.set()

    def _setup_audio(self):
        _suppress_alsa_errors()
        self._audio_in = sd.RawInputStream(
            samplerate=self.rate,
            blocksize=self.frames_per_chunk,
            dtype="int16",
            channels=1,
            device=AUDIO_INPUT_DEVICE,
        )
        self._audio_out = sd.RawOutputStream(
            samplerate=self.rate,
            blocksize=self.frames_per_chunk,
            dtype="int16",
            channels=1,
            device=AUDIO_OUTPUT_DEVICE,
        )
        self._audio_in.start()
        self._audio_out.start()

    def _setup_ws(self):
        headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Beta: realtime=v1",
        ]
        self._ws = websocket.WebSocketApp(
            _build_realtime_url(),
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

    def run(self):
        self._setup_audio()
        self._setup_ws()
        try:
            self._ws.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            try:
                if self._audio_in:
                    self._audio_in.stop()
                    self._audio_in.close()
            except Exception:
                pass
            try:
                if self._audio_out:
                    self._audio_out.stop()
                    self._audio_out.close()
            except Exception:
                pass


def run_realtime():
    assistant = RealtimeAssistant()
    assistant.run()
