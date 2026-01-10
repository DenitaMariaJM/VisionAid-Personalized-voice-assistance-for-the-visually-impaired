import base64
import json
import os
import tempfile
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
    REALTIME_OUTPUT_SUPPRESS_SECONDS,
    REALTIME_RESPONSE_STYLE,
    REALTIME_SAMPLE_RATE,
    REALTIME_SILENCE_DURATION,
    REALTIME_SILENCE_THRESHOLD,
    REALTIME_VOICE,
    REALTIME_WAKE_WORDS,
)
from .context import build_context
from .db import init_db, log_interaction
from .memory import store_memory
from .stt_whisper import transcribe_audio
from .utils.command_validation import is_confident_command
from .utils.language_guard import is_english


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

        init_db()

    def _send_event(self, event):
        try:
            if self._ws:
                self._ws.send(json.dumps(event))
        except Exception as exc:
            print(f"WebSocket send failed: {exc}")

    def _reset_vad(self):
        self._buffer = bytearray()
        self._speech_detected = False
        self._speech_duration = 0.0
        self._silence_duration = 0.0

    def _write_wav(self, pcm_bytes):
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.sample_width)
                wf.setframerate(self.rate)
                wf.writeframes(pcm_bytes)
            return temp_file.name
        finally:
            temp_file.close()

    def _build_context(self, user_text):
        extra_context, image_path = build_context(user_text)
        self._last_image_path = image_path
        return extra_context

    def _handle_user_audio(self, pcm_bytes):
        if not pcm_bytes or self._response_in_flight:
            return

        wav_path = None
        try:
            wav_path = self._write_wav(pcm_bytes)
            user_text = transcribe_audio(wav_path)
        finally:
            if wav_path:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        if not user_text:
            return

        if not is_english(user_text):
            self._assistant_text = ""
            self._response_in_flight = True
            event = {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio", "text"],
                    "instructions": (
                        f"{REALTIME_RESPONSE_STYLE}\n\n"
                        "Respond with: I can only communicate in English. "
                        "Please repeat in English."
                    ),
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

        instructions = f"{REALTIME_RESPONSE_STYLE}\n\nUser said: {user_text}"
        if extra_context:
            instructions = f"{instructions}\n\n{extra_context}"

        event = {
            "type": "response.create",
            "response": {
                "output_modalities": ["audio", "text"],
                "instructions": instructions,
                "metadata": {"user_text": user_text},
            },
        }
        self._assistant_text = ""
        self._response_in_flight = True
        self._send_event(event)

    def _mic_stream_loop(self):
        while not self._stop_event.is_set():
            try:
                pcm, overflowed = self._audio_in.read(self.frames_per_chunk)
                if overflowed:
                    print("Audio input overflowed.")
            except Exception as exc:
                print(f"Mic error: {exc}")
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
                    threading.Thread(
                        target=self._handle_user_audio,
                        args=(audio_bytes,),
                        daemon=True,
                    ).start()
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
                    print(f"Audio output error: {exc}")
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
                        f"User: {self._last_user_text}\nAssistant: {assistant_text}"
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
        print(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} {close_msg}")
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
