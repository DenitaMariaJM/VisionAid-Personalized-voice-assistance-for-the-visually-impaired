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
    REALTIME_TOOL_GUIDANCE,
    REALTIME_TOOL_CHOICE,
    REALTIME_TRANSCRIPTION_MODEL,
    REALTIME_TRANSCRIPT_TIMEOUT,
    REALTIME_USE_LOCAL_FALLBACK,
    REALTIME_VOICE,
    REALTIME_WAKE_WORDS,
    TOOL_ACCESS_RECHECK_SECONDS,
    validate_config,
)
from .db import init_db, log_interaction
from .logging_utils import configure_logging
from .memory import build_memory_entry, load_memory, store_memory
from .stt_whisper import transcribe_audio
from .tools import TOOLS, execute_tool, serialize_tool_result
from .tool_access import check_camera_access, should_recheck
from .utils.language_guard import is_english

logger = logging.getLogger(__name__)

VISION_QUERY_KEYWORDS = (
    "around me",
    "around",
    "surroundings",
    "what do you see",
    "what is there",
    "what's there",
    "look",
    "see",
    "scan",
    "check obstacles",
    "obstacle",
    "obstacles",
    "ahead",
    "in front",
    "left",
    "right",
    "take a picture",
    "take a photo",
    "capture",
    "camera",
)
MEMORY_QUERY_KEYWORDS = (
    "remember",
    "recall",
    "what did i say",
    "what do you remember",
    "last time",
    "previously",
)


def _debug(message):
    print(f"[visionaid] {message}", flush=True)
    logger.info(message)


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


def _needs_vision(text):
    lowered = text.lower()
    return any(key in lowered for key in VISION_QUERY_KEYWORDS)


def _needs_memory(text):
    lowered = text.lower()
    return any(key in lowered for key in MEMORY_QUERY_KEYWORDS)


def _format_memory(results):
    if not results:
        return ""
    return "Memory recall: " + " | ".join(results)


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
        self._tool_calls = {}
        self._camera_ok = None
        self._camera_checked_at = None
        self._language_warning_sent = False
        self._mic_started = False
        self._logged_first_event = False

        init_db()
        load_memory()

    def _send_event(self, event):
        try:
            if self._ws:
                self._ws.send(json.dumps(event))
        except Exception as exc:
            logger.warning("websocket_send_failed error=%s", exc)

    def _ensure_camera_access(self):
        if should_recheck(self._camera_checked_at, TOOL_ACCESS_RECHECK_SECONDS):
            cam_ok, cam_msg = check_camera_access()
            self._camera_ok = cam_ok
            self._camera_checked_at = time.time()
            logger.info("camera_recheck %s", cam_msg)
        return bool(self._camera_ok)

    def _execute_tool_call(self, call_id, name, arguments):
        _debug(f"tool_call_execute name={name} call_id={call_id}")
        camera_ok = True
        if name in ("capture_image", "analyze_image"):
            camera_ok = self._ensure_camera_access()
            if not camera_ok:
                logger.warning("camera_access_failed_precheck")
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}
        result = execute_tool(name, args)
        if isinstance(result, dict) and result.get("image_path"):
            self._last_image_path = result.get("image_path")
            self._camera_ok = True
        if name in ("capture_image", "analyze_image") and not result.get("ok", True):
            if not camera_ok:
                result = {
                    "ok": False,
                    "error": (
                        "Camera access failed. Check permissions and "
                        "CAMERA_INDEX."
                    ),
                }

        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": serialize_tool_result(result),
            },
        }
        self._send_event(event)
        self._send_event(
            {
                "type": "response.create",
                "response": {
                    "output_modalities": ["audio", "text"],
                    "instructions": (
                        f"{REALTIME_RESPONSE_STYLE}\n\n{REALTIME_TOOL_GUIDANCE}"
                    ),
                    "tools": TOOLS,
                    "tool_choice": REALTIME_TOOL_CHOICE,
                },
            }
        )

    def _find_tool_entry(self, call_id, item_id):
        if call_id and call_id in self._tool_calls:
            return self._tool_calls[call_id]
        if item_id and item_id in self._tool_calls:
            return self._tool_calls[item_id]
        return None

    def _ensure_tool_entry(self, call_id, item_id, name):
        entry = self._find_tool_entry(call_id, item_id)
        if not entry and not name:
            return None
        if not entry:
            entry = {"name": name or "", "arguments": "", "keys": set()}
            logger.info("tool_call_registered name=%s call_id=%s item_id=%s",
                        name, call_id, item_id)
        if name:
            entry["name"] = name
        if call_id:
            entry["call_id"] = call_id
            entry["keys"].add(call_id)
        if item_id:
            entry["item_id"] = item_id
            entry["keys"].add(item_id)
        for key in entry["keys"]:
            self._tool_calls[key] = entry
        return entry

    def _clear_tool_entry(self, entry):
        if not entry:
            return
        for key in entry.get("keys", []):
            self._tool_calls.pop(key, None)

    def _handle_tool_event(self, data):
        event_type = str(data.get("type", ""))
        item = data.get("item") or {}
        if not isinstance(item, dict):
            item = {}
        item_type = item.get("type")
        function_block = item.get("function") or item.get("tool") or {}
        call_id = data.get("call_id") or item.get("call_id")
        item_id = data.get("item_id") or item.get("id") or data.get("id")
        name = (
            data.get("name")
            or item.get("name")
            or function_block.get("name")
        )
        arguments = (
            data.get("arguments")
            or item.get("arguments")
            or function_block.get("arguments")
            or ""
        )
        delta = (
            data.get("delta")
            or data.get("arguments_delta")
            or item.get("arguments_delta")
            or function_block.get("arguments_delta")
            or data.get("arguments")
            or function_block.get("arguments")
        )
        if isinstance(delta, dict):
            delta = delta.get("arguments_delta") or delta.get("arguments")

        if item_type in ("function_call", "tool_call") or "function_call" in event_type:
            _debug(
                f"tool_event type={event_type} item_type={item_type} "
                f"name={name} call_id={call_id} item_id={item_id}"
            )

        if event_type in ("response.output_item.added", "response.output_item.updated"):
            if item_type in ("function_call", "tool_call"):
                entry = self._ensure_tool_entry(call_id, item_id, name)
                if entry and arguments:
                    entry["arguments"] = arguments
                return True

        if event_type in (
            "response.output_item.delta",
            "response.function_call_arguments.delta",
        ):
            entry = self._ensure_tool_entry(call_id, item_id, name)
            if entry and delta:
                entry["arguments"] += delta
                return True

        if event_type in (
            "response.output_item.done",
            "response.output_item.completed",
            "response.function_call_arguments.done",
        ) or (
            item_type in ("function_call", "tool_call")
            and str(item.get("status", "")) in ("completed", "done")
        ):
            entry = self._find_tool_entry(call_id, item_id)
            entry = entry or self._ensure_tool_entry(call_id, item_id, name)
            if not entry:
                return False
            if arguments and not entry.get("arguments"):
                entry["arguments"] = arguments
            self._clear_tool_entry(entry)
            resolved_call_id = entry.get("call_id") or call_id or item_id
            resolved_name = entry.get("name") or name
            if resolved_call_id and resolved_name:
                self._execute_tool_call(
                    resolved_call_id,
                    resolved_name,
                    entry.get("arguments", ""),
                )
                return True

        if "tool" in event_type or "function_call" in event_type:
            logger.debug("unhandled_tool_event type=%s data=%s", event_type, data)
        return False

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

    def _handle_user_text(self, user_text):
        if not user_text or self._response_in_flight:
            return

        _debug(f"user_text_received length={len(user_text)}")

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

        if REALTIME_WAKE_WORDS:
            lowered = user_text.lower()
            if not any(word in lowered for word in REALTIME_WAKE_WORDS):
                _debug("wake_word_missing")
                return
            for word in REALTIME_WAKE_WORDS:
                lowered = lowered.replace(word, "").strip()
            user_text = lowered.strip()
            if not user_text:
                _debug("wake_word_only")
                return

        self._last_user_text = user_text
        self._last_image_path = None
        user_text, tool_choice = self._apply_inference_layer(user_text)
        _debug(f"response_tool_choice {tool_choice}")

        event = {
            "type": "response.create",
            "response": {
                "output_modalities": ["audio", "text"],
                "instructions": (
                    f"{REALTIME_RESPONSE_STYLE}\n\n{REALTIME_TOOL_GUIDANCE}"
                ),
                "input": [{"type": "input_text", "text": user_text}],
                "max_output_tokens": REALTIME_MAX_OUTPUT_TOKENS,
                "tools": TOOLS,
                "tool_choice": tool_choice,
                "metadata": {"user_text": user_text},
            },
        }
        self._assistant_text = ""
        self._response_in_flight = True
        self._send_event(event)

    def _apply_inference_layer(self, user_text):
        context_lines = []
        tool_choice = REALTIME_TOOL_CHOICE

        needs_vision = _needs_vision(user_text)
        needs_memory = _needs_memory(user_text)
        _debug(f"inference_plan vision={needs_vision} memory={needs_memory}")

        if needs_vision:
            _debug("inference_layer vision_required")
            result = execute_tool("analyze_image", {"query": user_text})
            ok_value = result.get("ok") if isinstance(result, dict) else None
            keys = sorted(result.keys()) if isinstance(result, dict) else []
            _debug(f"vision_tool_result ok={ok_value} keys={keys}")
            if isinstance(result, dict) and result.get("image_path"):
                self._last_image_path = result["image_path"]
            if isinstance(result, dict) and result.get("analysis"):
                context_lines.append(f"Vision analysis: {result['analysis']}")
            else:
                error = ""
                if isinstance(result, dict):
                    error = result.get("error", "")
                context_lines.append(error or "Vision analysis unavailable.")
            tool_choice = "none"

        if needs_memory:
            _debug("inference_layer memory_required")
            result = execute_tool("search_memory", {"query": user_text, "k": 2})
            keys = sorted(result.keys()) if isinstance(result, dict) else []
            _debug(f"memory_tool_result keys={keys}")
            memory_line = ""
            if isinstance(result, dict):
                memory_line = _format_memory(result.get("results", []))
            if memory_line:
                context_lines.append(memory_line)
                tool_choice = "none"

        if context_lines:
            context = "\n".join(context_lines)
            user_text = f"{user_text}\n\nContext:\n{context}"

        return user_text, tool_choice

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
                if not self._mic_started:
                    self._mic_started = True
                    _debug("mic_stream_reading")
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
                    _debug(f"speech_started peak={peak:.3f}")
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
                    _debug(
                        f"audio_commit bytes={len(audio_bytes)} "
                        f"speech_seconds={self._speech_duration:.2f}"
                    )
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
        _debug("websocket_opened")
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio", "text"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": self.rate},
                        "turn_detection": {"type": "semantic_vad"},
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
                "instructions": (
                    f"{REALTIME_RESPONSE_STYLE}\n\n{REALTIME_TOOL_GUIDANCE}"
                ),
                "tools": TOOLS,
                "tool_choice": REALTIME_TOOL_CHOICE,
            },
        }
        self._send_event(session_update)
        _debug("session_update_sent")

        threading.Thread(target=self._mic_stream_loop, daemon=True).start()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        if self._handle_tool_event(data):
            return

        event_type = data.get("type")
        if not self._logged_first_event and event_type:
            _debug(f"realtime_event_first type={event_type}")
            self._logged_first_event = True
        if event_type == "error":
            _debug(f"realtime_error {data}")
        logger.debug("realtime_event type=%s", event_type)
        transcript = _extract_transcript(data)
        if transcript:
            self._pending_audio = None
            self._awaiting_transcript = False
            preview = transcript
            if len(preview) > 80:
                preview = preview[:77] + "..."
            _debug(f"transcript_received text={preview}")
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
            if assistant_text and not is_english(assistant_text):
                if not self._language_warning_sent:
                    _debug("assistant_non_english_response")
                    self._language_warning_sent = True
                    self._assistant_text = ""
                    self._response_in_flight = True
                    self._send_event(
                        {
                            "type": "response.create",
                            "response": {
                                "output_modalities": ["audio", "text"],
                                "instructions": (
                                    "Respond with: I can only communicate in "
                                    "English. Please repeat in English."
                                ),
                                "max_output_tokens": 40,
                            },
                        }
                    )
                return
            self._language_warning_sent = False
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
        _debug("audio_setup_start")
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
        _debug("audio_streams_started")

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
