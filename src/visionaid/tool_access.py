"""Runtime access checks for hardware tools."""

import time

import sounddevice as sd

from .config import (
    AUDIO_INPUT_DEVICE,
    CAMERA_AUTO_PROBE,
    CAMERA_INDEX,
    CAMERA_PROBE_MAX,
    REALTIME_CHUNK_MS,
    REALTIME_SAMPLE_RATE,
)
from .vision import try_capture_frame


def check_microphone_access():
    frames = int(REALTIME_SAMPLE_RATE * (REALTIME_CHUNK_MS / 1000.0))
    try:
        stream = sd.RawInputStream(
            samplerate=REALTIME_SAMPLE_RATE,
            blocksize=frames,
            dtype="int16",
            channels=1,
            device=AUDIO_INPUT_DEVICE,
        )
        stream.start()
        stream.read(frames)
        stream.stop()
        stream.close()
        return True, "Microphone access OK."
    except Exception as exc:
        return False, f"Microphone access failed: {exc}"


def check_camera_access():
    used_index, frame = try_capture_frame(CAMERA_INDEX)
    if frame is None and CAMERA_AUTO_PROBE:
        for idx in range(CAMERA_PROBE_MAX):
            if idx == CAMERA_INDEX:
                continue
            used_index, frame = try_capture_frame(idx)
            if frame is not None:
                break
    if frame is None:
        return False, "Camera access failed."
    if used_index != CAMERA_INDEX:
        return True, f"Camera access OK (auto-probed index {used_index})."
    return True, "Camera access OK."


def should_recheck(last_checked_at, interval_seconds):
    if interval_seconds <= 0:
        return False
    if last_checked_at is None:
        return True
    return (time.time() - last_checked_at) >= interval_seconds
