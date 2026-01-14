"""Runtime access checks for hardware tools."""

import glob
import os
import stat
import time

import sounddevice as sd

from .config import (
    AUDIO_INPUT_DEVICE,
    CAMERA_BACKEND,
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


def _format_mode(mode: int) -> str:
    return stat.filemode(mode)


def _describe_device(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{path} mode={_format_mode(st.st_mode)} uid={st.st_uid} gid={st.st_gid}"
    except Exception:
        return f"{path} (stat failed)"


def _device_readable(path: str) -> bool:
    return os.path.exists(path) and os.access(path, os.R_OK)


def _video_device_paths():
    return sorted(glob.glob("/dev/video*"))


def check_camera_access():
    device_path = f"/dev/video{CAMERA_INDEX}"
    if os.path.exists(device_path) and not _device_readable(device_path):
        devices = ", ".join(_describe_device(p) for p in _video_device_paths()) or "none"
        return (
            False,
            "Camera permission denied. "
            f"Current={_describe_device(device_path)}; devices=[{devices}]. "
            "Fix: add your user to the 'video' group and re-login "
            "(e.g., `sudo usermod -aG video $USER`).",
        )

    used_index, frame = try_capture_frame(CAMERA_INDEX)
    if frame is None and CAMERA_AUTO_PROBE:
        for idx in range(CAMERA_PROBE_MAX):
            if idx == CAMERA_INDEX:
                continue
            used_index, frame = try_capture_frame(idx)
            if frame is not None:
                break
    if frame is None:
        devices = ", ".join(_describe_device(p) for p in _video_device_paths()) or "none"
        backend = CAMERA_BACKEND or "default"
        return (
            False,
            "Camera access failed (OpenCV could not read a frame). "
            f"backend={backend} camera_index={CAMERA_INDEX} devices=[{devices}]. "
            "If permissions are OK, try setting `CAMERA_BACKEND = None` or "
            "changing `CAMERA_INDEX` in `src/visionaid/config.py`.",
        )
    if used_index != CAMERA_INDEX:
        return True, f"Camera access OK (auto-probed index {used_index})."
    return True, "Camera access OK."


def should_recheck(last_checked_at, interval_seconds):
    if interval_seconds <= 0:
        return False
    if last_checked_at is None:
        return True
    return (time.time() - last_checked_at) >= interval_seconds
