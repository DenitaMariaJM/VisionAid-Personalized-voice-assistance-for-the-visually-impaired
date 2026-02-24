"""Runtime access checks for hardware tools (Raspberry Pi compatible)."""

import glob
import os
import stat
import subprocess
import cv2
import sounddevice as sd

from .config import (
    USE_LIBCAMERA,
    AUDIO_INPUT_DEVICE,
    CAMERA_BACKEND,
    CAMERA_AUTO_PROBE,
    CAMERA_INDEX,
    CAMERA_PROBE_MAX,
    REALTIME_CHUNK_MS,
    REALTIME_SAMPLE_RATE,
)


# ==============================
# MICROPHONE CHECK (unchanged)
# ==============================

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


# ==============================
# CAMERA HELPERS
# ==============================

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


# ==============================
# CAMERA CHECK (Pi version)
# ==============================

def check_camera_access():
    """
    Dual-mode camera check:
    - Raspberry Pi → libcamera (rpicam-still)
    - Laptop/USB cam → OpenCV
    """

    device_path = f"/dev/video{CAMERA_INDEX}"

    # Permission diagnostics (useful mainly for USB cams)
    if os.path.exists(device_path) and not _device_readable(device_path):
        devices = ", ".join(_describe_device(p) for p in _video_device_paths()) or "none"
        return (
            False,
            "Camera permission denied. "
            f"Current={_describe_device(device_path)}; devices=[{devices}]. "
            "Fix: add your user to the 'video' group and re-login "
            "(e.g., `sudo usermod -aG video $USER`).",
        )

    # ============================
    # Raspberry Pi → libcamera
    # ============================
    if USE_LIBCAMERA:
        try:
            subprocess.run(
                ["rpicam-still", "-n", "--timeout", "200", "-o", "/dev/null"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True, "Camera access OK (libcamera)."

        except Exception:
            devices = ", ".join(_describe_device(p) for p in _video_device_paths()) or "none"
            backend = CAMERA_BACKEND or "libcamera"
            return (
                False,
                "Camera access failed (libcamera test failed). "
                f"backend={backend} camera_index={CAMERA_INDEX} devices=[{devices}]. "
                "Check ribbon cable or that `rpicam-still` works.",
            )

    # ============================
    # Laptop / USB cam → OpenCV
    # ============================
    else:
        cam = cv2.VideoCapture(CAMERA_INDEX)
        if not cam.isOpened():
            devices = ", ".join(_describe_device(p) for p in _video_device_paths()) or "none"
            return (
                False,
                "Camera access failed (OpenCV could not open device). "
                f"camera_index={CAMERA_INDEX} devices=[{devices}].",
            )

        cam.release()
        return True, "Camera access OK (OpenCV)."