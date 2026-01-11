"""Hardware access checks for microphone and camera."""

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from visionaid.logging_utils import configure_logging
from visionaid.tool_access import check_camera_access, check_microphone_access

logger = logging.getLogger(__name__)


def check_microphone():
    return check_microphone_access()


def check_camera():
    return check_camera_access()


def run_checks():
    configure_logging()
    ok_mic, mic_msg = check_microphone()
    ok_cam, cam_msg = check_camera()
    logger.info(mic_msg)
    logger.info(cam_msg)
    return ok_mic and ok_cam


if __name__ == "__main__":
    run_checks()
