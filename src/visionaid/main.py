"""Main entry point for the realtime assistant."""

import logging

from .config import validate_config
from .logging_utils import configure_logging
from .realtime_client import run_realtime
from .tool_access import check_camera_access, check_microphone_access

logger = logging.getLogger(__name__)


def main():
    configure_logging()
    validate_config()
    mic_ok, mic_msg = check_microphone_access()
    cam_ok, cam_msg = check_camera_access()
    logger.info("startup_check mic=%s camera=%s", mic_msg, cam_msg)
    if not mic_ok:
        logger.warning("microphone_not_ready")
    if not cam_ok:
        logger.warning("camera_not_ready")
    run_realtime()


if __name__ == "__main__":
    main()
