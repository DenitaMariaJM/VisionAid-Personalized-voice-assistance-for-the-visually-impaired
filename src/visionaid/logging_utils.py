"""Logging configuration helpers."""

import logging
import os
import sys


def configure_logging():
    level_name = os.getenv("VISIONAID_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
