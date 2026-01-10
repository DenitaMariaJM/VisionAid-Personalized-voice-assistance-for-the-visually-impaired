"""Main entry point for the realtime assistant."""

from .config import validate_config
from .logging_utils import configure_logging
from .realtime_client import run_realtime


def main():
    configure_logging()
    validate_config()
    run_realtime()


if __name__ == "__main__":
    main()
