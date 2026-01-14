"""Main entry point for the non-realtime pipeline."""

from .logging_utils import configure_logging
from .pipeline import run_pipeline


def main():
    configure_logging()
    run_pipeline()


if __name__ == "__main__":
    main()
