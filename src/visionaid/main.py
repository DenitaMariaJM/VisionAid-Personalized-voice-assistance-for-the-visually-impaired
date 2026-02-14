"""
Main entry point for the VisionAid non-realtime pipeline.

Responsibilities:
- One-time system initialization
- Memory consolidation
- Start the live assistive pipeline
"""

from .logging_utils import configure_logging
from .pipeline import run_pipeline
from .db import init_db
from .memory import load_memory
from .episodic_summary import summarize_pending_days
import logging

logger = logging.getLogger(__name__)

def main():
    configure_logging()

    # 1️⃣ Ensure DB & tables exist
    init_db()

    # 2️⃣ Load semantic memory
    load_memory()

    # 3️⃣ Run episodic summarization, but never block startup on failure.
    try:
        summarize_pending_days()
    except Exception as exc:
        logger.exception("episodic_summary_failed error=%s", exc)

    # 4️⃣ Start live assistant (blocking)
    run_pipeline()

if __name__ == "__main__":
    main()
