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

def main():
    configure_logging()

    # 1️⃣ Ensure DB & tables exist
    init_db()

    # 2️⃣ Load semantic memory
    load_memory()

    # 3️⃣ 🔴 RUN EPISODIC SUMMARY HERE
    summarize_pending_days()

    # 4️⃣ Start live assistant (blocking)
    run_pipeline()

if __name__ == "__main__":
    main()
