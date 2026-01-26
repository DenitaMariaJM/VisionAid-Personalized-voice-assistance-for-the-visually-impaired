"""SQLite persistence for assistant interactions."""

import logging
import os
from pathlib import Path
import sqlite3

_DEFAULT_DB_PATH = Path.cwd() / "assistant.db"
DB_NAME = os.getenv("VISIONAID_DB_PATH", str(_DEFAULT_DB_PATH))
logger = logging.getLogger(__name__)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    print("[DB] init_db called from", __file__)

    # User profile
    c.execute("""
    CREATE TABLE IF NOT EXISTS user_profile (
        user_id INTEGER PRIMARY KEY,
        vision_level TEXT,
        response_style TEXT,
        language TEXT
    )
    """)

    # Interactions
    c.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        response TEXT,
        image_path TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Semantic memory
    c.execute("""
       CREATE TABLE IF NOT EXISTS semantic_memory (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       text TEXT NOT NULL,
       embedding BLOB NOT NULL
    )

   """)

  # episodic summary
    c.execute("""
       CREATE TABLE IF NOT EXISTS episodic_memory (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       day TEXT NOT NULL UNIQUE,
       summary TEXT NOT NULL,
       embedding BLOB NOT NULL
    )
    """)

  # user facts
    c.execute("""
    	CREATE TABLE IF NOT EXISTS user_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
    """)

    conn.commit()
    conn.close()


def log_interaction(query, response, image_path=None):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO interactions (query, response, image_path)
            VALUES (?, ?, ?)
            """,
            (query, response, image_path),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("db_log_interaction_failed error=%s", exc)
