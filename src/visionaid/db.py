"""SQLite persistence for assistant interactions."""

import logging
import sqlite3

DB_NAME = "assistant.db"
logger = logging.getLogger(__name__)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

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
    CREATE TABLE IF NOT EXISTS memory (
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
