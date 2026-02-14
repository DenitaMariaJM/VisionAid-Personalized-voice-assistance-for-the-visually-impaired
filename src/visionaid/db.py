"""SQLite persistence for assistant interactions."""

import logging
import os
from pathlib import Path
import sqlite3

_DEFAULT_DB_PATH = Path.cwd() / "assistant.db"
DB_NAME = os.getenv("VISIONAID_DB_PATH", str(_DEFAULT_DB_PATH))
logger = logging.getLogger(__name__)
DEFAULT_USER_ID = 1


def _ensure_column(conn, table_name, column_name, ddl_fragment):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cur.fetchall()]
    if column_name not in cols:
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl_fragment}")


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    logger.debug("init_db called from %s", __file__)

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
        intent TEXT,
        image_path TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )

    """)

    # Semantic memory
    c.execute("""
       CREATE TABLE IF NOT EXISTS semantic_memory (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       text TEXT NOT NULL,
       embedding BLOB NOT NULL,
       created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
        embedding BLOB NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    _ensure_column(conn, "semantic_memory", "created_at", "created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
    _ensure_column(conn, "user_facts", "created_at", "created_at DATETIME DEFAULT CURRENT_TIMESTAMP")

    conn.commit()
    conn.close()


def log_interaction(query, response, intent=None, image_path=None):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO interactions (query, response, intent, image_path)
            VALUES (?, ?, ?, ?)
            """,
            (query, response, intent, image_path),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("db_log_interaction_failed error=%s", exc)


def get_user_profile(user_id: int = DEFAULT_USER_ID) -> dict:
    default_profile = {
        "user_id": user_id,
        "vision_level": "blind",
        "response_style": "concise",
        "language": "english",
    }
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            """
            INSERT OR IGNORE INTO user_profile (user_id, vision_level, response_style, language)
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                default_profile["vision_level"],
                default_profile["response_style"],
                default_profile["language"],
            ),
        )
        conn.commit()
        c.execute(
            """
            SELECT user_id, vision_level, response_style, language
            FROM user_profile
            WHERE user_id = ?
            """,
            (user_id,),
        )
        row = c.fetchone()
        conn.close()
        if not row:
            return default_profile
        return {
            "user_id": row[0],
            "vision_level": row[1] or default_profile["vision_level"],
            "response_style": row[2] or default_profile["response_style"],
            "language": row[3] or default_profile["language"],
        }
    except Exception as exc:
        logger.warning("db_get_user_profile_failed error=%s", exc)
        return default_profile


def update_user_profile(
    *,
    user_id: int = DEFAULT_USER_ID,
    vision_level: str | None = None,
    response_style: str | None = None,
    language: str | None = None,
) -> dict:
    profile = get_user_profile(user_id=user_id)
    new_vision = vision_level if vision_level is not None else profile["vision_level"]
    new_style = response_style if response_style is not None else profile["response_style"]
    new_language = language if language is not None else profile["language"]
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO user_profile (user_id, vision_level, response_style, language)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                vision_level = excluded.vision_level,
                response_style = excluded.response_style,
                language = excluded.language
            """,
            (user_id, new_vision, new_style, new_language),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("db_update_user_profile_failed error=%s", exc)
    return get_user_profile(user_id=user_id)


def get_recent_interactions(limit: int = 5) -> list[dict]:
    if limit <= 0:
        return []
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute(
            """
            SELECT query, response, intent, image_path, timestamp
            FROM interactions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = c.fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("db_get_recent_interactions_failed error=%s", exc)
        return []

    items = []
    for query, response, intent, image_path, timestamp in rows:
        items.append(
            {
                "query": query or "",
                "response": response or "",
                "intent": intent or "",
                "image_path": image_path or "",
                "timestamp": timestamp or "",
            }
        )
    return items
