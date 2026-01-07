import sqlite3

DB_NAME = "assistant.db"

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

    conn.commit()
    conn.close()
