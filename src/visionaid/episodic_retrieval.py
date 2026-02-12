import sqlite3
import numpy as np
from .db import DB_NAME
from .memory import get_embedding

def search_episodic_memory(query: str, k: int = 1) -> list[str]:
    """
    Return most recent episodic summaries.
    Similarity matching is unreliable for recall/comparison queries.
    """

    import sqlite3
    from .db import DB_NAME

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        SELECT summary
        FROM episodic_memory
        ORDER BY day DESC
        LIMIT ?
    """, (k,))

    rows = c.fetchall()
    conn.close()

    return [row[0] for row in rows] if rows else []
