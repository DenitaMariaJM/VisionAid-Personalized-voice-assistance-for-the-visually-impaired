import sqlite3

import numpy as np

from .db import DB_NAME
from .memory import get_embedding

def search_episodic_memory(query: str, k: int = 1) -> list[str]:
    """
    Return episodic summaries ranked by semantic similarity.
    Falls back to most recent summaries if embeddings are unavailable.
    """
    if k <= 0:
        return []

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT day, summary, embedding
        FROM episodic_memory
        ORDER BY day DESC
    """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        return []

    query_text = (query or "").strip()
    if not query_text:
        return [f"[episodic {day}] {summary}" for day, summary, _ in rows[:k]]

    q_vec = get_embedding(query_text)
    if q_vec is None:
        return [f"[episodic {day}] {summary}" for day, summary, _ in rows[:k]]
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0:
        return [f"[episodic {day}] {summary}" for day, summary, _ in rows[:k]]

    scored: list[tuple[float, str, str]] = []
    for day, summary, blob in rows:
        if not blob or not summary:
            continue
        try:
            vec = np.frombuffer(blob, dtype="float32")
            if vec.size != q_vec.size:
                continue
            denom = float(np.linalg.norm(vec) * q_norm)
            if denom == 0:
                continue
            similarity = float(np.dot(vec, q_vec) / denom)
            scored.append((similarity, day, summary))
        except Exception:
            continue

    if not scored:
        return [f"[episodic {day}] {summary}" for day, summary, _ in rows[:k]]

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [f"[episodic {day}] {summary}" for _, day, summary in scored[:k]]
