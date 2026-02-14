"""
Day-wise episodic memory summarization for VisionAid.

This module:
- Detects unsummarized past days from interactions
- Builds blind-assist–aware daily summaries
- Stores exactly one summary per day in episodic_memory
"""

import sqlite3
import logging

from openai import OpenAI

from .db import DB_NAME
from .config import EPISODIC_SUMMARY_MAX_CHARS
from .memory import get_embedding  # reuse your existing embedding logic

logger = logging.getLogger(__name__)
client = OpenAI()


# ---------------------------------------------------------
# SQL HELPERS
# ---------------------------------------------------------


def _get_unsummarized_days():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        SELECT DISTINCT DATE(timestamp) AS day
        FROM interactions
        WHERE DATE(timestamp) <= DATE('now', '-1 day')
          AND day NOT IN (
              SELECT day FROM episodic_memory
          )
        ORDER BY day ASC
    """)

    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]



def _get_interactions_for_day(day):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        SELECT query, response, image_path
        FROM interactions
        WHERE DATE(timestamp) = ?
        ORDER BY timestamp ASC
    """, (day,))

    rows = c.fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------
# INTERACTION FILTERING (CRITICAL)
# ---------------------------------------------------------

def _filter_interactions(rows):
    """
    Keep anything that produced visual/environmental information.
    Episodic memory must be over-inclusive.
    """
    filtered = []
    visual_query_keywords = (
        "left", "right", "front", "ahead", "obstacle", "stairs", "door",
        "path", "around", "room", "where am i", "what do you see", "describe",
    )
    hazard_response_keywords = (
        "stairs", "step", "obstacle", "blocked", "hazard", "edge", "wet",
        "narrow", "crowded", "traffic", "vehicle", "bike",
    )

    for query, response, image_path in rows:
        q = (query or "").strip().lower()
        r = (response or "").strip()
        r_lower = r.lower()
        has_visual_query = any(token in q for token in visual_query_keywords)
        has_hazard_signal = any(token in r_lower for token in hazard_response_keywords)
        has_meaningful_response = len(r) > 20 and ("left" in r_lower or "right" in r_lower or "front" in r_lower)

        # Keep all camera-grounded turns and short hazard/navigation alerts.
        if image_path or has_visual_query or has_hazard_signal or has_meaningful_response:
            filtered.append((query, response, image_path))

    return filtered



# ---------------------------------------------------------
# SUMMARY GENERATION
# ---------------------------------------------------------
def _build_summary_prompt(day, interactions):
    """
    Build a feature-rich episodic memory prompt for blind assistive recall.
    """
    observations = []

    for q, r, img in interactions:
        if r:
            observations.append(f"- {r[:300]}")

    content = "\n".join(observations)

    return [
        {
            "role": "system",
            "content": (
                "You are creating a long-term episodic memory for a blind user "
                "using a vision-assisted wearable device. Your goal is NOT to "
                "summarize briefly, but to extract and preserve as many visual "
                "and environmental features as possible so that future questions "
                "about this scene can be answered without re-capturing an image.\n\n"
                "Focus on:\n"
                "- Objects present (furniture, doors, stairs, screens, people if visible)\n"
                "- Spatial layout (left/right, near/far, open/closed spaces)\n"
                "- Lighting conditions (bright, dim, dark, artificial, natural)\n"
                "- Obstacles or hazards (stairs, narrow paths, clutter)\n"
                "- Indoor vs outdoor cues\n"
                "- Familiarity cues (home-like, office-like, public space)\n\n"
                "Do NOT include dialogue, assumptions, emotions, or guesses.\n"
                "Write in clear, factual sentences optimized for future recall."
            )
        },
        {
            "role": "user",
            "content": (
                f"Date: {day}\n\n"
                f"Vision observations from the day:\n{content}"
            )
        }
    ]



def _summarize_day(day, interactions):
    if not interactions:
        return None

    messages = _build_summary_prompt(day, interactions)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=200,
    )

    summary = response.choices[0].message.content.strip()
    return summary[:EPISODIC_SUMMARY_MAX_CHARS]



# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------

def _store_episode(day, summary):
    embedding = get_embedding(summary)
    if embedding is None:
        return False

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        INSERT INTO episodic_memory (day, summary, embedding)
        VALUES (?, ?, ?)
    """, (
        day,
        summary,
        embedding.tobytes()
    ))

    conn.commit()
    conn.close()
    return True


# ---------------------------------------------------------
# PUBLIC ENTRYPOINT
# ---------------------------------------------------------

def summarize_pending_days():
    days = _get_unsummarized_days()
    logger.info("episodic_summary: days to summarize = %s", days)

    if not days:
        logger.info("episodic_summary: nothing to summarize")
        return

    for day in days:
        try:
            rows = _get_interactions_for_day(day)
            logger.info("episodic_summary: %s raw rows = %d", day, len(rows))

            filtered = _filter_interactions(rows)
            logger.info("episodic_summary: %s filtered rows = %d", day, len(filtered))

            summary = _summarize_day(day, filtered)
            if not summary:
                logger.error("episodic_summary: %s FAILED to summarize", day)
                continue

            stored = _store_episode(day, summary)
            if stored:
                logger.info("episodic_summary: STORED summary for %s", day)
            else:
                logger.error("episodic_summary: %s FAILED to store embedding", day)
        except Exception as exc:
            logger.exception("episodic_summary: %s failed error=%s", day, exc)
