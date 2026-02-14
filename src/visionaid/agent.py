"""LLM-based action planning for VisionAid."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional
from .db import DB_NAME

from openai import OpenAI

from .config import AGENT_LLM_FALLBACK_ONLY, AGENT_MAX_TOKENS, AGENT_MODEL

logger = logging.getLogger(__name__)
client = OpenAI()


@dataclass(frozen=True)
class ActionPlan:
    intent: str  # "vision" | "memory" | "both" | "chat"
    vision_prompt: Optional[str] = None
    memory_query: Optional[str] = None


_SYSTEM = (
    "You are an action-planning agent for a voice assistant for visually impaired users.\n"
    "Given the user's utterance, decide which actions are needed before answering:\n"
    "- vision: needs a camera image (environment, obstacles, navigation, what's in hand/wearing).\n"
    "- memory: needs searching stored memory (remind, what did I ask, earlier, last time).\n"
    "- both: needs both vision and memory.\n"
    "- chat: neither needed.\n"
    "Return ONLY valid JSON with keys: intent, vision_prompt, memory_query.\n"
    "intent must be one of: vision, memory, both, chat.\n"
    "If vision is needed, set vision_prompt to a short prompt for an image model.\n"
    "If memory is needed, set memory_query to a short search query.\n"
)


def _fallback_plan(user_text: str) -> ActionPlan:
    lowered = (user_text or "").lower()
    vision_triggers = (
        "in front",
        "where am i",
        "what am i seeing",
        "what do i see",
        "what can you see",
        "what do you see",
        "describe what you see",
        "describe my surroundings",
        "what's around me",
        "what is around me",
        "ahead",
        "around",
        "surround",
        "surroundings",
        "left",
        "right",
        "obstacle",
        "stairs",
        "door",
        "path",
        "what is this",
        "what's this",
        "in my hand",
        "holding",
        "wearing",
        "outfit",
        "shirt",
        "camera",
        "capture",
        "take a picture",
        "take photo",
        "snap",
        "see",
        "look",
    )
    memory_triggers = ("remind", "earlier", "last time", "before", "what did i ask", "memory")
    location_history_triggers = (
        "where am i",
        "where is this",
        "what place is this",
        "this place",
        "this location",
        "been here before",
        "have i been here",
        "visited this place",
        "visited here",
    )
    needs_vision = any(t in lowered for t in vision_triggers)
    needs_memory = any(t in lowered for t in memory_triggers)
    if any(t in lowered for t in location_history_triggers):
        needs_memory = True
    if needs_vision and needs_memory:
        return ActionPlan(intent="both", vision_prompt=user_text, memory_query=user_text)
    if needs_vision:
        return ActionPlan(intent="vision", vision_prompt=user_text)
    if needs_memory:
        return ActionPlan(intent="memory", memory_query=user_text)
    return ActionPlan(intent="chat")

def plan_actions(user_text: str) -> ActionPlan:
    """
    Context-aware, low-latency action planner.
    """

    if not user_text:
        return ActionPlan(intent="chat")

    lowered = user_text.lower()

    # 1. FAST HEURISTIC
    heuristic = _fallback_plan(user_text)

    # 2. FOLLOW-UP / CONTINUATION
    followup_tokens = ("it", "this", "that", "there", "again", "same")
    is_followup = (
        len(lowered.split()) <= 6
        and any(t in lowered.split() for t in followup_tokens)
    )

    if heuristic.intent == "chat" and is_followup:
        last_intent = get_last_intent_from_logs()
        if last_intent in ("vision", "memory", "both"):
            return ActionPlan(
                intent=last_intent,
                vision_prompt=user_text if last_intent in ("vision", "both") else None,
                memory_query=user_text if last_intent in ("memory", "both") else None,
            )

    # 3. EPISODIC MEMORY (PREVIOUS DAYS)
    episodic_triggers = (
        "yesterday", "earlier", "last time", "before", "previously"
    )

    if heuristic.intent == "chat" and any(t in lowered for t in episodic_triggers):
        if episodic_memory_exists():
            return ActionPlan(intent="memory", memory_query=user_text)

    # 4. CONFIDENT HEURISTIC
    if heuristic.intent in ("vision", "memory", "both"):
        return heuristic

    # 5. OPTIONAL LLM FALLBACK
    if AGENT_LLM_FALLBACK_ONLY:
        return heuristic

    try:
        resp = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=AGENT_MAX_TOKENS,
        )

        data = json.loads(resp.choices[0].message.content or "{}")
        intent = data.get("intent", "chat")

        if intent not in ("vision", "memory", "both", "chat"):
            intent = "chat"

        return ActionPlan(
            intent=intent,
            vision_prompt=user_text if intent in ("vision", "both") else None,
            memory_query=user_text if intent in ("memory", "both") else None,
        )

    except Exception as exc:
        logger.warning("action_planning_failed error=%s", exc)
        return heuristic
    
def get_last_intent_from_logs() -> Optional[str]:
    """
    Fetch the last action intent from interaction logs.
    Metadata-only, very fast.
    """
    import sqlite3

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        SELECT intent
        FROM interactions
        WHERE intent IS NOT NULL
        ORDER BY timestamp DESC, id DESC
        LIMIT 1
    """)

    row = c.fetchone()
    conn.close()

    return row[0] if row else None

def episodic_memory_exists() -> bool:
    """
    Check if episodic memory has at least one stored day.
    """
    import sqlite3

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT 1 FROM episodic_memory LIMIT 1")
    exists = c.fetchone() is not None

    conn.close()
    return exists
def classify_memory_type(query: str) -> str:
    """
    Decide whether the query requires episodic or semantic memory.
    """
    lowered = query.lower()

    episodic_keywords = (
        "yesterday", "earlier", "last time", "before",
        "previously", "place", "room", "around me",
        "environment", "location", "scene",
        "where am i", "where is this", "been here", "visited"
    )

    semantic_keywords = (
        "what did i say", "what did i ask",
        "did i tell you", "remember that i",
        "my name", "preference", "allergic"
    )

    if any(k in lowered for k in episodic_keywords):
        return "episodic"

    if any(k in lowered for k in semantic_keywords):
        return "semantic"

    # Default to semantic so general "remember/recall" queries are query-based.
    return "semantic"
