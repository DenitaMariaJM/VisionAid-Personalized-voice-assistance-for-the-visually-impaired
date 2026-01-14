"""LLM-based action planning for VisionAid."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

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
        "see",
        "look",
    )
    memory_triggers = ("remind", "earlier", "last time", "before", "what did i ask", "memory")
    needs_vision = any(t in lowered for t in vision_triggers)
    needs_memory = any(t in lowered for t in memory_triggers)
    if needs_vision and needs_memory:
        return ActionPlan(intent="both", vision_prompt=user_text, memory_query=user_text)
    if needs_vision:
        return ActionPlan(intent="vision", vision_prompt=user_text)
    if needs_memory:
        return ActionPlan(intent="memory", memory_query=user_text)
    return ActionPlan(intent="chat")


def plan_actions(user_text: str) -> ActionPlan:
    if not user_text:
        return ActionPlan(intent="chat")
    heuristic = _fallback_plan(user_text)
    if heuristic.intent != "chat" and AGENT_LLM_FALLBACK_ONLY:
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
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        intent = str(data.get("intent", "")).strip().lower()
        if intent not in ("vision", "memory", "both", "chat"):
            raise ValueError("bad intent")
        vision_prompt = data.get("vision_prompt")
        memory_query = data.get("memory_query")
        return ActionPlan(
            intent=intent,
            vision_prompt=(vision_prompt or None),
            memory_query=(memory_query or None),
        )
    except Exception as exc:
        logger.warning("action_planning_failed error=%s", exc)
        return heuristic
