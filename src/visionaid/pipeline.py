"""Non-realtime voice pipeline with action routing."""

import logging
import re

from openai import OpenAI

from .audio_io import record_utterance_wav_bytes
from .config import (
    ASSISTANT_MAX_TOKENS,
    ASSISTANT_MODEL,
    MEMORY_ENABLED,
    MEMORY_STORE_EVERY_TURN,
    MEMORY_TOP_K,
    TTS_ENABLED,
    validate_config,
)
from .db import get_recent_interactions, get_user_profile, log_interaction, update_user_profile
from .memory import (
    build_memory_entry,
    search_memory,
    search_user_facts,
    store_memory,
    store_user_fact,
)
from .stt_whisper import transcribe_audio
from .tool_access import check_camera_access, check_microphone_access
from .tts import speak
from .vision import analyze_image, capture_image
from .agent import plan_actions, classify_memory_type
from .episodic_retrieval import search_episodic_memory

logger = logging.getLogger(__name__)
client = OpenAI()


_SYSTEM_ASSISTANT = (
    "You are VisionAid, an assistive voice system for the visually impaired. "
    "Always respond ONLY in English. "
    "Be concise, direct, and practical. "
    "Answer the user's question using available context (vision analysis and/or memory). "
    "If memory context is provided, use it and do not claim you have no memory. "
    "If the user asks for navigation help, prioritize safety and direction guidance (front/left/right). "
    "If the user asks to identify/describe something, describe the key, useful details. "
    "If clarification is needed, ask a short follow-up question."
)

_VISION_UNAVAILABLE = (
    "I cannot safely describe your current surroundings because I could not access a usable camera image. "
    "Please adjust the camera position and ask again."
)


def _is_location_history_query(text: str) -> bool:
    lowered = (text or "").lower()
    triggers = (
        "where am i",
        "where is this",
        "what place is this",
        "this place",
        "this location",
        "around me",
        "been here before",
        "have i been here",
        "visited this place",
        "visited here",
    )
    return any(token in lowered for token in triggers)


def _is_unhelpful_memory_text(text: str) -> bool:
    lowered = (text or "").lower()
    blockers = (
        "i don't have memory of past interactions",
        "i dont have memory of past interactions",
        "i do not have memory of past interactions",
        "i need more context to help you",
        "could you describe your current location or surroundings",
    )
    return any(token in lowered for token in blockers)


def _filter_memories(memories: list[str]) -> list[str]:
    cleaned = []
    for item in memories:
        if not item:
            continue
        if _is_unhelpful_memory_text(item):
            continue
        cleaned.append(item)
    return cleaned


def _profile_system_prompt(profile: dict) -> str:
    if not profile:
        return ""

    parts = []
    vision_level = (profile.get("vision_level") or "").strip().lower()
    response_style = (profile.get("response_style") or "").strip().lower()
    language = (profile.get("language") or "").strip().lower()

    if vision_level == "blind":
        parts.append("User profile: fully blind. Prioritize spatial orientation and obstacle cues.")
    elif vision_level in ("low_vision", "partially_sighted"):
        parts.append("User profile: low vision. Include orientation cues and practical visual clarifications.")

    if response_style == "step_by_step":
        parts.append("Response style: give short step-by-step guidance.")
    elif response_style == "detailed":
        parts.append("Response style: include slightly more detail while staying practical.")
    else:
        parts.append("Response style: concise.")

    if language and language != "english":
        parts.append(f"Preferred language is {language}, but currently respond in English.")

    return " ".join(parts).strip()


def _extract_profile_updates(user_text: str) -> tuple[dict, str | None]:
    if not user_text:
        return {}, None

    lowered = user_text.lower()
    updates = {}
    notes = []

    if "i am blind" in lowered or "fully blind" in lowered or "completely blind" in lowered:
        updates["vision_level"] = "blind"
        notes.append("vision level set to blind")
    elif "low vision" in lowered or "partially sighted" in lowered or "partially blind" in lowered:
        updates["vision_level"] = "low_vision"
        notes.append("vision level set to low vision")

    if "step by step" in lowered or "one step at a time" in lowered:
        updates["response_style"] = "step_by_step"
        notes.append("response style set to step-by-step")
    elif "more detail" in lowered or "be detailed" in lowered or "detailed responses" in lowered:
        updates["response_style"] = "detailed"
        notes.append("response style set to detailed")
    elif "be concise" in lowered or "keep it short" in lowered or "be brief" in lowered:
        updates["response_style"] = "concise"
        notes.append("response style set to concise")

    language_match = re.search(r"(?:respond|speak)\s+in\s+([a-z]+)", lowered)
    if language_match:
        requested_language = language_match.group(1).strip()
        if requested_language == "english":
            updates["language"] = "english"
            notes.append("language set to english")
        else:
            notes.append(f"requested language '{requested_language}' noted (English currently enforced)")

    if not notes:
        return {}, None
    return updates, "Profile updated: " + "; ".join(notes) + "."


def _extract_user_facts(user_text: str) -> list[str]:
    if not user_text:
        return []

    def _clean(value: str) -> str:
        return (value or "").strip().strip(" .,!?:;\"'")

    facts = []
    text = user_text.strip()

    patterns = [
        (r"\bmy name is ([a-zA-Z][a-zA-Z '\-]{0,40})", "User's name is {value}."),
        (r"\bi am allergic to ([a-zA-Z0-9 ,'\-]{2,80})", "User is allergic to {value}."),
        (r"\bi prefer ([a-zA-Z0-9 ,'\-]{2,80})", "User prefers {value}."),
        (r"\bi like ([a-zA-Z0-9 ,'\-]{2,80})", "User likes {value}."),
        (r"\bi live in ([a-zA-Z0-9 ,'\-]{2,80})", "User lives in {value}."),
        (r"\bmy favorite ([a-zA-Z]{3,20}) is ([a-zA-Z0-9 ,'\-]{2,80})", "User's favorite {slot} is {value}."),
    ]

    for pattern, template in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            if "{slot}" in template:
                slot = _clean(match.group(1)).lower()
                value = _clean(match.group(2))
                if slot and value:
                    facts.append(template.format(slot=slot, value=value))
            else:
                value = _clean(match.group(1))
                if value:
                    facts.append(template.format(value=value))

    deduped = []
    seen = set()
    for fact in facts:
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _final_response(user_text: str, vision_analysis: str, memories: list[str], profile: dict) -> str:
    parts = []
    if memories:
        parts.append("Relevant memory:\n" + "\n---\n".join(memories))
    if vision_analysis:
        parts.append("Image analysis:\n" + vision_analysis)
    context = "\n\n".join(parts).strip()

    messages = [{"role": "system", "content": _SYSTEM_ASSISTANT}]
    profile_prompt = _profile_system_prompt(profile)
    if profile_prompt:
        messages.append({"role": "system", "content": profile_prompt})
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model=ASSISTANT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=ASSISTANT_MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def _recent_interaction_memories(limit: int = 5) -> list[str]:
    items = get_recent_interactions(limit=limit)
    if not items:
        return []
    results = []
    for item in items:
        response = (item.get("response") or "").strip()
        query = (item.get("query") or "").strip()
        if not response:
            continue
        if _is_unhelpful_memory_text(response):
            continue
        stamp = item.get("timestamp") or "unknown-time"
        results.append(f"[interaction {stamp}] User: {query} | Assistant: {response}")
    return results


def run_pipeline():
    validate_config()
    user_profile = get_user_profile()
   

    mic_ok, mic_msg = check_microphone_access()
    cam_ok, cam_msg = check_camera_access()
    logger.info("startup_check mic=%s camera=%s", mic_msg, cam_msg)
    if not mic_ok:
        logger.warning("microphone_not_ready")
    if not cam_ok:
        logger.warning("camera_not_ready")

    logger.info("ready Speak now. Press Ctrl+C to exit.")
    try:
        while True:
            wav_bytes = record_utterance_wav_bytes()
            if not wav_bytes:
                continue

            user_text = transcribe_audio(wav_bytes)
            if not user_text:
                continue
            logger.info("user_text %s", user_text)

            if MEMORY_ENABLED:
                user_facts = _extract_user_facts(user_text)
                for fact in user_facts:
                    store_user_fact(fact)

            updates, profile_ack = _extract_profile_updates(user_text)
            if updates or profile_ack:
                if updates:
                    user_profile = update_user_profile(**updates)
                assistant_text = profile_ack or "Profile updated."
                print(assistant_text, flush=True)
                if TTS_ENABLED:
                    speak(assistant_text)
                log_interaction(user_text, assistant_text, intent="profile", image_path=None)
                continue

            plan = plan_actions(user_text)
            logger.info("action_plan intent=%s", plan.intent)

            image_path = None
            vision_analysis = ""
            vision_error = None
            if plan.intent in ("vision", "both"):
                ok, msg = check_camera_access()
                if not ok:
                    logger.warning("camera_access_failed %s", msg)
                    vision_error = msg
                else:
                    image_path = capture_image()
                    if image_path:
                        vision_prompt = plan.vision_prompt or user_text
                        vision_analysis = analyze_image(image_path, vision_prompt)
                        if vision_analysis:
                            preview = vision_analysis if len(vision_analysis) <= 140 else (vision_analysis[:137] + "...")
                            logger.info("vision_analysis chars=%s text=%s", len(vision_analysis), preview)
                        else:
                            logger.warning("vision_analysis_empty image_path=%s", image_path)
                            vision_error = "captured image could not be analyzed"
                    else:
                        vision_error = "camera capture returned no image"

            memories: list[str] = []

            should_query_memory = (
                MEMORY_ENABLED
                and (
                    plan.intent in ("memory", "both")
                    or _is_location_history_query(user_text)
                    or (plan.intent == "vision" and bool(vision_error))
                )
            )

            if should_query_memory:
                try:
                    memory_query = plan.memory_query or user_text
                    memory_type = classify_memory_type(memory_query)
                    facts = search_user_facts(memory_query, k=2)

                    if memory_type == "episodic":
                        memories = search_episodic_memory(memory_query, k=1)
                        if not memories:
                            # Fallback to semantic memory for same-session recall.
                            memories = search_memory(memory_query, k=MEMORY_TOP_K)
                    else:
                        memories = search_memory(memory_query, k=MEMORY_TOP_K)

                    if facts:
                        memories = facts + memories

                    memories = _filter_memories(memories)

                    if _is_location_history_query(user_text):
                        recent_location_context = _recent_interaction_memories(limit=8)
                        if recent_location_context:
                            memories = recent_location_context + memories

                    # Final fallback: raw recent interaction logs.
                    if not memories:
                        memories = _recent_interaction_memories(limit=5)
                except Exception as exc:
                    logger.warning("memory_retrieval_failed error=%s", exc)
                    memories = []


            # Token/perf: if the vision model already answered the question and
            # no memory context is needed, use it directly.
            if plan.intent in ("vision", "both") and not vision_analysis:
                if memories:
                    assistant_text = (
                        f"{_VISION_UNAVAILABLE} "
                        f"From memory: {memories[0]}"
                    )
                else:
                    assistant_text = _VISION_UNAVAILABLE
                if vision_error:
                    logger.info("vision_unavailable_reason %s", vision_error)
            elif plan.intent == "vision" and vision_analysis and not memories:
                assistant_text = vision_analysis
            else:
                assistant_text = _final_response(user_text, vision_analysis, memories, user_profile)
            if assistant_text:
                print(assistant_text, flush=True)
                if TTS_ENABLED:
                    speak(assistant_text)

            if user_text and assistant_text:
                if MEMORY_ENABLED:
                    should_store = MEMORY_STORE_EVERY_TURN or (plan.intent in ("memory", "both"))
                    lowered = user_text.lower()
                    if any(phrase in lowered for phrase in ("remember this", "save this", "remember that")):
                        should_store = True
                    if should_store:
                        try:
                            if _is_unhelpful_memory_text(assistant_text):
                                store_memory(build_memory_entry(user_text, None))
                            else:
                                store_memory(build_memory_entry(user_text, assistant_text))
                        except Exception as exc:
                            logger.warning("memory_store_failed error=%s", exc)
                log_interaction(
                    user_text,
                    assistant_text,
                    intent=plan.intent,
                    image_path=image_path
                )

    except KeyboardInterrupt:
        logger.info("shutdown")
