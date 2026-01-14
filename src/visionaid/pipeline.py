"""Non-realtime voice pipeline with action routing."""

import logging

from openai import OpenAI

from .agent import plan_actions
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
from .db import init_db, log_interaction
from .memory import build_memory_entry, load_memory, search_memory, store_memory
from .tool_access import check_camera_access, check_microphone_access
from .tts import speak
from .vision import analyze_image, capture_image

logger = logging.getLogger(__name__)
client = OpenAI()


_SYSTEM_ASSISTANT = (
    "You are VisionAid, an assistive voice system for the visually impaired. "
    "Always respond ONLY in English. "
    "Be concise, direct, and practical. "
    "Answer the user's question using available context (vision analysis and/or memory). "
    "If the user asks for navigation help, prioritize safety and direction guidance (front/left/right). "
    "If the user asks to identify/describe something, describe the key, useful details. "
    "If clarification is needed, ask a short follow-up question."
)


def _final_response(user_text: str, vision_analysis: str, memories: list[str]) -> str:
    parts = []
    if memories:
        parts.append("Relevant memory:\n" + "\n---\n".join(memories))
    if vision_analysis:
        parts.append("Image analysis:\n" + vision_analysis)
    context = "\n\n".join(parts).strip()

    messages = [{"role": "system", "content": _SYSTEM_ASSISTANT}]
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


def run_pipeline():
    validate_config()
    init_db()
    load_memory()

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

            from .stt_whisper import transcribe_audio

            user_text = transcribe_audio(wav_bytes)
            if not user_text:
                continue
            logger.info("user_text %s", user_text)

            plan = plan_actions(user_text)
            logger.info("action_plan intent=%s", plan.intent)

            image_path = None
            vision_analysis = ""
            if plan.intent in ("vision", "both"):
                ok, msg = check_camera_access()
                if not ok:
                    logger.warning("camera_access_failed %s", msg)
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

            memories: list[str] = []
            if MEMORY_ENABLED and plan.intent in ("memory", "both"):
                memories = search_memory(plan.memory_query or user_text, k=MEMORY_TOP_K)

            # Token/perf: if the vision model already answered the question and
            # no memory context is needed, use it directly.
            if plan.intent == "vision" and vision_analysis and not memories:
                assistant_text = vision_analysis
            else:
                assistant_text = _final_response(user_text, vision_analysis, memories)
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
                        store_memory(build_memory_entry(user_text, assistant_text))
                log_interaction(user_text, assistant_text, image_path=image_path)
    except KeyboardInterrupt:
        logger.info("shutdown")
