"""Build memory + vision context for assistant responses."""

from .config import (
    CONTEXT_MAX_CHARS,
    MEMORY_ENABLED,
    MEMORY_MIN_CHARS,
    MEMORY_SNIPPET_CHARS,
    MEMORY_TOP_K,
    VISION_ENABLED,
    VISION_SNIPPET_CHARS,
)
from .memory import search_memory
from .vision import analyze_image, capture_image


def _needs_vision(text):
    triggers = {
        "see",
        "look",
        "front",
        "left",
        "right",
        "ahead",
        "around",
        "in front",
        "obstacle",
        "camera",
        "image",
        "photo",
        "picture",
        "what is",
        "describe",
    }
    lowered = text.lower()
    return any(token in lowered for token in triggers)


def _truncate(text, max_chars):
    if not text:
        return ""
    trimmed = text.strip()
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[:max_chars].rstrip() + "..."


def build_context(user_text):
    """
    Returns (extra_context, image_path) based on memory + vision signals.
    """
    context_parts = []
    image_path = None

    if MEMORY_ENABLED and len(user_text.strip()) >= MEMORY_MIN_CHARS:
        memories = search_memory(user_text, k=MEMORY_TOP_K)
        if memories:
            trimmed_memories = [
                _truncate(item, MEMORY_SNIPPET_CHARS) for item in memories
            ]
            context_parts.append(
                "Relevant memory:\n- " + "\n- ".join(trimmed_memories)
            )

    if VISION_ENABLED and _needs_vision(user_text):
        image_path = capture_image()
        if image_path:
            vision_text = analyze_image(image_path, user_text)
            if vision_text:
                context_parts.append(
                    f"Vision analysis: {_truncate(vision_text, VISION_SNIPPET_CHARS)}"
                )
            else:
                context_parts.append("Vision analysis failed.")
        else:
            context_parts.append("Image capture failed.")

    extra_context = "\n\n".join(context_parts)
    extra_context = _truncate(extra_context, CONTEXT_MAX_CHARS)
    return extra_context, image_path
