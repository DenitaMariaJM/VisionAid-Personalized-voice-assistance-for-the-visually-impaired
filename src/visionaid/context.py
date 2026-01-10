from .config import MEMORY_ENABLED, MEMORY_TOP_K, VISION_ENABLED
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


def build_context(user_text):
    """
    Returns (extra_context, image_path) based on memory + vision signals.
    """
    context_parts = []
    image_path = None

    if MEMORY_ENABLED:
        memories = search_memory(user_text, k=MEMORY_TOP_K)
        if memories:
            context_parts.append("Relevant memory:\n- " + "\n- ".join(memories))

    if VISION_ENABLED and _needs_vision(user_text):
        image_path = capture_image()
        if image_path:
            vision_text = analyze_image(image_path, user_text)
            if vision_text:
                context_parts.append(f"Vision analysis: {vision_text}")
            else:
                context_parts.append("Vision analysis failed.")
        else:
            context_parts.append("Image capture failed.")

    return "\n\n".join(context_parts), image_path
