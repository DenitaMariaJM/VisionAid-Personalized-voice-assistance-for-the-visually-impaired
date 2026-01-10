import string

def is_confident_command(text):
    """
    Returns True only if text looks like a real sentence.
    """
    if not text:
        return False

    text = text.strip().lower()

    # Reject very short inputs
    if len(text) < 5:
        return False

    # Reject noise (no vowels)
    if not any(v in text for v in "aeiou"):
        return False

    words = text.split()
    meaningful = [w for w in words if len(w) > 2]

    # Must have at least two meaningful words
    if len(meaningful) < 2:
        return False

    # Reject mid-thought endings
    if text.endswith((" and", " or", " the", " to")):
        return False

    return True
