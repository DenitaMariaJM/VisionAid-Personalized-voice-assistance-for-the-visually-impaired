"""Language detection helpers for enforcing English-only responses."""

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
except Exception:  # pragma: no cover - fallback when langdetect isn't installed
    detect = None
    LangDetectException = Exception


def _basic_english_heuristic(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    non_ascii_letters = [c for c in letters if not c.isascii()]
    if non_ascii_letters:
        return False
    return True


def is_english(text):
    """
    Returns True if text appears to be English.
    """
    if not text or not text.strip():
        return False
    if detect is None:
        return _basic_english_heuristic(text)
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False
