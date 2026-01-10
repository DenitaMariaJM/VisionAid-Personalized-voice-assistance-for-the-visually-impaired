"""Runtime configuration constants for VisionAid."""

WAKE_WORD = "alexa"
MODEL = "gpt-4o-mini"

# OpenAI speech configuration
STT_MODEL = "gpt-4o-mini-transcribe"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_SPEED = 1.0

# Audio capture/playback settings
STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
STT_PHRASE_TIME_LIMIT = 6
STT_SILENCE_THRESHOLD = 0.01
STT_SILENCE_DURATION = 0.8
STT_MIN_SPEECH_DURATION = 0.3
STT_CHUNK_DURATION = 0.2
AUDIO_INPUT_DEVICE = None
AUDIO_OUTPUT_DEVICE = None

# Interaction behavior
REQUIRE_WAKE_WORD = False
DEBUG_SPEECH = True

# Performance tuning
MEMORY_ENABLED = True
MEMORY_TOP_K = 2
MEMORY_TIMEOUT = 1.0
MEMORY_PERSIST = True
VISION_ENABLED = True
VISION_TIMEOUT = 1.5
MEMORY_SNIPPET_CHARS = 240
VISION_SNIPPET_CHARS = 280
CONTEXT_MAX_CHARS = 520
MEMORY_MIN_CHARS = 12
MAX_RESPONSE_TOKENS = 150
VISION_MAX_TOKENS = 100
MEMORY_STORE_ASSISTANT = False
VISION_MAX_DIM = 640
VISION_JPEG_QUALITY = 70

# Realtime settings
REALTIME_ENABLED = True
REALTIME_MODEL = "gpt-4o-realtime-preview"
REALTIME_VOICE = "alloy"
REALTIME_SAMPLE_RATE = 24000
REALTIME_CHUNK_MS = 20
REALTIME_SILENCE_THRESHOLD = 0.01
REALTIME_SILENCE_DURATION = 0.8
REALTIME_MIN_SPEECH_DURATION = 0.3
REALTIME_OUTPUT_SUPPRESS_SECONDS = 0.6
REALTIME_MAX_OUTPUT_TOKENS = 140
REALTIME_MAX_BUFFER_SECONDS = 6.0
REALTIME_RESPONSE_STYLE = (
    "Respond in English. Be concise and action-oriented. Ask a brief "
    "follow-up question if needed."
)
REALTIME_WAKE_WORDS = []
REALTIME_TRANSCRIPTION_MODEL = STT_MODEL
REALTIME_TRANSCRIPT_TIMEOUT = 3.0
REALTIME_USE_LOCAL_FALLBACK = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config():
    _require(isinstance(STT_MODEL, str) and STT_MODEL.strip(),
             "STT_MODEL must be a non-empty string.")
    _require(isinstance(REALTIME_TRANSCRIPTION_MODEL, str)
             and REALTIME_TRANSCRIPTION_MODEL.strip(),
             "REALTIME_TRANSCRIPTION_MODEL must be a non-empty string.")
    _require(REALTIME_TRANSCRIPT_TIMEOUT >= 0,
             "REALTIME_TRANSCRIPT_TIMEOUT must be >= 0.")
    _require(isinstance(REALTIME_MODEL, str) and REALTIME_MODEL.strip(),
             "REALTIME_MODEL must be a non-empty string.")
    _require(isinstance(REALTIME_VOICE, str) and REALTIME_VOICE.strip(),
             "REALTIME_VOICE must be a non-empty string.")
    _require(REALTIME_SAMPLE_RATE > 0, "REALTIME_SAMPLE_RATE must be > 0.")
    _require(REALTIME_CHUNK_MS > 0, "REALTIME_CHUNK_MS must be > 0.")
    _require(REALTIME_MIN_SPEECH_DURATION > 0,
             "REALTIME_MIN_SPEECH_DURATION must be > 0.")
    _require(0 < REALTIME_SILENCE_THRESHOLD <= 1.0,
             "REALTIME_SILENCE_THRESHOLD must be in (0, 1].")
    _require(REALTIME_SILENCE_DURATION > 0,
             "REALTIME_SILENCE_DURATION must be > 0.")
    _require(REALTIME_OUTPUT_SUPPRESS_SECONDS >= 0,
             "REALTIME_OUTPUT_SUPPRESS_SECONDS must be >= 0.")
    _require(REALTIME_MAX_OUTPUT_TOKENS > 0,
             "REALTIME_MAX_OUTPUT_TOKENS must be > 0.")
    _require(REALTIME_MAX_BUFFER_SECONDS > 0,
             "REALTIME_MAX_BUFFER_SECONDS must be > 0.")
    _require(isinstance(REALTIME_WAKE_WORDS, list),
             "REALTIME_WAKE_WORDS must be a list.")
    _require(MAX_RESPONSE_TOKENS > 0, "MAX_RESPONSE_TOKENS must be > 0.")
    _require(VISION_MAX_TOKENS > 0, "VISION_MAX_TOKENS must be > 0.")
    _require(VISION_MAX_DIM > 0, "VISION_MAX_DIM must be > 0.")
    _require(1 <= VISION_JPEG_QUALITY <= 100,
             "VISION_JPEG_QUALITY must be in [1, 100].")
    _require(MEMORY_TOP_K >= 0, "MEMORY_TOP_K must be >= 0.")
