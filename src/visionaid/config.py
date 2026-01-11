"""Runtime configuration constants for VisionAid."""

AUDIO_INPUT_DEVICE = None
AUDIO_OUTPUT_DEVICE = None

STT_MODEL = "gpt-4o-mini-transcribe"
# Performance tuning
MEMORY_ENABLED = True
MEMORY_PERSIST = True
MEMORY_SNIPPET_CHARS = 240
VISION_MAX_TOKENS = 100
MEMORY_STORE_ASSISTANT = False
VISION_MAX_DIM = 640
VISION_JPEG_QUALITY = 70
CAMERA_INDEX = 0
CAMERA_AUTO_PROBE = True
CAMERA_PROBE_MAX = 4
CAMERA_BACKEND = "v4l2"
CAMERA_WARMUP_FRAMES = 4
CAMERA_FRAME_WIDTH = 0
CAMERA_FRAME_HEIGHT = 0
TOOL_ACCESS_RECHECK_SECONDS = 30.0

# Realtime settings
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
    "You are VisionAid, an assistive voice system for the visually impaired. "
    "Always respond ONLY in English. If the user speaks another language, reply: "
    'I can only communicate in English. Please repeat in English.' 
    "Be extremely concise, direct, and action-oriented. "
    "Prioritize navigation safety, obstacle awareness, and direction-based guidance (front/left/right). "
    "Never describe colors, decorations, or irrelevant visual details. "
    "If clarification is needed, ask a short follow-up question."
)
REALTIME_TOOL_GUIDANCE = (
    "You MUST use tools for all vision or memory-related questions. "
    "For any question about the environment, navigation, or obstacles, ALWAYS call capture_image first, "
    "then call analyze_image with the user's request and the captured image. "
    "Do NOT answer vision questions without using both tools. "
    "For memory, use search_memory, store_memory, list_memory, or clear_memory as appropriate. "
    "\n\n"
    "Example (vision):\n"
    "User: What is in front of me?\n"
    "Assistant: (call capture_image, then analyze_image)\n"
    "\n"
    "Example (memory):\n"
    "User: Remind me what I asked earlier.\n"
    "Assistant: (call search_memory with the query)\n"
    "\n"
    "Never answer a vision or memory question without using the appropriate tool(s)."
)
REALTIME_WAKE_WORDS = []
REALTIME_TRANSCRIPTION_MODEL = STT_MODEL
REALTIME_TRANSCRIPT_TIMEOUT = 3.0
REALTIME_USE_LOCAL_FALLBACK = False
REALTIME_TOOL_CHOICE = "auto"


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
    _require(REALTIME_TOOL_CHOICE in ("auto", "none"),
             "REALTIME_TOOL_CHOICE must be 'auto' or 'none'.")
    _require(VISION_MAX_TOKENS > 0, "VISION_MAX_TOKENS must be > 0.")
    _require(VISION_MAX_DIM > 0, "VISION_MAX_DIM must be > 0.")
    _require(1 <= VISION_JPEG_QUALITY <= 100,
             "VISION_JPEG_QUALITY must be in [1, 100].")
    _require(MEMORY_SNIPPET_CHARS > 0, "MEMORY_SNIPPET_CHARS must be > 0.")
    _require(isinstance(CAMERA_INDEX, int) and CAMERA_INDEX >= 0,
             "CAMERA_INDEX must be a non-negative integer.")
    _require(isinstance(CAMERA_AUTO_PROBE, bool),
             "CAMERA_AUTO_PROBE must be a boolean.")
    _require(isinstance(CAMERA_PROBE_MAX, int) and CAMERA_PROBE_MAX > 0,
             "CAMERA_PROBE_MAX must be a positive integer.")
    _require(CAMERA_BACKEND in (None, "", "v4l2", "dshow", "avfoundation"),
             "CAMERA_BACKEND must be None or one of: v4l2, dshow, avfoundation.")
    _require(isinstance(CAMERA_WARMUP_FRAMES, int) and CAMERA_WARMUP_FRAMES >= 0,
             "CAMERA_WARMUP_FRAMES must be >= 0.")
    _require(isinstance(CAMERA_FRAME_WIDTH, int) and CAMERA_FRAME_WIDTH >= 0,
             "CAMERA_FRAME_WIDTH must be >= 0.")
    _require(isinstance(CAMERA_FRAME_HEIGHT, int) and CAMERA_FRAME_HEIGHT >= 0,
             "CAMERA_FRAME_HEIGHT must be >= 0.")
    _require(TOOL_ACCESS_RECHECK_SECONDS >= 0,
             "TOOL_ACCESS_RECHECK_SECONDS must be >= 0.")
