"""Runtime configuration constants for VisionAid."""

AUDIO_INPUT_DEVICE = None
AUDIO_OUTPUT_DEVICE = None

STT_MODEL = "gpt-4o-mini-transcribe"
# Performance tuning
MEMORY_ENABLED = True
MEMORY_PERSIST = True
MEMORY_SNIPPET_CHARS = 240
VISION_MAX_TOKENS = 60
MEMORY_STORE_ASSISTANT = False
MEMORY_STORE_EVERY_TURN = False
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

# Non-realtime pipeline models
AGENT_MODEL = "gpt-4o-mini"
ASSISTANT_MODEL = "gpt-4o-mini"
VISION_MODEL = "gpt-4o-mini"
AGENT_MAX_TOKENS = 70
ASSISTANT_MAX_TOKENS = 120
MEMORY_TOP_K = 1
AGENT_LLM_FALLBACK_ONLY = True
TTS_ENABLED = True
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_FORMAT = "wav"

# Shared audio/VAD settings (also used by the non-realtime pipeline)
REALTIME_SAMPLE_RATE = 24000
REALTIME_CHUNK_MS = 20
REALTIME_SILENCE_THRESHOLD = 0.01
REALTIME_SILENCE_DURATION = 0.8
REALTIME_MAX_BUFFER_SECONDS = 6.0


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config():
    _require(isinstance(STT_MODEL, str) and STT_MODEL.strip(),
             "STT_MODEL must be a non-empty string.")
    _require(isinstance(AGENT_MODEL, str) and AGENT_MODEL.strip(),
             "AGENT_MODEL must be a non-empty string.")
    _require(isinstance(ASSISTANT_MODEL, str) and ASSISTANT_MODEL.strip(),
             "ASSISTANT_MODEL must be a non-empty string.")
    _require(isinstance(VISION_MODEL, str) and VISION_MODEL.strip(),
             "VISION_MODEL must be a non-empty string.")
    _require(isinstance(AGENT_MAX_TOKENS, int) and AGENT_MAX_TOKENS > 0,
             "AGENT_MAX_TOKENS must be a positive integer.")
    _require(isinstance(ASSISTANT_MAX_TOKENS, int) and ASSISTANT_MAX_TOKENS > 0,
             "ASSISTANT_MAX_TOKENS must be a positive integer.")
    _require(isinstance(MEMORY_TOP_K, int) and MEMORY_TOP_K > 0,
             "MEMORY_TOP_K must be a positive integer.")
    _require(isinstance(AGENT_LLM_FALLBACK_ONLY, bool),
             "AGENT_LLM_FALLBACK_ONLY must be a boolean.")
    _require(isinstance(TTS_ENABLED, bool), "TTS_ENABLED must be a boolean.")
    _require(isinstance(TTS_MODEL, str) and TTS_MODEL.strip(),
             "TTS_MODEL must be a non-empty string.")
    _require(isinstance(TTS_VOICE, str) and TTS_VOICE.strip(),
             "TTS_VOICE must be a non-empty string.")
    _require(TTS_FORMAT in ("wav", "mp3", "pcm"),
             "TTS_FORMAT must be one of: wav, mp3, pcm.")
    _require(REALTIME_SAMPLE_RATE > 0, "REALTIME_SAMPLE_RATE must be > 0.")
    _require(REALTIME_CHUNK_MS > 0, "REALTIME_CHUNK_MS must be > 0.")
    _require(0 < REALTIME_SILENCE_THRESHOLD <= 1.0,
             "REALTIME_SILENCE_THRESHOLD must be in (0, 1].")
    _require(REALTIME_SILENCE_DURATION > 0,
             "REALTIME_SILENCE_DURATION must be > 0.")
    _require(REALTIME_MAX_BUFFER_SECONDS > 0,
             "REALTIME_MAX_BUFFER_SECONDS must be > 0.")
    _require(VISION_MAX_TOKENS > 0, "VISION_MAX_TOKENS must be > 0.")
    _require(VISION_MAX_DIM > 0, "VISION_MAX_DIM must be > 0.")
    _require(1 <= VISION_JPEG_QUALITY <= 100,
             "VISION_JPEG_QUALITY must be in [1, 100].")
    _require(MEMORY_SNIPPET_CHARS > 0, "MEMORY_SNIPPET_CHARS must be > 0.")
    _require(isinstance(MEMORY_STORE_EVERY_TURN, bool),
             "MEMORY_STORE_EVERY_TURN must be a boolean.")
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
