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
VISION_ENABLED = True
VISION_TIMEOUT = 1.5

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
REALTIME_RESPONSE_STYLE = (
    "You are a helpful voice assistant. Always respond in English. Keep "
    "responses concise, clear, and action-oriented. If you are unsure, ask "
    "a brief follow-up question."
)
REALTIME_WAKE_WORDS = []
