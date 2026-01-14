# VisionAid – Personalized Voice Assistance for the Visually Impaired

##  Project Description

**VisionAid** is a personalized, voice-enabled assistive system designed to help **visually impaired users** understand and navigate their surroundings safely.  
The system uses **voice interaction, camera-based scene understanding, AI reasoning, and contextual memory** to provide **short, actionable, and accessibility-focused guidance**.

Unlike generic voice assistants, VisionAid prioritizes:
- Navigation safety
- Obstacle awareness
- Direction-based guidance
- Minimal and meaningful responses

---

##  Project Objectives

- Help visually impaired users understand **what is in front of them**
- Provide **clear movement guidance** (front / left / right)
- Avoid unnecessary visual descriptions (colors, decorations, artwork)
- Maintain contextual memory for follow-up questions
- Deliver calm, concise, and assistive voice responses

---

##  How the System Works

1. The system captures a single utterance from the microphone.
2. The utterance is transcribed to text (STT).
3. An action agent decides whether to use camera and/or memory.
4. If needed, the client captures an image and runs image understanding.
5. The final assistant model answers using the user query + vision/memory context.
6. The response is also spoken aloud (TTS).
7. The interaction is stored in a local database.

---

##  Models Used

Configured in `src/visionaid/config.py`:

Current defaults:

- `STT_MODEL = "gpt-4o-mini-transcribe"`: speech-to-text transcription (`src/visionaid/stt_whisper.py`)
- `AGENT_MODEL = "gpt-4o-mini"`: action planning agent (`src/visionaid/agent.py`)
- `VISION_MODEL = "gpt-4o-mini"`: image understanding (`src/visionaid/vision.py`)
- `ASSISTANT_MODEL = "gpt-4o-mini"`: final response model (`src/visionaid/pipeline.py`)
- `TTS_MODEL = "gpt-4o-mini-tts"`: text-to-speech (`src/visionaid/tts.py`)
- Embeddings (semantic memory) uses `model="text-embedding-3-small"` (`src/visionaid/memory.py`)

You can change these in `src/visionaid/config.py`.

---

##  Current Code Flow

Single turn (one user utterance):

1. `audio_io.record_utterance_wav_bytes()` records one utterance (simple VAD).
2. `stt_whisper.transcribe_audio()` turns WAV bytes into `user_text`.
3. `agent.plan_actions()` returns an `ActionPlan` (`vision` / `memory` / `both` / `chat`).
4. If `vision`/`both`:
   - `vision.capture_image()` saves a timestamped JPEG into `captured_images/`.
   - `vision.analyze_image()` produces a short textual description/answer.
5. If `memory`/`both`:
   - `memory.search_memory()` retrieves relevant snippets (FAISS + embeddings).
6. If pure `vision`, VisionAid returns the vision analysis directly to minimize tokens.
   Otherwise `pipeline._final_response()` calls the final model using the available context.
7. `tts.speak()` plays the final text response aloud (if `TTS_ENABLED=True`).
8. `db.log_interaction()` persists query/response/image_path to SQLite, and memory may be stored depending on `MEMORY_STORE_EVERY_TURN`.

---

##  Key Modules / Functions

- Audio capture: `src/visionaid/audio_io.py` (`record_utterance_wav_bytes`)
- Speech-to-text: `src/visionaid/stt_whisper.py` (`transcribe_audio`)
- Action planning: `src/visionaid/agent.py` (`plan_actions`, `ActionPlan`)
- Vision: `src/visionaid/vision.py` (`capture_image`, `analyze_image`)
- Memory: `src/visionaid/memory.py` (`search_memory`, `store_memory`, `load_memory`)
- Orchestration: `src/visionaid/pipeline.py` (`run_pipeline`)
- Text-to-speech: `src/visionaid/tts.py` (`speak`)
- Persistence: `src/visionaid/db.py` (`init_db`, `log_interaction`)

---

##  Key Features

###  Voice Interaction
- Local microphone capture (simple VAD)
- Speech-to-text using OpenAI transcription
- Text-to-speech using OpenAI TTS + local audio playback

###  Tool-Driven Actions
- Camera capture and vision analysis via an action-planning agent
- Memory search/store via an action-planning agent

###  Vision-Based Assistance
- Camera image capture (saved to `captured_images/`)
- Image understanding to answer general visual questions (not only obstacles)

###  Accessibility-Focused Responses
- Concise, practical voice responses
- Safety-first guidance when navigation-related

###  Contextual Memory
- Semantic memory using embeddings with SQLite persistence
- Supports follow-up queries (e.g., “And on the left?”)

### Data Storage
- SQLite database for storing interactions
- Stores query, response, image reference, and timestamp

### Observability
- Structured logging (set `VISIONAID_LOG_LEVEL`)

---

##  Repo Layout

```
src/visionaid/
  __main__.py
  main.py
  pipeline.py
  agent.py
  audio_io.py
  tts.py
  config.py
  tool_access.py
  stt_whisper.py
  logging_utils.py
  vision.py
  memory.py
  db.py
permissions.py
```

##  Quick Start

```
python main.py
```

##  Configuration Notes

- `OPENAI_API_KEY` must be set in your environment.
- Core settings live in `src/visionaid/config.py` (STT, VAD, models, token limits, TTS).
- Memory persistence can be toggled with `MEMORY_PERSIST`.
- To reduce token/latency, memory embeddings are not stored every turn by default; enable with `MEMORY_STORE_EVERY_TURN = True`.
- Captured image directory can be overridden with `VISIONAID_IMAGE_DIR=/path`.

##  Setup Notes

- Audio devices: if you get device errors, set `AUDIO_INPUT_DEVICE` and
  `AUDIO_OUTPUT_DEVICE` in `src/visionaid/config.py` to the correct device
  indices.
- Logging verbosity: set `VISIONAID_LOG_LEVEL=DEBUG` for more detail.
- Linux dependencies: install PortAudio headers before building audio
  packages.

```
sudo apt install -y portaudio19-dev libsndfile1 python3-dev build-essential
```

- Camera access: ensure your user has permission to access the camera device
  (e.g., add to the `video` group on Linux).

##  Permissions Check

You can run a quick hardware access check before launching the app:

```
python permissions.py
```
