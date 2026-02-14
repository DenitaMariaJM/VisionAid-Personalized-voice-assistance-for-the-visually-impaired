# VisionAid - Personalized Voice Assistance for the Visually Impaired

## Overview
VisionAid is a voice-first assistive agent designed for visually impaired users.
It combines:
- Microphone input + speech-to-text
- Camera capture + vision understanding
- Context memory (semantic + episodic + recent interactions)
- Spoken responses (text-to-speech)

The assistant focuses on practical guidance, short responses, and safety-oriented spatial cues.

## Core Capabilities
- Understand user speech and route each turn to `vision`, `memory`, `both`, or `chat`.
- Capture camera images only when needed.
- Describe surroundings and potential hazards.
- Remember prior turns and use them in follow-up questions.
- Personalize behavior using a persistent user profile.

## Runtime Flow
Single turn flow:
1. Record one utterance from the microphone (`audio_io.record_utterance_wav_bytes`).
2. Transcribe speech to text (`stt_whisper.transcribe_audio`).
3. Detect profile updates in user speech (for example: blind/low-vision, concise/step-by-step).
4. Plan actions (`agent.plan_actions`) as `vision`, `memory`, `both`, or `chat`.
5. If `vision`/`both`:
- Capture image (`vision.capture_image`).
- Analyze image (`vision.analyze_image`).
6. If `memory`/`both`, retrieval fallback chain is:
- Episodic memory (`episodic_retrieval.search_episodic_memory`)
- Semantic memory (`memory.search_memory`)
- Recent interaction logs (`db.get_recent_interactions`)
7. Generate final assistant reply (`pipeline._final_response`) with:
- System behavior instructions
- Active user profile instructions
- Retrieved memory and/or vision context
8. Speak response (`tts.speak`) and store interaction (`db.log_interaction`).
9. Store semantic memory entry for recall (`memory.store_memory`).

## Personalization
VisionAid actively uses `user_profile` in SQLite.

Profile fields:
- `vision_level` (for example: `blind`, `low_vision`)
- `response_style` (for example: `concise`, `step_by_step`, `detailed`)
- `language` (currently English output is enforced)

Voice examples that update profile:
- "I am blind"
- "I have low vision"
- "Be concise"
- "Give step by step guidance"
- "Respond in English"

## Memory Design
### Semantic Memory
- Embedding-based recall using FAISS (`text-embedding-3-small`).
- Stored in `semantic_memory` table with timestamps.
- Entries can include both user and assistant text.

### Episodic Memory
- Summarized day-level recall from previous interactions.
- Stored in `episodic_memory` with summary embeddings.
- Used for questions like "have I been here before?"

### Interaction Log Fallback
- If episodic and semantic retrieval return nothing, recent `interactions` rows are used as memory context.

### User Facts Memory
- Stable personal facts from speech (for example name, preferences, allergies) are stored in `user_facts`.
- Facts are embedding-indexed and retrieved during memory turns to improve personalized recall.

## Safety Behavior
- If camera/image analysis fails in a vision-required turn, VisionAid avoids guessing.
- It returns a safe prompt asking the user to reposition the camera.

## Models and Services
Configured in `src/visionaid/config.py`:
- `STT_MODEL = "gpt-4o-mini-transcribe"`
- `AGENT_MODEL = "gpt-4o-mini"`
- `VISION_MODEL = "gpt-4o-mini"`
- `ASSISTANT_MODEL = "gpt-4o-mini"`
- `TTS_MODEL = "gpt-4o-mini-tts"`
- Embeddings: `text-embedding-3-small`

## Project Structure
```
src/visionaid/
  __init__.py
  __main__.py
  main.py
  pipeline.py
  agent.py
  audio_io.py
  stt_whisper.py
  tts.py
  vision.py
  memory.py
  episodic_summary.py
  episodic_retrieval.py
  db.py
  tool_access.py
  config.py
  logging_utils.py
main.py
permissions.py
requirements.txt
setup.py
```

## Setup
### 1) System dependencies (Ubuntu/Debian)
```bash
sudo apt install -y portaudio19-dev libsndfile1 python3-dev build-essential
```

### 2) Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3) API key
```bash
export OPENAI_API_KEY="your_key_here"
```

### 4) Optional package install
```bash
pip install .
```

## Run
From repo root:
```bash
python main.py
```

Hardware pre-check:
```bash
python permissions.py
```

## Configuration Notes
Key settings in `src/visionaid/config.py`:
- `MEMORY_ENABLED = True`
- `MEMORY_PERSIST = True`
- `MEMORY_STORE_EVERY_TURN = True`
- `MEMORY_STORE_ASSISTANT = True`
- `EPISODIC_SUMMARY_MAX_CHARS = 1200`
- `VISION_MAX_TOKENS = 60`
- `ASSISTANT_MAX_TOKENS = 120`

Useful environment variables:
- `OPENAI_API_KEY`
- `VISIONAID_LOG_LEVEL` (for example `DEBUG`, `INFO`)
- `VISIONAID_DB_PATH` (custom SQLite path)
- `VISIONAID_IMAGE_DIR` (custom captured image directory)

## Usage Examples
Vision queries:
- "What am I seeing?"
- "What is in front of me?"
- "Capture and describe this."

Memory queries:
- "What did I ask earlier?"
- "Have I been here before?"
- "Remind me what you told me about this place."

Profile updates:
- "I am blind."
- "Give step by step directions."
- "Be concise."

## Troubleshooting
- If camera checks pass but no vision response appears, inspect planner intent logs:
- Look for `action_plan intent=vision` or `both`.
- If microphone captures are too short/long, tune VAD values in `config.py`.
- If memory answers seem empty, verify DB rows in `interactions`, `semantic_memory`, `episodic_memory`, and `user_facts`.
- Set `VISIONAID_LOG_LEVEL=DEBUG` for deeper diagnostics.

## Current Scope
VisionAid is a strong prototype and Raspberry Pi-friendly architecture for assistive interaction.
For production-grade deployment, add hardware field testing, stronger safety policy constraints, and automated tests.
