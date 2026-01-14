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

##  Key Features

###  Voice Interaction
- Local microphone capture (simple VAD)
- Speech-to-text using OpenAI transcription
- Text-to-speech using OpenAI TTS + local audio playback

###  Tool-Driven Actions
- Camera capture and vision analysis via an action-planning agent
- Memory search/store via an action-planning agent

###  Vision-Based Assistance
- Real-time camera image capture
- Multimodal AI (text + image)
- Focus on obstacles and navigation

###  Accessibility-Focused Responses
- Short, clear, and action-oriented
- Direction-aware (front / left / right)
- Safety-first language

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
  utils/
    language_guard.py
permissions.py
```

##  Quick Start

```
python main.py
```

##  Configuration Notes

- `OPENAI_API_KEY` must be set in your environment.
- Realtime transcription model and fallback settings live in
  `src/visionaid/config.py` (STT + VAD + models).
- Memory persistence can be toggled with `MEMORY_PERSIST`.

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
