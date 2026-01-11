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

1. The system streams audio in **realtime** over WebSocket.
2. The user speaks a query.
3. The model uses **tool calls** to request actions (camera capture, memory search).
4. The client executes tools locally and sends results back.
5. The model continues with the new context to answer.
6. The response is spoken aloud using streamed audio.
7. The interaction is stored in a local database.

---

##  Key Features

###  Voice Interaction
- Realtime audio streaming (WebSocket)
- Speech-to-text using OpenAI realtime transcription
- Text-to-speech using OpenAI audio output

###  Tool-Driven Actions
- Camera capture and vision analysis via tool calls
- Memory search/store/list/clear via tool calls
- Tool choice is model-driven (no keyword gating in the client)

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
  realtime_client.py
  config.py
  tools.py
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
  `src/visionaid/config.py` (`REALTIME_TRANSCRIPTION_MODEL`,
  `REALTIME_TRANSCRIPT_TIMEOUT`, `REALTIME_USE_LOCAL_FALLBACK`).
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
