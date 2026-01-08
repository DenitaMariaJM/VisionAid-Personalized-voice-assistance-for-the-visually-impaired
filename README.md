"# VisionAid-Personalized-voice-assistance-for-the-visually-impaired" 
# VisionAid – Personalized Voice Assistance for the Visually Impaired

## 📌 Project Description

**VisionAid** is a personalized, voice-enabled assistive system designed to help **visually impaired users** understand and navigate their surroundings safely.  
The system uses **voice interaction, camera-based scene understanding, AI reasoning, and contextual memory** to provide **short, actionable, and accessibility-focused guidance**.

Unlike generic voice assistants, VisionAid prioritizes:
- Navigation safety
- Obstacle awareness
- Direction-based guidance
- Minimal and meaningful responses

---

## 🎯 Project Objectives

- Help visually impaired users understand **what is in front of them**
- Provide **clear movement guidance** (front / left / right)
- Avoid unnecessary visual descriptions (colors, decorations, artwork)
- Maintain contextual memory for follow-up questions
- Deliver calm, concise, and assistive voice responses

---

## ⚙️ How the System Works

1. The system listens for a **wake word** (e.g., *Alexa*).
2. The user speaks a query.
3. The system identifies whether the query requires **vision-based understanding**.
4. If required, a real-time image is captured using the camera.
5. Past relevant interactions are retrieved using **semantic memory**.
6. The query and image are sent to a **vision-capable AI model**.
7. The system generates an **accessibility-focused response**.
8. The response is spoken aloud using **Google Text-to-Speech**.
9. The interaction is stored in a local database.

---

## ✨ Key Features

### 🗣 Voice Interaction
- Wake-word activation
- Speech-to-text using Google Speech Recognition
- Text-to-speech using Google TTS

### 📷 Vision-Based Assistance
- Real-time camera image capture
- Multimodal AI (text + image)
- Focus on obstacles and navigation

### 🧭 Accessibility-Focused Responses
- Short, clear, and action-oriented
- Direction-aware (front / left / right)
- Safety-first language

### 🧠 Contextual Memory
- Semantic memory using embeddings
- Supports follow-up queries (e.g., “And on the left?”)

### 💾 Data Storage
- SQLite database for storing interactions
- Stores query, response, image reference, and timestamp

---

## 🗂 Project Structure

