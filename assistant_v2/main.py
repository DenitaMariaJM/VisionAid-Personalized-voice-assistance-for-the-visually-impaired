import os
import sqlite3
from openai import OpenAI

from config import WAKE_WORD, MODEL
from voice import listen, speak
from vision import capture_image
from memory import store_memory, search_memory
from db import init_db

client = OpenAI()
init_db()

def understand_query(query):
    keywords = ["front", "left", "right", "see"]
    return any(k in query.lower() for k in keywords)

def run():
    speak("Assistant is ready")

    while True:
        try:
            text = listen().lower()

            if WAKE_WORD not in text:
                continue

            query = text.replace(WAKE_WORD, "").strip()

            needs_image = understand_query(query)
            image_path = None

            if needs_image:
                image_path = capture_image()

            memories = search_memory(query)

            prompt = f"""
            You are a voice assistant for a visually impaired user.

            Previous context:
            {memories}

            Current query:
            {query}
            """

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content

            speak(response)

            # Store memory
            store_memory(query + " " + response)

            # Save to DB
            conn = sqlite3.connect("assistant.db")
            c = conn.cursor()
            c.execute(
                "INSERT INTO interactions (query, response, image_path) VALUES (?, ?, ?)",
                (query, response, image_path)
            )
            conn.commit()
            conn.close()

        except Exception as e:
            speak("Sorry, I did not understand")

run()
