import asyncio
import websockets
import json
import pyttsx3
import wave
import os
import uuid

engine = pyttsx3.init()
engine.setProperty("rate", 170)

def synthesize_to_file(text, filename):
    engine.save_to_file(text, filename)
    engine.runAndWait()

    # 🔴 REQUIRED flush
    engine.stop()


async def tts_handler(websocket):
    print("🔊 TTS client connected")

    loop = asyncio.get_running_loop()

    try:
        async for message in websocket:
            data = json.loads(message)
            text = data.get("text", "").strip()

            if not text:
                await websocket.send("END")
                continue

            filename = f"temp_{uuid.uuid4().hex}.wav"

            # ✅ RUN BLOCKING TTS IN THREAD
            await loop.run_in_executor(
                None, synthesize_to_file, text, filename
            )

            with open(filename, "rb") as f:
                wav_bytes = f.read()
                await websocket.send(wav_bytes)




            await websocket.send("END")
            os.remove(filename)

    except websockets.ConnectionClosed:
        print("🔌 TTS client disconnected")

def start_tts_server():
    async def runner():
        async with websockets.serve(tts_handler, "localhost", 8765):
            print("✅ TTS Server running on ws://localhost:8765")
            await asyncio.Future()  # run forever

    asyncio.run(runner())
