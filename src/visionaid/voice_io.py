import os

import speech_recognition as sr
from playsound import playsound

WAKE_WORD = "al"
recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.3

_BEEP_PATH = os.path.join(os.path.dirname(__file__), "assets", "beep.wav")

def play_beep():
    """
    Plays a short beep to indicate mic is ON.
    """
    try:
        playsound(_BEEP_PATH)
    except Exception as exc:
        print(f"Beep failed: {exc}")

def listen_for_wake_word():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("🎧 Listening for wake word...")

        while True:
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio).lower()
                if WAKE_WORD in text:
                    print("🟢 Wake word detected")
                    return
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                print(f"Wake word recognition failed: {exc}")
                continue
            except Exception as exc:
                print(f"Wake word listener failed: {exc}")
                continue


def record_command(filename="command.wav"):
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("🎤 Listening for command...")
        audio = recognizer.listen(source)

    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())

    return filename
