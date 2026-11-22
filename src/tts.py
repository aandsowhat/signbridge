"""Offline speech without pausing recognition."""
from threading import Lock, Thread

_busy = Lock()


def speak(text):
    if not text.strip() or not _busy.acquire(blocking=False):
        return

    def play():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            print(f"Offline speech unavailable: {exc}", flush=True)
        finally:
            _busy.release()

    Thread(target=play, daemon=True).start()
