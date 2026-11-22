"""Laptop-only isolated-sign recognition with the Kaggle winning TFLite model."""
from collections import Counter, deque
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "models" / "words"
PHRASES = {("hello",): "Hello.", ("thankyou",): "Thank you.",
           ("water", "please"): "Water, please.", ("yes",): "Yes.", ("no",): "No."}


def gloss(label):
    return "THANK YOU" if label == "thankyou" else label.upper()


class WordModel:
    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter

        mapping = json.loads((ASSETS / "labels.json").read_text())
        if sorted(mapping.values()) != list(range(250)):
            raise ValueError("Expected 250 distinct class indices.")
        self.labels = sorted(mapping, key=mapping.get)
        self.interpreter = Interpreter(model_path=str(ASSETS / "model.tflite"), num_threads=2)
        self.run = self.interpreter.get_signature_runner("serving_default")

    def predict(self, frames):
        scores = self.run(inputs=np.asarray(frames, dtype=np.float32))["outputs"]
        if scores.shape != (250,) or not np.isfinite(scores).all():
            return "", 0.0
        weights = np.exp(scores - scores.max())
        index = int(scores.argmax())
        return self.labels[index], float(weights[index] / weights.sum())


def landmarks(result):
    frame = np.full((543, 3), np.nan, np.float32)
    for group, offset in ((result.face_landmarks, 0), (result.left_hand_landmarks, 468),
                          (result.pose_landmarks, 489), (result.right_hand_landmarks, 522)):
        if group is not None:
            frame[offset:offset + len(group.landmark)] = [(p.x, p.y, p.z) for p in group.landmark]
    return frame


class WordWorker:
    """Keep capture responsive by allowing at most one waiting frame."""
    def __init__(self):
        self.pending = Queue(maxsize=1)
        self.stop = Event()
        self.result = (0., "", 0., False, "Loading word model", 0)
        self.thread = Thread(target=self.work, daemon=True)
        self.thread.start()

    def submit(self, frame):
        try:
            self.pending.put_nowait((monotonic(), frame))
        except Full:
            pass

    def close(self):
        self.stop.set()
        self.thread.join()

    def work(self):
        try:
            from mediapipe import solutions
            model = WordModel()
            frames = deque(maxlen=24)
            votes = deque(maxlen=3)
            epoch = 0
            step = 0
            last = 0.
            with solutions.holistic.Holistic(model_complexity=1, refine_face_landmarks=False) as tracker:
                while not self.stop.is_set():
                    try:
                        stamp, frame = self.pending.get(timeout=.1)
                    except Empty:
                        continue
                    if stamp - last > .6:
                        frames.clear()
                        votes.clear()
                    last = stamp
                    result = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if result.left_hand_landmarks is None and result.right_hand_landmarks is None:
                        if frames:
                            epoch += 1
                        frames.clear()
                        votes.clear()
                        self.result = (stamp, "", 0., False, "Show one isolated sign", epoch)
                        continue
                    frames.append(landmarks(result))
                    step += 1
                    if len(frames) < 8 or step % 3:
                        continue
                    word, score = model.predict(frames)
                    votes.append((word, score))
                    winner = Counter(label for label, _ in reversed(votes)).most_common(1)[0][0]
                    scores = [value for label, value in votes if label == winner]
                    stable = bool(winner and len(scores) == 3 and min(scores) >= .55)
                    self.result = (stamp, winner, sum(scores) / len(scores), stable,
                                   "Ready" if stable else "Watching", epoch)
        except Exception as exc:
            message = f"Word mode unavailable: {exc}"
            print(message, flush=True)
            self.result = (0., "", 0., False, message, 0)


class Sentence:
    def __init__(self):
        self.signs = []
        self.last = None

    def edit(self, key, result, now):
        stamp, word, _, stable, _, epoch = result
        if key == 32 and stable and now - stamp < .8 and self.last != (epoch, word):
            self.signs.append(word)
            self.last = (epoch, word)
        elif key in (8, 127) and self.signs:
            self.signs.pop()
        elif key in (ord("c"), ord("C")):
            self.signs.clear()

    def text(self):
        return PHRASES.get(tuple(self.signs), " ".join(map(gloss, self.signs)))


def preview(frame, result, sentence):
    _, word, score, stable, status, _ = result
    canvas = cv2.copyMakeBorder(cv2.resize(frame, (640, 480)), 0, 144, 0, 0,
                                cv2.BORDER_CONSTANT, value=(42, 34, 25))
    lines = [f"Current: {gloss(word) or '-'}   Score: {score:.2f}   {status}",
             "Confirmed: " + " ".join(map(gloss, sentence.signs))[-55:]]
    if tuple(sentence.signs) in PHRASES:
        lines.append("English: " + sentence.text())
    lines.append("Space confirm | Backspace undo | C clear | Enter speak | M mode | Esc quit")
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, 508 + 30*i), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (174, 225, 108) if stable and i == 0 else (246, 247, 243), 1, cv2.LINE_AA)
    return canvas
