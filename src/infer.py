"""Recognize static ASL letters in the webcam preview."""
import sys
from collections import Counter, deque
from pathlib import Path
from time import monotonic, perf_counter

import cv2
import numpy as np
import onnxruntime as ort

from words import Sentence, WordWorker, preview as word_preview
from tts import speak
from preview import draw_preview
from landmarks import HandTracker, normalize

MODEL = Path(__file__).resolve().parents[1] / "models" / "classifier.onnx"


class Predictor:
    def __init__(self) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL), options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name


    def predict(self, points: np.ndarray | None) -> tuple[str, float]:
        if points is None:
            return "", 0.0
        try:
            features = normalize(points)
        except ValueError:
            return "", 0.0
        labels, probabilities = self.session.run(None, {self.input_name: features[None, :]})
        return str(labels[0]), float(probabilities.max())


class Smoother:
    def __init__(self):
        self.history = deque(maxlen=10)

    def update(self, letter, confidence):
        if not letter:
            self.history.clear()
            return "", 0.0
        self.history.append((letter, confidence))
        # In a tied vote, prefer the most recent tied letter.
        votes = Counter(label for label, _ in reversed(self.history))
        winner, count = votes.most_common(1)[0]
        confidence = sum(p for label, p in self.history if label == winner) / count
        return winner, confidence


class LocalInference:
    def __init__(self):
        self.predictor = Predictor()
        self.tracker = HandTracker()

    def process(self, frame):
        points, frame = self.tracker.process(frame)
        return (*self.predictor.predict(points), frame, "Local", None)

    def close(self):
        self.tracker.close()


def run_preview(backend, *, laptop_words=False) -> int:
    worker = None
    sentence = Sentence()
    smoother = Smoother()
    camera = cv2.VideoCapture(0)
    try:
        if not camera.isOpened():
            print("No webcam found. Check the connection and camera permissions.", file=sys.stderr)
            return 1
        source = None
        start, count, fps = perf_counter(), 0, 0.0
        word = candidate = ""
        since = start
        added = False
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Webcam stopped delivering frames.", file=sys.stderr)
                return 1
            if worker is not None:
                worker.submit(frame)
                result = worker.result
                now = monotonic()
                if result[0] and now - result[0] >= .8:
                    result = (result[0], "", 0., False, "Waiting for fresh input", result[-1])
                cv2.imshow("ASL fingerspelling", word_preview(cv2.flip(frame, 1), result, sentence))
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord("q")):
                    return 0
                if key in (ord("m"), ord("M")):
                    worker.close()
                    worker = None
                    sentence.last = None
                    smoother.history.clear()
                    candidate = ""
                    start, count = perf_counter(), 0
                elif key in (10, 13):
                    speak(sentence.text())
                else:
                    sentence.edit(key, result, now)
                continue
            letter, confidence, frame, name, latency = backend.process(cv2.flip(frame, 1))
            if name != source:
                smoother.history.clear()
                candidate = ""
                source = name
            letter, confidence = smoother.update(letter, confidence)
            count += 1
            now = perf_counter()
            if now - start >= 0.5:
                fps, start, count = count / (now - start), now, 0
            if letter != candidate:
                candidate, since, added = letter, now, False
            progress = min(1.0, now - since) if letter else 0.0
            if progress >= 1 and not added:
                word += letter
                added = True
            cv2.imshow("ASL fingerspelling", draw_preview(frame, letter, confidence,
                                                          progress, word, fps, name, latency, laptop_words))
            key = cv2.waitKey(1) & 0xff
            if laptop_words and key in (ord("m"), ord("M")):
                worker = WordWorker()
            elif laptop_words and key in (10, 13):
                speak(word)
            elif key == 32 and word and not word.endswith(" "):
                word += " "
            elif key in (8, 127):
                word = word[:-1]
            elif key in (27, ord("q")):
                return 0
    finally:
        camera.release()
        if worker is not None:
            worker.close()
        backend.close()
        cv2.destroyAllWindows()


def main() -> int:
    if not MODEL.is_file():
        print("Missing classifier.onnx. Run src/train.py first.", file=sys.stderr)
        return 1
    return run_preview(LocalInference(), laptop_words=True)


if __name__ == "__main__":
    raise SystemExit(main())
