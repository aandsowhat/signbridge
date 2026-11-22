"""Single-hand landmark tracking and webcam preview."""
import sys
from time import perf_counter

import cv2
import numpy as np


from normalization import normalize


class HandTracker:
    def __init__(self) -> None:
        from mediapipe import solutions

        self.api = solutions.hands
        self.draw = solutions.drawing_utils
        self.hands = self.api.Hands(max_num_hands=1, model_complexity=0)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return None, frame
        hand = result.multi_hand_landmarks[0]
        points = np.array([(p.x, p.y, p.z) for p in hand.landmark], dtype=np.float32)
        self.draw.draw_landmarks(frame, hand, self.api.HAND_CONNECTIONS)
        return points, frame

    def close(self) -> None:
        self.hands.close()


def main() -> int:
    camera = cv2.VideoCapture(0)
    tracker = None
    try:
        if not camera.isOpened():
            print("No webcam found. Connect a camera and check camera permissions.", file=sys.stderr)
            return 1
        tracker = HandTracker()
        start, count, fps = perf_counter(), 0, 0.0
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Webcam stopped delivering frames.", file=sys.stderr)
                return 1
            _, frame = tracker.process(cv2.flip(frame, 1))
            count += 1
            now = perf_counter()
            if now - start >= 0.5:
                fps, start, count = count / (now - start), now, 0
            cv2.putText(frame, f"{fps:.0f} FPS", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.imshow("Hand landmarks", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0
    finally:
        camera.release()
        if tracker is not None:
            tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
