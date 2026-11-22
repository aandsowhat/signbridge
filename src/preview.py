"""OpenCV layout for the live fingerspelling demo."""
import cv2
import numpy as np

INK = (42, 34, 25)
WHITE = (246, 247, 243)
MUTED = (193, 185, 170)
MINT = (174, 225, 108)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_preview(frame: np.ndarray, letter: str, confidence: float, progress: float,
                 word: str, fps: float, source: str = "Local", latency: float | None = None,
                 laptop_words: bool = False) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(640 / width, 480 / height)
    w, h = round(width * scale), round(height * scale)
    canvas = np.full((620, 880, 3), INK, dtype=np.uint8)
    x, y = (640 - w) // 2, (480 - h) // 2
    canvas[y:y+h, x:x+w] = cv2.resize(frame, (w, h))

    def text(value, x, y, size, color=WHITE, weight=1):
        cv2.putText(canvas, value, (x, y), FONT, size, color, weight, cv2.LINE_AA)

    def bar(y, fraction, color):
        cv2.rectangle(canvas, (668, y), (852, y+6), (80, 71, 60), -1)
        length = round(184 * min(1, max(0, fraction)))
        if length:
            cv2.rectangle(canvas, (668, y), (668+length, y+6), color, -1)

    text("PREDICTION", 668, 48, .56, MUTED)
    text(letter or "-", 665, 195, 4.4, MINT, 6)
    text(f"Confidence  {confidence:.0%}" if letter else "No hand", 668, 256, .57)
    bar(272, confidence, MINT)
    text("Added" if progress >= 1 else "Hold to add", 668, 343, .57)
    bar(359, progress, WHITE)
    caption = f"Inference: {source}"
    text(caption, 668, 412, .46, MUTED)
    if latency is not None:
        end = 668 + cv2.getTextSize(caption, FONT, .46, 1)[0][0]
        cv2.circle(canvas, (end + 6, 408), 1, MUTED, -1)
        text(f"{latency:.0f} ms", end + 12, 412, .46, MUTED)
    text(f"{fps:.0f} FPS", 668, 454, .48, MUTED)
    cv2.line(canvas, (0, 480), (879, 480), (80, 71, 60))
    text("FINGERSPELLING", 24, 507, .48, MINT)
    tail = word[-40:]
    while cv2.getTextSize(tail, FONT, 1.1, 2)[0][0] > 795:
        tail = tail[1:]
    text(("..." if len(tail) < len(word) else "") + tail if word else "Hold your first letter",
         24, 555, 1.1, WHITE if word else MUTED, 2)
    if laptop_words:
        text("M: word mode", 668, 375, .48, MUTED)
    text("Hold 1s to add   |   Space: separate words   |   Backspace: delete   |   Esc: quit",
         24, 599, .54, MUTED)
    return canvas
