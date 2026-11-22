"""Collect labeled hand landmarks from the webcam."""
import csv
import sys
import tkinter as tk
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from string import ascii_uppercase
from time import monotonic

import cv2

from landmarks import HandTracker, normalize

DATA = Path(__file__).resolve().parents[1] / "data" / "landmarks.csv"
HEADER = ["label", *(f"{axis}{i}" for i in range(21) for axis in "xyz")]


def main() -> int:
    with ExitStack() as cleanup:
        camera = cv2.VideoCapture(0)
        cleanup.callback(camera.release)
        if not camera.isOpened():
            print("No webcam found. Check the connection and camera permissions.", file=sys.stderr)
            return 1
        tracker = HandTracker()
        cleanup.callback(tracker.close)
        DATA.parent.mkdir(exist_ok=True)
        output = cleanup.enter_context(DATA.open("a+", newline="", encoding="utf-8"))
        output.seek(0)
        rows = csv.reader(line.removeprefix("\ufeff") for line in output)
        header = next(rows, None)
        if header is not None and header != HEADER:
            print("Unexpected header in landmarks.csv. Check the file before recording.", file=sys.stderr)
            return 1
        counts = Counter(row[0].removeprefix("\ufeff") for row in rows if row)
        writer = csv.writer(output)
        if header is None:
            writer.writerow(HEADER)
            output.flush()

        window = tk.Tk()
        cleanup.callback(window.destroy)
        window.title("Hold a letter to record | Esc to quit")
        preview = tk.Label(window)
        preview.pack()
        status = tk.Label(window, wraplength=540)
        status.pack()
        held = ""
        deadline = 0.0

        def press(event: tk.Event) -> None:
            nonlocal held, deadline
            key = event.keysym.upper()
            if len(key) == 1 and key in ascii_uppercase and key != held:
                held = key
                deadline = 0.0

        def release(event: tk.Event) -> None:
            nonlocal held
            if event.type == tk.EventType.FocusOut or event.keysym.upper() == held:
                held = ""

        window.bind("<KeyPress>", press)
        window.bind("<KeyRelease>", release)
        window.bind("<FocusOut>", release)
        closing = False

        def stop(event: tk.Event | None = None) -> None:
            nonlocal closing
            closing = True

        window.bind("<Escape>", stop)
        window.protocol("WM_DELETE_WINDOW", stop)
        window.update()
        window.focus_force()
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Webcam stopped delivering frames.", file=sys.stderr)
                return 1
            points, frame = tracker.process(cv2.flip(frame, 1))
            now = monotonic()
            if held and points is not None and now >= deadline:
                try:
                    features = normalize(points)
                except ValueError:
                    pass
                else:
                    writer.writerow([held, *features])
                    output.flush()
                    counts[held] += 1
                    deadline = max(now + 0.05, deadline + 0.1) if deadline else now + 0.1
            image = tk.PhotoImage(data=cv2.imencode(".ppm", frame)[1].tobytes(), format="PPM")
            preview.configure(image=image)
            message = "Show one hand" if points is None else (f"Recording {held}" if held else "Hold a letter")
            status.configure(text=message + "\n" + "   ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
            window.update()
            if closing:
                return 0


if __name__ == "__main__":
    raise SystemExit(main())

