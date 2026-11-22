# AGENTS.md — project context for Codex / ChatGPT

Read this file first in every session. It is the source of truth for scope and conventions.

## What we're building
A real-time ASL fingerspelling translator. A webcam sees a signer's hand, the app recognizes the letter, builds words, shows them on screen and speaks them aloud. Runs fully on-device (no cloud, no internet) so it is private and low-latency.

Hackathon: built from Fri 9pm to Sun 9am (36 hours). Deliverables: **working demo, landing page, presentation**. Solo builder (student), so keep everything small and shippable.

## Optional bonus track: QNX challenge
Prize challenge: build an embedded system on the **QNX OS** that uses **one of the open-source AI modules from https://oss.qnx.com**. Judged on: "cannot-fail" embedded app, needs real-time/reliability, creative AI use, runs on the embedded hardware (Raspberry Pi 5 loaner, pre-loaded with QNX), not cloud.
- This is a bonus. The laptop demo is the primary deliverable and must work on its own.
- Keep inference portable: train on the laptop, export the classifier to **ONNX**, run it with ONNX Runtime so the same model can run on the Pi.
- Known risks: camera support on QNX/Pi 5 (USB camera quirks; Pi Camera 3 needs extra packages), and MediaPipe may not be installable via apk. Fallback for hand tracking on QNX: a small ONNX hand-landmark or CNN model.

## Stack
- Python 3.10+
- OpenCV (`opencv-python`): webcam capture, drawing, display
- MediaPipe Hands: 21 hand landmarks per frame (OpenCV alone does not do hand tracking)
- scikit-learn (RandomForest or small MLP) for the classifier; `skl2onnx` for ONNX export
- `pyttsx3` (offline) for text-to-speech
- UI: simple OpenCV window first; Streamlit or Flask only if time allows
- Landing page: single static HTML/CSS (no framework), deployed to Vercel or GitHub Pages

## Scope (keep it tight)
- Start with a small letter set (about 8-10 static letters, e.g. A B C D E I L O U Y), then expand toward A-Y.
- J and Z need motion: out of scope.
- Extras, in order: confidence display, hold-to-confirm to add a letter, space and delete gestures, text-to-speech.
- Do NOT add: accounts, databases, cloud APIs, mobile apps, multi-hand or full-sentence ASL grammar.

## Repo layout
```
.
├── AGENTS.md
├── PLAN.md
├── PROMPTS.md
├── PITCH.md
├── requirements.txt
├── data/               # landmarks.csv (label + 63 features)
├── models/             # classifier.pkl, classifier.onnx
├── src/
│   ├── landmarks.py    # MediaPipe wrapper + normalization
│   ├── collect_data.py # press key to save labeled samples
│   ├── train.py        # train, report accuracy, export
│   ├── infer.py        # live prediction + smoothing + word builder
│   └── tts.py
└── web/                # landing page (index.html, style.css, demo.mp4)
```

## Conventions
- Normalize landmarks: subtract the wrist (landmark 0), divide by a hand-size scale (e.g. wrist to middle-finger MCP distance). Flatten to 63 features (x, y, z). Use the SAME function for training and inference.
- Smooth predictions with a majority vote over the last ~10 frames; only accept a letter after ~1 second of stable hold.
- Target 20+ FPS. Keep per-frame work light.
- Small functions, type hints, short comments. No heavy dependencies beyond the stack above.
- Every task must end with something runnable. Print clear errors (e.g. "no webcam found").
- Commit after each working step.

## Definition of done
1. `python src/infer.py` opens the webcam and shows the predicted letter with confidence.
2. Held letters build a word on screen and can be spoken.
3. Works reliably for a different person's hand and in different lighting (test this).
4. Landing page deployed with an embedded demo video.
5. 5-6 slide deck plus a recorded backup demo video.
