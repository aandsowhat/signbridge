# PROMPTS.md — paste these into Codex / ChatGPT one at a time

Start every session with: "Read AGENTS.md and PLAN.md first, then do the task below. Keep changes small and make sure it runs."
Test each step yourself before moving on, and commit after each one.

## PROMPT 0: scaffold
Create the repo layout from AGENTS.md with empty files, a `requirements.txt` (opencv-python, mediapipe, scikit-learn, numpy, pandas, pyttsx3, skl2onnx, onnxruntime), and a short README with run commands.
Done when: `pip install -r requirements.txt` succeeds.

## PROMPT 1: landmarks
In `src/landmarks.py`, wrap MediaPipe Hands so it takes a BGR frame and returns the 21 landmarks (or None) plus a drawn frame. Add a `normalize(landmarks)` function that subtracts the wrist and divides by wrist to middle-finger-MCP distance, returning 63 floats. In a small main block, show the webcam feed with the landmarks drawn and an FPS counter. Press `q` to quit.
Done when: it runs at 20+ FPS and prints a clear error if no webcam is found.

## PROMPT 2: data collector
In `src/collect_data.py`: show the webcam with landmarks; when I press a letter key, append the normalized 63 features plus that label to `data/landmarks.csv`. Hold the key to record continuously, about 10 samples per second. Show the count per label on screen. Press `Esc` to quit. Create the CSV with a header if it doesn't exist.
Done when: I can collect 200 samples for a letter in about 20 seconds.

## PROMPT 3: trainer
In `src/train.py`: load the CSV, split train/test, train a RandomForest (and compare with a small MLP), print overall and per-class accuracy plus a confusion matrix, save the best model to `models/classifier.pkl`, and export it to `models/classifier.onnx` with skl2onnx. Verify the ONNX model gives the same predictions as the pickle on the test set.
Done when: it prints accuracy and writes both model files.

## PROMPT 4: live inference
In `src/infer.py`: load the model, run the webcam loop, and show the predicted letter and confidence on screen. Smooth with a majority vote over the last 10 frames. Show "no hand" when nothing is detected.
Done when: it shows stable predictions at 20+ FPS.

## PROMPT 5: word builder and speech
Extend `infer.py`: add the letter to a word buffer after it is held stable for 1 second (show a progress bar for the hold). Add gestures or keys for space and delete. Add `src/tts.py` with an offline `speak(text)` using pyttsx3; speak the word on a "speak" key or gesture, without blocking the video loop (use a thread).
Done when: I can spell a word and hear it.

## PROMPT 6: demo UI
Make the live view look good: a large predicted letter, a confidence bar, the word being built along the bottom, a short help line with the keys, and a clean color theme. Keep it in OpenCV unless a Streamlit version takes under 30 minutes.
Done when: it looks good enough to screen-record for the demo video.

## PROMPT 7: landing page
In `web/`, build a single static page (HTML + CSS, no framework): hero with the project name and a one-line pitch, the problem (communication gaps with deaf and hard-of-hearing people), how it works in 3 steps (camera, hand tracking, letter recognition and speech), an embedded demo video, a "runs fully on-device" section, the tech stack, and the team. Responsive and accessible. Give me the steps to deploy to Vercel or GitHub Pages.
Done when: it works on a phone-sized screen.

## PROMPT 8 (QNX track, only if time allows)
Write a minimal script that loads `models/classifier.onnx` with ONNX Runtime and predicts from a normalized 63-float vector. Then adapt the capture loop to the camera that works on the QNX Raspberry Pi 5. If MediaPipe is not available there, propose the smallest alternative (an ONNX hand-landmark model, or a tiny CNN on cropped hand images) and implement it.
Done when: the Pi shows live predictions from the camera.
