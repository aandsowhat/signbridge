# PLAN.md — 36-hour roadmap (Fri 9pm to Sun 9am, Toronto time)

## Priorities
1. Working end-to-end demo by Saturday afternoon, even with only 5 letters.
2. Everything else (more letters, polish, QNX port) is added on top of a demo that already works.
3. Feature freeze Saturday midnight. Sunday morning is for the video, deck and submission only.

## Timeline

### Before 9pm Friday (no project code)
- Install opencv-python, mediapipe, scikit-learn, pyttsx3, skl2onnx; confirm the webcam works.
- Create the repo, add these four .md files, start the submission draft, pick a project name.
- Check the hackathon rules on pre-written code.
- Visit the QNX table: borrow a Raspberry Pi 5 kit and ask which AI packages install with `apk` (ONNX Runtime), whether OpenCV counts as the AI module, and which camera works.

### Fri 9pm to 1am: pipeline
- Webcam feed with 21 landmarks drawn live (PROMPT 1).
- Data collection script and first data (PROMPT 2). Record about 200 samples per letter, with varied angles, distances and lighting.

### Sat 1am to 3am: first model, then sleep
- Train, check accuracy, get live predictions on screen (PROMPTS 3-4).
- Sleep.

### Sat 10am to 2pm: make it feel good
- Smoothing, hold-to-confirm, word builder, space/delete, text-to-speech (PROMPT 5).
- Test with other people's hands and different lighting. Collect more data for confused letters.
- **Checkpoint:** a full demo works on the laptop. Commit and tag it.

### Sat 2pm to 7pm: three parallel streams
- **Demo polish:** clean UI (PROMPT 6), record a backup video.
- **Landing page:** build and deploy (PROMPT 7).
- **Deck:** draft from PITCH.md.
- **QNX track (time-boxed):** export to ONNX, get camera frames on the Pi, port inference. If the camera or runtime is still not working by 6pm, drop it.

### Sat 7pm to midnight: harden
- Bug fixes, edge cases (no hand in frame, two hands, low light).
- Rehearse the pitch twice. Feature freeze at midnight.

### Sun 12am to 5am
- Sleep (at least 4-5 hours).

### Sun 5am to 9am: ship
- Final demo run-through, re-record the video if needed.
- Finish the submission page, add links (repo, landing page, video).
- Submit by about 8am. Confirm the real deadline in the rules.

## Risk list and fallbacks
| Risk | Fallback |
|---|---|
| Low accuracy | Fewer letters, more data, collect with multiple people |
| Lighting differs at the judging table | Backup demo video, plain background, retrain with extra data |
| Webcam or driver issues | Backup video, second camera, phone as webcam |
| QNX port stalls | Ship the laptop demo; mention QNX as future work |
| Running out of time | Cut extras in this order: TTS, space/delete gestures, extra letters |
