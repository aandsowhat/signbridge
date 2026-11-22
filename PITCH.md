# PITCH.md — landing page and presentation outline

Fill in the project name and team before use. Keep the story simple: a real problem, a working demo, one clear differentiator (on-device).

## One-liner
Real-time ASL fingerspelling translator that runs entirely on-device: camera in, text and speech out, no cloud.

## Slide deck (5-6 slides, about 2-3 minutes)
1. **Title:** project name, one-liner, your name.
2. **Problem:** many deaf and hard-of-hearing people face communication gaps with people who don't sign. Interpreters aren't always available, and cloud tools raise privacy and latency concerns.
3. **Solution (live demo):** show the camera, spell a word, hear it spoken. Have the backup video ready.
4. **How it works:** webcam, OpenCV, MediaPipe hand landmarks (21 points), normalization, classifier, smoothing, text and speech. One simple diagram.
5. **Why on-device:** private (video never leaves the device), works offline, low latency. If the QNX port works: runs on a Raspberry Pi 5 with QNX, a real-time OS built for reliability.
6. **What's next:** full A-Z including motion letters (J, Z), common words and phrases, two-way translation (speech to text), more sign languages.

## Landing page sections
- Hero: name, one-liner, demo video, "Try it / View on GitHub" buttons
- The problem
- How it works (3 steps with icons)
- Why on-device
- Tech stack
- Team
- Footer with repo link and contact

## Demo script (60-90 seconds)
1. "This is [name]. Watch." Sign the letters of a short word (e.g. H-E-L-L-O), word builds on screen.
2. Trigger speech: it says the word out loud.
3. Show the hand tracking overlay and say one line about how it works.
4. Mention it runs offline (turn off wifi if the setup allows).
5. Close with the "what's next" line.

## Judging talking points
- Technical: custom data collection, landmark normalization, smoothing for stability.
- Impact: accessibility, privacy.
- Reliability: runs offline, real-time, no external dependencies.
- QNX (if done): deterministic real-time OS on an embedded board, with AI inference running locally.

## Checklist before submitting
- [ ] Repo public with a README and setup steps
- [ ] Landing page link works on a phone
- [ ] Demo video uploaded (under 2 minutes)
- [ ] Deck exported to PDF
- [ ] Submission includes team names, tech used, and all links
