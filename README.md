# ASL fingerspelling demo

From this folder on Windows, use Python 3.11 and a webcam.

    py -3.11 -m venv .venv
    .\.venv\Scripts\python -m pip install -r requirements.txt
    .\.venv\Scripts\python src/landmarks.py

Hold one hand in view. Press q in the preview window to quit.

To collect training samples, run `.\.venv\Scripts\python src/collect_data.py`.
Focus the preview and hold a letter key while signing it. Release to pause or
press Esc to quit. Counts include previous sessions in `data/landmarks.csv`.
Run live letter recognition with `.\.venv\Scripts\python.exe src/infer.py`. Press Q or Esc to quit.
Hold a letter for one second to add it. For a double letter, lower your hand briefly and sign it again. Space separates words and Backspace deletes the last character.
 
## QNX webcam inference

From laptop PowerShell, run `.\deploy.ps1 user@PI_IP` with the actual SSH user and IP.
On the Pi, run `cd ~/asl-demo` and `python3 pi/server.py`.
From a laptop terminal in this repository, run
`.\.venv\Scripts\python.exe src/infer_remote.py PI_IP`.

The app sends mirrored JPEG frames to TCP port 5005. Both packets use a four-byte
big-endian length prefix. The displayed milliseconds cover JPEG decoding through
classification on the server. If the connection fails, the app uses local inference
for the rest of the session. Restart the client to reconnect.

## Laptop isolated-sign mode

The pretrained Kaggle winning model and its 250-label index are included in
`models/words/`. Install the optional runtime once in the laptop environment.

    .\.venv\Scripts\python.exe -m pip install ai-edge-litert==2.4.0
    .\.venv\Scripts\python.exe src/infer.py

The app starts in fingerspelling. M switches modes using the same webcam.
In word mode, keep the upper body, face and both hands visible. Perform one
isolated sign, watch the current label, and tap Space while it says Ready.
Lower the hands out of tracking between repetitions. Backspace removes one
confirmed sign, C clears the sentence, Enter requests offline speech, and Esc
quits. In fingerspelling, Space still separates words and Enter reads the text.
If the word model or speech engine fails, see the terminal for its error.

Requested labels present: hello (113), no (153), thankyou (214), water (234),
yes (244). ME/I, YOU, WANT, HELP and EAT have no exact label. All 250 labels
can compete. Display names use capitals and expand thankyou to THANK YOU.

The laptop worker uses unmirrored RGB frames and MediaPipe Holistic image
landmarks. Each float32 frame has 543 XYZ rows in this order: face (468),
left hand (21), pose (33), right hand (21). Missing groups contain NaNs.
Raw landmarks go directly into serving_default(inputs), shape [T,543,3].
The graph handles feature preprocessing internally. The output is 250 raw
scores, indexed by labels.json. The displayed score is a relative softmax
score. Calibration and real-world accuracy have not been established.
The rolling window holds up to 24 processed frames. Starting at 8 frames,
every third frame is evaluated. Confirmation requires three consecutive
agreeing predictions with score at least 0.55. These are demo heuristics.

Confirmed glosses are displayed separately. The English line appears only
for explicitly mapped phrases (for example WATER PLEASE becomes Water, please.).
Other sequences stay as glosses. Word mode runs offline on the Windows laptop.
The Pi client and server keep their existing fingerspelling roles.

Model and vocabulary source: https://huggingface.co/sign/kaggle-asl-signs-1st-place
(model card lists MIT). Model SHA-256:
`f55a2bb1ebe6d1e912a98e31c6ef3f995c9ae261f408fe115f2008099d3f0bb7`.
