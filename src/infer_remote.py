"""Stream mirrored webcam frames to the QNX classifier."""
import argparse
import json
import math
import socket
import sys

import cv2

from frame_protocol import receive, send
from infer import LocalInference, run_preview


class RemoteInference:
    def __init__(self, host):
        self.host = host
        self.sock = None
        self.local = None

    def process(self, frame):
        if self.local is None:
            try:
                if self.sock is None:
                    self.sock = socket.create_connection((self.host, 5005), timeout=3)
                    self.sock.settimeout(5)
                h, w = frame.shape[:2]
                scale = min(320 / w, 240 / h, 1)
                small = cv2.resize(frame, (round(w * scale), round(h * scale)),
                                   interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if not ok:
                    raise ValueError("JPEG encoding failed")
                send(self.sock, jpeg.tobytes())
                reply = json.loads(receive(self.sock, 4096))
                if "error" in reply:
                    raise ValueError(reply["error"])
                letter, confidence = reply["letter"], float(reply["confidence"])
                latency = float(reply["inference_ms"])
                if (type(reply["hand_found"]) is not bool or not isinstance(letter, str)
                        or (letter not in tuple("ABCDEFGHIKLMNOPQRSTUVWXY") if reply["hand_found"] else letter != "")
                        or not 0 <= confidence <= 1 or not math.isfinite(latency) or latency < 0):
                    raise ValueError("Invalid Pi inference result")
                return letter, confidence, frame, "QNX Pi 5", latency
            except (OSError, EOFError, ValueError, KeyError, TypeError, cv2.error) as exc:
                print(f"Pi inference failed ({exc}). Switching to local inference.", file=sys.stderr)
                self.close()
                self.local = LocalInference()
        return self.local.process(frame)

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
        if self.local is not None:
            self.local.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pi_ip")
    sys.exit(run_preview(RemoteInference(parser.parse_args().pi_ip)))
