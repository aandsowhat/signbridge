"""Shared preprocessing for a single hand."""
import numpy as np


def normalize(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
    centered = points - points[0]
    scale = np.linalg.norm(centered[9])
    if scale <= 1e-8:
        raise ValueError("Wrist and middle-finger MCP must be distinct.")
    return (centered / scale).ravel()
