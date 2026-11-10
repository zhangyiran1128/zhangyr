"""Canonical pitch classes and the fixed root-quality vocabulary."""

import numpy as np

ROOT_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
QUALITY_TEMPLATES = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10), "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
}
STATES = [(root, quality) for root in range(12) for quality in QUALITY_TEMPLATES]
TEMPLATES = np.array([[float((pitch - root) % 12 in QUALITY_TEMPLATES[quality])
                       for pitch in range(12)] for root, quality in STATES])
TEMPLATES /= np.linalg.norm(TEMPLATES, axis=1, keepdims=True)


def chord_identity(label: str):
    root, quality = label.split(":", 1)
    if root == "N":
        return -1, "none"
    pitch = ROOT_NAMES.index(root[0]) + root.count("#") - root.count("b")
    return pitch % 12, quality


def classify_chroma(chroma):
    norm = np.linalg.norm(chroma)
    if norm < 1e-8:
        return "N:none"
    root, quality = STATES[int((TEMPLATES @ (chroma / norm)).argmax())]
    return f"{ROOT_NAMES[root]}:{quality}"
