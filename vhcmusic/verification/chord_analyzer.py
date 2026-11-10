"""Duration-weighted MIDI chord recognition with fixed templates and Viterbi decoding."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vhcmusic.representation.event_grid import EventGrid
from vhcmusic.representation.harmony import ROOT_NAMES, QUALITY_TEMPLATES, STATES, TEMPLATES

CHORD_CALIBRATION = "outputs/checkpoints/chord_calibration.json"


@dataclass
class ChordCandidate:
    root: int
    quality: str
    score: float

    @property
    def label(self) -> str:
        return "N:none" if self.root < 0 else f"{ROOT_NAMES[self.root]}:{self.quality}"


class ChordAnalyzer:
    def __init__(self, qualities: tuple[str, ...] | None = None, transition_scale: float = 1.0):
        self.qualities = qualities or tuple(QUALITY_TEMPLATES)
        self.states = [state for state in STATES if state[1] in self.qualities]
        self.templates = TEMPLATES[[STATES.index(state) for state in self.states]]
        self.states.append((-1, "none"))
        self.transition = transition_scale * np.array([
            [self._transition_penalty(a, b) for b in self.states] for a in self.states
        ])

    @classmethod
    def from_calibration(cls, path=None):
        payload = json.loads(Path(path or CHORD_CALIBRATION).read_text(encoding="utf-8"))
        if payload["dataset"] != "McGill Billboard" or payload["split"] != "val":
            raise ValueError("Chord calibration requires the McGill Billboard development split")
        return cls(transition_scale=float(payload["transition_scale"]))

    def pitch_class_activation(self, grid: EventGrid) -> tuple[np.ndarray, np.ndarray]:
        pc = np.zeros((grid.bars, 12))
        bass = np.full(grid.bars, -1)
        lowest = np.full(grid.bars, 128)
        for i, role in enumerate(grid.roles):
            if role == "drums":
                continue
            for bar, step, pitch_idx in np.argwhere(grid.data[i, ..., 0] > 0.5):
                pitch = grid.pitch_min + pitch_idx
                onset = bar * grid.subdivisions + step
                end = onset + max(1, int(round(grid.data[i, bar, step, pitch_idx, 1])))
                for j in range(bar, min(grid.bars, (end - 1) // grid.subdivisions + 1)):
                    span = min(end, (j + 1) * grid.subdivisions) - max(onset, j * grid.subdivisions)
                    pc[j, pitch % 12] += span
                    if pitch < lowest[j]:
                        lowest[j], bass[j] = pitch, pitch % 12
        return pc / np.maximum(pc.sum(axis=1, keepdims=True), 1.0), bass

    def candidates(self, pc: np.ndarray, bass: int, top_k: int = 6) -> list[ChordCandidate]:
        norm = np.linalg.norm(pc)
        if norm <= 1e-8:
            return [ChordCandidate(-1, "none", 1.0)]
        scores = self.templates @ (pc / norm)
        scores += [0.08 if root == bass else 0.0 for root, _ in self.states[:-1]]
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [ChordCandidate(*self.states[i], float(scores[i])) for i in order]

    def analyze(self, grid: EventGrid) -> dict:
        pc, bass = self.pitch_class_activation(grid)
        indices = {state: i for i, state in enumerate(self.states)}
        scores = np.full((grid.bars, len(self.states)), -np.inf)
        margins = np.zeros(grid.bars)
        top = np.zeros(grid.bars, dtype=int)
        for bar, (profile, bass_note) in enumerate(zip(pc, bass)):
            candidates = self.candidates(profile, int(bass_note), top_k=len(self.states))
            for candidate in candidates:
                scores[bar, indices[(candidate.root, candidate.quality)]] = candidate.score
            top[bar] = indices[(candidates[0].root, candidates[0].quality)]
            margins[bar] = candidates[0].score - candidates[1].score if len(candidates) > 1 else 1.0
        back = np.zeros_like(scores, dtype=int)
        previous = scores[0]
        for t in range(1, grid.bars):
            routes = previous[:, None] - self.transition
            back[t] = routes.argmax(axis=0)
            previous = scores[t] + routes.max(axis=0)
        path = [int(previous.argmax())]
        for t in range(grid.bars - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        outputs = []
        for bar, state in enumerate(reversed(path)):
            root, quality = self.states[state]
            name = "N" if root < 0 else ROOT_NAMES[root]
            outputs.append({"bar_id": bar + 1, "chord": f"{name}:{quality}", "root": name,
                            "quality": quality, "confidence": float(margins[bar]) if state == top[bar] else 0.0})
        return {"chords": outputs, "pitch_class_activation": pc.tolist(), "bass_pc": bass.tolist()}

    @staticmethod
    def _transition_penalty(a, b):
        root_a, quality_a = a
        root_b, quality_b = b
        if -1 in (root_a, root_b):
            return 0.0
        distance = min((root_a - root_b) % 12, (root_b - root_a) % 12)
        penalty = 0.02 * distance + 0.03 * (quality_a != quality_b)
        return penalty * 0.5 if distance == 5 else penalty
