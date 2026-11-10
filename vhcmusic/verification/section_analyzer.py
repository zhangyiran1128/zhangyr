"""Joint prediction of section boundaries and structural functions from MIDI."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.signal import find_peaks

from vhcmusic.planning.schema import SECTION_LABELS
from vhcmusic.representation.event_grid import EventGrid
from vhcmusic.utils.checkpoint import load_checkpoint

SECTION_CHECKPOINT = "outputs/checkpoints/section_analyzer.pt"
LABELS = (*SECTION_LABELS, "other")


def section_features(grid: EventGrid) -> np.ndarray:
    onsets = grid.data[..., 0] > 0.5
    bars = grid.bars
    counts = onsets.sum(axis=(0, 2, 3))
    activity = onsets.any(axis=(2, 3)).T
    pitch_hist = onsets[[r != "drums" for r in grid.roles]].sum(axis=(0, 2))
    chroma = np.zeros((bars, 12))
    for i, hist in enumerate(pitch_hist.T):
        chroma[:, (i + grid.pitch_min) % 12] += hist
    chroma /= np.maximum(chroma.sum(axis=1, keepdims=True), 1)
    profiles = onsets.transpose(1, 0, 2, 3).reshape(bars, -1).astype(np.float32)
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1)
    similarities = profiles @ profiles.T
    np.fill_diagonal(similarities, 0)
    repetition = similarities.max(axis=1)
    change = np.r_[0, np.abs(np.diff(chroma, axis=0)).sum(axis=1) / 2]
    velocity = grid.data[..., 2] * onsets
    mean = velocity.sum(axis=(0, 2, 3)) / np.maximum(counts, 1)
    std = np.sqrt(np.maximum((velocity**2).sum(axis=(0, 2, 3)) / np.maximum(counts, 1) - mean**2, 0))
    span = [np.ptp(pitches) if len(pitches) else 0 for pitches in map(np.flatnonzero, pitch_hist)]
    rhythmic = onsets.any(axis=(0, 3)).mean(axis=1)
    return np.column_stack((np.log1p(counts), activity, mean, std, np.asarray(span) / 87,
                            chroma, change, rhythmic, repetition)).astype(np.float32)


class SectionPredictor(nn.Module):
    def __init__(self, features: int, width: int = 128):
        super().__init__()
        self.input = nn.Linear(features, width)
        self.temporal = nn.GRU(width, width, num_layers=2, dropout=0.1, bidirectional=True, batch_first=True)
        self.boundary = nn.Linear(width * 2, 1)
        self.function = nn.Linear(width * 2, len(LABELS))

    def forward(self, x: torch.Tensor):
        hidden, _ = self.temporal(nn.functional.gelu(self.input(x)))
        return self.boundary(hidden).squeeze(-1), self.function(hidden)


def boundary_peaks(probabilities, threshold: float, gap: int) -> list[int]:
    # Pad the last bar so a peak immediately before the end can be detected.
    x = np.r_[0.0, np.asarray(probabilities)[1:], 0.0]
    peaks, _ = find_peaks(x, height=threshold, distance=gap)
    return [1, *(int(i) + 1 for i in peaks if i > 0)]


class SectionAnalyzer:
    def __init__(self, checkpoint: str | Path = SECTION_CHECKPOINT):
        payload = load_checkpoint(checkpoint)
        if payload["dataset"] != "SALAMI" or tuple(payload["labels"]) != LABELS:
            raise ValueError("Section Analyzer requires a compatible SALAMI checkpoint")
        self.mean = torch.as_tensor(payload["mean"], dtype=torch.float32)
        self.std = torch.as_tensor(payload["std"], dtype=torch.float32)
        self.threshold = float(payload["threshold"])
        self.gap = int(payload["min_gap"])
        self.model = SectionPredictor(len(self.mean), payload["width"])
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def analyze(self, grid: EventGrid) -> dict:
        features = torch.from_numpy(section_features(grid))
        logits, labels = self.model(((features - self.mean) / self.std)[None])
        scores = logits[0].sigmoid().numpy()
        label_probs = labels[0].softmax(-1).numpy()
        starts = boundary_peaks(scores, self.threshold, self.gap)
        confidences = {bar: float(scores[bar - 1]) for bar in starts}
        confidences[1] = 1.0
        sections = [
            {"section": LABELS[int(label_probs[start - 1:end - 1].mean(axis=0).argmax())],
             "start_bar": start, "end_bar": end - 1, "boundary_confidence": confidences[start]}
            for start, end in zip(starts, starts[1:] + [grid.bars + 1])
        ]
        return {"boundaries": starts, "sections": sections, "boundary_confidence": confidences}
