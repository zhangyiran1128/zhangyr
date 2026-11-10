"""Control-following metrics."""

from __future__ import annotations

import numpy as np


def accuracy(pred, target) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.size == 0:
        return 0.0
    return float((pred == target).mean())


def section_boundary_f1(target_boundaries, predicted_boundaries, tolerance: int = 1) -> dict[str, float]:
    target = sorted(set(map(int, target_boundaries)))
    pred = sorted(set(map(int, predicted_boundaries)))
    i = j = tp = 0
    while i < len(target) and j < len(pred):
        delta = pred[j] - target[i]
        if abs(delta) <= tolerance:
            tp += 1
            i += 1
            j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    if not target and not pred:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    return {"precision": tp / max(1, len(pred)), "recall": tp / max(1, len(target)),
            "f1": 2 * tp / (len(target) + len(pred))}


def rhythm_mae(target_density, predicted_density) -> float:
    target = np.asarray(target_density, dtype=np.float32)
    pred = np.asarray(predicted_density, dtype=np.float32)
    n = min(target.size, pred.size)
    if n == 0:
        return 0.0
    return float(np.abs(target[:n] - pred[:n]).mean())


def active_role_f1(target_activity: np.ndarray, predicted_activity: np.ndarray) -> dict[str, float]:
    target = (np.asarray(target_activity) > 0.5)
    pred = (np.asarray(predicted_activity) > 0.5)
    tp = float((target & pred).sum())
    fp = float((~target & pred).sum())
    fn = float((target & ~pred).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def tempo_accuracy(target_bpm, predicted_bpm, tolerance: float = 3.0) -> float:
    target = np.asarray(target_bpm, dtype=np.float32)
    pred = np.asarray(predicted_bpm, dtype=np.float32)
    n = min(target.size, pred.size)
    return float((np.abs(target[:n] - pred[:n]) <= tolerance).mean()) if n else 0.0


beat_accuracy = tempo_accuracy
key_accuracy = accuracy
chord_accuracy = accuracy
