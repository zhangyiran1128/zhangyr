"""Back-translation verification metrics."""

from __future__ import annotations

import numpy as np

from .control_metrics import accuracy, rhythm_mae, section_boundary_f1


def verification_f1(predicted_accept, target_accept) -> dict[str, float]:
    pred = np.asarray(predicted_accept).astype(bool)
    target = np.asarray(target_accept).astype(bool)
    tp = float((pred & target).sum())
    fp = float((pred & ~target).sum())
    fn = float((~pred & target).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def false_accept_rate(predicted_accept, target_accept) -> float:
    pred = np.asarray(predicted_accept).astype(bool)
    target = np.asarray(target_accept).astype(bool)
    failures = ~target
    if failures.sum() == 0:
        return 0.0
    return float((pred & failures).sum() / failures.sum())


boundary_f1 = section_boundary_f1
chord_acc = accuracy
density_mae = rhythm_mae
