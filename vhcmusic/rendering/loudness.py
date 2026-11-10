"""Loudness normalization utilities."""

from __future__ import annotations

import numpy as np


def peak_normalize(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs <= 1e-8:
        return audio
    return audio * (peak / max_abs)


def loudness_normalize(audio: np.ndarray, sample_rate: int, target_lufs: float = -14.0, ceiling: float = 0.98) -> np.ndarray:
    import pyloudnorm as pyln

    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    if not np.isfinite(loudness):
        return audio
    gain = 10 ** ((target_lufs - loudness) / 20)
    peak = np.max(np.abs(audio))
    return audio * min(gain, ceiling / peak)
