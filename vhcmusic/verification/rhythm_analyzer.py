"""Symbolic rhythm analyzer."""

from __future__ import annotations

import numpy as np

from vhcmusic.representation.event_grid import EventGrid, event_grid_to_pianoroll


class RhythmAnalyzer:
    """Extract rhythm density, syncopation, and track activity curves."""

    def __init__(self, density_scale: float = 24.0):
        if density_scale <= 0:
            raise ValueError("density_scale must be positive")
        self.density_scale = density_scale

    def analyze(self, grid: EventGrid) -> dict:
        data = grid.data
        onsets = data[..., 0] > 0.5
        velocities = data[..., 2]
        r, l, s, p, _ = data.shape
        onset_count = onsets.sum(axis=(0, 2, 3)).astype(np.float32)
        velocity_weighted = (onsets * velocities).sum(axis=(0, 2, 3)) / max(1.0, r * s)
        active_track = (onsets.sum(axis=(2, 3)) > 0).astype(np.float32)
        active_track_count = active_track.sum(axis=0)
        roll = event_grid_to_pianoroll(grid)
        coverage = roll.reshape(r, l, s, p).sum(axis=(0, 2, 3)) / max(1.0, r * s * p)
        sync = np.zeros(l, dtype=np.float32)
        weak_steps = np.ones(s, dtype=np.float32)
        weak_steps[:: max(1, s // 4)] = 0.0
        weak_steps[(s // 2) :: max(1, s // 4)] *= 0.5
        for bar in range(l):
            step_counts = onsets[:, bar].sum(axis=(0, 2)).astype(np.float32)
            total = step_counts.sum()
            sync[bar] = float((step_counts * weak_steps).sum() / max(1.0, total))
        onset_times = np.flatnonzero(onsets.any(axis=(0, 3)).reshape(l * s))
        ioi = np.diff(onset_times).astype(np.float32)
        hist = np.histogram(ioi, bins=np.arange(1, s + 2), density=False)[0].astype(np.float32) if ioi.size else np.zeros(s, dtype=np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        density_curve = np.clip(onset_count / (grid.beats_per_bar * 4 / grid.beat_unit) / self.density_scale, 0, 1)
        rhythmic_change = np.zeros(l, dtype=np.float32)
        step_profiles = onsets.sum(axis=3).transpose(1, 0, 2).reshape(l, r * s).astype(np.float32)
        for bar in range(1, l):
            rhythmic_change[bar] = float(np.linalg.norm(step_profiles[bar] - step_profiles[bar - 1]) / max(1.0, np.linalg.norm(step_profiles[bar - 1]) + np.linalg.norm(step_profiles[bar])))
        return {
            "onset_count": onset_count.tolist(),
            "duration_coverage": coverage.tolist(),
            "velocity_weighted_density": velocity_weighted.tolist(),
            "active_track_count": active_track_count.tolist(),
            "syncopation_proxy": sync.tolist(),
            "ioi_distribution": hist.tolist(),
            "rhythm_density_curve": density_curve.tolist(),
            "track_level_activity": active_track.tolist(),
            "rhythmic_change_curve": rhythmic_change.tolist(),
        }
