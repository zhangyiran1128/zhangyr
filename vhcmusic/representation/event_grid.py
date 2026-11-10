"""VHCMusic multitrack MIDI event-grid representation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .track_mapping import TRACK_ROLES


FEATURES = ("note_on", "duration", "velocity")


@dataclass
class EventGrid:
    """Container for `R x L x S x P x F` MIDI event tensors."""

    data: np.ndarray
    roles: tuple[str, ...] = TRACK_ROLES
    subdivisions: int = 16
    pitch_min: int = 21
    pitch_max: int = 108
    ticks_per_beat: int = 480
    beats_per_bar: int = 4
    tempo: float = 120.0
    beat_unit: int = 4

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 5:
            raise ValueError(f"EventGrid data must be 5D, got shape {self.data.shape}")
        if self.data.shape[0] != len(self.roles):
            raise ValueError("Role count does not match grid first dimension")
        if self.data.shape[2] != self.subdivisions:
            raise ValueError("subdivisions does not match grid shape")

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return self.data.shape

    @property
    def bars(self) -> int:
        return int(self.data.shape[1])

    @property
    def pitches(self) -> int:
        return int(self.data.shape[3])

    def role_index(self, role: str) -> int:
        return self.roles.index(role)

    def copy(self) -> "EventGrid":
        return EventGrid(
            data=self.data.copy(),
            roles=self.roles,
            subdivisions=self.subdivisions,
            pitch_min=self.pitch_min,
            pitch_max=self.pitch_max,
            ticks_per_beat=self.ticks_per_beat,
            beats_per_bar=self.beats_per_bar,
            tempo=self.tempo,
            beat_unit=self.beat_unit,
        )

    def metadata(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "data"}


def empty_event_grid(
    bars: int,
    roles: tuple[str, ...] = TRACK_ROLES,
    subdivisions: int = 16,
    pitch_min: int = 21,
    pitch_max: int = 108,
    feature_dim: int = len(FEATURES),
) -> EventGrid:
    pitches = int(pitch_max - pitch_min + 1)
    data = np.zeros((len(roles), bars, subdivisions, pitches, feature_dim), dtype=np.float32)
    return EventGrid(data=data, roles=roles, subdivisions=subdivisions, pitch_min=pitch_min, pitch_max=pitch_max)


def event_grid_to_pianoroll(grid: EventGrid) -> np.ndarray:
    """Return binary pianoroll `(R, L*S, P)` from note-on/duration events."""

    r, l, s, p, _ = grid.shape
    roll = np.zeros((r, l * s, p), dtype=np.float32)
    onsets = grid.data[..., 0] > 0.5
    durations = np.maximum(1, np.rint(grid.data[..., 1]).astype(int))
    for role in range(r):
        for bar in range(l):
            for step in range(s):
                for pitch in np.flatnonzero(onsets[role, bar, step]):
                    start = bar * s + step
                    end = min(l * s, start + int(durations[role, bar, step, pitch]))
                    roll[role, start:end, pitch] = 1.0
    return roll


def pianoroll_to_event_grid(
    roll: np.ndarray,
    bars: int,
    roles: tuple[str, ...] = TRACK_ROLES,
    subdivisions: int = 16,
    pitch_min: int = 21,
    pitch_max: int = 108,
    velocity: float = 0.75,
) -> EventGrid:
    """Convert binary pianoroll `(R, T, P)` to onset/duration events."""

    roll = np.asarray(roll) > 0
    grid = empty_event_grid(bars, roles, subdivisions, pitch_min, pitch_max)
    r, t, p = roll.shape
    if r != len(roles):
        raise ValueError("pianoroll role dimension mismatch")
    max_t = min(t, bars * subdivisions)
    for role in range(r):
        for pitch in range(p):
            active = False
            start = 0
            for time in range(max_t + 1):
                value = bool(time < max_t and roll[role, time, pitch])
                if value and not active:
                    active = True
                    start = time
                elif active and not value:
                    bar, step = divmod(start, subdivisions)
                    grid.data[role, bar, step, pitch, 0] = 1.0
                    grid.data[role, bar, step, pitch, 1] = max(1, time - start)
                    grid.data[role, bar, step, pitch, 2] = velocity
                    active = False
    return grid


def pad_or_crop_bars(grid: EventGrid, bars: int) -> EventGrid:
    """Return an EventGrid with exactly `bars` bars."""

    if grid.bars == bars:
        return grid.copy()
    out = empty_event_grid(bars, grid.roles, grid.subdivisions, grid.pitch_min, grid.pitch_max)
    keep = min(grid.bars, bars)
    out.data[:, :keep] = grid.data[:, :keep]
    out.ticks_per_beat = grid.ticks_per_beat
    out.beats_per_bar = grid.beats_per_bar
    out.tempo = grid.tempo
    out.beat_unit = grid.beat_unit
    return out


def grid_difference(before: EventGrid, after: EventGrid) -> np.ndarray:
    """Bar/role change map used by editing visualization and metrics."""

    if before.shape != after.shape:
        raise ValueError("Event grids must have the same shape for difference")
    diff = np.abs(after.data - before.data)
    return diff.mean(axis=(2, 3, 4))


def save_event_grid(path: str | Path, grid: EventGrid, plan: dict[str, Any] | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data": grid.data, "metadata": np.array([grid.metadata()], dtype=object)}
    if plan is not None:
        payload["plan"] = np.array([plan], dtype=object)
    np.savez_compressed(out, **payload)
    return out


def load_event_grid(path: str | Path) -> EventGrid:
    with np.load(path, allow_pickle=True) as npz:
        metadata = dict(npz["metadata"][0]) if "metadata" in npz else {}
        return EventGrid(data=npz["data"], **metadata)
