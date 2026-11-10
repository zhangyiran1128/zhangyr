"""Pianoroll convenience wrappers."""

from __future__ import annotations

import numpy as np

from .event_grid import EventGrid, event_grid_to_pianoroll, pianoroll_to_event_grid


def to_pianoroll(grid: EventGrid) -> np.ndarray:
    return event_grid_to_pianoroll(grid)


def from_pianoroll(roll: np.ndarray, bars: int, **kwargs) -> EventGrid:
    return pianoroll_to_event_grid(roll, bars=bars, **kwargs)
