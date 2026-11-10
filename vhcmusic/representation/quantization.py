"""Bar-grid quantization helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationSpec:
    ticks_per_beat: int = 480
    beats_per_bar: int = 4
    beat_unit: int = 4
    subdivisions: int = 16

    @property
    def ticks_per_bar(self) -> int:
        return int(self.ticks_per_beat * self.beats_per_bar)

    @property
    def ticks_per_step(self) -> float:
        return self.ticks_per_bar / float(self.subdivisions)


def quantize_tick(tick: int, spec: QuantizationSpec) -> tuple[int, int]:
    """Return zero-based `(bar, step)` for a MIDI tick."""

    total_step = int(round(tick / spec.ticks_per_step))
    bar = total_step // spec.subdivisions
    step = total_step % spec.subdivisions
    return bar, step


def step_to_tick(bar: int, step: int, spec: QuantizationSpec) -> int:
    """Convert zero-based bar/step coordinates back to ticks."""

    return int(round((bar * spec.subdivisions + step) * spec.ticks_per_step))


def duration_to_steps(duration_ticks: int, spec: QuantizationSpec) -> int:
    """Quantize a note duration to grid steps."""

    return max(1, int(round(duration_ticks / spec.ticks_per_step)))
