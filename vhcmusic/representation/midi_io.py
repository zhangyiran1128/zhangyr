"""MIDI reading and writing for VHCMusic event grids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .event_grid import EventGrid, empty_event_grid
from .quantization import QuantizationSpec, duration_to_steps, quantize_tick, step_to_tick
from .track_mapping import TRACK_ROLES, TrackMapper


@dataclass
class MidiMetadata:
    ticks_per_beat: int = 480
    tempo: float = 120.0
    beats_per_bar: int = 4
    beat_unit: int = 4


def _require_miditoolkit():
    try:
        import miditoolkit

        return miditoolkit
    except ImportError as exc:
        raise ImportError("MIDI IO requires miditoolkit. Install with `pip install miditoolkit`.") from exc


def read_midi(path: str | Path):
    """Load a MIDI file with miditoolkit."""

    miditoolkit = _require_miditoolkit()
    midi_path = Path(path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    return miditoolkit.MidiFile(str(midi_path))


def midi_metadata(midi) -> MidiMetadata:
    tempo = 120.0
    if getattr(midi, "tempo_changes", None):
        tempo = float(midi.tempo_changes[0].tempo)
    beats_per_bar, beat_unit = 4, 4
    if getattr(midi, "time_signature_changes", None):
        ts = midi.time_signature_changes[0]
        beats_per_bar, beat_unit = int(ts.numerator), int(ts.denominator)
    return MidiMetadata(
        ticks_per_beat=int(midi.ticks_per_beat),
        tempo=tempo,
        beats_per_bar=beats_per_bar,
        beat_unit=beat_unit,
    )


def midi_to_event_grid(
    midi_path: str | Path,
    bars: int | None = None,
    subdivisions: int = 16,
    pitch_min: int = 21,
    pitch_max: int = 108,
    roles: tuple[str, ...] = TRACK_ROLES,
) -> EventGrid:
    """Quantize a MIDI file to `R x L x S x P x F` event grid."""

    midi = read_midi(midi_path)
    meta = midi_metadata(midi)
    if len({(ts.numerator, ts.denominator) for ts in midi.time_signature_changes}) > 1:
        raise ValueError('The event grid requires a fixed time signature')
    spec = QuantizationSpec(
        ticks_per_beat=meta.ticks_per_beat,
        beats_per_bar=meta.beats_per_bar,
        beat_unit=meta.beat_unit,
        subdivisions=subdivisions,
    )
    max_tick = max((note.end for inst in midi.instruments for note in inst.notes), default=0)
    end_marker = next((m.time for m in midi.markers if m.text == 'VHCMusic:end'), None)
    inferred_bars = max(1, int(np.ceil((end_marker or max_tick) / spec.ticks_per_bar)))
    bars = int(bars or inferred_bars)
    grid = empty_event_grid(bars, roles, subdivisions, pitch_min, pitch_max)
    grid.ticks_per_beat = meta.ticks_per_beat
    grid.beats_per_bar = meta.beats_per_bar
    grid.tempo = meta.tempo
    grid.beat_unit = meta.beat_unit

    mapper = TrackMapper(roles)
    for inst in midi.instruments:
        role = mapper.infer_role(getattr(inst, "name", ""), int(inst.program), bool(inst.is_drum))
        if role not in roles:
            continue
        role_idx = roles.index(role)
        for note in inst.notes:
            if note.pitch < pitch_min or note.pitch > pitch_max:
                continue
            bar, step = quantize_tick(int(note.start), spec)
            if bar < 0 or bar >= bars:
                continue
            pitch_idx = int(note.pitch - pitch_min)
            duration = duration_to_steps(int(note.end - note.start), spec)
            velocity = float(np.clip(note.velocity / 127.0, 0.0, 1.0))
            current_velocity = grid.data[role_idx, bar, step, pitch_idx, 2]
            if grid.data[role_idx, bar, step, pitch_idx, 0] == 0 or velocity >= current_velocity:
                grid.data[role_idx, bar, step, pitch_idx, 0] = 1.0
                grid.data[role_idx, bar, step, pitch_idx, 1] = duration
                grid.data[role_idx, bar, step, pitch_idx, 2] = velocity
    return grid


def event_grid_to_midi(grid: EventGrid):
    """Decode an EventGrid to a miditoolkit MidiFile."""

    miditoolkit = _require_miditoolkit()
    midi = miditoolkit.MidiFile(ticks_per_beat=int(grid.ticks_per_beat))
    midi.tempo_changes = [miditoolkit.TempoChange(float(grid.tempo), 0)]
    midi.time_signature_changes = [miditoolkit.TimeSignature(int(grid.beats_per_bar), int(grid.beat_unit), 0)]
    spec = QuantizationSpec(
        ticks_per_beat=int(grid.ticks_per_beat),
        beats_per_bar=int(grid.beats_per_bar),
        beat_unit=int(grid.beat_unit),
        subdivisions=int(grid.subdivisions),
    )
    mapper = TrackMapper(grid.roles)
    end_tick = int(round(grid.bars * spec.ticks_per_bar))
    for role_idx, role in enumerate(grid.roles):
        inst = miditoolkit.Instrument(
            program=mapper.gm_program_for_role(role),
            is_drum=(role == "drums"),
            name=role,
        )
        onsets = grid.data[role_idx, ..., 0] > 0.5
        next_onset = {}
        positions = np.argwhere(onsets)
        for bar, step, pitch_idx in reversed(positions):
            start = step_to_tick(int(bar), int(step), spec)
            limit = next_onset.get(pitch_idx, end_tick)
            if limit <= start:
                continue
            duration, velocity = grid.data[role_idx, bar, step, pitch_idx, 1:]
            if not np.isfinite(duration) or not np.isfinite(velocity):
                raise ValueError("Decoded MIDI event has a non-finite duration or velocity")
            dur_steps = int(np.clip(np.rint(duration), 1, grid.bars * grid.subdivisions))
            end = min(limit, step_to_tick(int(bar), int(step) + dur_steps, spec))
            pitch = int(grid.pitch_min + pitch_idx)
            inst.notes.append(miditoolkit.Note(velocity=int(np.clip(np.rint(velocity * 127), 1, 127)),
                                               pitch=pitch, start=start, end=max(start + 1, end)))
            next_onset[pitch_idx] = start
        inst.notes.reverse()
        midi.instruments.append(inst)
    midi.markers = [miditoolkit.Marker('VHCMusic:end', end_tick)]
    return midi


def save_midi(midi, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(out))
    return out


def write_event_grid_midi(grid: EventGrid, path: str | Path) -> Path:
    """Decode and save an EventGrid as MIDI."""

    return save_midi(event_grid_to_midi(grid), path)


def batch_midi_to_grids(paths: Iterable[str | Path], **kwargs) -> list[EventGrid]:
    return [midi_to_event_grid(path, **kwargs) for path in paths]
