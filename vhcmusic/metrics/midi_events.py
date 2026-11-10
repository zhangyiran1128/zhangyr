"""Timing and note-event extraction from finalized MIDI files."""

from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from vhcmusic.representation.track_mapping import TrackMapper


@dataclass(frozen=True)
class NoteEvent:
    track: tuple
    role: str
    bar: int
    pitch: int
    onset: float
    duration: float
    velocity: int


def bar_ticks(midi, bars: int) -> np.ndarray:
    signatures = sorted((ts.time, ts.numerator, ts.denominator) for ts in midi.time_signature_changes)
    starts = [0.0]
    numerator, denominator = 4, 4
    index = 0
    for _ in range(bars):
        start = starts[-1]
        while index < len(signatures) and signatures[index][0] <= start:
            _, numerator, denominator = signatures[index]
            index += 1
        end = start + midi.ticks_per_beat * 4 * numerator / denominator
        if index < len(signatures):
            end = min(end, signatures[index][0])
        starts.append(end)
    return np.asarray(starts)


def bar_times(midi, bars: int) -> np.ndarray:
    changes = {0: 120.0, **{int(t.time): float(t.tempo) for t in midi.tempo_changes}}
    ticks = np.array(sorted(changes))
    seconds_per_tick = 60 / (midi.ticks_per_beat * np.array([changes[t] for t in ticks]))
    elapsed = np.r_[0, np.cumsum(np.diff(ticks) * seconds_per_tick[:-1])]
    edges = bar_ticks(midi, bars)
    index = np.searchsorted(ticks, edges, side="right") - 1
    return elapsed[index] + (edges - ticks[index]) * seconds_per_tick[index]


def note_events(midi, bars: int, roles: tuple[str, ...]) -> list[NoteEvent]:
    edges = bar_ticks(midi, bars)
    mapper = TrackMapper(roles)
    result = []
    counters = defaultdict(int)
    for inst in midi.instruments:
        key = (inst.name, inst.program, inst.is_drum)
        track = (*key, counters[key])
        counters[key] += 1
        role = mapper.infer_role(*key)
        for note in inst.notes:
            bar = int(np.searchsorted(edges, note.start, side="right"))
            result.append(NoteEvent(track, role, bar, note.pitch, note.start / midi.ticks_per_beat,
                                    (note.end - note.start) / midi.ticks_per_beat, note.velocity))
    return result


def event_curves(midi, bars: int, roles: tuple[str, ...], density_scale: float):
    counts = np.zeros((len(roles), bars))
    energy = np.zeros(bars)
    for note in note_events(midi, bars, roles):
        if 1 <= note.bar <= bars and note.role in roles:
            counts[roles.index(note.role), note.bar - 1] += 1
            energy[note.bar - 1] += note.velocity / 127
    beats = np.diff(bar_ticks(midi, bars)) / midi.ticks_per_beat
    return np.clip(counts.sum(0) / beats / density_scale, 0, 1), counts > 0, counts, energy


def event_matches(before, after, *, onset: float, duration: float, relative_duration: float, velocity: int):
    groups_a, groups_b = defaultdict(list), defaultdict(list)
    for notes, groups in ((before, groups_a), (after, groups_b)):
        for index, note in enumerate(notes):
            groups[(note.track, note.pitch)].append((index, note))
    pairs = []
    for key in sorted(groups_a.keys() & groups_b.keys()):
        a, b = groups_a[key], sorted(groups_b[key], key=lambda item: item[1].onset)
        times = np.array([note.onset for _, note in b])
        rows, cols = [], []
        for i, (_, note) in enumerate(a):
            low = int(np.searchsorted(times, note.onset - onset))
            high = int(np.searchsorted(times, note.onset + onset, side="right"))
            for j in range(low, high):
                other = b[j][1]
                if (abs(note.duration - other.duration) <= max(duration, relative_duration * note.duration)
                        and abs(note.velocity - other.velocity) <= velocity):
                    rows.append(i)
                    cols.append(j)
        if rows:
            adjacency = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(a), len(b)))
            assignment = maximum_bipartite_matching(adjacency, perm_type="column")
            pairs.extend((a[i][0], b[j][0]) for i, j in enumerate(assignment) if j >= 0)
    return pairs
