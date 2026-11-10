#!/usr/bin/env python
"""Align McGill Billboard chord annotations with independent symbolic recordings."""

import argparse
import csv
from pathlib import Path

from preprocess_common import chords_by_bar, chord_intervals, stable_split, write_jsonl
from vhcmusic.metrics.midi_events import bar_times
from vhcmusic.representation.midi_io import midi_to_event_grid, read_midi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, assigned = [], {}
    with args.alignment.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            song = row["song_id"]
            split = row.get("split") or stable_split(song, 518 / 740, 111 / 740)
            if split not in {"train", "val", "test"} or assigned.setdefault(song, split) != split:
                raise ValueError(f"Inconsistent Billboard split for {song}")
            midi = (args.alignment.parent / row["midi"]).resolve(strict=True)
            annotations = (args.input / row["annotation"]).resolve(strict=True)
            grid = midi_to_event_grid(midi)
            times = bar_times(read_midi(midi), grid.bars) + float(row.get("offset_seconds") or 0)
            chords = chords_by_bar(chord_intervals(annotations), times)
            if not any(chord is not None for chord in chords):
                raise ValueError(f"No aligned chord annotations for {song}")
            rows.append({"dataset": "McGill Billboard", "song_id": song, "split": split,
                         "midi": str(midi), "annotation": str(annotations), "chords": chords})
    if not rows:
        raise ValueError("The Billboard alignment manifest contains no recordings")
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "chords.jsonl", rows)


if __name__ == "__main__":
    main()
