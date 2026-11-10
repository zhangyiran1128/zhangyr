#!/usr/bin/env python
"""Select the fixed Viterbi penalty on aligned McGill Billboard development recordings."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import read_jsonl
from vhcmusic.representation.harmony import chord_identity
from vhcmusic.representation.midi_io import midi_to_event_grid
from vhcmusic.verification.chord_analyzer import ChordAnalyzer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoints/chord_calibration.json"))
    args = parser.parse_args()
    tracks, partitions = [], {}
    for row in read_jsonl(args.manifest):
        song, split = row["song_id"], row["split"]
        if row["dataset"] != "McGill Billboard" or partitions.setdefault(song, split) != split:
            raise ValueError("Calibration requires disjoint Billboard songs")
        if split == "val":
            grid = midi_to_event_grid(args.manifest.parent / row["midi"])
            labels = [chord_identity(c) if c is not None else None for c in row["chords"]]
            if grid.bars != len(labels):
                raise ValueError(f"Chord labels do not cover MIDI bars for {song}")
            tracks.append((grid, labels))
    if not tracks:
        raise ValueError("No Billboard development recordings available")
    results = []
    for scale in (0, 0.25, 0.5, 1, 1.5, 2, 3, 4):
        analyzer = ChordAnalyzer(transition_scale=scale)
        scores = []
        for grid, target in tracks:
            observed = [chord_identity(c["chord"]) for c in analyzer.analyze(grid)["chords"]]
            scores.append(np.mean([a == b for a, b in zip(observed, target) if b is not None]))
        results.append((float(np.mean(scores)), scale))
    score, scale = max(results, key=lambda r: (r[0], -abs(1 - r[1])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": "McGill Billboard", "split": "val", "transition_scale": scale,
               "development_accuracy": score, "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "songs": sorted(song for song, split in partitions.items() if split == "val")}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
