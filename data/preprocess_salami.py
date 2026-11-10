#!/usr/bin/env python
"""Align SALAMI functional annotations to independently aligned symbolic recordings."""

import argparse
import csv
import json
import re
from pathlib import Path

from preprocess_common import stable_split
from vhcmusic.metrics.midi_events import bar_times
from vhcmusic.representation.midi_io import midi_to_event_grid, read_midi


def function_label(label):
    value = re.sub(r"[-_ ]+", " ", label.lower().strip())
    for pattern, name in [
        (r"pre chorus", "pre_chorus"), (r"chorus|refrain", "chorus"), (r"verse", "verse"),
        (r"intro", "intro"), (r"outro|coda|fade out", "outro"), (r"bridge|middle 8", "bridge"),
        (r"breakdown", "breakdown"), (r"instrumental|solo", "instrumental"),
        (r"interlude|transition", "interlude"),
    ]:
        if re.search(pattern, value):
            return name
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, splits = [], {}
    with args.alignment.open(encoding="utf-8-sig", newline="") as stream:
        for item in csv.DictReader(stream):
            song = item["song_id"]
            split = item.get("split") or stable_split(song, 968 / 1383, 207 / 1383)
            if split not in {"train", "val", "test"} or splits.setdefault(song, split) != split:
                raise ValueError(f"Inconsistent split for SALAMI recording {song}")
            midi_path = (args.alignment.parent / item["midi"]).resolve(strict=True)
            annotations = sorted((args.input / "annotations" / song / "parsed").glob("textfile*_functions.txt"))
            if item.get("annotation"):
                annotations = [(args.input / item["annotation"]).resolve(strict=True)]
            if not annotations:
                raise FileNotFoundError(f"Missing SALAMI functional annotations for {song}")
            grid = midi_to_event_grid(midi_path)
            times = bar_times(read_midi(midi_path), grid.bars)[:-1]
            offset = float(item.get("offset_seconds") or 0)
            for annotation in annotations:
                events = {}
                for line in annotation.read_text(encoding="utf-8-sig").splitlines():
                    if not line.strip() or line.startswith("#"):
                        continue
                    seconds, label = line.split(maxsplit=1)
                    if label.strip().lower() in {"end", "end of piece", "silence"}:
                        continue
                    time = float(seconds) - offset
                    if time > times[-1] + 0.5:
                        continue
                    bar = int(abs(times - time).argmin()) + 1
                    events[bar] = function_label(label)
                if not events:
                    raise ValueError(f"No aligned section labels in {annotation}")
                starts = sorted(set(events) | {1})
                labels = [events.get(1, "other")] * grid.bars
                for a, b in zip(starts, starts[1:] + [grid.bars + 1]):
                    labels[a - 1:b - 1] = [events.get(a, "other")] * (b - a)
                rows.append({"dataset": "SALAMI", "song_id": song, "split": split,
                             "midi": str(midi_path), "annotation": str(annotation.resolve()),
                             "boundaries": starts, "labels": labels})
    if not rows:
        raise ValueError("The SALAMI alignment manifest contains no recordings")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sections.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
