"""Shared generator dataset preprocessing and song-level splitting."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vhcmusic.metrics.midi_events import bar_times
from vhcmusic.representation.event_grid import save_event_grid
from vhcmusic.representation.harmony import ROOT_NAMES, chord_identity
from vhcmusic.representation.midi_io import midi_to_event_grid, read_midi


def stable_split(song_id, train=0.8, val=0.1):
    value = int(hashlib.sha256(song_id.encode()).hexdigest(), 16) / 2**256
    return "train" if value < train else "val" if value < train + val else "test"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def chord_intervals(path):
    aliases = {"M": "maj", "m": "min", "M7": "maj7", "m7": "min7", "7": "7",
               "major": "maj", "minor": "min", "hdim": "hdim7"}
    intervals = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        a, b, label = line.split(maxsplit=2)
        label = label.split("/", 1)[0]
        if label in {"N", "X", "N:none", "no_chord"}:
            label = "N:none"
        else:
            root, sep, quality = label.partition(":")
            if not sep:
                root, quality = (label[:-1], "min") if label.endswith("m") else (label, "maj")
            quality = aliases.get(quality, quality)
            if quality not in {"maj", "min", "dim", "aug", "sus2", "sus4", "7", "maj7", "min7", "dim7", "hdim7"}:
                continue
            pitch, quality = chord_identity(f"{root}:{quality}")
            label = f"{ROOT_NAMES[pitch]}:{quality}"
        intervals.append((float(a), float(b), label))
    return intervals


def chords_by_bar(intervals, edges):
    result = []
    for start, end in zip(edges, edges[1:]):
        overlap, label = max(((max(0.0, min(b, end) - max(a, start)), chord)
                              for a, b, chord in intervals), default=(0.0, None))
        result.append(label if overlap else None)
    return result


def process_midi_dataset(args, dataset_name):
    root, output = Path(args.input).resolve(strict=True), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {}
    if args.manifest:
        manifest = Path(args.manifest)
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                path = (manifest.parent / row["midi"]).resolve(strict=True)
                if path in metadata:
                    raise ValueError(f"Duplicate MIDI in {manifest}: {path}")
                metadata[path] = row
        midis = sorted(metadata)
    else:
        midis = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".mid", ".midi"})
        if dataset_name == "slakh2100":
            mixes = [p for p in midis if p.stem.lower() in {"all_src", "mix", "mixture"}]
            if mixes:
                midis = mixes
    if not midis:
        raise FileNotFoundError(f"No MIDI files under {root}")
    items, rejected, splits = [], [], {}
    for path in midis:
        meta = metadata.get(path, {})
        relative = path.relative_to(root)
        song = meta.get("song_id") or (path.parent.name if path.parent != root and
                                      path.parent.name.lower() not in {"midi", "mid", "mids"} else path.stem)
        split_dirs = {"train": "train", "validation": "val", "valid": "val", "val": "val", "test": "test"}
        split = meta.get("split") or next((split_dirs[p.lower()] for p in relative.parts[:-1] if p.lower() in split_dirs), None)
        split = split or stable_split(song, args.train_ratio, args.val_ratio)
        if split not in {"train", "val", "test"} or splits.setdefault(song, split) != split:
            raise ValueError(f"Conflicting splits for {dataset_name}:{song}")
        try:
            midi = read_midi(path)
            if not any(inst.notes for inst in midi.instruments):
                raise ValueError("No note events")
            grid = midi_to_event_grid(path, subdivisions=args.subdivisions, pitch_min=args.pitch_min, pitch_max=args.pitch_max)
            if grid.bars < args.min_bars or not np.any(grid.data[..., 0]):
                raise ValueError("Too short or no in-range notes")
        except (OSError, ValueError, EOFError, IndexError, KeyError) as exc:
            rejected.append({"source": str(path), "reason": str(exc)})
            continue
        annotation = {}
        if meta.get("annotation"):
            annotation = json.loads((manifest.parent / meta["annotation"]).read_text(encoding="utf-8-sig"))
        native = path.parent / "chord_midi.txt"
        if dataset_name == "pop909" and native.is_file():
            existing = {s["bar_id"]: s for s in annotation.get("syntax", [])}
            for i, chord in enumerate(chords_by_bar(chord_intervals(native), bar_times(midi, grid.bars)), 1):
                if chord is not None and "chord" not in existing.get(i, {}):
                    existing.setdefault(i, {"bar_id": i})["chord"] = chord
            annotation["syntax"] = list(existing.values())
        for start in range(0, grid.bars, args.max_bars):
            stop = min(start + args.max_bars, grid.bars)
            if stop - start < args.min_bars or not grid.data[:, start:stop, ..., 0].any():
                continue
            from dataclasses import replace

            part = replace(grid, data=grid.data[:, start:stop].copy())
            for bar, step, pitch in np.argwhere(part.data[..., 0].any(axis=0)):
                part.data[:, bar, step, pitch, 1] = np.minimum(
                    part.data[:, bar, step, pitch, 1], (part.bars - bar) * grid.subdivisions - step)
            section = [dict(s, start_bar=max(s["start_bar"], start + 1) - start,
                            end_bar=min(s["end_bar"], stop) - start)
                       for s in annotation.get("structure", []) if s["start_bar"] <= stop and s["end_bar"] > start]
            grammar = [dict(s, bar_id=s["bar_id"] - start)
                       for s in annotation.get("syntax", []) if start < s["bar_id"] <= stop]
            piece = {"syntax": grammar}
            if section:
                piece["structure"] = section
            key = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:12]
            target = output / split / f"{dataset_name}_{key}_{start + 1:04}.npz"
            save_event_grid(target, part)
            items.append({"dataset": dataset_name, "song_id": song, "split": split, "source": str(path),
                          "path": str(target), "bars": part.bars, "first_bar": start + 1, "annotation": piece})
    if not items:
        raise ValueError(f"No valid excerpts in {root}: {rejected[:3]}")
    write_jsonl(output / "items.jsonl", items)
    write_jsonl(output / "rejected.jsonl", rejected)
    (output / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": dataset_name, "items": len(items), "rejected": len(rejected), "output": str(output)}))


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", help="CSV with midi, song_id, split and optional annotation paths")
    parser.add_argument("--max-bars", type=int, default=64)
    parser.add_argument("--min-bars", type=int, default=4)
    parser.add_argument("--subdivisions", type=int, default=16)
    parser.add_argument("--pitch-min", type=int, default=21)
    parser.add_argument("--pitch-max", type=int, default=108)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser
