#!/usr/bin/env python
"""Freeze curated benchmark tasks against the held-out generator song manifests."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generator_data import generator_items
from data.preprocess_common import write_jsonl
from scripts.evaluate import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--curated", type=Path, required=True, help="Fixed independently annotated JSONL tasks")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    songs = {(item["dataset"], item["song_id"]): item["split"] for item in generator_items(args.processed)}
    rows, seen = [], set()
    for row in read_jsonl(args.curated):
        identity = (row["dataset"], row["song_id"])
        open_ended = row["dataset"] == "unp90"
        if open_ended and (not row.get("author_id") or not row.get("prompt")):
            raise ValueError("UNP-90 records require the original prompt and its author ID")
        if (not open_ended and songs.get(identity) != "test") or row["id"] in seen:
            raise ValueError(f"Task {row['id']} does not identify a unique held-out sample")
        if not open_ended and not row.get("reference") and not row.get("targets") and not row.get("reference_audio"):
            raise ValueError(f"Task {row['id']} has no independently annotated targets or reference audio")
        seen.add(row["id"])
        for key in ("original_midi", "reference_audio", "plan", "edit_request", "prompt_file"):
            if key in row:
                row[key] = str((args.curated.parent / row[key]).resolve(strict=True))
        if "keyframes" in row:
            base = args.curated.parent
            row["keyframes"] = [dict(frame, path=str((base / frame["path"]).resolve(strict=True))) if isinstance(frame, dict)
                                else str((base / frame).resolve(strict=True)) for frame in row["keyframes"]]
        rows.append(dict(row, split="test"))
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "test_tasks.jsonl"
    write_jsonl(output, rows)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    (args.output / "test_tasks.sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"tasks": len(rows), "sha256": digest}))


if __name__ == "__main__":
    main()
