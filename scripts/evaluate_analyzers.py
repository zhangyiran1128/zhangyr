#!/usr/bin/env python
"""Evaluate back-translation against held-out annotations and perturbation labels."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import aggregate, read_jsonl
from vhcmusic.metrics.control_metrics import section_boundary_f1
from vhcmusic.metrics.verification_metrics import false_accept_rate, verification_f1
from vhcmusic.planning.music_planner import normalize_plan
from vhcmusic.representation.harmony import chord_identity
from vhcmusic.representation.midi_io import midi_to_event_grid
from vhcmusic.utils.checkpoint import load_checkpoint
from vhcmusic.verification.chord_analyzer import ChordAnalyzer
from vhcmusic.verification.rhythm_analyzer import RhythmAnalyzer
from vhcmusic.verification.section_analyzer import SectionAnalyzer
from vhcmusic.verification.verification_report import build_verification_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--section-checkpoint", required=True)
    parser.add_argument("--chord-calibration", required=True)
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/reliability_summary.json"))
    args = parser.parse_args()
    section = SectionAnalyzer(args.section_checkpoint)
    chord = ChordAnalyzer.from_calibration(args.chord_calibration)
    fitted = load_checkpoint(args.section_checkpoint)["splits"]
    bill_dev = json.loads(Path(args.chord_calibration).read_text(encoding="utf-8"))["songs"]
    scale = load_checkpoint(args.generator_checkpoint)["statistics"]["density_scale"]
    root = args.manifest.parent
    rows, decisions = [], defaultdict(list)
    suite = "back_translation_reliability"

    def add(row, method, scores):
        rows.append({"suite": suite, "method": method, "seed": int(row.get("seed", 16)),
                     "scores": scores})

    for row in read_jsonl(args.manifest):
        if row["split"] != "test":
            raise ValueError("Analyzer reliability requires held-out labels")
        song = str(row.get("song_id"))
        if (row.get("dataset") == "SALAMI" and song in fitted["train"] + fitted["val"]
                or row.get("dataset") == "McGill Billboard" and song in bill_dev):
            raise ValueError(f"Analyzer evaluation overlaps development data: {song}")
        grid = midi_to_event_grid(root / row["midi"], bars=row.get("bars"))
        ref = row.get("reference", {})
        if "boundaries" in ref:
            predicted = section.analyze(grid)["boundaries"]
            scores = {"boundary_f1": section_boundary_f1(
                [b for b in ref["boundaries"] if b > 1], predicted[1:], tolerance=1)["f1"]}
            if "labels" in ref:
                labels = [s["section"] for s in section.analyze(grid)["sections"]
                          for _ in range(s["end_bar"] - s["start_bar"] + 1)]
                scores["function_accuracy"] = float(np.mean(np.asarray(labels) == ref["labels"]))
            add(row, "Section Analyzer", scores)
        if "chords" in ref:
            labels = chord.analyze(grid)["chords"]
            matches = [chord_identity(p["chord"]) == chord_identity(t)
                       for p, t in zip(labels, ref["chords"], strict=True) if t is not None]
            if matches:
                add(row, "Chord Analyzer", {"chord_acc": float(np.mean(matches))})
        if "density" in ref:
            predicted = RhythmAnalyzer(scale).analyze(grid)["rhythm_density_curve"]
            if len(predicted) != len(ref["density"]):
                raise ValueError("Reference density must cover all bars")
            add(row, "Rhythm Analyzer", {"density_mae": float(np.abs(np.asarray(predicted) - ref["density"]).mean())})
        if "expected_accept" in row:
            plan = normalize_plan(json.loads((root / row["plan"]).read_text(encoding="utf-8-sig")))
            before, mask = None, None
            if "original_midi" in row:
                before = midi_to_event_grid(root / row["original_midi"], bars=grid.bars)
                mask = np.zeros((len(grid.roles), grid.bars), dtype=bool)
                for bar, role in row["editable"]:
                    mask[grid.roles.index(role), bar - 1] = True
            report = build_verification_report(plan, grid, before, mask, section_analyzer=section,
                                               chord_analyzer=chord, density_scale=scale)
            decisions[int(row.get("seed", 16))].append((report["pass"], bool(row["expected_accept"])))
    for seed, pairs in decisions.items():
        predictions, labels = zip(*pairs)
        add({"seed": seed}, "Verification Report", {
            "verification_f1": verification_f1(predictions, labels)["f1"],
            "false_accept_rate": false_accept_rate(predictions, labels),
        })
    if not rows:
        raise ValueError("No independent analyzer targets were present")
    per_seed, summary = aggregate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"per_seed": per_seed, "summary": summary}, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
