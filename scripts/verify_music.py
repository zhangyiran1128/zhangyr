#!/usr/bin/env python
"""Run Table A1 verification on a fixed MIDI and target plan."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.local_edit import grid_from_midi
from vhcmusic.planning.music_planner import normalize_plan
from vhcmusic.planning.schema import plan_bars
from vhcmusic.utils.checkpoint import load_checkpoint
from vhcmusic.utils.config import ensure_dir, load_config
from vhcmusic.verification.chord_analyzer import ChordAnalyzer
from vhcmusic.verification.section_analyzer import SECTION_CHECKPOINT, SectionAnalyzer
from vhcmusic.verification.verification_report import build_verification_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_vhcmusic.yaml")
    parser.add_argument("--midi", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--section-checkpoint", default=SECTION_CHECKPOINT)
    parser.add_argument("--chord-calibration")
    parser.add_argument("--generator-checkpoint", required=True)
    parser.add_argument("--original")
    parser.add_argument("--mask")
    parser.add_argument("--output", default="outputs/verify")
    args = parser.parse_args()
    if bool(args.original) != bool(args.mask):
        parser.error("--original and --mask must be provided together")
    cfg = load_config(args.config)
    plan = normalize_plan(json.loads(Path(args.plan).read_text(encoding="utf-8-sig")))
    bars = plan_bars(plan)
    grid = grid_from_midi(args.midi, cfg, bars=bars)
    original = grid_from_midi(args.original, cfg, bars=bars) if args.original else None
    density = load_checkpoint(args.generator_checkpoint)["statistics"]["density_scale"]
    report = build_verification_report(plan, grid, original, np.load(args.mask) if args.mask else None,
                                       section_analyzer=SectionAnalyzer(args.section_checkpoint),
                                       chord_analyzer=ChordAnalyzer.from_calibration(args.chord_calibration),
                                       density_scale=density)
    out = ensure_dir(args.output)
    (out / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "mismatch_bars": report["mismatch_bars"]}))


if __name__ == "__main__":
    main()
