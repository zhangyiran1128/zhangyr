#!/usr/bin/env python
"""Run GPT-4o-guided local repair using online analyzer observations."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_music import generate_grid, restore_model
from scripts.local_edit import grid_from_midi
from vhcmusic.editing.masked_sampling import edit_grid
from vhcmusic.planning.music_planner import normalize_plan
from vhcmusic.planning.schema import plan_bars, validate_plan
from vhcmusic.representation.midi_io import write_event_grid_midi
from vhcmusic.utils.config import ensure_dir, load_config
from vhcmusic.utils.seed import seed_everything
from vhcmusic.verification.closed_loop import run_closed_loop
from vhcmusic.verification.section_analyzer import SECTION_CHECKPOINT, SectionAnalyzer
from vhcmusic.verification.chord_analyzer import ChordAnalyzer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/edit_vhcmusic.yaml")
    parser.add_argument("--train-config", default="configs/train_vhcmusic.yaml")
    parser.add_argument("--midi", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--section-checkpoint", default=SECTION_CHECKPOINT)
    parser.add_argument("--chord-calibration")
    parser.add_argument("--output", default="outputs/closed_loop")
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--strategy", choices=["local", "random", "global", "none"], default="local")
    args = parser.parse_args()
    settings = load_config(args.config)
    cfg = load_config(args.train_config)
    seed_everything(args.seed)
    out = ensure_dir(args.output)
    plan = normalize_plan(json.loads(Path(args.plan).read_text(encoding="utf-8-sig")))
    bars = plan_bars(plan)
    validate_plan(plan, bars=bars, roles=tuple(cfg["data"]["roles"]))
    if bars > cfg["data"]["max_bars"]:
        raise ValueError("Repair requires an excerpt of at most 64 bars")
    original = grid_from_midi(args.midi, cfg, bars=bars)
    device = torch.device("cuda" if cfg["device"]["prefer_cuda"] and torch.cuda.is_available() else "cpu")
    model = restore_model(cfg, args.checkpoint or settings["input"]["checkpoint"], device)
    edit_cfg = settings["editing"]
    previous = Path(args.midi)

    def repair(grid, round_plan, mask, strength, action, index):
        nonlocal previous
        path = out / f"round_{index:02d}.mid"
        if action in {"global_resample", "random_resample"}:
            sample_cfg = cfg
            if action == "random_resample":
                sample_cfg = {**cfg, "diffusion": {**cfg["diffusion"], "classifier_free_guidance": 0.0}}
            result = generate_grid(model, round_plan, sample_cfg)
            write_event_grid_midi(result, path)
        else:
            result = edit_grid(model, grid, round_plan, mask, strength,
                               cfg["diffusion"]["classifier_free_guidance"], edit_cfg["ddim_steps"])
            write_event_grid_midi(result, path)
        np.save(out / f"round_{index:02d}_mask.npy", mask.mask)
        previous = path
        return grid_from_midi(path, cfg, bars=bars)

    chord = ChordAnalyzer.from_calibration(args.chord_calibration)
    _, reports = run_closed_loop(original, plan, repair, section_analyzer=SectionAnalyzer(args.section_checkpoint),
                                 chord_analyzer=chord, density_scale=model.statistics["density_scale"],
                                 max_rounds=edit_cfg["max_rounds"], strength=edit_cfg["edit_strength"],
                                 strategy=args.strategy)
    (out / "repair_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    final = out / "repaired.mid"
    if previous.resolve() != final.resolve():
        shutil.copyfile(previous, final)
    outcome = {"accepted": reports[-1]["pass"] if reports else None, "rounds": max(0, len(reports) - 1)}
    (out / "repair_outcome.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
