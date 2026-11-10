#!/usr/bin/env python
"""Apply a user-selected role/bar mask to a trained VHCMusic generator."""

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generator_data import derive_plan
from scripts.generate_music import restore_model
from vhcmusic.editing.edit_mask import mask_from_request
from vhcmusic.editing.masked_sampling import edit_grid
from vhcmusic.planning.music_planner import normalize_plan
from vhcmusic.planning.schema import validate_plan
from vhcmusic.representation.event_grid import grid_difference
from vhcmusic.representation.midi_io import midi_to_event_grid, write_event_grid_midi
from vhcmusic.utils.config import ensure_dir, load_config
from vhcmusic.utils.seed import seed_everything
from vhcmusic.verification.chord_analyzer import ChordAnalyzer
from vhcmusic.verification.section_analyzer import SectionAnalyzer
from vhcmusic.verification.verification_report import build_verification_report


def build_edit_plan(request, original, statistics, mask, base_plan=None):
    plan = copy.deepcopy(base_plan) if base_plan is not None else derive_plan(original, statistics)
    plan["global_constraints"].update(request.get("global_constraints", {}))
    update = {(c["bar_id"], c["track_role"]): c for c in request.get("constraints", [])}
    existing = {(c["bar_id"], c["track_role"]): c for c in plan["constraints"]}
    plan["constraints"] = []
    for r, role in enumerate(original.roles):
        for l in range(original.bars):
            key = (l + 1, role)
            con = {"bar_id": l + 1, "track_role": role, "target_density": None, "target_activity": None,
                   "target_register": None, "boundary_smoothness": 0.5, **existing.get(key, {})}
            if mask.mask[r, l]:
                for attr in ("target_density", "target_activity", "target_register", "boundary_smoothness"):
                    if attr in request:
                        con[attr] = request[attr]
                    if attr in update.get(key, {}):
                        con[attr] = update[key][attr]
            con["editable"] = bool(mask.mask[r, l])
            plan["constraints"].append(con)
    by_bar = {}
    for con in plan["constraints"]:
        if con["editable"]:
            by_bar.setdefault(con["bar_id"], []).append(con)
    quarter_notes = original.beats_per_bar * 4 / original.beat_unit
    onset_count = (original.data[..., 0] > 0.5).sum(axis=(2, 3))
    for syn in plan["syntax"]:
        bar = syn["bar_id"] - 1
        selected = by_bar.get(syn["bar_id"], ())
        count = float(onset_count[:, bar].sum())
        density_changed = False
        for con in selected:
            role = con["track_role"]
            idx = original.roles.index(role)
            if con["target_density"] is not None and ("target_density" in request
                    or "target_density" in update.get((syn["bar_id"], role), {})):
                count += con["target_density"] * statistics["role_density_scale"][idx] - onset_count[idx, bar]
                density_changed = True
            if con["target_activity"] is not None:
                active = set(syn["active_roles"])
                if con["target_activity"] >= 0.5:
                    active.add(role)
                else:
                    active.discard(role)
                syn["active_roles"] = [role for role in original.roles if role in active]
        if density_changed:
            syn["density"] = float(np.clip(count / quarter_notes / statistics["density_scale"], 0, 1))
        if selected and "chord" in request:
            syn["chord"] = request["chord"]
    if validate_plan(plan, bars=original.bars, roles=original.roles).bars != original.bars:
        raise ValueError("The edit plan must cover the entire input excerpt")
    return plan


def grid_from_midi(path, cfg, bars=None):
    data = cfg["data"]
    return midi_to_event_grid(path, bars=bars, subdivisions=data["subdivisions"],
                              pitch_min=data["pitch_min"], pitch_max=data["pitch_max"], roles=tuple(data["roles"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/edit_vhcmusic.yaml")
    parser.add_argument("--train-config", default="configs/train_vhcmusic.yaml")
    parser.add_argument("--midi", required=True)
    parser.add_argument("--edit-request", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--checkpoint")
    parser.add_argument("--section-checkpoint")
    parser.add_argument("--chord-calibration")
    parser.add_argument("--output", default="outputs/edit")
    parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args()
    edit_cfg = load_config(args.config)
    train_cfg = load_config(args.train_config)
    settings = edit_cfg["editing"]
    seed_everything(args.seed)
    out = ensure_dir(args.output)
    original = grid_from_midi(args.midi, train_cfg)
    if original.bars > train_cfg["data"]["max_bars"]:
        raise ValueError("Edit a MIDI excerpt of at most 64 bars")
    request = json.loads(Path(args.edit_request).read_text(encoding="utf-8-sig"))
    mask = mask_from_request(original.bars, request.get("target_bars"), request.get("target_track_role"), original.roles)
    if not mask.mask.any():
        raise ValueError("The request has no editable track-bar positions")
    device = torch.device("cuda" if train_cfg["device"]["prefer_cuda"] and torch.cuda.is_available() else "cpu")
    model = restore_model(train_cfg, args.checkpoint or edit_cfg["input"]["checkpoint"], device)
    base = normalize_plan(json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))) if args.plan else None
    plan = build_edit_plan(request, original, model.statistics, mask, base)
    edited = edit_grid(model, original, plan, mask, settings["edit_strength"],
                       train_cfg["diffusion"]["classifier_free_guidance"], settings["ddim_steps"])
    (out / "edit_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    np.save(out / "edit_mask.npy", mask.mask)
    np.save(out / "diff_map.npy", grid_difference(original, edited))
    write_event_grid_midi(original, out / "before.mid")
    path = out / "edited.mid"
    if int(settings["edit_strength"] * model.scheduler.steps) == 0:
        shutil.copyfile(out / "before.mid", path)
    else:
        write_event_grid_midi(edited, path)
    if args.section_checkpoint:
        finalized = grid_from_midi(path, train_cfg, bars=original.bars)
        report = build_verification_report(plan, finalized, original, mask.mask,
                                           section_analyzer=SectionAnalyzer(args.section_checkpoint),
                                           chord_analyzer=ChordAnalyzer.from_calibration(args.chord_calibration),
                                           density_scale=model.statistics["density_scale"])
        (out / "verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
