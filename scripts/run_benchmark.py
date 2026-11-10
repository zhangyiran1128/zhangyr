#!/usr/bin/env python
"""Generate fixed benchmark outputs across the three manuscript seeds."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import read_jsonl
from scripts.generate_music import generate_grid, restore_model
from scripts.local_edit import build_edit_plan, grid_from_midi
from vhcmusic.editing.edit_mask import mask_from_request
from vhcmusic.editing.masked_sampling import edit_grid
from vhcmusic.planning.music_planner import MusicPlanner, normalize_plan
from vhcmusic.planning.schema import plan_bars, validate_plan
from vhcmusic.representation.midi_io import write_event_grid_midi
from vhcmusic.utils.config import ensure_dir, load_config
from vhcmusic.utils.seed import seed_everything
from vhcmusic.verification.chord_analyzer import ChordAnalyzer
from vhcmusic.verification.closed_loop import run_closed_loop
from vhcmusic.verification.section_analyzer import SectionAnalyzer


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def fixed_plan(task, root, plan_dir, roles):
    if "plan" in task:
        plan = normalize_plan(json.loads(read_json(root / task["plan"])))
    else:
        prompt_path = root / task["prompt_file"] if "prompt_file" in task else None
        payload = json.loads(read_json(prompt_path)) if prompt_path else {
            key: task[key] for key in ("prompt", "video_description", "duration_seconds", "tempo", "key",
                                       "keyframes", "motion_curve") if key in task}
        if not payload:
            raise ValueError(f"Missing fixed plan or prompt for {task['id']}")
        if "keyframes" in payload:
            base = prompt_path.parent if prompt_path else root
            payload["keyframes"] = [dict(frame, path=str(base / frame["path"])) if isinstance(frame, dict)
                                    else str(base / frame) for frame in payload["keyframes"]]
        key = hashlib.sha256(json.dumps([task["id"], task["bars"], payload], sort_keys=True).encode()).hexdigest()
        cached = plan_dir / f"{key}.json"
        if cached.exists():
            plan = json.loads(read_json(cached))
        else:
            plan = MusicPlanner(roles=roles).plan(payload, bars=task["bars"])
            cached.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if validate_plan(plan, bars=task["bars"], roles=roles).bars != task["bars"]:
        raise ValueError(f"Plan length differs from task {task['id']}")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--train-config", default="configs/train_vhcmusic.yaml")
    parser.add_argument("--edit-config", default="configs/edit_vhcmusic.yaml")
    parser.add_argument("--output", type=Path, default=Path("outputs/generated"))
    parser.add_argument("--method", default="VHCMusic")
    parser.add_argument("--suite", action="append", help="Restrict execution to these suite names")
    parser.add_argument("--strategy", choices=("initial", "vanilla", "edit", "local", "random", "global"), default="initial")
    parser.add_argument("--section-checkpoint")
    parser.add_argument("--chord-calibration")
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    cfg = load_config(args.train_config)
    settings = load_config(args.edit_config)["editing"]
    device = torch.device("cuda" if cfg["device"]["prefer_cuda"] and torch.cuda.is_available() else "cpu")
    roles = tuple(cfg["data"]["roles"])
    root = args.tasks.parent
    output = ensure_dir(args.output)
    plan_dir = ensure_dir(output / "fixed_plans")
    records = [row for row in read_jsonl(args.tasks) if not args.suite or row["suite"] in args.suite]
    if not records or any(row["split"] != "test" for row in records):
        raise ValueError("Benchmark requires fixed, held-out tasks")
    if any(not 1 <= int(row["bars"]) <= cfg["data"]["max_bars"] for row in records):
        raise ValueError("Task bars exceed the generator sequence length")
    plans = {row["id"]: fixed_plan(row, root, plan_dir, roles) for row in records} if args.strategy != "vanilla" else {}
    closed = args.strategy in {"local", "random", "global"}
    section = chord = None
    if closed:
        if not args.section_checkpoint or not args.chord_calibration:
            parser.error("Closed-loop strategies require both analyzer checkpoints")
        section = SectionAnalyzer(args.section_checkpoint)
        chord = ChordAnalyzer.from_calibration(args.chord_calibration)
    results, timing, round_outputs = [], [], []
    for seed in args.seeds or cfg["project"]["seeds"]:
        model = restore_model(cfg, Path(cfg["diffusion"]["checkpoint_dir"]) / f"seed_{seed}" / "best.pt", device)
        n_parameters = sum(p.numel() for p in model.parameters())
        guidance = cfg["diffusion"]["classifier_free_guidance"]
        for row in records:
            task_id = row["id"]
            seed_everything(seed)
            identity = hashlib.sha256(f"{args.method}:{seed}:{task_id}".encode()).hexdigest()[:20]
            folder = ensure_dir(output / "samples" / identity)
            is_edit = "edit_request" in row
            plan = plans.get(task_id)
            if is_edit:
                original = grid_from_midi(root / row["original_midi"], cfg, bars=row["bars"])
                request = json.loads(read_json(root / row["edit_request"]))
                mask = mask_from_request(original.bars, request.get("target_bars"), request.get("target_track_role"), roles)
                selected = {(bar + 1, role) for r, role in enumerate(roles)
                            for bar in range(original.bars) if mask.mask[r, bar]}
                if selected != {tuple(pair) for pair in row["editable"]}:
                    raise ValueError(f"Edit request and fixed benchmark mask differ for {task_id}")
                plan = build_edit_plan(request, original, model.statistics, mask, plan)
            elif args.strategy in {"vanilla", "edit"}:
                raise ValueError("The selected editing strategy requires edit_request tasks")
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            grid = (edit_grid(model, original, plan, mask, settings["edit_strength"], guidance, settings["ddim_steps"])
                    if is_edit else generate_grid(model, plan, cfg))
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
            rounds = 0
            if closed:
                initial_midi = write_event_grid_midi(grid, folder / "round_00.mid")
                grid = grid_from_midi(initial_midi, cfg, bars=row["bars"])

                def repair(current, new_plan, new_mask, strength, action, index):
                    if action in {"global_resample", "random_resample"}:
                        sample = cfg if action == "global_resample" else {
                            **cfg, "diffusion": {**cfg["diffusion"], "classifier_free_guidance": 0.0}}
                        candidate = generate_grid(model, new_plan, sample)
                    else:
                        candidate = edit_grid(model, current, new_plan, new_mask, strength, guidance, settings["ddim_steps"])
                    candidate_midi = write_event_grid_midi(candidate, folder / f"round_{index:02d}.mid")
                    return grid_from_midi(candidate_midi, cfg, bars=row["bars"])

                grid, reports = run_closed_loop(grid, plan, repair, section_analyzer=section, chord_analyzer=chord,
                                                density_scale=model.statistics["density_scale"],
                                                max_rounds=settings["max_rounds"], strength=settings["edit_strength"],
                                                strategy=args.strategy, preservation_reference=original if is_edit else None)
                rounds = len(reports) - 1
                round_outputs.extend({"task_id": task_id, "method": args.method, "seed": seed, "round": i,
                                      "midi": (folder / f"round_{i:02d}.mid").relative_to(output).as_posix()}
                                     for i in range(rounds + 1))
                (folder / "repair_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
            midi = write_event_grid_midi(grid, folder / "final.mid")
            results.append({"task_id": task_id, "method": args.method, "seed": seed,
                            "midi": midi.relative_to(output).as_posix(), "rounds": rounds})
            duration = grid.bars * grid.beats_per_bar * 4 / grid.beat_unit * 60 / grid.tempo
            measure = {"parameters": n_parameters, "inference_seconds": latency, "throughput_xrt": duration / latency}
            if peak is not None:
                measure["peak_memory_mib"] = peak / 2**20
            timing.extend({"suite": "efficiency_analysis", "method": args.method, "seed": seed,
                           "sample_id": task_id, "metric": name, "value": value} for name, value in measure.items())
    for name, rows, keys in (
        ("final_outputs.jsonl", results, ("method", "seed", "task_id")),
        ("efficiency.jsonl", timing, ("method", "seed", "sample_id", "metric")),
        ("round_outputs.jsonl", round_outputs, ("method", "seed", "task_id", "round")),
    ):
        if not rows:
            continue
        path = output / name
        if path.exists():
            merged = {tuple(row[k] for k in keys): row for row in read_jsonl(path)}
            if name == "round_outputs.jsonl":
                rerun = {(row["method"], row["seed"], row["task_id"]) for row in rows}
                merged = {key: value for key, value in merged.items() if key[:3] not in rerun}
            merged.update({tuple(row[k] for k in keys): row for row in rows})
            rows = merged.values()
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(path)
    print(output / "final_outputs.jsonl")


if __name__ == "__main__":
    main()
