#!/usr/bin/env python
"""Score finalized benchmark outputs against fixed reference annotations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vhcmusic.metrics.generation_metrics import audio_feature_kld, clap_similarity, fad_score, imagebind_similarity
from vhcmusic.metrics.listening import listening_scores
from vhcmusic.metrics.midi_events import bar_times
from vhcmusic.metrics.offline import score_audio_output, score_output, temporal_alignment
from vhcmusic.rendering.fluidsynth_render import render_midi_to_wav, render_profile
from vhcmusic.representation.midi_io import read_midi
from vhcmusic.utils.config import ensure_dir, load_config


def read_jsonl(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["method"], row["seed"])].append(row["scores"])
    per_seed = []
    for (suite, method, seed), values in sorted(groups.items()):
        keys = sorted(set().union(*(item.keys() for item in values)))
        per_seed.append({"suite": suite, "method": method, "seed": seed, "n": len(values),
                         **{key: float(np.mean([item[key] for item in values if key in item])) for key in keys}})
    grouped = defaultdict(list)
    for row in per_seed:
        grouped[(row["suite"], row["method"])].append(row)
    result = []
    for (suite, method), values in sorted(grouped.items()):
        metrics = sorted(set().union(*(row.keys() for row in values)) - {"suite", "method", "seed", "n"})
        for metric in metrics:
            scores = [row[metric] for row in values if metric in row]
            result.append({"suite": suite, "method": method, "metric": metric,
                           "mean": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                           "seeds": len(scores)})
    return per_seed, result


def evaluate(config, tasks=None, outputs=None):
    data = config["data"]
    task_path = Path(tasks or data["test_tasks"])
    outputs_path = Path(outputs or data["outputs"])
    calibration = json.loads(Path(data["calibration"]).read_text(encoding="utf-8"))
    if calibration["version"] != 2 or calibration["density_scale"] <= 0:
        raise ValueError("Invalid offline calibration")
    fixed_tasks = read_jsonl(task_path)
    refs = {r["id"]: r for r in fixed_tasks}
    if len(refs) != len(fixed_tasks) or any(row.get("split") != "test" for row in refs.values()):
        raise ValueError("Benchmark tasks require unique IDs and the held-out test split")
    calibrated_songs = set(calibration["training_songs"]) | set(calibration["development_songs"])
    if any(f"{row['dataset']}:{row['song_id']}" in calibrated_songs for row in refs.values()):
        raise ValueError("Benchmark task overlaps offline calibration data")
    out = ensure_dir(config["output"]["dir"])
    render = load_config(config["render_config"])["render"].copy()
    render.pop("final_name", None)
    seen = set()
    scores = []
    quality = defaultdict(list)
    artifacts = read_jsonl(outputs_path)
    if any("midi" in artifact for artifact in artifacts) and render_profile(render) != calibration["render_profile"]:
        raise ValueError("Evaluation FluidSynth, SoundFont and mix settings must match offline calibration")
    for index, artifact in enumerate(artifacts):
        task_id, method, seed = artifact["task_id"], artifact["method"], int(artifact["seed"])
        round_id = int(artifact["round"]) if "round" in artifact else None
        identity = (task_id, method, seed, round_id)
        if identity in seen:
            raise ValueError(f"Duplicate final output {identity}")
        seen.add(identity)
        row = refs[task_id].copy()
        suite = row["suite"]
        audio = None
        if "midi" in artifact:
            row["midi"] = str((outputs_path.parent / artifact["midi"]).resolve(strict=True))
            midi = read_midi(row["midi"])
            audio = render_midi_to_wav(row["midi"], out / "rendered" / f"{index:06}" / "Music.wav",
                                       duration_seconds=float(bar_times(midi, int(row["bars"]))[-1]), **render)
            if "rounds" in artifact:
                row["rounds"] = artifact["rounds"]
            sample_scores = score_output(row, calibration, task_path.parent, audio)
        else:
            audio = (outputs_path.parent / artifact["audio"]).resolve(strict=True)
            sample_scores = score_audio_output(audio, row.get("reference", {}))
        if "motion_peaks" in row:
            sample_scores["ta_beat"] = temporal_alignment(audio, row["motion_peaks"])
        if "prompt" in row and "clap" in config.get("quality", {}):
            sample_scores["clap"] = clap_similarity(audio, row["prompt"], **config["quality"]["clap"])
        if "keyframes" in row and "imagebind" in config.get("quality", {}):
            sample_scores["imagebind"] = imagebind_similarity(
                audio, [task_path.parent / (frame["path"] if isinstance(frame, dict) else frame)
                        for frame in row["keyframes"]], **config["quality"]["imagebind"])
        if "reference_audio" in row and round_id is None:
            quality[(suite, method, seed)].append((task_path.parent / row["reference_audio"], audio))
        sample = {"task_id": task_id, "suite": suite, "method": method, "seed": seed, "scores": sample_scores}
        if round_id is not None:
            sample.update(round=round_id, suite=f"repair_round_{round_id:02d}")
        scores.append(sample)
    for (suite, method, seed), pairs in quality.items():
        reference, generated = zip(*pairs)
        metrics = {"kld": audio_feature_kld(reference, generated)}
        if "fad" in config.get("quality", {}):
            metrics["fad"] = fad_score(reference, generated, **config["quality"]["fad"])
        scores.append({"suite": suite, "method": method, "seed": seed, "scores": metrics})
    intervals = {}
    if "ratings" in data:
        samples, intervals = listening_scores(read_jsonl(data["ratings"]))
        scores.extend(samples)
    for field in ("efficiency", "failures"):
        if field not in data:
            continue
        observations = defaultdict(list)
        for row in read_jsonl(data[field]):
            key = (row["suite"], row["method"], int(row["seed"]), row["sample_id"], row["metric"])
            observations[key].append(float(row["value"]))
        for (suite, method, seed, sample_id, metric), values in observations.items():
            scores.append({"task_id": sample_id, "suite": suite, "method": method, "seed": seed,
                           "scores": {metric: float(np.mean(values))}})
    per_seed, summary = aggregate(scores)
    for item in summary:
        key = (item["suite"], item["method"], item["metric"])
        if key in intervals:
            item["ci_low"], item["ci_high"] = intervals[key]
    payload = {"samples": scores, "per_seed": per_seed, "summary": summary}
    (out / "eval_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_vhcmusic.yaml")
    parser.add_argument("--tasks")
    parser.add_argument("--outputs")
    args = parser.parse_args()
    payload = evaluate(load_config(args.config), args.tasks, args.outputs)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
