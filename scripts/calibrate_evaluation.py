#!/usr/bin/env python
"""Freeze offline normalization and checkerboard peak threshold before test scoring."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate import read_jsonl
from vhcmusic.metrics.control_metrics import section_boundary_f1
from vhcmusic.metrics.midi_events import bar_ticks, bar_times, event_curves
from vhcmusic.metrics.offline import audio_features, detect_boundaries, structural_novelty
from vhcmusic.planning.schema import TRACK_ROLES
from vhcmusic.rendering.fluidsynth_render import render_midi_to_wav, render_profile
from vhcmusic.representation.midi_io import read_midi
from vhcmusic.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--render-config", default="configs/render.yaml")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/calibration.json"))
    args = parser.parse_args()
    training = read_jsonl(args.training)
    dev = read_jsonl(args.development)
    train_ids = {f"{r['dataset']}:{r['song_id']}" for r in training}
    dev_ids = {f"{r['dataset']}:{r['song_id']}" for r in dev}
    if train_ids & dev_ids or any(r["split"] != "train" for r in training) or any(r["split"] != "val" for r in dev):
        raise ValueError("Calibration requires disjoint training and development songs")
    counts_by_beat, by_role, energies = [], [], []
    for row in training:
        midi = read_midi(args.training.parent / row["midi"])
        bars = int(row["bars"])
        _, _, counts, energy = event_curves(midi, bars, TRACK_ROLES, 1)
        counts_by_beat.extend(counts.sum(0) / (np.diff(bar_ticks(midi, bars)) / midi.ticks_per_beat))
        by_role.append(counts)
        energies.extend(energy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render = load_config(args.render_config)["render"].copy()
    render.pop("final_name", None)
    profile = render_profile(render)
    feature_sets = []
    for i, row in enumerate(dev):
        midi_path = args.development.parent / row["midi"]
        midi = read_midi(midi_path)
        bars = int(row["bars"])
        audio = render_midi_to_wav(midi_path, args.output.parent / "development_audio" / f"{i:06}" / "Music.wav",
                                   duration_seconds=float(bar_times(midi, bars)[-1]), **render)
        features, _ = audio_features(audio, midi, bars)
        feature_sets.append((features, [b for b in row["boundaries"] if b != 1]))
    stacked = np.concatenate([features for features, _ in feature_sets])
    mean, std = stacked.mean(0), stacked.std(0).clip(min=1e-6)
    curves = [(structural_novelty(features, mean, std), reference) for features, reference in feature_sets]
    candidates = sorted(set(float(value) for curve, _ in curves for value in curve if value > 0) | {1.0})
    if len(candidates) > 200:
        candidates = np.quantile(candidates, np.linspace(0, 1, 200)).tolist()
    threshold = max(candidates, key=lambda th: (np.mean([
        section_boundary_f1(reference, detect_boundaries(curve, th), tolerance=1)["f1"]
        for curve, reference in curves]), th))
    payload = {"version": 2, "render_profile": profile, "feature_mean": mean.tolist(), "feature_std": std.tolist(),
               "novelty_threshold": threshold, "training_songs": sorted(train_ids), "development_songs": sorted(dev_ids),
               "training_manifest_sha256": hashlib.sha256(args.training.read_bytes()).hexdigest(),
               "development_manifest_sha256": hashlib.sha256(args.development.read_bytes()).hexdigest(),
               "density_scale": max(1.0, float(np.percentile(counts_by_beat, 95))),
               "role_density_scale": dict(zip(TRACK_ROLES, np.percentile(np.concatenate(by_role, axis=1), 95, axis=1).clip(min=1).tolist())),
               "energy_scale": max(1.0, float(np.percentile(energies, 95)))}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
