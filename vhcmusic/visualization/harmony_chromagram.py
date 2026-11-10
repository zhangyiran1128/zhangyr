"""Chromagram and target chord-mask visualization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vhcmusic.representation.midi_io import midi_to_event_grid
from vhcmusic.verification.chord_analyzer import QUALITY_TEMPLATES, ROOT_NAMES

from .plot_utils import save_figure, setup_paper_style


def chord_mask(plan: dict, bars: int) -> np.ndarray:
    mask = np.zeros((12, bars), dtype=np.float32)
    for syn in plan["syntax"]:
        bar = int(syn["bar_id"]) - 1
        if bar >= bars:
            continue
        label = syn["chord"]
        if label == "N:none":
            continue
        root, quality = label.split(":", 1)
        if root not in ROOT_NAMES or quality not in QUALITY_TEMPLATES:
            continue
        root_idx = ROOT_NAMES.index(root)
        for interval in QUALITY_TEMPLATES[quality]:
            mask[(root_idx + interval) % 12, bar] = 1.0
    return mask


def plot_harmony_chromagram(midi: str | Path, plan_json: str | Path, output: str | Path):
    setup_paper_style()
    grid = midi_to_event_grid(midi)
    pc = np.zeros((12, grid.bars), dtype=np.float32)
    onsets = grid.data[..., 0] > 0.5
    for bar in range(grid.bars):
        for pitch_idx in np.flatnonzero(onsets[:, bar].any(axis=(0, 1))):
            pc[(grid.pitch_min + pitch_idx) % 12, bar] += float(onsets[:, bar, :, pitch_idx].sum())
    pc /= np.maximum(1.0, pc.sum(axis=0, keepdims=True))
    plan = json.loads(Path(plan_json).read_text(encoding="utf-8"))
    mask = chord_mask(plan, grid.bars)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].imshow(pc, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("Generated pitch-class activation")
    axes[0].set_ylabel("Pitch class")
    axes[1].imshow(mask, aspect="auto", origin="lower", cmap="gray_r")
    axes[1].set_title("Target chord pitch-class mask")
    axes[1].set_xlabel("Bar")
    axes[1].set_ylabel("Pitch class")
    return save_figure(fig, output, "harmony_chromagram")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(plot_harmony_chromagram(args.midi, args.plan, args.output))


if __name__ == "__main__":
    main()
