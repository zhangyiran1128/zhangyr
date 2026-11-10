"""Log-mel difference map for local editing."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import save_figure, setup_paper_style


def plot_local_edit_diff(before: str | Path, after: str | Path, output: str | Path):
    setup_paper_style()
    y0, sr = librosa.load(before, sr=22050, mono=True)
    y1, _ = librosa.load(after, sr=sr, mono=True)
    n = min(y0.size, y1.size)
    y0, y1 = y0[:n], y1[:n]
    m0 = np.log1p(librosa.feature.melspectrogram(y=y0, sr=sr, n_mels=96))
    m1 = np.log1p(librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=96))
    diff = np.abs(m1 - m0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(diff, aspect="auto", origin="lower", cmap="inferno")
    ax.set_title("Local edit log-mel difference")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel bin")
    fig.colorbar(im, ax=ax, fraction=0.025)
    return save_figure(fig, output, "local_edit_diff")


def plot_midi_edit_diff(before, after, output, mask_file):
    from vhcmusic.representation.event_grid import event_grid_to_pianoroll, grid_difference
    from vhcmusic.representation.midi_io import midi_to_event_grid

    setup_paper_style()
    a = midi_to_event_grid(before)
    b = midi_to_event_grid(after, bars=a.bars, roles=a.roles)
    mask = np.asarray(np.load(mask_file), dtype=bool)
    if mask.shape != a.shape[:2]:
        raise ValueError("The edit mask must match the MIDI's role and bar dimensions")
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    for grid, ax, title in ((a, axes[0], "Before"), (b, axes[1], "After")):
        roll = event_grid_to_pianoroll(grid)
        for role, notes in enumerate(roll):
            time, pitch = np.nonzero(notes)
            ax.scatter(time / a.subdivisions + 1, pitch + a.pitch_min, s=1.5, label=a.roles[role])
        for bar in np.flatnonzero(mask.any(axis=0)):
            ax.axvspan(bar + 1, bar + 2, color="firebrick", alpha=0.08)
        ax.set(ylabel="Pitch", title=title)
    axes[0].legend(ncol=len(a.roles), loc="upper right", fontsize=7)
    change = axes[2].imshow(grid_difference(a, b), aspect="auto", cmap="magma",
                            extent=(1, a.bars + 1, len(a.roles) - 0.5, -0.5))
    for role, bar in np.argwhere(mask):
        axes[2].add_patch(plt.Rectangle((bar + 1, role - 0.5), 1, 1, fill=False, edgecolor="cyan", linewidth=0.8))
    axes[2].set(yticks=range(len(a.roles)), yticklabels=a.roles, xlabel="Bar")
    fig.colorbar(change, ax=axes[2], label="Mean event-grid change")
    return save_figure(fig, output, "local_edit_midi_diff")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask", help="Render MIDI piano rolls using the saved role/bar edit mask")
    args = parser.parse_args()
    if args.mask:
        print(plot_midi_edit_diff(args.before, args.after, args.output, args.mask))
    else:
        print(plot_local_edit_diff(args.before, args.after, args.output))


if __name__ == "__main__":
    main()
