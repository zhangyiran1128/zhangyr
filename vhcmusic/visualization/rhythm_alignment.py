"""Rhythm alignment visualization."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import save_figure, setup_paper_style


def plot_rhythm_alignment(audio: str | Path, output: str | Path, video_peaks: str | Path | None = None):
    setup_paper_style()
    y, sr = librosa.load(audio, sr=22050, mono=True)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.frames_to_time(np.arange(onset.size), sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(times, onset, label="Onset envelope", lw=1.2)
    for bt in beat_times:
        ax.axvline(bt, color="black", alpha=0.25, lw=0.7)
    if video_peaks:
        peaks = np.load(video_peaks)
        ax.vlines(peaks, ymin=0, ymax=float(onset.max()), color="tab:red", alpha=0.4, label="Video motion peaks")
    ax.set_title(f"Rhythm alignment, tempo={float(tempo):.1f}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strength")
    ax.legend(loc="upper right")
    return save_figure(fig, output, "rhythm_alignment")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--video-peaks", default=None)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(plot_rhythm_alignment(args.audio, args.output, args.video_peaks))


if __name__ == "__main__":
    main()
