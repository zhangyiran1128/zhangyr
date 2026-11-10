"""Audio self-similarity visualization for structure control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from .plot_utils import save_figure, setup_paper_style


def compute_self_similarity(audio_path: str | Path) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    feat = np.vstack([chroma, mfcc])
    feat = librosa.util.normalize(feat, axis=0)
    return feat.T @ feat


def plot_structure_similarity(audio_path: str | Path, output: str | Path, plan_json: str | Path | None = None):
    setup_paper_style()
    sim = compute_self_similarity(audio_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, origin="lower", aspect="auto", cmap="magma")
    ax.set_title("Structure self-similarity")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")
    fig.colorbar(im, ax=ax, fraction=0.046)
    if plan_json:
        plan = json.loads(Path(plan_json).read_text(encoding="utf-8"))
        bars = max(s["end_bar"] for s in plan["structure"])
        frames_per_bar = sim.shape[0] / max(1, bars)
        for sec in plan["structure"]:
            x = (sec["start_bar"] - 1) * frames_per_bar
            ax.axvline(x, color="cyan", lw=0.8)
            ax.axhline(x, color="cyan", lw=0.8)
    return save_figure(fig, output, "structure_similarity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--plan", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(plot_structure_similarity(args.audio, args.output, args.plan))


if __name__ == "__main__":
    main()
