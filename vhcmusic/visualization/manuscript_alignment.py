"""Bar-aligned structure, harmony, rhythm, and verification figures."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vhcmusic.metrics.midi_events import event_curves
from vhcmusic.metrics.offline import audio_features
from vhcmusic.planning.music_planner import normalize_plan
from vhcmusic.planning.schema import TRACK_ROLES, plan_bars
from vhcmusic.representation.harmony import classify_chroma
from vhcmusic.representation.midi_io import read_midi
from .plot_utils import save_figure, setup_paper_style


def plot_alignment(midi_path, audio, plan, report, calibration, output):
    setup_paper_style()
    midi = read_midi(midi_path)
    bars = plan_bars(plan)
    x = np.arange(1, bars + 1)
    roles = tuple(plan.get("roles", TRACK_ROLES))
    features, chroma = audio_features(Path(audio), midi, bars)
    density, activity, counts, velocity_energy = event_curves(midi, bars, roles, calibration["density_scale"])
    energy = np.clip(velocity_energy / calibration["energy_scale"], 0, 1)
    log_energy = features[:, -1]
    audio_energy = (log_energy - log_energy.min()) / max(1e-8, np.ptp(log_energy))
    target_energy = np.empty(bars)
    structure = plan["structure"]
    syntax = sorted(plan["syntax"], key=lambda row: row["bar_id"])
    for section in structure:
        target_energy[section["start_bar"] - 1:section["end_bar"]] = section["energy"]
    target_density = [row["density"] for row in syntax]
    observed = report["analyzers"]
    observed_chords = [row["chord"] for row in observed["chord"]["chords"]]
    generated_chords = [classify_chroma(bar_chroma) for bar_chroma in chroma]
    mismatch = {
        "Section": report["structure_mismatch"]["mismatch_bars"],
        "Energy": (np.flatnonzero(np.abs(energy - target_energy) > 0.10) + 1).tolist(),
        "Density": report["rhythm_mismatch"]["mismatch_bars"],
        "Chord": report["chord_mismatch"]["mismatch_bars"],
        "Track activity": report["track_activity_mismatch"]["mismatch_bars"],
    }
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for ax in axes[:3]:
        for index, section in enumerate(structure):
            ax.axvspan(section["start_bar"] - 0.5, section["end_bar"] + 0.5, color=f"C{index % 10}", alpha=0.10)
    for section in structure:
        midpoint = (section["start_bar"] + section["end_bar"]) / 2
        axes[0].text(midpoint, 0.83, section["section"], ha="center")
    for section in observed["section"]["sections"]:
        axes[0].axvline(section["start_bar"] - 0.5, color="firebrick", linestyle="--")
        axes[0].text((section["start_bar"] + section["end_bar"]) / 2, 0.24, section["section"], ha="center")
    axes[0].set(ylim=(0, 1), yticks=[0.24, 0.83], yticklabels=["Recovered", "Plan"])
    axes[1].plot(x, target_energy, label="Plan energy")
    axes[1].plot(x, energy, label="MIDI energy")
    axes[1].plot(x, target_density, linestyle="--", label="Plan density")
    axes[1].plot(x, density, linestyle="--", label="MIDI density")
    axes[1].plot(x, observed["rhythm"]["rhythm_density_curve"], linestyle=":", label="Back-translated density")
    axes[1].legend(ncol=3, fontsize=8)
    chord_rows = ([row["chord"] for row in syntax], generated_chords, observed_chords)
    for i, chord_sequence in enumerate(chord_rows):
        for bar, label in enumerate(chord_sequence, 1):
            color = "firebrick" if i and bar in mismatch["Chord"] else "black"
            axes[2].text(bar, i, label, ha="center", va="center", color=color, fontsize=7, rotation=45 if bars > 24 else 0)
    axes[2].set(ylim=(2.5, -0.5), yticks=range(3), yticklabels=["Plan", "Audio", "Back-translated"])
    normalized = counts / np.maximum(counts.max(axis=1, keepdims=True), 1)
    axes[3].imshow(normalized, aspect="auto", extent=(0.5, bars + 0.5, len(roles) - 0.5, -0.5), cmap="viridis")
    recovered = np.asarray(observed["rhythm"]["track_level_activity"], dtype=bool)
    for r, b in np.argwhere(recovered != activity):
        axes[3].scatter(b + 1, r, marker="x", color="red")
    axes[3].set(yticks=range(len(roles)), yticklabels=roles, ylabel="Activity")
    mismatch_map = np.array([[bar in positions for bar in x] for positions in mismatch.values()])
    axes[4].imshow(mismatch_map, aspect="auto", extent=(0.5, bars + 0.5, len(mismatch) - 0.5, -0.5),
                   cmap="Reds", vmin=0, vmax=1)
    axes[4].set(yticks=range(len(mismatch)), yticklabels=list(mismatch), xlabel="Bar", xlim=(0.5, bars + 0.5))
    outputs = [save_figure(fig, output, "three_layer_alignment")]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, target_energy, label="Planned energy")
    ax.plot(x, audio_energy, label="Rendered energy")
    for section in structure:
        ax.axvline(section["start_bar"] - 0.5, color="gray", alpha=0.6)
    ax.set(xlabel="Bar", ylabel="Normalized energy", xlim=(0.5, bars + 0.5))
    ax.legend()
    outputs.append(save_figure(fig, output, "structure_energy"))

    values = (features - np.array(calibration["feature_mean"])) / np.array(calibration["feature_std"])
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    fig, ax = plt.subplots(figsize=(6, 5))
    plot = ax.imshow(values @ values.T, extent=(0.5, bars + 0.5, 0.5, bars + 0.5),
                     origin="lower", cmap="magma", vmin=-1, vmax=1)
    for section in structure[1:]:
        boundary = section["start_bar"] - 0.5
        ax.axhline(boundary, color="cyan", linewidth=0.6)
        ax.axvline(boundary, color="cyan", linewidth=0.6)
    ax.set(xlabel="Bar", ylabel="Bar")
    fig.colorbar(plot, ax=ax, label="Cosine similarity")
    outputs.append(save_figure(fig, output, "structure_bar_similarity"))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("midi", "audio", "plan", "report", "calibration", "output"):
        parser.add_argument(f"--{option}", required=True)
    args = parser.parse_args()

    def load(path):
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))

    print(plot_alignment(args.midi, args.audio, normalize_plan(load(args.plan)), load(args.report),
                         load(args.calibration), args.output))


if __name__ == "__main__":
    main()
