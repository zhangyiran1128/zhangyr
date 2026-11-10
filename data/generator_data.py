"""Song-isolated generator data and MIDI-derived conditioning."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vhcmusic.planning.schema import default_global_constraints, validate_plan
from vhcmusic.representation.event_grid import load_event_grid

GENERATOR_DATASETS = ("pop909", "symmv", "slakh2100")


def generator_items(root):
    root = Path(root)
    items = []
    assigned = {}
    for dataset in GENERATOR_DATASETS:
        manifest = root / dataset / "items.jsonl"
        if not manifest.is_file():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["dataset"].lower() != dataset:
                raise ValueError(f"Unexpected dataset in {manifest}")
            song = (dataset, row["song_id"])
            split = row["split"]
            if split not in {"train", "val", "test"} or assigned.setdefault(song, split) != split:
                raise ValueError(f"Invalid or overlapping split for {song}")
            row["path"] = str(Path(row["path"]).resolve(strict=True))
            items.append(row)
    if not items:
        raise FileNotFoundError(f"No generator manifests in {root}")
    return items


def grid_statistics(items):
    note_rate, role_counts, energies = [], [], []
    for row in items:
        if row["split"] != "train":
            continue
        grid = load_event_grid(row["path"])
        onsets = grid.data[..., 0] > 0.5
        counts = onsets.sum(axis=(2, 3))
        note_rate.extend(counts.sum(axis=0) / (grid.beats_per_bar * 4 / grid.beat_unit))
        role_counts.append(counts)
        energies.extend((grid.data[..., 2] * onsets).sum(axis=(0, 2, 3)))
    if not role_counts:
        raise ValueError("Training split has no event grids")
    return {"density_scale": max(1.0, float(np.percentile(note_rate, 95))),
            "role_density_scale": np.percentile(np.concatenate(role_counts, axis=1), 95, axis=1).clip(min=1).tolist(),
            "energy_scale": max(1.0, float(np.percentile(energies, 95)))}


def derive_plan(grid, stats, annotation=None):
    from vhcmusic.verification.chord_analyzer import ChordAnalyzer
    from vhcmusic.verification.rhythm_analyzer import RhythmAnalyzer

    annotation = annotation or {}
    rhythm = RhythmAnalyzer(stats["density_scale"]).analyze(grid)
    chords = ChordAnalyzer(transition_scale=0).analyze(grid)["chords"]
    chords = [row["chord"] for row in chords]
    onsets = grid.data[..., 0] > 0.5
    counts = onsets.sum(axis=(2, 3))
    activity = counts > 0
    energy = np.clip((grid.data[..., 2] * onsets).sum(axis=(0, 2, 3)) / stats["energy_scale"], 0, 1)
    sections = annotation.get("structure")
    if sections is None:
        # Detect broad arrangement changes when functional annotations are unavailable.
        from scipy.signal import find_peaks

        change = np.r_[0, np.abs(np.diff(energy)) + np.count_nonzero(np.diff(activity, axis=1), axis=0) / len(grid.roles)]
        peaks, _ = find_peaks(np.r_[change, 0], height=0.4, distance=4)
        starts = [1, *(int(p) + 1 for p in peaks if p >= 3 and grid.bars - p >= 3)]
        sections = []
        for index, (start, end) in enumerate(zip(starts, starts[1:] + [grid.bars + 1])):
            local_energy = float(energy[start - 1:end - 1].mean())
            label = "instrumental" if len(starts) == 1 else (
                "intro" if index == 0 else "outro" if index == len(starts) - 1 else
                "chorus" if local_energy > float(np.median(energy)) else "verse")
            sections.append({"section": label, "start_bar": start, "end_bar": end - 1,
                             "energy": local_energy, "tension": float(np.mean(rhythm["rhythmic_change_curve"][start - 1:end - 1])),
                             "repeat_group": label, "transition_to_next": "none" if end > grid.bars else
                             ("build" if energy[end - 1] > local_energy else "release")})
    syntax, constraints = [], []
    annotated = {s["bar_id"]: s for s in annotation.get("syntax", [])}
    for l in range(grid.bars):
        chord = chords[l]
        sync = rhythm["syncopation_proxy"][l]
        syntax.append({"bar_id": l + 1, "chord": chord, "harmonic_func": "other",
                       "syncopation": sync, "groove_type": "syncopated" if sync >= 0.5 else "straight",
                       "thickness": float(activity[:, l].mean()), "cadence_strength": float(l == grid.bars - 1 or
                       any(s["end_bar"] == l + 1 for s in sections)),
                       "density": rhythm["rhythm_density_curve"][l],
                       "active_roles": [role for i, role in enumerate(grid.roles) if activity[i, l]],
                       **annotated.get(l + 1, {})})
        for i, role in enumerate(grid.roles):
            pitches = np.nonzero(onsets[i, l])[1] + grid.pitch_min
            register = None
            if pitches.size:
                center = np.median(pitches)
                register = "low" if center < 48 else "high" if center >= 72 else "mid"
            constraints.append({"bar_id": l + 1, "track_role": role, "editable": True,
                                "target_density": min(1.0, float(counts[i, l] / stats["role_density_scale"][i])),
                                "target_activity": float(activity[i, l]), "target_register": register,
                                "boundary_smoothness": 1.0 - rhythm["rhythmic_change_curve"][l]})
    plan = {"structure": sections, "syntax": syntax, "constraints": constraints,
            "global_constraints": default_global_constraints()}
    validate_plan(plan, bars=grid.bars, roles=grid.roles)
    return plan


class GeneratorDataset(Dataset):
    def __init__(self, items, split, *, statistics=None, max_bars=64):
        self.items = [row for row in items if row["split"] == split]
        if not self.items:
            raise ValueError(f"Generator {split} split has no records")
        self.statistics = statistics
        self.max_bars = max_bars
        self.plans = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        grid = load_event_grid(row["path"])
        if grid.bars > self.max_bars:
            raise ValueError(f"Preprocess {row['path']} with at most {self.max_bars} bars")
        tensor = torch.from_numpy(grid.data)
        if self.statistics is None:
            return tensor, None
        if idx not in self.plans:
            self.plans[idx] = derive_plan(grid, self.statistics, row.get("annotation"))
        return tensor, self.plans[idx]


def collate_grids(batch):
    grids, plans = zip(*batch)
    lengths = torch.tensor([grid.shape[1] for grid in grids])
    bars = int(lengths.max())
    padded = grids[0].new_zeros((len(grids), grids[0].shape[0], bars, *grids[0].shape[2:]))
    for i, grid in enumerate(grids):
        padded[i, :, :grid.shape[1]] = grid
    valid = torch.arange(bars)[None] < lengths[:, None]
    return padded, list(plans), valid
