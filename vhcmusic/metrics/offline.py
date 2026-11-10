"""Independent outcome evaluation under Section 4.3 and Appendix B."""

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from vhcmusic.metrics.control_metrics import active_role_f1, section_boundary_f1
from vhcmusic.metrics.midi_events import bar_times, event_curves, note_events
from vhcmusic.metrics.editing_metrics import (
    DENSITY_ENERGY_TOLERANCE, DIRECTIONAL_CHANGE, note_edit_scores, target_success_rate,
)
from vhcmusic.planning.schema import TRACK_ROLES
from vhcmusic.representation.midi_io import midi_metadata, read_midi
from vhcmusic.representation.harmony import chord_identity, classify_chroma

BOUNDARY_TOLERANCE = 1
TEMPO_BINS = [40, 60, 70, 90, 110, 140, 160, 210]
KERNEL_HALF_WIDTH = 4
SAMPLE_RATE = 44100
HOP = 512


def audio_features(path: Path, midi, bars: int):
    import librosa

    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    edges = bar_times(midi, bars)
    if not len(y) or len(y) / SAMPLE_RATE + 0.05 < edges[-1]:
        raise ValueError(f"Audio does not cover the annotated MIDI bars: {path}")
    chroma = librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)
    frames = np.arange(chroma.shape[1]) * HOP / SAMPLE_RATE
    bar_chroma, bar_features = [], []
    for start, end in zip(edges, edges[1:]):
        idx = np.flatnonzero((frames >= start) & (frames < end))
        if not idx.size:
            idx = [int(np.abs(frames - (start + end) / 2).argmin())]
        c = chroma[:, idx].mean(1)
        bar_chroma.append(c)
        bar_features.append(np.r_[c, mfcc[:, idx].mean(1), np.log(np.maximum(rms[:, idx] ** 2, 1e-12)).mean()])
    return np.array(bar_features), np.array(bar_chroma)


def structural_novelty(features, mean, std):
    x = (features - mean) / std
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    similarity = x @ x.T
    half = KERNEL_HALF_WIDTH
    weights = np.r_[-np.ones(half), np.ones(half)] * np.exp(-0.5 * (np.r_[np.arange(-half, 0), np.arange(1, half + 1)] / half)**2)
    kernel = np.outer(weights, weights)
    kernel /= np.abs(kernel).sum()
    novelty = np.zeros(len(x))
    for i in range(half, len(x) - half + 1):
        novelty[i] = np.sum(similarity[i - half:i + half, i - half:i + half] * kernel)
    return novelty


def detect_boundaries(novelty, threshold):
    peaks, _ = find_peaks(np.r_[novelty, 0.0], height=threshold)
    return (peaks + 1).tolist()


def _key(chroma):
    major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    profiles = np.array([np.roll(p - p.mean(), r) for p in (major, minor) for r in range(12)])
    best = int((profiles @ chroma / np.linalg.norm(profiles, axis=1)).argmax())
    return best % 12, ("maj", "min")[best // 12]


def tempo_key_scores(reference, tempo=None, key=None):
    scores = {}
    if "tempo" in reference and tempo is not None:
        delta = abs(int(np.digitize(tempo, TEMPO_BINS))
                    - int(np.digitize(float(reference["tempo"]), TEMPO_BINS)))
        scores.update(tb=100.0 * (delta == 0), tbt=100.0 * (delta <= 1))
    if "key" in reference and key is not None:
        target = chord_identity(reference["key"])
        relative = ((target[0] + (9 if target[1] == "maj" else 3)) % 12,
                    "min" if target[1] == "maj" else "maj")
        scores.update(ck=100.0 * (key == target), ckd=100.0 * (key in (target, relative)))
    return scores


def score_audio_output(audio, reference):
    import librosa

    if not ({"tempo", "key"} & reference.keys()):
        return {}
    y, _ = librosa.load(audio, sr=SAMPLE_RATE, mono=True)
    tempo, key = None, None
    if "tempo" in reference:
        estimate, _ = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE, hop_length=HOP)
        tempo = float(np.ravel(estimate)[0])
    if "key" in reference:
        key = _key(librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE, hop_length=HOP).mean(1))
    return tempo_key_scores(reference, tempo, key)


def score_output(row: dict, calibration: dict, root: Path, audio: Path | None = None) -> dict:
    midi = read_midi(root / row["midi"])
    bars = int(row["bars"])
    if bars < 1:
        raise ValueError("Evaluation requires a positive annotated bar count")
    roles = tuple(row.get("roles", TRACK_ROLES))
    ref = row.get("reference", {})
    density, activity, counts, energy = event_curves(midi, bars, roles, calibration["density_scale"])
    scores = {}
    mean = np.asarray(calibration["feature_mean"])
    std = np.asarray(calibration["feature_std"])
    structure = {"boundaries", "chords", "key"} & ref.keys()
    boundaries, chords = None, None
    if structure or any(t["attribute"] in {"boundaries", "chord"} for t in row.get("targets", [])):
        if audio is None:
            raise ValueError("Structural and harmonic evaluation require rendered audio")
        features, chroma = audio_features(audio, midi, bars)
        boundaries = detect_boundaries(structural_novelty(features, mean, std), calibration["novelty_threshold"])
        chords = [classify_chroma(c) for c in chroma]
    if "boundaries" in ref:
        target = [b for b in ref["boundaries"] if b != 1]
        scores["boundary_f1"] = section_boundary_f1(target, boundaries, BOUNDARY_TOLERANCE)["f1"]
    if "chords" in ref:
        if len(ref["chords"]) != bars:
            raise ValueError("Reference chord count differs from bars")
        matches = [chord_identity(label) == chord_identity(observed)
                   for label, observed in zip(ref["chords"], chords) if label is not None]
        if matches:
            scores["chord_acc"] = float(np.mean(matches))
    if "density" in ref:
        target = np.asarray(ref["density"])
        if target.shape != density.shape:
            raise ValueError("Reference density shape differs from bars")
        scores["rhythm_mae"] = float(np.abs(density - target).mean())
    if "activity" in ref:
        if np.asarray(ref["activity"]).shape != activity.shape:
            raise ValueError("Reference activity shape differs from role/bar geometry")
        scores["track_f1"] = active_role_f1(ref["activity"], activity)["f1"]
    scores.update(tempo_key_scores(ref, midi_metadata(midi).tempo, _key(chroma.mean(0)) if "key" in ref else None))
    if "original_midi" in row:
        before = read_midi(root / row["original_midi"])
        mask = {(int(bar), role) for bar, role in row["editable"]}
        if any(not 1 <= bar <= bars or role not in roles for bar, role in mask):
            raise ValueError("Editable mask contains an invalid bar or role")
        scores.update(note_edit_scores(note_events(before, bars, roles), note_events(midi, bars, roles), mask))
        if len(mask) == len(roles) * bars:
            scores.pop("npr")
        if row.get("targets"):
            original = event_curves(before, bars, roles, calibration["density_scale"])
            observed = (density, activity, counts, energy)
            notes = note_events(midi, bars, roles)
            success = target_success_rate(
                _target_hit(target, original, observed, roles, calibration, boundaries, chords, notes)
                for target in row["targets"]
            )
            if success is not None:
                scores["tsr"] = success
    if "rounds" in row:
        scores["repair_rounds"] = int(row["rounds"])
    return scores


def _target_hit(target, original, edited, roles, calibration, boundaries, chords, notes):
    attr = target["attribute"]
    bars = np.asarray(target["bars"], dtype=int) - 1
    role = target.get("role")
    if not len(bars) or bars.min() < 0 or bars.max() >= len(edited[0]):
        raise ValueError("Target bars outside the evaluation span")
    if attr == "boundaries":
        wanted = target["value"]
        relevant = set(bars + 1)
        found = [b for b in boundaries if any(abs(b - r) <= BOUNDARY_TOLERANCE for r in relevant)]
        return section_boundary_f1(wanted, found, BOUNDARY_TOLERANCE)["f1"] == 1
    if attr == "chord":
        return all(chord_identity(chords[b]) == chord_identity(target["value"]) for b in bars)
    if attr == "register":
        pitches = [n.pitch for n in notes if n.bar - 1 in bars and (role is None or n.role == role)]
        low, high = target["value"]
        return bool(pitches and low <= np.median(pitches) <= high)

    def value(curves):
        density, activity, counts, energy = curves
        if attr == "density":
            values = density if role is None else np.clip(counts[roles.index(role)] / calibration["role_density_scale"][role], 0, 1)
        elif attr == "activity":
            values = activity.mean(0) if role is None else activity[roles.index(role)]
        elif attr == "energy":
            values = np.clip(energy / calibration["energy_scale"], 0, 1)
        else:
            raise ValueError(f"Unsupported editing target attribute: {attr}")
        return float(np.mean(values[bars]))

    current = value(edited)
    if "direction" in target:
        baseline = value(original)
        direction = {"increase": 1, "decrease": -1}[target["direction"]]
        magnitude = float(target.get("minimum_change", DIRECTIONAL_CHANGE * baseline))
        return direction * (current - baseline) > 0 and direction * (current - baseline) >= magnitude
    tolerance = DENSITY_ENERGY_TOLERANCE if attr in {"density", "energy"} else 0.0
    return abs(current - float(target["value"])) <= tolerance



def temporal_alignment(audio, motion_peaks, tolerance=0.1):
    import librosa

    wave, _ = librosa.load(audio, sr=SAMPLE_RATE, mono=True)
    _, beats = librosa.beat.beat_track(y=wave, sr=SAMPLE_RATE, hop_length=HOP, units="time")
    if not len(motion_peaks):
        raise ValueError("TA/Beat requires at least one annotated motion peak")
    return float(np.mean([np.any(np.abs(beats - float(peak)) <= tolerance) for peak in motion_peaks]))
