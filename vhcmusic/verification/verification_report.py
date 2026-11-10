"""Table A1 online checks for the repair agent."""

from typing import Any

import numpy as np

from vhcmusic.metrics.control_metrics import active_role_f1
from vhcmusic.planning.schema import validate_plan
from vhcmusic.representation.event_grid import EventGrid
from vhcmusic.representation.harmony import chord_identity

from .chord_analyzer import ChordAnalyzer
from .rhythm_analyzer import RhythmAnalyzer
from .section_analyzer import SectionAnalyzer

SECTION_TOLERANCE_BARS = 0
CHORD_CONFIDENCE_MARGIN = 0.05
RHYTHM_MAX_ERROR = 0.10
NON_TARGET_MAX_MISMATCH = 0.10


def build_verification_report(
    target_plan: dict[str, Any],
    generated_grid: EventGrid,
    original_grid: EventGrid | None = None,
    edit_mask: np.ndarray | None = None,
    *,
    section_analyzer: SectionAnalyzer | None = None,
    chord_analyzer: ChordAnalyzer | None = None,
    density_scale: float = 24.0,
) -> dict[str, Any]:
    if validate_plan(target_plan, bars=generated_grid.bars, roles=generated_grid.roles).bars != generated_grid.bars:
        raise ValueError('The target plan must cover all verified bars')
    section = (section_analyzer or SectionAnalyzer()).analyze(generated_grid)
    chord = (chord_analyzer or ChordAnalyzer.from_calibration()).analyze(generated_grid)
    rhythm = RhythmAnalyzer(density_scale).analyze(generated_grid)
    target_sections = {s["start_bar"]: s for s in target_plan["structure"]}
    observed_sections = {s["start_bar"]: s for s in section["sections"]}
    section_errors = sorted(b for b in target_sections.keys() | observed_sections.keys()
                            if b not in target_sections or b not in observed_sections
                            or target_sections[b]["section"] != observed_sections[b]["section"])
    observed_chords = {c["bar_id"]: c for c in chord["chords"]}
    syntax = sorted(target_plan["syntax"], key=lambda s: s["bar_id"])
    chord_errors = [s["bar_id"] for s in syntax
                    if chord_identity(s["chord"]) != chord_identity(observed_chords[s["bar_id"]]["chord"])
                    or observed_chords[s["bar_id"]]["confidence"] < CHORD_CONFIDENCE_MARGIN]
    density_error = np.abs(np.asarray(rhythm["rhythm_density_curve"][:len(syntax)])
                           - [s["density"] for s in syntax])
    density_errors = (np.flatnonzero(density_error > RHYTHM_MAX_ERROR + 1e-8) + 1).tolist()
    activity = np.asarray(rhythm["track_level_activity"])[:, :len(syntax)] > 0.5
    target_activity = np.array([[role in s["active_roles"] for s in syntax] for role in generated_grid.roles])
    track_errors = (np.flatnonzero(np.any(activity != target_activity, axis=0)) + 1).tolist()
    if (original_grid is None) != (edit_mask is None):
        raise ValueError("Preservation verification requires the original grid and edit mask together")
    preservation = {"available": False, "score": 0.0}
    protected_bars = []
    if original_grid is not None:
        if original_grid.shape != generated_grid.shape or np.shape(edit_mask) != generated_grid.shape[:2]:
            raise ValueError("Preservation comparison requires matching grid and mask dimensions")
        before, after = original_grid.data, generated_grid.data
        protected = ~np.asarray(edit_mask, dtype=bool)[..., None, None]
        a = (before[..., 0] > 0.5) & protected
        b = (after[..., 0] > 0.5) & protected
        changed = (a ^ b) | (a & b & np.any(before[..., 1:] != after[..., 1:], axis=-1))
        preservation = {"available": bool(protected.any()), "score": int(changed.sum()) / max(1, int((a | b).sum()))}
        protected_bars = (np.flatnonzero(changed.any(axis=(0, 2, 3))) + 1).tolist()
    mismatch = sorted(set(section_errors + chord_errors + density_errors + track_errors))
    passed = not mismatch and preservation["score"] <= NON_TARGET_MAX_MISMATCH
    return {
        "pass": passed,
        "structure_mismatch": {"mismatch_bars": section_errors, "target_boundaries": list(target_sections),
                               "predicted_boundaries": list(observed_sections), "tolerance_bars": SECTION_TOLERANCE_BARS},
        "chord_mismatch": {"mismatch_bars": chord_errors, "minimum_confidence": CHORD_CONFIDENCE_MARGIN},
        "rhythm_mismatch": {"mae": float(density_error.mean()), "mismatch_bars": density_errors,
                            "maximum_error": RHYTHM_MAX_ERROR},
        "track_activity_mismatch": {"mismatch_bars": track_errors, **active_role_f1(target_activity, activity)},
        "preservation_mismatch": {**preservation, "mismatch_bars": protected_bars,
                                  "maximum_mismatch": NON_TARGET_MAX_MISMATCH},
        "mismatch_bars": mismatch,
        "analyzers": {"section": section, "chord": chord, "rhythm": rhythm},
    }
