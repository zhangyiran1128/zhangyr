"""Appendix B edit success, preservation, and leakage."""

import numpy as np

from .midi_events import event_matches

NOTE_MATCH = {"onset": 0.25, "duration": 0.25, "relative_duration": 0.2, "velocity": 10}
DENSITY_ENERGY_TOLERANCE = 0.05
DIRECTIONAL_CHANGE = 0.15


def target_success_rate(target_results):
    applicable = [bool(result) for result in target_results if result is not None]
    return float(np.mean(applicable)) if applicable else None


def note_edit_scores(before, after, mask):
    def protected(note):
        return (note.bar, note.role) not in mask

    changes = protected_changes = protected_total = protected_matches = 0
    for is_protected in (True, False):
        a = [n for n in before if protected(n) == is_protected]
        b = [n for n in after if protected(n) == is_protected]
        pairs = event_matches(a, b, **NOTE_MATCH)
        matched = len(pairs)
        ai, bi = {i for i, _ in pairs}, {i for _, i in pairs}
        a_left = [note for i, note in enumerate(a) if i not in ai]
        b_left = [note for i, note in enumerate(b) if i not in bi]
        modifications = len(event_matches(a_left, b_left, onset=NOTE_MATCH["onset"],
                                          duration=float("inf"), relative_duration=0, velocity=127))
        edits = len(a_left) + len(b_left) - modifications
        changes += edits
        if is_protected:
            protected_changes, protected_total, protected_matches = edits, len(a) + len(b), matched
    return {"npr": 2 * protected_matches / protected_total if protected_total else 1.0,
            "leakage": protected_changes / changes if changes else 0.0}
