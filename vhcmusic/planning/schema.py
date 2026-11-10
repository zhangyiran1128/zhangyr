"""Schema and validation for VHCMusic hierarchical plans."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


TRACK_ROLES = ("lead", "bass", "rhythm", "harmony_pad", "accompaniment", "drums")
SECTION_LABELS = (
    "intro",
    "verse",
    "pre_chorus",
    "chorus",
    "bridge",
    "breakdown",
    "outro",
    "instrumental",
    "interlude",
)
TRANSITIONS = ("none", "build", "release", "fill", "cut", "riser", "cadence", "smooth_lift", "build_up", "contrast", "stop")
HARMONIC_FUNCS = ("I", "ii", "iii", "IV", "V", "vi", "vii_dim", "tonic", "predominant", "dominant", "passing", "modal", "other")
GROOVE_TYPES = ("straight", "pop_drive", "ballad", "funk", "rock", "half_time", "four_on_floor", "latin", "sparse", "syncopated", "dance", "cinematic")
REGISTERS = ("low", "mid_low", "mid", "mid_high", "high", "wide")
REPAIR_ACTIONS = ("accept", "local_rewrite", "adjust_edit_strength", "global_resample")

CHORD_RE = re.compile(r"^[A-G](?:#|b)?:(?:maj|min|dim|aug|sus2|sus4|7|maj7|min7|dim7|hdim7|none)$")


class PlanValidationError(ValueError):
    """Raised when a hierarchical plan violates the schema."""


def _range01(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise PlanValidationError(f"{field} must be numeric")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise PlanValidationError(f"{field} must be in [0, 1], got {value}")
    return value


def _require(obj: dict[str, Any], field: str) -> Any:
    if field not in obj:
        raise PlanValidationError(f"Missing field: {field}")
    return obj[field]


def _validate_choice(value: Any, field: str, choices: tuple[str, ...]) -> str:
    if value not in choices:
        raise PlanValidationError(f"Illegal {field}: {value}. Allowed: {choices}")
    return str(value)


def default_global_constraints() -> dict[str, bool]:
    return {
        "keep_structure": True,
        "keep_chord_progression": True,
        "keep_main_motif": True,
        "keep_rhythm_pattern": False,
        "allow_arrangement_change": False,
    }


PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["structure", "syntax", "constraints", "global_constraints"],
    "properties": {
        "structure": {"type": "array"},
        "syntax": {"type": "array"},
        "constraints": {"type": "array"},
        "global_constraints": {"type": "object"},
    },
}


@dataclass(frozen=True)
class PlanSummary:
    """Compact validated plan summary."""

    bars: int
    roles: tuple[str, ...]
    sections: int


def validate_plan(plan: dict[str, Any], bars: int | None = None, roles: tuple[str, ...] = TRACK_ROLES) -> PlanSummary:
    """Validate a VHCMusic hierarchical plan.

    The plan uses one-based inclusive bar indices in JSON. Internally most tensor
    code converts them to zero-based slices.
    """

    if not isinstance(plan, dict):
        raise PlanValidationError("Plan must be a dictionary")
    for top_field in PLAN_JSON_SCHEMA["required"]:
        _require(plan, top_field)

    structure = plan["structure"]
    syntax = plan["syntax"]
    constraints = plan["constraints"]
    globals_ = plan["global_constraints"]
    if not isinstance(structure, list) or not structure:
        raise PlanValidationError("structure must be a non-empty list")
    if not isinstance(syntax, list) or not syntax:
        raise PlanValidationError("syntax must be a non-empty list")
    if not isinstance(constraints, list):
        raise PlanValidationError("constraints must be a list")

    seen_bars: set[int] = set()
    max_bar = 0
    last_end = 0
    for sec in structure:
        _validate_choice(_require(sec, "section"), "section", SECTION_LABELS)
        start = int(_require(sec, "start_bar"))
        end = int(_require(sec, "end_bar"))
        if start < 1 or end < start:
            raise PlanValidationError(f"Invalid section boundary: {start}-{end}")
        if start != last_end + 1:
            raise PlanValidationError(f"Overlapping or unsorted section at bar {start}")
        last_end = end
        max_bar = max(max_bar, end)
        _range01(_require(sec, "energy"), "energy")
        _range01(_require(sec, "tension"), "tension")
        if _require(sec, "repeat_group") is not None and not isinstance(sec["repeat_group"], str):
            raise PlanValidationError("repeat_group must be a string")
        _validate_choice(_require(sec, "transition_to_next"), "transition_to_next", TRANSITIONS)

    if bars is not None and max_bar > bars:
        raise PlanValidationError(f"Plan extends to {max_bar} bars, expected at most {bars}")

    for syn in syntax:
        bar_id = int(_require(syn, "bar_id"))
        if bar_id < 1 or bar_id > max_bar:
            raise PlanValidationError(f"syntax bar_id {bar_id} outside 1..{max_bar}")
        if bar_id in seen_bars:
            raise PlanValidationError(f'Duplicate syntax bar {bar_id}')
        seen_bars.add(bar_id)
        chord = _require(syn, "chord")
        if chord != "N:none" and not CHORD_RE.match(chord):
            raise PlanValidationError(f"Illegal chord label: {chord}")
        _validate_choice(_require(syn, "harmonic_func"), "harmonic_func", HARMONIC_FUNCS)
        _range01(_require(syn, "syncopation"), "syncopation")
        _validate_choice(_require(syn, "groove_type"), "groove_type", GROOVE_TYPES)
        _range01(_require(syn, "thickness"), "thickness")
        _range01(_require(syn, "cadence_strength"), "cadence_strength")
        _range01(_require(syn, "density"), "density")
        active_roles = _require(syn, "active_roles")
        if not isinstance(active_roles, list) or any(role not in roles for role in active_roles):
            raise PlanValidationError(f"active_roles must be a list drawn from {roles}")

    missing_syntax = set(range(1, max_bar + 1)) - seen_bars
    if missing_syntax:
        raise PlanValidationError(f"Missing syntax bars: {sorted(missing_syntax)[:8]}")

    seen_constraints = set()
    for con in constraints:
        bar_id = int(_require(con, "bar_id"))
        if bar_id < 1 or bar_id > max_bar:
            raise PlanValidationError(f"constraint bar_id {bar_id} outside 1..{max_bar}")
        _validate_choice(_require(con, "track_role"), "track_role", roles)
        identity = (bar_id, con["track_role"])
        if identity in seen_constraints:
            raise PlanValidationError(f"Duplicate constraint {identity}")
        seen_constraints.add(identity)
        if not isinstance(_require(con, "editable"), bool):
            raise PlanValidationError("editable must be bool")
        for key in ("target_density", "target_activity"):
            value = _require(con, key)
            if value is not None:
                _range01(value, key)
        if _require(con, "target_register") is not None:
            _validate_choice(con["target_register"], "target_register", REGISTERS)
        _range01(_require(con, "boundary_smoothness"), "boundary_smoothness")

    expected_globals = default_global_constraints()
    for key in expected_globals:
        if key not in globals_ or not isinstance(globals_[key], bool):
            raise PlanValidationError(f"global_constraints.{key} must be bool")

    return PlanSummary(bars=max_bar, roles=tuple(roles), sections=len(structure))


def validate_or_fallback(plan: dict[str, Any] | None, bars: int = 32, roles: tuple[str, ...] = TRACK_ROLES) -> dict[str, Any]:
    """Return a validated plan or a deterministic rule-based fallback plan."""

    if plan is not None:
        try:
            validate_plan(plan, bars=bars, roles=roles)
            return plan
        except PlanValidationError:
            pass
    return make_fallback_plan(bars=bars, roles=roles)


def make_fallback_plan(bars: int = 32, roles: tuple[str, ...] = TRACK_ROLES) -> dict[str, Any]:
    """Create a complete, schema-valid pop-form plan from rules."""

    bars = max(1, int(bars))
    boundaries = [
        ("intro", 1, max(4, bars // 8), 0.35, 0.25),
        ("verse", max(4, bars // 8) + 1, max(8, bars // 2), 0.55, 0.45),
        ("chorus", max(8, bars // 2) + 1, max(12, (bars * 3) // 4), 0.85, 0.75),
        ("outro", max(12, (bars * 3) // 4) + 1, bars, 0.45, 0.30),
    ]
    structure = []
    for name, start, end, energy, tension in boundaries:
        if start <= bars and start <= end:
            structure.append(
                {
                    "section": name,
                    "start_bar": start,
                    "end_bar": min(end, bars),
                    "energy": energy,
                    "tension": tension,
                    "repeat_group": f"{name}_theme",
                    "transition_to_next": "build" if name in {"intro", "verse"} else "release",
                }
            )
    progression = ["C:maj", "G:maj", "A:min", "F:maj"]
    funcs = ["I", "V", "vi", "IV"]
    syntax = []
    constraints = []
    for bar in range(1, bars + 1):
        sec = next(s for s in structure if s["start_bar"] <= bar <= s["end_bar"])
        density = float(sec["energy"])
        active = ["lead", "bass", "harmony_pad"] if density < 0.7 else list(roles)
        if sec["section"] == "intro":
            active = ["lead", "harmony_pad"]
        syntax.append(
            {
                "bar_id": bar,
                "chord": progression[(bar - 1) % len(progression)],
                "harmonic_func": funcs[(bar - 1) % len(funcs)],
                "syncopation": 0.25 + 0.25 * density,
                "groove_type": "pop_drive" if density >= 0.6 else "straight",
                "thickness": density,
                "cadence_strength": 0.8 if bar % 4 == 0 else 0.2,
                "density": density,
                "active_roles": active,
            }
        )
        for role in roles:
            constraints.append(
                {
                    "bar_id": bar,
                    "track_role": role,
                    "editable": True,
                    "target_density": density if role in active else 0.1,
                    "target_activity": 1.0 if role in active else 0.0,
                    "target_register": _default_register(role),
                    "boundary_smoothness": 0.6,
                }
            )
    plan = {
        "structure": structure,
        "syntax": syntax,
        "constraints": constraints,
        "global_constraints": default_global_constraints(),
    }
    validate_plan(plan, roles=roles)
    return plan


def _default_register(role: str) -> str:
    return {
        "bass": "low",
        "lead": "mid_high",
        "harmony_pad": "mid",
        "accompaniment": "mid",
        "rhythm": "mid",
        "drums": "wide",
    }.get(role, "mid")


def plan_bars(plan: dict[str, Any]) -> int:
    """Return the number of bars covered by a plan."""

    return max(int(sec["end_bar"]) for sec in plan["structure"])


def plan_to_bar_maps(plan: dict[str, Any], roles: tuple[str, ...] = TRACK_ROLES) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    """Expand section/syntax/constraint lists to bar-indexed lookup tables."""

    validate_plan(plan, roles=roles)
    bars = plan_bars(plan)
    section_by_bar: list[dict[str, Any]] = [{} for _ in range(bars)]
    for sec in plan["structure"]:
        for bar in range(int(sec["start_bar"]), int(sec["end_bar"]) + 1):
            section_by_bar[bar - 1] = sec
    syntax_by_bar = sorted(copy.deepcopy(plan["syntax"]), key=lambda x: int(x["bar_id"]))
    constraint_by_bar_role = {
        (int(c["bar_id"]) - 1, str(c["track_role"])): copy.deepcopy(c)
        for c in plan["constraints"]
    }
    return section_by_bar, syntax_by_bar, constraint_by_bar_role
