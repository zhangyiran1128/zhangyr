"""Agent-guided online verification and masked repair."""

import copy

import numpy as np

from vhcmusic.editing.edit_mask import EditMask, mask_from_constraints
from vhcmusic.editing.local_repair import agent_decision
from .verification_report import build_verification_report


def run_closed_loop(initial_grid, target_plan, repair_fn, *, section_analyzer, density_scale,
                    chord_analyzer=None, max_rounds=3, strength=0.5, decision_fn=agent_decision,
                    strategy="local", preservation_reference=None):
    if strategy not in {"local", "random", "global", "none"}:
        raise ValueError(f"Unknown repair strategy {strategy}")
    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")
    grid = initial_grid
    reference = initial_grid if preservation_reference is None else preservation_reference
    roles = grid.roles
    permitted = mask_from_constraints(target_plan, roles, grid.bars).mask.astype(bool)
    flags = target_plan["global_constraints"]
    reports = []
    if strategy == "none":
        return grid, reports
    for index in range(max_rounds + 1):
        report = build_verification_report(target_plan, grid, reference, permitted,
                                           section_analyzer=section_analyzer, chord_analyzer=chord_analyzer,
                                           density_scale=density_scale)
        report["round"] = index
        reports.append(report)
        if index == max_rounds:
            break
        if strategy == "local":
            decision = decision_fn(target_plan, report, roles, permitted, strength)
        else:
            alternate = "random_resample" if strategy == "random" else "global_resample"
            decision = {"action": "accept" if report["pass"] else alternate,
                        "editable": [], "strength": strength, "diagnosis": strategy}
        action = decision["action"]
        report["decision"] = decision
        if action == "accept":
            if not report["pass"]:
                raise ValueError("Repair agent requested acceptance before all Table A1 checks passed")
            break
        if report["pass"]:
            raise ValueError("Repair agent requested an edit after all Table A1 checks passed")
        strength = float(decision["strength"])
        if not 0 < strength <= 1:
            raise ValueError("Repair strength must be in (0, 1]")
        if action in {"global_resample", "random_resample"}:
            if strategy == "local" and (not permitted.all() or any(value for key, value in flags.items() if key.startswith("keep_"))):
                raise ValueError("Global resampling conflicts with the plan's protection constraints")
            edit = np.ones_like(permitted)
        elif action in {"local_rewrite", "adjust_edit_strength"}:
            edit = np.zeros_like(permitted)
            for item in decision["editable"]:
                r, l = roles.index(item["role"]), int(item["bar"]) - 1
                if not 0 <= l < grid.bars or not permitted[r, l]:
                    raise ValueError("Repair agent selected a protected bar or role")
                edit[r, l] = True
            if not edit.any():
                raise ValueError("Repair agent must select at least one editable bar and role")
        else:
            raise ValueError(f"Unknown repair action: {action}")
        round_plan = copy.deepcopy(target_plan)
        for item in round_plan["constraints"]:
            item["editable"] = bool(edit[roles.index(item["track_role"]), item["bar_id"] - 1])
        mask = EditMask(edit, roles)
        grid = repair_fn(grid, round_plan, mask, strength, action, index + 1)
    return grid, reports
