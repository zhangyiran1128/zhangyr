"""Edit-mask construction from realization constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vhcmusic.planning.schema import TRACK_ROLES, validate_plan


@dataclass
class EditMask:
    """Binary edit mask over `(R, L)` role/bar coordinates."""

    mask: np.ndarray
    roles: tuple[str, ...] = TRACK_ROLES

    def __post_init__(self) -> None:
        self.mask = (np.asarray(self.mask) > 0).astype(np.float32)
        if self.mask.ndim != 2:
            raise ValueError("EditMask must be 2D `(R, L)`")

    @property
    def bars(self) -> int:
        return int(self.mask.shape[1])

    def to_tensor(self, device=None):
        import torch

        return torch.tensor(self.mask[None], dtype=torch.float32, device=device)


def mask_from_constraints(plan: dict[str, Any], roles: tuple[str, ...] = TRACK_ROLES, bars: int | None = None) -> EditMask:
    """Construct `M in {0,1}^{R x L}` from `constraint_plan.editable`."""

    validate_plan(plan, bars=bars, roles=roles)
    bars = bars or max(int(s["end_bar"]) for s in plan["structure"])
    mask = np.zeros((len(roles), bars), dtype=np.float32)
    for con in plan["constraints"]:
        role = con["track_role"]
        bar = int(con["bar_id"]) - 1
        if role in roles and 0 <= bar < bars and bool(con["editable"]):
            mask[roles.index(role), bar] = 1.0
    return EditMask(mask, roles)


def mask_from_request(
    bars: int,
    target_bars: tuple[int, int] | list[int] | None = None,
    target_role: str | None = None,
    roles: tuple[str, ...] = TRACK_ROLES,
) -> EditMask:
    """Create an edit mask from CLI-style target bars and role."""

    mask = np.zeros((len(roles), bars), dtype=np.float32)
    if target_bars is None:
        start, end = 1, bars
    elif len(target_bars) == 2:
        start, end = int(target_bars[0]), int(target_bars[1])
    else:
        selected = [int(b) for b in target_bars]
        for b in selected:
            if 1 <= b <= bars:
                if target_role:
                    mask[roles.index(target_role), b - 1] = 1.0
                else:
                    mask[:, b - 1] = 1.0
        return EditMask(mask, roles)
    role_indices = [roles.index(target_role)] if target_role else list(range(len(roles)))
    for role_idx in role_indices:
        mask[role_idx, max(0, start - 1) : min(bars, end)] = 1.0
    return EditMask(mask, roles)


def expand_mask_to_grid(mask: EditMask, subdivisions: int, pitches: int, feature_dim: int = 3) -> np.ndarray:
    """Expand `(R, L)` mask to event-grid shape `(R, L, S, P, F)`."""

    return mask.mask[:, :, None, None, None].repeat(subdivisions, axis=2).repeat(pitches, axis=3).repeat(feature_dim, axis=4)
