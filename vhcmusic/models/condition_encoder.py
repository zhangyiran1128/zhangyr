"""Hierarchical plan encoder for conditional diffusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vhcmusic.representation.harmony import ROOT_NAMES, chord_identity
from vhcmusic.planning.schema import (
    GROOVE_TYPES,
    HARMONIC_FUNCS,
    REGISTERS,
    SECTION_LABELS,
    TRACK_ROLES,
    TRANSITIONS,
    default_global_constraints,
    plan_bars,
    plan_to_bar_maps,
)


ROOTS = (*ROOT_NAMES, 'N')
QUALITIES = ("maj", "min", "dim", "aug", "sus2", "sus4", "7", "maj7", "min7", "dim7", "hdim7", "none")


@dataclass
class ConditionBatch:
    section: torch.Tensor
    transition: torch.Tensor
    structure_cont: torch.Tensor
    chord_root: torch.Tensor
    chord_quality: torch.Tensor
    harmonic_func: torch.Tensor
    groove: torch.Tensor
    syntax_cont: torch.Tensor
    active_roles: torch.Tensor
    target_register: torch.Tensor
    constraint_cont: torch.Tensor
    editable: torch.Tensor
    repetition: torch.Tensor
    flags: torch.Tensor


def parse_chord(label: str) -> tuple[int, int]:
    root, quality = chord_identity(label)
    return root % len(ROOTS), QUALITIES.index(quality)



class ConditionEncoder(nn.Module):
    """Encode structure, syntax, and realization constraints to `(B, R, L, D)`."""

    def __init__(self, dim: int = 512, roles: tuple[str, ...] = TRACK_ROLES, mode: str = 'full'):
        super().__init__()
        self.dim = dim
        self.roles = roles
        if mode not in {"full", "without_structure", "without_grammar", "without_constraints", "flat"}:
            raise ValueError(f"Unknown condition mode: {mode}")
        self.mode = mode
        self.section_emb = nn.Embedding(len(SECTION_LABELS), dim)
        self.transition_emb = nn.Embedding(len(TRANSITIONS), dim)
        self.repeat_emb = nn.Embedding(65, dim)
        self.root_emb = nn.Embedding(len(ROOTS), dim)
        self.quality_emb = nn.Embedding(len(QUALITIES), dim)
        self.func_emb = nn.Embedding(len(HARMONIC_FUNCS), dim)
        self.groove_emb = nn.Embedding(len(GROOVE_TYPES), dim)
        self.register_emb = nn.Embedding(len(REGISTERS) + 1, dim)
        self.structure_proj = nn.Linear(2, dim)
        self.syntax_proj = nn.Linear(5, dim)
        self.constraint_proj = nn.Linear(6, dim)
        self.editable_emb = nn.Embedding(2, dim)
        self.flags_emb = nn.Embedding(2 ** len(default_global_constraints()), dim)
        self.null = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(
        self,
        plans: list[dict[str, Any]],
        bars: int | None = None,
        device: torch.device | str | None = None,
        drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = self.tensorize(plans, bars=bars, device=device)
        structure = self.section_emb(batch.section) + self.transition_emb(batch.transition)
        structure = structure + self.repeat_emb(batch.repetition) + self.structure_proj(batch.structure_cont)
        syntax = self.root_emb(batch.chord_root) + self.quality_emb(batch.chord_quality)
        syntax = syntax + self.func_emb(batch.harmonic_func) + self.groove_emb(batch.groove)
        syntax_values = batch.syntax_cont[:, None].expand(-1, len(self.roles), -1, -1)
        constraints = self.register_emb(batch.target_register) + self.editable_emb(batch.editable.long())
        constraints = constraints + self.constraint_proj(batch.constraint_cont)
        grammar = syntax[:, None] + self.syntax_proj(torch.cat((syntax_values, batch.active_roles[..., None]), dim=-1))
        realization = constraints + self.flags_emb(batch.flags)[:, None, None]
        cond = ((self.mode != "without_structure") * structure[:, None]
                + (self.mode != "without_grammar") * grammar
                + (self.mode != "without_constraints") * realization)
        if self.mode == "flat":
            weights = cond.new_zeros((len(plans), 1, cond.shape[2], 1))
            for index, plan in enumerate(plans):
                weights[index, :, :plan_bars(plan)] = 1
            cond = ((cond * weights).mean(dim=1, keepdim=True).sum(dim=2, keepdim=True)
                    / weights.sum(dim=2, keepdim=True))
            cond = cond.expand(-1, len(self.roles), batch.section.shape[1], -1)
        if drop_mask is not None:
            mask = drop_mask.to(cond.device).view(-1, 1, 1, 1).float()
            cond = cond * (1.0 - mask) + self.null * mask
        return cond

    def tensorize(self, plans: list[dict[str, Any]], bars: int | None = None, device: torch.device | str | None = None) -> ConditionBatch:
        bars = bars or max(plan_bars(p) for p in plans)
        b = len(plans)
        r = len(self.roles)
        section = torch.zeros(b, bars, dtype=torch.long)
        transition = torch.zeros(b, bars, dtype=torch.long)
        structure_cont = torch.zeros(b, bars, 2, dtype=torch.float32)
        chord_root = torch.zeros(b, bars, dtype=torch.long)
        chord_quality = torch.zeros(b, bars, dtype=torch.long)
        harmonic_func = torch.zeros(b, bars, dtype=torch.long)
        groove = torch.zeros(b, bars, dtype=torch.long)
        syntax_cont = torch.zeros(b, bars, 4, dtype=torch.float32)
        active_roles = torch.zeros(b, r, bars, dtype=torch.float32)
        target_register = torch.zeros(b, r, bars, dtype=torch.long)
        constraint_cont = torch.zeros(b, r, bars, 6, dtype=torch.float32)
        editable = torch.zeros(b, r, bars, dtype=torch.float32)

        repetition = torch.zeros(b, bars, dtype=torch.long)
        flags = torch.zeros(b, dtype=torch.long)
        for i, plan in enumerate(plans):
            groups = {}
            flags[i] = sum(int(plan["global_constraints"][name]) << bit
                           for bit, name in enumerate(default_global_constraints()))
            sec_by_bar, syn_by_bar, con_by_bar_role = plan_to_bar_maps(plan, roles=self.roles)
            for l in range(min(bars, len(sec_by_bar))):
                sec = sec_by_bar[l]
                group = sec["repeat_group"]
                if group:
                    repetition[i, l] = groups.setdefault(group, len(groups) + 1)
                section[i, l] = SECTION_LABELS.index(sec["section"])
                transition[i, l] = TRANSITIONS.index(sec["transition_to_next"])
                structure_cont[i, l] = torch.tensor([float(sec["energy"]), float(sec["tension"])])
            for l in range(min(bars, len(syn_by_bar))):
                syn = syn_by_bar[l]
                root_id, quality_id = parse_chord(str(syn["chord"]))
                chord_root[i, l] = root_id
                chord_quality[i, l] = quality_id
                harmonic_func[i, l] = HARMONIC_FUNCS.index(syn["harmonic_func"])
                groove[i, l] = GROOVE_TYPES.index(syn["groove_type"])
                syntax_cont[i, l] = torch.tensor(
                    [float(syn["syncopation"]), float(syn["thickness"]), float(syn["cadence_strength"]), float(syn["density"])],
                )
                for role in syn["active_roles"]:
                    if role in self.roles:
                        active_roles[i, self.roles.index(role), l] = 1.0
            for l in range(bars):
                for role_idx, role in enumerate(self.roles):
                    con = con_by_bar_role.get((l, role))
                    if con is None:
                        con = {
                            "editable": True,
                            "target_density": 0.5,
                            "target_activity": active_roles[i, role_idx, l].item(),
                            "target_register": "mid",
                            "boundary_smoothness": 0.5,
                        }
                    editable[i, role_idx, l] = 1.0 if con["editable"] else 0.0
                    target_register[i, role_idx, l] = REGISTERS.index(con["target_register"]) if con["target_register"] is not None else len(REGISTERS)
                    constraint_cont[i, role_idx, l] = torch.tensor(
                        [float(con["target_density"] or 0), float(con["target_activity"] or 0),
                         float(con["boundary_smoothness"]), float(con["target_density"] is not None),
                         float(con["target_activity"] is not None), float(con["target_register"] is not None)],
                        )
        tensors = (section, transition, structure_cont, chord_root, chord_quality, harmonic_func, groove,
                   syntax_cont, active_roles, target_register, constraint_cont, editable, repetition, flags)
        return ConditionBatch(*(tensor.to(device) for tensor in tensors))
