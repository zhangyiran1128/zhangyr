"""Map raw MIDI instruments to VHCMusic functional track roles."""

from __future__ import annotations

from dataclasses import dataclass


TRACK_ROLES = ("lead", "bass", "rhythm", "harmony_pad", "accompaniment", "drums")

GM_PROGRAMS = {
    "lead": 80,
    "bass": 33,
    "rhythm": 25,
    "harmony_pad": 89,
    "accompaniment": 1,
    "drums": 0,
}


@dataclass(frozen=True)
class TrackInfo:
    role: str
    program: int
    is_drum: bool = False
    name: str = ""


class TrackMapper:
    """Heuristic instrument-to-role mapper for heterogeneous MIDI datasets."""

    def __init__(self, roles: tuple[str, ...] = TRACK_ROLES):
        self.roles = roles

    def role_index(self, role: str) -> int:
        if role not in self.roles:
            raise KeyError(f"Unknown track role: {role}")
        return self.roles.index(role)

    def infer_role(self, name: str = "", program: int = 0, is_drum: bool = False) -> str:
        if is_drum:
            return "drums"
        lowered = (name or "").lower()
        if "bass" in lowered or 32 <= program <= 39:
            return "bass"
        if any(k in lowered for k in ["melody", "lead", "vocal", "soprano"]) or 72 <= program <= 87:
            return "lead"
        if any(k in lowered for k in ["drum", "perc", "beat"]):
            return "drums"
        if any(k in lowered for k in ["pad", "strings", "choir"]) or 88 <= program <= 95 or 40 <= program <= 52:
            return "harmony_pad"
        if any(k in lowered for k in ["guitar", "strum", "rhythm"]) or 24 <= program <= 31:
            return "rhythm"
        return "accompaniment"

    def gm_program_for_role(self, role: str) -> int:
        return GM_PROGRAMS.get(role, 0)
