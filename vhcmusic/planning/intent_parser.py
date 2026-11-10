"""Lightweight intent parser for local/offline planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import TRACK_ROLES, make_fallback_plan


class IntentParser:
    """Parse prompt JSON or plain text into coarse planning hints.

    This parser is intentionally deterministic. It is not a replacement for an
    LLM planner; it provides a reproducible fallback and test target.
    """

    def parse(self, source: str | Path | dict[str, Any], bars: int = 32) -> dict[str, Any]:
        if isinstance(source, dict):
            prompt = str(source.get("prompt", ""))
            video = str(source.get("video_description", ""))
            bars = int(source.get("bars", bars))
        else:
            text = Path(source).read_text(encoding="utf-8") if Path(str(source)).exists() else str(source)
            try:
                payload = json.loads(text)
                return self.parse(payload, bars=bars)
            except json.JSONDecodeError:
                prompt, video = text, ""

        hints = {
            "prompt": prompt,
            "video_description": video,
            "bars": bars,
            "roles": list(TRACK_ROLES),
            "energy_bias": self._energy_bias(prompt + " " + video),
            "dense_drums": any(w in (prompt + " " + video).lower() for w in ["drum", "beat", "dense", "impact", "chorus"]),
        }
        return hints

    @staticmethod
    def _energy_bias(text: str) -> float:
        text = text.lower()
        if any(w in text for w in ["calm", "simple", "soft", "sparse"]):
            return -0.2
        if any(w in text for w in ["powerful", "impact", "full", "energetic", "chorus"]):
            return 0.2
        return 0.0


def parse_to_fallback_plan(source: str | Path | dict[str, Any], bars: int = 32) -> dict[str, Any]:
    parser = IntentParser()
    hints = parser.parse(source, bars=bars)
    plan = make_fallback_plan(bars=int(hints["bars"]))
    if hints["energy_bias"]:
        for sec in plan["structure"]:
            sec["energy"] = min(1.0, max(0.0, float(sec["energy"]) + hints["energy_bias"]))
            sec["tension"] = min(1.0, max(0.0, float(sec["tension"]) + hints["energy_bias"] * 0.5))
        for syn in plan["syntax"]:
            syn["density"] = min(1.0, max(0.0, float(syn["density"]) + hints["energy_bias"]))
            syn["thickness"] = syn["density"]
    return plan
