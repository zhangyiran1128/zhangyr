"""Schema-constrained hierarchical planning with fixed GPT-4o parameters."""

import base64
import json
from pathlib import Path

from .gpt4o import object_schema, request_json
from .prompt_templates import AGENT_SYSTEM_PROMPT, build_planning_prompt
from .schema import GROOVE_TYPES, HARMONIC_FUNCS, REGISTERS, SECTION_LABELS, TRACK_ROLES, TRANSITIONS
from .schema import default_global_constraints, validate_plan


def planning_schema(roles):
    integer, number = {"type": "integer"}, {"type": "number"}
    boolean, string = {"type": "boolean"}, {"type": "string"}
    enum = lambda values: {"type": "string", "enum": list(values)}
    array = lambda item: {"type": "array", "items": item}
    local = object_schema({"bar_id": integer, "track_role": enum(roles), "editable": boolean,
                           "target_density": {"type": ["number", "null"]},
                           "target_activity": {"type": ["number", "null"]},
                           "target_register": {"anyOf": [enum(REGISTERS), {"type": "null"}]},
                           "boundary_smoothness": number})
    return object_schema({
        "structure_planning": array(object_schema({
            "section": enum(SECTION_LABELS), "start_bar": integer, "end_bar": integer, "energy": number,
            "tension": number, "repeat_group": {"type": ["string", "null"]}, "transition_to_next": enum(TRANSITIONS)})),
        "musical_grammar_planning": array(object_schema({
            "bar_id": integer, "chord": string, "harmonic_func": enum(HARMONIC_FUNCS), "syncopation": number,
            "groove_type": enum(GROOVE_TYPES), "thickness": number, "cadence_strength": number,
            "density": number, "active_roles": array(enum(roles))})),
        "realization_constraints": object_schema({
            "local_constraints": array(local),
            "global_constraints": object_schema({name: boolean for name in default_global_constraints()})}),
    })


def normalize_plan(plan):
    if "structure_planning" not in plan:
        return plan
    constraints = plan["realization_constraints"]
    return {"structure": plan["structure_planning"], "syntax": plan["musical_grammar_planning"],
            "constraints": constraints["local_constraints"], "global_constraints": constraints["global_constraints"]}


class MusicPlanner:
    def __init__(self, agent_fn=None, roles=TRACK_ROLES):
        self.agent_fn = agent_fn
        self.roles = tuple(roles)

    def plan(self, prompt, bars=32, video_description=None):
        data = prompt if isinstance(prompt, dict) else {"prompt": prompt}
        description = video_description or data.get("video_description") or ""
        if data.get("motion_curve"):
            description += "\nMotion intensity curve: " + json.dumps(data["motion_curve"])
        if data.get("duration_seconds"):
            description += f"\nTarget duration in seconds: {data['duration_seconds']}"
        text = build_planning_prompt(data.get("prompt", ""), bars, list(self.roles), description)
        if data.get("tempo") or data.get("key"):
            text += "\nTempo and key requirements: " + json.dumps({k: data[k] for k in ("tempo", "key") if k in data})
        user = text
        if data.get("keyframes"):
            user = [{"type": "text", "text": text}]
            for frame in data["keyframes"]:
                if isinstance(frame, dict):
                    user.append({"type": "text", "text": f"Frame at {frame['seconds']} seconds"})
                    frame = frame["path"]
                frame = Path(frame)
                mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                        ".webp": "image/webp"}[frame.suffix.lower()]
                encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
                user.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
        if self.agent_fn:
            result = self.agent_fn(AGENT_SYSTEM_PROMPT, user)
            result = json.loads(result) if isinstance(result, str) else result
        else:
            result = request_json(AGENT_SYSTEM_PROMPT, user, planning_schema(self.roles), "hierarchical_music_plan")
        plan = normalize_plan(result)
        if validate_plan(plan, bars=bars, roles=self.roles).bars != bars:
            raise ValueError("GPT-4o plan must cover the requested bars")
        for name in ("tempo", "key", "duration_seconds"):
            if name in data:
                plan[name] = data[name]
        return plan
