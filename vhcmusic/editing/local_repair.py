"""Schema-constrained repair decisions based on online MIDI observations."""

import json

import numpy as np

from vhcmusic.planning.gpt4o import object_schema, request_json
from vhcmusic.planning.prompt_templates import REPAIR_DECISION_TEMPLATE
from vhcmusic.planning.schema import REPAIR_ACTIONS


def agent_decision(plan, observations, roles, allowed, strength):
    selection = object_schema({"bar": {"type": "integer"}, "role": {"type": "string", "enum": list(roles)}})
    schema = object_schema({"action": {"type": "string", "enum": list(REPAIR_ACTIONS)},
                            "editable": {"type": "array", "items": selection},
                            "strength": {"type": "number"}, "diagnosis": {"type": "string"}})
    user = {"target_plan": plan, "observations": observations, "current_strength": strength,
            "editable_positions": [{"bar": int(l) + 1, "role": roles[r]} for r, l in np.argwhere(allowed)]}
    return request_json(REPAIR_DECISION_TEMPLATE, json.dumps(user), schema, "music_repair")
