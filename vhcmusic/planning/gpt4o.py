"""Pinned GPT-4o backend for planning and repair."""

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODEL = "gpt-4o-2024-08-06"


def request_json(system: str, user: str | list, schema: dict, name: str) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set OPENAI_API_KEY for GPT-4o planning or supply a fixed plan")
    payload = {"model": MODEL, "temperature": 0.2, "top_p": 1.0, "max_completion_tokens": 16384,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "response_format": {"type": "json_schema",
                                   "json_schema": {"name": name, "strict": True, "schema": schema}}}
    request = Request("https://api.openai.com/v1/chat/completions",
                      data=json.dumps(payload).encode("utf-8"),
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=180) as stream:
            result = json.load(stream)
    except HTTPError as exc:
        raise RuntimeError(f"GPT-4o request failed with HTTP {exc.code}") from exc
    choice = result["choices"][0]
    if choice["finish_reason"] != "stop" or not choice["message"].get("content"):
        raise RuntimeError(f"GPT-4o returned no complete {name}")
    return json.loads(choice["message"]["content"])


def object_schema(fields: dict) -> dict:
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
