"""Stdlib ATIF export from canonical Context history, independent of execution."""

import json
import os
from pathlib import Path
import tempfile

from aworld._version import __version__


def write_json(path, value, *, default=None):
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".aworld-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, default=default, allow_nan=False)
            stream.write("\n")
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def build_trajectory(history, *, result, agent, model_name=None):
    steps, calls = [], {}
    for entry in history:
        if entry.kind == "input":
            steps.append({"step_id": len(steps) + 1, "source": "user", "message": entry.data,
                          "extra": {"run_id": entry.run_id}})
        elif entry.kind == "assistant":
            step = {"step_id": len(steps) + 1, "source": "agent", "message": entry.data["content"],
                    "llm_call_count": 1, "extra": {"run_id": entry.run_id}}
            if entry.data["tool_calls"]:
                step["tool_calls"] = [{"tool_call_id": call["id"], "function_name": call["name"],
                                      "arguments": call["arguments"]} for call in entry.data["tool_calls"]]
                for call in step["tool_calls"]:
                    calls[(entry.run_id, call["tool_call_id"])] = step
            steps.append(step)
        elif entry.kind == "tool.result":
            data = entry.data
            step = calls[(entry.run_id, data["tool_call_id"])]
            observation = step.setdefault("observation", {"results": []})
            observation["results"].append({"source_call_id": data["tool_call_id"],
                "content": json.dumps(data["content"], ensure_ascii=False, allow_nan=False),
                "extra": {"is_error": data["is_error"]}})
    return {"schema_version": "ATIF-v1.7", "session_id": result.session_id,
            "trajectory_id": result.run_id,
            "agent": {"name": "aworld", "version": __version__, "model_name": model_name,
                      "extra": {"skills": [{"name": skill.name, "location": skill.location} for skill in agent.skills],
                                "tools": [tool.name for tool in agent.tools]}},
            "steps": steps, "extra": {"run_id": result.run_id, "status": result.status.value,
                                      "stop_reason": result.stop_reason.value,
                                      "history_scope": "session", "token_usage": "unavailable"}}
