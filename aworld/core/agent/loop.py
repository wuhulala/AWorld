"""One model/tool loop. Output events observe execution; they never dispatch it."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Sequence

from aworld.core.context.simple import ContextEntry
from aworld.core.session.protocols import RunContext
from aworld.core.tool.function import Tool, ToolExecutionError
from aworld.core.tool.local import default_tools
from aworld.core.tool.registry import ToolRegistry
from .skill import Skill
from .messages import AssistantMessage, Model, ModelRequest, ToolCall, ToolResultMessage, UserMessage


class AgentLoopLimitError(RuntimeError):
    pass


def _messages(entries: tuple[ContextEntry, ...]):
    """Project facts into model messages; leave incomplete tool turns in history.

    A cancelled run may retain a declared call with no result. Omit that turn
    from the request view instead of deleting history or fabricating a result.
    """
    messages = []
    for entry in entries:
        if entry.kind == "input":
            if not isinstance(entry.data, str):
                raise TypeError("Agent history input must be text")
            messages.append(UserMessage(entry.data))
        elif entry.kind == "assistant":
            data = entry.data
            messages.append(AssistantMessage(data["content"], tuple(
                ToolCall(call["id"], call["name"], call["arguments"]) for call in data["tool_calls"]
            )))
        elif entry.kind == "tool.result" and isinstance(entry.data, dict) and "tool_call_id" in entry.data:
            data = entry.data
            messages.append(ToolResultMessage(data["tool_call_id"], data["name"], data["content"], data["is_error"]))
    view = []
    i = 0
    while i < len(messages):
        message = messages[i]
        i += 1
        if isinstance(message, AssistantMessage) and message.tool_calls:
            results = []
            while i < len(messages) and isinstance(messages[i], ToolResultMessage):
                results.append(messages[i])
                i += 1
            expected = {call.id: call.name for call in message.tool_calls}
            actual = {result.tool_call_id: result.name for result in results}
            if expected == actual and len(results) == len(message.tool_calls):
                view.extend([message, *results])
        elif not isinstance(message, ToolResultMessage):
            view.append(message)
    return tuple(view)


class Agent:
    """Configured execution capability; conversation and turn state stay in Context/Run."""

    def __init__(
        self, *, model: Model, tools: Sequence[Tool] | ToolRegistry | None = None,
        skills: Sequence[Skill] = (), system_prompt: str = "",
        max_turns: int = 20,
    ) -> None:
        if not callable(getattr(model, "complete", None)):
            raise TypeError("model must implement async complete()")
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be text")
        skills = tuple(skills)
        if not all(isinstance(skill, Skill) for skill in skills):
            raise TypeError("skills must contain Skill values")
        if len({skill.name for skill in skills}) != len(skills):
            raise ValueError("Skill names must be unique")
        registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(default_tools() if tools is None else tools)
        registry = registry.extend(tool for skill in skills for tool in skill.tools)
        contributions = [system_prompt]
        for skill in skills:
            contributions.append(f"Skill {skill.name}: {skill.description}")
            if skill.location:
                registry.get("read")
                contributions.append(f"Read {skill.location} when this skill applies; relative references are based on its directory.")
            if skill.instructions:
                contributions.append(skill.instructions)
        self._model = model
        self._tools = registry
        self._system_prompt = "\n\n".join(part for part in contributions if part)
        self._skills = skills
        self._max_turns = max_turns

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def skills(self) -> tuple[Skill, ...]:
        return self._skills

    def validate_input(self, input: object) -> None:
        if not isinstance(input, str) or not input.strip():
            raise ValueError("Agent input must be non-empty text")

    async def run(self, input: object, context: RunContext) -> str:
        self.validate_input(input)
        seen_calls = set()
        for turn in range(1, self._max_turns + 1):
            # Prepare again after every tool batch; no cached second message history.
            history = await context.get_context()
            for entry in history:
                if entry.kind == "assistant":
                    seen_calls.update(call["id"] for call in entry.data["tool_calls"])
            request = ModelRequest(
                self._system_prompt, _messages(history),
                self._tools.schemas(),
            )
            context.emit("model.started", {"turn": turn})
            response = await self._model.complete(deepcopy(request))
            if not isinstance(response, AssistantMessage):
                raise TypeError("Model must return AssistantMessage")
            response = deepcopy(response)
            ids = [call.id for call in response.tool_calls]
            if len(set(ids)) != len(ids) or seen_calls.intersection(ids):
                raise ValueError("Tool call IDs must be unique in context")
            seen_calls.update(ids)
            context.append("assistant", {
                "content": response.content,
                "tool_calls": [{"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                               for call in response.tool_calls],
            })
            if response.content:
                context.set_output(response.content)
            context.emit("model.finished", {"turn": turn, "content": response.content, "tool_calls": len(ids)})
            if not response.tool_calls:
                return response.content
            for call in response.tool_calls:
                context.emit("tool.started", {"id": call.id, "name": call.name, "arguments": dict(call.arguments)})
                try:
                    content = await self._tools.execute(call.name, call.arguments, context)
                    is_error = False
                except asyncio.CancelledError:
                    raise
                except ToolExecutionError as exc:
                    content = exc.result
                    is_error = True
                except Exception as exc:
                    content = {"error": type(exc).__name__, "message": str(exc)}
                    is_error = True
                result = {"tool_call_id": call.id, "name": call.name, "content": content, "is_error": is_error}
                context.append("tool.result", result)
                context.emit("tool.finished", result)
        raise AgentLoopLimitError(f"max_turns_exceeded: {self._max_turns} model turns")
