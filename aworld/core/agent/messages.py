"""Minimal model boundary: text, assistant tool calls and tool results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from aworld.core.tool.function import ToolSchema


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (self.id, self.name)):
            raise ValueError("Tool call id and name must be non-empty")
        object.__setattr__(self, "arguments", deepcopy(dict(self.arguments)))


@dataclass(frozen=True)
class UserMessage:
    content: str
    role: str = field(default="user", init=False)


@dataclass(frozen=True)
class AssistantMessage:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    role: str = field(default="assistant", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("Assistant content must be text")
        calls = tuple(self.tool_calls)
        if not all(isinstance(call, ToolCall) for call in calls):
            raise TypeError("tool_calls must contain ToolCall values")
        object.__setattr__(self, "tool_calls", deepcopy(calls))


@dataclass(frozen=True)
class ToolResultMessage:
    tool_call_id: str
    name: str
    content: object
    is_error: bool = False
    role: str = field(default="tool", init=False)


@dataclass(frozen=True)
class ModelRequest:
    system_prompt: str
    messages: tuple[UserMessage | AssistantMessage | ToolResultMessage, ...]
    tools: tuple[ToolSchema, ...]


class Model(Protocol):
    async def complete(self, request: ModelRequest) -> AssistantMessage: ...
