"""Optional HTTP adapter for text/function-tool Chat Completions endpoints."""

from __future__ import annotations

import json
import math
from urllib.parse import urlsplit

from aworld.core.agent.messages import AssistantMessage, ModelRequest, ToolCall, ToolResultMessage


def request_payload(request: ModelRequest, model: str, reasoning_effort: str | None = None) -> dict:
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    for message in request.messages:
        if isinstance(message, AssistantMessage):
            value = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                value["tool_calls"] = [{"id": call.id, "type": "function", "function": {
                    "name": call.name, "arguments": json.dumps(dict(call.arguments), ensure_ascii=False, allow_nan=False),
                }} for call in message.tool_calls]
        elif isinstance(message, ToolResultMessage):
            value = {"role": "tool", "tool_call_id": message.tool_call_id,
                     "content": json.dumps({"result": message.content, "is_error": message.is_error}, ensure_ascii=False, allow_nan=False)}
        else:
            value = {"role": "user", "content": message.content}
        messages.append(value)
    payload = {"model": model, "messages": messages, "stream": False}
    if request.tools:
        payload["tools"] = [{"type": "function", "function": {"name": tool.name,
            "description": tool.description, "parameters": dict(tool.parameters)}} for tool in request.tools]
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return payload


def parse_response(payload: dict) -> AssistantMessage:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Expected one complete model choice")
    choice = choices[0]
    reason, message = choice.get("finish_reason"), choice.get("message")
    if reason not in ("stop", "tool_calls") or not isinstance(message, dict):
        raise ValueError("Model response is incomplete or unsupported")
    if message.get("role") != "assistant" or message.get("refusal"):
        raise ValueError("Model did not return an executable assistant response")
    calls = []
    for item in message.get("tool_calls") or []:
        if item.get("type") != "function":
            raise ValueError("Only function tools are supported")
        function = item.get("function", {})
        arguments = json.loads(function.get("arguments", ""))
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object")
        calls.append(ToolCall(item.get("id"), function.get("name"), arguments))
    if (reason == "tool_calls") != bool(calls):
        raise ValueError("Model finish reason does not match its tool calls")
    content = message.get("content")
    if content is None:
        content = ""
    if not content and not calls:
        raise ValueError("Model returned an empty final response")
    return AssistantMessage(content, tuple(calls))


class ChatCompletionsModel:
    """HTTPX is imported only on explicit provider construction; never installed."""
    def __init__(self, *, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, timeout: float = 60, reasoning_effort: str | None = None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Specify --model or AWORLD_MODEL")
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be an HTTP(S) base URL without credentials/query")
        if parsed.hostname == "api.openai.com" and not api_key:
            raise ValueError("Set AWORLD_API_KEY or OPENAI_API_KEY in the environment")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("request timeout must be positive finite seconds")
        try:
            import httpx
        except ImportError:
            raise RuntimeError('Live models need the optional dependency: pip install "aworld[llm]"') from None
        self._httpx = httpx
        self._model, self._reasoning_effort = model, reasoning_effort
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._client = httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    async def complete(self, request: ModelRequest) -> AssistantMessage:
        try:
            response = await self._client.post(self._url, json=request_payload(request, self._model, self._reasoning_effort))
        except self._httpx.HTTPError as exc:
            raise RuntimeError(f"Model transport failed: {type(exc).__name__}") from None
        if response.is_error:
            raise RuntimeError(f"Model request failed: HTTP {response.status_code}")
        return parse_response(response.json())

    async def aclose(self):
        await self._client.aclose()
