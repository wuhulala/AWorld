"""Provider serialization and a real local HTTP model/tool/model roundtrip."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from aworld.core.agent import Agent
from aworld.core.agent.messages import AssistantMessage, ModelRequest, ToolCall, ToolResultMessage, UserMessage
from aworld.core.session import RunStatus, create_session
from aworld.core.tool import Tool
from aworld.models.chat_completions import ChatCompletionsModel, parse_response, request_payload


def payload(message, reason="stop"):
    return {"choices": [{"message": {"role": "assistant", **message}, "finish_reason": reason}]}


def test_message_roles_tool_ids_json_and_empty_capability_set():
    request = ModelRequest("instructions", (UserMessage("hello"), AssistantMessage("", (ToolCall("id", "sum", {"a": 2}),)), ToolResultMessage("id", "sum", {"value": 5}, True)), ())
    value = request_payload(request, "configured-model", "none")
    assert [item["role"] for item in value["messages"]] == ["system", "user", "assistant", "tool"]
    assert value["messages"][-1]["tool_call_id"] == "id"
    assert json.loads(value["messages"][-1]["content"])["is_error"] is True
    assert "tools" not in value
    assert value["reasoning_effort"] == "none"


@pytest.mark.parametrize("response", [
    payload({"content": "partial"}, "length"), payload({"content": "filtered"}, "content_filter"),
    payload({"content": "", "refusal": "refused"}), payload({"content": ""}),
    payload({"tool_calls": [{"id": "id", "type": "function", "function": {"name": "read", "arguments": "[]"}}]}, "tool_calls"),
    payload({"content": "answer"}, "tool_calls"), {"choices": []},
])
def test_invalid_or_incomplete_responses_cannot_trigger_success_or_tools(response):
    with pytest.raises((ValueError, TypeError)):
        parse_response(response)


def test_actual_http_tool_roundtrip_is_independent_of_legacy_chain():
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if body["messages"][-1]["role"] == "tool":
                result = payload({"content": "5"})
            else:
                result = payload({"content": None, "tool_calls": [{"id": "sum-1", "type": "function", "function": {"name": "sum", "arguments": '{"a":2,"b":3}'}}]}, "tool_calls")
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    async def run():
        model = ChatCompletionsModel(model="local-test", base_url=f"http://127.0.0.1:{server.server_port}/v1")
        async def sum_values(arguments, context):
            return arguments["a"] + arguments["b"]
        session = await create_session(agent=Agent(model=model, tools=[Tool("sum", "Add", {"type": "object"}, sum_values)]))
        try:
            result = await (await session.submit("add")).result()
            assert result.status == RunStatus.COMPLETED
            assert result.output == "5"
        finally:
            await session.close()
            await model.aclose()
    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert len(requests) == 2
    assert requests[0]["tools"][0]["function"]["name"] == "sum"
    assert requests[1]["messages"][-1]["tool_call_id"] == "sum-1"


def test_real_cli_http_model_reads_a_local_file(tmp_path):
    (tmp_path / "note").write_text("provider tool output")
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if body["messages"][-1]["role"] == "tool":
                tool = json.loads(body["messages"][-1]["content"])
                assert "provider tool output" in tool["result"]["content"]
                result = payload({"content": "verified"})
            else:
                result = payload({"tool_calls": [{"id": "read-note", "type": "function", "function": {
                    "name": "read", "arguments": '{"path":"note"}'}}]}, "tool_calls")
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = dict(os.environ)
    environment["AWORLD_API_KEY"] = "local-fixture-key"
    try:
        result = subprocess.run([sys.executable, "-m", "aworld", "run", "--model", "local-fixture", "--base-url",
            f"http://127.0.0.1:{server.server_port}/v1", "--cwd", str(tmp_path), "--task", "read note", "--json"],
            cwd=Path(__file__).resolve().parents[1], env=environment, capture_output=True, text=True, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["output"] == "verified"
    assert len(requests) == 2


def test_cli_ctrl_c_during_http_request_exits_and_cleans_up():
    started, release = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            started.set()
            release.wait(10)
            self.send_response(200)
            self.end_headers()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    process = subprocess.Popen([sys.executable, "-m", "aworld", "run", "--model", "local-fixture", "--base-url",
        f"http://127.0.0.1:{server.server_port}/v1", "--task", "wait"],
        cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert started.wait(5), "CLI did not reach the model endpoint"
        process.send_signal(signal.SIGINT)
        output, error = process.communicate(timeout=3)
        assert process.returncode == 130, error
        assert "Traceback" not in error
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
