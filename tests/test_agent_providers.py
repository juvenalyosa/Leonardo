"""Provider adapters with fake SDK clients (no network) and the external MCP endpoint."""
import asyncio
import socket
import time
from types import SimpleNamespace as NS

import pytest

from leonardo.agent.providers.anthropic_provider import AnthropicProvider
from leonardo.agent.providers.base import TextDelta, ThinkingDelta, ToolCall, TurnEnd
from leonardo.agent.providers.openai_provider import OpenAIProvider


class _Stream:
    def __init__(self, events, final):
        self.events, self.final = events, final

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final

    def get_final_response(self):
        return self.final


class _AnthropicClient:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        client = self

        class _Messages:
            def stream(self, **kw):
                client.calls.append(kw)
                return client.turns.pop(0)
        self.messages = _Messages()
        self.beta = NS(messages=_Messages())


def _delta(kind, text):
    field = "text" if kind == "text_delta" else "thinking"
    return NS(type="content_block_delta", delta=NS(type=kind, **{field: text}))


def test_anthropic_turns_tool_results_and_refusal():
    tool_block = NS(type="tool_use", id="tu_1", name="find_pockets", input={"object": "3PTB"})
    turn1 = _Stream([_delta("thinking_delta", "plan"), _delta("text_delta", "Looking")],
                    NS(content=[NS(type="text", text="Looking"), tool_block], stop_reason="tool_use",
                       usage=NS(input_tokens=100, output_tokens=20)))
    turn2 = _Stream([], NS(content=[], stop_reason="refusal", usage=NS(input_tokens=5, output_tokens=0),
                           stop_details=NS(explanation="policy")))
    client = _AnthropicClient([turn1, turn2])
    prov = AnthropicProvider("claude-sonnet-5", api_key="x", system="S", tools=[{"name": "t"}], client=client)
    prov.add_user_text("hi")
    events = list(prov.stream_turn())
    assert isinstance(events[0], ThinkingDelta) and isinstance(events[1], TextDelta)
    assert isinstance(events[2], ToolCall) and events[2].arguments == {"object": "3PTB"}
    assert isinstance(events[3], TurnEnd) and events[3].stop_reason == "tool_use"
    assert client.calls[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert client.calls[0]["output_config"] == {"effort": "high"}
    prov.add_tool_results([{"id": "tu_1", "content": "{}", "is_error": False}])
    assert prov.messages[1]["role"] == "assistant" and prov.messages[2]["content"][0]["tool_use_id"] == "tu_1"
    end = list(prov.stream_turn())[-1]
    assert end.stop_reason == "refusal" and end.detail == "policy"


def test_anthropic_haiku_has_no_adaptive_thinking_and_opus_uses_fallbacks():
    final = NS(content=[NS(type="text", text="ok")], stop_reason="end_turn", usage=NS(input_tokens=1, output_tokens=1))
    client = _AnthropicClient([_Stream([], final), _Stream([], final)])
    haiku = AnthropicProvider("claude-haiku-4-5", api_key="x", client=client)
    haiku.add_user_text("x")
    list(haiku.stream_turn())
    assert "thinking" not in client.calls[0] and "output_config" not in client.calls[0]
    opus = AnthropicProvider("claude-opus-5", api_key="x", client=client)
    opus.add_user_text("x")
    list(opus.stream_turn())
    assert client.calls[1]["fallbacks"] == "default" and client.calls[1]["betas"]


def test_anthropic_truncated_tool_call_is_not_run():
    final = NS(content=[NS(type="tool_use", id="t", name="run_md", input={"obj": "x"})], stop_reason="max_tokens",
               usage=NS(input_tokens=1, output_tokens=1))
    prov = AnthropicProvider("claude-sonnet-5", api_key="x", client=_AnthropicClient([_Stream([], final)]))
    prov.add_user_text("x")
    events = list(prov.stream_turn())
    assert not any(isinstance(e, ToolCall) for e in events) and events[-1].stop_reason == "max_tokens"


class _OpenAIClient:
    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = []
        client = self

        class _Responses:
            def stream(self, **kw):
                client.calls.append(kw)
                return client.turns.pop(0)
        self.responses = _Responses()


def test_openai_function_calls_and_previous_response_id():
    call = NS(type="function_call", call_id="call_1", name="find_pockets", arguments='{"object": "3PTB"}')
    turn1 = _Stream([NS(type="response.output_text.delta", delta="Hi")],
                    NS(id="resp_1", output=[call], status="completed", usage=NS(input_tokens=7, output_tokens=3)))
    turn2 = _Stream([], NS(id="resp_2", output=[], status="completed", usage=NS(input_tokens=1, output_tokens=1)))
    client = _OpenAIClient([turn1, turn2])
    prov = OpenAIProvider("gpt-5", api_key="x", system="S", tools=[], client=client)
    prov.add_user_text("hi")
    events = list(prov.stream_turn())
    assert isinstance(events[0], TextDelta) and events[1].arguments == {"object": "3PTB"}
    assert client.calls[0]["reasoning"]["effort"] == "high"
    prov.add_tool_results([{"id": "call_1", "content": "{}", "is_error": True}])
    list(prov.stream_turn())
    assert client.calls[1]["previous_response_id"] == "resp_1"
    assert client.calls[1]["input"] == [{"type": "function_call_output", "call_id": "call_1", "output": "ERROR: {}"}]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_external_mcp_http_endpoint():
    pytest.importorskip("uvicorn")
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    from leonardo.agent.mcp_server import ExternalMcpServer, build_server
    from leonardo.agent.registry import ToolRegistry

    reg = ToolRegistry()

    @reg.tool("ping", "Ping.", {"properties": {"x": {"type": "integer"}}}, permission="read")
    def ping(ctx, x=1):
        return {"pong": x}

    @reg.tool("heavy", "Heavy.", permission="compute_heavy")
    def heavy(ctx):
        return {"ran": True}

    ext = ExternalMcpServer(build_server(reg, run_on_main=lambda fn: fn(), confirm=lambda s, a: False), port=_free_port())
    url = ext.start()
    time.sleep(1.0)

    async def run():
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = [t.name for t in (await session.list_tools()).tools]
                ok = await session.call_tool("ping", {"x": 3})
                denied = await session.call_tool("heavy", {})
                return names, ok, denied
    try:
        names, ok, denied = asyncio.run(run())
    finally:
        ext.stop()
    assert set(names) == {"ping", "heavy"}
    assert not ok.isError and '"pong": 3' in ok.content[0].text
    assert denied.isError and "declined" in denied.content[0].text


def test_external_mcp_server_starts_inside_pymol_and_reports_a_busy_port(monkeypatch):
    """PyMOL redirects sys.stdout to an object without isatty(): uvicorn's default logging config failed
    ("Unable to configure formatter 'default'") and Leonardo reported it as "port in use"."""
    pytest.importorskip("uvicorn")
    import sys

    from leonardo.agent.mcp_server import ExternalMcpServer, build_server
    from leonardo.agent.registry import ToolRegistry

    class PymolStdout:
        def write(self, text):
            return len(text)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", PymolStdout())
    monkeypatch.setattr(sys, "stderr", PymolStdout())
    reg = ToolRegistry()
    port = _free_port()
    first = ExternalMcpServer(build_server(reg, run_on_main=lambda fn: fn()), port=port)
    assert first.start().endswith(f":{port}/mcp")
    try:
        second = ExternalMcpServer(build_server(reg, run_on_main=lambda fn: fn()), port=port)
        with pytest.raises(RuntimeError, match="in use by another program"):
            second.start()
    finally:
        first.stop()
    # stop() waits for the port: a restart on the same port works right away (changing Settings)
    again = ExternalMcpServer(build_server(reg, run_on_main=lambda fn: fn()), port=port)
    try:
        assert again.start().endswith(f":{port}/mcp")
    finally:
        again.stop()


def test_unanswered_tool_calls_are_closed_before_the_next_message():
    from types import SimpleNamespace

    prov = AnthropicProvider("claude-sonnet-5", api_key="x", client=object())
    prov.add_user_text("hi")
    prov.messages.append({"role": "assistant", "content": [SimpleNamespace(type="tool_use", id="tu_1")]})
    prov.add_user_text("again")
    last = prov.messages[-1]["content"]
    assert last[0]["tool_use_id"] == "tu_1" and last[0]["is_error"] and last[-1]["text"] == "again"

    oai = OpenAIProvider("gpt-5", api_key="x", client=object())
    oai._open_calls = ["call_1"]
    oai.add_user_text("again")
    assert oai.pending_input[0] == {"type": "function_call_output", "call_id": "call_1",
                                    "output": "ERROR: not run (the turn was interrupted)"}
