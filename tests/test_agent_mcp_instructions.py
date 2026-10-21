"""External MCP clients get Leonardo's guidance (modules, workflows such as cryptic site -> docking box), not only
the tool list."""
import asyncio

import pytest

pytest.importorskip("mcp")
from leonardo.agent.mcp_server import build_server, mcp_instructions  # noqa: E402
from leonardo.agent.registry import ToolRegistry  # noqa: E402


def test_instructions_have_the_science_without_the_in_app_plan_loop():
    text = mcp_instructions()
    for name in ("find_cryptic_pockets", "prepare_protein", "ramachandran_plot", "check_simulation_system",
                 "docking_box", "resolve_components"):
        assert name in text, name
    for name in ("propose_plan", "mark_step", "ask_user"):
        assert name not in text, name


def test_client_receives_the_instructions_at_initialization():
    from mcp.shared.memory import create_client_server_memory_streams
    from mcp import ClientSession

    server = build_server(ToolRegistry(), run_on_main=lambda f: f())

    async def run():
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async def serve():
                await server.run(server_streams[0], server_streams[1], server.create_initialization_options(),
                                 raise_exceptions=True)

            task = asyncio.create_task(serve())
            async with ClientSession(client_streams[0], client_streams[1]) as session:
                result = await session.initialize()
            task.cancel()
            return result

    result = asyncio.run(run())
    assert "find_cryptic_pockets" in (result.instructions or "")
