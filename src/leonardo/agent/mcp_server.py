"""
Leonardo's tools as an MCP server.

The same server object serves two clients:

- the in-app assistant, through an in-memory MCP client session (``McpToolClient``);
- optionally, external MCP clients (Claude Desktop, Claude Code, ...) over streamable
  HTTP on 127.0.0.1 (``ExternalMcpServer``). External calls have no approved plan, so
  the host's ``confirm`` callback is asked before any tool whose permission class needs
  a confirmation.

Tool handlers must run on the Qt main thread (PyMOL and the Leonardo dialog are not
thread-safe): the host passes ``run_on_main(fn) -> value``.
"""

import asyncio
import json
import threading

from .registry import APPROVAL_POLICIES, ToolError, result_to_text

SERVER_NAME = "leonardo"


def build_server(registry, run_on_main, context=None, confirm=None, approval_policy="heavy"):
    """Low-level MCP server exposing ``registry`` (explicit JSON schemas, no signature inference)."""
    import mcp.types as types
    from mcp.server.lowlevel import Server

    server = Server(SERVER_NAME)
    needs_confirmation = APPROVAL_POLICIES.get(approval_policy, APPROVAL_POLICIES["heavy"])

    @server.list_tools()
    async def _list_tools():
        return [types.Tool(name=s.name, description=s.description, inputSchema=s.parameters,
                           annotations=types.ToolAnnotations(title=s.title,
                                                             readOnlyHint=s.permission in ("read", "view"),
                                                             destructiveHint=s.permission == "destructive"))
                for s in registry.specs()]

    def _reply(payload, is_error):
        text = result_to_text(payload)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=bool(is_error))

    @server.call_tool(validate_input=False)      # the registry validates (and reports) arguments
    async def _call_tool(name, arguments):
        try:
            spec, args = registry.validate(name, arguments or {})
        except ToolError as exc:
            return _reply({"error": str(exc)}, True)
        if callable(confirm) and spec.permission in needs_confirmation:
            ok = await asyncio.to_thread(confirm, spec, args)
            if not ok:
                return _reply({"error": "declined by the user"}, True)
        try:
            if spec.main_thread:
                result = await asyncio.to_thread(run_on_main, lambda: registry.call(name, args, context=context))
            else:
                result = await asyncio.to_thread(registry.call, name, args, context)
        except ToolError as exc:
            return _reply({"error": str(exc)}, True)
        return _reply(result, False)

    return server


class _LoopThread:
    """An asyncio loop on a daemon thread, for the sync (Qt / worker) side to submit coroutines."""

    def __init__(self, name):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name=name, daemon=True)
        self.thread.start()

    def run(self, coro, timeout=None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)


class McpToolClient:
    """In-memory MCP client for the in-app assistant: call_tool() is synchronous."""

    def __init__(self, server):
        self._server = server
        self._lt = _LoopThread("leonardo-mcp-client")
        self._session = None
        self._ready = threading.Event()
        self._closing = None
        self._lt.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._main()))
        if not self._ready.wait(15):
            raise RuntimeError("the in-app MCP session did not start")

    async def _main(self):
        from mcp.shared.memory import create_connected_server_and_client_session

        self._closing = asyncio.Event()
        async with create_connected_server_and_client_session(self._server) as session:
            self._session = session
            self._ready.set()
            await self._closing.wait()

    def list_tool_names(self):
        res = self._lt.run(self._session.list_tools(), timeout=30)
        return [t.name for t in res.tools]

    def call_tool(self, name, arguments):
        """Returns the tool's JSON result as a dict; raises ToolError when the tool reports an error."""
        res = self._lt.run(self._session.call_tool(name, dict(arguments or {})))
        text = "".join(getattr(c, "text", "") for c in (res.content or []))
        try:
            data = json.loads(text) if text else {}
        except ValueError:
            data = {"result": text}
        if res.isError:
            raise ToolError(str(data.get("error", text)) if isinstance(data, dict) else text)
        return data if isinstance(data, dict) else {"result": data}

    def close(self):
        if self._closing is not None:
            self._lt.loop.call_soon_threadsafe(self._closing.set)
        self._lt.stop()


class ExternalMcpServer:
    """Streamable-HTTP MCP endpoint on 127.0.0.1 (off by default; Settings > MCP server)."""

    def __init__(self, server, port=8765):
        self.server = server
        self.port = int(port)
        self._uvicorn = None
        self._thread = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self):
        import contextlib

        import uvicorn
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.routing import Mount

        manager = StreamableHTTPSessionManager(app=self.server, stateless=True)

        async def handle(scope, receive, send):
            await manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(_app):
            async with manager.run():
                yield

        app = Starlette(routes=[Mount("/mcp", app=handle)], lifespan=lifespan)
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        self._uvicorn = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._uvicorn.run, name="leonardo-mcp-http", daemon=True)
        self._thread.start()
        return self.url

    def stop(self):
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        self._uvicorn = None
