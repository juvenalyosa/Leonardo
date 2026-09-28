"""Scripted provider for tests and offline demos (no network)."""

from .base import Provider, TextDelta, ToolCall, TurnEnd


class ScriptedProvider(Provider):
    """
    Plays back a list of turns. Each turn is a list of items: a str (streamed as text),
    ("tool", name, args) for a tool call, or ("stop", reason). A turn without tool
    calls ends with "end_turn".
    """

    name = "scripted"

    def __init__(self, turns, model="scripted", system="", tools=None, effort="high"):
        super().__init__(model, system=system, tools=tools, effort=effort)
        self.turns = list(turns)
        self.transcript = []
        self._n = 0

    def add_user_text(self, text):
        self.transcript.append(("user", str(text)))

    def add_tool_results(self, results):
        self.transcript.append(("tool_results", [dict(r) for r in results]))

    def stream_turn(self, cancel=None):
        if not self.turns:
            yield TurnEnd("end_turn")
            return
        turn = self.turns.pop(0)
        if callable(turn):          # built from earlier tool results
            turn = turn()
        calls = 0
        for item in turn:
            if callable(cancel) and cancel():
                yield TurnEnd("cancelled")
                return
            if isinstance(item, str):
                yield TextDelta(item)
            elif item[0] == "tool":
                self._n += 1
                calls += 1
                yield ToolCall(f"call_{self._n}", item[1], dict(item[2] or {}))
            elif item[0] == "stop":
                yield TurnEnd(item[1])
                return
        self.transcript.append(("assistant", turn))
        yield TurnEnd("tool_use" if calls else "end_turn", {"input_tokens": 10, "output_tokens": 5})
