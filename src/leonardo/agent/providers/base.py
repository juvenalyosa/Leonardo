"""Provider-neutral streaming interface of the Leonardo agent."""

from dataclasses import dataclass, field


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: object          # dict, or the raw JSON string the model produced


@dataclass
class TurnEnd:
    stop_reason: str           # "end_turn", "tool_use", "max_tokens", "refusal", "cancelled", "error"
    usage: dict = field(default_factory=dict)   # {"input_tokens": int, "output_tokens": int}
    detail: str = ""


class ProviderError(RuntimeError):
    """Configuration or API error shown to the user (bad key, unknown model, network)."""


class Provider:
    """
    One conversation with one model. The provider keeps the transcript in its own API
    format; the orchestrator only sees the neutral events above.

    stream_turn() yields TextDelta/ThinkingDelta/ToolCall events and ends with exactly
    one TurnEnd. When the turn has tool calls, the caller answers all of them with one
    add_tool_results() before the next stream_turn().
    """

    name = "base"

    def __init__(self, model, system="", tools=None, effort="high"):
        self.model = str(model)
        self.system = str(system or "")
        self.tools = list(tools or [])
        self.effort = str(effort or "high")

    def set_tools(self, tools):
        self.tools = list(tools or [])

    def add_user_text(self, text):
        raise NotImplementedError

    def add_tool_results(self, results):
        """results: list of dicts {"id", "content" (str), "is_error" (bool)}."""
        raise NotImplementedError

    def stream_turn(self, cancel=None):
        raise NotImplementedError
