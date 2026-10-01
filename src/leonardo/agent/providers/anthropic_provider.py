"""Claude (Anthropic Messages API) provider for the Leonardo agent."""

from .base import Provider, ProviderError, TextDelta, ThinkingDelta, ToolCall, TurnEnd

# models that take adaptive thinking + effort; Haiku 4.5 uses neither
_NO_ADAPTIVE = ("claude-haiku-4-5",)
# server-side refusal fallback (routes a declined turn to another model inside the same call)
_FALLBACK_MODELS = ("claude-opus-5", "claude-fable-5-1", "claude-opus-5-5")
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model, api_key, system="", tools=None, effort="high", max_tokens=32000, client=None):
        super().__init__(model, system=system, tools=tools, effort=effort)
        self.max_tokens = int(max_tokens)
        self.messages = []
        if client is not None:
            self.client = client
        else:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError("The 'anthropic' package is not installed in PyMOL's Python.") from exc
            self.client = anthropic.Anthropic(api_key=api_key or None, max_retries=3)

    def add_user_text(self, text):
        # an assistant turn whose tool calls never got results (cut reply, crash) would make the API
        # reject every later request: answer them as errors in the same user message
        last = self.messages[-1] if self.messages else None
        open_ids = [getattr(b, "id", None) or (b.get("id") if isinstance(b, dict) else None)
                    for b in (last["content"] if last and last["role"] == "assistant"
                              and isinstance(last["content"], list) else [])
                    if (getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else "")) == "tool_use"]
        if open_ids:
            self.messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": i, "content": "not run (the turn was interrupted)",
                 "is_error": True} for i in open_ids] + [{"type": "text", "text": str(text)}]})
            return
        self.messages.append({"role": "user", "content": str(text)})

    def add_tool_results(self, results):
        # every tool_result of one assistant turn goes back in a single user message
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r["id"], "content": str(r.get("content", "")),
             "is_error": bool(r.get("is_error", False))}
            for r in results
        ]})

    def _request_kwargs(self):
        kw = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "tools": self.tools,
            "messages": self.messages,
        }
        if not self.model.startswith(_NO_ADAPTIVE):
            kw["thinking"] = {"type": "adaptive", "display": "summarized"}
            kw["output_config"] = {"effort": self.effort}
        return kw

    def _open_stream(self):
        kw = self._request_kwargs()
        if self.model.startswith(_FALLBACK_MODELS):
            return self.client.beta.messages.stream(betas=[_FALLBACK_BETA], fallbacks="default", **kw)
        return self.client.messages.stream(**kw)

    def stream_turn(self, cancel=None):
        import anthropic

        try:
            with self._open_stream() as stream:
                for event in stream:
                    if callable(cancel) and cancel():
                        yield TurnEnd("cancelled")
                        return
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield TextDelta(event.delta.text)
                        elif event.delta.type == "thinking_delta":
                            yield ThinkingDelta(event.delta.thinking)
                response = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise ProviderError("Anthropic rejected the API key (check it in AI Assistant settings).") from exc
        except anthropic.NotFoundError as exc:
            raise ProviderError(f"Model {self.model!r} is not available for this API key.") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError("Anthropic rate limit reached; wait a moment and retry.") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Cannot reach the Anthropic API (network).") from exc

        # the full content (thinking, fallback and tool_use blocks included) goes back unchanged
        self.messages.append({"role": "assistant", "content": response.content})
        usage = {"input_tokens": int(getattr(response.usage, "input_tokens", 0) or 0),
                 "output_tokens": int(getattr(response.usage, "output_tokens", 0) or 0)}
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            yield TurnEnd("refusal", usage, str(getattr(details, "explanation", "") or "declined by the model"))
            return
        tool_uses = [b for b in response.content if getattr(b, "type", "") == "tool_use"]
        if response.stop_reason == "model_context_window_exceeded":
            yield TurnEnd("max_tokens", usage, "the conversation no longer fits in the model's context; "
                                               "start a New chat")
            return
        if response.stop_reason == "max_tokens" and tool_uses:
            # a truncated tool input must not run
            yield TurnEnd("max_tokens", usage, "the reply was cut before the tool call was complete")
            return
        for block in tool_uses:
            yield ToolCall(block.id, block.name, block.input)
        yield TurnEnd("tool_use" if tool_uses else str(response.stop_reason or "end_turn"), usage)


def list_anthropic_models(api_key):
    """Model ids available to this key (newest first), via the Models API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or None)
    return [m.id for m in client.models.list(limit=100)]
