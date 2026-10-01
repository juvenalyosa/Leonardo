"""OpenAI (Responses API) provider for the Leonardo agent."""

import json

from .base import Provider, ProviderError, TextDelta, ThinkingDelta, ToolCall, TurnEnd


def _is_reasoning_model(model):
    m = str(model).lower()
    return m.startswith(("o1", "o3", "o4", "gpt-5")) or "reason" in m


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model, api_key, system="", tools=None, effort="high", max_output_tokens=32000, client=None):
        super().__init__(model, system=system, tools=tools, effort=effort)
        self.max_output_tokens = int(max_output_tokens)
        self.pending_input = []          # items sent with the next request
        self.previous_response_id = None  # conversation state kept by the Responses API
        self._open_calls = []             # function calls of the last response still waiting for output
        if client is not None:
            self.client = client
        else:
            try:
                import openai
            except ImportError as exc:
                raise ProviderError("The 'openai' package is not installed in PyMOL's Python.") from exc
            self.client = openai.OpenAI(api_key=api_key or None, max_retries=3)

    def add_user_text(self, text):
        for call_id in self._open_calls:
            self.pending_input.append({"type": "function_call_output", "call_id": call_id,
                                       "output": "ERROR: not run (the turn was interrupted)"})
        self._open_calls = []
        self.pending_input.append({"role": "user", "content": str(text)})

    def add_tool_results(self, results):
        self._open_calls = []
        for r in results:
            out = str(r.get("content", ""))
            if r.get("is_error"):
                out = "ERROR: " + out
            self.pending_input.append({"type": "function_call_output", "call_id": r["id"], "output": out})

    def _request_kwargs(self):
        kw = {
            "model": self.model,
            "instructions": self.system,
            "input": list(self.pending_input),
            "tools": self.tools,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.previous_response_id:
            kw["previous_response_id"] = self.previous_response_id
        if _is_reasoning_model(self.model):
            effort = {"xhigh": "high", "max": "high"}.get(self.effort, self.effort)
            kw["reasoning"] = {"effort": effort, "summary": "auto"}
        return kw

    def stream_turn(self, cancel=None):
        import openai

        ended = None                      # a response.incomplete / response.failed event
        try:
            with self.client.responses.stream(**self._request_kwargs()) as stream:
                for event in stream:
                    if callable(cancel) and cancel():
                        yield TurnEnd("cancelled")
                        return
                    etype = getattr(event, "type", "")
                    if etype == "response.output_text.delta":
                        yield TextDelta(event.delta)
                    elif etype == "response.reasoning_summary_text.delta":
                        yield ThinkingDelta(event.delta)
                    elif etype in ("response.incomplete", "response.failed"):
                        ended = event.response
                try:
                    response = stream.get_final_response()
                except RuntimeError:
                    # the SDK only returns completed responses; incomplete/failed ones are reported here
                    if ended is None:
                        raise ProviderError("OpenAI ended the reply without a result; retry.") from None
                    response = ended
        except openai.AuthenticationError as exc:
            raise ProviderError("OpenAI rejected the API key (check it in AI Assistant settings).") from exc
        except openai.NotFoundError as exc:
            raise ProviderError(f"Model {self.model!r} is not available for this API key.") from exc
        except openai.RateLimitError as exc:
            raise ProviderError("OpenAI rate limit reached; wait a moment and retry.") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:
            raise ProviderError("Cannot reach the OpenAI API (network).") from exc

        if getattr(response, "status", "") == "failed":
            err = getattr(response, "error", None)
            raise ProviderError(f"OpenAI could not complete the reply: {getattr(err, 'message', '') or 'failed'}")
        self.pending_input = []
        self.previous_response_id = response.id
        usage_obj = getattr(response, "usage", None)
        usage = {"input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
                 "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0)}
        calls = [item for item in (response.output or []) if getattr(item, "type", "") == "function_call"]
        refusals = [c for item in (response.output or []) for c in (getattr(item, "content", None) or [])
                    if getattr(c, "type", "") == "refusal"]
        if refusals and not calls:
            yield TurnEnd("refusal", usage, str(getattr(refusals[0], "refusal", "") or "declined by the model"))
            return
        self._open_calls = [item.call_id for item in calls]
        if getattr(response, "status", "") == "incomplete" and calls:
            yield TurnEnd("max_tokens", usage, "the reply was cut before the tool call was complete")
            return
        for item in calls:
            try:
                args = json.loads(item.arguments or "{}")
            except ValueError:
                args = item.arguments          # the registry reports invalid JSON back to the model
            yield ToolCall(item.call_id, item.name, args)
        if getattr(response, "status", "") == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "") or "incomplete"
            yield TurnEnd("max_tokens", usage, f"the reply was cut ({reason})")
            return
        yield TurnEnd("tool_use" if calls else "end_turn", usage)


def list_openai_models(api_key):
    """Chat/reasoning model ids available to this key, newest first."""
    import openai

    client = openai.OpenAI(api_key=api_key or None)
    models = list(client.models.list())
    models.sort(key=lambda m: getattr(m, "created", 0), reverse=True)
    keep = []
    for m in models:
        mid = str(m.id)
        low = mid.lower()
        if not low.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
            continue
        if any(x in low for x in ("audio", "realtime", "tts", "transcribe", "image", "search", "embedding")):
            continue
        keep.append(mid)
    return keep
