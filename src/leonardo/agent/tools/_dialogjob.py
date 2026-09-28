"""Start a Leonardo dialog action as an assistant job (headless modals, polled completion)."""

from . import _headless


def start_dialog_job(ctx, kind, start, is_done, collect, description=""):
    """
    start(): kicks off the dialog action (on the main thread);
    is_done() -> bool: the action finished (busy flag cleared, task gone, ...);
    collect() -> dict: result, or raises to report a failure.
    Captured message boxes are returned with the result (``messages``).
    """
    _headless.begin()
    try:
        start()
    except Exception:
        _headless.take_messages()
        _headless.end()
        raise

    finished = {"done": False}

    def poll():
        if finished["done"]:
            return {"done": True}
        if not is_done():
            return {"done": False, "progress": _progress(ctx)}
        finished["done"] = True
        messages = _headless.take_messages()
        _headless.end()
        errors = [m["text"] for m in messages if m["kind"] == "critical"]
        try:
            result = dict(collect() or {})
        except Exception as exc:
            return {"done": True, "error": "; ".join(errors) or f"{type(exc).__name__}: {exc}"}
        if messages:
            result["leonardo_messages"] = messages
        return {"done": True, "result": result}

    def cancel():
        if not finished["done"]:
            finished["done"] = True
            _headless.take_messages()
            _headless.end()

    return ctx.jobs.start(kind, poll, cancel=cancel, description=description)


def _progress(ctx):
    try:
        state = dict(getattr(ctx.dlg, "_task_monitor_state", {}) or {})
        if state:
            last = state[max(state)]
            return f"{last.get('status', '')} {last.get('message', '')}".strip()
    except Exception:
        pass
    return ""
