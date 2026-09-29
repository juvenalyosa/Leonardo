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
        warnings = [m["text"][:600] for m in messages if m["kind"] == "warning"]
        try:
            result = dict(collect() or {})
        except Exception as exc:
            import traceback

            traceback.print_exc()
            detail = str(exc) or "; ".join(errors)          # the tool's own diagnosis first
            extra = [e for e in errors if e and e not in detail]
            if extra:
                detail += " | Leonardo: " + " / ".join(extra[:2])
            if warnings:
                detail += " | Leonardo warnings: " + " / ".join(warnings[:3])
            return {"done": True, "error": detail}
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


class _Call:
    def __init__(self, args, kwargs):
        self.args, self.kwargs = tuple(args), dict(kwargs)

    def arg(self, index, name):
        if name in self.kwargs:
            return self.kwargs[name]
        return self.args[index] if index < len(self.args) else None


def start_finalize_job(ctx, kind, start, finalize_attr, collect, busy_button="", description=""):
    """
    For Leonardo modules that submit a background task and finish in ``dlg.<finalize_attr>(*args)``:
    the finalize call is captured (it still runs), and the job ends when it has run, when nothing was
    submitted, or when ``busy_button`` is enabled again (a failed task never reaches finalize).
    ``collect(call)`` gets the finalize call (None if it never ran; ``call.arg(i, name)`` reads an argument
    given by position or keyword) and builds the result.
    """
    dlg = ctx.dlg
    captured = {}
    original = getattr(dlg, finalize_attr)

    def finalize(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            captured["call"] = _Call(args, kwargs)
            setattr(dlg, finalize_attr, original)

    setattr(dlg, finalize_attr, finalize)

    def restore():
        if getattr(dlg, finalize_attr) is finalize:
            setattr(dlg, finalize_attr, original)

    def wrapped_start():
        before = set(getattr(dlg, "_task_monitor_state", {}) or {})
        try:
            start()
        except Exception:
            restore()
            raise
        after = set(getattr(dlg, "_task_monitor_state", {}) or {})
        captured["submitted"] = bool(after - before)

    def button_enabled():
        btn = getattr(dlg, busy_button, None) if busy_button else None
        try:
            return btn is None or bool(btn.isEnabled())
        except Exception:
            return True

    def is_done():
        if "call" in captured or not captured.get("submitted", True):
            return True
        return bool(busy_button) and button_enabled()

    def wrapped_collect():
        restore()
        return collect(captured.get("call"))

    return start_dialog_job(ctx, kind, wrapped_start, is_done, wrapped_collect, description=description)


def start_deferred_job(ctx, kind, run, collect, description=""):
    """
    For synchronous Leonardo handlers that take minutes (they pump the event loop themselves):
    ``run()`` is scheduled on the Qt main thread after the tool call returns, so the assistant can
    follow it as a job; ``collect(outcome)`` gets {"error": str|None} and builds the result.
    """
    from pymol.Qt import QtCore

    outcome = {"done": False, "error": None}

    def fire():
        try:
            run()
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            outcome["done"] = True

    def start():
        QtCore.QTimer.singleShot(0, fire)

    return start_dialog_job(ctx, kind, start, lambda: outcome["done"], lambda: collect(outcome),
                            description=description)
