"""
Long-running jobs started by agent tools (docking, MD, redocking...).

A tool that starts a long calculation registers it here and returns immediately with
``{"job_id": ..., "status": "running"}``; ``job_status`` / ``job_cancel`` (registered
as tools) let the assistant, or an external MCP client, follow it. The poll function
runs on the Qt main thread like any tool and must return quickly.
"""

import itertools
import threading
import time


class JobStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs = {}
        self._ids = itertools.count(1)

    def start(self, kind, poll, cancel=None, description=""):
        """poll() -> {"done": bool, "result": dict, "error": str, "progress": str}."""
        with self._lock:
            job_id = f"{kind}-{next(self._ids)}"
            self._jobs[job_id] = {"kind": kind, "poll": poll, "cancel": cancel, "description": description,
                                  "started": time.time(), "final": None}
        return {"job_id": job_id, "status": "running", "kind": kind, "description": description}

    def status(self, job_id):
        with self._lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            return {"job_id": job_id, "status": "unknown", "error": "no such job"}
        if job["final"] is not None:
            return dict(job["final"])
        try:
            state = dict(job["poll"]() or {})
        except Exception as exc:
            state = {"done": True, "error": f"{type(exc).__name__}: {exc}"}
        elapsed = round(time.time() - job["started"], 1)
        if state.get("done"):
            final = {"job_id": job_id, "status": "failed" if state.get("error") else "done",
                     "elapsed_s": elapsed, "result": dict(state.get("result") or {})}
            if state.get("error"):
                final["error"] = str(state["error"])
            job["final"] = final
            return dict(final)
        return {"job_id": job_id, "status": "running", "elapsed_s": elapsed,
                "progress": str(state.get("progress", "") or "")}

    def cancel(self, job_id):
        with self._lock:
            job = self._jobs.get(str(job_id))
        if job is None:
            return {"job_id": job_id, "status": "unknown"}
        if job["final"] is None and callable(job["cancel"]):
            try:
                job["cancel"]()
            except Exception:
                pass
        job["final"] = job["final"] or {"job_id": job_id, "status": "cancelled"}
        return dict(job["final"])


def register_job_tools(registry, store):
    @registry.tool("job_status", "Status of a long calculation started by another tool (docking, MD, ...). "
                   "Returns running/done/failed and, when done, the result.",
                   {"properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
                   permission="read", title="Job status")
    def job_status(ctx, job_id):
        return store.status(job_id)

    @registry.tool("job_cancel", "Cancel a running long calculation.",
                   {"properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
                   permission="compute_light", title="Cancel job")
    def job_cancel(ctx, job_id):
        return store.cancel(job_id)
