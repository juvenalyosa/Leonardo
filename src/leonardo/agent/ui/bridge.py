"""
Qt plumbing of the assistant: main-thread tool execution, the worker thread that talks
to the model, and the adapter that turns orchestrator events/questions into signals.
"""

import threading
from concurrent.futures import Future

from pymol.Qt import QtCore


class MainThreadRunner(QtCore.QObject):
    """run(fn) executes fn on the Qt main thread and returns its value (blocking the caller)."""

    _invoke = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._invoke.connect(self._run, QtCore.Qt.QueuedConnection)

    def _run(self, job):
        fn, fut = job
        if not fut.set_running_or_notify_cancel():
            return
        try:
            fut.set_result(fn())
        except BaseException as exc:     # re-raised in the calling thread
            fut.set_exception(exc)

    def run(self, fn, timeout=None):
        if QtCore.QThread.currentThread() == self.thread():
            return fn()
        fut = Future()
        self._invoke.emit((fn, fut))
        return fut.result(timeout)


class UiAdapter(QtCore.QObject):
    """
    The orchestrator's ``ui``: on_event() emits ``event``; ask() emits ``question`` and
    blocks the worker thread until the chat answers with answer() (or the session is
    cancelled).
    """

    event = QtCore.Signal(str, object)
    question = QtCore.Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._answer = None
        self._answered = threading.Event()

    def on_event(self, kind, payload):
        self.event.emit(str(kind), dict(payload or {}))

    def ask(self, kind, payload):
        self._answer = None
        self._answered.clear()
        self.question.emit(str(kind), dict(payload or {}))
        self._answered.wait()
        return dict(self._answer or {"cancelled": True})

    def answer(self, data):
        self._answer = dict(data or {})
        self._answered.set()

    def release(self):
        """Unblock a pending ask() (session cancelled / window closed)."""
        if not self._answered.is_set():
            self._answer = {"cancelled": True, "decision": "reject", "approved": False}
            self._answered.set()


class AgentWorker(QtCore.QThread):
    """Runs one user request (AgentSession.send) off the GUI thread."""

    finished_request = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, session, text, parent=None):
        super().__init__(parent)
        self.session = session
        self.text = str(text)

    def run(self):
        try:
            outcome = self.session.send(self.text)
            self.finished_request.emit(str(outcome))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
