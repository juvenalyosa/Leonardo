"""External MCP requests are confirmed without a nested event loop (Qt 5 on macOS 26 crashed inside exec_())."""
import os
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _controller(app):
    from leonardo.agent.ui.bridge import MainThreadRunner
    from leonardo.agent.ui.controller import AssistantController

    ctl = SimpleNamespace(panel=QtWidgets.QWidget(), runner=MainThreadRunner(),
                          EXTERNAL_CONFIRM_TIMEOUT_S=AssistantController.EXTERNAL_CONFIRM_TIMEOUT_S)
    return ctl, AssistantController._confirm_external


def _ask_from_thread(ctl, confirm, timeout=None):
    out = {}
    spec = SimpleNamespace(title="Run docking")
    th = threading.Thread(target=lambda: out.setdefault("ok", confirm(ctl, spec, {"x": 1}, timeout=timeout)))
    th.start()
    return th, out


def _wait_box(app, ctl, seconds=5.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        boxes = list(getattr(ctl, "_external_boxes", ()))
        if boxes:
            return boxes[0]
        time.sleep(0.01)
    raise AssertionError("no confirmation box")


@pytest.mark.parametrize("button, expected", [("Yes", True), ("No", False)])
def test_external_request_is_answered_without_exec(app, monkeypatch, button, expected):
    def no_exec(*a, **k):
        raise AssertionError("exec_() runs a nested event loop")

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec_", no_exec)
    ctl, confirm = _controller(app)
    th, out = _ask_from_thread(ctl, confirm)
    box = _wait_box(app, ctl)
    assert box.isVisible()
    box.button(getattr(QtWidgets.QMessageBox, button)).click()
    for _ in range(200):
        app.processEvents()
        if not th.is_alive():
            break
        th.join(0.01)
    assert out["ok"] is expected


def test_unanswered_external_request_is_declined(app):
    ctl, confirm = _controller(app)
    th, out = _ask_from_thread(ctl, confirm, timeout=0.3)
    _wait_box(app, ctl)
    for _ in range(200):
        app.processEvents()
        if not th.is_alive():
            break
        th.join(0.01)
    assert out["ok"] is False
