"""Chat panel: the composer answers whatever the agent is waiting for (question, plan, confirmation)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets

from leonardo.agent.ui.chat_panel import ChatPanel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def panel(app):
    p = ChatPanel()
    answers = []
    p.answer_given.connect(answers.append)
    p.set_busy(True)
    assert p.input.isReadOnly() and not p.btn_send.isEnabled()
    return p, answers


def _type_and_send(p, text):
    p.input.setPlainText(text)
    p._send()


def test_question_answered_from_composer(panel):
    p, answers = panel
    p.ask_question({"question": "Which chain?", "options": []})
    assert not p.input.isReadOnly() and p.btn_send.isEnabled()
    assert p.btn_send.text() == "Answer"
    _type_and_send(p, "chain A")
    assert answers == [{"text": "chain A"}]
    # answered: back to busy, and a second send does not answer twice
    assert p.input.isReadOnly() and p.btn_send.text() == "Send"
    _type_and_send(p, "again")
    assert answers == [{"text": "chain A"}]


def test_plan_revised_from_composer(panel):
    p, answers = panel
    p.show_plan({"goal": "Dock BEN", "steps": [{"title": "Fetch 3PTB"}]}, interactive=True)
    _type_and_send(p, "use exhaustiveness 32")
    assert answers == [{"decision": "revise", "feedback": "use exhaustiveness 32"}]


def test_confirm_skipped_with_reason_from_composer(panel):
    p, answers = panel
    p.ask_confirm({"tool": "run_md", "title": "Molecular dynamics (OpenMM)", "args": {}, "permission": "compute_heavy"})
    _type_and_send(p, "too long")
    assert answers == [{"approved": False, "feedback": "too long"}]


def test_request_end_clears_pending(panel):
    p, answers = panel
    p.ask_question({"question": "Which chain?", "options": []})
    p.set_busy(False)
    _type_and_send(p, "new request")
    assert answers == []
