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


def test_old_card_cannot_answer_a_new_question(panel):
    p, answers = panel
    p.ask_question({"question": "Which chain?", "options": ["A", "B"]})
    p.set_busy(False)                      # request stopped without an answer
    p.set_busy(True)
    p.ask_confirm({"tool": "run_md", "title": "MD", "args": {}, "permission": "compute_heavy"})
    stale = [w for w in p.findChildren(QtWidgets.QPushButton) if w.text() == "A"][0]
    assert not stale.isEnabled()
    stale.click()
    assert answers == [] and p._pending is not None and p._pending[0] == "confirm"


def test_tool_values_and_errors_are_shown_literally(panel):
    from leonardo.agent.ui.chat_panel import _md_escape, _result_summary

    assert _md_escape("C[C@H](N)C(=O)O") == "C\\[C@H\\]\\(N\\)C\\(=O\\)O"
    assert "<b>" not in _result_summary({"path": "/tmp/a*b*c.pdb", "x": "<b>bold</b>"})
    p, _answers = panel
    p.tool_event({"tool": "t", "title": "T", "status": "running", "args": {}, "permission": "read"})
    p.tool_event({"tool": "t", "title": "T", "status": "failed", "error": "'<' not supported between 'a' and <int>"})
    labels = [w.text() for w in p.findChildren(QtWidgets.QLabel)]
    assert any("&lt;int&gt;" in t for t in labels)


def test_long_plan_and_tool_cards_fit_the_chat_width(app):
    """A long time estimate or long tool arguments must wrap, never widen the feed past the window."""
    p = ChatPanel()
    p.resize(500, 700)
    p.show()
    long_estimate = ("~10-30 min en este i7-8850H (6 núcleos, sin GPU relevante para Psi4); Leonardo no tiene "
                     "una estimación propia para optimizaciones DFT")
    p.show_plan({"goal": "Optimizar la cafeína con DFT y mostrar HOMO y LUMO", "estimated_time": long_estimate,
                 "steps": [{"title": "Optimización DFT con Psi4", "heavy": True,
                            "detail": "B3LYP-D3BJ/def2-SVP en fase gas, molécula completa (24 átomos), 6 hilos"}]})
    p.tool_event({"tool": "run_psi4", "title": "Quantum chemistry (Psi4)", "permission": "compute_heavy",
                  "status": "running", "args": {"operation": "optimize", "object": "dock_ligand_preview",
                                                "method": "b3lyp-d3bj", "basis": "def2-SVP", "threads": 6},
                  "estimate": "~4 min with 6 threads (246 basis functions, ~12 gradient(s))"})
    for _ in range(5):
        app.processEvents()
    viewport = p.scroll.viewport().width()
    assert p.scroll.widget().width() <= viewport
    cards = [w for w in p.scroll.widget().findChildren(QtWidgets.QFrame) if w.objectName() in ("cardPlan", "card")]
    assert cards and all(c.width() <= viewport for c in cards)
    assert any(long_estimate in lab.text() for lab in p.findChildren(QtWidgets.QLabel) if lab.wordWrap())
    p.close()
