"""Quitting PyMOL with Leonardo open: Leonardo is closed before PyMOL's exit() (a PySide6 crash otherwise)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets


def test_quit_closes_leonardo_first(monkeypatch):
    import ProtLigInteract as P
    from pymol import cmd

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    calls = []
    monkeypatch.setattr(cmd, "_quit", lambda *a, **k: calls.append(("pymol quit", a)), raising=False)
    P._install_quit_teardown()
    P._install_quit_teardown()                     # idempotent: one wrapper only
    assert getattr(cmd._quit, "_leonardo_teardown", False)

    dlg = QtWidgets.QDialog()
    child = QtWidgets.QLabel("x", dlg)
    dlg.show()
    app.processEvents()
    monkeypatch.setattr(P, "_dialog", dlg)
    cmd._quit(0)
    assert calls == [("pymol quit", (0,))]          # PyMOL still quits, after the teardown
    assert P._dialog is None
    is_valid, _delete = P._qt_binding_ops(dlg)
    assert not is_valid(dlg) and not is_valid(child)   # gone before exit()


def test_teardown_without_leonardo_open_is_harmless(monkeypatch):
    import ProtLigInteract as P

    monkeypatch.setattr(P, "_dialog", None)
    P.teardown_for_quit()
