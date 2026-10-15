"""Monte Carlo ensemble redocking: every option the assistant can set is a control in the Leonardo window."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets


@pytest.fixture(scope="module")
def dialog():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2

    dlg = code_v2.ProtLigInteractDialog()
    app.processEvents()
    yield dlg
    dlg.close()


def test_controls_exist_and_follow_the_ensemble_checkbox(dialog):
    controls = (dialog.spin_redock_mc_n, dialog.spin_redock_mc_temp, dialog.spin_redock_mc_radius,
                dialog.cb_redock_mc_keep_ligand_space, dialog.spin_redock_mc_seed, dialog.cb_redock_mc_reproducible)
    dialog.cb_redock_use_mc_ensemble.setChecked(False)
    assert not any(w.isEnabled() for w in controls)
    dialog.cb_redock_use_mc_ensemble.setChecked(True)
    assert all(w.isEnabled() for w in controls)
    assert dialog.btn_redock_show_mc.isEnabled()


def test_seed_and_reproducible_reach_the_sampler(dialog, monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd_args, **kwargs):
        import json

        seen.update(json.load(open(cmd_args[-1])))
        raise RuntimeError("stop after writing the arguments")

    dialog.spin_redock_mc_seed.setValue(4242)
    dialog.cb_redock_mc_reproducible.setChecked(True)
    monkeypatch.setattr("subprocess.Popen", fake_run)
    with pytest.raises(RuntimeError):
        dialog._redock_mc_sample(str(tmp_path / "in.pdb"), str(tmp_path / "out"), [("A", "1")], 5, 300.0,
                                 seed=dialog.spin_redock_mc_seed.value())
    assert seen["seed"] == 4242 and seen["reproducible"] is True


def test_new_controls_are_saved_with_the_project(dialog):
    import inspect

    import code_v2

    src = inspect.getsource(code_v2)
    for name in ("cb_redock_mc_keep_ligand_space", "spin_redock_mc_seed", "cb_redock_mc_reproducible"):
        assert f'"{name}",' in src


def test_show_ensemble_without_a_run_says_so(dialog, monkeypatch):
    shown = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    dialog._redock_last_mc_ensemble = {}
    assert dialog._on_show_redock_mc_ensemble() is None
    assert shown and "No ensemble yet" in shown[0]


def test_assistant_values_appear_in_the_window(dialog):
    """run_redocking(ensemble_seed=..., ensemble_reproducible=...) must set the visible controls too."""
    import inspect
    import re

    from leonardo.agent.tools import redocking

    src = inspect.getsource(redocking)
    assert re.search(r"spin_redock_mc_seed\.setValue\(int\(ensemble_seed\)\)", src)
    assert re.search(r"cb_redock_mc_reproducible\.setChecked\(bool\(ensemble_reproducible\)\)", src)
