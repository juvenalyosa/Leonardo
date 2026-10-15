"""Long jobs run in the background and are reported every 30 min (Monte Carlo ensemble sampling, assistant jobs)."""
import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets

FAKE_SAMPLER = r'''
import json, os, sys, time
args = json.load(open(sys.argv[1]))
n = int(args["n_structures"])
delay = float(os.environ.get("FAKE_MC_DELAY", "0.05"))
out = []
for k in range(1, n + 1):
    time.sleep(delay)
    path = os.path.join(args["out_dir"], f"mc_{k:03d}.pdb")
    open(path, "w").write("END\n")
    out.append({"path": path, "relative_energy_kj_mol": -1.0 * k, "site_rmsd_to_start_A": 0.3})
    print(f"PROGRESS {k}/{n} structures (step {k})", flush=True)
json.dump({"ok": True, "structures": out, "acceptance": 0.3}, open(os.path.join(args["out_dir"], "mc_ensemble.json"), "w"))
'''


@pytest.fixture(scope="module")
def dialog():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2

    dlg = code_v2.ProtLigInteractDialog()
    app.processEvents()
    yield dlg
    dlg.close()


@pytest.fixture
def fake_sampler(dialog, tmp_path, monkeypatch):
    script = tmp_path / "fake_mc.py"
    script.write_text(FAKE_SAMPLER)
    monkeypatch.setattr(dialog, "_modeler_import_module", lambda name: SimpleNamespace(__file__=str(script)))
    dialog._redock_mc_cache = {}
    return tmp_path


def _plan(tmp_path, n=4):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / "best_model.pdb").write_text("END\n")
    return {"label": "WT", "best_pdb": str(work / "best_model.pdb"), "work": str(work), "site": [("A", "1")],
            "n": n, "temp": 300.0, "exclude": [], "keep_space": False, "seed": 1}


def _tick_until(dialog, cond, seconds=20.0):
    app = QtWidgets.QApplication.instance()
    deadline = time.time() + seconds
    while time.time() < deadline:
        dialog._redock_mc_bg_tick()
        app.processEvents()
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_background_sampling_reports_and_fills_the_cache(dialog, fake_sampler, monkeypatch):
    monkeypatch.setattr(type(dialog), "_redock_mc_report_every_s", 0.2)
    dialog._redock_mc_reports = []
    bg = dialog._redock_mc_start_background("key-1", _plan(fake_sampler), owner="agent")
    assert bg["state"] == "running" and dialog._redock_mc_running()
    assert not dialog.btn_run_redocking.isEnabled() and dialog.btn_redock_cancel_mc.isEnabled()
    assert _tick_until(dialog, lambda: dialog._redock_mc_bg["state"] != "running")
    assert dialog._redock_mc_bg["state"] == "done" and dialog._redock_mc_bg["produced"] == 4
    cached = dialog._redock_mc_cache["key-1"]
    assert len(cached["result"]["structures"]) == 4 and cached["plan"]["label"] == "WT"
    assert any("started in the background" in r for r in dialog._redock_mc_reports)
    assert any("4/4 structures" in r for r in dialog._redock_mc_reports)
    assert dialog.btn_run_redocking.isEnabled() and not dialog.btn_redock_cancel_mc.isEnabled()


def test_periodic_report_flags_a_stalled_sampler(dialog, fake_sampler, monkeypatch):
    monkeypatch.setenv("FAKE_MC_DELAY", "30")              # no structure for a long time
    monkeypatch.setattr(type(dialog), "_redock_mc_report_every_s", 0.3)
    dialog._redock_mc_reports = []
    dialog._redock_mc_start_background("key-2", _plan(fake_sampler, n=2), owner="agent")
    assert _tick_until(dialog, lambda: any("no new structure" in r for r in dialog._redock_mc_reports), 10)
    assert dialog._redock_mc_cancel_background()
    assert dialog._redock_mc_bg["state"] == "cancelled" and dialog._redock_mc_bg["proc"].poll() is not None
    assert "key-2" not in dialog._redock_mc_cache


def test_progress_text_estimates_the_remaining_time(dialog):
    dialog._redock_mc_bg = {"label": "WT", "started": time.time() - 3600, "produced": 3, "n": 6, "state": "running"}
    text = dialog._redock_mc_progress_text()
    assert "3/6 structures after 1 h 00 min" in text and "about 1 h 00 min left" in text


def test_assistant_jobs_are_reported_every_30_minutes():
    from leonardo.agent.ui.controller import AssistantController

    notices, errors = [], []
    state = {"status": "running", "elapsed": 0.0}

    class Jobs:
        def list(self):
            return [{"job_id": "redocking-1", "kind": "redocking", "description": "redocking of x",
                     "status": state["status"] if state["status"] != "running" else "running",
                     "elapsed_s": state["elapsed"]}]

        def status(self, job_id):
            if state["status"] == "running":
                return {"status": "running", "elapsed_s": state["elapsed"], "progress": "MC 3/20"}
            return {"status": state["status"], "elapsed_s": state["elapsed"], "error": "boom"}

    ctl = SimpleNamespace(context=SimpleNamespace(jobs=Jobs()),
                          panel=SimpleNamespace(add_notice=notices.append, add_error=errors.append),
                          JOB_REPORT_EVERY_S=1800.0)
    report = AssistantController._poll_and_report_jobs
    t0 = 1_000_000.0
    assert report(ctl, now=t0) == 1 and not notices                  # just started
    state["elapsed"] = 1799.0
    report(ctl, now=t0 + 1799.0)
    assert not notices
    state["elapsed"] = 1801.0
    report(ctl, now=t0 + 1801.0)
    assert len(notices) == 1 and "0 h 30 min" in notices[0] and "MC 3/20" in notices[0]
    state["elapsed"] = 2500.0
    report(ctl, now=t0 + 2500.0)
    assert len(notices) == 1                                          # next report at 60 min
    state["elapsed"], state["status"] = 3700.0, "failed"
    assert report(ctl, now=t0 + 3700.0) == 0
    assert errors and "failed" in errors[0]
    report(ctl, now=t0 + 3800.0)
    assert len(errors) == 1                                           # the end is reported once
