"""Assistant diagnostics/output tools: debug log, task list, readiness, figures, vibrational FF refinement."""
import os
import subprocess

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from leonardo.agent.jobs import JobStore  # noqa: E402
from leonardo.agent.registry import ToolError, ToolRegistry  # noqa: E402
from leonardo.agent.tools import diagnostics  # noqa: E402


def test_debug_log_tail_filters_and_limits():
    lines = [f"[10:00:{i:02d}] step {i}" for i in range(10)] + ["[10:01:00] Traceback: boom"]
    assert diagnostics.debug_log_tail(lines, 3) == lines[-3:]
    assert diagnostics.debug_log_tail(lines, 50, contains="traceback") == ["[10:01:00] Traceback: boom"]


def test_task_rows_sorted_with_running_time():
    state = {
        "7": {"tool_label": "OpenMM", "task_label": "MD", "status": "Running", "progress": 40.0,
              "message": "step 4000", "job_dir": "/tmp/md", "started_at": 100.0},
        "3": {"tool_label": "Psi4", "task_label": "freq", "status": "Completed", "started_at": 50.0},
    }
    rows = diagnostics.task_rows(state, now=160.0)
    assert [r["id"] for r in rows] == ["3", "7"]
    assert rows[1]["running_s"] == 60.0 and rows[1]["progress"] == 40.0


def test_job_store_lists_assistant_jobs():
    store = JobStore()
    store.start("md", lambda: {"done": False}, description="MD 1 ns")
    listed = store.list()
    assert listed[0]["kind"] == "md" and listed[0]["status"] == "running"


def test_readiness_summary_on_this_installation():
    from leonardo.scientific_readiness_runtime import build_suite_readiness_report

    out = diagnostics.readiness_summary(build_suite_readiness_report(workflow_ids=["docking", "psi4_qm"]))
    assert [w["id"] for w in out["workflows"]] == ["docking", "psi4_qm"]
    assert set(out["status_counts"]) >= {"ready", "ready_with_gaps", "missing_required"}


def test_vib_refine_command_prefers_parametrized_files_then_bootstrap(tmp_path):
    ins = {"mol2": "l.mol2", "prmtop": "l.prmtop", "inpcrd": "l.inpcrd", "sdf": "l.sdf"}
    argv = diagnostics.vib_refine_command("py", ins, str(tmp_path), qm_engine="mopac", n_iter=3)
    assert argv[:3] == ["py", "-m", "leonardo.run_ligand_vib_refine"]
    assert argv[argv.index("--mol2") + 1] == "l.mol2" and "--qm-start-sdf" in argv
    assert argv[argv.index("--n-iter") + 1] == "3"
    boot = diagnostics.vib_refine_command("py", {"sdf": "l.sdf"}, str(tmp_path))
    assert boot[boot.index("--bootstrap-template-sdf") + 1] == "l.sdf"
    with pytest.raises(ToolError, match="parametrize"):
        diagnostics.vib_refine_command("py", {}, str(tmp_path))


def test_vib_refine_module_starts_as_subprocess():
    pkg_root = os.path.join(os.path.dirname(__file__), "..", "src")
    env = dict(os.environ, PYTHONPATH=os.path.abspath(pkg_root))
    out = subprocess.run([diagnostics._python_executable(), "-m", "leonardo.run_ligand_vib_refine", "--help"],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0 and "--qm-engine" in out.stdout


def test_tools_registered_with_permissions():
    reg = ToolRegistry()
    diagnostics.register(reg)
    perms = {s.name: s.permission for s in reg.specs()}
    assert perms == {"read_diagnostics_log": "read", "list_tasks": "read", "readiness_report": "read",
                     "export_figure": "compute_light", "refine_ligand_vibrational_ff": "compute_heavy"}


@pytest.fixture(scope="module")
def dialog():
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2

    dlg = code_v2.ProtLigInteractDialog()
    app.processEvents()
    yield dlg
    dlg.close()


def test_log_tasks_and_figure_on_real_dialog(dialog, tmp_path):
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    ctx = AgentContext(dlg=dialog)
    ctx.workspace = lambda sub="": str(tmp_path / sub) if os.makedirs(tmp_path / sub, exist_ok=True) is None else ""
    reg = build_registry(ctx)
    dialog._append_debug_log("assistant diagnostics probe")
    log = reg.call("read_diagnostics_log", {"contains": "diagnostics probe"}, ctx)
    assert log["count"] == 1
    tasks = reg.call("list_tasks", {}, ctx)
    assert "leonardo_tasks" in tasks and "assistant_jobs" in tasks
    cmd.reinitialize()
    cmd.fragment("ala")
    fig = reg.call("export_figure", {"name": "ala fig", "width": 400, "height": 400, "dpi": 72,
                                     "title": "Alanine", "scale_bar": True, "legend": False}, ctx)
    assert os.path.getsize(fig["path"]) > 1000
    assert getattr(dialog, "_settings", {}).get("title_text", "") != "Alanine"      # settings restored
    assert "Interactions.Title" in cmd.get_names("all") or "title_label" in " ".join(cmd.get_names("all"))
