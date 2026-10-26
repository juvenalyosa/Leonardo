"""Advanced Tools: assistant tools as forms (leonardo.agent.tool_forms) and the window that runs them."""
import os

import pytest

from leonardo.agent import tool_forms as TF
from leonardo.agent.registry import ToolRegistry


def _registry():
    reg = ToolRegistry()

    @reg.tool("demo_tool", "Demo tool for forms.", {"properties": {
        "object": {"type": "string"},
        "mode": {"type": "string", "enum": ["fast", "exact"]},
        "steps": {"type": "integer", "minimum": 1, "maximum": 100},
        "temperature_K": {"type": "number"},
        "keep": {"type": "boolean"},
        "residues": {"type": "array", "items": {"type": "string"}},
        "weights": {"type": "array", "items": {"type": "number"}},
        "extra": {"type": "object"},
    }, "required": ["object"]}, permission="compute_light")
    def demo_tool(ctx, object, mode="fast", steps=10, temperature_K=300.0, keep=True, residues=None,
                  weights=None, extra=None):
        return {"object": object, "mode": mode, "steps": steps, "keep": keep, "residues": residues,
                "_card": f"**{object}** done"}

    return reg


def test_fields_follow_schema_and_handler_defaults():
    spec = _registry().get("demo_tool")
    fields = {f["name"]: f for f in TF.field_specs(spec)}
    assert TF.field_specs(spec)[0]["name"] == "object"            # required first
    assert fields["object"]["required"] and fields["object"]["is_object"]
    assert [fields[n]["kind"] for n in ("mode", "steps", "temperature_K", "keep", "residues", "weights", "extra")] \
        == ["enum", "int", "number", "bool", "list_str", "list_number", "json"]
    assert TF.default_text(fields["mode"]) == "fast"
    assert TF.default_text(fields["steps"]) == "10"
    assert TF.default_text(fields["residues"]) == ""


def test_collect_args_leaves_empty_fields_to_the_tool_default():
    fields = TF.field_specs(_registry().get("demo_tool"))
    args = TF.collect_args(fields, {"object": "1UBQ", "mode": "", "steps": "", "keep": True,
                                    "residues": "ILE44, LYS48", "weights": "0.5;1", "extra": '{"a": 1}'})
    assert args == {"object": "1UBQ", "residues": ["ILE44", "LYS48"], "weights": [0.5, 1.0], "extra": {"a": 1}}
    assert TF.collect_args(fields, {"object": "x", "keep": False, "mode": "exact", "steps": "5"}) == \
        {"object": "x", "keep": False, "mode": "exact", "steps": 5}


def test_collect_args_reports_every_bad_field():
    fields = TF.field_specs(_registry().get("demo_tool"))
    with pytest.raises(ValueError) as err:
        TF.collect_args(fields, {"object": "", "steps": "500", "temperature_K": "warm", "extra": "{"})
    text = str(err.value)
    for part in ("object is required", "steps: 500 is outside", "temperature_K: 'warm' is not a number",
                 "extra: not valid JSON"):
        assert part in text


def test_result_summary_prefers_the_tool_card():
    assert TF.result_summary({"_card": "**x** done", "a": 1}) == "**x** done"
    assert "job **build-1**" in TF.result_summary({"job_id": "build-1", "status": "running"})
    assert "**a**: 1" in TF.result_summary({"a": 1})


# -- with Leonardo's real tools and the window ------------------------------------------------------------- #
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture()
def real_registry():
    pytest.importorskip("pymol.Qt")
    from leonardo.agent.tools import AgentContext, build_registry

    ctx = AgentContext(dlg=None)
    return ctx, build_registry(ctx)


def test_every_tool_is_grouped_once_and_windowless_tools_come_first(real_registry):
    _ctx, reg = real_registry
    sections = TF.group_tools(reg.specs())
    names = [s.name for _sec, groups in sections for _t, specs in groups for s in specs]
    assert sorted(names) == reg.names()
    assert sections[0][0] == "Without a window"
    windowless = {s.name for _t, specs in sections[0][1] for s in specs}
    assert {"build_membrane_system", "build_material", "design_disulfides", "analyze_interface",
            "build_validated_system", "run_pipeline"} <= windowless
    assert not any(t == TF.OTHER_GROUP for _sec, groups in sections for t, _ in groups), \
        "a tool module is missing from tool_forms.WINDOWLESS_GROUPS / WINDOWED_GROUPS"
    for spec in reg.specs():
        TF.field_specs(spec)                       # every schema becomes a form


def _wait(app, cond, timeout=10.0):
    import time

    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_window_runs_a_tool_and_follows_its_background_job():
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    from leonardo.agent.tools import AgentContext
    from leonardo.agent.ui.tools_dialog import AdvancedToolsDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ctx = AgentContext(dlg=None)
    reg = _registry()
    polls = {"n": 0}

    @reg.tool("slow_tool", "Starts a background job.", {"properties": {}}, permission="compute_heavy")
    def slow_tool(c):
        def poll():
            polls["n"] += 1
            return {"done": polls["n"] >= 2, "result": {"_card": "job finished"}, "progress": "half"}
        return c.jobs.start("slow", poll, description="demo job")

    dlg = AdvancedToolsDialog(reg, ctx)
    dlg._job_timer.setInterval(20)
    dlg.show()
    item = dlg.tree.findItems("Demo tool", QtCore_flags())[0]
    dlg.tree.setCurrentItem(item)
    dlg._widgets["object"].setEditText("1UBQ")
    dlg._widgets["steps"].setText("3")
    dlg.btn_run.click()
    assert _wait(app, lambda: "1UBQ" in dlg.result.toPlainText() and dlg.btn_run.isEnabled())
    assert "done" in dlg.result.toPlainText()

    dlg.tree.setCurrentItem(dlg.tree.findItems("Slow tool", QtCore_flags())[0])
    dlg.btn_run.click()
    assert _wait(app, lambda: "job finished" in dlg.result.toPlainText())
    assert dlg.btn_run.isEnabled() and not dlg.btn_cancel_job.isVisible()

    dlg.tree.setCurrentItem(dlg.tree.findItems("Demo tool", QtCore_flags())[0])
    dlg._widgets["object"].setEditText("")      # the first loaded PyMOL object is preselected when there is one
    dlg.btn_run.click()                                      # required object left empty
    assert _wait(app, lambda: "object is required" in dlg.result.toPlainText())
    dlg.close()


def QtCore_flags():
    from pymol.Qt import QtCore

    return QtCore.Qt.MatchRecursive | QtCore.Qt.MatchExactly
