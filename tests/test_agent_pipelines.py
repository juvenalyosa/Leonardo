"""Reusable pipelines: parameters, links between steps, library, and running them in the orchestrator."""
import json
import os

import pytest

from leonardo.agent import pipelines as P
from leonardo.agent.orchestrator import AgentSession
from leonardo.agent.providers.fake import ScriptedProvider
from leonardo.agent.registry import ToolError, ToolRegistry


@pytest.fixture(autouse=True)
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "LIB_DIR", str(tmp_path / "pipes"))
    P.clear_history()
    yield
    P.clear_history()


def test_history_to_pipeline_with_parameters_and_step_links():
    P.record("fetch_structure", {"identifier": "3PTB", "object_name": "3PTB"}, {"object": "3PTB"})
    P.record("list_objects", {}, {})                                        # look-ups are not steps
    P.record("run_md", {"object": "3PTB", "production_ns": 0.1}, {"object": "openmm_3PTB_155109"})
    P.record("trajectory_rmsd_rmsf", {"object": "openmm_3PTB_155109", "selection": "polymer and name CA"}, {})
    P.record("analyze_interactions", {"object": "openmm_3PTB_155109", "ligand_resn": "BEN"}, {})
    pipe = P.build_from_history("md", parameters={"protein": "3PTB", "ligand": "BEN"})
    steps = pipe["steps"]
    assert [s["tool"] for s in steps] == ["fetch_structure", "run_md", "trajectory_rmsd_rmsf", "analyze_interactions"]
    assert steps[0]["args"] == {"identifier": "{{protein}}", "object_name": "{{protein}}"}
    assert steps[2]["args"]["object"] == "{{step:2.object}}"                 # the MD output, renamed every run
    assert steps[3]["args"]["ligand_resn"] == "{{ligand}}"
    filled, vals = P.instantiate(pipe, {"protein": "1HPV", "ligand": "MK1"})
    assert filled[0]["args"]["identifier"] == "1HPV" and vals["ligand"] == "MK1"
    assert P.resolve_refs(filled[2]["args"], {2: {"object": "openmm_1HPV_1"}})["object"] == "openmm_1HPV_1"
    with pytest.raises(KeyError):
        P.resolve_refs(filled[2]["args"], {})
    with pytest.raises(ValueError):
        P.instantiate(pipe, {"receptor": "x"})


def test_placeholders_keep_types_and_match_whole_tokens():
    assert P.fill("{{compounds}}", {"compounds": [{"smiles": "CCO"}]}) == [{"smiles": "CCO"}]
    assert P.fill("{{p}} and resn {{l}}", {"p": "3PTB", "l": "BEN"}) == "3PTB and resn BEN"
    # '3PTB' must not eat '3PTBX' or 'A3PTB'
    assert P._substitute("3PTBX or A3PTB or (3PTB)", [("3PTB", "{{p}}")]) == "3PTBX or A3PTB or ({{p}})"
    assert P._substitute(7.4, [(7.4, "{{ph}}")]) == "{{ph}}"


def test_library_save_load_list_delete(tmp_path):
    pipe = P.build_from_history("Dock + MD", steps=[{"tool": "fetch_structure", "args": {"identifier": "3PTB"}}],
                                parameters={"protein": "3PTB"})
    path, exported = P.save(pipe, export_path=str(tmp_path))
    assert os.path.isfile(path) and os.path.isfile(exported)
    assert P.load("Dock + MD")["steps"][0]["args"]["identifier"] == "{{protein}}"
    assert P.load(exported)["name"] == "Dock + MD"
    assert [p["name"] for p in P.list_all()] == ["Dock + MD"]
    P.delete("Dock + MD")
    with pytest.raises(FileNotFoundError):
        P.load("Dock + MD")


class UI:
    def __init__(self):
        self.asked = []

    def on_event(self, kind, payload):
        pass

    def ask(self, kind, payload):
        self.asked.append((kind, payload.get("title")))
        return {"decision": "approve"} if kind == "plan" else {"approved": True}


def _registry(log):
    from leonardo.agent.jobs import JobStore, register_job_tools
    from leonardo.agent.tools import pipeline_tools

    reg = ToolRegistry()
    register_job_tools(reg, JobStore())

    @reg.tool("fetch_structure", "f", {"properties": {"identifier": {"type": "string"}}}, permission="compute_light")
    def fetch(ctx, identifier):
        log.append(("fetch", identifier))
        return {"object": identifier}

    @reg.tool("run_md", "m", {"properties": {"object": {"type": "string"}}}, permission="compute_heavy")
    def run_md(ctx, object):
        if object == "BAD":
            raise ToolError("no hydrogens")
        log.append(("md", object))
        return {"object": f"openmm_{object}_1"}

    @reg.tool("trajectory_rmsd_rmsf", "r", {"properties": {"object": {"type": "string"}}}, permission="compute_light")
    def rmsd(ctx, object):
        log.append(("rmsd", object))
        return {"rmsd": 1.0}

    pipeline_tools.register(reg)
    return reg


def _session(turns, reg):
    ui = UI()
    sess = AgentSession(ScriptedProvider(turns), reg, ui, execute=lambda n, a: reg.call(n, a, None),
                        approval_policy="heavy", poll_interval=0.01)
    return sess, ui


def test_orchestrator_records_saves_and_reruns_with_new_molecules():
    log = []
    reg = _registry(log)
    plan = {"goal": "g", "steps": [{"title": "t", "detail": "d"}]}
    turns = [[("tool", "propose_plan", plan)],
             [("tool", "fetch_structure", {"identifier": "3PTB"})],
             [("tool", "run_md", {"object": "3PTB"})],
             [("tool", "trajectory_rmsd_rmsf", {"object": "openmm_3PTB_1"})],
             [("tool", "save_pipeline", {"name": "md", "parameters": {"protein": "3PTB"}})],
             [("tool", "run_pipeline", {"name": "md", "values": {"protein": "1HPV"}})],
             ["done"]]
    sess, ui = _session(turns, reg)
    sess.send("go")
    assert log[-3:] == [("fetch", "1HPV"), ("md", "1HPV"), ("rmsd", "openmm_1HPV_1")]
    # one confirmation for the whole pipeline, not one per heavy step inside it
    assert [t for k, t in ui.asked if k == "confirm_tool"].count("Pipeline: md") == 1
    assert sum(1 for k, t in ui.asked if k == "confirm_tool") == 2           # run_md once in the session + pipeline


def test_pipeline_stops_at_the_failing_step():
    log = []
    reg = _registry(log)
    P.save(P.build_from_history("md", steps=[{"tool": "run_md", "args": {"object": "3PTB"}},
                                             {"tool": "trajectory_rmsd_rmsf", "args": {"object": "{{step:1.object}}"}}],
                                parameters={"protein": "3PTB"}))
    plan = {"goal": "g", "steps": [{"title": "t", "detail": "d"}]}
    prov = ScriptedProvider([[("tool", "propose_plan", plan)],
                             [("tool", "run_pipeline", {"name": "md", "values": {"protein": "BAD"}})], ["ok"]])
    ui = UI()
    sess = AgentSession(prov, reg, ui, execute=lambda n, a: reg.call(n, a, None), approval_policy="heavy")
    sess.send("go")
    res = [r for kind, rs in prov.transcript if kind == "tool_results" for r in rs][-1]
    assert res["is_error"] and '"stopped_at"' in res["content"] and "no hydrogens" in res["content"]
    assert ("rmsd", "openmm_BAD_1") not in log


def test_pipeline_dialog_builds_the_request(tmp_path):
    pytest.importorskip("pymol")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pymol.Qt import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from leonardo.agent.ui.pipeline_dialog import PipelineDialog

    P.save(P.build_from_history("screen", steps=[{"tool": "run_virtual_screening",
                                                  "args": {"compounds": [{"smiles": "CCO"}], "exhaustiveness": 8}}],
                                parameters={"compounds": [{"smiles": "CCO"}]}))
    dlg = PipelineDialog()
    widget, _default = dlg._fields["compounds"]
    widget.setPlainText('[{"smiles": "c1ccccc1O", "name": "phenol"}]')
    dlg._run()
    assert "screen" in dlg.run_text and "phenol" in dlg.run_text
    assert app is not None


def test_short_and_numeric_parameters_are_substituted_precisely():
    pipe = P.build_from_history("p", steps=[
        {"tool": "add_hydrogens", "args": {"object": "3PTB", "chain": "B", "resn": "ZN"}},
        {"tool": "run_md", "args": {"object": "3PTB", "n": 10, "x": 10, "ph": 7.4}}],
        parameters={"chain": "B", "zinc": "ZN", "n": 10, "ph": 7.4})
    a1, a2 = pipe["steps"][0]["args"], pipe["steps"][1]["args"]
    assert a1["chain"] == "{{chain}}" and a1["resn"] == "{{zinc}}"
    assert a2["n"] == "{{n}}" and a2["x"] == 10 and a2["ph"] == "{{ph}}"


def test_short_object_names_become_step_links():
    """Objects named 'ubq' or 'wt' (under 4 characters) stayed literal: a re-run with another molecule worked on
    the old object."""
    P.record("load_structure_file", {"path": "/data/1UBQ.pdb", "object_name": "ubq"}, {"object": "ubq"})
    P.record("model_mutant", {"object": "ubq", "mutations": ["K48R"], "method": "local"}, {"object": "wt"})
    P.record("align_structures", {"mobile": "wt", "target": "ubq and chain A"}, {})
    steps = P.build_from_history("m", parameters={"structure_file": "/data/1UBQ.pdb"})["steps"]
    assert steps[1]["args"]["object"] == "{{step:1.object}}"
    assert steps[2]["args"] == {"mobile": "{{step:2.object}}", "target": "{{step:1.object}} and chain A"}
    assert steps[0]["args"] == {"path": "{{structure_file}}", "object_name": "ubq"}


def test_resume_from_step_reuses_the_earlier_results():
    log = []
    reg = _registry(log)
    P.save(P.build_from_history("md", steps=[{"tool": "run_md", "args": {"object": "3PTB"}},
                                             {"tool": "trajectory_rmsd_rmsf", "args": {"object": "{{step:1.object}}"}},
                                             {"tool": "run_md", "args": {"object": "{{step:1.object}}"}}],
                                parameters={"protein": "3PTB"}))
    plan = {"goal": "g", "steps": [{"title": "t", "detail": "d"}]}
    prov = ScriptedProvider([[("tool", "propose_plan", plan)],
                             [("tool", "run_pipeline", {"name": "md"})],
                             [("tool", "run_pipeline", {"name": "md", "from_step": 2})], ["ok"]])
    sess = AgentSession(prov, reg, UI(), execute=lambda n, a: reg.call(n, a, None), approval_policy="heavy")
    sess.send("go")
    last = [r for kind, rs in prov.transcript if kind == "tool_results" for r in rs][-1]
    assert not last["is_error"], last["content"]
    assert log[-2:] == [("rmsd", "openmm_3PTB_1"), ("md", "openmm_3PTB_1")]


def test_invalid_pipeline_step_reports_the_real_error():
    log = []
    reg = _registry(log)
    P.save(P.build_from_history("bad", steps=[{"tool": "run_md", "args": {"object": 5}}], parameters={}))
    plan = {"goal": "g", "steps": [{"title": "t", "detail": "d"}]}
    prov = ScriptedProvider([[("tool", "propose_plan", plan)], [("tool", "run_pipeline", {"name": "bad"})], ["ok"]])
    AgentSession(prov, reg, UI(), execute=lambda n, a: reg.call(n, a, None), approval_policy="heavy").send("go")
    res = [r for kind, rs in prov.transcript if kind == "tool_results" for r in rs][-1]
    assert res["is_error"] and "is not of type 'string'" in res["content"]


def test_destructive_pipeline_steps_are_still_confirmed():
    reg = _registry([])
    P.save(P.build_from_history("wipe", steps=[{"tool": "delete_pipeline", "args": {"name": "x"}}], parameters={}))
    plan = {"goal": "g", "steps": [{"title": "t", "detail": "d"}]}
    ui = UI()
    prov = ScriptedProvider([[("tool", "propose_plan", plan)], [("tool", "run_pipeline", {"name": "wipe"})], ["ok"]])
    AgentSession(prov, reg, ui, execute=lambda n, a: reg.call(n, a, None), approval_policy="plan_only").send("go")
    assert [k for k, _t in ui.asked].count("confirm_tool") == 1


def test_each_request_starts_gated_with_a_fresh_budget():
    log = []
    reg = _registry(log)
    plan = {"goal": "g", "steps": [{"title": "t", "detail": "d"}]}
    prov = ScriptedProvider([[("tool", "propose_plan", plan)], [("tool", "fetch_structure", {"identifier": "A"})],
                             ["ok"], [("tool", "fetch_structure", {"identifier": "B"})], ["ok"]])
    sess = AgentSession(prov, reg, UI(), execute=lambda n, a: reg.call(n, a, None), max_tool_calls=1)
    sess.send("first")
    sess.send("second")
    assert log == [("fetch", "A")]                                   # B needs a new plan
    res = [r for kind, rs in prov.transcript if kind == "tool_results" for r in rs][-1]
    assert "propose_plan" in res["content"] and "limit" not in res["content"]


def test_bad_session_tool_arguments_do_not_escape():
    reg = _registry([])
    prov = ScriptedProvider([[("tool", "mark_step", {"step": "first"})], ["ok"]])
    assert AgentSession(prov, reg, UI(), execute=lambda n, a: reg.call(n, a, None)).send("go") == "done"


def test_update_pipeline_keeps_step_links():
    reg = _registry([])
    P.save(P.build_from_history("md", steps=[{"tool": "fetch_structure", "args": {"identifier": "3PTB"}},
                                             {"tool": "run_md", "args": {"object": "{{step:1.object}}"}},
                                             {"tool": "trajectory_rmsd_rmsf", "args": {"object": "{{step:2.object}}"}}],
                                parameters={}))
    out = reg.call("update_pipeline", {"name": "md", "insert_steps": [{"position": 1, "tool": "fetch_structure",
                                                                         "args": {"identifier": "1ABC"}}]}, None)
    assert out["steps"][3]["args"]["object"] == "{{step:3.object}}"
    with pytest.raises(ToolError):
        reg.call("update_pipeline", {"name": "md", "remove_steps": [3]}, None)


# save/import/rename are quick (compute_light, not confirmed under the default policy): they must not replace a
# saved pipeline or an existing file; only delete_pipeline (destructive, always asked) removes one
_STEPS = [{"tool": "fetch_structure", "args": {"identifier": "3PTB"}}]


def _saved_md(reg):
    reg.call("save_pipeline", {"name": "md", "steps": _STEPS}, None)
    return open(P.path_for("md")).read()


def test_save_pipeline_does_not_overwrite_a_pipeline_or_a_file(tmp_path):
    reg = _registry([])
    first = _saved_md(reg)
    with pytest.raises(ToolError, match="not overwritten"):
        reg.call("save_pipeline", {"name": "md", "steps": [{"tool": "run_md", "args": {"object": "X"}}]}, None)
    assert open(P.path_for("md")).read() == first
    victim = tmp_path / "notes.json"
    victim.write_text("keep me")
    with pytest.raises(ToolError, match="not overwritten"):
        reg.call("save_pipeline", {"name": "other", "steps": _STEPS, "export_path": str(victim)}, None)
    assert victim.read_text() == "keep me" and not os.path.exists(P.path_for("other"))


def test_import_pipeline_does_not_overwrite_a_saved_one(tmp_path):
    reg = _registry([])
    first = _saved_md(reg)
    shared = tmp_path / "from_a_colleague.json"           # another pipeline that is also called "md"
    shared.write_text(json.dumps(dict(P.load("md"), steps=[{"tool": "run_md", "args": {"object": "Y"}}])))
    with pytest.raises(ToolError, match="not overwritten"):
        reg.call("import_pipeline", {"path": str(shared)}, None)
    assert open(P.path_for("md")).read() == first


def test_update_pipeline_rename_does_not_overwrite_another_pipeline():
    reg = _registry([])
    first = _saved_md(reg)
    reg.call("save_pipeline", {"name": "dock", "steps": _STEPS}, None)
    with pytest.raises(ToolError, match="not overwritten"):
        reg.call("update_pipeline", {"name": "dock", "rename_to": "md"}, None)
    assert open(P.path_for("md")).read() == first and os.path.exists(P.path_for("dock"))
    reg.call("update_pipeline", {"name": "md", "description": "edited in place"}, None)
    assert P.load("md")["description"] == "edited in place"


def test_update_pipeline_numeric_parameter_replaces_only_its_argument():
    reg = _registry([])
    P.save(P.build_from_history("dock", steps=[{"tool": "run_md", "args": {"object": "A", "exhaustiveness": 8,
                                                                           "num_modes": 8}}], parameters={}))
    with pytest.raises(ToolError, match="several arguments"):
        reg.call("update_pipeline", {"name": "dock", "add_parameter": {"effort": 8}}, None)
    assert P.load("dock")["steps"][0]["args"] == {"object": "A", "exhaustiveness": 8, "num_modes": 8}
    out = reg.call("update_pipeline", {"name": "dock", "add_parameter": {"exhaustiveness": 8}}, None)
    assert out["steps"][0]["args"] == {"object": "A", "exhaustiveness": "{{exhaustiveness}}", "num_modes": 8}


def test_import_of_a_non_pipeline_json_and_run_past_the_end_are_clear_errors(tmp_path):
    """A JSON list crashed import_pipeline with AttributeError; an external run_pipeline with from_step past
    the end returned an empty step list as if there were nothing to do."""
    reg = _registry([])
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2]")
    with pytest.raises(ToolError, match="is not a Leonardo pipeline"):
        reg.call("import_pipeline", {"path": str(bad)}, None)
    _saved_md(reg)
    with pytest.raises(ToolError, match="past the end"):
        reg.call("run_pipeline", {"name": "md", "from_step": 99}, None)
