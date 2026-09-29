"""Reusable pipelines: parameters, links between steps, library, and running them in the orchestrator."""
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
