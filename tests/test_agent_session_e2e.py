"""The assistant end to end with a scripted model on Leonardo's real tools (headless window, no network):
plan -> approval -> confirmed/declined heavy steps -> a MODELLER job followed to the end -> errors back to
the model -> save the session as a pipeline -> re-run it twice with other molecules -> replay it with
``python -m leonardo.replay --set``; plus the tool-call limit and the MCP server on the real registry."""
import json
import os
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
cmd = pytest.importorskip("pymol").cmd
pytest.importorskip("pdbfixer")

from leonardo.agent import pipelines as P  # noqa: E402
from leonardo.agent.orchestrator import AgentSession  # noqa: E402
from leonardo.agent.providers.base import ToolCall  # noqa: E402
from leonardo.agent.providers.fake import ScriptedProvider  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "tests", "data", "components")
UBQ = os.path.join(DATA, "1UBQ.pdb")
CYTC = os.path.join(DATA, "1HRC_full_nowat.pdb")
PLAN = {"goal": "K48R mutant of ubiquitin", "steps": [{"title": "load", "detail": "1UBQ"},
                                                     {"title": "mutate", "detail": "K48R"}]}


@pytest.fixture(scope="module")
def leo(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("leo"))
    from leonardo import replay

    app, ctx, reg = replay._headless_leonardo(work)
    yield app, ctx, reg, work
    ctx.dlg.close()


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "LIB_DIR", str(tmp_path / "pipelines"))
    P.clear_history()
    yield str(tmp_path / "pipelines")
    P.clear_history()


class UI:
    def __init__(self, decide):
        self.decide, self.events, self.asked = decide, [], []

    def on_event(self, kind, payload):
        self.events.append((kind, payload))

    def ask(self, kind, payload):
        self.asked.append((kind, payload.get("tool")))
        return self.decide(kind, payload)


def _session(leo, turns, decide, **kw):
    app, ctx, reg, _work = leo

    def execute(name, args):
        out = reg.call(name, args, ctx)
        app.processEvents()
        return out
    return AgentSession(ScriptedProvider(turns), reg, UI(decide), execute=execute, poll_interval=0.05, **kw)


def _results(sess):
    return [r for kind, rs in sess.provider.transcript if kind == "tool_results" for r in rs]


def _decide(kind, payload):
    if kind == "plan":
        return {"decision": "approve"}
    if payload["args"].get("mutations") == ["Q2E"]:
        return {"approved": False, "feedback": "not this one"}
    return {"approved": True}


def test_session_plan_confirm_jobs_errors_pipeline_and_replay(leo, library, tmp_path):
    pytest.importorskip("modeller")
    cmd.reinitialize()
    turns = [
        [("tool", "list_objects", {}), ("tool", "load_structure_file", {"path": UBQ, "object_name": "ubq"})],
        [("tool", "propose_plan", PLAN)],
        [("tool", "mark_step", {"step": 1, "status": "running"}),
         ("tool", "load_structure_file", {"path": UBQ, "object_name": "ubq"})],
        [("tool", "model_mutant", {"object": "ubq", "mutations": ["K48R"], "method": "local"}),
         ("tool", "align_structures", {"mobile": "ubq_K48R", "target": "ubq"}),
         ("tool", "model_mutant", {"object": "ubq", "mutations": ["Q2E"], "num_models": 1}),
         ("tool", "model_mutant", {"object": "ubq", "mutations": ["K6R"], "num_models": 1, "name": "ubq_k6r"}),
         ("tool", "focus_view", {"selection": "no_such_object"})],
        ["Done."],
    ]
    sess = _session(leo, turns, _decide)
    assert sess.send("make the K48R mutant of ubiquitin") == "done"
    res = _results(sess)
    assert res[0]["is_error"] is False and res[1]["is_error"]          # loading is gated until a plan exists
    assert "propose_plan" in res[1]["content"]
    local, align, declined, job, failed = res[5:10]
    assert json.loads(local["content"])["max_shift_outside_mutation_A"] == 0.0
    assert json.loads(align["content"])["rmsd_A"] == 0.0
    assert declined["is_error"] and "not this one" in declined["content"]
    job = json.loads(job["content"])                                  # MODELLER job followed to its end
    assert job["object"] == "ubq_k6r" and job["job_id"].startswith("modeling")
    found = []
    cmd.iterate("ubq_k6r and resi 6 and name CA", "found.append(resn)", space={"found": found})
    assert found == ["ARG"]
    assert failed["is_error"] and "no_such_object" in failed["content"]
    heavy = [tool for kind, tool in sess.ui.asked if kind == "confirm_tool"]
    assert heavy == ["model_mutant", "model_mutant", "model_mutant"]  # the view/light steps were not asked
    assert [h["tool"] for h in P.HISTORY][:3] == ["load_structure_file", "model_mutant", "align_structures"]

    # -- save the first three steps as a pipeline and re-run it twice with cytochrome c ------------- #
    values = {"structure_file": CYTC, "mutation": "K13A"}
    turns = [
        [("tool", "propose_plan", {"goal": "save and re-run", "steps": [{"title": "pipeline", "detail": "x"}]})],
        [("tool", "save_pipeline", {"name": "local mutant", "first_step": 1, "last_step": 3,
                                    "parameters": {"structure_file": UBQ, "mutation": "K48R"}})],
        [("tool", "run_pipeline", {"name": "local mutant", "values": values})],
        [("tool", "run_pipeline", {"name": "local mutant", "values": values})],
        ["Both runs done."],
    ]
    sess = _session(leo, turns, _decide)
    assert sess.send("save it and run it on cytochrome c twice") == "done"
    saved, run1, run2 = (json.loads(r["content"]) for r in _results(sess)[1:4])
    assert [s["args"] for s in saved["steps"]] == [
        {"path": "{{structure_file}}", "object_name": "ubq"},
        {"object": "{{step:1.object}}", "mutations": ["{{mutation}}"], "method": "local"},
        {"mobile": "{{step:2.object}}", "target": "{{step:1.object}}"}]
    for run in (run1, run2):                     # the second run used to stop at step 1 (name already taken)
        assert run["steps_done"] == 3 and "stopped_at" not in run
        assert run["results"]["2"]["max_shift_outside_mutation_A"] == 0.0
    assert run1["results"]["1"]["object"] != run2["results"]["1"]["object"]
    assert all(cmd.count_states(run["results"]["1"]["object"]) == 1 for run in (run1, run2))

    # -- replay the saved pipeline outside the assistant, with another mutation ------------------- #
    pack = tmp_path / "pack"
    pack.mkdir()
    with open(os.path.join(library, "local_mutant.json")) as fh:
        (pack / "pipeline.json").write_text(fh.read())
    (pack / "expected.json").write_text(json.dumps({"metrics": {
        "mutation": {"path": "2.mutations.0", "value": "K72E", "kind": "exact"},
        "kept": {"path": "2.max_shift_outside_mutation_A", "value": 0.0, "kind": "exact"},
        "rmsd": {"path": "3.rmsd_A", "value": 0.0, "tol": 0.01}}}))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(REPO, "legacy"), os.path.join(REPO, "src"),
                                                       os.environ.get("PYTHONPATH", "")]),
               QT_QPA_PLATFORM="offscreen")
    out = subprocess.run([sys.executable, "-m", "leonardo.replay", "verify", str(pack), "--set", "mutation=K72E",
                          "--set", f"structure_file={CYTC}", "--report", str(tmp_path / "report.json")],
                         env=env, capture_output=True, text=True, timeout=240)
    report = json.load(open(tmp_path / "report.json"))
    assert out.returncode == 0 and report["ok"], out.stdout[-2000:] + out.stderr[-2000:]


def test_unknown_replay_parameter_is_refused(tmp_path):
    from leonardo import replay

    assert replay.parse_values(["n=10", "ph=7.4", "name=1HRC", "chains=[\"A\", \"B\"]"]) == {
        "n": 10, "ph": 7.4, "name": "1HRC", "chains": ["A", "B"]}
    with pytest.raises(ValueError, match="NAME=VALUE"):
        replay.parse_values(["protein"])
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pipeline.json").write_text(json.dumps({"name": "p", "parameters": {"protein": {"default": "1UBQ"}},
                                                    "steps": [{"tool": "list_objects", "args": {}}]}))
    with pytest.raises(ValueError, match="unknown parameter"):
        replay.run_pipeline(str(pack), str(tmp_path), values={"ligand": "x"})


def test_tool_call_limit_and_malformed_arguments_on_the_real_registry(leo):
    turns = [[("tool", "list_objects", {})] for _ in range(305)]
    sess = _session(leo, turns, _decide)
    assert sess.send("loop") == "stopped"
    res = _results(sess)
    assert len(res) == 301 and "Tool-call limit (300)" in res[-1]["content"]
    assert sum(1 for r in res if not r["is_error"]) == 300
    sess.tool_calls = 0
    bad = sess._handle_call(ToolCall("c1", "list_objects", "{not json"))
    assert bad["is_error"] and "not valid JSON" in bad["content"]


def test_mcp_server_on_the_real_registry_confirms_heavy_tools(leo):
    from leonardo.agent.mcp_server import McpToolClient, build_server

    _app, ctx, reg, _work = leo
    cmd.reinitialize()
    cmd.load(UBQ, "ubq")
    asked = []
    client = McpToolClient(build_server(reg, run_on_main=lambda fn: fn(), context=ctx,
                                        confirm=lambda spec, args: asked.append(spec.name) or False))
    try:
        assert set(client.list_tool_names()) == set(reg.names())
        assert client.call_tool("describe_object", {"object": "ubq"})["chains"] == {"A": 76}
        with pytest.raises(Exception, match="declined"):
            client.call_tool("model_mutant", {"object": "ubq", "mutations": ["K48R"], "method": "local"})
        assert asked == ["model_mutant"] and "ubq_K48R" not in cmd.get_names("objects")
    finally:
        client.close()
