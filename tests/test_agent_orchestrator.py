"""Plan/approval/execution loop of the AI assistant, with a scripted provider (no network)."""
import pytest

from leonardo.agent.jobs import JobStore, register_job_tools
from leonardo.agent.mcp_server import McpToolClient, build_server
from leonardo.agent.orchestrator import AgentSession
from leonardo.agent.providers.fake import ScriptedProvider
from leonardo.agent.registry import ToolError, ToolRegistry


class UI:
    def __init__(self, answers):
        self.answers = list(answers)
        self.events = []
        self.asked = []

    def on_event(self, kind, payload):
        self.events.append((kind, payload))

    def ask(self, kind, payload):
        self.asked.append((kind, payload))
        return self.answers.pop(0)


def _registry(log):
    reg = ToolRegistry()
    jobs = JobStore()
    register_job_tools(reg, jobs)

    @reg.tool("list_objects", "List PyMOL objects.", permission="read")
    def list_objects(ctx):
        log.append("list")
        return {"objects": ["3PTB"]}

    @reg.tool("color_by_chain", "Color by chain.", {"properties": {"object": {"type": "string"}},
                                                    "required": ["object"]}, permission="view")
    def color(ctx, object):
        log.append(f"color {object}")
        return {"ok": True}

    @reg.tool("run_docking", "Dock.", {"properties": {"exhaustiveness": {"type": "integer"}}}, permission="compute_heavy")
    def dock(ctx, exhaustiveness=8):
        log.append(f"dock {exhaustiveness}")
        state = {"n": 0}

        def poll():
            state["n"] += 1
            return {"done": state["n"] >= 2, "result": {"best_affinity": -7.1}, "progress": f"{state['n']}/2"}
        return jobs.start("docking", poll)

    @reg.tool("fail_tool", "Always fails.", permission="view")
    def fail(ctx):
        raise ToolError("no such selection")
    return reg


def _session(turns, answers, log, policy="heavy"):
    reg = _registry(log)
    ui = UI(answers)
    prov = ScriptedProvider(turns)
    client = McpToolClient(build_server(reg, run_on_main=lambda fn: fn()))
    sess = AgentSession(prov, reg, ui, execute=client.call_tool, approval_policy=policy, poll_interval=0.0)
    return sess, prov, ui


PLAN = {"goal": "Dock BEN", "steps": [{"title": "Dock", "detail": "smina"}]}


def test_plan_approval_then_heavy_step_confirmation_and_polling():
    log = []
    turns = [
        [("tool", "list_objects", {}), "Looking."],
        [("tool", "propose_plan", PLAN)],
        [("tool", "mark_step", {"step": 1, "status": "running"}), ("tool", "run_docking", {"exhaustiveness": 16})],
        ["Done: -7.1 kcal/mol."],
    ]
    sess, prov, ui = _session(turns, [{"decision": "approve"}, {"approved": True}], log)
    assert sess.send("dock benzamidine") == "done"
    assert log == ["list", "dock 16"]
    assert [k for k, _ in ui.asked] == ["plan", "confirm_tool"]
    done = [p for k, p in ui.events if k == "tool" and p.get("status") == "done"]
    assert done[-1]["result"]["best_affinity"] == -7.1 and done[-1]["result"]["job_id"] == "docking-1"
    assert any(k == "progress" for k, _ in ui.events)
    assert sess.phase == "executing"


def test_execution_tools_are_refused_before_a_plan():
    log = []
    turns = [[("tool", "run_docking", {})], ["I need a plan first."]]
    sess, prov, ui = _session(turns, [], log)
    sess.send("dock")
    assert log == []
    results = [r for kind, r in prov.transcript if kind == "tool_results"][0]
    assert results[0]["is_error"] and "propose_plan" in results[0]["content"]


def test_declined_step_and_rejected_plan():
    log = []
    turns = [[("tool", "propose_plan", PLAN)], [("tool", "run_docking", {})], ["ok, skipped"]]
    sess, prov, ui = _session(turns, [{"decision": "approve"}, {"approved": False, "feedback": "too slow"}], log)
    sess.send("dock")
    assert log == []
    declined = [r for kind, r in prov.transcript if kind == "tool_results"][1][0]
    assert declined["is_error"] and "too slow" in declined["content"]

    log2 = []
    sess2, prov2, ui2 = _session([[("tool", "propose_plan", PLAN)]], [{"decision": "reject"}], log2)
    assert sess2.send("dock") == "stopped" and sess2.phase == "planning"


def test_view_tools_run_without_confirmation_and_errors_reach_the_model():
    log = []
    turns = [[("tool", "propose_plan", PLAN)],
             [("tool", "color_by_chain", {"object": "3PTB"}), ("tool", "fail_tool", {}),
              ("tool", "color_by_chain", {"objekt": "x"})],
             ["done"]]
    sess, prov, ui = _session(turns, [{"decision": "approve"}], log)
    sess.send("color it")
    assert log == ["color 3PTB"]
    res = [r for kind, r in prov.transcript if kind == "tool_results"][1]
    assert not res[0]["is_error"] and res[1]["is_error"] and "no such selection" in res[1]["content"]
    assert res[2]["is_error"] and "invalid arguments" in res[2]["content"]


def test_every_step_policy_asks_for_view_tools_too():
    log = []
    turns = [[("tool", "propose_plan", PLAN)], [("tool", "color_by_chain", {"object": "A"})], ["ok"]]
    sess, prov, ui = _session(turns, [{"decision": "approve"}, {"approved": True}], log, policy="every_step")
    sess.send("x")
    assert [k for k, _ in ui.asked] == ["plan", "confirm_tool"]


def test_registry_renders_for_both_providers():
    reg = _registry([])
    a = {t["name"]: t for t in reg.anthropic_tools()}
    o = {t["name"]: t for t in reg.openai_tools()}
    assert set(a) == set(o) == {"list_objects", "color_by_chain", "run_docking", "fail_tool", "job_status",
                                "job_cancel"}
    assert a["color_by_chain"]["input_schema"]["required"] == ["object"]
    assert o["run_docking"]["type"] == "function"
    assert [t["name"] for t in reg.anthropic_tools(planning=True)] == ["job_status", "list_objects"]
    with pytest.raises(ValueError):
        reg.tool("list_objects", "dup")(lambda ctx: None)


def test_mcp_server_lists_and_calls_registry_tools():
    log = []
    reg = _registry(log)
    client = McpToolClient(build_server(reg, run_on_main=lambda fn: fn()))
    assert "run_docking" in client.list_tool_names()
    assert client.call_tool("list_objects", {}) == {"objects": ["3PTB"]}
    with pytest.raises(ToolError, match="no such selection"):
        client.call_tool("fail_tool", {})
    with pytest.raises(ToolError, match="invalid arguments"):
        client.call_tool("color_by_chain", {"objekt": 1})
    client.close()


def test_external_calls_need_confirmation_for_heavy_tools():
    log = []
    reg = _registry(log)
    asked = []

    def confirm(spec, args):
        asked.append(spec.name)
        return False
    client = McpToolClient(build_server(reg, run_on_main=lambda fn: fn(), confirm=confirm))
    with pytest.raises(ToolError, match="declined"):
        client.call_tool("run_docking", {})
    assert client.call_tool("color_by_chain", {"object": "x"}) == {"ok": True}
    assert asked == ["run_docking"] and log == ["color x"]


def test_external_calls_are_confirmed_under_every_policy_without_a_plan():
    # an external MCP client never had a plan approved: "plan_only" must not let it run MD/docking unasked,
    # and "heavy" must not let it run quick calculations unasked (in the app they need plan approval)
    log = []
    reg = _registry(log)

    @reg.tool("find_pockets", "Pockets.", permission="compute_light")
    def pockets(ctx):
        log.append("pockets")
        return {"n": 1}

    for policy, tool in (("plan_only", "run_docking"), ("heavy", "find_pockets")):
        asked = []

        def confirm(spec, args):
            asked.append(spec.name)
            return False
        client = McpToolClient(build_server(reg, run_on_main=lambda fn: fn(), confirm=confirm,
                                            approval_policy=policy))
        with pytest.raises(ToolError, match="declined"):
            client.call_tool(tool, {})
        assert client.call_tool("list_objects", {}) == {"objects": ["3PTB"]}
        client.close()
        assert asked == [tool], policy
    assert log == ["list", "list"]


def test_tool_call_limit_also_stops_calls_refused_before_a_plan():
    # a model that keeps calling a gated (or unknown) tool while planning must still hit the limit
    log = []
    turns = [[("tool", "run_docking", {})], [("tool", "no_such_tool", {})]] * 8
    sess, prov, ui = _session(turns, [], log)
    sess.max_tool_calls = 5
    assert sess.send("dock") == "stopped"
    results = [r for kind, rs in prov.transcript if kind == "tool_results" for r in rs]
    assert len(results) == 6 and "Tool-call limit (5)" in results[-1]["content"]
    assert log == []

    turns = [[("tool", "mark_step", {"step": 1, "status": "running"})] for _ in range(8)]
    sess, prov, ui = _session(turns, [], log)
    sess.max_tool_calls = 3
    assert sess.send("loop") == "stopped"


def test_job_cancelled_from_elsewhere_is_followed_to_its_end():
    # an uninterruptible job reports "cancelling" until it ends: that is not an unexpected state
    log = []
    states = iter([{"status": "running"}, {"status": "cancelling"}, {"status": "cancelling"},
                   {"status": "cancelled", "error": ""}])
    reg = _registry(log)

    def execute(name, args):
        if name == "job_status":
            return next(states)
        return {"job_id": "md-1", "status": "running"}
    sess = AgentSession(ScriptedProvider([]), reg, UI([]), execute=execute, poll_interval=0.0)
    with pytest.raises(ToolError) as err:
        sess._wait_if_pending({"job_id": "md-1", "status": "running"})
    assert "md-1 cancelled" in str(err.value) and "unexpected" not in str(err.value)
