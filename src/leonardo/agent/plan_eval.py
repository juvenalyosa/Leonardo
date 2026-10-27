"""
Planning benchmark of the assistant: does it pick the key tools for a request, in a workable order?

Each case is a request (and optionally structures loaded first) with what a good plan must contain:

- ``all``: groups of tools; the plan (or the read-only calls made while planning) must use one tool of each
  group, e.g. ``[["select_pocket", "set_docking_box"], ["run_docking"]]``;
- ``order``: pairs ``[a, b]``: the first use of ``a`` comes before the first use of ``b``;
- ``none``: tools that must not appear (calculations the request did not ask for);
- ``prefer``: good practice that is reported but does not fail the case (e.g. prepare_protein before MD).

Only the planning phase runs: read-only tools execute, the plan is captured and rejected, so no
calculation starts. ``run_case`` drives one case with any provider; ``scripts/eval_agent_planning.py`` runs
the benchmark with the provider and model of the assistant's settings.
"""

import json
import os
import threading
import time

from . import plan_check
from .orchestrator import SYSTEM_PROMPT, AgentSession

CASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "benchmarks",
                          "agent_planning", "cases.json")
QUESTION_ANSWER = ("I cannot answer questions right now: choose sensible defaults, state them as assumptions in "
                   "the plan, and propose it.")


def system_prompt(recipes=True, state=True):
    """The assistant's system prompt; without recipes and state, the earlier one (for A/B runs)."""
    if recipes and state:
        return SYSTEM_PROMPT
    start = SYSTEM_PROMPT.index("- Each request starts with an automatic")
    end = SYSTEM_PROMPT.index("\n- ", start) + 1
    return SYSTEM_PROMPT[:start] + SYSTEM_PROMPT[end:]


def load_cases(path=None):
    with open(path or CASES_FILE, encoding="utf-8") as fh:
        return json.load(fh)["cases"]


def plan_tools(plan):
    return [str(t) for step in list((plan or {}).get("steps") or []) for t in (step.get("tools") or [])]


def grade(case, called, plan, problems=()):
    """Score one case from the read-only tools called while planning and the proposed plan."""
    expect = case.get("expect") or {}
    seq = list(called) + plan_tools(plan)
    first = {}
    for i, name in enumerate(seq):
        first.setdefault(name, i)
    missing = [group for group in expect.get("all", []) if not any(t in first for t in group)]
    bad_order = [pair for pair in expect.get("order", [])
                 if pair[0] in first and pair[1] in first and first[pair[0]] > first[pair[1]]]
    unwanted = [t for t in expect.get("none", []) if t in first]
    preferred = [t for t in expect.get("prefer", []) if t in first]
    n_groups = len(expect.get("all", []))
    passed = bool(plan or not case.get("needs_plan", True)) and not missing and not bad_order and not unwanted
    return {
        "id": case["id"],
        "passed": passed,
        "has_plan": bool(plan),
        "coverage": round(1.0 - len(missing) / n_groups, 3) if n_groups else 1.0,
        "missing": missing,
        "bad_order": bad_order,
        "unwanted": unwanted,
        "prefer_hit": f"{len(preferred)}/{len(expect.get('prefer', []))}",
        "plan_problems": list(problems),
        "tools": seq,
    }


class _CaptureUI:
    """Answers the session like a user who rejects the plan after reading it (nothing heavy runs)."""

    def __init__(self):
        self.plan = None
        self.questions = 0

    def on_event(self, kind, payload):
        pass

    def ask(self, kind, payload):
        if kind == "plan":
            self.plan = dict(payload.get("plan") or {})
            return {"decision": "reject"}
        if kind == "question":
            self.questions += 1
            return {"text": QUESTION_ANSWER}
        return {"approved": False}


def run_case(case, provider, registry, execute, state_facts=None, state_summary=None, max_tool_calls=40,
             timeout=300.0):
    """Run the planning phase of one case; returns the grade with usage, questions and time.

    A case still running after ``timeout`` seconds is cancelled and fails with outcome "timeout".
    """
    ui = _CaptureUI()
    called = []

    def tracked(name, args):
        called.append(name)
        return execute(name, args)

    session = AgentSession(provider, registry, ui, execute=tracked, max_tool_calls=max_tool_calls,
                           state_facts=state_facts, state_summary=state_summary)
    started = time.time()
    timer = threading.Timer(timeout, session.cancel) if timeout else None
    if timer:
        timer.daemon = True
        timer.start()
    try:
        outcome = session.send(case["request"])
    except Exception as exc:          # a provider or harness failure counts against the case, not the run
        outcome = f"error: {type(exc).__name__}: {exc}"
    finally:
        if timer:
            timer.cancel()
    if session.cancelled():
        outcome = "timeout"
    problems = plan_check.check_plan(ui.plan, registry, set(state_facts() if state_facts else ())) if ui.plan else []
    result = grade(case, called, ui.plan, problems)
    if outcome == "timeout":
        result["passed"] = False
    result.update({"outcome": outcome, "questions": ui.questions, "usage": dict(session.usage),
                   "seconds": round(time.time() - started, 1), "plan": ui.plan})
    return result


def summary(results):
    n = len(results)
    if not n:
        return {"cases": 0}
    return {
        "cases": n,
        "passed": sum(r["passed"] for r in results),
        "pass_rate": round(sum(r["passed"] for r in results) / n, 3),
        "mean_coverage": round(sum(r["coverage"] for r in results) / n, 3),
        "with_plan": sum(r["has_plan"] for r in results),
        "plans_with_problems": sum(bool(r["plan_problems"]) for r in results),
        "questions": sum(r.get("questions", 0) for r in results),
    }
