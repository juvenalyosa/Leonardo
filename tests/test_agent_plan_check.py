"""Prerequisite check of the assistant's plans: missing steps go back to the model before the user sees them."""
import pytest

from leonardo.agent import plan_check as PC
from leonardo.agent.orchestrator import AgentSession
from leonardo.agent.providers.fake import ScriptedProvider


@pytest.fixture(scope="module")
def registry():
    from leonardo.agent.tools import AgentContext, _headless, build_registry

    _headless.install()
    return build_registry(AgentContext())


def _plan(*steps):
    return {"goal": "g", "steps": [{"title": f"s{i}", "detail": "", "tools": list(t)} for i, t in enumerate(steps, 1)]}


def test_every_tool_named_by_the_rules_is_registered(registry):
    names = set(registry.names())
    named = {t for _d, tools in PC.FACTS.values() for t in tools} | set(PC.ALSO_PRODUCES) | set(PC.REQUIRES)
    assert named <= names, sorted(named - names)
    assert {f for need in PC.REQUIRES.values() for f in need} | \
        {f for made in PC.ALSO_PRODUCES.values() for f in made} <= set(PC.FACTS)


def test_complete_docking_plan_passes(registry):
    plan = _plan(["fetch_structure"], ["prepare_protein"], ["find_pockets", "select_pocket"],
                 ["set_ligand_from_smiles", "prepare_docking_ligand"], ["run_docking"], ["analyze_interactions"],
                 ["show_2d_interaction_map"])
    assert PC.check_plan(plan, registry) == []


def test_docking_without_pockets_or_ligand_names_the_missing_tools(registry):
    problems = PC.check_plan(_plan(["fetch_structure"], ["run_docking"]), registry)
    text = " ".join(problems)
    assert len(problems) == 3 and "find_pockets" in text and "select_pocket or set_docking_box" in text \
        and "prepare_docking_ligand" in text


def test_cryptic_site_docking_needs_fpockets_pocket_list(registry):
    plan = _plan(["find_cryptic_pockets"], ["set_docking_box"], ["prepare_docking_ligand"], ["run_docking"])
    problems = PC.check_plan(plan, registry, facts={"structure"})
    assert len(problems) == 1 and "find_pockets" in problems[0]


def test_session_facts_and_order(registry):
    # already done in the session: nothing to report
    assert PC.check_plan(_plan(["run_docking"]), registry,
                         facts={"structure", "pockets_found", "docking_box", "ligand"}) == []
    # analysis before the MD that makes the trajectory
    problems = PC.check_plan(_plan(["trajectory_rmsd_rmsf"], ["run_md"]), registry, facts={"structure"})
    assert len(problems) == 1 and "Step 1" in problems[0] and "run_md" in problems[0]
    # an object-taking tool with nothing loaded
    assert "fetch_structure" in PC.check_plan(_plan(["find_pockets"]), registry)[0]


def test_unknown_tools_are_reported_and_steps_without_tools_pass(registry):
    problems = PC.check_plan(_plan(["dock_everything"]), registry)
    assert problems and "not a Leonardo tool" in problems[0]
    assert PC.check_plan({"goal": "g", "steps": [{"title": "t", "detail": "d"}]}, registry) == []


class UI:
    def __init__(self):
        self.asked = []

    def on_event(self, kind, payload):
        pass

    def ask(self, kind, payload):
        self.asked.append((kind, payload))
        return {"decision": "reject"}


def test_session_sends_findings_back_once_then_shows_them(registry):
    bad = _plan(["fetch_structure"], ["run_docking"])
    good = _plan(["fetch_structure"], ["find_pockets", "select_pocket"], ["prepare_docking_ligand"], ["run_docking"])

    # fixed after the feedback: the user only sees the revised plan, without warnings
    ui = UI()
    prov = ScriptedProvider([[("tool", "propose_plan", bad)], [("tool", "propose_plan", good)]])
    sess = AgentSession(prov, registry, ui, execute=lambda n, a: {}, state_facts=lambda: set())
    sess.send("dock ibuprofen into 1ABC")
    feedback = next(entry[1][0]["content"] for entry in prov.transcript if entry[0] == "tool_results")
    assert "Plan check" in feedback and "find_pockets" in feedback
    assert len(ui.asked) == 1 and ui.asked[0][1]["plan"] == good and "checks" not in ui.asked[0][1]["plan"]

    # kept as it is: the user sees it, with the findings
    ui = UI()
    prov = ScriptedProvider([[("tool", "propose_plan", bad)], [("tool", "propose_plan", bad)]])
    sess = AgentSession(prov, registry, ui, execute=lambda n, a: {}, state_facts=lambda: set())
    sess.send("dock ibuprofen into 1ABC")
    assert len(ui.asked) == 1 and len(ui.asked[0][1]["plan"]["checks"]) == 3
