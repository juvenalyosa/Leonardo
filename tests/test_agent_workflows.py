"""Workflow recipes (suggest_workflow) and the session-state summary sent with each request."""
import os

import pytest

from leonardo.agent import plan_check, plan_eval, session_state, workflows
from leonardo.agent.orchestrator import SYSTEM_PROMPT, AgentSession
from leonardo.agent.providers.fake import ScriptedProvider

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def registry():
    from leonardo.agent.tools import AgentContext, _headless, build_registry

    _headless.install()
    return build_registry(AgentContext())


def test_recipes_use_registered_tools_and_pass_the_plan_check(registry):
    names = set(registry.names())
    assert len({r[0] for r in workflows.RECIPES}) == len(workflows.RECIPES)
    for rid, *_rest in workflows.RECIPES:
        r = workflows.recipe(rid)
        used = {t for s in r["steps"] for t in s["tools"]}
        assert used <= names, (rid, sorted(used - names))
        assert plan_check.check_plan(workflows.as_plan(rid), registry, facts={"structure"}) == [], rid


@pytest.mark.parametrize("goal, expected", [
    ("Haz docking de ibuprofeno en la COX-2", "dock_ligand"),
    ("¿Dónde podría unirse un fármaco que no se vea? Luego haz docking", "dock_cryptic_site"),
    ("Redock the crystal ligand and check the RMSD", "redock_validation"),
    ("Screen this library of compounds against the pocket", "virtual_screening"),
    ("Corre una dinámica y dime si la proteína es estable", "md_stability"),
    ("How often is the salt bridge kept during MD?", "md_complex_contacts"),
    ("Estima la energía libre de unión", "binding_free_energy"),
    ("Mete el receptor en una membrana de POPC", "membrane_protein"),
    ("Run umbrella sampling of the unbinding", "enhanced_sampling"),
    ("Muéstrame el HOMO y el LUMO de la cafeína con DFT", "qm_orbitals"),
    ("Espectro UV-vis del benceno con TDDFT", "uv_spectrum"),
    ("Modela la reacción en el sitio activo con un clúster QM", "cluster_reaction"),
    ("Pon la proteína sobre una superficie de oro", "protein_on_surface"),
    ("Calcula el pKa del ibuprofeno", "ligand_pka"),
])
def test_goals_find_their_recipe(goal, expected):
    found = [r["id"] for r in workflows.match(goal)]
    assert found and found[0] == expected, (goal, found)


def test_tool_answers_with_recipes_or_says_none_fits(registry):
    out = registry.call("suggest_workflow", {"goal": "dock aspirin into COX-1"}, None)
    assert out["recipes"][0]["id"] == "dock_ligand" and out["recipes"][0]["steps"][1]["tools"] == ["find_pockets"]
    assert registry.call("suggest_workflow", {"goal": "zzz"}, None)["recipes"] == []
    assert registry.get("suggest_workflow").planning_allowed


def test_session_state_describes_objects_and_docking_state():
    from pymol import cmd

    cmd.delete("all")
    assert "none loaded" in session_state.describe() and "nothing set" in session_state.describe()
    cmd.load(os.path.join(ROOT, "examples", "3ptb_ben", "3PTB.pdb"), "3PTB")
    try:
        class Dlg:
            _fpocket_pockets = [{"id": 1}]
            _docking_ligand_file = "/tmp/x/ben.sdf"

        text = session_state.describe(Dlg())
        assert "3PTB (" in text and "chains A" in text and "BEN" in text and "waters" in text
        assert "fpocket pocket list" in text and "active ligand (ben.sdf)" in text
    finally:
        cmd.delete("all")


def test_session_sends_the_state_before_the_request_and_ab_prompt():
    prov = ScriptedProvider([["ok"]])

    class UI:
        def on_event(self, *a):
            pass

    sess = AgentSession(prov, plan_check_registry(), UI(), execute=lambda n, a: {},
                        state_summary=lambda: "Session state (automatic, at this request):\n- objects: 1UBQ")
    sess.send("hola")
    assert prov.transcript[0] == ("user", "Session state (automatic, at this request):\n- objects: 1UBQ\n\nhola")
    base = plan_eval.system_prompt(recipes=False, state=False)
    assert "suggest_workflow" in SYSTEM_PROMPT and "suggest_workflow" not in base and len(base) < len(SYSTEM_PROMPT)


def plan_check_registry():
    from leonardo.agent.registry import ToolRegistry

    return ToolRegistry()


@pytest.mark.parametrize("goal, wrong", [
    ("Muestra la tripsina como superficie y centra la vista", "protein_on_surface"),
    ("Build a homology model and validate it", "qm_orbitals"),
    ("UV-vis spectrum with TDDFT", "qm_orbitals"),
])
def test_ambiguous_words_do_not_pick_the_wrong_recipe(goal, wrong):
    assert wrong not in [r["id"] for r in workflows.match(goal)]
