"""Planning benchmark: case file, grading and one case driven by a scripted provider (no network)."""
import pytest

from leonardo.agent import plan_eval as PE
from leonardo.agent.providers.fake import ScriptedProvider


@pytest.fixture(scope="module")
def registry():
    from leonardo.agent.tools import AgentContext, _headless, build_registry

    _headless.install()
    return build_registry(AgentContext())


def test_cases_name_only_registered_tools_and_load_existing_scenes(registry):
    import os

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    names = set(registry.names())
    cases = PE.load_cases()
    assert len(cases) >= 25 and len({c["id"] for c in cases}) == len(cases)
    for case in cases:
        e = case["expect"]
        named = {t for g in e.get("all", []) for t in g} | {t for p in e.get("order", []) for t in p} \
            | set(e.get("none", [])) | set(e.get("prefer", []))
        assert named <= names, (case["id"], sorted(named - names))
        for rel in case.get("scene") or ():
            assert os.path.isfile(os.path.join(root, rel)), rel


def test_grade_groups_order_and_unwanted():
    case = {"id": "c", "expect": {"all": [["a"], ["b", "c"]], "order": [["a", "b"]], "none": ["x"], "prefer": ["p"]}}
    plan = {"steps": [{"tools": ["b"]}, {"tools": ["a"]}]}
    g = PE.grade(case, [], plan)
    assert not g["passed"] and g["bad_order"] == [["a", "b"]] and g["coverage"] == 1.0
    g = PE.grade(case, ["a"], {"steps": [{"tools": ["c", "p"]}]})
    assert g["passed"] and g["prefer_hit"] == "1/1"
    assert not PE.grade(case, [], {"steps": [{"tools": ["a", "b", "x"]}]})["passed"]
    assert not PE.grade(case, ["a", "b"], None)["passed"]                     # no plan
    assert PE.grade(dict(case, needs_plan=False), ["a", "b"], None)["passed"]  # read-only answer is fine


def test_run_case_captures_the_plan_and_runs_nothing_heavy(registry):
    case = {"id": "dock", "request": "dock ibuprofen into 5KIR",
            "expect": {"all": [["fetch_structure"], ["find_pockets"], ["run_docking"]], "none": ["run_md"]}}
    plan = {"goal": "dock", "steps": [
        {"title": "load", "detail": "", "tools": ["fetch_structure"]},
        {"title": "pockets", "detail": "", "tools": ["find_pockets", "select_pocket"]},
        {"title": "ligand", "detail": "", "tools": ["set_ligand_from_smiles"]},
        {"title": "dock", "detail": "", "tools": ["run_docking"]}]}
    executed = []
    prov = ScriptedProvider([[("tool", "list_objects", {})], [("tool", "ask_user", {"question": "pH?"})],
                             [("tool", "propose_plan", plan)]])
    result = PE.run_case(case, prov, registry, lambda n, a: executed.append(n) or {"objects": []},
                         state_facts=lambda: set())
    assert executed == ["list_objects"]          # the read-only look-up ran; nothing after the rejected plan
    assert result["passed"] and result["questions"] == 1 and result["plan_problems"] == []
    assert result["tools"][:2] == ["list_objects", "fetch_structure"]
    assert PE.summary([result])["pass_rate"] == 1.0
