"""Replay packs: re-running a recorded pipeline without the AI gives the recorded answer."""
import json
import os
import shutil
import subprocess
import sys

import pytest

from leonardo import replay

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(REPO, "examples", "films", "01_molecular_docking")


def test_value_paths_and_checks():
    results = {2: {"repeats": [{"top1_rmsd_A": 0.39}], "top1_success": "3/3 repeats"}}
    assert replay._value_at(results, "2.repeats.0.top1_rmsd_A") == 0.39
    with pytest.raises(KeyError):
        replay._value_at(results, "2.missing")
    assert replay.check({"kind": "abs", "value": 0.39, "tol": 0.15}, 0.50)
    assert not replay.check({"kind": "abs", "value": 0.39, "tol": 0.05}, 0.50)
    assert replay.check({"kind": "exact", "value": "3/3 repeats"}, "3/3 repeats")
    assert not replay.check({"kind": "exact", "value": 20260915}, 20260916)
    assert replay.check({"kind": "max", "value": 2.0}, 1.2) and not replay.check({"kind": "max", "value": 2.0}, 2.5)


def test_film08_pack_asks_for_a_reproducible_seeded_ensemble():
    pipe = json.load(open(os.path.join(REPO, "examples", "films", "08_ensemble_redocking", "pipeline.json")))
    args = pipe["steps"][1]["args"]
    assert args["ensemble_structures"] == 3 and args["ensemble_seed"] == 1 and args["ensemble_reproducible"] is True
    exp = json.load(open(os.path.join(REPO, "examples", "films", "08_ensemble_redocking", "expected.json")))
    assert all(m.get("value") is not None for m in exp["metrics"].values())


def test_redocking_tool_passes_the_ensemble_seed_and_reproducible_mode():
    from leonardo.agent.registry import ToolRegistry
    from leonardo.agent.tools import redocking

    reg = ToolRegistry()
    redocking.register(reg)
    props = {s.name: s for s in reg.specs()}["run_redocking"].parameters["properties"]
    assert props["ensemble_seed"]["type"] == "integer" and props["ensemble_reproducible"]["type"] == "boolean"


def test_film01_pack_is_complete_and_seeded():
    pipe = json.load(open(os.path.join(PACK, "pipeline.json")))
    exp = json.load(open(os.path.join(PACK, "expected.json")))
    assert [s["tool"] for s in pipe["steps"]] == ["load_structure_file", "run_redocking"]
    assert pipe["steps"][1]["args"]["seed"] == 20260915
    assert all(m.get("value") is not None for m in exp["metrics"].values())
    assert os.path.isfile(os.path.join(PACK, "3PTB.pdb"))


@pytest.mark.skipif(not (shutil.which("smina") or os.path.isfile(os.path.join(sys.prefix, "bin", "smina"))),
                    reason="smina not installed")
def test_film01_pack_verifies(tmp_path):
    pytest.importorskip("pymol")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(REPO, "src"), os.path.join(REPO, "legacy")]),
               QT_QPA_PLATFORM="offscreen")
    report = tmp_path / "report.json"
    out = subprocess.run([sys.executable, "-m", "leonardo.replay", "verify", PACK, "--report", str(report)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=900)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert json.load(open(report))["ok"]


def test_list_reducers_do_not_depend_on_order():
    results = {2: {"poses": [{"rmsd_A": 0.6}, {"rmsd_A": 0.48}, {"rmsd_A": 12.4}]}}
    assert replay._value_at(results, "2.poses.*.rmsd_A") == [0.6, 0.48, 12.4]
    assert replay._value_at(results, "2.poses.*.rmsd_A|max") == 12.4
    assert replay._value_at(results, "2.poses.*.rmsd_A|min") == 0.48
    assert replay._value_at(results, "2.poses.*.rmsd_A|count_le:2.0") == 2
    assert replay._value_at(results, "2.poses.*.rmsd_A|count") == 3
    assert replay._value_at(results, "2.poses.*.rmsd_A|sorted") == [0.48, 0.6, 12.4]


def test_trajectory_distance_measures_every_frame():
    pytest.importorskip("pymol")
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    cmd.fragment("ser", "traj")
    cmd.create("traj", "traj", 1, 2)
    cmd.translate([0.0, 0.0, 1.0], "traj and name OG", state=2, camera=0)
    ctx = AgentContext(None)
    reg = build_registry(ctx)
    d1 = cmd.get_distance("traj and name OG", "traj and name CA", state=1)
    res = reg.call("trajectory_distance", {"object": "traj", "atom_a": "name OG", "atom_b": "name CA",
                                           "cutoff_A": 2.6}, ctx)
    assert res["frames"] == 2 and abs(res["distance_A"][0] - round(d1, 2)) < 1e-6
    assert res["max_A"] > res["min_A"] and res["fraction_within_cutoff"] == 0.5
    with pytest.raises(Exception):
        reg.call("trajectory_distance", {"object": "traj", "atom_a": "name C*", "atom_b": "name CA"}, ctx)


def test_count_reduces_a_plain_list():
    results = {2: {"scan": {"rows": [{"e": 1}, {"e": 2}, {"e": 3}]}}}
    assert replay._value_at(results, "2.scan.rows|count") == 3


def test_count_of_a_null_list_is_zero():
    # was 1: [None] was counted as one item, so '2.designs|count' passed an "exact 1" check with no designs
    assert replay._value_at({2: {"designs": None}}, "2.designs|count") == 0
    assert replay._value_at({2: {"designs": []}}, "2.designs|count") == 0


def test_model_mutant_can_keep_hetero_groups():
    from leonardo.agent.registry import ToolRegistry
    from leonardo.agent.tools import modeling

    reg = ToolRegistry()
    modeling.register(reg)
    props = {s.name: s for s in reg.specs()}["model_mutant"].parameters["properties"]
    assert props["keep_hetero"]["type"] == "array"
    pytest.importorskip("pymol")
    from pymol import cmd

    cmd.reinitialize()
    cmd.load(os.path.join(REPO, "examples", "films", "02_dhla_reaction_qmmm", "2DHC.pdb"), "wt")
    cmd.create("mut", "wt and polymer")
    copier = modeling._hetero_copier("wt", ["DCE"])
    changed, post = copier("mut")
    assert changed and post["components_added"] and cmd.count_atoms("mut and resn DCE") == 4
    with pytest.raises(Exception):
        modeling._hetero_copier("wt", ["XYZ"])


def _fake_pack(tmp_path, metrics):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "expected.json").write_text(json.dumps({"metrics": metrics}))
    return pack


def test_record_fails_on_a_missing_metric_and_keeps_its_reference(tmp_path, monkeypatch):
    pack = _fake_pack(tmp_path, {"rmsd": {"path": "1.rmsd_A", "value": 0.4, "tol": 0.1},
                                 "poses": {"path": "1.poses.*.rmsd_A|max", "value": 0.5, "tol": 0.1}})
    monkeypatch.setattr(replay, "run_pipeline", lambda pack, work: ({1: {"poses": []}}, []))
    monkeypatch.setattr(replay, "environment", lambda: {"platform": "test", "leonardo": "test"})
    rc = replay.main(["record", str(pack), "--report", str(tmp_path / "r.json")])
    assert rc == 1                                            # was 0, "ALL CHECKS PASSED"
    exp = json.load(open(pack / "expected.json"))["metrics"]
    assert exp["rmsd"]["value"] == 0.4 and exp["poses"]["value"] == 0.5     # not overwritten with null


def test_verify_reports_an_uncomparable_metric_instead_of_crashing(tmp_path, monkeypatch):
    pack = _fake_pack(tmp_path, {"success": {"path": "1.top1_success", "value": 3, "tol": 0},
                                 "mean": {"path": "1.poses.*.rmsd_A|mean", "value": 0.5, "tol": 0.1}})
    monkeypatch.setattr(replay, "run_pipeline", lambda pack, work: ({1: {"top1_success": "3/3 repeats",
                                                                         "poses": []}}, []))
    monkeypatch.setattr(replay, "environment", lambda: {"platform": "test", "leonardo": "test"})
    assert replay.main(["verify", str(pack), "--report", str(tmp_path / "r.json")]) == 1
    assert json.load(open(tmp_path / "r.json"))["ok"] is False


def test_replay_stops_at_a_cancelled_job(tmp_path, monkeypatch):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pipeline.json").write_text(json.dumps({"name": "p", "parameters": {},
                                                    "steps": [{"tool": "run_md", "args": {}}]}))

    class Jobs:
        def status(self, job_id):
            return {"job_id": job_id, "status": "cancelled"}

    class Ctx:
        jobs = Jobs()

    class App:
        def processEvents(self):
            pass

    class Reg:
        def call(self, name, args, ctx):
            return {"job_id": "md-1", "status": "running"}

    monkeypatch.setattr(replay, "_headless_leonardo", lambda work: (App(), Ctx(), Reg()))
    with pytest.raises(RuntimeError, match=r"step 1 \(run_md\) cancelled"):
        replay.run_pipeline(str(pack), str(tmp_path))


def test_local_mutant_keeps_every_other_atom_in_place(tmp_path):
    pytest.importorskip("pdbfixer")
    pytest.importorskip("pymol")
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    cmd.load(os.path.join(REPO, "examples", "films", "02_dhla_reaction_qmmm", "2DHC.pdb"), "wt")
    ctx = AgentContext(None)
    ctx.workspace = lambda sub="": str(tmp_path / sub) if os.makedirs(tmp_path / sub, exist_ok=True) is None else ""
    reg = build_registry(ctx)
    res = reg.call("model_mutant", {"object": "wt", "mutations": ["W125A"], "chain": "A", "method": "local",
                                    "name": "w125a"}, ctx)
    assert res["max_shift_outside_mutation_A"] < 1e-3 and res["unchanged_atoms"] > 2000
    assert cmd.count_atoms("w125a and resi 125 and resn ALA and name CB") == 1
    assert cmd.count_atoms("w125a and resi 125 and name NE1") == 0
    assert cmd.count_atoms("w125a and resn DCE") == 4 and cmd.count_atoms("w125a and resn HOH") > 0
    with pytest.raises(Exception):
        reg.call("model_mutant", {"object": "wt", "mutations": ["G125A"], "chain": "A", "method": "local"}, ctx)


def test_local_mutant_of_a_structure_with_alternate_locations(tmp_path):
    # the mutant is built from alt A only; the check compared it with alt B and rejected every such structure
    # ("changed atoms outside the mutation (max shift 0.400 A)")
    pytest.importorskip("pdbfixer")
    pytest.importorskip("pymol")
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    alt = tmp_path / "1pga_alt.pdb"
    with open(alt, "w") as fh:
        for ln in open(os.path.join(REPO, "examples", "films", "06_protein_stability_design", "1PGA.pdb")):
            if ln.startswith("ATOM") and ln[22:26].strip() == "20":
                fh.write(ln[:16] + "A" + ln[17:54] + "  0.50" + ln[60:])
                fh.write(ln[:16] + "B" + ln[17:30] + f"{float(ln[30:38]) + 0.4:8.3f}" + ln[38:54] + "  0.50" + ln[60:])
            else:
                fh.write(ln)
    cmd.reinitialize()
    cmd.load(str(alt), "wt")
    assert cmd.count_atoms("wt and alt B") > 0
    ctx = AgentContext(None)
    ctx.workspace = lambda sub="": str(tmp_path / sub) if os.makedirs(tmp_path / sub, exist_ok=True) is None else ""
    reg = build_registry(ctx)
    res = reg.call("model_mutant", {"object": "wt", "mutations": ["K10A"], "chain": "A", "method": "local",
                                    "name": "k10a"}, ctx)
    assert res["max_shift_outside_mutation_A"] < 1e-3
    assert cmd.count_atoms("k10a and resi 10 and resn ALA and name CB") == 1
