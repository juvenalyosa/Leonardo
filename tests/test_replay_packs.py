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
                         capture_output=True, text=True, env=env, timeout=900)
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
