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
