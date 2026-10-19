"""Assistant tools analyze_protein_structure / prepare_protein and the Prepare Protein window."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
pytest.importorskip("pdbfixer")

from leonardo.agent.tools import AgentContext, build_registry  # noqa: E402

PREP = os.path.join(os.path.dirname(__file__), "data", "prep")


@pytest.fixture()
def env(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymol import cmd

    import leonardo.agent.presentation as pres

    cmd.reinitialize()
    cmd.load(os.path.join(PREP, "2LZT.pdb"), "lyz")
    ctx = AgentContext(dlg=None)

    def workspace(sub=""):
        path = os.path.join(str(tmp_path), sub)
        os.makedirs(path, exist_ok=True)
        return path

    ctx.workspace = workspace
    original = pres.save_step_scene
    pres.save_step_scene = lambda *a, **k: {"scene": ""}
    yield ctx, build_registry(ctx), cmd
    pres.save_step_scene = original
    del app


def test_registration(env):
    _ctx, reg, _cmd = env
    assert reg.get("analyze_protein_structure").permission == "read"
    spec = reg.get("prepare_protein")
    assert spec.permission == "compute_light" and spec.parameters["required"] == ["object"]


def test_analyze_and_prepare(env):
    ctx, reg, cmd = env
    a = reg.call("analyze_protein_structure", {"object": "lyz", "fetch_sequence": False}, ctx)
    assert a["chains"][0]["residues"] == 129 and a["waters"] > 100 and "_card" in a
    job = reg.call("prepare_protein", {"object": "lyz", "ph": 5.0, "minimize": False}, ctx)
    assert job["status"] == "running"
    t0 = time.time()
    while True:
        st = reg.call("job_status", {"job_id": job["job_id"]}, ctx)
        if st.get("status") != "running" or time.time() - t0 > 300:
            break
        time.sleep(0.5)
    assert st.get("status") == "done", st
    res = st["result"]
    assert res["object"] == "lyz_prep" and "lyz_prep" in cmd.get_names("objects")
    assert cmd.count_atoms("lyz_prep and hydro") > 900
    if res["tools"]["propka"]:
        assert "lyz_prep_protonated" in res["selections"].values()
        assert cmd.count_atoms("lyz_prep_protonated and resi 35") > 0       # Glu35 protonated at pH 5
    assert "Colours:" in res["_card"]
    with pytest.raises(Exception):
        reg.call("prepare_protein", {"object": "lyz"}, ctx)                 # lyz_prep exists already


def test_dialog_analyzes_and_lists_options(env, tmp_path):
    from leonardo.protein_prep_dialog import ProteinPrepDialog

    dlg = ProteinPrepDialog(None, workspace=lambda: str(tmp_path / "dlg"))
    assert dlg.cmb_object.findText("lyz") >= 0
    dlg._on_analyze()                                                       # synchronous without a task pool
    assert dlg._analysis is not None and dlg.tree_analysis.topLevelItemCount() >= 8
    dlg.txt_seq.setPlainText("A: KVFGRCELAAAMKRHGLDNYRG\nB = MKT AYI")
    assert dlg._sequences() == {"A": "KVFGRCELAAAMKRHGLDNYRG", "B": "MKTAYI"}
    dlg.close()
