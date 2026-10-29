import os
import time

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("gemmi")
from leonardo import cryptic_pocket_runtime as cp  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "prep")


@pytest.fixture(scope="module")
def lysozyme():
    return cp.read_chain(os.path.join(DATA, "2LZT.pdb"))


def test_read_chain_keeps_protein_heavy_atoms(lysozyme):
    assert lysozyme.n_res == 129 and len(lysozyme.ca_index) == 129
    assert lysozyme.res_keys[0] == "A_1" and lysozyme.res_names[0] == "LYS"


def test_sparse_hessian_matches_dense(lysozyme):
    ca = lysozyme.xyz[lysozyme.ca_index][:60]
    n = len(ca)
    d = ca[:, None, :] - ca[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", d, d)
    contact = (r2 < 15.0 ** 2) & ~np.eye(n, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        blk = -d[:, :, :, None] * d[:, :, None, :] / r2[:, :, None, None]
    blk[~contact] = 0.0
    dense = np.zeros((n, 3, n, 3))
    dense[:] = blk.transpose(0, 2, 1, 3)
    for i in range(n):
        dense[i, :, i, :] = -blk[i].sum(axis=0)
    assert np.abs(cp.anm_hessian(ca).toarray() - dense.reshape(3 * n, 3 * n)).max() < 1e-10
    vals, modes = cp.anm_modes(ca, n_modes=10)
    assert vals.shape == (10,) and modes.shape == (10, n, 3) and vals[0] > 1e-6


def test_conformers_have_the_requested_rmsd(lysozyme):
    vals, modes = cp.anm_modes(lysozyme.xyz[lysozyme.ca_index])
    for x in cp.conformers(lysozyme, vals, modes, n_conf=3, rmsd=2.0):
        d = x[lysozyme.ca_index] - lysozyme.xyz[lysozyme.ca_index]
        assert abs(np.sqrt((d ** 2).sum(axis=1).mean()) - 2.0) < 1e-6


@pytest.mark.skipif(not cp.have_extension(), reason="C++ extension not built (python -m leonardo.cryptic_build)")
def test_cpp_and_numpy_agree(lysozyme):
    a = cp.pocket_grid(lysozyme.xyz, lysozyme.atom_res, lysozyme.n_res)
    b = cp.pocket_grid(lysozyme.xyz, lysozyme.atom_res, lysozyme.n_res, force_numpy=True)
    assert len(a["points"]) == len(b["points"]) > 50
    assert np.allclose(np.sort(a["cluster_volume"]), np.sort(b["cluster_volume"]))
    assert np.abs(a["residue_volume"] - b["residue_volume"]).max() == 0.0
    assert np.abs(a["residue_score"] - b["residue_score"]).max() < 1e-3


def test_lysozyme_active_site_cleft_is_found(lysozyme):
    """Lysozyme's substrate cleft (Glu35, Asp52, Trp62/63, Asp101, Trp108) is a pocket in the structure."""
    res = cp.analyze(lysozyme, model="physics")
    prob = dict(zip(res["residues"], res["probability"]))
    assert all(0.0 <= p <= 1.0 for p in prob.values())
    cleft = ["A_35", "A_52", "A_62", "A_63", "A_101", "A_108"]
    others = [k for k in prob if k not in cleft]
    assert np.mean([prob[k] for k in cleft]) > np.percentile([prob[k] for k in others], 75)
    assert res["sites"] and set(cleft) & set(sum((s["residues"] for s in res["sites"][:3]), []))
    for s in res["sites"]:
        assert s["kind"] in ("already open", "grows with motion", "forms with motion")


def test_model_file_is_consistent():
    model = cp.load_model()
    assert set(model["features"]) <= set(cp.FEATURES)
    assert len(model["coef"]) == len(model["features"]) == len(model["mean"])
    assert "CryptoBench" in model["trained_on"]


def test_agent_tool(tmp_path):
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    cmd.load(os.path.join(DATA, "2LZT.pdb"), "lyz")
    ctx = AgentContext(dlg=None)

    def workspace(sub=""):
        path = os.path.join(str(tmp_path), sub)
        os.makedirs(path, exist_ok=True)
        return path

    ctx.workspace = workspace
    reg = build_registry(ctx)
    assert reg.get("find_cryptic_pockets").permission == "compute_light"
    job = reg.call("find_cryptic_pockets", {"object": "lyz", "top": 3, "model": "physics"}, ctx)
    t0 = time.time()
    while True:
        st = reg.call("job_status", {"job_id": job["job_id"]}, ctx)
        if st.get("status") != "running" or time.time() - t0 > 120:
            break
        time.sleep(0.2)
    assert st.get("status") == "done", st
    res = st["result"]
    assert res["sites"] and "_card" in res
    assert cmd.count_atoms(res["sites"][0]["selection"]) > 0
    del app


def test_agent_chain_cryptic_site_to_docking_box(tmp_path):
    """What the assistant does after find_cryptic_pockets: the site's docking_box goes to set_docking_box, and
    Leonardo's docking box is then centred on that site."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2
    from pymol import cmd

    from leonardo.agent.orchestrator import SYSTEM_PROMPT
    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    cmd.load(os.path.join(DATA, "2LZT.pdb"), "lyz")
    dlg = code_v2.ProtLigInteractDialog()
    ctx = AgentContext(dlg=dlg)

    def workspace(sub=""):
        path = os.path.join(str(tmp_path), sub)
        os.makedirs(path, exist_ok=True)
        return path

    ctx.workspace = workspace
    reg = build_registry(ctx)
    job = reg.call("find_cryptic_pockets", {"object": "lyz", "model": "physics"}, ctx)
    t0 = time.time()
    while True:
        st = reg.call("job_status", {"job_id": job["job_id"]}, ctx)
        if st.get("status") != "running" or time.time() - t0 > 120:
            break
        app.processEvents()
        time.sleep(0.2)
    site = st["result"]["sites"][0]
    box = site["docking_box"]
    assert all(16.0 <= s <= 26.0 for s in box["size"])
    reg.call("set_docking_box", {"center": box["center"], "size": box["size"]}, ctx)
    got = [dlg.spin_cx.value(), dlg.spin_cy.value(), dlg.spin_cz.value()]
    assert np.allclose(got, box["center"], atol=0.01) and dlg._dock_grid_user_defined
    # the assistant is told about the tools and the chain
    for name in ("find_cryptic_pockets", "prepare_protein", "ramachandran_plot", "check_simulation_system",
                 "analyze_interface", "run_enhanced_sampling", "docking_box"):
        assert name in SYSTEM_PROMPT, name
    dlg.close()


def test_crypsis_copy_and_its_models():
    from leonardo.crypsis import __version__, descriptors, pipeline

    assert __version__ == "0.2.0"
    n = 22 + len(descriptors.DESCRIPTORS) + len(descriptors.ENSEMBLE)
    assert pipeline._load("physics").n_features == n
    for esm, pcs in (("650M", 256), ("150M", 128)):
        hyb = pipeline._load(f"hybrid_{esm}")
        assert hyb.n_features == n + pcs and hyb.extra["pca_components"].shape[1] == pipeline.ESM_MODELS[esm][2]


def test_physics_and_v1_models(lysozyme):
    res = cp.analyze(lysozyme, model="physics")
    assert res["model"] == "physics" and res["esm"] is None and "0.864" in res["benchmark"]
    assert abs((res["probability"] >= res["threshold"]).mean() - 0.1) < 0.02      # the top 10% are called
    v1 = cp.analyze(lysozyme, model="v1")
    assert v1["model"] == "v1" and v1["threshold"] == cp.THRESHOLD
    with pytest.raises(ValueError):
        cp.analyze(lysozyme, model="nonsense")


def test_agent_tool_asks_for_fair_esm_before_the_hybrid(tmp_path):
    if cp.have_esm():
        pytest.skip("fair-esm is installed here")
    pytest.importorskip("pymol.Qt")
    from pymol import cmd

    from leonardo.agent.registry import ToolError
    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    cmd.load(os.path.join(DATA, "2LZT.pdb"), "lyz")
    ctx = AgentContext(dlg=None)
    ctx.workspace = lambda sub="": str(tmp_path)
    reg = build_registry(ctx)
    with pytest.raises(ToolError, match="fair-esm"):
        reg.call("find_cryptic_pockets", {"object": "lyz", "model": "hybrid"}, ctx)


def test_auto_model_follows_the_computer(monkeypatch):
    monkeypatch.setattr(cp, "have_esm", lambda: True)
    for mem, expected in ((32.0, ("hybrid", "650M")), (17.2, ("hybrid", "650M")), (8.6, ("hybrid", "150M")),
                          (4.0, ("physics", "650M"))):
        monkeypatch.setattr(cp, "total_memory_gb", lambda m=mem: m)
        assert cp.default_model() == expected, mem
    monkeypatch.setattr(cp, "have_esm", lambda: False)
    monkeypatch.setattr(cp, "total_memory_gb", lambda: 64.0)
    assert cp.default_model() == ("physics", "650M")
