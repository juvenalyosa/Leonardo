"""Membrane systems: orientation (OPM / predicted TM segments), bilayer build, metrics, equilibration, tools."""
import io
import os
import time

import numpy as np
import pytest

app = pytest.importorskip("openmm.app")
unit = pytest.importorskip("openmm.unit")

from leonardo import membrane_runtime as mr  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "components")
HELIX = os.path.join(DATA, "tm_helix_tilted.pdb")


def _ca(top, pos):
    return np.array([pos[a.index] for a in top.atoms() if a.name == "CA"])


def test_tm_helix_is_oriented_along_z_and_centred():
    pdb = app.PDBFile(HELIX)
    pos = [tuple(p.value_in_unit(unit.nanometer)) for p in pdb.positions]
    new, info = mr.orient_by_tm_segments(pdb.topology, pos)
    ca = _ca(pdb.topology, new)
    axis = np.linalg.svd(ca - ca.mean(axis=0))[2][0]
    assert abs(axis[2]) > 0.98                                     # helix axis along the bilayer normal
    assert abs(ca.mean(axis=0)[2]) < 0.2 and info["segments"]


def test_orientation_refuses_soluble_proteins():
    pdb = app.PDBFile(os.path.join(DATA, "3KS3_zn_site.pdb"))
    pos = [tuple(p.value_in_unit(unit.nanometer)) for p in pdb.positions]
    with pytest.raises(ValueError, match="membrane protein"):
        mr.orient_by_tm_segments(pdb.topology, pos)


def test_fetch_opm_strips_dummy_membrane_atoms(tmp_path, monkeypatch):
    text = ("REMARK      1/2 of bilayer thickness:   15.0\n"
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
            "HETATM    2  O   DUM     1       0.000   0.000  15.000\n")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp(text.encode()))
    got = mr.fetch_opm("1abc", str(tmp_path / "o.pdb"))
    assert got["half_thickness_nm"] == 1.5 and "DUM" not in open(got["path"]).read()
    assert mr.fetch_opm("bad id!", str(tmp_path / "x.pdb")) is None


def test_build_popc_membrane_around_tm_helix(tmp_path):
    rep = mr.build_membrane_system(HELIX, lipid="POPC", family="amber14", orientation="tm", padding_nm=0.8,
                                   minimize_steps=50, out_dir=str(tmp_path))
    assert rep["ok"], rep.get("error")
    mem = rep["membrane"]
    assert mem["n_lipids"] > 20 and 3.3 < mem["thickness_PP_nm"] < 4.1
    assert 0.55 < mem["area_per_lipid_nm2"] < 0.80                  # voids next to the protein close on equilibration
    assert rep["tm_calpha_in_core_fraction"] > 0.6                   # helix ends stick out of the +-1.5 nm core
    assert abs(rep["total_charge"] - round(rep["total_charge"])) < 0.01
    eq = mr.equilibrate_membrane(rep["pdb"], rep["system_xml"], str(tmp_path / "eq"), stages_ps=(0.2, 0.2),
                                 restraint_k=(1000, 100), report_every_ps=0.1)
    assert eq["ok"] and eq["frames"] == 4 and os.path.isfile(eq["trajectory"])


def test_agent_membrane_tool_runs_in_the_background(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert qt_app is not None
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    cmd.load(HELIX, "helix")
    ctx = AgentContext(dlg=None)

    def ws(sub=""):
        p = tmp_path / sub
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    ctx.workspace = ws
    reg = build_registry(ctx)
    job = reg.call("build_membrane_system", {"object": "helix", "orientation": "tm", "padding_A": 8}, ctx)
    deadline = time.time() + 900
    status = reg.call("job_status", {"job_id": job["job_id"]}, ctx)
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(2)
        status = reg.call("job_status", {"job_id": job["job_id"]}, ctx)
    assert status["status"] == "done", status
    assert "helix_popc" in cmd.get_names("objects")
    assert status["result"]["membrane"]["n_lipids"] > 20


def test_parse_lipid_spec_mixtures_and_leaflets():
    assert mr.parse_lipid_spec("popc:chl1", "3:1") == ("POPC:CHL1", "3:1", ["POPC", "CHL1"])
    assert mr.parse_lipid_spec("POPC:CHL1//POPE") == ("POPC:CHL1//POPE", "1:1//1", ["POPC", "CHL1", "POPE"])
    assert mr.is_mixture("POPE:POPG") and mr.is_mixture("PSM") and not mr.is_mixture("POPC")
    with pytest.raises(ValueError, match="does not match"):
        mr.parse_lipid_spec("POPC:CHL1", "3:1:1")
    with pytest.raises(ValueError, match="positive"):
        mr.parse_lipid_spec("POPC:CHL1", "3:0")


@pytest.mark.skipif(mr._packmol_memgen()[0] is None, reason="PACKMOL-Memgen not installed")
def test_build_popc_cholesterol_mixture_charmm(tmp_path):
    rep = mr.build_membrane_system(HELIX, lipid="POPC:CHL1", lipid_ratio="3:1", family="charmm36",
                                   orientation="tm", padding_nm=0.8, water_nm=1.2, minimize_steps=100,
                                   out_dir=str(tmp_path), seed=11)
    assert rep["ok"], rep.get("error")
    counts = rep["packing"]["counts"]
    assert set(counts) == {"POPC", "CHL1"} and 2.0 <= counts["POPC"] / counts["CHL1"] <= 4.5
    assert abs(rep["total_charge"]) < 1e-3 and rep["packing"]["ions"]["NA"] > 0
    assert 3.3 < rep["membrane"]["thickness_PP_nm"] < 4.6
    assert rep["tm_calpha_in_core_fraction"] >= 0.6
    assert rep["energy_minimized_kj"] < 0 and os.path.isfile(rep["cif"])
    # the 4-letter lipid names survive in mmCIF; PDB keeps CHL/POP and the metrics still find them
    pdb = app.PDBFile(rep["pdb"])
    m = mr.membrane_metrics(pdb.topology, [tuple(p) for p in pdb.positions.value_in_unit(unit.nanometer)],
                            rep["box_nm"], lipid_names=rep["lipid_residue_names"])
    assert m["n_lipids"] == sum(counts.values())
