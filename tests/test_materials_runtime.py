"""Materials and hybrid systems: metal slabs, nanoparticles, graphene, a protein on Au(111), QM clusters."""
import os
import time

import numpy as np
import pytest

app = pytest.importorskip("openmm.app")
unit = pytest.importorskip("openmm.unit")
pytest.importorskip("ase")

from leonardo import materials_runtime as mat  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "components")
UBQ = os.path.join(DATA, "1UBQ.pdb")


def test_au111_slab_geometry():
    slab = mat.build_material("Au", "slab", "111", size_nm=(3, 3), layers=4)
    P = slab["positions_nm"]
    assert abs(P[:, 2].max()) < 1e-9 and slab["layers"] == 4
    layers = sorted(set(np.round(P[:, 2], 3)))
    assert len(layers) == 4 and abs((layers[-1] - layers[-2]) - 0.2356) < 0.002    # Au(111) spacing a/sqrt(3)
    per_layer = slab["n_atoms"] / 4
    area = slab["cell_nm"][0] * slab["cell_nm"][1]
    assert abs(per_layer / area - 13.87) < 0.1                                       # atoms/nm^2 of Au(111)


def test_graphene_and_nanoparticle():
    g = mat.build_material("graphene", size_nm=(3, 3))
    assert abs(g["n_atoms"] / (g["cell_nm"][0] * g["cell_nm"][1]) - 38.2) < 0.2
    assert mat.build_material("Au", "nanoparticle", shells=3)["n_atoms"] == 55
    assert mat.build_material("Au", "nanoparticle", shells=4)["n_atoms"] == 147
    with pytest.raises(ValueError):
        mat.build_material("Unobtainium")


def test_interface_ff_parameters_load_and_match_heinz():
    xml = mat.material_forcefield_xml("Au", ["Au"])
    import io

    ff = app.ForceField(io.StringIO(xml))
    assert "MAU" in ff._templates
    sigma, eps, _a = mat.METAL_LJ["Au"]
    assert abs(sigma * 2 ** (1 / 6) - 0.2951) < 1e-3 and abs(eps / 4.184 - 5.29) < 0.01   # r0 2.951 A, 5.29 kcal/mol


def test_protein_on_gold_with_site_facing_solution(tmp_path):
    rep = mat.build_protein_surface_system(UBQ, material="Au", facet="111", layers=3, away_from_surface="A:44",
                                           water_above_nm=1.0, minimize_steps=100, out_dir=str(tmp_path))
    assert rep["ok"], rep.get("error") or rep.get("warnings")
    sa = rep["surface_analysis"]
    assert sa["site_height_above_surface_nm"] > 1.5 and sa["protein_surface_LJ_kj_mol"] < 0
    pdb = app.PDBFile(rep["pdb"])
    X = np.array(pdb.positions.value_in_unit(unit.nanometer))
    au = [a.index for a in pdb.topology.atoms() if a.residue.name == "MAU"]
    ow = [a.index for a in pdb.topology.atoms() if a.residue.name == "HOH" and a.name == "O"]
    zau = X[au, 2]
    assert not ((X[ow, 2] > zau.min() - 0.05) & (X[ow, 2] < zau.max() + 0.05)).any()        # no water in the metal
    import openmm as mm

    with open(rep["system_xml"]) as fh:
        system = mm.XmlSerializer.deserialize(fh.read())
    assert all(system.getParticleMass(i).value_in_unit(unit.dalton) == 0 for i in au[:50])  # frozen substrate
    cluster = mat.cut_qm_cluster(rep["pdb"], "A:32", radius_nm=0.45)
    assert cluster["composition"].get("Au", 0) > 0 and os.path.isfile(cluster["xyz"])


def test_thiol_anchor_keeps_cys_sulfur_on_gold(tmp_path):
    pep = os.path.join(DATA, "GGCGG.pdb")
    rep = mat.build_protein_surface_system(pep, material="Au", layers=3, toward_surface="A:3", gap_nm=0.25,
                                           lateral_padding_nm=1.0, water_above_nm=1.0, minimize_steps=300,
                                           anchor_cys="A:3", out_dir=str(tmp_path))
    assert rep["ok"], rep.get("error") or rep.get("warnings")
    assert rep["anchors"] and "CYS3" in rep["anchors"][0]
    pdb = app.PDBFile(rep["pdb"])
    X = np.array(pdb.positions.value_in_unit(unit.nanometer))
    sg = [a.index for a in pdb.topology.atoms() if a.residue.name == "CYS" and a.name == "SG"][0]
    au = [a.index for a in pdb.topology.atoms() if a.residue.name == "MAU"]
    d = np.sort(np.linalg.norm(X[au] - X[sg], axis=1))[:3] * 10
    assert (d < 2.75).all(), d                                 # S sits in a hollow site, 3 Au at ~2.4-2.6 A
    with pytest.raises(ValueError, match="no Cys SG"):
        mat._add_thiol_anchors(None, pdb.topology, [], "A:1", [])


def test_agent_surface_tool_runs_in_the_background(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert qt_app is not None
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    cmd.load(UBQ, "ubq")
    ctx = AgentContext(dlg=None)

    def ws(sub=""):
        p = tmp_path / sub
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    ctx.workspace = ws
    reg = build_registry(ctx)
    slab = reg.call("build_material", {"material": "graphene", "size_nm": 2}, ctx)
    assert slab["object"] in cmd.get_names("objects")
    job = reg.call("build_protein_surface_system", {"object": "ubq", "material": "Au", "layers": 3,
                                                    "away_from_surface": "A:44", "water_above_A": 10}, ctx)
    deadline = time.time() + 900
    status = reg.call("job_status", {"job_id": job["job_id"]}, ctx)
    while status["status"] == "running" and time.time() < deadline:
        time.sleep(2)
        status = reg.call("job_status", {"job_id": job["job_id"]}, ctx)
    assert status["status"] == "done", status
    assert "ubq_on_Au" in cmd.get_names("objects")
    cl = reg.call("cut_qm_cluster", {"object": "ubq_on_Au", "center": "A:32", "radius_A": 4.5}, ctx)
    assert cl["composition"].get("Au", 0) > 0


def test_hydroxylated_quartz_is_neutral_with_clayff_charges():
    slab = mat.build_material("SiO2", "slab", None, size_nm=(2.5, 2.5), layers=2)
    types = slab["types"]
    assert abs(sum(slab["charges"])) < 1e-9 and types.count("oh") == types.count("ho") == len(slab["bonds"])
    assert 9.0 < slab["silanols_per_nm2"] < 10.2                       # fully hydroxylated quartz (001): 9.6
    P = slab["positions_nm"]
    si = P[[i for i, t in enumerate(types) if t == "st"]]
    o = P[[i for i, t in enumerate(types) if t in ("ob", "oh")]]
    L = np.array(slab["cell_nm"] + [99.0])
    d = o[:, None] - si[None]
    d -= L * np.round(d / L)
    assert np.all((np.linalg.norm(d, axis=2) < 0.175).sum(axis=0) == 4)   # every Si keeps four O
    for oi, hi in slab["bonds"]:
        assert abs(np.linalg.norm(P[oi] - P[hi]) - 0.1) < 1e-6
    xml = mat.material_forcefield_xml("SiO2", slab["symbols"], model=slab)
    ff = app.ForceField()
    ff.loadFile(__import__("io").StringIO(xml))
    top, pos = mat.material_topology(slab)
    system = ff.createSystem(top)
    nb = next(f for f in system.getForces() if f.__class__.__name__ == "NonbondedForce")
    q = sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(nb.getNumParticles()))
    assert abs(q) < 1e-6 and system.getNumParticles() == slab["n_atoms"]


def test_titania_needs_user_parameters_and_neutral_charges():
    slab = mat.build_material("TiO2", "slab", None, size_nm=(2, 2), layers=2)
    assert slab["symbols"].count("O") == 2 * slab["symbols"].count("Ti")
    with pytest.raises(ValueError, match="no bundled MM parameters"):
        mat.material_forcefield_xml("TiO2", slab["symbols"], model=slab)
    with pytest.raises(ValueError, match="neutral"):
        mat.material_forcefield_xml("TiO2", slab["symbols"], model=slab,
                                    params={"Ti": [2.0, 0.2, 0.1], "O": [-0.9, 0.3, 0.6]})
    xml = mat.material_forcefield_xml("TiO2", slab["symbols"], model=slab,
                                      params={"Ti": [2.2, 0.2, 0.1], "O": {"charge": -1.1, "sigma_nm": 0.3,
                                                                          "epsilon_kj": 0.6}})
    assert 'charge="2.200000"' in xml and 'charge="-1.100000"' in xml


def test_protein_on_silica_is_neutral_and_water_stays_out(tmp_path):
    rep = mat.build_protein_surface_system(UBQ, material="SiO2", layers=1, away_from_surface="A:44",
                                           minimize_steps=100, out_dir=str(tmp_path))
    assert rep["ok"], rep.get("error") or rep["warnings"]
    assert abs(rep["total_charge"]) < 1e-3 and rep["slab"]["facet"] == "001"
    sa = rep["surface_analysis"]
    assert "protein_surface_coulomb_unscreened_kj_mol" in sa and sa["min_distance_A"] > 1.5
    pdb = app.PDBFile(rep["pdb"])
    X = np.array([tuple(p) for p in pdb.positions.value_in_unit(unit.nanometer)])
    slab = [a.index for a in pdb.topology.atoms() if a.residue.name in ("QSI", "QOB", "QOH")]
    wat = [a.index for a in pdb.topology.atoms() if a.residue.name == "HOH" and a.name == "O"]
    z0, z1 = X[slab, 2].min() + 0.1, X[slab, 2].max() - 0.1
    assert not np.any((X[wat, 2] > z0) & (X[wat, 2] < z1))
    with pytest.raises(ValueError, match="thiolate"):
        mat.build_protein_surface_system(UBQ, material="SiO2", anchor_cys="A:1", out_dir=str(tmp_path / "x"))
