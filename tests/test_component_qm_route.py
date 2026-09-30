"""QM/MM route for components without MM parameters (heme c, metal sites): CCD completion, QM-region
plans and a real MOPAC QM/MM single point through Leonardo's worker."""
import os
import shutil

import numpy as np
import pytest

app = pytest.importorskip("openmm.app")
unit = pytest.importorskip("openmm.unit")
pytest.importorskip("gemmi")

from leonardo import component_library_runtime as cl  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "components")


@pytest.fixture(autouse=True)
def offline_ccd(tmp_path, monkeypatch):
    (tmp_path / "ccd").mkdir()
    shutil.copy(os.path.join(DATA, "ccd", "HEC.cif"), tmp_path / "ccd" / "HEC.cif")
    monkeypatch.setenv("LEONARDO_PARAMETERS_DIR", str(tmp_path))


def _load(name):
    pdb = app.PDBFile(os.path.join(DATA, name))
    return pdb.topology, [tuple(p.value_in_unit(unit.nanometer)) for p in pdb.positions]


def test_heme_c_completed_from_the_ccd_with_bound_form_hydrogens():
    top, pos = _load("1HRC_heme_c.pdb")
    hec = [r for r in top.residues() if r.name == "HEC"][0]
    sg = {name: [pos[a.index] for r in top.residues() if r.id == resi for a in r.atoms() if a.name == "SG"]
          for name, resi in (("CAB", "14"), ("CAC", "17"))}
    model = cl.complete_from_ccd(hec, pos, ph=7.4, fetch=False, external=sg)
    atoms = {n: np.array(x) for n, _e, x in model["atoms"]}
    assert len(atoms) == 75 and sum(1 for _n, e, _x in model["atoms"] if e == "H") == 32
    assert model["charge"] == -4 and model["has_metal"]                  # porphyrin 2- + two propionates
    xh = [np.linalg.norm(atoms[a] - atoms[b]) * 10 for a, b in model["bonds"] if "H" in (a[0], b[0])]
    assert 0.95 < min(xh) and max(xh) < 1.12
    from scipy.spatial import cKDTree

    assert not cKDTree(np.array(list(atoms.values()))).query_pairs(0.09)   # no two atoms closer than 0.9 A
    for c, resi in (("CAB", "14"), ("CAC", "17")):                          # no H left pointing at the Cys S
        s_xyz = np.array(sg[c][0])
        assert min(np.linalg.norm(atoms[h] - s_xyz) for h in atoms if h.startswith("H")) * 10 > 1.5


def test_qm_region_plan_for_cytochrome_c_and_zinc_site():
    top, pos = _load("1HRC_heme_c.pdb")
    plan = cl.resolve_components(top, pos, family="amber14")
    site = cl.plan_qm_region(top, pos, plan, fetch_ccd=False)["sites"][0]
    assert site["component"] == "HEC A105"
    assert {s["resi"] for s in site["side_chains"]} == {"14", "17", "18", "80"}
    fe3 = [o for o in site["charge_options"] if o["metal"] == "FE" and o["oxidation"] == 3][0]
    assert fe3["qm_charge"] == -1 and fe3["multiplicities"] == [2, 6]
    top, pos = _load("3KS3_zn_site.pdb")
    plan = cl.resolve_components(top, pos, family="amber14")
    site = cl.plan_qm_region(top, pos, plan, fetch_ccd=False)["sites"][0]
    assert {s["resi"] for s in site["side_chains"]} >= {"94", "96", "119"}
    assert site["charge_options"][0]["qm_charge"] == 2


def test_placeholder_template_builds_heme_c_system_without_mm_parameters():
    pdb = app.PDBFile(os.path.join(DATA, "1HRC_full_nowat.pdb"))
    m = app.Modeller(pdb.topology, pdb.positions)
    pos = [tuple(p.value_in_unit(unit.nanometer)) for p in m.positions]
    plan = cl.resolve_components(m.topology, pos, family="amber14")
    ren = {}
    files = cl.prepare_modeller(m, plan, "amber14", renames=ren)
    qf, qv, keys, info = cl.install_qm_placeholders(m, plan, fetch_ccd=False)
    assert keys == [("A", "105", "HEC")] and info[0]["qm_atoms"] == 75
    ff = app.ForceField(*files, *qf)
    variants, _notes = cl.coordination_variants(m.topology, [tuple(p.value_in_unit(unit.nanometer)) for p in m.positions])
    for k, v in qv.items():
        variants[k] = v
    m.addHydrogens(ff, pH=7.4, variants=variants)
    cl.restore_names(m.topology, ren)
    system = ff.createSystem(m.topology, nonbondedMethod=app.NoCutoff)
    cys = [r for r in m.topology.residues() if r.id in ("14", "17") and r.name.startswith("CY")]
    assert all("HG" not in {a.name for a in r.atoms()} for r in cys)          # thioether Cys are CYX
    assert system.getNumParticles() == m.topology.getNumAtoms()


def test_real_mopac_qmmm_on_cytochrome_c(tmp_path):
    if shutil.which("mopac") is None and not os.path.isfile(os.path.join(os.path.dirname(os.__file__), "..", "..",
                                                                          "bin", "mopac")):
        pytest.skip("MOPAC not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert qt_app is not None                                              # keep the application alive
    import code_v2
    from pymol import cmd

    from leonardo.agent.tools._widgets import set_combo, set_value

    cmd.reinitialize()
    cmd.load(os.path.join(DATA, "1HRC_full_nowat.pdb"), "cytc")
    dlg = code_v2.ProtLigInteractDialog()
    set_combo(dlg, "cmb_qmmm_object", "cytc")
    set_combo(dlg, "cmb_qmmm_operation", "probe")
    set_combo(dlg, "cmb_qmmm_mode", "gas")
    set_combo(dlg, "cmb_qmmm_region_mode", "pymol_selection")
    set_value(dlg, "edit_qmmm_qm_selection", "resn HEC or (resi 18+80 and not name N+CA+C+O+H+HA)")
    set_value(dlg, "edit_qmmm_mm_selection", "")
    set_value(dlg, "spin_qmmm_charge", -1)
    set_value(dlg, "spin_qmmm_mult", 2)
    set_combo(dlg, "cmb_openmm_ff_family", "amber14", required=False)
    pack = dlg._qmmm_collect_run_payload()
    assert pack["ok"], pack
    out = dlg._run_openmm_worker_job(pack["payload"])
    assert out["ok"], out.get("error")
    assert not out.get("removed_residues")
    probe = out["qmmm_probe"]
    assert probe["region"]["qm_atom_count"] == 97                        # heme c 75 + His18/Met80 side chains with H
    assert probe["qmmm"]["heat_of_formation_kcal_mol"] < 0
    dlg.close()
