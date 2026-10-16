"""Regressions found by driving the MD / QM tools headlessly on small systems (peptides, ions, radicals)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets

from leonardo.agent.tools import AgentContext, build_registry, mopac, qmanalysis  # noqa: E402
from leonardo.agent.tools import psi4 as psi4_tool  # noqa: E402
from leonardo.openmm_qmmm_runtime import add_missing_terminal_oxt, compose_openmm_input_pdb  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "components")


def _heavy_polymer_lines(path):
    return [ln.rstrip("\n") for ln in open(path) if ln.startswith(("ATOM", "HETATM")) and ln[76:78].strip() != "H"]


# ------------------------------------------------------------------ C-terminal OXT before OpenMM
def test_missing_cterminal_oxt_is_added_and_openmm_builds_the_peptide(tmp_path):
    app = pytest.importorskip("openmm.app")
    lines = _heavy_polymer_lines(os.path.join(DATA, "GGCGG.pdb"))        # a peptide written without OXT
    out = compose_openmm_input_pdb(lines, [])
    assert out["added_oxt"] == ["A:5 GLY"]
    oxt = [ln for ln in out["lines"] if ln[12:16].strip() == "OXT"]
    assert len(oxt) == 1 and oxt[0][22:26].strip() == "5"
    xyz = {ln[12:16].strip(): np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
           for ln in out["lines"] if ln[22:26].strip() == "5" and ln.startswith("ATOM")}
    assert np.linalg.norm(xyz["OXT"] - xyz["C"]) == pytest.approx(1.25, abs=0.01)
    a, b = xyz["O"] - xyz["C"], xyz["OXT"] - xyz["C"]
    assert np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))) == pytest.approx(120, abs=10)
    path = tmp_path / "pep.pdb"
    path.write_text("\n".join(out["lines"]) + "\n")
    pdb = app.PDBFile(str(path))
    ff = app.ForceField("amber14-all.xml")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addHydrogens(ff, pH=7.0)
    ff.createSystem(mod.topology)            # before the fix: "No template found ... missing a terminal capping group"


def test_oxt_not_duplicated_and_caps_and_ligands_left_alone():
    ubq = _heavy_polymer_lines(os.path.join(DATA, "1UBQ.pdb"))
    ubq = [ln for ln in ubq if ln.startswith("ATOM")]
    lines, added = add_missing_terminal_oxt(ubq)                       # 1UBQ already has its OXT
    assert added == [] and lines == ubq
    capped = _heavy_polymer_lines(os.path.join(DATA, "GGCGG.pdb")) + [
        "HETATM   99  N   NME A   6       9.000   0.000   0.000  1.00  0.00           N",
        "HETATM  100  C   NME A   6      10.000   0.000   0.000  1.00  0.00           C"]
    assert add_missing_terminal_oxt(capped)[1] == []
    ligand = ["HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C"]
    assert add_missing_terminal_oxt(ligand)[1] == []


PHENOL_BLOCK = """\
HETATM    1  C   PHL A 201      26.434  31.354  31.347  0.00  0.00           C
HETATM    2  C   PHL A 201      25.078  31.480  31.656  0.00  0.00           C
HETATM    3  C   PHL A 201      24.163  30.534  31.190  0.00  0.00           C
HETATM    4  C   PHL A 201      24.601  29.461  30.414  0.00  0.00           C
HETATM    5  C   PHL A 201      25.955  29.333  30.105  0.00  0.00           C
HETATM    6  C   PHL A 201      26.860  30.278  30.572  0.00  0.00           C
HETATM    7  O   PHL A 201      28.174  30.116  30.246  0.00  0.00           O
HETATM    8  H   PHL A 201      27.134  32.096  31.715  0.00  0.00           H
HETATM    9  H   PHL A 201      24.734  32.315  32.260  0.00  0.00           H
HETATM   10  H   PHL A 201      23.108  30.634  31.432  0.00  0.00           H
HETATM   11  H   PHL A 201      23.890  28.724  30.051  0.00  0.00           H
HETATM   12  H   PHL A 201      26.303  28.499  29.501  0.00  0.00           H
HETATM   13  H   PHL A 201      28.681  30.845  30.639  0.00  0.00           H
CONECT    1    2    8    6
CONECT    2    1    3    9
CONECT    3    2    4   10
CONECT    4    3    5   11
CONECT    5    4    6   12
CONECT    6    5    1    7
CONECT    7    6   13
""".splitlines()


def test_ligand_with_element_only_atom_names_keeps_all_its_atoms(tmp_path):
    app = pytest.importorskip("openmm.app")
    # PyMOL names SDF atoms by element only; OpenMM's PDB reader merged them (13 -> 3 atoms) and the ligand,
    # parameterized just before, matched no template and was silently dropped from run_md
    out = compose_openmm_input_pdb(_heavy_polymer_lines(os.path.join(DATA, "GGCGG.pdb")), [PHENOL_BLOCK])
    path = tmp_path / "cplx.pdb"
    path.write_text("\n".join(out["lines"]) + "\n")
    pdb = app.PDBFile(str(path))
    lig = [r for r in pdb.topology.residues() if r.name == "PHL"][0]
    names = [a.name for a in lig.atoms()]
    assert len(names) == 13 and len(set(names)) == 13 and names[:2] == ["C1", "C2"] and "O1" in names
    assert sum(1 for b in pdb.topology.bonds() if b[0].residue is lig) == 13      # CONECT bonds kept
    assert out["renamed_atoms"] == ["PHL A:201"]
    unchanged = ["HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C"]
    assert compose_openmm_input_pdb([], [unchanged])["renamed_atoms"] == []


# ------------------------------------------------------------------ Psi4 charge / multiplicity
@pytest.fixture(scope="module")
def dialog(tmp_path_factory):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2
    from pymol import cmd
    from rdkit import Chem
    from rdkit.Chem import AllChem

    cmd.reinitialize()
    folder = tmp_path_factory.mktemp("mols")
    for name, smi in (("acetate", "CC(=O)[O-]"), ("methyl", "[CH3]"), ("ethanol", "CCO"), ("ammonium", "[NH4+]")):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=7)
        Chem.MolToMolFile(mol, str(folder / f"{name}.sdf"))
        cmd.load(str(folder / f"{name}.sdf"), name)
    dlg = code_v2.ProtLigInteractDialog()
    work = str(tmp_path_factory.mktemp("ws"))
    dlg._dock_workspace_dir = lambda: work
    dlg._save_settings = lambda *a, **k: None
    dlg._on_psi4_run = lambda *a, **k: None          # the widgets are what Leonardo's run reads
    app.processEvents()
    yield dlg
    dlg.close()


def _psi4_spin(dialog, **args):
    ctx = AgentContext(dialog)
    build_registry(ctx).call("run_psi4", dict({"operation": "energy", "method": "hf", "basis": "sto-3g"}, **args), ctx)
    auto = dialog.cb_psi4_auto_spin.isChecked()
    if auto:      # what Leonardo's run would use with auto charge/multiplicity on
        atoms, _sel, _fb = dialog._psi4_collect_atoms_from_active_selection(state=1)
        return dialog._psi4_auto_charge_multiplicity(atoms=atoms, quiet=True)[:2]
    return dialog.spin_psi4_charge.value(), dialog.spin_psi4_mult.value()


def test_psi4_charge_and_multiplicity_from_the_molecule_not_the_last_job(dialog):
    assert _psi4_spin(dialog, object="ammonium", charge=1, multiplicity=1) == (1, 1)
    # before: auto mode reused the +1 left in the spin box -> ethanol ran as a cation radical (1, 2)
    assert _psi4_spin(dialog, object="ethanol") == (0, 1)
    assert _psi4_spin(dialog, object="acetate") == (-1, 1)          # formal charge of the SDF; before (0, 2)
    assert _psi4_spin(dialog, object="methyl") == (0, 2)
    assert _psi4_spin(dialog, object="acetate", charge=-1) == (-1, 1)


def test_cluster_reaction_does_not_inherit_the_previous_charge(dialog, monkeypatch):
    from pymol import cmd

    cmd.load(os.path.join(DATA, "GGCGG.pdb"), "pepc")
    cmd.remove("pepc and elem H")
    cmd.h_add("pepc")
    ctx = AgentContext(dialog)
    reg = build_registry(ctx)
    reg.call("build_reaction_cluster", {"object": "pepc", "center_selection": "pepc and resi 3 and name SG",
                                        "radius_A": 3.0}, ctx)
    monkeypatch.setattr(dialog, "_on_psi4_cluster_run_workflow", lambda *a, **k: None)
    dialog.spin_psi4_charge.setValue(1)          # left over from an earlier job (e.g. NH4+)
    dialog.spin_psi4_mult.setValue(2)
    reg.call("run_cluster_reaction", {"bonds_forming": [["pepc and resi 3 and name SG", "pepc and resi 3 and name N"]],
                                      "method": "hf", "basis": "sto-3g"}, ctx)
    if dialog.cb_psi4_auto_spin.isChecked():     # before the fix: auto mode, which reads that spin box
        atoms = list(dialog._psi4_cluster_state.get("reactant_atoms") or dialog._psi4_cluster_state.get("atoms"))
        got = dialog._psi4_auto_charge_multiplicity(atoms=atoms, quiet=True)[:2]
    else:
        got = (dialog.spin_psi4_charge.value(), dialog.spin_psi4_mult.value())
    assert got == (0, 1)


def test_psi4_orbital_energies_of_an_open_shell_job(tmp_path):
    (tmp_path / "psi4_run.out").write_text(
        "    Alpha Occupied:\n\n       1A    -11.005535     2A     -0.904490\n       5A     -0.356248\n\n"
        "    Alpha Virtual:\n\n       6A      0.626225     7A      0.765629\n\n"
        "    Beta Occupied:\n\n       1A    -10.974161     4A     -0.520292\n\n"
        "    Beta Virtual:\n\n       5A      0.334958\n\n    Final Occupation by Irrep:\n")
    e = psi4_tool._orbital_energies(str(tmp_path))          # before: {} (only "Doubly Occupied" was read)
    assert e["homo_hartree"] == -0.356248 and e["lumo_hartree"] == 0.626225


# ------------------------------------------------------------------ open-shell UV-vis
@pytest.mark.skipif(not mopac.mopac_executable(), reason="MOPAC not installed")
def test_mopac_radical_with_uv_keeps_its_results(tmp_path):
    coords = [[0.0, 0.0, 0.0], [1.08, 0.0, 0.0], [-0.54, 0.935, 0.0], [-0.54, -0.935, 0.0]]
    res = mopac.run_mopac_job(mopac.mopac_executable(), str(tmp_path), ["C", "H", "H", "H"], coords, "PM7", 0, 2,
                              None, False, False, True, 6, {})
    # before: "UHF and RHF cannot both be used" from the CIS step failed the whole job
    assert res["open_shell"] and "uv_states" not in res and "closed-shell" in res["uv_note"]


def test_tddft_on_a_radical_selection_uses_a_doublet(dialog, monkeypatch):
    seen = {}
    monkeypatch.setattr(qmanalysis, "_psi4_child_job", lambda ctx, kind, workdir, args, collect, desc: seen.update(args))
    ctx = AgentContext(dialog)
    build_registry(ctx).call("run_uv_spectrum", {"selection": "methyl", "basis": "sto-3g", "states": 2}, ctx)
    assert seen["geometry"].startswith("0 2\n")          # before: "0 1", which Psi4 rejects
    build_registry(ctx).call("run_uv_spectrum", {"selection": "acetate", "basis": "sto-3g", "states": 2}, ctx)
    assert seen["geometry"].startswith("-1 1\n")


# ------------------------------------------------------------------ build_validated_system(minimize_steps=0)
def test_zero_minimize_steps_means_no_minimization():
    from leonardo import component_library_runtime as cl

    rep = cl.build_and_validate(os.path.join(DATA, "3KS3_zn_site.pdb"), family="amber14", minimize_steps=0)
    # before: OpenMM's maxIterations=0 minimized to convergence (-210 -> -1280 kJ/mol)
    assert rep["ok"] and rep["energy_minimized_kj"] == rep["energy_start_kj"]


# ------------------------------------------------------------------ protein on graphene: sane energies
def test_graphene_sheet_does_not_repel_itself(tmp_path):
    pytest.importorskip("ase")
    import openmm as mm
    from openmm import unit

    from leonardo import materials_runtime as mat

    rep = mat.build_protein_surface_system(os.path.join(DATA, "GGCGG.pdb"), material="graphene", layers=1,
                                           water_above_nm=0.8, ionic_strength_molar=0.0, minimize_steps=50,
                                           out_dir=str(tmp_path))
    # before: the frozen sheet's own C-C pairs (0.14 nm, sigma 0.34 nm) gave ~+1.8e7 kJ/mol
    assert rep["ok"] and rep["energy_minimized_kj"] < 0, (rep["energy_start_kj"], rep["energy_minimized_kj"])
    system = mm.XmlSerializer.deserialize(open(os.path.join(str(tmp_path), "surface_system.xml")).read())
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    n_c = sum(1 for i in range(system.getNumParticles()) if system.getParticleMass(i).value_in_unit(unit.dalton) == 0)
    excluded = sum(1 for k in range(nb.getNumExceptions()) if nb.getExceptionParameters(k)[0] >= 0
                   and system.getParticleMass(nb.getExceptionParameters(k)[0]).value_in_unit(unit.dalton) == 0)
    assert excluded == n_c * 12 // 2              # 3 + 6 + 3 neighbours within 0.3 nm per C, each pair once


# ------------------------------------------------------------------ short membrane equilibration
def test_equilibration_shorter_than_one_report_interval_still_writes_a_frame(tmp_path):
    import openmm as mm
    from openmm import app, unit

    from leonardo import membrane_runtime as mr

    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, boxSize=mm.Vec3(1.8, 1.8, 1.8) * unit.nanometer)
    pdb_path, xml_path = str(tmp_path / "box.pdb"), str(tmp_path / "box.xml")
    with open(pdb_path, "w") as fh:
        app.PDBFile.writeFile(mod.topology, mod.positions, fh)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.8 * unit.nanometer,
                             constraints=app.HBonds)
    with open(xml_path, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(system))
    # equilibrate_membrane_system(stage_ps=1, stages=1): shorter than the 10 ps report interval; the tool
    # loads the trajectory into PyMOL, which failed on the empty DCD ("Could not load trajectory")
    eq = mr.equilibrate_membrane(pdb_path, xml_path, str(tmp_path / "eq"), stages_ps=(0.1,), restraint_k=(1000,))
    from pymol import cmd

    cmd.load(pdb_path, "box_eq")
    cmd.load_traj(eq["trajectory"], "box_eq", state=1)
    assert eq["ok"] and cmd.count_states("box_eq") >= 1 and os.path.getsize(eq["trajectory"]) > 1000


@pytest.mark.skipif(not mopac.mopac_executable(), reason="MOPAC not installed")
def test_mopac_cis_geometry_check_is_retried_and_a_failed_cis_keeps_the_ground_state(tmp_path, monkeypatch):
    coords = [[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]
    real_run = mopac._run
    seen = []

    def geo_ok_once(exe, workdir, stem, keys, *a, **k):
        seen.append((stem, keys))
        if stem == "cis" and "GEO-OK" not in keys:
            raise RuntimeError('MOPAC 1SCF failed: JOB STOPPED. TO CONTINUE, SPECIFY "GEO-OK".')
        return real_run(exe, workdir, stem, keys, *a, **k)

    monkeypatch.setattr(mopac, "_run", geo_ok_once)
    res = mopac.run_mopac_job(mopac.mopac_executable(), str(tmp_path), ["C", "H", "H", "H", "H"], coords, "PM7", 0,
                              1, None, False, False, True, 4, {})
    # before: the CIS geometry check (a cluster cut from a larger system) failed the whole MOPAC job
    assert [s for s, k in seen if s == "cis"] == ["cis", "cis"] and "GEO-OK" in seen[-1][1]
    assert res["uv_states"] and res["eigenvalues_ev"]

    def cis_fails(exe, workdir, stem, keys, *a, **k):
        if stem == "cis":
            raise RuntimeError("MOPAC 1SCF failed: something else")
        return real_run(exe, workdir, stem, keys, *a, **k)

    monkeypatch.setattr(mopac, "_run", cis_fails)
    (tmp_path / "b").mkdir()
    res = mopac.run_mopac_job(mopac.mopac_executable(), str(tmp_path / "b"), ["C", "H", "H", "H", "H"], coords,
                              "PM7", 0, 1, None, False, False, True, 4, {})
    assert "uv_states" not in res and "not available" in res["uv_note"] and res["eigenvalues_ev"]


def test_mopac_keeps_the_oxygens_of_an_xyz_molecule_and_drops_real_water(tmp_path):
    from pymol import cmd

    # acetic acid from an XYZ file: PyMOL flags each lone-O "UNK" residue as solvent, and the old
    # "not solvent" filter sent MOPAC the molecule without its oxygens (QM clusters lost all their O)
    xyz = tmp_path / "acoh.xyz"
    xyz.write_text("8\nacetic acid\nC 0.000 0.000 0.000\nC 1.500 0.000 0.000\nO 2.100 1.050 0.000\n"
                   "O 2.150 -1.150 0.000\nH 3.100 -1.050 0.000\nH -0.380 1.020 0.000\nH -0.380 -0.520 0.880\n"
                   "H -0.380 -0.520 -0.880\n")
    cmd.delete("all")
    cmd.load(str(xyz), "acoh")
    cmd.pseudoatom("wat", pos=[10.0, 0.0, 0.0], resn="HOH", elem="O", name="O")
    assert cmd.count_atoms("wat and solvent") == 1 and cmd.count_atoms("acoh and solvent") == 2   # the trap
    symbols, _coords, _q = mopac._molecule_from_selection("acoh or wat", add_hydrogens=False)
    assert symbols.count("O") == 2 and len(symbols) == 8
    cmd.delete("all")
