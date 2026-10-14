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
