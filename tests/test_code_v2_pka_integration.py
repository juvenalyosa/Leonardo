import os

import code_v2
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


def _dialog(*names):
    class DummyDialog:
        pass

    for name in names + ("_pka_method", "_pka_cache_dir", "_pka_backend_options"):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    dlg = DummyDialog()
    dlg._append_debug_log = lambda *_a: None
    return dlg


def test_locked_state_is_recognized(tmp_path):
    dlg = _dialog("_pka_ligand_is_locked")
    path = tmp_path / "state.sdf"
    path.write_text("")
    assert not dlg._pka_ligand_is_locked(str(path))
    dlg._pka_locked_ligand_file = str(path)
    assert dlg._pka_ligand_is_locked(str(path)) and not dlg._pka_ligand_is_locked(str(tmp_path / "other.sdf"))


def test_default_method_follows_the_pka_window():
    dlg = _dialog("_pka_set_default_method")
    assert dlg._pka_method() == "empirical"
    dlg._pka_set_default_method("pm7")
    assert dlg._pka_method() == "pm7"


def test_psi4_is_never_used_for_library_scale_protonation():
    dlg = _dialog("_pka_set_default_method")
    dlg._psi4_tool_path = lambda: "/opt/psi4/bin/psi4"
    dlg._pka_set_default_method("psi4")
    assert dlg._pka_method() == "psi4" and dlg._pka_method(bulk=True) == "pm7"
    assert dlg._pka_backend_options("psi4") == {"psi4_exe": "/opt/psi4/bin/psi4"}
    assert dlg._pka_backend_options("pm7") is None


def test_md_components_are_protonated_at_the_md_ph(tmp_path):
    """Crystal HETATM + neutral CCD SMILES used to be parametrized neutral (COOH)."""
    mol = Chem.AddHs(Chem.MolFromSmiles("OC(=O)c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    heavy = Chem.RemoveHs(mol)
    for atom in heavy.GetAtoms():
        info = Chem.AtomPDBResidueInfo(f"{atom.GetSymbol()}{atom.GetIdx() + 1:<3d}"[:4], residueName="BZO",
                                       residueNumber=1, chainId="A", isHeteroAtom=True)
        atom.SetMonomerInfo(info)
    src = tmp_path / "component.pdb"
    Chem.MolToPDBFile(heavy, str(src))
    dlg = _dialog("_openmm_build_hydrated_component_pdb", "_openmm_ligand_target_ph")
    dlg._current_small_molecule_target_ph = lambda: 7.4
    out = tmp_path / "hydrated.pdb"
    res = dlg._openmm_build_hydrated_component_pdb(str(src), str(out), {"resn": "BZO", "chain": "A", "resi": "1"},
                                                   smiles_hint="OC(=O)c1ccccc1", target_ph=7.4)
    assert res["ok"] and res["formal_charge"] == -1 and res["protonation"]["method"] == "empirical"
    res_low = dlg._openmm_build_hydrated_component_pdb(str(src), str(out), {"resn": "BZO"},
                                                       smiles_hint="OC(=O)c1ccccc1", target_ph=2.0)
    assert res_low["formal_charge"] == 0


def test_meeko_prep_protonates_library_ligands_only_when_asked(tmp_path):
    pytest.importorskip("meeko")
    dlg = _dialog("_vs_prepare_ligand_with_meeko", "_select_best_ligand_fragment", "_ligand_geometry_is_bad",
                  "_dock_pdbqt_syntax_ok", "_dock_pdbqt_type_token", "_dock_valid_pdbqt_types")
    out = tmp_path / "lig.pdbqt"

    def charge_sum(path):
        return sum(float(ln[70:76]) for ln in open(path) if ln.startswith(("ATOM", "HETATM")))

    ok, msg = dlg._vs_prepare_ligand_with_meeko(Chem.MolFromSmiles("NC(=N)c1ccccc1"), str(out), target_ph=7.4)
    assert ok, msg
    assert charge_sum(out) == pytest.approx(1.0, abs=0.01)                 # amidinium for VS/lead-opt
    ok, msg = dlg._vs_prepare_ligand_with_meeko(Chem.MolFromSmiles("NC(=N)c1ccccc1"), str(out))
    assert ok and charge_sum(out) == pytest.approx(0.0, abs=0.01)          # docking path: input kept as given


def test_md_component_keeps_crystal_atom_names_and_coordinates(tmp_path):
    """OpenMM matches residue templates by atom name: protonation must not rename heavy atoms."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "benchmarks", "redocking", "3ptb_ben", "ben_crystal_template.pdb")
    dlg = _dialog("_openmm_build_hydrated_component_pdb", "_openmm_ligand_target_ph")
    dlg._current_small_molecule_target_ph = lambda: 7.4
    out = tmp_path / "ben_hydrated.pdb"
    res = dlg._openmm_build_hydrated_component_pdb(src, str(out), {"resn": "BEN", "chain": "A", "resi": "1"},
                                                   smiles_hint="[NH2+]=C(N)c1ccccc1", target_ph=7.4)
    assert res["ok"] and res["formal_charge"] == 1

    def heavy(path):
        return sorted((ln[12:16].strip(), ln[17:20].strip(), round(float(ln[30:38]), 3), round(float(ln[38:46]), 3),
                       round(float(ln[46:54]), 3)) for ln in open(path)
                      if ln.startswith(("ATOM", "HETATM")) and ln[76:78].strip() != "H")
    assert heavy(src) == heavy(str(out))


def test_locked_state_note_warns_when_ph_changed(tmp_path):
    from leonardo.pka_dialog import write_state_sdf

    mol = Chem.AddHs(Chem.MolFromSmiles("NC(=[NH2+])c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    path = write_state_sdf(mol, {"method": "pm7", "ph": 7.4, "charge": 1, "population": 1.0},
                           str(tmp_path / "state.sdf"))
    dlg = _dialog("_pka_ligand_is_locked", "_pka_locked_state_note")
    dlg._pka_locked_ligand_file = path
    note, warn = dlg._pka_locked_state_note(path, 7.4)
    assert note == "protonation_locked:pka_window(pm7, pH 7.40)" and warn is None
    _note, warn = dlg._pka_locked_state_note(path, 5.0)
    assert warn and "pH 7.40" in warn


def _load_sdf(tmp_path, smiles, name):
    from pymol import cmd

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    path = tmp_path / f"{name}.sdf"
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()
    cmd.load(str(path), name)
    return mol


def test_selection_of_an_editor_molecule_keeps_its_bond_orders_without_open_babel(tmp_path):
    # a molecule drawn in the ligand editor and built in 3D reaches PyMOL as SDF: its bond orders are used
    # directly (before, they were dropped and Open Babel had to guess them again, and could fail)
    from pymol import cmd

    cmd.delete("all")
    _load_sdf(tmp_path, "NC(=N)c1ccccc1", "ligand_edited")
    cmd.select("sele", "ligand_edited")
    dlg = _dialog("_pka_selection_molecule")
    dlg._dock_tool_path = lambda name: None                  # no Open Babel at all
    dlg._openmm_component_smiles_hint = lambda info: ""
    mol, label = dlg._pka_selection_molecule("sele")
    assert Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=True)).replace("[H]/", "").replace("/", "") in (
        "N=C(N)c1ccccc1", "NC(=N)c1ccccc1")
    assert mol.GetNumConformers() == 1 and "bond orders from PyMOL" in label
    cmd.delete("all")


def test_selection_over_two_objects_is_refused_with_their_names(tmp_path):
    from pymol import cmd

    cmd.delete("all")
    _load_sdf(tmp_path, "CCO", "lig_one")
    _load_sdf(tmp_path, "CCN", "lig_two")
    cmd.select("sele", "lig_one or lig_two")
    dlg = _dialog("_pka_selection_molecule")
    dlg._dock_tool_path = lambda name: None
    dlg._openmm_component_smiles_hint = lambda info: ""
    with pytest.raises(RuntimeError, match=r"several objects \(lig_one, lig_two\)"):
        dlg._pka_selection_molecule("sele")
    cmd.delete("all")


def test_pymol_guessed_aromatic_amidine_is_not_trusted(tmp_path):
    # from a PDB with hydrogens PyMOL marks the amidine bonds aromatic: the selection falls back to a real
    # perception instead of returning N:C(:N)
    from pymol import cmd

    cmd.delete("all")
    mol = Chem.AddHs(Chem.MolFromSmiles("NC(=N)c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    pdb = tmp_path / "amidine.pdb"
    Chem.MolToPDBFile(mol, str(pdb), flavor=2 | 8)
    cmd.load(str(pdb), "amidine")
    cmd.select("sele", "amidine")
    dlg = _dialog("_pka_selection_molecule")
    dlg._dock_tool_path = lambda name: None
    dlg._openmm_component_smiles_hint = lambda info: ""
    out, _label = dlg._pka_selection_molecule("sele")
    assert not any(b.GetIsAromatic() and not b.IsInRing() for b in out.GetBonds())
    assert Chem.MolToSmiles(Chem.RemoveHs(out)).count("=") == 1
    cmd.delete("all")
