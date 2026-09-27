import os

import code_v2
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


def _dialog(*names):
    class DummyDialog:
        pass

    for name in names + ("_pka_method", "_pka_cache_dir"):
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
