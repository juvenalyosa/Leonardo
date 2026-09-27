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
