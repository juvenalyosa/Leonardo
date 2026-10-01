import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
from rdkit import Chem  # noqa: E402

from leonardo.pka_dialog import PkaDialog, write_state_sdf  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _rows(tbl, cols):
    return [[tbl.item(r, c).text() for c in range(cols)] for r in range(tbl.rowCount())]


def test_dialog_computes_sites_states_and_hands_off(app, tmp_path):
    sent = []
    dlg = PkaDialog(use_for_docking=lambda m, i: sent.append((m, i)) or "ok", cache_dir=str(tmp_path))
    dlg.edit_smiles.setText("NCC(=O)O")
    dlg._on_run()                                  # no submit hook: runs synchronously
    assert [r[1] for r in _rows(dlg.tbl_sites, 3)] == ["amine1", "carboxylic_aliphatic"]
    assert _rows(dlg.tbl_states, 3)[0][1:] == ["+0", "[NH3+]CC(=O)[O-]"]
    dlg.spin_ph.setValue(1.0)                      # states follow the pH without recomputing
    assert _rows(dlg.tbl_states, 3)[0][1:] == ["+1", "[NH3+]CC(=O)O"]
    dlg.spin_ph.setValue(7.4)
    dlg.btn_to_docking.click()
    mol, info = sent[-1]
    assert info["charge"] == 0 and info["state"] == [True, False] and mol.GetNumConformers() == 1
    assert not dlg.btn_to_param.isEnabled()        # no parametrization hook given


def test_dialog_keeps_3d_coordinates_and_warns_on_mixed_population(app, tmp_path):
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("c1c[nH]cn1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    dlg = PkaDialog(cache_dir=str(tmp_path), default_ph=6.9)
    dlg.set_molecule(mol, "imidazole")
    dlg._on_run()
    assert "Mixed population" in dlg.txt_warnings.toPlainText()          # ~50/50 at its pKa
    out, info = dlg.selected_state_molecule()
    ref = Chem.RemoveHs(mol).GetConformer().GetPositions()
    got = Chem.RemoveHs(out).GetConformer().GetPositions()
    assert abs(ref - got).max() < 1e-6                                     # heavy atoms untouched
    path = write_state_sdf(out, info, str(tmp_path / "state.sdf"))
    back = next(iter(Chem.SDMolSupplier(path, removeHs=False)))
    assert back.GetProp("LEONARDO_NET_CHARGE") == str(info["charge"]) and back.GetProp("LEONARDO_PROTONATION_PH") == "6.90"
