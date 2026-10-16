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


def test_dialog_level_of_theory_follows_the_method(app, tmp_path):
    defaults = []
    dlg = PkaDialog(cache_dir=str(tmp_path), set_default_method=lambda m, lv=None: defaults.append((m, lv)),
                    default_level={"theory": "pbe0", "basis": "def2-svpd"})
    dlg.show()
    app.processEvents()
    dlg.cmb_method.setCurrentIndex(dlg.cmb_method.findData("empirical"))
    assert not dlg.cmb_hamiltonian.isVisible() and not dlg.cmb_basis.isVisible()
    dlg.cmb_method.setCurrentIndex(dlg.cmb_method.findData("pm7"))
    assert dlg.cmb_hamiltonian.isVisible() and not dlg.cmb_theory.isVisible()
    assert dlg.level_options() == {"hamiltonian": "PM7"} and "Calibrated" in dlg.lbl_level_status.text()
    dlg.cmb_hamiltonian.setCurrentIndex(dlg.cmb_hamiltonian.findData("AM1"))
    assert "not calibrated" in dlg.lbl_level_status.text()
    dlg.cmb_method.setCurrentIndex(dlg.cmb_method.findData("psi4"))
    assert dlg.cmb_theory.isVisible() and dlg.cmb_basis.isVisible() and dlg.cmb_solvent.isVisible()
    assert dlg.level_options() == {"theory": "pbe0", "basis": "def2-svpd", "pcm_solver": "CPCM"}
    dlg.cmb_basis.setEditText("6-311++G(2d,p)")                  # any basis Psi4 knows can be typed
    dlg.cmb_theory.setEditText("r2scan")
    assert dlg.level_options()["basis"] == "6-311++g(2d,p)" and dlg.level_options()["theory"] == "r2scan"
    dlg.cb_default_method.setChecked(True)                         # the default level travels with the method
    assert defaults[-1] == ("psi4", {"theory": "r2scan", "basis": "6-311++g(2d,p)", "pcm_solver": "CPCM"})
    dlg.cmb_theory.setEditText("bad;name")
    dlg._sync_level_widgets()
    assert "Invalid level" in dlg.lbl_level_status.text()
    dlg.close()


def test_dialog_passes_the_level_to_the_backend(app, tmp_path, monkeypatch):
    from leonardo import pka_dialog as D

    seen = {}

    def fake(mol, method, focus_ph, cache_dir, cancel, backend_options):
        seen.update(method=method, options=dict(backend_options or {}))
        raise D.P.PkaError("stop")

    monkeypatch.setattr(D.P, "compute_pka", fake)
    dlg = PkaDialog(cache_dir=str(tmp_path), backend_options=lambda m: {"psi4_exe": "/x/psi4"})
    dlg.edit_smiles.setText("CC(=O)O")
    dlg.cmb_method.setCurrentIndex(dlg.cmb_method.findData("psi4"))
    dlg.cmb_theory.setCurrentIndex(dlg.cmb_theory.findData("m06-2x"))
    dlg.cmb_basis.setCurrentIndex(dlg.cmb_basis.findData("aug-cc-pvdz"))
    dlg._on_run()
    assert seen["method"] == "psi4"
    assert seen["options"] == {"psi4_exe": "/x/psi4", "theory": "m06-2x", "basis": "aug-cc-pvdz", "pcm_solver": "CPCM"}
