"""Crystal ligands for redocking: bond orders from the RCSB CCD and the docking-pH protonation reach the PDBQT."""
import os

import pytest

pytest.importorskip("rdkit")

# benzamidine (3PTB, BEN A 1) as deposited: heavy atoms only, no bond orders
BEN_PDB = """\
HETATM 1632  C1  BEN A   1      -1.853  14.311  16.658  1.00 19.86           C
HETATM 1633  C2  BEN A   1      -2.107  15.653  16.758  1.00 19.86           C
HETATM 1634  C3  BEN A   1      -1.774  16.341  17.932  1.00 19.86           C
HETATM 1635  C4  BEN A   1      -1.175  15.662  19.005  1.00 19.86           C
HETATM 1636  C5  BEN A   1      -0.914  14.295  18.885  1.00 19.86           C
HETATM 1637  C6  BEN A   1      -1.257  13.634  17.708  1.00 19.86           C
HETATM 1638  C   BEN A   1      -2.193  13.627  15.496  1.00 19.86           C
HETATM 1639  N1  BEN A   1      -2.797  14.235  14.491  1.00 19.86           N
HETATM 1640  N2  BEN A   1      -1.762  12.391  15.309  1.00 19.86           N
END
"""


class _Dlg:
    _fetch_json_url = None


def _ccd_smiles(payload):
    import code_v2

    dlg = _Dlg()
    dlg._fetch_json_url = lambda url, timeout=20: payload
    return code_v2.ProtLigInteractDialog._rcsb_chemcomp_smiles(dlg, "BEN")


def test_ccd_smiles_reads_the_keys_the_rcsb_api_actually_returns():
    # data.rcsb.org/rest/v1/core/chemcomp/BEN spells them "SMILES" / "SMILES_stereo"
    assert _ccd_smiles({"rcsb_chem_comp_descriptor": {"SMILES": "NC(=N)c1ccccc1",
                                                      "SMILES_stereo": "[H]/N=C(/N)c1ccccc1"}}) \
        == "[H]/N=C(/N)c1ccccc1"
    assert _ccd_smiles({"rcsb_chem_comp_descriptor": {"SMILES": "NC(=N)c1ccccc1"}}) == "NC(=N)c1ccccc1"
    # deposited descriptors as the fallback
    assert _ccd_smiles({"pdbx_chem_comp_descriptor": [
        {"type": "InChI", "descriptor": "InChI=1S/C7H8N2"},
        {"type": "SMILES_CANONICAL", "program": "CACTVS", "descriptor": "NC(=N)c1ccccc1"}]}) == "NC(=N)c1ccccc1"
    assert _ccd_smiles({}) == ""


@pytest.fixture(scope="module")
def dialog():
    pytest.importorskip("meeko")
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2

    dlg = code_v2.ProtLigInteractDialog()
    app.processEvents()
    yield dlg
    dlg.close()


def test_crystal_benzamidine_is_docked_aromatic_and_as_the_amidinium(dialog, tmp_path):
    lig = tmp_path / "ben.pdb"
    lig.write_text(BEN_PDB)
    dialog._current_pubchem_smiles_hint = lambda: ""
    out, info = dialog._prepare_ligand_for_docking(str(lig), str(tmp_path / "job"), target_ph=7.4,
                                                   smiles_hint="[H]/N=C(/N)c1ccccc1")
    assert out.endswith(".pdbqt") and os.path.isfile(out), info
    rows = [ln for ln in open(out) if ln.startswith(("ATOM", "HETATM"))]
    types = [ln[77:79].strip() for ln in rows]
    assert types.count("A") == 6                         # aromatic ring (not a cyclohexane)
    assert types.count("HD") == 4                        # NH2 + NH2+: amidinium at pH 7.4
    assert abs(sum(float(ln[70:76]) for ln in rows) - 1.0) < 0.05
    assert info["qc_status"] != "red"
