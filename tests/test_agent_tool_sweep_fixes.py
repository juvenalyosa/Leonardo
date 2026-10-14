"""Regressions found by driving the MD / QM tools headlessly on small systems (peptides, ions, radicals)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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
