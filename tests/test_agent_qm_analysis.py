"""MOPAC output parsing, orbital labels, Slater orbitals/cubes and ESP scaling for the QM analysis tools."""
import math
import shutil

import numpy as np
import pytest

from leonardo.agent.registry import ToolError
from leonardo.agent.tools import mopac, qmviz

AUX = """ AO_ATOMINDEX[004]=
  1  1  1  1
 ATOM_SYMTYPE[004]=
  S PX PY PZ
 AO_ZETA[004]=
  1.500000 1.200000 1.200000
  1.200000
 ATOM_PQN[004]=
 2 2 2 2
 HEAT_OF_FORMATION:KCAL/MOL=-0.481295758515372D+02
 EIGENVALUES[004]=
  -20.0 -10.5  2.0  3.0
 EIGENVECTORS[00016]=
  1 0 0 0  0 1 0 0
  0 0 1 0  0 0 0 1
 MOLECULAR_ORBITAL_OCCUPANCIES[00004]=
 2.0 2.0 0.0 0.0
# comment line with 3 numbers 1 2 3
"""

CIS_OUT = """
  STATE       ENERGY (EV)        Q.N.  SPIN   SYMMETRY              POLARIZATION
         ABSOLUTE     RELATIVE                                X           Y           Z

    1+   -0.000000    0.000000     1+  SINGLET     A'
    2     4.103022    4.103022     1   TRIPLET     A"
    3     5.194164    5.194164     1   SINGLET     A"        0.0000      0.0000      0.0078
    4     6.575881    6.575881     2   SINGLET     A'        0.0485      0.0058      0.0000
    5    10.338233   10.338233     5   SINGLET     A'        0.2686      0.1100      0.0000

           The "+" symbol indicates the root used.
"""


def test_aux_blocks_scalars_and_scf():
    assert mopac.aux_block(AUX, "AO_ZETA") == [1.5, 1.2, 1.2, 1.2]
    assert mopac.aux_block(AUX, "ATOM_SYMTYPE", as_str=True) == ["S", "PX", "PY", "PZ"]
    assert mopac.aux_scalar(AUX, "HEAT_OF_FORMATION") == pytest.approx(-48.1295758515372)
    scf = mopac.parse_scf(AUX)
    assert scf["homo_index"] == 2 and scf["coefficients"].shape == (4, 4)
    assert scf["basis"][1] == (0, "PX", 1.2, 2)


def test_cis_singlets_with_relative_intensity():
    states = mopac.parse_cis(CIS_OUT)
    assert [s["state"] for s in states] == [3, 4, 5]             # the ground state and triplets are left out
    assert states[-1]["relative_intensity"] == 1.0
    assert states[0]["wavelength_nm"] == pytest.approx(1239.84193 / 5.194164, abs=0.1)


def test_orbital_labels():
    assert qmviz.resolve_orbitals(["homo", "lumo", "homo-2", "lumo+1", "7"], 5) == [
        ("HOMO", 5), ("LUMO", 6), ("HOMO-2", 3), ("LUMO+1", 7), ("MO 7 (LUMO+1)", 7)]
    with pytest.raises(ToolError):
        qmviz.resolve_orbitals(["sigma*"], 5)


def test_slater_orbitals_are_normalized_and_cube_round_trips(tmp_path):
    basis = [(0, "S", 1.0, 1), (0, "PZ", 1.3, 2), (0, "XY", 1.8, 3)]
    lo, h, n = qmviz.grid_for([[0.0, 0.0, 0.0]], spacing_bohr=0.15, margin_bohr=12.0)
    mos, dens = qmviz.sto_grid_values(basis, np.eye(3), [[0.0, 0.0, 0.0]], lo, h, n, occupations=[2, 0, 0])
    for row in mos:
        assert (row ** 2).sum() * h ** 3 == pytest.approx(1.0, abs=0.03)
    assert dens.sum() * h ** 3 == pytest.approx(2.0, abs=0.06)
    cube = qmviz.write_cube(str(tmp_path / "s.cube"), ["C"], [[0.0, 0.0, 0.0]], lo, h, n, mos[0])
    assert np.allclose(qmviz.read_cube_values(cube), mos[0], rtol=1e-4, atol=1e-8)


def test_charged_molecule_gets_a_surface_range_not_a_symmetric_one():
    dens = np.array([0.002] * 50 + [0.1] * 50)
    esp = np.concatenate([np.linspace(0.08, 0.23, 50), np.full(50, 1.0)])
    lo, hi = qmviz.surface_esp_range(dens, esp, 0.002)
    assert 0.08 <= lo < hi <= 0.23


def test_spectrum_images(tmp_path):
    ir = qmviz.ir_png([500, 1700, 3000], [1, 3, 2], str(tmp_path / "ir.png"), "IR")
    uv = qmviz.uv_png([5.1, 6.6], [0.27, 0.43], str(tmp_path / "uv.png"), "UV")
    assert (tmp_path / "ir.png").stat().st_size > 5000 and (tmp_path / "uv.png").stat().st_size > 5000
    assert ir and uv and math.isclose(qmviz.EV_NM / 5.1, 243.1, abs_tol=0.1)


@pytest.mark.skipif(not mopac.mopac_executable() or not shutil.which("mopac") and not mopac.mopac_executable(),
                    reason="MOPAC not installed")
def test_real_mopac_water(tmp_path):
    res = mopac.run_mopac_job(mopac.mopac_executable(), str(tmp_path), ["O", "H", "H"],
                              [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]], "PM7", 0, 1, None, True, True, True, 3, {})
    assert res["homo_index"] == 4 and len(res["eigenvalues_ev"]) == 6
    assert len(res["freq_cm"]) == 3 and all(f > 1000 for f in res["freq_cm"])
    assert res["uv_states"] and abs(sum(res["esp_charges"])) < 0.01
