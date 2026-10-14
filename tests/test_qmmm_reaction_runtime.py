"""QM/MM reaction paths driven from Python with the real MOPAC binary (skipped when MOPAC is not installed):
relaxed distance scan inside a fixed MM environment, dimer transition-state search and frequency validation."""
import importlib.util
import shutil

import numpy as np
import pytest

from leonardo.qmmm_reaction_runtime import EmbeddedQMMMModel
from openmm_qmmm_runtime import run_openmm_qmmm_relaxed_scan, run_openmm_qmmm_ts_search

# propan-1-ol + water: QM = C3H2-O-H (C2-C3 cut -> one H link atom); MM = C1H3-C2H2 + water
_PROPANOL_SYM = ["C", "H", "H", "H", "C", "H", "H", "C", "H", "H", "O", "H", "O", "H", "H"]
_PROPANOL_XYZ = np.array([
    [-2.52, 0.00, 0.00], [-2.90, 1.02, 0.00], [-2.90, -0.51, 0.88], [-2.90, -0.51, -0.88],   # C1 H3  (MM)
    [-1.00, 0.00, 0.00], [-0.64, -0.51, 0.89], [-0.64, -0.51, -0.89],                        # C2 H2  (MM, cut)
    [-0.49, 1.44, 0.00], [-0.86, 1.95, 0.89], [-0.86, 1.95, -0.89],                          # C3 H2  (QM)
    [0.94, 1.44, 0.00], [1.25, 2.35, 0.00],                                                  # O H    (QM)
    [1.30, -1.30, 0.20], [1.00, -0.40, 0.40], [2.25, -1.35, 0.30]])                          # water  (MM)
_PROPANOL_Z = [{"C": 6, "H": 1, "O": 8}[s] for s in _PROPANOL_SYM]
_PROPANOL_QM = [7, 8, 9, 10, 11]
_PROPANOL_MM = [0, 1, 2, 3, 4, 5, 6, 12, 13, 14]
_PROPANOL_BONDS = [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6), (4, 7), (7, 8), (7, 9), (7, 10), (10, 11), (12, 13), (12, 14)]
_PROPANOL_Q = np.array([-0.18, 0.06, 0.06, 0.06, 0.12, 0.03, 0.03, 0, 0, 0, 0, 0, -0.834, 0.417, 0.417])
_PROPANOL_SIG = np.array([0.34, 0.25, 0.25, 0.25, 0.34, 0.25, 0.25, 0.34, 0.25, 0.25, 0.307, 0.0, 0.315, 0.0, 0.0])
_PROPANOL_EPS = np.array([0.46, 0.066, 0.066, 0.066, 0.46, 0.066, 0.066, 0.46, 0.066, 0.066, 0.71, 0.0, 0.636, 0.0, 0.0])

MOPAC = shutil.which("mopac") or shutil.which("MOPAC")
HAS_ASE = importlib.util.find_spec("ase") is not None
needs_mopac = pytest.mark.skipif(MOPAC is None or not HAS_ASE, reason="MOPAC and ASE are required")

# HCN -> HNC isomerization: H starts bridging above the C-N bond; one inert far-away MM site
_HCN_XYZ = np.array([[0.0, 0.0, 0.0], [1.18, 0.0, 0.0], [0.55, 1.05, 0.0], [30.0, 0.0, 0.0]])
_HCN_Z = [6, 7, 1, 1]


def _propanol_model(tmp_path):
    return EmbeddedQMMMModel(_PROPANOL_XYZ, _PROPANOL_Z, _PROPANOL_QM, _PROPANOL_MM, _PROPANOL_Q, str(tmp_path),
                             mopac_executable=MOPAC, bond_pairs=_PROPANOL_BONDS, lj_sigma_nm=_PROPANOL_SIG,
                             lj_epsilon_kj=_PROPANOL_EPS)


@needs_mopac
def test_relaxed_scan_reaches_targets_and_is_monotonic(tmp_path):
    scan = _propanol_model(tmp_path).relaxed_distance_scan(10, 11, [0.96, 1.10, 1.25])   # stretch the O-H bond
    rows = scan["rows"]
    assert all(r["converged"] for r in rows)
    for r in rows:
        assert abs(r["distance_ang"] - r["target_distance_ang"]) < 0.05                   # stiff enough restraint
    e = [r["energy_rel_kj_mol"] for r in rows]
    assert e[0] == 0.0 and e[0] < e[1] < e[2]                                            # stretching costs energy
    assert not scan["interior_maximum"]


@needs_mopac
def test_dimer_finds_the_hcn_hnc_saddle_with_one_imaginary_mode(tmp_path):
    model = EmbeddedQMMMModel(_HCN_XYZ, _HCN_Z, [0, 1, 2], [3], [0.0, 0.0, 0.0, 0.0], str(tmp_path), mopac_executable=MOPAC)
    mode = np.zeros((3, 3))
    mode[2] = [1.0, 0.0, 0.0]            # the H slides from C to N
    ts = model.refine_ts_dimer(_HCN_XYZ[:3], mode_guess=mode)
    assert ts["converged"] and ts["curvature_ev_ang2"] < 0.0
    x = ts["qm_xyz"]
    d_ch, d_nh = np.linalg.norm(x[2] - x[0]), np.linalg.norm(x[2] - x[1])
    assert 1.1 < d_ch < 1.6 and 1.1 < d_nh < 1.7                                        # H bridging C and N
    val = model.vibrations(x)["validation"]
    assert val["ok"] and val["imaginary_mode_count"] == 1 and val["primary_imag_cm1"] > 500.0


def _propanol_rows():
    return [{"index": i, "name": s + str(i), "resn": "LIG" if i in _PROPANOL_QM else "MM", "chain": "A", "resi": "1"}
            for i, s in enumerate(_PROPANOL_SYM)]


@needs_mopac
def test_openmm_relaxed_scan_returns_highest_point_probe_and_files(tmp_path):
    cfg = {"qm_indices": list(_PROPANOL_QM), "mm_indices": list(_PROPANOL_MM), "mopac_executable": MOPAC}
    out = run_openmm_qmmm_relaxed_scan(_PROPANOL_XYZ, _PROPANOL_Z, _PROPANOL_Q, _propanol_rows(), (10, 11), 0.96, 1.20,
                                       n_points=3, config=cfg, workdir=str(tmp_path), bond_pairs=_PROPANOL_BONDS,
                                       lj_sigma_nm=_PROPANOL_SIG, lj_epsilon_kj=_PROPANOL_EPS)
    assert out["ok"] is True and out["operation"] == "reaction_scan"
    scan = out["scan"]
    assert scan["relaxed"] and scan["n_points"] == 3 and scan["n_successful"] == 3
    assert scan["highest_point"]["point"] == 3 and scan["barrier_estimate_kj_mol"] > 0.0
    assert (tmp_path / "scan_path.xyz").is_file() and (tmp_path / "scan_profile.csv").is_file()
    assert out["qmmm"]["link_atoms"]["n_link"] == 1


@needs_mopac
def test_openmm_ts_search_validates_frequencies(tmp_path):
    rows = [{"index": i, "name": n, "resn": "HCN" if i < 3 else "DUM", "chain": "A", "resi": "1"}
            for i, n in enumerate(["C", "N", "H", "X"])]
    cfg = {"qm_indices": [0, 1, 2], "mm_indices": [3], "mopac_executable": MOPAC}
    out = run_openmm_qmmm_ts_search(_HCN_XYZ, _HCN_Z, [0.0, 0.0, 0.0, 0.0], rows, config=cfg, workdir=str(tmp_path),
                                    mode_pair=(0, 2))
    assert out["ok"] is True and out["operation"] == "transition_state"
    assert out["ts"]["converged"] and out["ts"]["frequency_validation"]["imaginary_mode_count"] == 1
    assert (tmp_path / "ts.xyz").is_file()


_HCN_BENT = np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.03, 0.30, 0.0], [30.0, 0.0, 0.0]])   # H on C, slightly bent


@needs_mopac
def test_transfer_scan_then_automatic_ts_refinement(tmp_path):
    """Scan xi = d(C-H) - d(H-N) across HCN -> HNC: smooth profile with an interior maximum; the dimer, started
    there with the path tangent, converges to the saddle point with one imaginary mode."""
    rows = [{"index": i, "name": n, "resn": "HCN" if i < 3 else "DUM", "chain": "A", "resi": "1"}
            for i, n in enumerate(["C", "N", "H", "X"])]
    cfg = {"qm_indices": [0, 1, 2], "mm_indices": [3], "mopac_executable": MOPAC}
    out = run_openmm_qmmm_relaxed_scan(_HCN_BENT, _HCN_Z, [0.0] * 4, rows, (0, 2), -1.0, 1.0, n_points=9,
                                       transfer_atom=1, config=cfg, workdir=str(tmp_path))
    scan = out["scan"]
    assert scan["coordinate"] == "transfer_xi" and scan["interior_maximum"]
    xi = [r["xi_ang"] for r in scan["rows"]]
    assert xi == sorted(xi)                                                    # the coordinate advances monotonically
    ts = out["ts"]
    assert ts["converged"] and ts["stopped_reason"] == ""
    assert ts["frequency_validation"]["imaginary_mode_count"] == 1 and ts["frequencies_cm1"][0] < -500.0
    assert ts["barrier_kj_mol"] > 0.0


@needs_mopac
def test_dimer_stops_instead_of_climbing_from_a_minimum(tmp_path):
    """From a minimum there is no negative curvature: the search must stop with a reason, not wander uphill."""
    model = EmbeddedQMMMModel(_HCN_XYZ, _HCN_Z, [0, 1, 2], [3], [0.0] * 4, str(tmp_path), mopac_executable=MOPAC)
    start = np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.07, 0.05, 0.0]])          # ~linear HCN (a minimum)
    ts = model.refine_ts_dimer(start, mode_guess=None, max_positive_curvature_steps=5, steps=100)
    assert ts["converged"] is False and "no negative curvature" in ts["stopped_reason"]
    assert ts["steps"] < 20


def test_scan_barrier_is_measured_from_the_reactant_minimum():
    from leonardo.qmmm_reaction_runtime import scan_summary

    # the scan starts 2.7 kJ/mol above the reactant minimum (as in the DhlA SN2 scan)
    energies = [0.0, -2.7, -2.1, 1.2, 8.2, 18.7, 28.7, 31.3, 20.5, -3.9, -32.0, -55.2, -71.4]
    rows = [{"point": i + 1, "energy_kj_mol": e} for i, e in enumerate(energies)]
    out = scan_summary(rows)
    assert out["highest_point"]["point"] == 8 and out["reactant_point"]["point"] == 2
    assert abs(out["barrier_estimate_kj_mol"] - 34.0) < 1e-9
    assert abs(out["barrier_from_first_point_kj_mol"] - 31.3) < 1e-9
    assert abs(out["reaction_energy_kj_mol"] - (-71.4 + 2.7)) < 1e-9
    assert out["interior_maximum"] and rows[1]["energy_rel_kj_mol"] == -2.7
