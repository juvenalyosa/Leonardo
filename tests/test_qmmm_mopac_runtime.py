import math
from types import SimpleNamespace

import numpy as np

from qmmm_mopac_runtime import EV_TO_HARTREE
from qmmm_mopac_runtime import OPENMM_COULOMB_CONSTANT
from qmmm_mopac_runtime import assemble_force_correction
from qmmm_mopac_runtime import build_mopac_input_text
from qmmm_mopac_runtime import build_mopac_reaction_input_text
from qmmm_mopac_runtime import compute_phi_on_qm_atoms
from qmmm_mopac_runtime import electrostatic_backreaction_openmm
from qmmm_mopac_runtime import mopac_build_irc_profile_payload
from qmmm_mopac_runtime import mopac_reaction_keywords_for_stage
from qmmm_mopac_runtime import mopac_validate_ts_frequencies
from qmmm_mopac_runtime import parse_mopac_arc_geometries
from qmmm_mopac_runtime import parse_mopac_aux_text
from qmmm_mopac_runtime import qmmm_assess_interface
from qmmm_mopac_runtime import qmmm_mopac_keywords_for_operation
from qmmm_mopac_runtime import qmmm_recommend_region
from qmmm_mopac_runtime import run_mopac_reaction_workflow
from qmmm_mopac_runtime import qmmm_step_file_based


def test_compute_phi_on_qm_atoms_positive_and_distance_dependent():
    xyz_qm = np.array([[0.0, 0.0, 0.0]], dtype=float)
    xyz_mm_near = np.array([[1.0, 0.0, 0.0]], dtype=float)
    xyz_mm_far = np.array([[2.0, 0.0, 0.0]], dtype=float)
    q_mm = np.array([1.0], dtype=float)

    phi_near = compute_phi_on_qm_atoms(xyz_qm, xyz_mm_near, q_mm)
    phi_far = compute_phi_on_qm_atoms(xyz_qm, xyz_mm_far, q_mm)

    assert phi_near.shape == (1,)
    assert phi_near[0] > 0.0
    assert phi_near[0] > phi_far[0]


def test_electrostatic_backreaction_openmm_newton_third_law_pair():
    xyz_qm = np.array([[0.0, 0.0, 0.0]], dtype=float)
    xyz_mm = np.array([[1.0, 0.0, 0.0]], dtype=float)  # 0.1 nm
    q_qm = np.array([1.0], dtype=float)
    q_mm = np.array([-1.0], dtype=float)

    res = electrostatic_backreaction_openmm(xyz_qm, xyz_mm, q_qm, q_mm)
    fq = np.asarray(res["forces_qm_kj_mol_nm"], dtype=float)
    fm = np.asarray(res["forces_mm_kj_mol_nm"], dtype=float)
    e = float(res["energy_kj_mol"])

    expected_energy = -OPENMM_COULOMB_CONSTANT / 0.1
    assert math.isclose(e, expected_energy, rel_tol=1e-8, abs_tol=1e-8)
    assert fq.shape == (1, 3)
    assert fm.shape == (1, 3)
    assert np.allclose(fq[0] + fm[0], np.zeros(3), atol=1e-10)
    assert fq[0, 0] > 0.0
    assert abs(fq[0, 1]) < 1e-12
    assert abs(fq[0, 2]) < 1e-12


def test_assemble_force_correction_formula_blocks():
    n_atoms = 4
    qm_idx = [0, 1]
    mm_idx = [2, 3]
    f_mm_qm_internal = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    f_mm_qmmm_coul = np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    f_qm_internal = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=float)
    f_qmmm_elec_qm = np.array([[0.2, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=float)
    f_qmmm_elec_mm = np.array([[-0.4, 0.0, 0.0], [-0.6, 0.0, 0.0]], dtype=float)

    corr = assemble_force_correction(
        n_atoms=n_atoms,
        qm_indices=qm_idx,
        mm_indices=mm_idx,
        f_mm_qm_internal=f_mm_qm_internal,
        f_mm_qmmm_coul=f_mm_qmmm_coul,
        f_qm_internal=f_qm_internal,
        f_qmmm_elec_qm=f_qmmm_elec_qm,
        f_qmmm_elec_mm=f_qmmm_elec_mm,
    )

    assert corr.shape == (n_atoms, 3)
    assert math.isclose(float(corr[0, 0]), -1.0 - 0.5 + 4.0 + 0.2, rel_tol=1e-12)
    assert math.isclose(float(corr[1, 0]), -2.0 - 1.0 + 8.0 + 0.3, rel_tol=1e-12)
    assert math.isclose(float(corr[2, 0]), -0.4, rel_tol=1e-12)
    assert math.isclose(float(corr[3, 0]), -0.6, rel_tol=1e-12)


def test_parse_mopac_aux_text_extracts_energy_charges_gradients():
    aux_text = """
TOTAL_ENERGY:EV= -25.0000
ATOM_CHARGES[2]=
  0.10000000 -0.10000000
GRADIENTS:KCAL/MOL/ANGSTROM[6]=
  1.00000000 0.00000000 0.00000000
 -1.00000000 0.00000000 0.00000000
"""
    parsed = parse_mopac_aux_text(aux_text)
    assert parsed["ok"] is True
    assert math.isclose(float(parsed["energy_hartree"]), -25.0 * EV_TO_HARTREE, rel_tol=1e-12)
    charges = np.asarray(parsed["atom_charges"], dtype=float)
    grads = np.asarray(parsed["gradients_kj_mol_nm"], dtype=float)
    assert charges.shape == (2,)
    assert np.allclose(charges, np.array([0.1, -0.1]), atol=1e-12)
    assert grads.shape == (2, 3)
    assert math.isclose(float(grads[0, 0]), 41.84, rel_tol=1e-12)


def test_build_mopac_input_text_includes_charge_and_multiplicity():
    text = build_mopac_input_text(
        qm_atomic_numbers=[6, 8],
        xyz_qm_ang=np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=float),
        keywords="PM7 1SCF GRAD AUX QMMM",
        total_charge=-1,
        multiplicity=2,
        title="toy",
    )
    first = str(text.splitlines()[0]).upper()
    assert "CHARGE=-1" in first
    assert "DOUBLET" in first
    assert "QMMM" in first


def test_qmmm_mopac_keywords_remove_operation_conflicts():
    kw_opt = qmmm_mopac_keywords_for_operation("optimization", "PM6 1SCF GRAD AUX QMMM")
    assert "PM6" in kw_opt
    assert "EF" in kw_opt
    assert "1SCF" not in kw_opt
    assert "GRAD" not in kw_opt

    kw_thermo = qmmm_mopac_keywords_for_operation("thermochemistry", "PM7 EF AUX")
    assert "FORCE" in kw_thermo
    assert "THERMO" in kw_thermo
    assert "EF" not in kw_thermo


def test_mopac_reaction_keywords_are_stage_specific_and_safe():
    qst2 = mopac_reaction_keywords_for_stage("qst2", base_keywords="PM6 1SCF GRAD QMMM")
    assert "PM6" in qst2
    assert "SADDLE" in qst2
    assert "1SCF" not in qst2
    assert "QMMM" not in qst2

    ts = mopac_reaction_keywords_for_stage("ts", base_keywords="PM7 EF 1SCF")
    assert "TS" in ts
    assert "EF" not in ts
    assert "1SCF" not in ts

    irc = mopac_reaction_keywords_for_stage("irc", base_keywords="PM7 TS EF 1SCF", irc_direction="both", x_priority_ang=0.025)
    assert "IRC=1*" in irc
    assert "FORCE" in irc
    assert "X-PRIORITY=0.0250" in irc
    assert "TS" not in irc
    assert "1SCF" not in irc


def test_build_mopac_reaction_input_text_writes_two_ordered_geometries():
    reactant = [
        {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0},
        {"element": "O", "x": 1.2, "y": 0.0, "z": 0.0},
    ]
    product = [
        {"element": "C", "x": 0.1, "y": 0.0, "z": 0.0},
        {"element": "O", "x": 1.3, "y": 0.0, "z": 0.0},
    ]
    text = build_mopac_reaction_input_text(
        reactant,
        product_atoms=product,
        keywords="PM7 SADDLE AUX XYZ",
        total_charge=-1,
        multiplicity=2,
    )
    first = text.splitlines()[0].upper()
    assert "SADDLE" in first
    assert "CHARGE=-1" in first
    assert "DOUBLET" in first
    assert text.count("\nC ") == 2
    assert text.count("\nO ") == 2


def test_parse_mopac_aux_text_preserves_negative_ts_frequency():
    aux = """
ATOM_EL[3]=
 C O H
ATOM_X_OPT:ANGSTROMS[9]=
 0.0 0.0 0.0
 1.2 0.0 0.0
 -0.8 0.0 0.0
TOTAL_ENERGY:EV= -25.0
ATOM_CHARGES[3]=
 0.1 -0.2 0.1
GRADIENTS:KCAL/MOL/ANGSTROM[9]=
 0.0 0.0 0.0
 0.0 0.0 0.0
 0.0 0.0 0.0
VIB._FREQ:CM(-1)[3]=
 -735.5 112.0 245.0
"""
    parsed = parse_mopac_aux_text(aux)
    assert parsed["ok"] is True
    assert parsed["atom_symbols"] == ["C", "O", "H"]
    assert np.asarray(parsed["optimized_coords_ang"]).shape == (3, 3)
    assert np.allclose(np.asarray(parsed["freq_cm_signed"]), np.array([-735.5, 112.0, 245.0]))
    val = mopac_validate_ts_frequencies(parsed["freq_cm_signed"], parsed["freq_cm_imag"])
    assert val["ok"] is True
    assert val["imaginary_mode_count"] == 1
    assert math.isclose(float(val["primary_imag_cm1"]), 735.5, rel_tol=1e-12)


def test_parse_mopac_arc_geometries_and_irc_profile():
    arc = """
 HEAT OF FORMATION =       10.0000 KCAL/MOL
 C      0.00000000 1      0.00000000 1      0.00000000 1
 H      1.00000000 1      0.00000000 1      0.00000000 1

 HEAT OF FORMATION =       25.0000 KCAL/MOL
 C      0.10000000 1      0.00000000 1      0.00000000 1
 H      1.10000000 1      0.00000000 1      0.00000000 1

 HEAT OF FORMATION =       12.0000 KCAL/MOL
 C      0.20000000 1      0.00000000 1      0.00000000 1
 H      1.20000000 1      0.00000000 1      0.00000000 1
"""
    parsed = parse_mopac_arc_geometries(arc, expected_atoms=2)
    assert parsed["ok"] is True
    coords = np.asarray(parsed["coords_ang"], dtype=float)
    energy = np.asarray(parsed["energy_hartree"], dtype=float)
    assert coords.shape == (3, 2, 3)
    assert energy.shape == (3,)
    profile = mopac_build_irc_profile_payload(coords, energy, parsed["symbols"])
    assert profile["ok"] is True
    assert profile["ts_frame_index"] == 1
    assert profile["n_points"] == 3


def test_run_mopac_reaction_workflow_mock_qst2_ts_freq_irc(tmp_path):
    reactant = [
        {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0},
        {"element": "O", "x": 1.25, "y": 0.0, "z": 0.0},
        {"element": "H", "x": -0.75, "y": 0.0, "z": 0.0},
    ]
    product = [
        {"element": "C", "x": 0.0, "y": 0.0, "z": 0.0},
        {"element": "O", "x": 1.05, "y": 0.0, "z": 0.0},
        {"element": "H", "x": -0.45, "y": 0.0, "z": 0.0},
    ]
    seen_keywords = []

    def _write_aux(cwd, coords, freqs=None):
        freq_block = ""
        if freqs is not None:
            freq_block = "VIB._FREQ:CM(-1)[3]=\n " + " ".join(str(x) for x in freqs) + "\n"
        aux = (
            "ATOM_EL[3]=\n C O H\n"
            "ATOM_X_OPT:ANGSTROMS[9]=\n"
            f" {coords[0][0]} {coords[0][1]} {coords[0][2]}\n"
            f" {coords[1][0]} {coords[1][1]} {coords[1][2]}\n"
            f" {coords[2][0]} {coords[2][1]} {coords[2][2]}\n"
            "TOTAL_ENERGY:EV= -20.0\n"
            "ATOM_CHARGES[3]=\n 0.1 -0.2 0.1\n"
            "GRADIENTS:KCAL/MOL/ANGSTROM[9]=\n 0 0 0 0 0 0 0 0 0\n"
            + freq_block
        )
        (cwd / "job.aux").write_text(aux, encoding="utf-8")

    def _runner(cmdline, cwd=None, capture_output=None, text=None, timeout=None):
        import pathlib

        cdir = pathlib.Path(cwd)
        first_line = (cdir / "job.mop").read_text(encoding="utf-8").splitlines()[0].upper()
        seen_keywords.append(first_line)
        stage = cdir.name
        if "01_qst2" in stage:
            _write_aux(cdir, [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0], [-0.60, 0.0, 0.0]])
        elif "02_ts" in stage:
            _write_aux(cdir, [[0.0, 0.0, 0.0], [1.12, 0.0, 0.0], [-0.58, 0.0, 0.0]])
        elif "03_ts_frequency" in stage:
            _write_aux(cdir, [[0.0, 0.0, 0.0], [1.12, 0.0, 0.0], [-0.58, 0.0, 0.0]], freqs=[-550.0, 100.0, 200.0])
        elif "04_irc" in stage:
            _write_aux(cdir, [[0.0, 0.0, 0.0], [1.12, 0.0, 0.0], [-0.58, 0.0, 0.0]])
            arc = """
 HEAT OF FORMATION =       1.0000 KCAL/MOL
 C      0.00000000 1      0.00000000 1      0.00000000 1
 O      1.30000000 1      0.00000000 1      0.00000000 1
 H     -0.80000000 1      0.00000000 1      0.00000000 1

 HEAT OF FORMATION =       5.0000 KCAL/MOL
 C      0.00000000 1      0.00000000 1      0.00000000 1
 O      1.12000000 1      0.00000000 1      0.00000000 1
 H     -0.58000000 1      0.00000000 1      0.00000000 1

 HEAT OF FORMATION =       2.0000 KCAL/MOL
 C      0.00000000 1      0.00000000 1      0.00000000 1
 O      1.00000000 1      0.00000000 1      0.00000000 1
 H     -0.40000000 1      0.00000000 1      0.00000000 1
"""
            (cdir / "job.arc").write_text(arc, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    out = run_mopac_reaction_workflow(
        reactant,
        product_atoms=product,
        operation="mopac_reaction_path",
        base_keywords="PM7 1SCF QMMM",
        workdir=str(tmp_path),
        runner=_runner,
        irc_max_points=25,
    )
    assert out["ok"] is True
    assert len(out["stages"]) == 4
    assert out["ts_validation"]["ok"] is True
    assert out["irc_profile"]["ok"] is True
    assert out["irc_profile"]["n_points"] == 3
    assert (tmp_path / "mopac_irc_profile.csv").is_file()
    assert (tmp_path / "mopac_irc_path.xyz").is_file()
    assert any("SADDLE" in kw for kw in seen_keywords)
    assert any("TS" in kw for kw in seen_keywords)
    assert all("1SCF" not in kw for kw in seen_keywords)


def test_qmmm_recommend_region_adds_nearby_pocket_and_metal():
    atom_rows = [
        {"index": 0, "name": "CA", "resn": "ALA", "chain": "A", "resi": "1", "element": "C", "x": 8.0, "y": 0.0, "z": 0.0},
        {"index": 1, "name": "CB", "resn": "ALA", "chain": "A", "resi": "1", "element": "C", "x": 8.5, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "CG", "resn": "PHE", "chain": "A", "resi": "2", "element": "C", "x": 2.5, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "CZ", "resn": "PHE", "chain": "A", "resi": "2", "element": "C", "x": 2.9, "y": 0.0, "z": 0.0},
        {"index": 4, "name": "C1", "resn": "LIG", "chain": "L", "resi": "10", "element": "C", "x": 0.0, "y": 0.0, "z": 0.0},
        {"index": 5, "name": "O1", "resn": "LIG", "chain": "L", "resi": "10", "element": "O", "x": 1.0, "y": 0.0, "z": 0.0},
        {"index": 6, "name": "ZN", "resn": "ZN", "chain": "Z", "resi": "1", "element": "ZN", "x": 2.2, "y": 0.0, "z": 0.0},
    ]

    rec = qmmm_recommend_region(
        atom_rows=atom_rows,
        operation="optimization",
        qm_pocket_cutoff_ang=3.2,
        include_pocket=True,
        include_metals=True,
        max_qm_atoms=50,
    )

    assert rec["ok"] is True
    assert set(rec["qm_indices"]) == {2, 3, 4, 5, 6}
    assert set(rec["mm_indices"]) == {0, 1}
    assert rec["center_residue"]["resn"] == "LIG"


def test_qmmm_assess_interface_flags_covalent_boundary():
    atom_rows = [
        {"index": 0, "name": "C1", "resn": "LIG", "chain": "L", "resi": "1", "element": "C", "x": 0.0, "y": 0.0, "z": 0.0},
        {"index": 1, "name": "SG", "resn": "CYS", "chain": "A", "resi": "45", "element": "S", "x": 1.7, "y": 0.0, "z": 0.0},
    ]

    iface = qmmm_assess_interface(atom_rows=atom_rows, qm_indices=[0], mm_indices=[1], bond_pairs=[(0, 1)])
    assert iface["ok"] is False
    assert iface["cross_boundary_bonds"]
    assert any("Covalent" in str(w) for w in iface["warnings"])


def test_qmmm_step_file_based_smoke_with_mock_runner(tmp_path):
    xyz_full = np.array(
        [
            [0.0, 0.0, 0.0],  # QM 0
            [1.0, 0.0, 0.0],  # QM 1
            [3.0, 0.0, 0.0],  # MM 2
        ],
        dtype=float,
    )
    atomic_numbers = [6, 8, 11]
    qm_indices = [0, 1]
    mm_indices = [2]
    q_mm = [1.0]

    def _runner(cmdline, cwd=None, capture_output=None, text=None, timeout=None):
        assert isinstance(cmdline, list)
        assert str(cmdline[0]) == "mopac"
        assert str(cmdline[1]).endswith(".mop")
        assert cwd is not None
        mop_path = tmp_path / "step.mop"
        mol_in_path = tmp_path / "mol.in"
        assert mop_path.is_file()
        assert mol_in_path.is_file()
        aux_text = """
TOTAL_ENERGY:EV= -40.0000
ATOM_CHARGES[2]=
 0.20000000 -0.20000000
GRADIENTS:KCAL/MOL/ANGSTROM[6]=
  1.00000000 0.00000000 0.00000000
 -2.00000000 0.00000000 0.00000000
"""
        (tmp_path / "step.aux").write_text(aux_text, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    out = qmmm_step_file_based(
        xyz_full_ang=xyz_full,
        atomic_numbers_full=atomic_numbers,
        qm_indices=qm_indices,
        mm_indices=mm_indices,
        q_mm=q_mm,
        workdir=str(tmp_path),
        runner=_runner,
    )
    assert out["ok"] is True
    assert out["qm_indices"] == qm_indices
    assert out["mm_indices"] == mm_indices
    assert math.isclose(float(out["energy_qm_hartree"]), -40.0 * EV_TO_HARTREE, rel_tol=1e-12)
    f_qm = np.asarray(out["forces_qm_internal_kj_mol_nm"], dtype=float)
    assert f_qm.shape == (2, 3)
    assert math.isclose(float(f_qm[0, 0]), -41.84, rel_tol=1e-12)
    assert math.isclose(float(f_qm[1, 0]), 83.68, rel_tol=1e-12)
    f_elec_qm = np.asarray(out["forces_qmmm_elec_qm_kj_mol_nm"], dtype=float)
    f_elec_mm = np.asarray(out["forces_qmmm_elec_mm_kj_mol_nm"], dtype=float)
    assert f_elec_qm.shape == (2, 3)
    assert f_elec_mm.shape == (1, 3)
    assert np.allclose(np.sum(f_elec_qm, axis=0) + np.sum(f_elec_mm, axis=0), np.zeros(3), atol=1e-8)


def test_qmmm_step_file_based_preserves_operation_keywords(tmp_path):
    xyz_full = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    seen_mopac_text = {}

    def _runner(cmdline, cwd=None, capture_output=None, text=None, timeout=None):
        mop_path = tmp_path / "step.mop"
        seen_mopac_text["text"] = mop_path.read_text(encoding="utf-8")
        aux_text = """
TOTAL_ENERGY:EV= -10.0000
ATOM_CHARGES[1]=
 0.00000000
GRADIENTS:KCAL/MOL/ANGSTROM[3]=
  0.00000000 0.00000000 0.00000000
"""
        (tmp_path / "step.aux").write_text(aux_text, encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    out = qmmm_step_file_based(
        xyz_full_ang=xyz_full,
        atomic_numbers_full=[6, 8],
        qm_indices=[0],
        mm_indices=[1],
        q_mm=[0.0],
        mopac_keywords="PM7 EF AUX QMMM",
        workdir=str(tmp_path),
        runner=_runner,
    )

    assert out["ok"] is True
    first_line = seen_mopac_text["text"].splitlines()[0].upper()
    assert "EF" in first_line
    assert "QMMM" in first_line
    assert "1SCF" not in first_line


# ---------------------------------------------------------------------------
# mol.in format, energy units and element symbols (regressions found with real MOPAC 23.2.3)
# ---------------------------------------------------------------------------
from qmmm_mopac_runtime import HARTREE_TO_KCAL, KCAL_TO_HARTREE, build_mopac_mol_in_text  # noqa: E402


def test_build_mopac_mol_in_text_layout_and_units():
    phi_au = np.array([0.1, -0.05])
    txt = build_mopac_mol_in_text(phi_au, ["O", "H"], np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]]))
    lines = txt.strip().splitlines()
    assert len(lines) == 4                                   # title, counts, one line per QM atom
    assert lines[1].split() == ["2", "0"]                    # QM atoms, link atoms
    sym, x, y, z, v = lines[2].split()
    assert sym == "O" and float(x) == 0.0
    assert math.isclose(float(v), 0.1 * HARTREE_TO_KCAL, rel_tol=1e-9)        # kcal/mol/e, not Eh/e
    assert math.isclose(float(lines[3].split()[4]), -0.05 * HARTREE_TO_KCAL, rel_tol=1e-9)


def test_build_mopac_mol_in_text_rejects_mismatched_sizes():
    import pytest
    with pytest.raises(ValueError):
        build_mopac_mol_in_text([0.1, 0.2], ["O"], np.zeros((2, 3)))


def test_parse_mopac_aux_heat_of_formation_is_kcal_not_ev():
    # A plain PM7 1SCF GRAD job writes only HEAT_OF_FORMATION (kcal/mol, Fortran D exponent).
    aux_text = """
HEAT_OF_FORMATION:KCAL/MOL=-0.577823D+02
ATOM_CHARGES[03]=
 -0.64500 0.32300 0.32300
GRADIENTS:KCAL/MOL/ANGSTROM[09]=
 1.0 0.0 0.0 -0.5 0.0 0.0 -0.5 0.0 0.0
"""
    parsed = parse_mopac_aux_text(aux_text)
    assert parsed["ok"] is True
    assert parsed["energy_source"] == "heat_of_formation_kcal_mol"
    assert math.isclose(parsed["energy_hartree"], -57.7823 * KCAL_TO_HARTREE, rel_tol=1e-9)
    assert math.isclose(parsed["heat_of_formation_kcal_mol"], -57.7823, rel_tol=1e-9)
    assert math.isclose(parsed["heat_of_formation_ev"], -57.7823 / 23.060548, rel_tol=1e-5)


def test_parse_mopac_aux_takes_final_gradients_not_the_first_optimization_step():
    # Layout of a real MOPAC 23 "PM7 EF GRAD AUX" water run: every optimization step writes
    # GRADIENTS_UPDATED before the final GRADIENTS block of the converged geometry.
    aux_text = """
 ATOM_EL[03]=
   O  H  H
 HEAT_OF_FORM_UPDATED:KCAL/MOL=-0.488821D+02
 GRADIENT_NORM_UPDATED:KCAL/MOL/ANG=+0.121321D+03
 ATOM_X_UPDATED:ANGSTROMS[09]=
    0.0000    0.0000    0.0000
    1.1000    0.0000    0.0000
   -0.3000    1.0000    0.0000
 GRADIENTS_UPDATED:KCAL/MOL/ANGSTROM[09]=
  -56.4973  -39.9191   0.0000  82.0217  -9.6842   0.0000 -25.5245  49.6032   0.0000
 HEAT_OF_FORMATION:KCAL/MOL=-0.577998D+02
 GRADIENT_NORM:KCAL/MOL/ANGSTROM=+0.302314D+00
 ATOM_X_OPT:ANGSTROMS[09]=
    0.0191    0.0110    0.0000
    0.9666   -0.0021    0.0000
   -0.2397    0.9419    0.0000
 ATOM_CHARGES[03]=
  -0.64918  0.32462  0.32456
 GRADIENTS:KCAL/MOL/ANGSTROM[09]=
  -0.2362  -0.0117   0.0000   0.1615   0.0493   0.0000   0.0747  -0.0376   0.0000
"""
    parsed = parse_mopac_aux_text(aux_text)
    assert parsed["ok"] is True
    final = np.array([[-0.2362, -0.0117, 0.0], [0.1615, 0.0493, 0.0], [0.0747, -0.0376, 0.0]])
    assert np.allclose(parsed["gradients_kcal_mol_ang"], final)
    assert np.allclose(parsed["optimized_coords_ang"][0], [0.0191, 0.0110, 0.0])
    assert math.isclose(parsed["heat_of_formation_kcal_mol"], -57.7998, rel_tol=1e-9)


def test_mopac_input_writer_knows_transition_metals_and_rejects_unknown():
    import pytest
    txt = build_mopac_input_text([25, 8], np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    assert txt.splitlines()[3].split()[0] == "Mn"            # used to be written silently as carbon
    with pytest.raises(ValueError):
        build_mopac_input_text([0], np.zeros((1, 3)))


def _water_step(q_mm, tmp_path):
    import shutil
    exe = shutil.which("mopac") or shutil.which("MOPAC")
    xyz = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0], [-3.0, 0.0, 0.0]])
    return qmmm_step_file_based(
        xyz_full_ang=xyz, atomic_numbers_full=[8, 1, 1, 11], qm_indices=[0, 1, 2], mm_indices=[3],
        q_mm=[q_mm], mopac_executable=exe, workdir=str(tmp_path), timeout_s=60,
    )


def test_qmmm_step_runs_with_real_mopac(tmp_path):
    """End-to-end with the real MOPAC binary: the mocked tests above never exercised mol.in."""
    import shutil
    import pytest
    if not (shutil.which("mopac") or shutil.which("MOPAC")):
        pytest.skip("MOPAC is not installed")
    neutral = _water_step(0.0, tmp_path / "neutral")
    charged = _water_step(1.0, tmp_path / "charged")
    assert neutral["ok"] and charged["ok"]
    # zero external charge reproduces the gas-phase PM7 heat of formation of water (-57.8 kcal/mol)
    assert math.isclose(neutral["heat_of_formation_kcal_mol"], -57.78, abs_tol=0.1)
    # a +1 charge 3 A from the oxygen polarizes the water (O more negative) and stabilizes it
    assert charged["q_qm"][0] < neutral["q_qm"][0] - 0.03
    assert charged["energy_qm_hartree"] < neutral["energy_qm_hartree"]
    assert (charged["heat_of_formation_kcal_mol"] - neutral["heat_of_formation_kcal_mol"]) < -2.0


# ---------------------------------------------------------------------------
# Link atoms + QM–MM Lennard-Jones: analytic forces vs finite differences with the real MOPAC binary
# ---------------------------------------------------------------------------
from qmmm_mopac_runtime import qmmm_link_atom_plan, qmmm_redistribute_boundary_charges  # noqa: E402

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


def test_link_atom_plan_caps_the_cut_bond_and_moves_the_boundary_charge():
    plan = qmmm_link_atom_plan(_PROPANOL_XYZ, _PROPANOL_Z, _PROPANOL_QM, _PROPANOL_MM, bond_pairs=_PROPANOL_BONDS)
    assert [(L["q"], L["m"]) for L in plan["links"]] == [(7, 4)]
    g = plan["links"][0]["g"]
    assert math.isclose(g * np.linalg.norm(_PROPANOL_XYZ[4] - _PROPANOL_XYZ[7]), 1.09, rel_tol=1e-9)   # C-H 1.09 A
    assert plan["charge_redistribution"] == {4: [0, 5, 6]}
    q = qmmm_redistribute_boundary_charges(_PROPANOL_Q[_PROPANOL_MM], _PROPANOL_MM, plan["charge_redistribution"])
    assert q[_PROPANOL_MM.index(4)] == 0.0
    assert math.isclose(q.sum(), _PROPANOL_Q[_PROPANOL_MM].sum(), abs_tol=1e-12)      # total charge conserved
    assert (7, 4) in plan["lj_excluded_pairs"] and (10, 12) not in plan["lj_excluded_pairs"]


def test_link_atom_and_lj_forces_match_finite_differences_with_real_mopac(tmp_path):
    import shutil
    import pytest
    exe = shutil.which("mopac") or shutil.which("MOPAC")
    if not exe:
        pytest.skip("MOPAC is not installed")
    plan = qmmm_link_atom_plan(_PROPANOL_XYZ, _PROPANOL_Z, _PROPANOL_QM, _PROPANOL_MM, bond_pairs=_PROPANOL_BONDS)

    def run(x, tag):
        return qmmm_step_file_based(
            x, _PROPANOL_Z, _PROPANOL_QM, _PROPANOL_MM, _PROPANOL_Q[_PROPANOL_MM], mopac_executable=exe,
            mopac_keywords="PM7 1SCF GRAD AUX QMMM PRECISE", workdir=str(tmp_path / tag), link_plan=plan,
            lj_sigma_nm=_PROPANOL_SIG, lj_epsilon_kj=_PROPANOL_EPS, include_lj=True, timeout_s=60)

    ref = run(_PROPANOL_XYZ, "ref")
    assert ref["link_atoms"]["n_link"] == 1 and ref["energy_lj_kj_mol"] != 0.0
    qp = {g: k for k, g in enumerate(_PROPANOL_QM)}
    mp = {g: k for k, g in enumerate(_PROPANOL_MM)}
    force = np.zeros_like(_PROPANOL_XYZ)
    for g, k in qp.items():
        force[g] = (ref["forces_qm_internal_kj_mol_nm"][k] + ref["forces_qmmm_elec_qm_kj_mol_nm"][k]
                    + ref["forces_lj_qm_kj_mol_nm"][k])
    for g, k in mp.items():
        force[g] = (ref["forces_qmmm_elec_mm_kj_mol_nm"][k] + ref["forces_link_mm_kj_mol_nm"][k]
                    + ref["forces_lj_mm_kj_mol_nm"][k])
    h = 0.002   # Angstrom
    analytic, numeric = [], []
    for atom in (7, 10, 4, 12):          # QM anchor, QM oxygen, MM boundary carbon, water oxygen
        for c in range(3):
            xp, xm = _PROPANOL_XYZ.copy(), _PROPANOL_XYZ.copy()
            xp[atom, c] += h
            xm[atom, c] -= h
            ep = run(xp, "p")["energy_qmmm_total_kj_mol"]
            em = run(xm, "m")["energy_qmmm_total_kj_mol"]
            numeric.append(-(ep - em) / (2 * h / 10.0))
            analytic.append(force[atom, c])
    analytic, numeric = np.asarray(analytic), np.asarray(numeric)
    assert np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric) < 0.01      # measured: 0.09 %
