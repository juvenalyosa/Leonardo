import numpy as np
import pytest

from ligand_vib_refine_runtime import build_mode_comparison_rows
from ligand_vib_refine_runtime import build_qm_optimization_stages
from ligand_vib_refine_runtime import build_refined_frcmod_from_system
from ligand_vib_refine_runtime import build_scale_search_config
from ligand_vib_refine_runtime import compute_mode_fit_metrics
from ligand_vib_refine_runtime import _infer_element_from_mol2_name_type
from ligand_vib_refine_runtime import load_first_sdf_record_atoms
from ligand_vib_refine_runtime import parse_mopac_vibrational_aux_text
from ligand_vib_refine_runtime import qm_result_matches_stage
from ligand_vib_refine_runtime import _scale_regularization_weight
from ligand_vib_refine_runtime import load_mol2_full_records
from ligand_vib_refine_runtime import run_bounded_pso
from ligand_vib_refine_runtime import validate_qm_start_sdf_matches_template_mol2
from ligand_vib_refine_runtime import write_mol2_with_coords


def test_run_bounded_pso_moves_toward_simple_minimum():
    bounds = {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}

    def objective(point):
        x = float(point["x"])
        y = float(point["y"])
        return (x - 0.25) ** 2 + (y + 0.5) ** 2

    payload = run_bounded_pso(
        objective_fn=objective,
        bounds=bounds,
        initial={"x": 1.5, "y": 1.5},
        swarm_size=6,
        n_iter=8,
        seed=7,
    )
    best = dict(payload["best_scales"])
    assert abs(float(best["x"]) - 0.25) < 0.25
    assert abs(float(best["y"]) + 0.5) < 0.25
    assert float(payload["best_objective"]) < 0.1


def test_compute_mode_fit_metrics_rewards_better_ff_match():
    qm_vib = {
        "freq_real": np.asarray([0.1, 2.0, 3.0, 4.0, 5.0, 6.0, 400.0, 800.0, 1200.0], dtype=float),
        "freq_imag": np.zeros((9,), dtype=float),
        "ir": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 22.0, 14.0], dtype=float),
        "trv": np.asarray(["T", "T", "R", "R", "R", "T", "V", "V", "V"], dtype=object),
    }
    qm_coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    mm_good = compute_mode_fit_metrics(
        qm_vib_data=qm_vib,
        mm_freq_cm=np.asarray([-1.0, 0.2, 0.4, 0.7, 2.1, 4.2, 398.0, 805.0, 1188.0], dtype=float),
        mm_intensity_rel=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.2], dtype=float),
        mm_coords_ang=qm_coords + 0.02,
        qm_coords_ang=qm_coords,
    )
    mm_bad = compute_mode_fit_metrics(
        qm_vib_data=qm_vib,
        mm_freq_cm=np.asarray([-1.0, 0.2, 0.4, 0.7, 2.1, 4.2, 550.0, 1050.0, 1550.0], dtype=float),
        mm_intensity_rel=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.2], dtype=float),
        mm_coords_ang=qm_coords + 0.25,
        qm_coords_ang=qm_coords,
    )
    assert mm_good["ok"] is True
    assert mm_bad["ok"] is True
    assert float(mm_good["objective"]) < float(mm_bad["objective"])
    assert float(mm_good["freq_rmse_cm"]) < float(mm_bad["freq_rmse_cm"])


def test_compute_mode_fit_metrics_penalizes_geometry_drift_even_with_matching_modes():
    qm_vib = {
        "freq_real": np.asarray([0.1, 2.0, 3.0, 4.0, 5.0, 6.0, 400.0, 800.0, 1200.0], dtype=float),
        "freq_imag": np.zeros((9,), dtype=float),
        "ir": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 22.0, 14.0], dtype=float),
        "trv": np.asarray(["T", "T", "R", "R", "R", "T", "V", "V", "V"], dtype=object),
    }
    qm_coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    mm_near = compute_mode_fit_metrics(
        qm_vib_data=qm_vib,
        mm_freq_cm=np.asarray([-1.0, 0.2, 0.4, 0.7, 2.1, 4.2, 398.0, 805.0, 1188.0], dtype=float),
        mm_intensity_rel=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.2], dtype=float),
        mm_coords_ang=qm_coords + np.asarray([[0.00, 0.00, 0.00], [0.03, 0.00, 0.00], [0.00, -0.02, 0.00]], dtype=float),
        qm_coords_ang=qm_coords,
    )
    mm_far = compute_mode_fit_metrics(
        qm_vib_data=qm_vib,
        mm_freq_cm=np.asarray([-1.0, 0.2, 0.4, 0.7, 2.1, 4.2, 398.0, 805.0, 1188.0], dtype=float),
        mm_intensity_rel=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.2], dtype=float),
        mm_coords_ang=qm_coords + np.asarray([[0.00, 0.00, 0.00], [0.22, 0.00, 0.00], [0.00, -0.18, 0.12]], dtype=float),
        qm_coords_ang=qm_coords,
    )
    assert float(mm_near["geom_rmsd_ang"]) < float(mm_far["geom_rmsd_ang"])
    assert float(mm_far["geom_guard_penalty_ang"]) > 0.0
    assert float(mm_near["objective"]) < float(mm_far["objective"])


def test_write_mol2_with_coords_updates_positions(tmp_path):
    src = "benchmarks/abfe/ptb3_ben/BEN_from_complex_gaff2.mol2"
    payload = load_mol2_full_records(src)
    coords = np.asarray([[float(a["x"]), float(a["y"]), float(a["z"])] for a in payload["atoms"]], dtype=float)
    shifted = np.array(coords, copy=True)
    shifted[:, 0] += 1.234
    outp = tmp_path / "shifted.mol2"
    write_mol2_with_coords(src, shifted, str(outp))
    new_payload = load_mol2_full_records(str(outp))
    new_coords = np.asarray([[float(a["x"]), float(a["y"]), float(a["z"])] for a in new_payload["atoms"]], dtype=float)
    assert new_coords.shape == shifted.shape
    assert np.allclose(new_coords[:, 0], shifted[:, 0], atol=1.0e-4)
    assert np.allclose(new_coords[:, 1:], shifted[:, 1:], atol=1.0e-4)


def test_build_mode_comparison_rows_aligns_qm_and_ff_values():
    qm_like = {
        "qm_freq_cm": [100.0, 200.0],
        "qm_ir_rel": [0.4, 1.0],
    }
    refined = {
        "mm_freq_cm": [105.0, 195.0],
        "mm_ir_rel": [0.5, 0.9],
    }
    initial = {
        "mm_freq_cm": [120.0, 180.0],
        "mm_ir_rel": [0.2, 0.7],
    }
    rows = build_mode_comparison_rows(qm_like, refined, initial)
    assert len(rows) == 2
    assert rows[0]["mode_rank"] == 1
    assert abs(float(rows[1]["ff_refined_abs_error_cm"]) - 5.0) < 1.0e-8


def test_mol2_element_inference_handles_gaff_aromatic_types():
    assert _infer_element_from_mol2_name_type("C14", "ca") == "C"
    assert _infer_element_from_mol2_name_type("N1", "nv") == "N"
    assert _infer_element_from_mol2_name_type("O3", "os") == "O"
    assert _infer_element_from_mol2_name_type("Cl1", "cl") == "Cl"


def test_build_refined_frcmod_has_amber_section_layout():
    openmm = pytest.importorskip("openmm")
    from openmm import unit
    import os
    import tempfile

    system = openmm.System()
    for mass in (12.0, 12.0, 14.0, 1.0):
        system.addParticle(mass * unit.dalton)
    bond = openmm.HarmonicBondForce()
    bond.addBond(0, 1, 0.14 * unit.nanometer, 300000.0 * unit.kilojoule_per_mole / (unit.nanometer * unit.nanometer))
    system.addForce(bond)
    angle = openmm.HarmonicAngleForce()
    angle.addAngle(0, 1, 2, 2.1 * unit.radian, 350.0 * unit.kilojoule_per_mole / (unit.radian * unit.radian))
    system.addForce(angle)
    tors = openmm.PeriodicTorsionForce()
    tors.addTorsion(0, 1, 2, 3, 2, np.pi * unit.radian, 5.0 * unit.kilojoule_per_mole)
    system.addForce(tors)
    nb = openmm.NonbondedForce()
    for q, sig, eps in ((0.0, 0.34, 0.2), (0.0, 0.34, 0.2), (0.0, 0.325, 0.15), (0.0, 0.1069, 0.0157)):
        nb.addParticle(q * unit.elementary_charge, sig * unit.nanometer, eps * unit.kilojoule_per_mole)
    system.addForce(nb)
    mol2_text = """@<TRIPOS>MOLECULE
LIG
    4     3     1     0     0
SMALL
bcc

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 ca         1 LIG       0.000000
      2 C2          1.4000    0.0000    0.0000 ca         1 LIG       0.000000
      3 N1          2.5000    0.0000    0.0000 nv         1 LIG       0.000000
      4 H1          3.4000    0.0000    0.0000 hn         1 LIG       0.000000
@<TRIPOS>BOND
     1     1     2 1
     2     2     3 1
     3     3     4 1
@<TRIPOS>SUBSTRUCTURE
     1 LIG         1 TEMP              0 ****  ****    0 ROOT
"""
    with tempfile.TemporaryDirectory() as td:
        mol2_path = os.path.join(td, "toy.mol2")
        with open(mol2_path, "w", encoding="utf-8") as fh:
            fh.write(mol2_text)
        frcmod = build_refined_frcmod_from_system(system, mol2_path)
    assert "MASS\nca" in frcmod
    assert "\nBOND\nca-ca" in frcmod
    assert "\nANGLE\nca-ca-nv" in frcmod
    assert "\nDIHE\n" in frcmod
    assert "\nNONBON\nca" in frcmod


def test_build_scale_search_config_adds_typed_parameters():
    payload = {
        "atom_types": ["ca", "ca", "nv", "hn"],
        "bond_types_by_index": [("ca", "ca"), ("ca", "nv"), ("hn", "nv")],
        "angle_types_by_index": [("ca", "ca", "nv"), ("ca", "nv", "hn")],
        "torsion_types_by_index": [("ca", "ca", "nv", "hn")],
        "term_type_counts": {
            "atom": {"ca": 2, "nv": 1, "hn": 1},
            "bond": {("ca", "ca"): 1, ("ca", "nv"): 1, ("hn", "nv"): 1},
            "angle": {("ca", "ca", "nv"): 1, ("ca", "nv", "hn"): 1},
            "torsion": {("ca", "ca", "nv", "hn"): 1},
        },
    }
    config = build_scale_search_config(payload, fit_mode="typed_hybrid", max_typed_bond_types=2, max_typed_angle_types=1, max_typed_torsion_types=1, max_typed_lj_types=2)
    assert config["fit_mode"] == "typed_hybrid"
    assert "bond_k_scale" in config["bounds"]
    assert any(name.startswith("typed_bond__") for name in config["bounds"])
    assert any(name.startswith("typed_angle__") for name in config["bounds"])
    assert any(name.startswith("typed_torsion__") for name in config["bounds"])
    assert any(name.startswith("typed_sigma__") for name in config["bounds"])
    assert any(name.startswith("typed_epsilon__") for name in config["bounds"])
    assert config["bounds"]["bond_k_scale"][0] > 0.70
    assert config["bounds"]["epsilon_scale"][1] < 1.40


def test_build_scale_search_config_prioritizes_hydrogen_bonds_and_widens_bounds():
    payload = {
        "atom_types": ["ca", "ca", "ha", "nv", "hn"],
        "bond_types_by_index": [("ca", "ca"), ("ca", "ha"), ("hn", "nv"), ("ca", "nv")],
        "angle_types_by_index": [],
        "torsion_types_by_index": [],
        "term_type_counts": {
            "atom": {"ca": 2, "ha": 1, "nv": 1, "hn": 1},
            "bond": {("ca", "ca"): 5, ("ca", "nv"): 4, ("ca", "ha"): 3, ("hn", "nv"): 2},
            "angle": {},
            "torsion": {},
        },
    }
    config = build_scale_search_config(payload, fit_mode="typed_hybrid", max_typed_bond_types=2, max_typed_angle_types=0, max_typed_torsion_types=0, max_typed_lj_types=0)
    bond_names = [x["name"] for x in config["selected_typed_parameters"]["bond"]]
    assert "typed_bond__ca__ha" in bond_names
    assert "typed_bond__hn__nv" in bond_names
    assert config["bounds"]["typed_bond__ca__ha"][1] > config["bounds"]["bond_k_scale"][1]
    assert config["bounds"]["typed_bond__hn__nv"][1] > config["bounds"]["bond_k_scale"][1]


def test_hydrogen_bond_regularization_is_looser_than_generic_bond():
    assert _scale_regularization_weight("typed_bond__ca__ha") < _scale_regularization_weight("typed_bond__ca__ca")


def test_qm_result_matches_stage_requires_frequency_only_when_requested():
    cached_opt = {
        "ok": True,
        "method": "hf",
        "basis": "sto-3g",
        "optimized_xyz": "optimized.xyz",
    }
    cached_freq = dict(cached_opt)
    cached_freq["vib_modes_npz"] = "vib_modes.npz"
    assert qm_result_matches_stage(cached_opt, method="hf", basis="sto-3g", require_frequency=False) is True
    assert qm_result_matches_stage(cached_opt, method="hf", basis="sto-3g", require_frequency=True) is False
    assert qm_result_matches_stage(cached_freq, method="hf", basis="sto-3g", require_frequency=True) is True


def test_build_qm_optimization_stages_orders_and_deduplicates():
    stages = build_qm_optimization_stages(
        target_method="wb97x-d",
        target_basis="def2-SVP",
        low_method="hf",
        low_basis="sto-3g",
        medium_method="b3lyp",
        medium_basis="6-31g(d)",
        staging_mode="auto",
    )
    assert [s["label"] for s in stages] == ["low", "medium", "high"]
    assert stages[0]["method"] == "hf"
    assert stages[1]["method"] == "b3lyp"
    assert stages[2]["method"] == "wb97x-d"
    stages2 = build_qm_optimization_stages(
        target_method="b3lyp",
        target_basis="6-31g(d)",
        low_method="hf",
        low_basis="sto-3g",
        medium_method="b3lyp",
        medium_basis="6-31g(d)",
        staging_mode="auto",
    )
    assert [s["label"] for s in stages2] == ["low", "medium"]
    stages3 = build_qm_optimization_stages(
        target_method="wb97x-d",
        target_basis="def2-SVP",
        staging_mode="off",
    )
    assert [s["label"] for s in stages3] == ["high"]


def test_load_first_sdf_record_atoms_reads_v2000_geometry(tmp_path):
    sdf = tmp_path / "lig.sdf"
    sdf.write_text(
        "\n".join(
            [
                "",
                "LIG",
                "  test",
                "",
                "  3  2  0  0  0  0            999 V2000",
                "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "    1.2000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0",
                "    2.1000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "  1  2  1  0  0  0  0",
                "  2  3  1  0  0  0  0",
                "M  END",
                "$$$$",
            ]
        ),
        encoding="utf-8",
    )
    payload = load_first_sdf_record_atoms(str(sdf))
    atoms = list(payload["atoms"])
    assert payload["name"] == "LIG"
    assert len(atoms) == 3
    assert atoms[0]["element"] == "C"
    assert atoms[1]["element"] == "N"
    assert atoms[2]["element"] == "H"


def test_validate_qm_start_sdf_matches_template_mol2_rejects_sequence_mismatch(tmp_path):
    mol2 = tmp_path / "lig.mol2"
    mol2.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                "LIG",
                "    3     2     1     0     0",
                "SMALL",
                "bcc",
                "",
                "@<TRIPOS>ATOM",
                "      1 C1          0.0000    0.0000    0.0000 c3         1 LIG       0.000000",
                "      2 N1          1.2000    0.0000    0.0000 n3         1 LIG       0.000000",
                "      3 H1          2.1000    0.0000    0.0000 h1         1 LIG       0.000000",
                "@<TRIPOS>BOND",
                "     1     1     2 1",
                "     2     2     3 1",
                "@<TRIPOS>SUBSTRUCTURE",
                "     1 LIG         1 TEMP              0 ****  ****    0 ROOT",
            ]
        ),
        encoding="utf-8",
    )
    good_sdf = tmp_path / "good.sdf"
    good_sdf.write_text(
        "\n".join(
            [
                "LIG",
                "  test",
                "",
                "  3  2  0  0  0  0            999 V2000",
                "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "    1.2000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0",
                "    2.1000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "  1  2  1  0  0  0  0",
                "  2  3  1  0  0  0  0",
                "M  END",
                "$$$$",
            ]
        ),
        encoding="utf-8",
    )
    bad_sdf = tmp_path / "bad.sdf"
    bad_sdf.write_text(
        "\n".join(
            [
                "LIG",
                "  test",
                "",
                "  3  2  0  0  0  0            999 V2000",
                "    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0",
                "    1.2000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "    2.1000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "  1  2  1  0  0  0  0",
                "  2  3  1  0  0  0  0",
                "M  END",
                "$$$$",
            ]
        ),
        encoding="utf-8",
    )
    ok = validate_qm_start_sdf_matches_template_mol2(str(good_sdf), str(mol2))
    assert ok["ok"] is True
    with pytest.raises(ValueError, match="atom ordering does not match"):
        validate_qm_start_sdf_matches_template_mol2(str(bad_sdf), str(mol2))


def test_parse_mopac_vibrational_aux_text_extracts_vibrational_subset():
    aux = """
ATOM_EL[03]=
 O H H
ATOM_X_OPT:ANGSTROMS[09]=
 0.0000 0.0000 0.0000
 0.9573 0.0000 0.0000
 -0.2399 0.9268 0.0000
HEAT_OF_FORMATION:KCAL/MOL=-0.577912D+02
ZERO_POINT_ENERGY:KCAL/MOL= 0.101176D+02
VIB._FREQ:CM(-1)[09]=
 1406.50 2807.52 2863.37 -4.69 -3.66 0.03 167.34 -58.25 99.82
VIB._T_DIP:ELECTRONS[09]=
 0.522 1.244 0.391 0.001 0.002 0.000 1.448 0.000 0.822
NORMAL_MODES[081]=
""" + " ".join(["0.0"] * 81)
    parsed = parse_mopac_vibrational_aux_text(aux)
    assert parsed["ok"] is True
    assert parsed["atom_symbols"] == ["O", "H", "H"]
    assert parsed["optimized_coords_ang"].shape == (3, 3)
    assert np.allclose(parsed["freq_cm_real"], np.asarray([1406.50, 2807.52, 2863.37]))
    assert parsed["freq_cm_imag"].shape == (3,)
    # intensity = |transition dipole|^2 (the raw T_DIP values were used before)
    assert np.allclose(parsed["ir_intensity_rel"], np.asarray([0.522, 1.244, 0.391]) ** 2)


def test_psi4_stationary_runs_an_open_shell_molecule_with_an_unrestricted_reference(tmp_path):
    # OH radical (doublet), HF/STO-3G: a few seconds. Without UHF Psi4 stops: "RHF reference is only for singlets".
    psi4 = pytest.importorskip("psi4")
    import ligand_vib_refine_runtime as L

    psi4.core.clean_options()
    atoms =[{"element": "O", "x": 0.0, "y": 0.0, "z": 0.0}, {"element": "H", "x": 0.97, "y": 0.0, "z": 0.0}]
    res = L._run_psi4_stationary_from_atoms(atoms, str(tmp_path), method="hf", basis="sto-3g", multiplicity=2,
                                            run_frequency=False, threads=1, memory_gb=1)
    assert res["ok"] is True, res.get("error")
    assert res["multiplicity"] == 2 and res["energy_opt_hartree"] < -74.0


def test_psi4_stationary_starts_from_clean_options_and_accepts_a_relative_out_dir(tmp_path, monkeypatch):
    # a staged run executes several Psi4 jobs in one process: options of a previous stage (here a Cartesian
    # retry) must not carry over, and a relative out_dir must not break the final result write
    psi4 = pytest.importorskip("psi4")
    import ligand_vib_refine_runtime as L

    psi4.set_options({"opt_coordinates": "cartesian"})
    monkeypatch.chdir(tmp_path)
    atoms = [{"element": "H", "x": 0.0, "y": 0.0, "z": 0.0}, {"element": "H", "x": 0.80, "y": 0.0, "z": 0.0}]
    res = L._run_psi4_stationary_from_atoms(atoms, "h2_job", method="hf", basis="sto-3g", run_frequency=False,
                                            threads=1, memory_gb=1)
    assert res["ok"] is True, res.get("error")
    assert (tmp_path / "h2_job" / "psi4_result.json").is_file()
    assert str(psi4.core.get_option("OPTKING", "OPT_COORDINATES")).upper() != "CARTESIAN"


def test_psi4_frequency_from_xyz_runs_an_open_shell_molecule(tmp_path):
    psi4 = pytest.importorskip("psi4")
    import ligand_vib_refine_runtime as L

    psi4.core.clean_options()          # no reference left over from another test in this process
    xyz = tmp_path / "oh.xyz"
    xyz.write_text("2\nOH\nO 0.0 0.0 0.0\nH 0.97 0.0 0.0\n")
    freq = L.run_psi4_frequency_from_xyz(str(xyz), str(tmp_path / "freq"), method="hf", basis="sto-3g",
                                         multiplicity=2, threads=1, memory_gb=1)
    assert freq["ok"] is True, freq.get("error")


def test_mopac_vibrational_parse_keeps_imaginary_mode_of_a_saddle_point(tmp_path):
    # Real MOPAC 23 "PM7 FORCE AUX" output at the HCN <-> HNC transition state: one imaginary mode
    # (-1390.73) among the 3N-6 vibrations, followed by the six translations/rotations.
    import ligand_vib_refine_runtime as L

    aux = """
 ATOM_EL[03]=
   C  N  H
 ATOM_X_OPT:ANGSTROMS[09]=
    0.0005    0.0009    0.0000
    1.2039   -0.0416   -0.0000
    0.4592    1.2518    0.0000
 HEAT_OF_FORMATION:KCAL/MOL=+0.115929D+03
 VIB._FREQ:CM(-1)[09]=
 -1390.73  2015.04  2322.00     0.02     0.03     0.03    56.42    29.84    20.20
 VIB._T_DIP:ELECTRONS[09]=
 0.300 0.200 0.100 0.000 0.000 0.000 0.010 0.010 0.010
 NORMAL_MODES[081]=
""" + " ".join(["0.1"] * 81)
    parsed = parse_mopac_vibrational_aux_text(aux)
    assert parsed["ok"] is True
    assert np.allclose(parsed["freq_cm_real"], [0.0, 2015.04, 2322.00])
    assert np.allclose(parsed["freq_cm_imag"], [1390.73, 0.0, 0.0])
    assert np.allclose(parsed["ir_intensity_rel"], [0.090, 0.040, 0.010])
    assert parsed["mode_cart_ang"].shape == (3, 3, 3)

    opt = {"parsed": parsed, "method": "PM7"}
    force = {"parsed": parsed, "method": "PM7"}
    result = L._build_mopac_result_from_payloads(opt, force_payload=force, out_dir=str(tmp_path))
    vib = L.psi4_load_vib_data(result, str(tmp_path))
    summary = L.make_summary_payload(result, vib, {}, {}, {}, {})
    assert summary["stationary_point_ok"] is False
    assert summary["qm_imaginary_count"] == 1
    assert summary["qm_vibrational_mode_count"] == 2      # the imaginary mode is not fitted as a vibration
