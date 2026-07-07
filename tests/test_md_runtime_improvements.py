import numpy as np

from abfe_runtime import build_abfe_openmm_context_setup_plan
from abfe_runtime import build_abfe_openmm_staged_prep_plan
from openmm_qmmm_runtime import run_openmm_qmmm_distance_scan_probe
from openmm_qmmm_runtime import run_openmm_qmmm_single_point_probe


def test_abfe_context_setup_plan_sanitizes_md_inputs_and_gpu_precision():
    plan = build_abfe_openmm_context_setup_plan(
        temperature_k=-10.0,
        timestep_ps=0.01,
        friction_inv_ps=0.0,
        picked_platform_name="CUDA",
        has_platform=True,
        platform_properties={},
    )

    assert plan["ok"] is True
    assert float(plan["temperature_k"]) == 300.0
    assert float(plan["timestep_ps"]) == 0.004
    assert float(plan["friction_inv_ps"]) == 5.0
    assert plan["integration_regime"] == "hmr_candidate"
    assert plan["requires_hydrogen_mass_repartitioning"] is True
    assert plan["platform_properties"]["Precision"] == "mixed"
    assert "defaulted_invalid_temperature" in plan["notes"]
    assert "clamped_timestep_to_4fs" in plan["notes"]
    assert "defaulted_invalid_friction" in plan["notes"]
    assert "added_mixed_precision_for_gpu_platform" in plan["notes"]
    assert int(plan["recommended_checkpoint_interval_steps"]) > int(plan["recommended_report_interval_steps"])


def test_abfe_context_setup_plan_ignores_empty_platform_name():
    plan = build_abfe_openmm_context_setup_plan(
        temperature_k=310.0,
        timestep_ps=0.002,
        friction_inv_ps=4.0,
        picked_platform_name="",
        has_platform=True,
        platform_properties={"Precision": "single"},
    )

    assert plan["has_platform"] is False
    assert len(plan["context_attempts"]) == 1
    assert plan["context_attempts"][0]["mode"] == "default"


def test_abfe_staged_prep_plan_reports_analysis_discard_recommendation():
    plan = build_abfe_openmm_staged_prep_plan(
        minimize_max_iters=1000,
        has_lig_hold=True,
        sim_ns=0.05,
        timestep_ps=0.01,
    )

    assert plan["ok"] is True
    assert float(plan["sim_ns_used"]) == 0.2
    assert float(plan["timestep_ps_used"]) == 0.004
    assert float(plan["recommended_analysis_discard_ps"]) == float(plan["equilibration_ps_total"])
    assert 0.0 < float(plan["recommended_analysis_discard_fraction"]) <= 0.5
    assert "clamped_short_duration" in plan["notes"]
    assert "clamped_timestep_to_4fs" in plan["notes"]


def test_openmm_qmmm_single_point_probe_returns_structured_backend_failure(tmp_path):
    xyz_full = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float)
    atomic_numbers = [6, 8]
    charges = np.array([0.0, 0.0], dtype=float)
    atom_rows = [
        {"index": 0, "name": "C1", "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 1, "name": "O1", "resn": "LIG", "chain": "L", "resi": "10"},
    ]

    def _fail_step(**_kwargs):
        raise RuntimeError("mopac boom")

    out = run_openmm_qmmm_single_point_probe(
        xyz_full_ang=xyz_full,
        atomic_numbers_full=atomic_numbers,
        mm_charges_e=charges,
        atom_rows=atom_rows,
        config={"qm_indices": [0], "mm_indices": [1]},
        workdir=str(tmp_path),
        qmmm_step_fn=_fail_step,
    )

    assert out["ok"] is False
    assert "mopac boom" in out["error"]
    assert out["qm_indices"] == [0]
    assert out["mm_indices"] == [1]


def test_openmm_qmmm_distance_scan_probe_skips_failed_points():
    xyz_full = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    atomic_numbers = [6, 6, 8]
    charges = np.array([0.1, 0.0, 0.0], dtype=float)
    atom_rows = [
        {"index": 0, "name": "CA", "resn": "ALA", "chain": "A", "resi": "1"},
        {"index": 1, "name": "C1", "resn": "LIG", "chain": "L", "resi": "10"},
        {"index": 2, "name": "O1", "resn": "LIG", "chain": "L", "resi": "10"},
    ]

    def _mock_step(
        xyz_full_ang,
        atomic_numbers_full,
        qm_indices,
        mm_indices,
        q_mm,
        **_kwargs,
    ):
        _ = (atomic_numbers_full, mm_indices, q_mm)
        xyz = np.asarray(xyz_full_ang, dtype=float)
        d = float(np.linalg.norm(xyz[qm_indices[1]] - xyz[qm_indices[0]]))
        if d < 1.3:
            return {"ok": False, "reason": "too_close"}
        return {
            "ok": True,
            "energy_qm_hartree": float((d - 1.6) ** 2 - 10.0),
            "energy_qmmm_elec_kj_mol": float(-1.0 * d),
            "forces_qm_internal_kj_mol_nm": np.zeros((len(qm_indices), 3), dtype=float),
            "forces_qmmm_elec_qm_kj_mol_nm": np.zeros((len(qm_indices), 3), dtype=float),
            "forces_qmmm_elec_mm_kj_mol_nm": np.zeros((len(mm_indices), 3), dtype=float),
        }

    out = run_openmm_qmmm_distance_scan_probe(
        xyz_full_ang=xyz_full,
        atomic_numbers_full=atomic_numbers,
        mm_charges_e=charges,
        atom_rows=atom_rows,
        scan_pair=(1, 2),
        scan_start_ang=1.0,
        scan_end_ang=2.0,
        n_points=5,
        config={},
        qmmm_step_fn=_mock_step,
    )

    assert out["ok"] is True
    assert int(out["n_successful"]) == 3
    assert int(out["n_failed"]) == 2
    first = dict(out["rows"][0])
    assert first["ok"] is False
    assert first["error"] == "too_close"
    best = dict(out["best_point"])
    assert best["ok"] is True
    assert abs(float(best["distance_ang"]) - 1.5) <= 0.26
