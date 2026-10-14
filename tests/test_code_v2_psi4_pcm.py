import os
import types

import code_v2
import numpy as np
import psi4_runtime
from psi4_runtime import psi4_element_token, psi4_pcm_supported_solvents, psi4_qst_build_guess_atoms


class _DummyDialog:
    pass


def _build_dummy_dialog():
    for name in (
        "_psi4_pcm_force_c1",
        "_psi4_pcm_solvent_candidates",
        "_psi4_molecule_block",
        "_psi4_qst_script_payload",
        "_psi4_make_script",
    ):
        setattr(_DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return _DummyDialog()


def _pcm_job():
    return {
        "operation": "frequencies",
        "method": "hf",
        "basis": "sto-3g",
        "molecule_block": "\n".join(
            [
                "0 1",
                "O 0.000000 0.000000 0.000000",
                "H 0.757160 0.586260 0.000000",
                "H -0.757160 0.586260 0.000000",
                "units angstrom",
                "no_reorient",
                "no_com",
            ]
        ),
        "memory_gb": 1,
        "threads": 1,
        "implicit": True,
        "solvent": "Water",
        "scf_type": "pk",
        "pcm_solver": "IEFPCM",
    }


def test_psi4_pcm_force_c1_inserts_symmetry_before_orientation_flags():
    dlg = _build_dummy_dialog()

    out = dlg._psi4_pcm_force_c1("0 1\nC 0 0 0\nunits angstrom\nno_reorient\nno_com")

    assert "symmetry c1" in out
    assert out.index("symmetry c1") < out.index("no_reorient")


def test_psi4_pcm_force_c1_replaces_existing_symmetry():
    dlg = _build_dummy_dialog()

    out = dlg._psi4_pcm_force_c1("0 1\nC 0 0 0\nsymmetry c2v\nunits angstrom\nno_reorient")

    assert "symmetry c2v" not in out.lower()
    assert out.lower().count("symmetry c1") == 1


def test_psi4_make_script_uses_pcm_input_and_forces_c1_for_solvent_jobs():
    dlg = _build_dummy_dialog()

    script = dlg._psi4_make_script(_pcm_job())

    assert 'pcm_block = r"""Units = Angstrom' in script
    assert "\\nMedium {" not in script
    assert "pcm__input" in script
    assert "psi4.set_options({'pcm': True, 'pcm_scf_type': 'total', 'pcm__input': pcm_block})" in script
    assert "psi4.pcm_helper" not in script
    assert "symmetry c1" in script
    assert "result['pcm_input_mode'] = 'pcm__input'" in script
    assert "result['pcm_forced_symmetry'] = 'c1'" in script
    assert "allow_cubeprop = False" in script
    assert "result['cubeprop_skipped'] = 'pcm_enabled'" in script


def test_psi4_make_script_without_pcm_keeps_plain_options():
    dlg = _build_dummy_dialog()
    job = dict(_pcm_job())
    job["implicit"] = False

    script = dlg._psi4_make_script(job)

    assert "pcm__input" not in script
    assert "result['pcm_input_mode']" not in script
    assert "psi4.pcm_helper" not in script


def test_psi4_make_script_supports_single_point_and_normalizes_embedded_basis():
    dlg = _build_dummy_dialog()
    job = dict(_pcm_job())
    job["operation"] = "single point"
    job["method"] = "hf / sto-3g"
    job["basis"] = "def2-SVP"
    job["implicit"] = False

    script = dlg._psi4_make_script(job)

    assert "e, wfn = psi4.energy(method_name, return_wfn=True)" in script
    assert "psi4.optimize(method_name" not in script
    assert "method_name = 'hf'" in script
    assert "result['method'] = 'hf'" in script
    assert "result['basis'] = 'sto-3g'" in script
    assert "'basis': 'sto-3g'" in script


def test_psi4_make_script_sets_open_shell_reference_from_multiplicity():
    dlg = _build_dummy_dialog()
    job = dict(_pcm_job())
    job["operation"] = "energy"
    job["implicit"] = False
    job["molecule_block"] = "\n".join(
        [
            "0 2",
            "H 0.0 0.0 -0.9",
            "H 0.0 0.0 0.0",
            "H 0.0 0.0 0.9",
            "units angstrom",
            "symmetry c1",
            "no_reorient",
            "no_com",
        ]
    )

    script = dlg._psi4_make_script(job)

    assert "'reference': 'uhf'" in script


def test_psi4_qst2_guess_rejects_collapsed_aligned_midpoint():
    dlg = _build_dummy_dialog()
    for name in (
        "_psi4_reorder_atoms_to_template",
        "_psi4_build_qst_guess_atoms",
    ):
        setattr(dlg, name, types.MethodType(getattr(code_v2.ProtLigInteractDialog, name), dlg))
    dlg._psi4_element_token = lambda element: psi4_element_token(element)
    dlg._psi4_align_coords_kabsch = lambda _ref, _mob: np.asarray(
        [
            [-0.7, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [-0.85, -0.95, 0.0],
            [0.55, 0.95, 0.0],
        ],
        dtype=float,
    )

    react = [
        {"index": 1, "name": "O1", "element": "O", "symbol": "O", "x": -0.7, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "O2", "element": "O", "symbol": "O", "x": 0.7, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "H1", "element": "H", "symbol": "H", "x": -0.55, "y": 0.95, "z": 0.0},
        {"index": 4, "name": "H2", "element": "H", "symbol": "H", "x": 0.55, "y": -0.95, "z": 0.0},
    ]
    prod = [
        {"index": 1, "name": "O1", "element": "O", "symbol": "O", "x": -0.7, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "O2", "element": "O", "symbol": "O", "x": 0.7, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "H1", "element": "H", "symbol": "H", "x": -0.55, "y": 0.95, "z": 0.0},
        {"index": 4, "name": "H2", "element": "H", "symbol": "H", "x": 0.55, "y": -0.95, "z": 0.0},
    ]

    old_align = psi4_runtime.psi4_qst_align_coords_kabsch
    psi4_runtime.psi4_qst_align_coords_kabsch = dlg._psi4_align_coords_kabsch
    try:
        guess, note = dlg._psi4_build_qst_guess_atoms(react, prod)
    finally:
        psi4_runtime.psi4_qst_align_coords_kabsch = old_align

    assert len(guess) == 4
    assert "guess=direct_midpoint" in note
    assert "gradients=psi4_optking_ts" in note


def test_psi4_qst3_builds_quadratic_path_through_ts_guess():
    react = [
        {"index": 1, "name": "N", "element": "N", "x": 0.0, "y": 0.0, "z": 0.10},
        {"index": 2, "name": "H1", "element": "H", "x": 0.0, "y": 0.94, "z": -0.27},
        {"index": 3, "name": "H2", "element": "H", "x": 0.8138, "y": -0.47, "z": -0.27},
        {"index": 4, "name": "H3", "element": "H", "x": -0.8138, "y": -0.47, "z": -0.27},
    ]
    prod = [
        {"index": 1, "name": "N", "element": "N", "x": 0.0, "y": 0.0, "z": -0.10},
        {"index": 2, "name": "H1", "element": "H", "x": 0.0, "y": 0.94, "z": 0.27},
        {"index": 3, "name": "H2", "element": "H", "x": 0.8138, "y": -0.47, "z": 0.27},
        {"index": 4, "name": "H3", "element": "H", "x": -0.8138, "y": -0.47, "z": 0.27},
    ]
    ts_guess = [
        {"index": 1, "name": "N", "element": "N", "x": 0.0, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "H1", "element": "H", "x": 0.0, "y": 0.94, "z": 0.0},
        {"index": 3, "name": "H2", "element": "H", "x": 0.8138, "y": -0.47, "z": 0.0},
        {"index": 4, "name": "H3", "element": "H", "x": -0.8138, "y": -0.47, "z": 0.0},
    ]

    qst = psi4_qst_build_guess_atoms(react, prod, ts_guess_atoms=ts_guess, ts_weight=1.0, n_images=9)

    assert qst["payload"]["mode"] == "qst3"
    assert "path=quadratic_synchronous" in qst["note"]
    center = min(qst["payload"]["images"], key=lambda im: abs(im["lambda"] - 0.5))
    assert center["lambda"] == 0.5
    center_xyz = np.asarray([[a["x"], a["y"], a["z"]] for a in center["atoms"]], dtype=float)
    guess_xyz = np.asarray([[a["x"], a["y"], a["z"]] for a in ts_guess], dtype=float)
    assert np.allclose(center_xyz - center_xyz.mean(axis=0), guess_xyz - guess_xyz.mean(axis=0), atol=1.0e-7)


def test_psi4_make_script_qst_ts_prescans_before_ts_opt_and_frequency():
    dlg = _build_dummy_dialog()
    dlg._psi4_element_token = lambda element: psi4_element_token(element)
    react = [
        {"index": 1, "name": "H1", "element": "H", "x": -0.4, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "H2", "element": "H", "x": 0.4, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "H3", "element": "H", "x": 1.4, "y": 0.0, "z": 0.0},
    ]
    prod = [
        {"index": 1, "name": "H1", "element": "H", "x": -1.4, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "H2", "element": "H", "x": -0.4, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "H3", "element": "H", "x": 0.4, "y": 0.0, "z": 0.0},
    ]
    qst = psi4_qst_build_guess_atoms(react, prod, n_images=5)
    job = {
        "operation": "ts",
        "method": "hf",
        "basis": "sto-3g",
        "molecule_block": dlg._psi4_molecule_block(qst["guess_atoms"], charge=0, multiplicity=2),
        "memory_gb": 1,
        "threads": 1,
        "implicit": False,
        "scf_type": "pk",
        "charge": 0,
        "multiplicity": 2,
        "qst": dlg._psi4_qst_script_payload(qst["payload"], charge=0, multiplicity=2),
    }

    script = dlg._psi4_make_script(job)

    assert "qst_payload =" in script
    assert "psi4.energy(method_name, molecule=qmol)" in script
    assert "result['qst_scan_selected']" in script
    assert "result['opt_type'] = 'TS'" in script
    assert "result['gradient_engine'] = 'Psi4/OptKing optimize driver'" in script
    assert "psi4.optimize(method_name, return_wfn=True, return_history=True)" in script
    assert "psi4.frequency(method_name, return_wfn=True)" in script


def test_psi4_make_script_irc_validates_ts_and_writes_profile():
    dlg = _build_dummy_dialog()
    job = dict(_pcm_job())
    job.update(
        {
            "operation": "irc",
            "implicit": False,
            "irc_direction": "BOTH",
            "irc_points": 8,
            "irc_step": 0.10,
            "irc_require_valid_ts": True,
        }
    )

    script = dlg._psi4_make_script(job)

    assert "'opt_type': 'IRC'" in script
    assert "efreq, wfn_freq = psi4.frequency(method_name, return_wfn=True)" in script
    assert "_validate_irc_ts_frequencies" in script
    assert "result['irc_ts_validation']" in script
    assert "IRC requires exactly one imaginary TS frequency" in script
    assert "_irc_dirs = ['BACKWARD', 'FORWARD']" in script
    assert "_combine_irc_histories(_irc_histories.get('BACKWARD'), _irc_histories.get('FORWARD'))" in script
    assert "psi4.optimize(method_name, return_wfn=True, return_history=True)" in script
    assert "_build_irc_profile(hA, hE, symbols)" in script
    assert "result['irc_profile_csv'] = 'irc_profile.csv'" in script
    assert "result['irc_coordinate_amu05_bohr']" in script


def test_psi4_cluster_worker_runs_qst2_qst3_then_irc(tmp_path):
    dlg = _build_dummy_dialog()
    for name in (
        "_psi4_reorder_atoms_to_template",
        "_psi4_align_coords_kabsch",
        "_psi4_build_qst_guess_atoms",
        "_psi4_molecule_block",
        "_psi4_collect_optimized_atoms_from_run_result",
        "_psi4_execute_cluster_reaction_worker",
    ):
        setattr(dlg, name, types.MethodType(getattr(code_v2.ProtLigInteractDialog, name), dlg))
    dlg._psi4_element_token = lambda element: psi4_element_token(element)
    stage_calls = []

    def fake_execute(job, job_dir, psi4_bin, timeout_s, use_pcm, solvent, progress_cb=None):
        assert os.path.isdir(job_dir)
        stage_calls.append((str(job_dir).split("/")[-1], job["operation"]))
        if callable(progress_cb):
            progress_cb({"progress": 100.0, "message": "fake complete"})
        return {
            "logs": [f"fake {job['operation']}"],
            "attempt_summaries": [{"operation": job["operation"]}],
            "run_result": {
                "success": True,
                "job_dir": str(job_dir),
                "result": {},
            },
        }

    dlg._psi4_execute_with_fallback_worker = fake_execute
    react = [
        {"index": 1, "name": "O1", "element": "O", "x": -0.7, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "O2", "element": "O", "x": 0.7, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "H1", "element": "H", "x": -0.55, "y": 0.95, "z": 0.0},
        {"index": 4, "name": "H2", "element": "H", "x": 0.55, "y": -0.95, "z": 0.0},
    ]
    prod = [
        {"index": 1, "name": "O1", "element": "O", "x": -0.7, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "O2", "element": "O", "x": 0.7, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "H1", "element": "H", "x": 0.55, "y": 0.95, "z": 0.0},
        {"index": 4, "name": "H2", "element": "H", "x": -0.55, "y": -0.95, "z": 0.0},
    ]

    result = dlg._psi4_execute_cluster_reaction_worker({
        "method": "hf",
        "basis": "sto-3g",
        "charge": 0,
        "mult": 1,
        "memory_gb": 1,
        "threads": 1,
        "geom_maxiter": 20,
        "timeout_s": 60,
        "use_pcm": False,
        "run_qst3": True,
        "run_irc": True,
        "do_preopt": True,
        "do_classical_min": False,
        "psi4_bin": "psi4",
        "job_dir": str(tmp_path),
        "react_atoms": react,
        "prod_atoms": prod,
    })

    assert result["ok"] is True
    assert list(result["stage_payloads"]) == ["react_preopt", "prod_preopt", "qst2_ts", "qst3_ts", "irc"]
    assert [op for _stage, op in stage_calls] == ["optimize", "optimize", "ts", "ts", "irc"]
    assert result["final_operation"] == "irc"


def test_psi4_pcm_supported_solvents_exposes_full_builtin_table_without_explicit():
    sols = psi4_pcm_supported_solvents()

    assert "Water" in sols
    assert "Propylene Carbonate" in sols
    assert "THF" in sols
    assert "Benzene" in sols
    assert "Cyclohexane" in sols
    assert "Explicit" not in sols


def test_psi4_pcm_solvent_candidates_use_aliases_without_hidden_water_fallback():
    dlg = _build_dummy_dialog()

    assert dlg._psi4_pcm_solvent_candidates("THF")[:2] == ["THF", "Tetrahydrofurane"]
    assert "Water" not in dlg._psi4_pcm_solvent_candidates("THF")
    assert dlg._psi4_pcm_solvent_candidates("Benzene") == ["Benzene", "C6H6"]
    assert dlg._psi4_pcm_solvent_candidates("not-a-real-solvent") == ["not-a-real-solvent"]


def test_psi4_make_script_preserves_nonwater_supported_solvent_name():
    dlg = _build_dummy_dialog()
    job = dict(_pcm_job())
    job["solvent"] = "THF"

    script = dlg._psi4_make_script(job)

    assert "Solvent = THF" in script
    assert "Solvent = Water" not in script
