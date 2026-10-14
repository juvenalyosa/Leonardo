import os

import numpy as np
from leadopt_runtime import leadopt_finalize_refine_estimate, mpo_score, run_leadopt_entries
from redock_runtime import pick_best_pose_row
from smina_runtime import run_smina_repeats
from vs_runtime import run_virtual_screening_entries


def test_smina_runtime_ignores_malformed_pose_affinities(tmp_path):
    def _run_command(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        return True, "", "", 0

    def _parse_pose_scores(_out_pose, _out_log):
        return [
            {"mode": 1, "affinity": "bad"},
            {"mode": 2, "affinity": -8.4},
        ]

    payload = run_smina_repeats(
        base_cmd=["smina", "-r", "r.pdbqt", "-l", "l.pdbqt"],
        job_dir=str(tmp_path),
        n_repeats=1,
        seed0=0,
        run_command=_run_command,
        parse_pose_scores=_parse_pose_scores,
        timeout=10,
    )

    assert payload["best_repeat"] is not None
    assert payload["repeat_rows"][0]["ok"] is True
    assert float(payload["repeat_rows"][0]["best_affinity"]) == -8.4


def test_vs_runtime_accepts_item_aware_fallback_and_bad_pose_rows(tmp_path):
    lig_dir = tmp_path / "ligands"
    pose_dir = tmp_path / "poses"
    lig_dir.mkdir()
    pose_dir.mkdir()

    def _safe_name(name):
        return str(name).replace(" ", "_")

    def _write_sdf(_mol, lig_sdf, _name):
        with open(lig_sdf, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True, ""

    def _prepare_with_meeko(_mol, _out_pdbqt):
        return False, "meeko_failed"

    def _fallback_prepare_ligand(lig_sdf, fallback_job, item, _target_ph):
        assert item["identifier"] == "L1"
        _ = lig_sdf
        os.makedirs(str(fallback_job), exist_ok=True)
        out_pdbqt = os.path.join(str(fallback_job), "ligand_prepared.pdbqt")
        with open(out_pdbqt, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return out_pdbqt, {"notes": "fallback_ok"}

    def _run_command(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        return True, "", "", 0

    def _parse_pose_scores(_out_pose, _out_log):
        return [
            {"mode": 1, "affinity": "bad"},
            {"mode": 2, "affinity": -7.9},
        ]

    def _convert_pose_for_view(out_pose, _pose_dir, _stem):
        return str(out_pose)

    rows, cancelled = run_virtual_screening_entries(
        entries=[{"index": 1, "name": "Lig A", "identifier": "L1", "mol": object()}],
        smina_bin="smina",
        receptor_input="rec.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(10.0, 10.0, 10.0),
        exhaustiveness=8,
        timeout_lig=30,
        lig_dir=str(lig_dir),
        pose_dir=str(pose_dir),
        target_ph=7.4,
        safe_name_fn=_safe_name,
        write_sdf=_write_sdf,
        prepare_with_meeko=_prepare_with_meeko,
        fallback_prepare_ligand=_fallback_prepare_ligand,
        run_command=_run_command,
        parse_pose_scores=_parse_pose_scores,
        convert_pose_for_view=_convert_pose_for_view,
    )

    assert cancelled is False
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert float(rows[0]["affinity"]) == -7.9
    assert "fallback_ok" in str(rows[0]["notes"])


def test_leadopt_runtime_ignores_malformed_pose_affinities(tmp_path):
    work_dir = tmp_path / "work"
    lig_dir = work_dir / "ligands"
    pose_dir = work_dir / "poses"
    lig_dir.mkdir(parents=True)
    pose_dir.mkdir(parents=True)

    def _safe_name(name):
        return str(name).replace(" ", "_")

    def _write_sdf(_mol, lig_sdf, _name):
        with open(lig_sdf, "w", encoding="utf-8") as fh:
            fh.write("$$$$\n")
        return True, ""

    def _prepare_with_meeko(_mol, out_pdbqt):
        with open(out_pdbqt, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        return True, "ok"

    def _fallback_prepare_ligand(_lig_sdf, _fallback_job, _item, _target_ph):
        raise AssertionError("fallback should not be used")

    def _run_command(argv, cwd=None, timeout=0):
        _ = (cwd, timeout)
        out_pose = argv[argv.index("--out") + 1]
        out_log = argv[argv.index("--log") + 1]
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("ATOM 1 C\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        return True, "", "", 0

    def _parse_pose_scores(_out_pose, _out_log):
        return [
            {"mode": 1, "affinity": "bad"},
            {"mode": 2, "affinity": -8.6},
        ]

    def _convert_pose_for_view(out_pose, _pose_dir, _stem):
        return str(out_pose)

    rows, cancelled = run_leadopt_entries(
        entries=[
            {
                "name": "Lead A",
                "source": "pool",
                "smiles": "CCO",
                "admet": 0.8,
                "qed": 0.6,
                "sim_seed": 0.5,
                "mw": 46.0,
                "logp": 0.1,
                "hba": 1,
                "hbd": 1,
                "tpsa": 20.0,
                "rotb": 0,
                "mol": object(),
            }
        ],
        smina_bin="smina",
        receptor_input="rec.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(10.0, 10.0, 10.0),
        exhaustiveness=8,
        timeout_lig=30,
        work_dir=str(work_dir),
        lig_dir=str(lig_dir),
        pose_dir=str(pose_dir),
        target_ph=7.4,
        safe_name_fn=_safe_name,
        write_sdf=_write_sdf,
        prepare_with_meeko=_prepare_with_meeko,
        fallback_prepare_ligand=_fallback_prepare_ligand,
        run_command=_run_command,
        parse_pose_scores=_parse_pose_scores,
        convert_pose_for_view=_convert_pose_for_view,
        mpo_fn=mpo_score,
    )

    assert cancelled is False
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    assert float(rows[0]["affinity"]) == -8.6
    assert np.isfinite(float(rows[0]["mpo"]))


def test_leadopt_finalize_refine_estimate_preserves_zero_neff():
    payload = leadopt_finalize_refine_estimate(
        est={
            "ok": True,
            "method": "pymbar.exp",
            "dg_kj": -10.0,
            "dg_err_kj": 1.0,
            "ci_low_kj": -11.0,
            "ci_high_kj": -9.0,
            "n_eff": 0,
        },
        energies=[-1.0, -2.0, -3.0, -4.0],
        dmins=[2.0, 2.5, 3.0, 3.5],
        min_samples=4,
    )

    assert payload["ok"] is True
    assert int(payload["update"]["refine_n_eff"]) == 0


def test_redock_runtime_pick_best_pose_row_ignores_bad_affinity_strings():
    best = pick_best_pose_row(
        [
            {"mode": 1, "affinity": "bad", "rmsd_ref": float("nan")},
            {"mode": 2, "affinity": -7.1, "rmsd_ref": float("nan")},
        ]
    )

    assert best is not None
    assert float(best["affinity"]) == -7.1
