import os

from redock_runtime import (
    build_redocking_smina_plan,
    run_redocking_single_run_pipeline,
    run_redocking_single_system_orchestrator,
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text), encoding="utf-8")
    return str(path)


def _run_redocking_with_ligand_template(tmp_path, repeats):
    rec_in = tmp_path / "rec_input.pdbqt"
    lig_in = tmp_path / "lig_in.sdf"
    template_mol2 = tmp_path / "charged_template.mol2"
    _write(rec_in, "RECEPTOR\n")
    _write(lig_in, "$$$$\n")
    _write(template_mol2, "@<TRIPOS>MOLECULE\nLIG\n")
    calls = {"prepare_ligand": [], "run_repeats": []}

    def _prepare_receptor(_obj, _job_dir, target_ph=7.4, flow_label="", auto_use_cached_hetatm=False):
        _ = (target_ph, flow_label, auto_use_cached_hetatm)
        return str(rec_in), {"notes": ["rec_ok"]}

    def _prepare_ligand(src, job_dir, target_ph=7.4, smiles_hint="", preserve_input_charges=False):
        prepared = os.path.join(str(job_dir), "lig_prepared.pdbqt")
        with open(prepared, "w", encoding="utf-8") as fh:
            fh.write("ATOM      1  C   LIG     1       0.000   0.000   0.000\n")
        calls["prepare_ligand"].append(
            {
                "src": str(src),
                "target_ph": float(target_ph),
                "smiles_hint": str(smiles_hint),
                "preserve_input_charges": bool(preserve_input_charges),
                "prepared": str(prepared),
            }
        )
        return str(prepared), {"notes": ["lig_ok"]}

    def _run_repeats(
        base_cmd,
        job_dir,
        n_repeats,
        seed0,
        run_command,
        parse_pose_scores,
        build_rows_for_repeat,
        timeout=7200,
    ):
        _ = (seed0, run_command, parse_pose_scores, timeout)
        calls["run_repeats"].append({"n_repeats": int(n_repeats), "base_cmd": list(base_cmd)})
        out_pose = os.path.join(str(job_dir), "docked_poses.pdbqt")
        out_log = os.path.join(str(job_dir), "docking.log")
        with open(out_pose, "w", encoding="utf-8") as fh:
            fh.write("MODEL 1\nREMARK minimizedAffinity -8.0\nENDMDL\n")
        with open(out_log, "w", encoding="utf-8") as fh:
            fh.write("ok\n")
        poses = [{"mode": 1, "affinity": -8.0, "rmsd_lb": 0.0, "rmsd_ub": 0.0}]
        rows = list(build_rows_for_repeat(1, poses, out_pose, out_log) or [])
        repeat_rows = [
            {
                "repeat": i,
                "seed": 100 + i,
                "ok": True,
                "best_affinity": -8.0 - (0.01 * i),
                "best_rmsd": 1.25,
                "n_poses": 1,
                "pose_file": str(out_pose),
                "log_file": str(out_log),
                "error": "",
            }
            for i in range(1, int(n_repeats) + 1)
        ]
        return {
            "ok": True,
            "repeat_rows": repeat_rows,
            "best_rows": rows,
            "best_repeat": dict(rows[0]),
            "best_repeat_id": 1,
            "best_pose_file": str(out_pose),
        }

    def _single_run_pipeline(**kwargs):
        return run_redocking_single_run_pipeline(
            **kwargs,
            run_repeats_fn=_run_repeats,
        )

    payload = run_redocking_single_system_orchestrator(
        system_label="WT",
        receptor_obj="wt_rec",
        reference_obj="ref_complex",
        ref_fields=("A", "101", "LIG"),
        ref_mol=object(),
        center=(1.0, 2.0, 3.0),
        size=(18.0, 18.0, 18.0),
        params={"ph": 7.4, "repeats": int(repeats), "seed0": 100, "num_modes": 1},
        source_receptor_obj="wt_rec",
        flex_enabled=False,
        flex_residues=[],
        auto_use_cached_hetatm=False,
        ligand_override_file="",
        smina_bin="smina",
        picked_ligand_file=str(lig_in),
        workspace_dir=str(tmp_path),
        safe_name_fn=lambda x: str(x).replace(" ", "_"),
        file_exists_fn=os.path.isfile,
        makedirs_fn=os.makedirs,
        prepare_receptor_fn=_prepare_receptor,
        strip_reference_residue_fn=lambda *_args: 0,
        prepare_flexible_receptor_fn=None,
        should_preserve_charges_fn=lambda _path: False,
        current_charge_mode_fn=lambda: "preserve",
        parameterized_ligand_mol2=str(template_mol2),
        samefile_or_equal_fn=lambda _a, _b: False,
        chemically_equivalent_fn=lambda _a, _b: True,
        pubchem_smiles_hint_fn=lambda: "",
        rcsb_smiles_by_het_fn=lambda _rn: "",
        prepare_ligand_fn=_prepare_ligand,
        has_rdkit=True,
        convert_pdbqt_to_sdf_fn=lambda _src, _dst: False,
        convert_pdbqt_to_pdb_fn=lambda _src, _dst: False,
        load_pose_molecules_fn=lambda _path: [],
        best_rmsd_fn=lambda _ref, _pose: 1.25,
        load_pose_object_fn=lambda _path, _obj: {"ok": True},
        run_command=lambda *_args, **_kwargs: (True, "", "", 0),
        parse_pose_scores_fn=lambda _pose, _log: [{"mode": 1, "affinity": -8.0}],
        repeat_statistics_fn=lambda vals, bootstrap_n=0, seed=0: {
            "n": int(len(list(vals or []))),
            "mean": 0.0,
            "ci_low": -1.0,
            "ci_high": 1.0,
            "confidence": "high",
        },
        timeout=30,
        compute_rmsd_values_for_repeat_fn=lambda **kwargs: [1.25 for _ in range(int(kwargs.get("poses_count", 0)))],
        single_run_pipeline_fn=_single_run_pipeline,
    )
    return payload, calls, str(template_mol2)


def test_redocking_one_sample_uses_parameterized_ligand_template(tmp_path):
    payload, calls, template_mol2 = _run_redocking_with_ligand_template(tmp_path, repeats=1)

    assert payload["ok"] is True
    assert int(payload["plan_payload"]["n_repeats"]) == 1
    assert int(payload["run_payload"]["repeat_count"]) == 1
    assert int(calls["run_repeats"][0]["n_repeats"]) == 1
    assert str(calls["prepare_ligand"][0]["src"]) == template_mol2
    assert bool(calls["prepare_ligand"][0]["preserve_input_charges"]) is True
    assert any("using charged MOL2 template" in str(msg) for msg in payload.get("debug_messages", []))


def test_redocking_ten_samples_use_parameterized_ligand_template(tmp_path):
    payload, calls, template_mol2 = _run_redocking_with_ligand_template(tmp_path, repeats=10)

    assert payload["ok"] is True
    assert int(payload["plan_payload"]["n_repeats"]) == 10
    assert int(payload["run_payload"]["repeat_count"]) == 10
    assert int(calls["run_repeats"][0]["n_repeats"]) == 10
    assert len(payload["run_payload"]["repeat_rows"]) == 10
    assert str(calls["prepare_ligand"][0]["src"]) == template_mol2
    assert bool(calls["prepare_ligand"][0]["preserve_input_charges"]) is True


def test_redocking_smina_plan_honors_zero_min_rmsd_filter():
    plan = build_redocking_smina_plan(
        smina_bin="smina",
        receptor_for_dock="rec.pdbqt",
        ligand_input="lig.pdbqt",
        center=(0.0, 0.0, 0.0),
        size=(18.0, 18.0, 18.0),
        params={"repeats": 1, "min_rmsd": 0.0},
    )
    cmd = list(plan["base_cmd"])
    assert cmd[cmd.index("--min_rmsd_filter") + 1] == "0.000"
