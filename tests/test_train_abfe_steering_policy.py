import json

import numpy as np
from train_abfe_steering_policy import (
    ABFE_STEERING_POLICY_FEATURE_NAMES,
    ABFE_STEERING_POLICY_OOD_FEATURE_NAMES,
    build_ood_profile,
    evaluate_group_cv,
    load_trace_rows,
    select_model_kind,
)


def test_load_trace_rows_derives_groups(tmp_path):
    trace_path = tmp_path / "case_a" / "policy_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "actionable": True,
                "policy_action_name": "keep",
                "reward_estimate": 0.1,
                "ligand_smiles": "NC(=N)c1ccccc1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_trace_rows([trace_path])
    assert len(rows) == 1
    row = dict(rows[0])
    assert str(row["target_group"]) == "case_a"
    assert str(row["scaffold_group"]) != ""
    assert str(row["ligand_scaffold"]) != ""


def test_grouped_policy_eval_and_ood_profile():
    rows = []
    for idx in range(12):
        rows.append(
            {
                "actionable": True,
                "policy_action_name": "keep" if idx % 2 == 0 else "reduce_pull",
                "reward_estimate": 0.5 + (0.1 * (idx % 3)),
                "trace_group": f"trace_{idx // 3}",
                "target_group": "target_a" if idx < 6 else "target_b",
                "scaffold_group": "scaf_a" if idx % 4 < 2 else "scaf_b",
                "phase": "pull",
                "size_scale": 0.3 + (0.02 * idx),
                "charge_density": 0.05,
                "formal_charge_e": 1.0 if idx % 2 == 0 else 0.0,
                "ring_count": 1 + (idx % 2),
                "rotatable_bonds": 2 + (idx % 3),
                "metal_contact_expected": False,
                "pocket_n_atoms": 32.0 + idx,
                "pocket_volume_bbox_a3": 420.0 + (15.0 * idx),
                "pocket_atom_density_per_a3": 0.04 + (0.002 * idx),
                "pocket_frac_hydrophobic": 0.45,
                "pocket_frac_polar": 0.25,
                "pocket_charge_density": -0.01,
                "far_unbound_min_dist_a": 3.5,
                "far_unbound_site_com_a": 10.0,
                "far_unbound_max_contact_count_4a": 4,
            }
        )

    model_sel = select_model_kind(
        rows,
        alpha=1.0,
        min_action_samples=2,
        feature_names=list(ABFE_STEERING_POLICY_FEATURE_NAMES),
        requested_kind="auto",
    )
    assert str(model_sel["model_kind"]) in {"ridge", "huber"}

    eval_target = evaluate_group_cv(
        rows,
        alpha=1.0,
        min_action_samples=2,
        feature_names=list(ABFE_STEERING_POLICY_FEATURE_NAMES),
        model_kind=str(model_sel["model_kind"]),
        group_key="target_group",
    )
    assert bool(eval_target["ok"]) is True
    assert int(eval_target["n_groups"]) == 2
    assert np.isfinite(float(eval_target["reward_rmse"]))

    ood = build_ood_profile(rows, ABFE_STEERING_POLICY_OOD_FEATURE_NAMES)
    assert bool(ood["ok"]) is True
    assert len(list(ood["ood_feature_names"])) >= 3
    assert np.isfinite(float(ood["ood_distance_threshold"]))
