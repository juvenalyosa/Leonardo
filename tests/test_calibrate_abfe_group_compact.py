import numpy as np
import pandas as pd

import calibrate_abfe_group_compact as cagc


def test_merge_docking_table_scales_kcal(tmp_path):
    base = pd.DataFrame(
        {
            "target": ["T1", "T1", "T2"],
            "ligand": ["a", "b", "c"],
            "exp_dg_kj_model": [-10.0, -12.0, -8.0],
            "dg_expavg_kj": [-9.0, -11.0, -7.0],
            "dg_pop_or_jeff_kj": [-10.5, -12.5, -8.5],
        }
    )
    dock_csv = tmp_path / "dock.csv"
    pd.DataFrame(
        {
            "target": ["T1", "T1", "T2"],
            "ligand": ["a", "b", "c"],
            "affinity": [-2.0, -3.0, -4.0],
        }
    ).to_csv(dock_csv, index=False)

    merged, info = cagc.merge_docking_table(
        base,
        docking_csv=dock_csv,
        target_col="target",
        ligand_col="ligand",
        affinity_col="affinity",
        affinity_unit="kcal",
    )

    assert info["docking_col"] == "dock_affinity_kj"
    assert np.allclose(merged["dock_affinity_kj"].to_numpy(float), np.array([-8.368, -12.552, -16.736]), atol=1e-6)


def test_append_groupcv_stack_predictions_adds_fold_rows():
    fold_df = pd.DataFrame(
        [
            {"model": "cal_combo_ep", "fold_target": "T1", "n_test": 2, "rmse": 0.1, "mae": 0.1, "pearson": 1.0, "spearman": 1.0, "alpha": np.nan},
            {"model": "cal_combo_ep", "fold_target": "T2", "n_test": 2, "rmse": 0.1, "mae": 0.1, "pearson": 1.0, "spearman": 1.0, "alpha": np.nan},
            {"model": "cal_combo_ep", "fold_target": "T3", "n_test": 2, "rmse": 0.1, "mae": 0.1, "pearson": 1.0, "spearman": 1.0, "alpha": np.nan},
        ]
    )
    pred_df = pd.DataFrame(
        {
            "target": ["T1", "T1", "T2", "T2", "T3", "T3"],
            "ligand": ["a", "b", "c", "d", "e", "f"],
            "sample_weight": [1.0] * 6,
            "cal_combo_ep": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "cal_expavg": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            "dock_affinity_kj": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }
    )
    pred_df["exp_dg_kj"] = 1.5 * pred_df["cal_combo_ep"] - 0.25 * pred_df["cal_expavg"] + 0.5 * pred_df["dock_affinity_kj"] + 1.0

    fold_out, pred_out = cagc.append_groupcv_stack_predictions(
        fold_df,
        pred_df,
        feature_cols=["cal_combo_ep", "cal_expavg", "dock_affinity_kj"],
        predictor_name="stack_cal_combo_ep_cal_expavg_dock_loto",
    )

    assert "stack_cal_combo_ep_cal_expavg_dock_loto" in pred_out.columns
    assert int((fold_out["model"] == "stack_cal_combo_ep_cal_expavg_dock_loto").sum()) == 3
    vals = pred_out["stack_cal_combo_ep_cal_expavg_dock_loto"].to_numpy(float)
    assert np.all(np.isfinite(vals))
