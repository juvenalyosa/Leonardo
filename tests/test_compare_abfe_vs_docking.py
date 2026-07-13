import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import compare_abfe_vs_docking as cabd


def test_fit_affine_loto_predictions_recovers_linear_relation():
    df = pd.DataFrame(
        {
            "target": ["A", "A", "B", "B"],
            "ligand": ["l1", "l2", "l3", "l4"],
            "exp_dg_kj": [3.0, 5.0, 7.0, 9.0],
            "dock_affinity_kj": [1.0, 2.0, 3.0, 4.0],
        }
    )
    pred = cabd.fit_affine_loto_predictions(
        df,
        predictor_col="dock_affinity_kj",
        target_col="target",
        exp_col="exp_dg_kj",
    )
    assert np.allclose(pred, df["exp_dg_kj"].to_numpy(dtype=float), atol=1.0e-8)

    scored = df.copy()
    scored["dock_affinity_kj_loto_affine"] = pred
    overall, per_target = cabd.build_metrics_table(
        scored,
        predictor_cols=["dock_affinity_kj_loto_affine"],
        exp_col="exp_dg_kj",
        target_col="target",
    )
    assert int(overall.iloc[0]["n"]) == 4
    assert float(overall.iloc[0]["rmse_kj"]) < 1.0e-8
    assert int(per_target["target"].nunique()) == 2


def test_fit_linear_loto_stack_predictions_recovers_linear_relation():
    df = pd.DataFrame(
        {
            "target": ["A", "A", "B", "B", "C", "C"],
            "ligand": ["l1", "l2", "l3", "l4", "l5", "l6"],
            "rel_hybrid": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "dock_affinity_kj": [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }
    )
    df["exp_dg_kj"] = 2.0 * df["rel_hybrid"] + 0.5 * df["dock_affinity_kj"] + 1.0
    pred = cabd.fit_linear_loto_stack_predictions(
        df,
        predictor_cols=["rel_hybrid", "dock_affinity_kj"],
        target_col="target",
        exp_col="exp_dg_kj",
    )
    assert np.allclose(pred, df["exp_dg_kj"].to_numpy(dtype=float), atol=1.0e-8)


def test_run_merges_raw_abfe_cv_predictions_and_docking():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        abfe_csv = root / "abfe.csv"
        pred_csv = root / "pred.csv"
        dock_csv = root / "dock.csv"
        out_dir = root / "out"

        pd.DataFrame(
            {
                "target": ["T1", "T1", "T2", "T2"],
                "ligand": ["a", "b", "c", "d"],
                "exp_dg_kj": [-10.0, -12.0, -8.0, -6.0],
                "dg_expavg_kj": [-11.0, -11.5, -8.5, -5.0],
                "dg_pop_or_jeff_kj": [-9.5, -10.5, -7.0, -7.5],
            }
        ).to_csv(abfe_csv, index=False)

        pd.DataFrame(
            {
                "target": ["T1", "T1", "T2", "T2"],
                "ligand": ["a", "b", "c", "d"],
                "exp_dg_kj": [-10.0, -12.0, -8.0, -6.0],
                "raw_expavg": [-11.0, -11.5, -8.5, -5.0],
                "raw_pop": [-9.5, -10.5, -7.0, -7.5],
                "cal_combo": [-10.2, -11.8, -8.1, -6.2],
                "rel_hybrid": [-10.1, -11.6, -7.9, -6.1],
            }
        ).to_csv(pred_csv, index=False)

        pd.DataFrame(
            {
                "target": ["T1", "T1", "T2", "T2"],
                "ligand": ["a", "b", "c", "d"],
                "affinity": [-2.390057, -2.868068, -1.912046, -1.434034],
            }
        ).to_csv(dock_csv, index=False)

        class _Args:
            pass

        ns = _Args()
        ns.abfe_csv = str(abfe_csv)
        ns.predictions_csv = str(pred_csv)
        ns.prediction_cols = ["cal_combo", "rel_hybrid"]
        ns.docking_csv = str(dock_csv)
        ns.target_col = "target"
        ns.ligand_col = "ligand"
        ns.exp_col = "exp_dg_kj"
        ns.docking_target_col = "target"
        ns.docking_ligand_col = "ligand"
        ns.docking_affinity_col = ["affinity"]
        ns.docking_unit = "kcal"
        ns.fit_docking_affine_loto = True
        ns.out_dir = str(out_dir)

        payload = cabd.run(ns)
        merged = pd.read_csv(payload["files"]["merged_predictions_csv"])
        overall = pd.read_csv(payload["files"]["overall_metrics_csv"])
        common = pd.read_csv(payload["files"]["common_subset_metrics_csv"])

        expected_cols = {
            "abfe_raw_expavg_kj",
            "abfe_raw_pop_or_jeff_kj",
            "cal_combo",
            "rel_hybrid",
            "dock_affinity_kj",
            "dock_affinity_kj_loto_affine",
        }
        assert expected_cols.issubset(set(merged.columns))
        assert {"cal_combo", "rel_hybrid", "dock_affinity_kj", "dock_affinity_kj_loto_affine"}.issubset(
            set(overall["predictor"].astype(str))
        )
        assert int(common["n"].max()) == 4
        assert bool(payload["head_to_head"]["ok"]) is True


def test_run_fills_exp_from_yml_and_builds_stack():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_root = root / "benchmarks/data"
        abfe_csv = root / "abfe.csv"
        pred_csv = root / "pred.csv"
        dock_csv = root / "dock.csv"
        out_dir = root / "out"

        rows = []
        pred_rows = []
        dock_rows = []
        yml_values = {
            ("T1", "a"): -10.0,
            ("T1", "b"): -12.0,
            ("T2", "c"): -8.0,
            ("T2", "d"): -6.0,
            ("T3", "e"): -14.0,
            ("T3", "f"): -16.0,
        }
        for (target, ligand), exp in yml_values.items():
            rows.append(
                {
                    "target": target,
                    "ligand": ligand,
                    "exp_dg_kj": np.nan,
                    "dg_expavg_kj": exp - 0.8,
                    "dg_pop_or_jeff_kj": exp + 0.6,
                }
            )
            pred_rows.append(
                {
                    "target": target,
                    "ligand": ligand,
                    "exp_dg_kj": exp,
                    "rel_hybrid": exp + 0.2,
                }
            )
            dock_rows.append(
                {
                    "target": target,
                    "ligand": ligand,
                    "affinity": exp - 1.0,
                }
            )

        pd.DataFrame(rows).to_csv(abfe_csv, index=False)
        pd.DataFrame(pred_rows).to_csv(pred_csv, index=False)
        pd.DataFrame(dock_rows).to_csv(dock_csv, index=False)

        for target in ["T1", "T2", "T3"]:
            tdir = data_root / target
            tdir.mkdir(parents=True, exist_ok=True)
            payload = {}
            for (tgt, lig), exp in yml_values.items():
                if tgt != target:
                    continue
                payload[lig] = {
                    "measurement": {
                        "type": "dg",
                        "value": float(exp),
                        "unit": "kilojoules / mole",
                    }
                }
            blocks = []
            for lig, info in payload.items():
                blocks.extend(
                    [
                        f"{lig}:",
                        "  measurement:",
                        "    type: dg",
                        f"    value: {float(info['measurement']['value'])}",
                        "    unit: kilojoules / mole",
                    ]
                )
            yml_text = "\n".join(blocks)
            (tdir / "ligands.yml").write_text(yml_text + "\n", encoding="utf-8")

        class _Args:
            pass

        ns = _Args()
        ns.abfe_csv = str(abfe_csv)
        ns.predictions_csv = str(pred_csv)
        ns.prediction_cols = ["rel_hybrid"]
        ns.data_root = str(data_root)
        ns.temperature_k = 300.0
        ns.fill_exp_from_yml = True
        ns.docking_csv = str(dock_csv)
        ns.target_col = "target"
        ns.ligand_col = "ligand"
        ns.exp_col = "exp_dg_kj"
        ns.docking_target_col = "target"
        ns.docking_ligand_col = "ligand"
        ns.docking_affinity_col = ["affinity"]
        ns.docking_unit = "kj"
        ns.fit_docking_affine_loto = True
        ns.fit_rel_hybrid_dock_stack_loto = True
        ns.out_dir = str(out_dir)

        payload = cabd.run(ns)
        merged = pd.read_csv(payload["files"]["merged_predictions_csv"])
        overall = pd.read_csv(payload["files"]["overall_metrics_csv"])

        assert int(payload["fill_info"]["n_exp_before"]) == 0
        assert int(payload["fill_info"]["n_exp_after"]) == 6
        assert int(payload["fill_info"]["n_exp_filled"]) == 6
        assert int(merged["exp_dg_kj"].notna().sum()) == 6
        assert "stack_rel_hybrid_dock_affinity_kj_loto" in merged.columns
        assert "stack_rel_hybrid_dock_affinity_kj_loto" in set(overall["predictor"].astype(str))
