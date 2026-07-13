#!/usr/bin/env python3
"""Compare docking affinity, raw ABFE, and calibrated ABFE on shared ligand rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


KEY_COLS_DEFAULT = ("target", "ligand")
PREDICTION_META_COLS = {
    "target",
    "ligand",
    "exp_dg_kj",
    "sample_weight",
    "endpoint_group",
    "split",
    "fold",
    "repeat",
    "row_index",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--abfe-csv",
        default="benchmark_results/_analysis/manifest_combined_results_all_targets_with_pocket.csv",
        help="ABFE benchmark CSV with target, ligand, exp_dg_kj, and raw ABFE outputs.",
    )
    p.add_argument(
        "--predictions-csv",
        default="benchmark_results/all/groupcv_predictions.csv",
        help="Optional calibrated ABFE prediction CSV (for example groupcv_predictions.csv).",
    )
    p.add_argument(
        "--prediction-cols",
        nargs="*",
        default=None,
        help="Explicit calibrated prediction columns to include. Default: auto-detect numeric predictors.",
    )
    p.add_argument(
        "--data-root",
        default="benchmarks/data",
        help="Benchmark data root used to recover missing experimental DeltaG values from ligands.yml.",
    )
    p.add_argument(
        "--temperature-k",
        type=float,
        default=300.0,
        help="Temperature used when converting Ki/Kd/IC50-style measurements to DeltaG.",
    )
    p.add_argument(
        "--fill-exp-from-yml",
        action="store_true",
        default=True,
        help="Recover missing experimental DeltaG values from benchmarks/data/<target>/ligands.yml.",
    )
    p.add_argument(
        "--no-fill-exp-from-yml",
        dest="fill_exp_from_yml",
        action="store_false",
    )
    p.add_argument(
        "--docking-csv",
        default="",
        help="Optional docking CSV with target, ligand, and affinity/score columns.",
    )
    p.add_argument("--target-col", default="target")
    p.add_argument("--ligand-col", default="ligand")
    p.add_argument("--exp-col", default="exp_dg_kj")
    p.add_argument("--docking-target-col", default="target")
    p.add_argument("--docking-ligand-col", default="ligand")
    p.add_argument(
        "--docking-affinity-col",
        action="append",
        default=[],
        help="Docking affinity column. Can be repeated. Default: affinity.",
    )
    p.add_argument(
        "--docking-unit",
        choices=("kcal", "kj"),
        default="kcal",
        help="Units for docking affinity columns.",
    )
    p.add_argument(
        "--fit-docking-affine-loto",
        action="store_true",
        default=True,
        help="Add leave-one-target-out affine docking calibration columns.",
    )
    p.add_argument(
        "--no-fit-docking-affine-loto",
        dest="fit_docking_affine_loto",
        action="store_false",
    )
    p.add_argument(
        "--out-dir",
        default="benchmark_results/abfe_vs_docking_comparison",
        help="Output directory for merged tables and reports.",
    )
    p.add_argument(
        "--fit-rel-hybrid-dock-stack-loto",
        action="store_true",
        default=True,
        help="Add leave-one-target-out linear stack using rel_hybrid and raw docking affinity when available.",
    )
    p.add_argument(
        "--no-fit-rel-hybrid-dock-stack-loto",
        dest="fit_rel_hybrid_dock_stack_loto",
        action="store_false",
    )
    return p.parse_args()


def _safe_float(x, default=np.nan) -> float:
    try:
        out = float(x)
    except Exception:
        out = float(default)
    return float(out) if np.isfinite(out) else float(default)


def _normalize_unit(unit: str) -> str:
    return str(unit or "").strip().lower().replace(" ", "").replace("µ", "u")


def _to_molar(value: float, unit: str) -> float:
    u = _normalize_unit(unit)
    if not np.isfinite(value):
        return float("nan")
    factor = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
        "fm": 1e-15,
    }
    if u in factor:
        return float(value * factor[u])
    if u in {"microm", "micromolar", "umolar", "u.m", "u.m."}:
        return float(value * 1e-6)
    if u in {"nanom", "nanomolar"}:
        return float(value * 1e-9)
    if u in {"millim", "millimolar"}:
        return float(value * 1e-3)
    if u in {"molar"}:
        return float(value)
    return float("nan")


def _dg_from_molar_k(activity_m: float, temperature_k: float) -> float:
    if not np.isfinite(activity_m) or activity_m <= 0.0:
        return float("nan")
    r_kj = 0.008314462618
    return float(r_kj * float(temperature_k) * math.log(float(activity_m)))


def _dg_from_measurement(meas: dict, temperature_k: float) -> tuple[float, str]:
    if not isinstance(meas, dict):
        return float("nan"), ""

    add = meas.get("additional", [])
    if isinstance(add, list):
        for item in add:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type", "")).strip().lower()
            if typ not in {"dg", "delta_g", "deltag"}:
                continue
            val = _safe_float(item.get("value"))
            if not np.isfinite(val):
                continue
            unit = _normalize_unit(str(item.get("unit", "")))
            if "kcal" in unit:
                return float(val * 4.184), "measurement.additional.dg_kcal"
            return float(val), "measurement.additional.dg"

    typ = str(meas.get("type", "")).strip().lower()
    val = _safe_float(meas.get("value"))
    unit = str(meas.get("unit", ""))

    if typ in {"dg", "delta_g", "deltag"} and np.isfinite(val):
        norm_unit = _normalize_unit(unit)
        if "kcal" in norm_unit:
            return float(val * 4.184), "measurement.dg_kcal"
        return float(val), "measurement.dg"
    if typ in {"pki", "pkd", "pic50"} and np.isfinite(val):
        r_kj = 0.008314462618
        dg = float(-r_kj * float(temperature_k) * math.log(10.0) * float(val))
        return dg, f"measurement.{typ}"
    if typ in {"ki", "kd", "ic50"} and np.isfinite(val):
        mol = _to_molar(float(val), unit)
        dg = _dg_from_molar_k(mol, temperature_k=float(temperature_k))
        if np.isfinite(dg):
            return dg, f"measurement.{typ}"
    return float("nan"), ""


def build_exp_map_from_ligands_yml(data_root: Path, temperature_k: float) -> Dict[tuple[str, str], tuple[float, str]]:
    out: Dict[tuple[str, str], tuple[float, str]] = {}
    if yaml is None or (not data_root.exists()):
        return out
    for target_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        yml_path = target_dir / "ligands.yml"
        if not yml_path.exists():
            continue
        try:
            data = yaml.safe_load(yml_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        target = str(target_dir.name)
        for ligand, rec in data.items():
            meas = rec.get("measurement") if isinstance(rec, dict) else None
            dg, src = _dg_from_measurement(meas if isinstance(meas, dict) else {}, temperature_k=float(temperature_k))
            if np.isfinite(dg):
                out[(target, str(ligand))] = (float(dg), str(src))
    return out


def fill_missing_exp_from_yml(
    df: pd.DataFrame,
    *,
    data_root: Path,
    temperature_k: float,
    exp_col: str = "exp_dg_kj",
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out[exp_col] = pd.to_numeric(out.get(exp_col, np.nan), errors="coerce")
    n_before = int(np.sum(np.isfinite(out[exp_col])))
    filled = 0
    sources: list[str] = []
    exp_map = build_exp_map_from_ligands_yml(data_root=data_root, temperature_k=float(temperature_k))
    if exp_map:
        for i, row in out.iterrows():
            if np.isfinite(float(out.at[i, exp_col])):
                continue
            key = (str(row.get("target", "")), str(row.get("ligand", "")))
            if key not in exp_map:
                continue
            dg, src = exp_map[key]
            out.at[i, exp_col] = float(dg)
            filled += 1
            sources.append(str(src))
    info = {
        "n_exp_before": int(n_before),
        "n_exp_after": int(np.sum(np.isfinite(out[exp_col]))),
        "n_exp_filled": int(filled),
        "fill_exp_from_yml": True,
        "data_root": str(data_root),
        "fill_sources": list(sorted(set(sources))),
    }
    return out, info


def _metric_rmse(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _metric_mae(y: np.ndarray, p: np.ndarray) -> float:
    if y.size == 0:
        return np.nan
    return float(np.mean(np.abs(p - y)))


def _metric_pearson(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 2:
        return np.nan
    sy = float(np.std(y))
    sp = float(np.std(p))
    if sy < 1.0e-12 or sp < 1.0e-12:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def _rank_average(x: Sequence[float]) -> np.ndarray:
    return pd.Series(list(x), dtype=float).rank(method="average").to_numpy(dtype=float)


def _metric_spearman(y: np.ndarray, p: np.ndarray) -> float:
    if y.size < 2:
        return np.nan
    return _metric_pearson(_rank_average(y), _rank_average(p))


def _fit_linear(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.column_stack([np.asarray(X, dtype=float), np.ones(len(X), dtype=float)])
    coef, _, _, _ = np.linalg.lstsq(x2, np.asarray(y, dtype=float), rcond=None)
    return np.asarray(coef, dtype=float)


def _predict_linear(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    x2 = np.column_stack([np.asarray(X, dtype=float), np.ones(len(X), dtype=float)])
    return np.asarray(x2 @ np.asarray(coef, dtype=float), dtype=float)


def choose_prediction_columns(pred_df: pd.DataFrame, explicit_cols: Sequence[str] | None = None) -> List[str]:
    if explicit_cols:
        cols = [str(c) for c in explicit_cols if str(c) in pred_df.columns]
        if cols:
            return cols
    cols = []
    for col in list(pred_df.columns):
        if str(col) in PREDICTION_META_COLS:
            continue
        ser = pd.to_numeric(pred_df[col], errors="coerce")
        if np.any(np.isfinite(ser.to_numpy(dtype=float))):
            cols.append(str(col))
    return cols


def load_abfe_base_table(
    abfe_csv: Path,
    *,
    target_col: str,
    ligand_col: str,
    exp_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(abfe_csv)
    for req in (target_col, ligand_col, exp_col):
        if req not in df.columns:
            raise RuntimeError(f"Missing required ABFE column: {req}")
    out = pd.DataFrame(
        {
            "target": df[target_col].astype(str),
            "ligand": df[ligand_col].astype(str),
            "exp_dg_kj": pd.to_numeric(df[exp_col], errors="coerce"),
        }
    )
    raw_map = {
        "dg_expavg_kj": "abfe_raw_expavg_kj",
        "dg_pop_or_jeff_kj": "abfe_raw_pop_or_jeff_kj",
        "dg_cumulant_kj": "abfe_raw_cumulant_kj",
    }
    for src, dst in raw_map.items():
        if src in df.columns:
            out[dst] = pd.to_numeric(df[src], errors="coerce")
    return out


def merge_prediction_table(
    merged: pd.DataFrame,
    pred_csv: Path,
    *,
    prediction_cols: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, List[str]]:
    pred = pd.read_csv(pred_csv)
    for req in KEY_COLS_DEFAULT:
        if req not in pred.columns:
            raise RuntimeError(f"Missing required calibrated-prediction column: {req}")
    use_cols = choose_prediction_columns(pred, explicit_cols=prediction_cols)
    keep = list(KEY_COLS_DEFAULT) + list(use_cols)
    sub = pred.loc[:, keep].copy()
    for col in use_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    out = merged.merge(sub, on=list(KEY_COLS_DEFAULT), how="left")
    return out, list(use_cols)


def _sanitize_name(text: str) -> str:
    return (
        str(text or "")
        .strip()
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def merge_docking_table(
    merged: pd.DataFrame,
    docking_csv: Path,
    *,
    target_col: str,
    ligand_col: str,
    affinity_cols: Sequence[str],
    affinity_unit: str,
) -> tuple[pd.DataFrame, List[str]]:
    dock = pd.read_csv(docking_csv)
    for req in (target_col, ligand_col):
        if req not in dock.columns:
            raise RuntimeError(f"Missing required docking column: {req}")
    use_aff_cols = list(affinity_cols or [])
    if not use_aff_cols:
        if "affinity" in dock.columns:
            use_aff_cols = ["affinity"]
        else:
            raise RuntimeError("No docking affinity column specified and `affinity` was not found.")

    sub = pd.DataFrame(
        {
            "target": dock[target_col].astype(str),
            "ligand": dock[ligand_col].astype(str),
        }
    )
    out_cols = []
    scale = 4.184 if str(affinity_unit).strip().lower() == "kcal" else 1.0
    for col in use_aff_cols:
        if col not in dock.columns:
            raise RuntimeError(f"Missing docking affinity column: {col}")
        name = f"dock_{_sanitize_name(col)}_kj"
        sub[name] = pd.to_numeric(dock[col], errors="coerce") * float(scale)
        out_cols.append(name)
    out = merged.merge(sub, on=list(KEY_COLS_DEFAULT), how="left")
    return out, out_cols


def fit_affine_loto_predictions(
    df: pd.DataFrame,
    *,
    predictor_col: str,
    target_col: str = "target",
    exp_col: str = "exp_dg_kj",
) -> np.ndarray:
    pred = np.full(len(df), np.nan, dtype=float)
    x_all = pd.to_numeric(df[predictor_col], errors="coerce").to_numpy(dtype=float)
    y_all = pd.to_numeric(df[exp_col], errors="coerce").to_numpy(dtype=float)
    groups = df[target_col].astype(str).to_numpy(dtype=object)
    uniq = sorted(set(groups.tolist()))
    for g in uniq:
        te = (groups == g) & np.isfinite(x_all) & np.isfinite(y_all)
        tr = (groups != g) & np.isfinite(x_all) & np.isfinite(y_all)
        if int(np.sum(te)) == 0 or int(np.sum(tr)) < 2:
            continue
        coef = _fit_linear(x_all[tr][:, None], y_all[tr])
        pred[te] = _predict_linear(x_all[te][:, None], coef)
    return np.asarray(pred, dtype=float)


def fit_linear_loto_stack_predictions(
    df: pd.DataFrame,
    *,
    predictor_cols: Sequence[str],
    target_col: str = "target",
    exp_col: str = "exp_dg_kj",
) -> np.ndarray:
    cols = [str(c) for c in list(predictor_cols or []) if str(c) in df.columns]
    pred = np.full(len(df), np.nan, dtype=float)
    if not cols:
        return pred
    x_all = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols])
    y_all = pd.to_numeric(df[exp_col], errors="coerce").to_numpy(dtype=float)
    groups = df[target_col].astype(str).to_numpy(dtype=object)
    finite_mask = np.isfinite(y_all) & np.all(np.isfinite(x_all), axis=1)
    uniq = sorted(set(groups.tolist()))
    min_train = int(len(cols) + 1)
    for g in uniq:
        te = (groups == g) & finite_mask
        tr = (groups != g) & finite_mask
        if int(np.sum(te)) == 0 or int(np.sum(tr)) < min_train:
            continue
        coef = _fit_linear(x_all[tr], y_all[tr])
        pred[te] = _predict_linear(x_all[te], coef)
    return np.asarray(pred, dtype=float)


def build_metrics_table(
    df: pd.DataFrame,
    *,
    predictor_cols: Sequence[str],
    exp_col: str = "exp_dg_kj",
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_cols = ["predictor", "n", "rmse_kj", "mae_kj", "pearson", "spearman"]
    per_target_cols = ["predictor", "target", "n", "rmse_kj", "mae_kj", "pearson", "spearman"]
    rows = []
    per_target_rows = []
    y_all = pd.to_numeric(df[exp_col], errors="coerce").to_numpy(dtype=float)
    for col in list(predictor_cols or []):
        if col not in df.columns:
            continue
        p_all = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y_all) & np.isfinite(p_all)
        y = y_all[mask]
        p = p_all[mask]
        rows.append(
            {
                "predictor": str(col),
                "n": int(y.size),
                "rmse_kj": _metric_rmse(y, p),
                "mae_kj": _metric_mae(y, p),
                "pearson": _metric_pearson(y, p),
                "spearman": _metric_spearman(y, p),
            }
        )
        sub = df.loc[mask, [target_col, exp_col, col]].copy()
        for target, g in sub.groupby(target_col, sort=True):
            yy = pd.to_numeric(g[exp_col], errors="coerce").to_numpy(dtype=float)
            pp = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(yy) & np.isfinite(pp)
            yy = yy[m]
            pp = pp[m]
            per_target_rows.append(
                {
                    "predictor": str(col),
                    "target": str(target),
                    "n": int(yy.size),
                    "rmse_kj": _metric_rmse(yy, pp),
                    "mae_kj": _metric_mae(yy, pp),
                    "pearson": _metric_pearson(yy, pp),
                    "spearman": _metric_spearman(yy, pp),
                }
            )
    overall = pd.DataFrame(rows, columns=overall_cols)
    if not overall.empty:
        overall = overall.sort_values(["rmse_kj", "mae_kj"], ascending=[True, True], kind="mergesort")
    per_target = pd.DataFrame(per_target_rows, columns=per_target_cols)
    if not per_target.empty:
        per_target = per_target.sort_values(["target", "rmse_kj"], kind="mergesort")
    return overall, per_target


def build_common_subset_metrics(
    df: pd.DataFrame,
    *,
    predictor_cols: Sequence[str],
    exp_col: str = "exp_dg_kj",
) -> pd.DataFrame:
    cols = [c for c in list(predictor_cols or []) if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=["predictor", "n", "rmse_kj", "mae_kj", "pearson", "spearman"])
    mask = np.isfinite(pd.to_numeric(df[exp_col], errors="coerce").to_numpy(dtype=float))
    for col in cols:
        mask = mask & np.isfinite(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
    keep_cols = [c for c in ["target", exp_col] + cols if c in df.columns]
    sub = df.loc[mask, keep_cols].copy()
    return build_metrics_table(sub, predictor_cols=cols, exp_col=exp_col, target_col="target")[0]


def summarize_head_to_head(overall: pd.DataFrame) -> dict:
    if overall.empty:
        return {"ok": False, "note": "no_metrics"}
    pred_names = set(overall["predictor"].astype(str).tolist())
    dock_rows = overall[overall["predictor"].astype(str).str.startswith("dock_")].copy()
    abfe_rows = overall[~overall["predictor"].astype(str).str.startswith("dock_")].copy()
    summary = {
        "ok": True,
        "predictors": list(sorted(pred_names)),
    }
    if not dock_rows.empty:
        best_dock_rmse = dock_rows.loc[dock_rows["rmse_kj"].idxmin()].to_dict()
        best_dock_pearson = dock_rows.loc[dock_rows["pearson"].idxmax()].to_dict()
        summary["best_docking_rmse"] = dict(best_dock_rmse)
        summary["best_docking_pearson"] = dict(best_dock_pearson)
    if not abfe_rows.empty:
        best_abfe_rmse = abfe_rows.loc[abfe_rows["rmse_kj"].idxmin()].to_dict()
        best_abfe_pearson = abfe_rows.loc[abfe_rows["pearson"].idxmax()].to_dict()
        summary["best_abfe_rmse"] = dict(best_abfe_rmse)
        summary["best_abfe_pearson"] = dict(best_abfe_pearson)
    if ("best_docking_rmse" in summary) and ("best_abfe_rmse" in summary):
        summary["delta_rmse_abfe_minus_docking"] = float(
            _safe_float(summary["best_abfe_rmse"].get("rmse_kj")) - _safe_float(summary["best_docking_rmse"].get("rmse_kj"))
        )
    if ("best_docking_pearson" in summary) and ("best_abfe_pearson" in summary):
        summary["delta_pearson_abfe_minus_docking"] = float(
            _safe_float(summary["best_abfe_pearson"].get("pearson")) - _safe_float(summary["best_docking_pearson"].get("pearson"))
        )
    return summary


def write_report(
    out_path: Path,
    *,
    abfe_csv: Path,
    predictions_csv: Path | None,
    docking_csv: Path | None,
    fill_info: dict | None,
    overall: pd.DataFrame,
    common_subset: pd.DataFrame,
    head_to_head: dict,
) -> None:
    lines = [
        "# ABFE vs Docking Comparison",
        "",
        "## Inputs",
        f"- abfe_csv: {abfe_csv}",
        f"- predictions_csv: {predictions_csv if predictions_csv else ''}",
        f"- docking_csv: {docking_csv if docking_csv else ''}",
    ]
    if isinstance(fill_info, dict) and bool(fill_info):
        lines.extend(
            [
                f"- fill_exp_from_yml: {bool(fill_info.get('fill_exp_from_yml', False))}",
                f"- data_root: {fill_info.get('data_root', '')}",
                f"- exp before/after/fill: {int(fill_info.get('n_exp_before', 0))} / "
                f"{int(fill_info.get('n_exp_after', 0))} / +{int(fill_info.get('n_exp_filled', 0))}",
            ]
        )
    lines.extend(
        [
            "",
            "## Overall Metrics",
            "",
            "| Predictor | n | RMSE (kJ/mol) | MAE (kJ/mol) | Pearson | Spearman |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['predictor']} | {int(row['n'])} | {float(row['rmse_kj']):.3f} | {float(row['mae_kj']):.3f} | "
            f"{float(row['pearson']):.3f} | {float(row['spearman']):.3f} |"
        )
    if not common_subset.empty:
        lines.extend(
            [
                "",
                "## Common-Subset Metrics",
                "",
                "| Predictor | n | RMSE (kJ/mol) | MAE (kJ/mol) | Pearson | Spearman |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in common_subset.iterrows():
            lines.append(
                f"| {row['predictor']} | {int(row['n'])} | {float(row['rmse_kj']):.3f} | {float(row['mae_kj']):.3f} | "
                f"{float(row['pearson']):.3f} | {float(row['spearman']):.3f} |"
            )
    if bool(head_to_head.get("ok", False)):
        lines.extend(["", "## Verdict"])
        if "best_docking_rmse" in head_to_head:
            lines.append(
                f"- best_docking_rmse: `{head_to_head['best_docking_rmse'].get('predictor','')}` "
                f"(RMSE={_safe_float(head_to_head['best_docking_rmse'].get('rmse_kj')):.3f})"
            )
        if "best_abfe_rmse" in head_to_head:
            lines.append(
                f"- best_abfe_rmse: `{head_to_head['best_abfe_rmse'].get('predictor','')}` "
                f"(RMSE={_safe_float(head_to_head['best_abfe_rmse'].get('rmse_kj')):.3f})"
            )
        if "delta_rmse_abfe_minus_docking" in head_to_head:
            lines.append(f"- delta_rmse_abfe_minus_docking: {float(head_to_head['delta_rmse_abfe_minus_docking']):.3f}")
        if "best_docking_pearson" in head_to_head:
            lines.append(
                f"- best_docking_pearson: `{head_to_head['best_docking_pearson'].get('predictor','')}` "
                f"(Pearson={_safe_float(head_to_head['best_docking_pearson'].get('pearson')):.3f})"
            )
        if "best_abfe_pearson" in head_to_head:
            lines.append(
                f"- best_abfe_pearson: `{head_to_head['best_abfe_pearson'].get('predictor','')}` "
                f"(Pearson={_safe_float(head_to_head['best_abfe_pearson'].get('pearson')):.3f})"
            )
        if "delta_pearson_abfe_minus_docking" in head_to_head:
            lines.append(
                f"- delta_pearson_abfe_minus_docking: {float(head_to_head['delta_pearson_abfe_minus_docking']):.3f}"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    abfe_csv = Path(args.abfe_csv).resolve()
    if not abfe_csv.exists():
        raise FileNotFoundError(f"ABFE CSV not found: {abfe_csv}")
    predictions_csv = Path(args.predictions_csv).resolve() if str(args.predictions_csv).strip() else None
    docking_csv = Path(args.docking_csv).resolve() if str(args.docking_csv).strip() else None
    if predictions_csv is not None and (not predictions_csv.exists()):
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")
    if docking_csv is not None and (not docking_csv.exists()):
        raise FileNotFoundError(f"Docking CSV not found: {docking_csv}")
    data_root_raw = str(getattr(args, "data_root", "benchmarks/data") or "").strip()
    data_root = Path(data_root_raw).resolve() if data_root_raw else None
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_abfe_base_table(
        abfe_csv,
        target_col=str(args.target_col),
        ligand_col=str(args.ligand_col),
        exp_col=str(args.exp_col),
    )
    fill_info = {
        "fill_exp_from_yml": False,
        "data_root": str(data_root) if data_root else "",
        "n_exp_before": int(np.sum(np.isfinite(pd.to_numeric(merged["exp_dg_kj"], errors="coerce").to_numpy(dtype=float)))),
        "n_exp_after": int(np.sum(np.isfinite(pd.to_numeric(merged["exp_dg_kj"], errors="coerce").to_numpy(dtype=float)))),
        "n_exp_filled": 0,
    }
    if bool(getattr(args, "fill_exp_from_yml", True)) and data_root is not None:
        merged, fill_info = fill_missing_exp_from_yml(
            merged,
            data_root=data_root,
            temperature_k=float(getattr(args, "temperature_k", 300.0)),
            exp_col="exp_dg_kj",
        )
    predictor_cols: List[str] = [c for c in merged.columns if c.startswith("abfe_raw_")]

    pred_cols_used: List[str] = []
    if predictions_csv is not None and predictions_csv.exists():
        merged, pred_cols_used = merge_prediction_table(
            merged,
            predictions_csv,
            prediction_cols=args.prediction_cols,
        )
        predictor_cols.extend(list(pred_cols_used))

    dock_cols_used: List[str] = []
    if docking_csv is not None and docking_csv.exists():
        merged, dock_cols_used = merge_docking_table(
            merged,
            docking_csv,
            target_col=str(args.docking_target_col),
            ligand_col=str(args.docking_ligand_col),
            affinity_cols=list(args.docking_affinity_col or []),
            affinity_unit=str(args.docking_unit),
        )
        predictor_cols.extend(list(dock_cols_used))
        if bool(args.fit_docking_affine_loto):
            for col in list(dock_cols_used):
                out_col = f"{col}_loto_affine"
                merged[out_col] = fit_affine_loto_predictions(
                    merged,
                    predictor_col=str(col),
                    target_col="target",
                    exp_col="exp_dg_kj",
                )
                if np.any(np.isfinite(pd.to_numeric(merged[out_col], errors="coerce").to_numpy(dtype=float))):
                    predictor_cols.append(out_col)
        if bool(getattr(args, "fit_rel_hybrid_dock_stack_loto", True)) and ("rel_hybrid" in merged.columns):
            for col in list(dock_cols_used):
                out_col = f"stack_rel_hybrid_{_sanitize_name(col)}_loto"
                merged[out_col] = fit_linear_loto_stack_predictions(
                    merged,
                    predictor_cols=["rel_hybrid", str(col)],
                    target_col="target",
                    exp_col="exp_dg_kj",
                )
                if np.any(np.isfinite(pd.to_numeric(merged[out_col], errors="coerce").to_numpy(dtype=float))):
                    predictor_cols.append(out_col)
        if bool(getattr(args, "fit_rel_hybrid_dock_stack_loto", True)) and ("dock_affinity_kj" in merged.columns):
            stack_cols = [c for c in ["cal_combo_ep", "cal_expavg", "dock_affinity_kj"] if c in merged.columns]
            if len(stack_cols) == 3:
                out_col = "stack_cal_combo_ep_cal_expavg_dock_loto"
                merged[out_col] = fit_linear_loto_stack_predictions(
                    merged,
                    predictor_cols=stack_cols,
                    target_col="target",
                    exp_col="exp_dg_kj",
                )
                if np.any(np.isfinite(pd.to_numeric(merged[out_col], errors="coerce").to_numpy(dtype=float))):
                    predictor_cols.append(out_col)

    predictor_cols = [c for c in predictor_cols if c in merged.columns]
    overall, per_target = build_metrics_table(
        merged,
        predictor_cols=predictor_cols,
        exp_col="exp_dg_kj",
        target_col="target",
    )
    common_subset = build_common_subset_metrics(
        merged,
        predictor_cols=predictor_cols,
        exp_col="exp_dg_kj",
    )
    head_to_head = summarize_head_to_head(overall)

    merged_csv = out_dir / "merged_predictions.csv"
    overall_csv = out_dir / "overall_metrics.csv"
    per_target_csv = out_dir / "per_target_metrics.csv"
    common_csv = out_dir / "common_subset_metrics.csv"
    summary_json = out_dir / "comparison_summary.json"
    report_md = out_dir / "comparison_report.md"

    merged.to_csv(merged_csv, index=False)
    overall.to_csv(overall_csv, index=False)
    per_target.to_csv(per_target_csv, index=False)
    common_subset.to_csv(common_csv, index=False)
    payload = {
        "abfe_csv": str(abfe_csv),
        "predictions_csv": str(predictions_csv) if predictions_csv else "",
        "docking_csv": str(docking_csv) if docking_csv else "",
        "predictor_cols": list(predictor_cols),
        "prediction_cols_used": list(pred_cols_used),
        "docking_cols_used": list(dock_cols_used),
        "fill_info": dict(fill_info),
        "head_to_head": dict(head_to_head),
        "files": {
            "merged_predictions_csv": str(merged_csv),
            "overall_metrics_csv": str(overall_csv),
            "per_target_metrics_csv": str(per_target_csv),
            "common_subset_metrics_csv": str(common_csv),
            "report_md": str(report_md),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(
        report_md,
        abfe_csv=abfe_csv,
        predictions_csv=predictions_csv,
        docking_csv=docking_csv,
        fill_info=fill_info,
        overall=overall,
        common_subset=common_subset,
        head_to_head=head_to_head,
    )
    return payload


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(f"Wrote: {payload['files']['merged_predictions_csv']}")
    print(f"Wrote: {payload['files']['overall_metrics_csv']}")
    print(f"Wrote: {payload['files']['per_target_metrics_csv']}")
    print(f"Wrote: {payload['files']['common_subset_metrics_csv']}")
    print(f"Wrote: {payload['files']['report_md']}")
    head = dict(payload.get("head_to_head", {}) or {})
    if bool(head.get("ok", False)):
        if "best_abfe_rmse" in head:
            print(
                f"Best ABFE RMSE: {head['best_abfe_rmse'].get('predictor')} = "
                f"{_safe_float(head['best_abfe_rmse'].get('rmse_kj')):.3f} kJ/mol"
            )
        if "best_docking_rmse" in head:
            print(
                f"Best docking RMSE: {head['best_docking_rmse'].get('predictor')} = "
                f"{_safe_float(head['best_docking_rmse'].get('rmse_kj')):.3f} kJ/mol"
            )


if __name__ == "__main__":
    main()
