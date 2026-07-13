#!/usr/bin/env python3
"""Compact descriptor calibration for ABFE-lite with stronger data handling.

Improvements implemented:
1) Experimental-label recovery from benchmark_data/<target>/ligands.yml
   (Ki/Kd/IC50/pIC50/pKi/pKd and direct dG entries)
2) Quality-weighted fitting (uncertainty/state-balance/frame-aware weights)
3) Relative-per-target hybrid model (DeltaDeltaG-style centered model + absolute anchor)
4) Known-target panel CV (target-aware model) to estimate practical ranking when
   scoring congeneric series for targets represented in training data.
5) Endpoint-aware residual correction (ki/kd/ic50/dg assay families) for cross-target transfer.
6) Hierarchical target-wise linear correction (shrunken) for known-target panel inference.

Evaluation outputs:
- Leave-one-target-out (LOTO): unknown-target generalization
- Within-target stratified CV: known-target panel performance
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class FoldRow:
    model: str
    fold_target: str
    n_test: int
    rmse: float
    mae: float
    pearson: float
    spearman: float
    alpha: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        default="benchmark_results/diverse_abfe_panel_best_all_available/manifest_combined_results.csv",
    )
    p.add_argument(
        "--data-root",
        default="benchmark_data",
        help="Root with <target>/ligands.yml used for experimental recovery",
    )
    p.add_argument(
        "--out-dir",
        default="benchmark_results/diverse_abfe_panel_best_all_available/calibration_compact",
    )
    p.add_argument(
        "--alphas",
        default="0,0.01,0.1,0.3,1,3,10,30,100",
        help="Comma-separated ridge alpha grid",
    )
    p.add_argument(
        "--temperature-k",
        type=float,
        default=300.0,
        help="Temperature used to convert affinity to DeltaG",
    )
    p.add_argument(
        "--no-fill-exp-from-yml",
        action="store_true",
        help="Disable experimental DeltaG recovery from ligands.yml",
    )
    p.add_argument(
        "--weight-mode",
        choices=["none", "quality"],
        default="quality",
        help="Sample weighting strategy for model fitting",
    )
    p.add_argument(
        "--within-target-repeats",
        type=int,
        default=250,
        help="Repeats for known-target stratified CV",
    )
    p.add_argument(
        "--within-target-test-frac",
        type=float,
        default=0.20,
        help="Test fraction per target in known-target CV",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260215,
        help="Random seed",
    )
    p.add_argument(
        "--endpoint-aware",
        action="store_true",
        default=True,
        help="Enable endpoint-aware residual correction (ki/kd/ic50/dg groups) in LOTO.",
    )
    p.add_argument(
        "--no-endpoint-aware",
        action="store_false",
        dest="endpoint_aware",
        help="Disable endpoint-aware residual correction.",
    )
    p.add_argument(
        "--endpoint-min-rows",
        type=int,
        default=10,
        help="Minimum rows per endpoint group to estimate endpoint residual offsets.",
    )
    p.add_argument(
        "--hier-shrink-k",
        type=float,
        default=8.0,
        help="Shrinkage strength for target-wise linear correction in within-target CV.",
    )
    p.add_argument(
        "--endpoint-filter",
        default="all",
        help="Optional endpoint group filter: all|ki_like|kd_like|ic50_like|dg_like|unknown",
    )
    p.add_argument(
        "--ic50-normalization",
        choices=["none", "cp_only", "shift_only", "cp_or_shift"],
        default="cp_only",
        help=(
            "Normalize ic50-like endpoints to ki-like scale: "
            "cp_only=Cheng-Prusoff when metadata exists; "
            "shift_only=dataset shift alignment; "
            "cp_or_shift=CP first then shift fallback"
        ),
    )
    p.add_argument(
        "--ic50-shift-min-rows",
        type=int,
        default=4,
        help="Minimum rows per endpoint within target to estimate target-specific ic50->ki shift.",
    )
    p.add_argument(
        "--docking-csv",
        default="",
        help="Optional docking CSV to merge into the calibration table.",
    )
    p.add_argument("--docking-target-col", default="target")
    p.add_argument("--docking-ligand-col", default="ligand")
    p.add_argument(
        "--docking-affinity-col",
        default="affinity",
        help="Docking affinity column to use when --docking-csv is provided.",
    )
    p.add_argument(
        "--docking-unit",
        choices=["kcal", "kj"],
        default="kcal",
        help="Units for the docking affinity column.",
    )
    p.add_argument(
        "--no-docking-stack",
        action="store_true",
        help="Disable the docking-aware stacked model even when docking data are available.",
    )
    return p.parse_args()


def rmse(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        return float(np.sqrt(np.mean((p - y) ** 2)))
    ws = np.asarray(w, dtype=float)
    ws = np.where(np.isfinite(ws), np.clip(ws, 1e-12, None), 1e-12)
    err2 = (p - y) ** 2
    return float(np.sqrt(np.sum(ws * err2) / np.sum(ws)))


def mae(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    if w is None:
        return float(np.mean(np.abs(p - y)))
    ws = np.asarray(w, dtype=float)
    ws = np.where(np.isfinite(ws), np.clip(ws, 1e-12, None), 1e-12)
    err = np.abs(p - y)
    return float(np.sum(ws * err) / np.sum(ws))


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    sy = float(np.std(y))
    sp = float(np.std(p))
    if sy < 1e-12 or sp < 1e-12:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    yr = pd.Series(y).rank(method="average").to_numpy(float)
    pr = pd.Series(p).rank(method="average").to_numpy(float)
    return pearson(yr, pr)


def _safe_float(v) -> float:
    try:
        out = float(v)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return out


def _normalize_unit(unit: str) -> str:
    s = (unit or "").strip().lower().replace("µ", "u")
    s = s.replace(" ", "")
    return s


def _dg_from_molar_k(activity_m: float, temperature_k: float) -> float:
    if not np.isfinite(activity_m) or activity_m <= 0.0:
        return float("nan")
    r_kj = 0.008314462618
    return float(r_kj * float(temperature_k) * np.log(activity_m))


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


def _dg_from_measurement(meas: dict, temperature_k: float) -> Tuple[float, str]:
    if not isinstance(meas, dict):
        return float("nan"), ""

    add = meas.get("additional", [])
    if isinstance(add, list):
        for item in add:
            if not isinstance(item, dict):
                continue
            t = str(item.get("type", "")).strip().lower()
            if t not in {"dg", "delta_g", "deltag"}:
                continue
            v = _safe_float(item.get("value"))
            if not np.isfinite(v):
                continue
            unit = _normalize_unit(str(item.get("unit", "")))
            if "kcal" in unit:
                return float(v * 4.184), "measurement.additional.dg_kcal"
            return float(v), "measurement.additional.dg"

    t = str(meas.get("type", "")).strip().lower()
    v = _safe_float(meas.get("value"))
    unit = str(meas.get("unit", ""))

    if t in {"dg", "delta_g", "deltag"} and np.isfinite(v):
        uu = _normalize_unit(unit)
        if "kcal" in uu:
            return float(v * 4.184), "measurement.dg_kcal"
        return float(v), "measurement.dg"

    if t in {"pki", "pkd", "pic50"} and np.isfinite(v):
        r_kj = 0.008314462618
        dg = float(-r_kj * float(temperature_k) * math.log(10.0) * v)
        return dg, f"measurement.{t}"

    if t in {"ki", "kd", "ic50"} and np.isfinite(v):
        mol = _to_molar(v, unit)
        dg = _dg_from_molar_k(mol, temperature_k=temperature_k)
        if np.isfinite(dg):
            return dg, f"measurement.{t}"

    return float("nan"), ""


def build_exp_map_from_ligands_yml(data_root: Path, temperature_k: float) -> Dict[Tuple[str, str], Tuple[float, str]]:
    out: Dict[Tuple[str, str], Tuple[float, str]] = {}
    if yaml is None:
        return out
    for target_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        yml = target_dir / "ligands.yml"
        if not yml.exists():
            continue
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        tgt = str(target_dir.name)
        for lig, rec in data.items():
            lig_name = str(lig)
            meas = rec.get("measurement") if isinstance(rec, dict) else None
            dg, src = _dg_from_measurement(meas if isinstance(meas, dict) else {}, temperature_k=temperature_k)
            if np.isfinite(dg):
                out[(tgt, lig_name)] = (float(dg), src)
    return out


def _extract_value_unit(x, default_unit: str = "") -> Tuple[float, str]:
    if isinstance(x, dict):
        v = _safe_float(x.get("value"))
        u = str(x.get("unit", default_unit or ""))
        return v, u
    v = _safe_float(x)
    return v, str(default_unit or "")


def _extract_measurement_field(
    meas: dict,
    direct_keys: Sequence[str],
    additional_types: Sequence[str],
    default_unit: str = "",
) -> Tuple[float, str]:
    if not isinstance(meas, dict):
        return float("nan"), ""

    for k in direct_keys:
        if k not in meas:
            continue
        v, u = _extract_value_unit(meas.get(k), default_unit=default_unit)
        if np.isfinite(v):
            return v, u

    add = meas.get("additional", [])
    if isinstance(add, list):
        want = {str(t).strip().lower() for t in additional_types}
        for item in add:
            if not isinstance(item, dict):
                continue
            typ = str(item.get("type", "")).strip().lower()
            if typ not in want:
                continue
            v = _safe_float(item.get("value"))
            if not np.isfinite(v):
                continue
            u = str(item.get("unit", default_unit or ""))
            return v, u

    return float("nan"), ""


def _cheng_prusoff_factor_from_measurement(meas: dict) -> float:
    if not isinstance(meas, dict):
        return float("nan")

    s_val, s_unit = _extract_measurement_field(
        meas,
        direct_keys=("substrate_concentration", "substrate_conc", "substrate", "s_conc", "s", "[s]"),
        additional_types=("substrate_concentration", "substrate_conc", "substrate", "s_conc", "s", "[s]"),
        default_unit="",
    )
    km_val, km_unit = _extract_measurement_field(
        meas,
        direct_keys=("km", "k_m", "michaelis_constant", "michaelis"),
        additional_types=("km", "k_m", "michaelis_constant", "michaelis"),
        default_unit="",
    )

    if not (np.isfinite(s_val) and np.isfinite(km_val) and s_val > 0.0 and km_val > 0.0):
        return float("nan")

    s_m = _to_molar(s_val, s_unit)
    km_m = _to_molar(km_val, km_unit)

    if np.isfinite(s_m) and np.isfinite(km_m) and s_m > 0.0 and km_m > 0.0:
        fac = 1.0 + float(s_m / km_m)
        return float(fac) if np.isfinite(fac) and fac > 1.0 else float("nan")

    # Fallback when units are missing/unknown but values are likely commensurate.
    if (_normalize_unit(s_unit) == "" and _normalize_unit(km_unit) == "") or (_normalize_unit(s_unit) == _normalize_unit(km_unit)):
        fac = 1.0 + float(s_val / km_val)
        return float(fac) if np.isfinite(fac) and fac > 1.0 else float("nan")

    return float("nan")


def _cheng_prusoff_term_kj_from_measurement(meas: dict, temperature_k: float) -> float:
    fac = _cheng_prusoff_factor_from_measurement(meas)
    if not np.isfinite(fac) or fac <= 1.0:
        return float("nan")
    r_kj = 0.008314462618
    term = float(r_kj * float(temperature_k) * np.log(fac))
    return term if np.isfinite(term) and term > 0.0 else float("nan")


def build_cp_term_map_from_ligands_yml(data_root: Path, temperature_k: float) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    if yaml is None:
        return out
    for target_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        yml = target_dir / "ligands.yml"
        if not yml.exists():
            continue
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        tgt = str(target_dir.name)
        for lig, rec in data.items():
            lig_name = str(lig)
            meas = rec.get("measurement") if isinstance(rec, dict) else None
            term = _cheng_prusoff_term_kj_from_measurement(meas if isinstance(meas, dict) else {}, temperature_k=temperature_k)
            if np.isfinite(term):
                out[(tgt, lig_name)] = float(term)
    return out


def _estimate_ic50_shift(
    y: np.ndarray,
    endpoint: np.ndarray,
    targets: np.ndarray,
    min_rows: int,
) -> Tuple[Dict[str, float], float, Dict[str, int]]:
    yy = np.asarray(y, dtype=float)
    ep = np.asarray(endpoint).astype(str)
    tg = np.asarray(targets).astype(str)

    valid = np.isfinite(yy)
    yy = yy[valid]
    ep = ep[valid]
    tg = tg[valid]

    target_delta: Dict[str, float] = {}
    target_weight: Dict[str, int] = {}

    for t in sorted(set(tg.tolist())):
        mt = tg == str(t)
        yi = yy[mt & (ep == "ic50_like")]
        yk = yy[mt & (ep == "ki_like")]
        if len(yi) < int(min_rows) or len(yk) < int(min_rows):
            continue
        d = float(np.nanmedian(yk) - np.nanmedian(yi))
        if np.isfinite(d):
            target_delta[str(t)] = d
            target_weight[str(t)] = int(min(len(yi), len(yk)))

    global_delta = float("nan")
    if target_delta:
        vals = np.asarray([target_delta[k] for k in target_delta], dtype=float)
        wts = np.asarray([max(1, target_weight.get(k, 1)) for k in target_delta], dtype=float)
        global_delta = float(np.sum(vals * wts) / np.sum(wts))

    if not np.isfinite(global_delta):
        yi = yy[ep == "ic50_like"]
        yk = yy[ep == "ki_like"]
        if len(yi) >= int(min_rows) and len(yk) >= int(min_rows):
            global_delta = float(np.nanmedian(yk) - np.nanmedian(yi))

    return target_delta, global_delta, target_weight


def _apply_ic50_normalization(
    df: pd.DataFrame,
    data_root: Path,
    temperature_k: float,
    mode: str,
    shift_min_rows: int,
) -> dict:
    out_info = {
        "ic50_normalization_mode": str(mode),
        "ic50_cp_applied": 0,
        "ic50_shift_target_applied": 0,
        "ic50_shift_global_applied": 0,
        "ic50_shift_global_kj": float("nan"),
        "ic50_shift_targets": {},
    }

    mode_norm = str(mode or "none").strip().lower()
    if mode_norm not in {"none", "cp_only", "shift_only", "cp_or_shift"}:
        mode_norm = "none"
    out_info["ic50_normalization_mode"] = mode_norm

    df["exp_dg_kj_model"] = pd.to_numeric(df.get("exp_dg_kj_filled", np.nan), errors="coerce")
    df["ic50_norm_method"] = "none"
    df["ic50_cp_term_kj"] = np.nan
    df["ic50_shift_kj"] = np.nan

    ic_mask = (df.get("endpoint_group", "").astype(str) == "ic50_like") & np.isfinite(df["exp_dg_kj_model"])
    if mode_norm == "none" or not np.any(ic_mask):
        return out_info

    cp_applied = np.zeros(len(df), dtype=bool)

    if mode_norm in {"cp_only", "cp_or_shift"}:
        cp_map = build_cp_term_map_from_ligands_yml(data_root=data_root, temperature_k=temperature_k)
        if cp_map:
            for i, row in df.loc[ic_mask].iterrows():
                key = (str(row["target"]), str(row["ligand"]))
                term = cp_map.get(key)
                if term is None or not np.isfinite(float(term)) or float(term) <= 0.0:
                    continue
                df.at[i, "exp_dg_kj_model"] = float(df.at[i, "exp_dg_kj_model"]) - float(term)
                df.at[i, "ic50_cp_term_kj"] = float(term)
                df.at[i, "ic50_norm_method"] = "cp"
                cp_applied[i] = True
            out_info["ic50_cp_applied"] = int(np.sum(cp_applied))

    do_shift = mode_norm in {"shift_only", "cp_or_shift"}
    if do_shift:
        if mode_norm == "shift_only":
            shift_mask = ic_mask.copy()
        else:
            shift_mask = ic_mask & (~cp_applied)

        target_delta, global_delta, _target_weight = _estimate_ic50_shift(
            y=df["exp_dg_kj_model"].to_numpy(float),
            endpoint=df.get("endpoint_group", "").astype(str).to_numpy(),
            targets=df.get("target", "").astype(str).to_numpy(),
            min_rows=int(shift_min_rows),
        )
        out_info["ic50_shift_targets"] = target_delta
        out_info["ic50_shift_global_kj"] = float(global_delta) if np.isfinite(global_delta) else float("nan")

        for i, row in df.loc[shift_mask].iterrows():
            t = str(row["target"])
            if t in target_delta and np.isfinite(float(target_delta[t])):
                d = float(target_delta[t])
                df.at[i, "exp_dg_kj_model"] = float(df.at[i, "exp_dg_kj_model"]) + d
                df.at[i, "ic50_shift_kj"] = d
                df.at[i, "ic50_norm_method"] = "shift_target"
                out_info["ic50_shift_target_applied"] += 1
            elif np.isfinite(global_delta):
                d = float(global_delta)
                df.at[i, "exp_dg_kj_model"] = float(df.at[i, "exp_dg_kj_model"]) + d
                df.at[i, "ic50_shift_kj"] = d
                df.at[i, "ic50_norm_method"] = "shift_global"
                out_info["ic50_shift_global_applied"] += 1

    return out_info


def _endpoint_group_from_source(src: str) -> str:
    s = str(src or "").strip().lower()
    if not s:
        return "unknown"
    if (
        ".pki" in s
        or ".ki" in s
        or "from_ki" in s
        or s.endswith("_ki")
        or s == "ki"
        or s == "pki"
    ):
        return "ki_like"
    if (
        ".pkd" in s
        or ".kd" in s
        or "from_kd" in s
        or s.endswith("_kd")
        or s == "kd"
        or s == "pkd"
    ):
        return "kd_like"
    if (
        ".pic50" in s
        or ".ic50" in s
        or "from_ic50" in s
        or s.endswith("_ic50")
        or s == "ic50"
        or s == "pic50"
    ):
        return "ic50_like"
    if ".dg" in s or "delta_g" in s or "metadata_dg" in s or s.endswith("_dg") or s == "dg":
        return "dg_like"
    return "unknown"


def _fit_endpoint_offsets(
    endpoints: np.ndarray,
    residuals: np.ndarray,
    sample_weight: np.ndarray,
    min_rows: int,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    ep = np.asarray(endpoints).astype(str)
    rr = np.asarray(residuals, dtype=float)
    ww = np.asarray(sample_weight, dtype=float)
    ww = np.where(np.isfinite(ww), np.clip(ww, 1e-12, None), 1e-12)

    offsets: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for g in sorted(set(ep.tolist())):
        m = ep == str(g)
        n = int(np.sum(m))
        if n < int(min_rows):
            continue
        r = rr[m]
        w = ww[m]
        if n == 0:
            continue
        off = float(np.sum(w * r) / np.sum(w))
        if np.isfinite(off):
            offsets[str(g)] = off
            counts[str(g)] = n
    return offsets, counts


def _apply_endpoint_offsets(pred: np.ndarray, endpoints: np.ndarray, offsets: Dict[str, float]) -> np.ndarray:
    p = np.asarray(pred, dtype=float).copy()
    ep = np.asarray(endpoints).astype(str)
    for i, g in enumerate(ep.tolist()):
        p[i] = p[i] + float(offsets.get(str(g), 0.0))
    return p


def _fit_target_linear_corrections(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    sample_weight: np.ndarray,
    shrink_k: float,
) -> Dict[str, Dict[str, float]]:
    yy = np.asarray(y_true, dtype=float)
    pp = np.asarray(y_pred, dtype=float)
    gg = np.asarray(groups).astype(str)
    ww = np.asarray(sample_weight, dtype=float)
    ww = np.where(np.isfinite(ww), np.clip(ww, 1e-12, None), 1e-12)

    out: Dict[str, Dict[str, float]] = {}
    for g in sorted(set(gg.tolist())):
        m = gg == str(g)
        n = int(np.sum(m))
        if n < 3:
            continue
        x = pp[m]
        y = yy[m]
        w = ww[m]
        X = np.column_stack([x, np.ones(n, dtype=float)])
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
        try:
            coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
            a_raw = float(coef[0])
            b_raw = float(coef[1])
        except Exception:
            a_raw, b_raw = 1.0, 0.0

        shrink = float(n / (n + max(0.0, float(shrink_k))))
        a = 1.0 + shrink * (a_raw - 1.0)
        b = shrink * b_raw
        if not np.isfinite(a):
            a = 1.0
        if not np.isfinite(b):
            b = 0.0
        out[str(g)] = {
            "n": float(n),
            "a_raw": float(a_raw),
            "b_raw": float(b_raw),
            "a": float(a),
            "b": float(b),
        }
    return out


def _apply_target_linear_corrections(
    pred: np.ndarray,
    groups: np.ndarray,
    corr: Dict[str, Dict[str, float]],
) -> np.ndarray:
    p = np.asarray(pred, dtype=float).copy()
    gg = np.asarray(groups).astype(str)
    for i, g in enumerate(gg.tolist()):
        c = corr.get(str(g))
        if not c:
            continue
        p[i] = float(c.get("a", 1.0)) * p[i] + float(c.get("b", 0.0))
    return p


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
    df: pd.DataFrame,
    *,
    docking_csv: Path,
    target_col: str,
    ligand_col: str,
    affinity_col: str,
    affinity_unit: str,
) -> Tuple[pd.DataFrame, dict]:
    dock = pd.read_csv(docking_csv)
    for req in (target_col, ligand_col, affinity_col):
        if req not in dock.columns:
            raise RuntimeError(f"Missing docking column: {req}")
    scale = 4.184 if str(affinity_unit).strip().lower() == "kcal" else 1.0
    out_col = f"dock_{_sanitize_name(affinity_col)}_kj"
    sub = pd.DataFrame(
        {
            "target": dock[target_col].astype(str),
            "ligand": dock[ligand_col].astype(str),
            out_col: pd.to_numeric(dock[affinity_col], errors="coerce") * float(scale),
        }
    )
    merged = df.merge(sub, on=["target", "ligand"], how="left")
    info = {
        "docking_csv": str(docking_csv),
        "docking_col": str(out_col),
        "docking_rows_total": int(len(sub)),
        "docking_rows_merged": int(np.sum(np.isfinite(pd.to_numeric(merged[out_col], errors="coerce").to_numpy(float)))),
    }
    return merged, info


def fit_loto_linear_stack_from_predictions(
    pred_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str = "target",
    exp_col: str = "exp_dg_kj",
    sample_weight_col: str = "sample_weight",
) -> np.ndarray:
    cols = [str(c) for c in list(feature_cols or []) if str(c) in pred_df.columns]
    out = np.full(len(pred_df), np.nan, dtype=float)
    if not cols:
        return out
    X = np.column_stack([pd.to_numeric(pred_df[c], errors="coerce").to_numpy(float) for c in cols])
    y = pd.to_numeric(pred_df[exp_col], errors="coerce").to_numpy(float)
    w = pd.to_numeric(pred_df.get(sample_weight_col, 1.0), errors="coerce").to_numpy(float)
    groups = pred_df[target_col].astype(str).to_numpy()
    valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if np.any(np.isfinite(w)):
        valid = valid & np.isfinite(w)
    min_train = int(len(cols) + 1)
    for g in sorted(set(groups.tolist())):
        te = (groups == str(g)) & valid
        tr = (groups != str(g)) & valid
        if int(np.sum(te)) == 0 or int(np.sum(tr)) < min_train:
            continue
        coef = fit_linear(X[tr], y[tr], sample_weight=w[tr])
        out[te] = pred_linear(X[te], coef)
    return np.asarray(out, dtype=float)


def append_groupcv_stack_predictions(
    fold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    predictor_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = [str(c) for c in list(feature_cols or []) if str(c) in pred_df.columns]
    if len(cols) != len(list(feature_cols or [])):
        return fold_df, pred_df
    pred = fit_loto_linear_stack_from_predictions(
        pred_df,
        feature_cols=cols,
        target_col="target",
        exp_col="exp_dg_kj",
        sample_weight_col="sample_weight",
    )
    if not np.any(np.isfinite(pred)):
        return fold_df, pred_df

    out_pred = pred_df.copy()
    out_pred[predictor_name] = pred

    rows = []
    for target, g in out_pred.groupby("target", sort=True):
        y = pd.to_numeric(g["exp_dg_kj"], errors="coerce").to_numpy(float)
        p = pd.to_numeric(g[predictor_name], errors="coerce").to_numpy(float)
        w = pd.to_numeric(g.get("sample_weight", 1.0), errors="coerce").to_numpy(float)
        m = np.isfinite(y) & np.isfinite(p)
        if np.any(np.isfinite(w)):
            m = m & np.isfinite(w)
        if int(np.sum(m)) == 0:
            continue
        rows.append(
            FoldRow(
                model=str(predictor_name),
                fold_target=str(target),
                n_test=int(np.sum(m)),
                rmse=rmse(y[m], p[m], w=w[m]),
                mae=mae(y[m], p[m], w=w[m]),
                pearson=pearson(y[m], p[m]),
                spearman=spearman(y[m], p[m]),
                alpha=float("nan"),
            ).__dict__
        )
    out_fold = fold_df.copy()
    if rows:
        out_fold = pd.concat([out_fold, pd.DataFrame(rows)], ignore_index=True)
    return out_fold, out_pred


def build_quality_weights(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)

    ci_low = pd.to_numeric(df.get("dg_expavg_ci_low_kj", np.nan), errors="coerce").to_numpy(float)
    ci_high = pd.to_numeric(df.get("dg_expavg_ci_high_kj", np.nan), errors="coerce").to_numpy(float)
    ci_w = np.abs(ci_high - ci_low)
    med_ciw = float(np.nanmedian(ci_w[np.isfinite(ci_w)])) if np.any(np.isfinite(ci_w)) else 20.0
    if not np.isfinite(med_ciw) or med_ciw <= 0.0:
        med_ciw = 20.0
    ci_w = np.where(np.isfinite(ci_w), ci_w, med_ciw)
    q_ci = 1.0 / (1.0 + (ci_w / 10.0) ** 2)

    n_bound = pd.to_numeric(df.get("n_bound", np.nan), errors="coerce").to_numpy(float)
    n_unbound = pd.to_numeric(df.get("n_unbound", np.nan), errors="coerce").to_numpy(float)
    both = np.isfinite(n_bound) & np.isfinite(n_unbound)
    sb = np.full(n, 0.5, dtype=float)
    if np.any(both):
        mn = np.minimum(n_bound[both], n_unbound[both])
        mx = np.maximum(n_bound[both], n_unbound[both])
        sb[both] = (mn + 1.0) / (mx + 1.0)
    q_state = np.sqrt(np.clip(sb, 0.05, 1.0))

    frames_valid = pd.to_numeric(df.get("frames_valid", np.nan), errors="coerce").to_numpy(float)
    med_f = float(np.nanmedian(frames_valid[np.isfinite(frames_valid)])) if np.any(np.isfinite(frames_valid)) else 500.0
    if not np.isfinite(med_f) or med_f <= 0.0:
        med_f = 500.0
    frames_valid = np.where(np.isfinite(frames_valid), frames_valid, med_f)
    q_frame = np.clip(frames_valid / 1000.0, 0.25, 1.0)

    trans = np.where(both, np.maximum(0.0, n_bound) + np.maximum(0.0, n_unbound), np.nan)
    med_t = float(np.nanmedian(trans[np.isfinite(trans)])) if np.any(np.isfinite(trans)) else 50.0
    if not np.isfinite(med_t) or med_t <= 0.0:
        med_t = 50.0
    trans = np.where(np.isfinite(trans), trans, med_t)
    q_trans = np.clip(np.log1p(trans) / np.log1p(200.0), 0.25, 1.0)

    q = q_ci * q_state * q_frame * q_trans
    q = np.clip(q, 0.05, 1.0)
    return q.astype(float)


def prep(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med[None, :])

    # Guard against extreme magnitudes that can destabilize XtX in ridge fitting.
    Xi = np.clip(Xi, -1.0e6, 1.0e6)
    try:
        lo = np.nanpercentile(Xi, 1.0, axis=0)
        hi = np.nanpercentile(Xi, 99.0, axis=0)
        good = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        lo = np.where(good, lo, -1.0e6)
        hi = np.where(good, hi, 1.0e6)
        Xi = np.minimum(np.maximum(Xi, lo[None, :]), hi[None, :])
    except Exception:
        pass

    mu = Xi.mean(axis=0)
    sd = Xi.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return med, mu, sd


def transform(X: np.ndarray, med: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Xi = np.where(np.isfinite(X), X, med[None, :])
    Z = (Xi - mu[None, :]) / sd[None, :]
    Z = np.clip(Z, -1.0e4, 1.0e4)
    return Z


def fit_linear(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
    X2 = np.column_stack([X, np.ones(len(X), dtype=float)])
    yy = y.astype(float)
    if sample_weight is None:
        coef, _, _, _ = np.linalg.lstsq(X2, yy, rcond=None)
        return coef
    w = np.asarray(sample_weight, dtype=float)
    w = np.where(np.isfinite(w), np.clip(w, 1e-12, None), 1e-12)
    sw = np.sqrt(w)
    Xw = X2 * sw[:, None]
    yw = yy * sw
    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return coef


def pred_linear(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    X2 = np.column_stack([X, np.ones(len(X), dtype=float)])
    return X2 @ coef


def fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray | float]:
    med, mu, sd = prep(X)
    Xs = transform(X, med, mu, sd)
    ym = float(np.mean(y))
    yc = y - ym

    if sample_weight is None:
        Xw = Xs
        yw = yc
    else:
        w = np.asarray(sample_weight, dtype=float)
        w = np.where(np.isfinite(w), np.clip(w, 1e-12, None), 1e-12)
        sw = np.sqrt(w)
        Xw = Xs * sw[:, None]
        yw = yc * sw

    A = Xw.T @ Xw + float(alpha) * np.eye(Xs.shape[1], dtype=float)
    b = Xw.T @ yw
    try:
        wcoef = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        wcoef = np.linalg.lstsq(A, b, rcond=None)[0]
    return {
        "med": med,
        "mu": mu,
        "sd": sd,
        "w": wcoef,
        "y_mean": ym,
        "alpha": float(alpha),
    }


def pred_ridge(X: np.ndarray, model: Dict[str, np.ndarray | float]) -> np.ndarray:
    med = np.asarray(model["med"], dtype=float)
    mu = np.asarray(model["mu"], dtype=float)
    sd = np.asarray(model["sd"], dtype=float)
    wcoef = np.asarray(model["w"], dtype=float)
    ym = float(model["y_mean"])
    Xs = transform(X, med, mu, sd)
    return Xs @ wcoef + ym


def _target_onehot(groups: np.ndarray, target_order: Sequence[str]) -> np.ndarray:
    """One-hot encode target labels with fixed column order."""
    g = np.asarray(groups).astype(str)
    order = list(target_order)
    idx = {t: i for i, t in enumerate(order)}
    T = np.zeros((len(g), len(order)), dtype=float)
    for i, tg in enumerate(g):
        j = idx.get(str(tg))
        if j is not None:
            T[i, j] = 1.0
    return T


def _fit_targetaware_full(
    X_compact: np.ndarray,
    groups: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    alpha: float,
) -> Dict[str, object]:
    """Fit target-aware ridge on full data (compact features + target one-hot)."""
    target_order = sorted(set(np.asarray(groups).astype(str).tolist()))
    T = _target_onehot(np.asarray(groups).astype(str), target_order)
    Xta = np.hstack([X_compact, T])
    model = fit_ridge(Xta, y, alpha=float(alpha), sample_weight=sample_weight)
    return {
        "target_order": target_order,
        "model": model,
    }


def _predict_targetaware(
    X_compact: np.ndarray,
    groups: np.ndarray,
    targetaware_pack: Dict[str, object],
) -> np.ndarray:
    target_order = list(targetaware_pack["target_order"])
    model = targetaware_pack["model"]
    T = _target_onehot(np.asarray(groups).astype(str), target_order)
    Xta = np.hstack([X_compact, T])
    return pred_ridge(Xta, model)  # type: ignore[arg-type]


def choose_alpha(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    uniq = sorted(set(groups.tolist()))
    if len(uniq) <= 1:
        return float(alphas[0])
    best_a = float(alphas[0])
    best_score = float("inf")
    for a in alphas:
        scores: List[float] = []
        for g in uniq:
            te = groups == g
            tr = ~te
            if np.sum(te) < 2 or np.sum(tr) < 3:
                continue
            w_tr = None if sample_weight is None else sample_weight[tr]
            w_te = None if sample_weight is None else sample_weight[te]
            m = fit_ridge(X[tr], y[tr], alpha=float(a), sample_weight=w_tr)
            p = pred_ridge(X[te], m)
            scores.append(rmse(y[te], p, w=w_te))
        if scores:
            s = float(np.mean(scores))
            if s < best_score:
                best_score = s
                best_a = float(a)
    return best_a


def _center_by_group(X: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    out = np.zeros_like(X, dtype=float)
    means: Dict[str, np.ndarray] = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if np.sum(m) == 0:
            continue
        mu = np.nanmean(X[m], axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        means[str(g)] = mu
        out[m] = X[m] - mu[None, :]
    return out, means


def _center_y_by_group(y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    out = np.zeros_like(y, dtype=float)
    means: Dict[str, float] = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        if np.sum(m) == 0:
            continue
        mu = float(np.mean(y[m]))
        means[str(g)] = mu
        out[m] = y[m] - mu
    return out, means


def _target_stratified_split(groups: np.ndarray, test_frac: float, rng: np.random.Generator) -> np.ndarray:
    groups = np.asarray(groups)
    n = len(groups)
    te = np.zeros(n, dtype=bool)
    for g in sorted(set(groups.tolist())):
        idx = np.where(groups == g)[0]
        m = len(idx)
        if m <= 1:
            continue
        n_te = max(1, int(round(float(test_frac) * m)))
        n_te = min(n_te, m - 1)
        pick = rng.choice(idx, size=n_te, replace=False)
        te[pick] = True
    if int(np.sum(te)) == 0 and n > 1:
        te[rng.integers(0, n)] = True
    return te


def _tune_alpha_within_target(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
    repeats: int,
    test_frac: float,
    seed: int,
) -> Tuple[float, Dict[float, float]]:
    rng = np.random.default_rng(seed)
    scores: Dict[float, List[float]] = {float(a): [] for a in alphas}
    nrep = max(10, int(repeats))
    for _ in range(nrep):
        te = _target_stratified_split(groups, test_frac=test_frac, rng=rng)
        tr = ~te
        if np.sum(te) < 2 or np.sum(tr) < 5:
            continue
        for a in alphas:
            m = fit_ridge(X[tr], y[tr], alpha=float(a), sample_weight=w[tr])
            p = pred_ridge(X[te], m)
            scores[float(a)].append(rmse(y[te], p, w=w[te]))
    mean_scores = {a: (float(np.mean(v)) if v else float("inf")) for a, v in scores.items()}
    best_a = sorted(mean_scores.items(), key=lambda kv: kv[1])[0][0]
    return float(best_a), mean_scores


def load_data(
    csv_path: str,
    data_root: Path,
    fill_exp_from_yml: bool,
    temperature_k: float,
    weight_mode: str,
    endpoint_filter: str,
    ic50_normalization: str,
    ic50_shift_min_rows: int,
    docking_csv: Optional[Path] = None,
    docking_target_col: str = "target",
    docking_ligand_col: str = "ligand",
    docking_affinity_col: str = "affinity",
    docking_unit: str = "kcal",
) -> Tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)
    req = ["target", "ligand", "exp_dg_kj", "dg_expavg_kj", "dg_pop_or_jeff_kj"]
    for c in req:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    for c in [
        "exp_dg_kj",
        "dg_expavg_kj",
        "dg_pop_or_jeff_kj",
        "dg_cumulant_kj",
        "energy_mean_kj",
        "energy_std_kj",
        "coul_mean_kj",
        "lj_mean_kj",
        "contact_frac_mean",
        "min_dist_mean_a",
        "lig_com_dist_mean_a",
        "p_bound",
        "p_unbound",
        "n_bound",
        "n_unbound",
        "frames_valid",
        "dg_expavg_ci_low_kj",
        "dg_expavg_ci_high_kj",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in df.columns:
        if str(c).startswith("pocket_") and str(c) != "pocket_desc_status":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["target"] = df["target"].astype(str)
    df["ligand"] = df["ligand"].astype(str)
    df["exp_dg_kj_orig"] = pd.to_numeric(df["exp_dg_kj"], errors="coerce")
    df["exp_dg_kj_filled"] = df["exp_dg_kj_orig"].copy()
    df["exp_source_filled"] = df.get("exp_source", "").astype(str) if "exp_source" in df.columns else ""

    n_before = int(np.sum(np.isfinite(df["exp_dg_kj_orig"])))
    n_filled = 0

    if fill_exp_from_yml:
        exp_map = build_exp_map_from_ligands_yml(data_root=data_root, temperature_k=temperature_k)
        for i, row in df.iterrows():
            key = (str(row["target"]), str(row["ligand"]))
            if key in exp_map:
                _, src = exp_map[key]
                if not str(df.at[i, "exp_source_filled"]).strip():
                    df.at[i, "exp_source_filled"] = str(src)
            if np.isfinite(float(row["exp_dg_kj_filled"])):
                continue
            if key not in exp_map:
                continue
            dg, src = exp_map[key]
            if np.isfinite(dg):
                df.at[i, "exp_dg_kj_filled"] = float(dg)
                df.at[i, "exp_source_filled"] = str(src)
                n_filled += 1

    df["endpoint_group"] = df["exp_source_filled"].astype(str).map(_endpoint_group_from_source)

    ic50_info = _apply_ic50_normalization(
        df=df,
        data_root=data_root,
        temperature_k=float(temperature_k),
        mode=str(ic50_normalization),
        shift_min_rows=int(ic50_shift_min_rows),
    )

    w = df[
        df["target"].notna()
        & df["ligand"].notna()
        & np.isfinite(df["exp_dg_kj_model"])
        & np.isfinite(df["dg_expavg_kj"])
        & np.isfinite(df["dg_pop_or_jeff_kj"])
    ].copy()

    endpoint_filter_norm = str(endpoint_filter or "all").strip().lower()
    if endpoint_filter_norm and endpoint_filter_norm != "all":
        w = w[w["endpoint_group"].astype(str) == endpoint_filter_norm].copy()
        if w.empty:
            raise RuntimeError(f"No rows left after endpoint_filter={endpoint_filter_norm}")

    if "contact_frac_mean" in w.columns and "min_dist_mean_a" in w.columns:
        w["contact_over_dist"] = w["contact_frac_mean"] / w["min_dist_mean_a"].replace(0, np.nan)

    docking_info = {
        "docking_csv": "",
        "docking_col": "",
        "docking_rows_total": 0,
        "docking_rows_merged": 0,
        "docking_rows_model": 0,
    }
    if docking_csv is not None and str(docking_csv).strip():
        docking_path = Path(docking_csv)
        if not docking_path.exists():
            raise FileNotFoundError(f"Docking CSV not found: {docking_path}")
        w, docking_info = merge_docking_table(
            w,
            docking_csv=docking_path,
            target_col=str(docking_target_col),
            ligand_col=str(docking_ligand_col),
            affinity_col=str(docking_affinity_col),
            affinity_unit=str(docking_unit),
        )
        dcol = str(docking_info.get("docking_col", "") or "")
        if dcol and dcol in w.columns:
            docking_info["docking_rows_model"] = int(np.sum(np.isfinite(pd.to_numeric(w[dcol], errors="coerce").to_numpy(float))))

    if weight_mode == "quality":
        w["sample_weight"] = build_quality_weights(w)
    else:
        w["sample_weight"] = 1.0

    endpoint_counts = (
        w["endpoint_group"].astype(str).value_counts(dropna=False).to_dict()
        if "endpoint_group" in w.columns
        else {}
    )

    info = {
        "n_rows_total": int(len(df)),
        "n_exp_before": int(n_before),
        "n_exp_after": int(np.sum(np.isfinite(df["exp_dg_kj_filled"]))),
        "n_exp_model_after": int(np.sum(np.isfinite(df["exp_dg_kj_model"]))),
        "n_exp_filled": int(n_filled),
        "n_rows_model": int(len(w)),
        "n_targets_model": int(len(sorted(w["target"].astype(str).unique().tolist()))),
        "fill_exp_from_yml": bool(fill_exp_from_yml),
        "weight_mode": str(weight_mode),
        "endpoint_filter": endpoint_filter_norm,
        "endpoint_counts": endpoint_counts,
        "pocket_descriptor_cols": [c for c in w.columns if str(c).startswith("pocket_") and str(c) != "pocket_desc_status"],
        "n_pocket_descriptor_cols": int(len([c for c in w.columns if str(c).startswith("pocket_") and str(c) != "pocket_desc_status"])),
        "ic50_normalization_mode": ic50_info.get("ic50_normalization_mode", "none"),
        "ic50_cp_applied": int(ic50_info.get("ic50_cp_applied", 0)),
        "ic50_shift_target_applied": int(ic50_info.get("ic50_shift_target_applied", 0)),
        "ic50_shift_global_applied": int(ic50_info.get("ic50_shift_global_applied", 0)),
        "ic50_shift_global_kj": float(ic50_info.get("ic50_shift_global_kj", float("nan"))),
        "ic50_shift_targets_n": int(len(ic50_info.get("ic50_shift_targets", {}))),
        "docking_csv": str(docking_info.get("docking_csv", "")),
        "docking_col": str(docking_info.get("docking_col", "")),
        "docking_rows_total": int(docking_info.get("docking_rows_total", 0)),
        "docking_rows_merged": int(docking_info.get("docking_rows_merged", 0)),
        "docking_rows_model": int(docking_info.get("docking_rows_model", 0)),
    }
    return w, info


def evaluate_loto(
    df: pd.DataFrame,
    alphas: Sequence[float],
    endpoint_aware: bool,
    endpoint_min_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    targets = sorted(df["target"].astype(str).unique().tolist())

    compact_cols = [
        "dg_expavg_kj",
        "dg_cumulant_kj",
        "energy_mean_kj",
        "coul_mean_kj",
        "contact_frac_mean",
        "pocket_volume_bbox_a3",
        "pocket_volume_sphere95_a3",
        "pocket_atom_density_per_a3",
        "pocket_polar_atom_frac",
        "pocket_n_residues",
        "pocket_n_atoms",
        "pocket_frac_hydrophobic",
        "pocket_frac_polar",
        "pocket_frac_positive",
        "pocket_frac_negative",
        "pocket_frac_aromatic",
        "pocket_charge_est",
        "pocket_charge_density",
    ]
    compact_cols = [c for c in compact_cols if c in df.columns]
    if len(compact_cols) < 3:
        raise RuntimeError("Not enough compact features available")

    all_cols = [
        "dg_expavg_kj",
        "dg_pop_or_jeff_kj",
        "dg_cumulant_kj",
        "energy_mean_kj",
        "energy_std_kj",
        "coul_mean_kj",
        "lj_mean_kj",
        "contact_frac_mean",
        "min_dist_mean_a",
        "lig_com_dist_mean_a",
        "p_bound",
        "p_unbound",
        "ligand_atoms_used",
        "n_bound",
        "n_unbound",
        "contact_over_dist",
        "pocket_volume_bbox_a3",
        "pocket_volume_sphere95_a3",
        "pocket_atom_density_per_a3",
        "pocket_polar_atom_frac",
        "pocket_n_residues",
        "pocket_n_atoms",
        "pocket_frac_hydrophobic",
        "pocket_frac_polar",
        "pocket_frac_positive",
        "pocket_frac_negative",
        "pocket_frac_aromatic",
        "pocket_charge_est",
        "pocket_charge_density",
    ]
    all_cols = [c for c in all_cols if c in df.columns]

    fold_rows: List[FoldRow] = []
    pred_rows: List[dict] = []

    y_all = df["exp_dg_kj_model"].to_numpy(float)
    xea_all = df["dg_expavg_kj"].to_numpy(float)
    xpo_all = df["dg_pop_or_jeff_kj"].to_numpy(float)
    w_all = pd.to_numeric(df.get("sample_weight", 1.0), errors="coerce").to_numpy(float)
    endpoint_all = df.get("endpoint_group", "unknown").astype(str).to_numpy()

    for tgt in targets:
        te = df["target"].astype(str) == str(tgt)
        tr = ~te
        if np.sum(te) < 2 or np.sum(tr) < 5:
            continue

        y_tr = y_all[tr]
        y_te = y_all[te]
        xea_tr = xea_all[tr]
        xea_te = xea_all[te]
        xpo_tr = xpo_all[tr]
        xpo_te = xpo_all[te]
        w_tr = w_all[tr]
        w_te = w_all[te]
        ep_tr = endpoint_all[tr]
        ep_te = endpoint_all[te]

        raw_expavg = xea_te.copy()
        raw_pop = xpo_te.copy()

        be = fit_linear(xea_tr[:, None], y_tr, sample_weight=w_tr)
        bc = fit_linear(np.column_stack([xea_tr, xpo_tr]), y_tr, sample_weight=w_tr)
        cal_expavg = pred_linear(xea_te[:, None], be)
        cal_combo = pred_linear(np.column_stack([xea_te, xpo_te]), bc)

        Xc_tr = df.loc[tr, compact_cols].to_numpy(float)
        Xc_te = df.loc[te, compact_cols].to_numpy(float)
        grp_tr = df.loc[tr, "target"].astype(str).to_numpy()
        ac = choose_alpha(Xc_tr, y_tr, grp_tr, alphas, sample_weight=w_tr)
        mc = fit_ridge(Xc_tr, y_tr, alpha=ac, sample_weight=w_tr)
        ridge_compact = pred_ridge(Xc_te, mc)

        # Endpoint-aware residual correction helps cross-target transfer when
        # assays are heterogeneous (Ki/Kd/IC50/pIC50 mixtures).
        cal_combo_ep = cal_combo.copy()
        ridge_compact_ep = ridge_compact.copy()
        if endpoint_aware:
            cal_combo_tr = pred_linear(np.column_stack([xea_tr, xpo_tr]), bc)
            rc_combo = y_tr - cal_combo_tr
            off_combo, _ = _fit_endpoint_offsets(ep_tr, rc_combo, w_tr, min_rows=int(endpoint_min_rows))
            cal_combo_ep = _apply_endpoint_offsets(cal_combo, ep_te, off_combo)

            ridge_compact_tr = pred_ridge(Xc_tr, mc)
            rc_ridge = y_tr - ridge_compact_tr
            off_ridge, _ = _fit_endpoint_offsets(ep_tr, rc_ridge, w_tr, min_rows=int(endpoint_min_rows))
            ridge_compact_ep = _apply_endpoint_offsets(ridge_compact, ep_te, off_ridge)

        Xa_tr = df.loc[tr, all_cols].to_numpy(float)
        Xa_te = df.loc[te, all_cols].to_numpy(float)
        aa = choose_alpha(Xa_tr, y_tr, grp_tr, alphas, sample_weight=w_tr)
        ma = fit_ridge(Xa_tr, y_tr, alpha=aa, sample_weight=w_tr)
        ridge_all = pred_ridge(Xa_te, ma)

        Xr_tr_c, _ = _center_by_group(Xc_tr, grp_tr)
        yr_tr_c, _ = _center_y_by_group(y_tr, grp_tr)
        mu_te = np.nanmean(Xc_te, axis=0)
        mu_te = np.where(np.isfinite(mu_te), mu_te, 0.0)
        Xr_te_c = Xc_te - mu_te[None, :]
        ar = choose_alpha(Xr_tr_c, yr_tr_c, grp_tr, alphas, sample_weight=w_tr)
        mr = fit_ridge(Xr_tr_c, yr_tr_c, alpha=ar, sample_weight=w_tr)
        rel_center = pred_ridge(Xr_te_c, mr)
        rel_center = rel_center - float(np.mean(rel_center))
        anchor = float(np.mean(cal_combo))
        rel_hybrid = anchor + rel_center

        pred_map = {
            "raw_expavg": raw_expavg,
            "raw_pop": raw_pop,
            "cal_expavg": cal_expavg,
            "cal_combo": cal_combo,
            "cal_combo_ep": cal_combo_ep,
            "ridge_compact": ridge_compact,
            "ridge_compact_ep": ridge_compact_ep,
            "ridge_all": ridge_all,
            "rel_hybrid": rel_hybrid,
        }

        ligs = df.loc[te, "ligand"].astype(str).tolist()
        for mname, pred in pred_map.items():
            a_used = np.nan
            if mname in {"ridge_compact", "ridge_compact_ep"}:
                a_used = float(ac)
            elif mname == "ridge_all":
                a_used = float(aa)
            elif mname == "rel_hybrid":
                a_used = float(ar)
            fold_rows.append(
                FoldRow(
                    model=mname,
                    fold_target=str(tgt),
                    n_test=int(len(y_te)),
                    rmse=rmse(y_te, pred, w=w_te),
                    mae=mae(y_te, pred, w=w_te),
                    pearson=pearson(y_te, pred),
                    spearman=spearman(y_te, pred),
                    alpha=a_used,
                )
            )

        for i in range(len(y_te)):
            pred_rows.append(
                {
                    "target": str(tgt),
                    "ligand": ligs[i],
                    "exp_dg_kj": float(y_te[i]),
                    "sample_weight": float(w_te[i]) if i < len(w_te) else 1.0,
                    "endpoint_group": str(ep_te[i]),
                    "raw_expavg": float(raw_expavg[i]),
                    "raw_pop": float(raw_pop[i]),
                    "cal_expavg": float(cal_expavg[i]),
                    "cal_combo": float(cal_combo[i]),
                    "cal_combo_ep": float(cal_combo_ep[i]),
                    "ridge_compact": float(ridge_compact[i]),
                    "ridge_compact_ep": float(ridge_compact_ep[i]),
                    "ridge_all": float(ridge_all[i]),
                    "rel_hybrid": float(rel_hybrid[i]),
                }
            )

    return pd.DataFrame([r.__dict__ for r in fold_rows]), pd.DataFrame(pred_rows)


def evaluate_within_target_cv(
    df: pd.DataFrame,
    alphas: Sequence[float],
    repeats: int,
    test_frac: float,
    seed: int,
    hier_shrink_k: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Known-target panel CV (stratified split within each target).

    This estimates performance when target identity is known and represented in
    training data (common medicinal chemistry scenario).
    """
    compact_cols = [
        "dg_expavg_kj",
        "dg_cumulant_kj",
        "energy_mean_kj",
        "coul_mean_kj",
        "contact_frac_mean",
        "pocket_volume_bbox_a3",
        "pocket_volume_sphere95_a3",
        "pocket_atom_density_per_a3",
        "pocket_polar_atom_frac",
        "pocket_n_residues",
        "pocket_n_atoms",
        "pocket_frac_hydrophobic",
        "pocket_frac_polar",
        "pocket_frac_positive",
        "pocket_frac_negative",
        "pocket_frac_aromatic",
        "pocket_charge_est",
        "pocket_charge_density",
    ]
    compact_cols = [c for c in compact_cols if c in df.columns]
    if len(compact_cols) < 3:
        raise RuntimeError("Not enough compact features available")

    work = df.reset_index(drop=True).copy()
    y_all = work["exp_dg_kj_model"].to_numpy(float)
    xea_all = work["dg_expavg_kj"].to_numpy(float)
    w_all = pd.to_numeric(work.get("sample_weight", 1.0), errors="coerce").to_numpy(float)
    groups = work["target"].astype(str).to_numpy()

    Xc_all = work.loc[:, compact_cols].to_numpy(float)

    # target one-hot (known target identity available in this scenario)
    uniq_targets = sorted(set(groups.tolist()))
    t_index = {t: i for i, t in enumerate(uniq_targets)}
    T = np.zeros((len(work), len(uniq_targets)), dtype=float)
    for i, g in enumerate(groups):
        T[i, t_index[g]] = 1.0
    Xta_all = np.hstack([Xc_all, T])

    tune_rep = int(np.clip(max(20, repeats // 5), 20, 100))
    alpha_compact, alpha_compact_scores = _tune_alpha_within_target(
        X=Xc_all,
        y=y_all,
        w=w_all,
        groups=groups,
        alphas=alphas,
        repeats=tune_rep,
        test_frac=test_frac,
        seed=seed,
    )
    alpha_targetaware, alpha_target_scores = _tune_alpha_within_target(
        X=Xta_all,
        y=y_all,
        w=w_all,
        groups=groups,
        alphas=alphas,
        repeats=tune_rep,
        test_frac=test_frac,
        seed=seed + 17,
    )

    rng = np.random.default_rng(seed)
    fold_rows: List[FoldRow] = []
    pred_rows: List[dict] = []

    models = ["raw_expavg", "cal_expavg", "ridge_compact", "ridge_targetaware", "ridge_targetaware_hier"]
    agg_sum = {m: np.zeros(len(work), dtype=float) for m in models}
    agg_n = {m: np.zeros(len(work), dtype=int) for m in models}

    for rep in range(int(repeats)):
        te = _target_stratified_split(groups, test_frac=test_frac, rng=rng)
        tr = ~te
        if np.sum(te) < 2 or np.sum(tr) < 5:
            continue

        y_tr = y_all[tr]
        y_te = y_all[te]
        w_tr = w_all[tr]
        w_te = w_all[te]
        xea_tr = xea_all[tr]
        xea_te = xea_all[te]

        raw_expavg = xea_te.copy()
        be = fit_linear(xea_tr[:, None], y_tr, sample_weight=w_tr)
        cal_expavg = pred_linear(xea_te[:, None], be)

        mc = fit_ridge(Xc_all[tr], y_tr, alpha=alpha_compact, sample_weight=w_tr)
        ridge_compact = pred_ridge(Xc_all[te], mc)

        mt = fit_ridge(Xta_all[tr], y_tr, alpha=alpha_targetaware, sample_weight=w_tr)
        ridge_targetaware = pred_ridge(Xta_all[te], mt)

        # Hierarchical target-wise linear correction (shrunken to avoid overfit).
        pred_tr_ta = pred_ridge(Xta_all[tr], mt)
        corr = _fit_target_linear_corrections(
            y_true=y_tr,
            y_pred=pred_tr_ta,
            groups=groups[tr],
            sample_weight=w_tr,
            shrink_k=float(hier_shrink_k),
        )
        ridge_targetaware_hier = _apply_target_linear_corrections(
            pred=ridge_targetaware,
            groups=groups[te],
            corr=corr,
        )

        pred_map = {
            "raw_expavg": raw_expavg,
            "cal_expavg": cal_expavg,
            "ridge_compact": ridge_compact,
            "ridge_targetaware": ridge_targetaware,
            "ridge_targetaware_hier": ridge_targetaware_hier,
        }

        te_idx = np.where(te)[0]
        for mname, pred in pred_map.items():
            fold_rows.append(
                FoldRow(
                    model=mname,
                    fold_target=f"repeat_{rep}",
                    n_test=int(len(y_te)),
                    rmse=rmse(y_te, pred, w=w_te),
                    mae=mae(y_te, pred, w=w_te),
                    pearson=pearson(y_te, pred),
                    spearman=spearman(y_te, pred),
                    alpha=float(
                        alpha_targetaware
                        if mname in {"ridge_targetaware", "ridge_targetaware_hier"}
                        else (alpha_compact if mname == "ridge_compact" else np.nan)
                    ),
                )
            )
            agg_sum[mname][te_idx] += pred
            agg_n[mname][te_idx] += 1

        for loc, idx in enumerate(te_idx):
            pred_rows.append(
                {
                    "repeat": int(rep),
                    "row_index": int(idx),
                    "target": str(groups[idx]),
                    "ligand": str(work.iloc[idx]["ligand"]),
                    "exp_dg_kj": float(y_te[loc]),
                    "sample_weight": float(w_te[loc]),
                    "raw_expavg": float(raw_expavg[loc]),
                    "cal_expavg": float(cal_expavg[loc]),
                    "ridge_compact": float(ridge_compact[loc]),
                    "ridge_targetaware": float(ridge_targetaware[loc]),
                    "ridge_targetaware_hier": float(ridge_targetaware_hier[loc]),
                }
            )

    fold_df = pd.DataFrame([r.__dict__ for r in fold_rows])
    pred_df = pd.DataFrame(pred_rows)

    oos = pd.DataFrame(
        {
            "row_index": np.arange(len(work), dtype=int),
            "target": work["target"].astype(str).to_numpy(),
            "ligand": work["ligand"].astype(str).to_numpy(),
            "exp_dg_kj": y_all,
            "sample_weight": w_all,
        }
    )
    for m in models:
        mean_pred = np.where(agg_n[m] > 0, agg_sum[m] / np.maximum(agg_n[m], 1), np.nan)
        oos[m] = mean_pred
        oos[f"{m}_n"] = agg_n[m]

    meta = {
        "alpha_compact": float(alpha_compact),
        "alpha_targetaware": float(alpha_targetaware),
        "alpha_compact_scores": alpha_compact_scores,
        "alpha_targetaware_scores": alpha_target_scores,
        "hier_shrink_k": float(hier_shrink_k),
        "models": models,
    }
    return fold_df, pred_df, oos, meta


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in fold_df.groupby("model", sort=False):
        rows.append(
            {
                "model": m,
                "rmse_mean": float(g["rmse"].mean()),
                "rmse_std": float(g["rmse"].std(ddof=0)),
                "mae_mean": float(g["mae"].mean()),
                "mae_std": float(g["mae"].std(ddof=0)),
                "pearson_mean": float(g["pearson"].mean()),
                "pearson_std": float(g["pearson"].std(ddof=0)),
                "spearman_mean": float(g["spearman"].mean()),
                "spearman_std": float(g["spearman"].std(ddof=0)),
                "n_folds": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_mean", ascending=True)


def plot_scatter(path: Path, y: np.ndarray, raw: np.ndarray, cal: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    for ax, (name, pred) in zip(axes, [("Raw", raw), ("Calibrated", cal)]):
        m = np.isfinite(y) & np.isfinite(pred)
        yt = y[m]
        yp = pred[m]
        if len(yt) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        lo = min(float(np.min(yt)), float(np.min(yp))) - 2
        hi = max(float(np.max(yt)), float(np.max(yp))) + 2
        ax.scatter(yt, yp, s=24, alpha=0.8)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Experimental DeltaG (kJ/mol)")
        ax.set_ylabel("Predicted DeltaG (kJ/mol)")
        ax.set_title(f"{name}  n={len(yt)}  r={pearson(yt, yp):.3f}  RMSE={rmse(yt, yp):.2f}")
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    alphas = [float(x.strip()) for x in str(args.alphas).split(",") if x.strip()]
    if not alphas:
        raise RuntimeError("No valid alphas provided")

    df, info = load_data(
        csv_path=args.csv,
        data_root=Path(args.data_root).resolve(),
        fill_exp_from_yml=not bool(args.no_fill_exp_from_yml),
        temperature_k=float(args.temperature_k),
        weight_mode=str(args.weight_mode),
        endpoint_filter=str(args.endpoint_filter),
        ic50_normalization=str(args.ic50_normalization),
        ic50_shift_min_rows=int(args.ic50_shift_min_rows),
        docking_csv=(Path(args.docking_csv).resolve() if str(args.docking_csv).strip() else None),
        docking_target_col=str(args.docking_target_col),
        docking_ligand_col=str(args.docking_ligand_col),
        docking_affinity_col=str(args.docking_affinity_col),
        docking_unit=str(args.docking_unit),
    )

    # Unknown-target generalization
    fold_df, pred_df = evaluate_loto(
        df,
        alphas=alphas,
        endpoint_aware=bool(args.endpoint_aware),
        endpoint_min_rows=int(args.endpoint_min_rows),
    )
    docking_col = str(info.get("docking_col", "") or "")
    if (not bool(args.no_docking_stack)) and docking_col and (docking_col in df.columns):
        pred_df = pred_df.merge(
            df.loc[:, ["target", "ligand", docking_col]].drop_duplicates(subset=["target", "ligand"]),
            on=["target", "ligand"],
            how="left",
        )
        fold_df, pred_df = append_groupcv_stack_predictions(
            fold_df,
            pred_df,
            feature_cols=["cal_combo_ep", "cal_expavg", docking_col],
            predictor_name="stack_cal_combo_ep_cal_expavg_dock_loto",
        )
    if fold_df.empty:
        raise RuntimeError("No folds evaluated")
    summary_df = summarize(fold_df)
    best_model = str(summary_df.iloc[0]["model"])
    summary_abfe_only = summary_df.loc[
        ~summary_df["model"].astype(str).str.startswith("stack_")
    ].reset_index(drop=True)
    best_model_abfe_only = (
        str(summary_abfe_only.iloc[0]["model"])
        if not summary_abfe_only.empty
        else str(best_model)
    )

    raw_name = (
        "raw_expavg"
        if best_model
        in {
            "cal_expavg",
            "cal_combo",
            "cal_combo_ep",
            "ridge_compact",
            "ridge_compact_ep",
            "ridge_all",
            "rel_hybrid",
            "stack_cal_combo_ep_cal_expavg_dock_loto",
            "raw_expavg",
        }
        else "raw_pop"
    )
    y = pred_df["exp_dg_kj"].to_numpy(float)
    raw = pred_df[raw_name].to_numpy(float)
    cal = pred_df[best_model].to_numpy(float)
    plot_scatter(out / "groupcv_scatter_best_vs_raw.png", y, raw, cal, f"LOTO best={best_model} vs {raw_name}")

    # Known-target panel CV
    wt_fold_df, wt_pred_df, wt_oos_df, wt_meta = evaluate_within_target_cv(
        df=df,
        alphas=alphas,
        repeats=int(args.within_target_repeats),
        test_frac=float(args.within_target_test_frac),
        seed=int(args.seed),
        hier_shrink_k=float(args.hier_shrink_k),
    )
    wt_summary_df = summarize(wt_fold_df)
    wt_best = str(wt_summary_df.iloc[0]["model"])
    wt_raw_name = "raw_expavg"

    y_wt = wt_oos_df["exp_dg_kj"].to_numpy(float)
    raw_wt = wt_oos_df[wt_raw_name].to_numpy(float)
    cal_wt = wt_oos_df[wt_best].to_numpy(float)
    plot_scatter(
        out / "within_target_scatter_best_vs_raw.png",
        y_wt,
        raw_wt,
        cal_wt,
        f"Within-target CV best={wt_best} vs {wt_raw_name}",
    )

    # Fit deploy compact and linear models on all rows
    yall = df["exp_dg_kj_model"].to_numpy(float)
    xea = df["dg_expavg_kj"].to_numpy(float)
    xpo = df["dg_pop_or_jeff_kj"].to_numpy(float)
    grp = df["target"].astype(str).to_numpy()
    w_all = pd.to_numeric(df.get("sample_weight", 1.0), errors="coerce").to_numpy(float)

    compact_cols = [
        c
        for c in ["dg_expavg_kj", "dg_cumulant_kj", "energy_mean_kj", "coul_mean_kj", "contact_frac_mean", "pocket_volume_bbox_a3", "pocket_volume_sphere95_a3", "pocket_atom_density_per_a3", "pocket_polar_atom_frac", "pocket_n_residues", "pocket_n_atoms", "pocket_frac_hydrophobic", "pocket_frac_polar", "pocket_frac_positive", "pocket_frac_negative", "pocket_frac_aromatic", "pocket_charge_est", "pocket_charge_density"]
        if c in df.columns
    ]
    Xc = df.loc[:, compact_cols].to_numpy(float)
    ac = choose_alpha(Xc, yall, grp, alphas, sample_weight=w_all)
    mc = fit_ridge(Xc, yall, alpha=ac, sample_weight=w_all)

    be = fit_linear(xea[:, None], yall, sample_weight=w_all)
    bc = fit_linear(np.column_stack([xea, xpo]), yall, sample_weight=w_all)

    Xc_center, x_means = _center_by_group(Xc, grp)
    y_center, y_means = _center_y_by_group(yall, grp)
    ar = choose_alpha(Xc_center, y_center, grp, alphas, sample_weight=w_all)
    mr = fit_ridge(Xc_center, y_center, alpha=ar, sample_weight=w_all)

    # Known-target panel model: compact features + target identity one-hot.
    at = float(wt_meta["alpha_targetaware"])
    ta_pack = _fit_targetaware_full(
        X_compact=Xc,
        groups=grp,
        y=yall,
        sample_weight=w_all,
        alpha=at,
    )
    ta_model = ta_pack["model"]
    ta_targets = ta_pack["target_order"]

    ep_grp = df.get("endpoint_group", "unknown").astype(str).to_numpy()
    cal_combo_all = pred_linear(np.column_stack([xea, xpo]), bc)
    off_combo_all, off_combo_counts = _fit_endpoint_offsets(
        ep_grp,
        yall - cal_combo_all,
        w_all,
        min_rows=int(args.endpoint_min_rows),
    )
    ridge_compact_all = pred_ridge(Xc, mc)
    off_ridge_all, off_ridge_counts = _fit_endpoint_offsets(
        ep_grp,
        yall - ridge_compact_all,
        w_all,
        min_rows=int(args.endpoint_min_rows),
    )

    pred_ta_all = _predict_targetaware(Xc, grp, ta_pack)
    corr_ta = _fit_target_linear_corrections(
        y_true=yall,
        y_pred=pred_ta_all,
        groups=grp,
        sample_weight=w_all,
        shrink_k=float(args.hier_shrink_k),
    )

    cal_expavg_all = pred_linear(xea[:, None], be)
    cal_combo_ep_all = _apply_endpoint_offsets(cal_combo_all, ep_grp, off_combo_all)
    stack_deploy = {}
    if (not bool(args.no_docking_stack)) and docking_col and (docking_col in df.columns):
        xdock_all = pd.to_numeric(df[docking_col], errors="coerce").to_numpy(float)
        stack_mask = np.isfinite(yall) & np.isfinite(cal_combo_ep_all) & np.isfinite(cal_expavg_all) & np.isfinite(xdock_all)
        if int(np.sum(stack_mask)) >= 6:
            stack_coef = fit_linear(
                np.column_stack([cal_combo_ep_all[stack_mask], cal_expavg_all[stack_mask], xdock_all[stack_mask]]),
                yall[stack_mask],
                sample_weight=w_all[stack_mask],
            )
            stack_deploy = {
                "enabled": True,
                "predictor_name": "stack_cal_combo_ep_cal_expavg_dock_loto",
                "feature_cols": ["cal_combo_ep", "cal_expavg", docking_col],
                "coef": np.asarray(stack_coef).tolist(),
                "n_rows_fit": int(np.sum(stack_mask)),
                "note": (
                    "Apply after computing cal_combo_ep and cal_expavg from the weighted ABFE models "
                    "and providing the merged docking affinity column."
                ),
            }
        else:
            stack_deploy = {
                "enabled": False,
                "predictor_name": "stack_cal_combo_ep_cal_expavg_dock_loto",
                "feature_cols": ["cal_combo_ep", "cal_expavg", docking_col],
                "n_rows_fit": int(np.sum(stack_mask)),
                "note": "Insufficient rows with finite docking and ABFE predictors to fit stack.",
            }

    deploy = {
        "n_rows": int(len(df)),
        "targets": sorted(df["target"].astype(str).unique().tolist()),
        "best_model_groupcv": best_model_abfe_only,
        "best_model_groupcv_abfe_only": best_model_abfe_only,
        "best_model_groupcv_overall": best_model,
        "best_model_within_target_cv": wt_best,
        "training_info": info,
        "linear_expavg_weighted": {"a": float(be[0]), "b": float(be[1])},
        "linear_combo_weighted": {"a": float(bc[0]), "b": float(bc[1]), "c": float(bc[2])},
        "ridge_compact_weighted": {
            "feature_cols": compact_cols,
            "alpha": float(ac),
            "med": np.asarray(mc["med"]).tolist(),
            "mu": np.asarray(mc["mu"]).tolist(),
            "sd": np.asarray(mc["sd"]).tolist(),
            "w": np.asarray(mc["w"]).tolist(),
            "y_mean": float(mc["y_mean"]),
        },
        "endpoint_residual_offsets_weighted": {
            "enabled": bool(args.endpoint_aware),
            "endpoint_min_rows": int(args.endpoint_min_rows),
            "cal_combo_offsets": off_combo_all,
            "cal_combo_counts": off_combo_counts,
            "ridge_compact_offsets": off_ridge_all,
            "ridge_compact_counts": off_ridge_counts,
            "endpoint_col": "endpoint_group",
            "note": "Apply as additive residual correction after base model prediction.",
        },
        "relative_hybrid_weighted": {
            "feature_cols": compact_cols,
            "alpha": float(ar),
            "med": np.asarray(mr["med"]).tolist(),
            "mu": np.asarray(mr["mu"]).tolist(),
            "sd": np.asarray(mr["sd"]).tolist(),
            "w": np.asarray(mr["w"]).tolist(),
            "y_mean": float(mr["y_mean"]),
            "train_target_feature_means": {k: np.asarray(v).tolist() for k, v in x_means.items()},
            "train_target_y_means": y_means,
            "anchor_model": "linear_combo_weighted",
            "note": "For panel-level inference; for single ligand use absolute weighted models.",
        },
        "ridge_targetaware_weighted": {
            "feature_cols": compact_cols,
            "target_order": list(ta_targets),
            "alpha": float(at),
            "med": np.asarray(ta_model["med"]).tolist(),
            "mu": np.asarray(ta_model["mu"]).tolist(),
            "sd": np.asarray(ta_model["sd"]).tolist(),
            "w": np.asarray(ta_model["w"]).tolist(),
            "y_mean": float(ta_model["y_mean"]),
            "note": "Use for known-target ligand panels where target is in target_order.",
        },
        "ridge_targetaware_hier_weighted": {
            "base_model": "ridge_targetaware_weighted",
            "hier_shrink_k": float(args.hier_shrink_k),
            "target_linear_corrections": corr_ta,
            "note": "Apply y <- a_target*y + b_target for known-target panels.",
        },
        "stack_abfe_docking_weighted": stack_deploy,
        "within_target_cv": {
            "repeats": int(args.within_target_repeats),
            "test_frac": float(args.within_target_test_frac),
            "alpha_compact": float(wt_meta["alpha_compact"]),
            "alpha_targetaware": float(wt_meta["alpha_targetaware"]),
            "hier_shrink_k": float(args.hier_shrink_k),
        },
    }

    # Save outputs
    fold_df.to_csv(out / "groupcv_fold_metrics.csv", index=False)
    summary_df.to_csv(out / "groupcv_summary_metrics.csv", index=False)
    pred_df.to_csv(out / "groupcv_predictions.csv", index=False)

    wt_fold_df.to_csv(out / "within_target_cv_fold_metrics.csv", index=False)
    wt_summary_df.to_csv(out / "within_target_cv_summary_metrics.csv", index=False)
    wt_pred_df.to_csv(out / "within_target_cv_predictions.csv", index=False)
    wt_oos_df.to_csv(out / "within_target_cv_oos_mean_predictions.csv", index=False)

    with open(out / "deploy_models.json", "w", encoding="utf-8") as fh:
        json.dump(deploy, fh, indent=2)

    lines: List[str] = []
    lines.append("# Compact Calibration Report")
    lines.append("")
    lines.append(f"- input_csv: `{Path(args.csv).resolve()}`")
    lines.append(f"- data_root: `{Path(args.data_root).resolve()}`")
    lines.append(f"- n_rows_model: {len(df)}")
    lines.append(f"- targets: {', '.join(sorted(df['target'].astype(str).unique().tolist()))}")
    lines.append(f"- alpha_grid: {alphas}")
    lines.append(f"- weight_mode: {args.weight_mode}")
    lines.append(f"- endpoint_filter: {info.get('endpoint_filter', 'all')}")
    lines.append(f"- endpoint_aware: {bool(args.endpoint_aware)}")
    lines.append(f"- endpoint_min_rows: {int(args.endpoint_min_rows)}")
    lines.append(f"- hier_shrink_k: {float(args.hier_shrink_k)}")
    lines.append(f"- best_model_groupcv_abfe_only: {best_model_abfe_only}")
    if best_model != best_model_abfe_only:
        lines.append(f"- best_model_groupcv_overall: {best_model}")
    if info.get("docking_col"):
        lines.append(
            f"- docking: {info.get('docking_col')} "
            f"(merged/model rows {int(info.get('docking_rows_merged', 0))}/{int(info.get('docking_rows_model', 0))})"
        )
        lines.append(f"- docking_stack_enabled: {not bool(args.no_docking_stack)}")
    lines.append(f"- fill_exp_from_yml: {not bool(args.no_fill_exp_from_yml)}")
    lines.append(f"- exp before/after/fill: {info['n_exp_before']} / {info['n_exp_after']} / +{info['n_exp_filled']}")
    if int(info.get("n_exp_model_after", info.get("n_exp_after", 0))) != int(info.get("n_exp_after", 0)):
        lines.append(f"- exp after model-normalization: {int(info.get('n_exp_model_after', 0))}")
    lines.append(f"- ic50_normalization: {info.get('ic50_normalization_mode', 'none')}")
    lines.append(
        f"- ic50_normalization_counts (cp/shift_target/shift_global): "
        f"{int(info.get('ic50_cp_applied', 0))}/"
        f"{int(info.get('ic50_shift_target_applied', 0))}/"
        f"{int(info.get('ic50_shift_global_applied', 0))}"
    )
    if np.isfinite(float(info.get("ic50_shift_global_kj", float("nan")))):
        lines.append(f"- ic50_shift_global_kj: {float(info.get('ic50_shift_global_kj')):.3f}")
    if int(info.get("ic50_shift_targets_n", 0)) > 0:
        lines.append(f"- ic50_shift_targets_n: {int(info.get('ic50_shift_targets_n'))}")
    if info.get("endpoint_counts"):
        lines.append(f"- endpoint_counts: {info['endpoint_counts']}")
    if int(info.get("n_pocket_descriptor_cols", 0)) > 0:
        lines.append(f"- pocket_descriptors: {info['n_pocket_descriptor_cols']} columns")
        lines.append(f"- pocket_descriptor_cols: {', '.join(info.get('pocket_descriptor_cols', []))}")
    lines.append("")
    lines.append("## Unknown-Target Generalization (LOTO)")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append(f"- best_model: `{best_model}`")
    lines.append(f"- compared_to_raw: `{raw_name}`")
    lines.append("")
    lines.append("## Known-Target Panel CV (Stratified within target)")
    lines.append("")
    lines.append(f"- repeats: {args.within_target_repeats}")
    lines.append(f"- test_frac: {args.within_target_test_frac}")
    lines.append(wt_summary_df.to_markdown(index=False))
    lines.append("")
    lines.append(f"- best_model: `{wt_best}`")
    lines.append(f"- compared_to_raw: `{wt_raw_name}`")
    lines.append("")
    lines.append("## Weighted Compact Deployment")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(deploy["ridge_compact_weighted"], indent=2))
    lines.append("```")
    (out / "groupcv_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out / 'groupcv_fold_metrics.csv'}")
    print(f"Wrote: {out / 'groupcv_summary_metrics.csv'}")
    print(f"Wrote: {out / 'groupcv_predictions.csv'}")
    print(f"Wrote: {out / 'groupcv_scatter_best_vs_raw.png'}")
    print(f"Wrote: {out / 'within_target_cv_fold_metrics.csv'}")
    print(f"Wrote: {out / 'within_target_cv_summary_metrics.csv'}")
    print(f"Wrote: {out / 'within_target_cv_predictions.csv'}")
    print(f"Wrote: {out / 'within_target_cv_oos_mean_predictions.csv'}")
    print(f"Wrote: {out / 'within_target_scatter_best_vs_raw.png'}")
    print(f"Wrote: {out / 'deploy_models.json'}")
    print(f"Wrote: {out / 'groupcv_report.md'}")
    print(f"Info: exp before={info['n_exp_before']} after={info['n_exp_after']} filled={info['n_exp_filled']}")


if __name__ == "__main__":
    main()
