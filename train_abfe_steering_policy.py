#!/usr/bin/env python3
"""Train and evaluate a safe ABFE steering policy from logged policy traces."""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from abfe_runtime import ABFE_STEERING_ACTION_NAMES
from abfe_runtime import ABFE_STEERING_POLICY_FEATURE_NAMES
from abfe_runtime import ABFE_STEERING_POLICY_OOD_FEATURE_NAMES
from abfe_runtime import build_abfe_steering_policy_feature_payload
from abfe_runtime import infer_abfe_steering_scaffold_key

try:
    from sklearn.linear_model import HuberRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    HAS_SKLEARN = True
except Exception:
    HuberRegressor = None
    Ridge = None
    GroupKFold = None
    HAS_SKLEARN = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace",
        action="append",
        default=[],
        help="Policy trace JSONL file or directory containing policy_trace.jsonl files. Can be repeated.",
    )
    p.add_argument(
        "--trace-glob",
        action="append",
        default=[],
        help="Glob pattern for policy_trace.jsonl files. Can be repeated.",
    )
    p.add_argument(
        "--out-dir",
        default="test_abfe_3ptb_ben/policy_model_training",
        help="Output directory for model JSON and reports.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength for per-action linear models.",
    )
    p.add_argument(
        "--min-action-samples",
        type=int,
        default=5,
        help="Minimum observed rows before fitting a full action model.",
    )
    p.add_argument(
        "--blend-alpha",
        type=float,
        default=0.25,
        help="Runtime blend factor for learned scores added to heuristic scores.",
    )
    p.add_argument(
        "--model-kind",
        default="auto",
        help="Action-value model family: auto, ridge, or huber.",
    )
    p.add_argument(
        "--include-trained-model-runs",
        action="store_true",
        help="Include traces already generated with a trained policy model.",
    )
    return p.parse_args()


def _safe_float(value, default=np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return float(out) if np.isfinite(out) else float(default)


def _metric_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _metric_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return np.nan
    return float(np.mean(np.abs(y_pred - y_true)))


def resolve_trace_paths(explicit_paths: Sequence[str], glob_patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    candidates = list(explicit_paths or [])
    patterns = list(glob_patterns or [])
    if (not candidates) and (not patterns):
        patterns = ["test_abfe_3ptb_ben/**/policy_trace.jsonl"]

    for item in candidates:
        path = Path(str(item)).expanduser().resolve()
        if path.is_file():
            if str(path) not in seen:
                paths.append(path)
                seen.add(str(path))
            continue
        if path.is_dir():
            for hit in sorted(path.rglob("policy_trace.jsonl")):
                resolved = hit.resolve()
                if str(resolved) not in seen:
                    paths.append(resolved)
                    seen.add(str(resolved))

    for pattern in patterns:
        for hit in sorted(glob.glob(str(pattern), recursive=True)):
            path = Path(hit).expanduser().resolve()
            if path.is_file() and str(path) not in seen:
                paths.append(path)
                seen.add(str(path))

    return paths


def load_trace_rows(trace_paths: Sequence[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in list(trace_paths or []):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            row = dict(payload)
            row["trace_path"] = str(path)
            row["trace_group"] = str(path.parent.name or path.stem)
            row["trace_lineno"] = int(lineno)
            target_hint = str(row.get("target_hint", "") or "").strip()
            row["target_group"] = target_hint if target_hint else str(row["trace_group"])
            ligand_smiles = str(row.get("ligand_smiles", "") or "").strip()
            ligand_scaffold = str(row.get("ligand_scaffold", "") or "").strip()
            if (not ligand_scaffold) and ligand_smiles:
                ligand_scaffold = str(infer_abfe_steering_scaffold_key(ligand_smiles) or "").strip()
            row["ligand_scaffold"] = str(ligand_scaffold)
            row["scaffold_group"] = (
                str(ligand_scaffold)
                if ligand_scaffold
                else (ligand_smiles if ligand_smiles else str(row["trace_group"]))
            )
            rows.append(row)
    return rows


def build_feature_matrix(rows: Sequence[dict], feature_names: Sequence[str]) -> np.ndarray:
    mats = []
    for row in list(rows or []):
        feat_payload = build_abfe_steering_policy_feature_payload(
            observation=dict(row or {}),
            policy_state=dict(row or {}),
        )
        feat_map = dict(feat_payload.get("features", {}) or {})
        mats.append([_safe_float(feat_map.get(name, 0.0), 0.0) for name in feature_names])
    if not mats:
        return np.zeros((0, len(list(feature_names or []))), dtype=float)
    return np.asarray(mats, dtype=float)


def fit_ridge_weights(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if X.shape[0] == 0:
        return np.zeros(X.shape[1], dtype=float)
    alpha = max(0.0, float(alpha))
    xtx = X.T @ X
    reg = alpha * np.eye(X.shape[1], dtype=float)
    xty = X.T @ y
    try:
        weights = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(xtx + reg, xty, rcond=None)[0]
    return np.asarray(weights, dtype=float)


def fit_linear_weights(X: np.ndarray, y: np.ndarray, *, alpha: float, model_kind: str) -> np.ndarray:
    kind = str(model_kind or "ridge").strip().lower()
    if (not HAS_SKLEARN) or kind not in {"ridge", "huber"}:
        return fit_ridge_weights(X, y, alpha=float(alpha))
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 0, dtype=float)
    try:
        if kind == "huber":
            model = HuberRegressor(
                alpha=float(max(1.0e-6, alpha)),
                fit_intercept=False,
                epsilon=1.35,
                max_iter=200,
            )
        else:
            model = Ridge(alpha=float(max(0.0, alpha)), fit_intercept=False)
        model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        coef = np.asarray(getattr(model, "coef_", np.zeros(X.shape[1], dtype=float)), dtype=float)
        if coef.size == X.shape[1]:
            return coef
    except Exception:
        pass
    return fit_ridge_weights(X, y, alpha=float(alpha))


def build_group_labels(rows: Sequence[dict], group_key: str) -> List[str]:
    key = str(group_key or "trace_group").strip()
    out = []
    for row in list(rows or []):
        val = str(row.get(key, "") or "").strip()
        if not val:
            val = str(row.get("trace_group", "") or "").strip()
        out.append(val)
    return out


def build_ood_profile(rows: Sequence[dict], feature_names: Sequence[str]) -> dict:
    names = [str(v) for v in list(feature_names or []) if str(v)]
    if not names:
        return {
            "ok": False,
            "ood_feature_names": [],
            "ood_center": [],
            "ood_scale": [],
            "ood_distance_threshold": np.nan,
            "ood_max_abs_z": np.nan,
        }
    X = build_feature_matrix(rows, names)
    if X.ndim != 2 or X.shape[0] == 0:
        return {
            "ok": False,
            "ood_feature_names": list(names),
            "ood_center": [],
            "ood_scale": [],
            "ood_distance_threshold": np.nan,
            "ood_max_abs_z": np.nan,
        }
    center = np.nanmean(X, axis=0)
    scale = np.nanstd(X, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    scale = np.where(np.logical_and(np.isfinite(scale), scale > 1.0e-6), scale, 1.0)
    z = (X - center[None, :]) / scale[None, :]
    dist = np.sqrt(np.mean(np.square(z), axis=1))
    max_abs = np.max(np.abs(z), axis=1)
    return {
        "ok": True,
        "ood_feature_names": list(names),
        "ood_center": [float(v) for v in np.asarray(center, dtype=float).tolist()],
        "ood_scale": [float(v) for v in np.asarray(scale, dtype=float).tolist()],
        "ood_distance_threshold": float(np.quantile(np.asarray(dist, dtype=float), 0.975)) if dist.size else np.nan,
        "ood_max_abs_z": float(np.quantile(np.asarray(max_abs, dtype=float), 0.99)) if max_abs.size else np.nan,
    }


def fit_action_models(
    rows: Sequence[dict],
    *,
    alpha: float,
    min_action_samples: int,
    feature_names: Sequence[str],
    model_kind: str = "ridge",
) -> Dict[str, dict]:
    feat_names = list(feature_names or [])
    global_rewards = np.asarray([_safe_float(r.get("reward_estimate", np.nan)) for r in list(rows or [])], dtype=float)
    global_rewards = global_rewards[np.isfinite(global_rewards)]
    global_mean = float(np.mean(global_rewards)) if global_rewards.size else 0.0

    models: Dict[str, dict] = {}
    for action_name in ABFE_STEERING_ACTION_NAMES:
        action_rows = [
            dict(r)
            for r in list(rows or [])
            if str(r.get("policy_action_name", "") or "") == str(action_name)
            and np.isfinite(_safe_float(r.get("reward_estimate", np.nan)))
        ]
        y = np.asarray([_safe_float(r.get("reward_estimate", np.nan)) for r in action_rows], dtype=float)
        action_mean = float(np.mean(y)) if y.size else float(global_mean)
        if len(action_rows) >= max(1, int(min_action_samples)):
            X = build_feature_matrix(action_rows, feat_names)
            weights = fit_linear_weights(X, y - action_mean, alpha=float(alpha), model_kind=str(model_kind))
        else:
            weights = np.zeros(len(feat_names), dtype=float)
        models[str(action_name)] = {
            "weights": [float(x) for x in np.asarray(weights, dtype=float).tolist()],
            "n_train": int(len(action_rows)),
            "mean_reward": float(action_mean),
        }
    return models


def predict_row_action_scores(row: dict, *, feature_names: Sequence[str], models: Dict[str, dict]) -> Dict[str, float]:
    feat_payload = build_abfe_steering_policy_feature_payload(
        observation=dict(row or {}),
        policy_state=dict(row or {}),
    )
    feat_map = dict(feat_payload.get("features", {}) or {})
    vector = np.asarray([_safe_float(feat_map.get(name, 0.0), 0.0) for name in list(feature_names or [])], dtype=float)
    out = {}
    for action_name in ABFE_STEERING_ACTION_NAMES:
        weights = np.asarray(list(dict(models.get(action_name, {}) or {}).get("weights", []) or []), dtype=float)
        if weights.size != vector.size:
            score = 0.0
        else:
            score = float(np.dot(vector, weights))
        out[str(action_name)] = float(score)
    return out


def evaluate_group_cv(
    rows: Sequence[dict],
    *,
    alpha: float,
    min_action_samples: int,
    feature_names: Sequence[str],
    model_kind: str,
    group_key: str,
) -> dict:
    group_labels = build_group_labels(rows, group_key)
    groups = sorted({str(v) for v in list(group_labels or []) if str(v)})
    if len(groups) < 2:
        return {
            "ok": False,
            "note": "need_at_least_two_trace_groups_for_cv",
            "n_groups": int(len(groups)),
            "group_key": str(group_key),
        }

    y_true = []
    y_pred = []
    match_rewards = []
    match_count = 0
    action_match_counts = {name: 0 for name in ABFE_STEERING_ACTION_NAMES}
    best_action_counts = {name: 0 for name in ABFE_STEERING_ACTION_NAMES}

    if HAS_SKLEARN and GroupKFold is not None:
        n_splits = min(5, len(groups))
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(np.arange(len(list(rows or []))), groups=np.asarray(group_labels, dtype=object))
    else:
        split_iter = []
        for group in groups:
            train_idx = [i for i, gg in enumerate(group_labels) if str(gg) != str(group)]
            test_idx = [i for i, gg in enumerate(group_labels) if str(gg) == str(group)]
            split_iter.append((np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)))

    rows_list = [dict(r) for r in list(rows or [])]
    for train_idx, test_idx in split_iter:
        train_rows = [dict(rows_list[int(i)]) for i in np.asarray(train_idx, dtype=int).tolist()]
        test_rows = [dict(rows_list[int(i)]) for i in np.asarray(test_idx, dtype=int).tolist()]
        if not train_rows or not test_rows:
            continue
        models = fit_action_models(
            train_rows,
            alpha=float(alpha),
            min_action_samples=int(min_action_samples),
            feature_names=list(feature_names or []),
            model_kind=str(model_kind),
        )
        for row in test_rows:
            actual_reward = _safe_float(row.get("reward_estimate", np.nan))
            if not np.isfinite(actual_reward):
                continue
            logged_action = str(row.get("policy_action_name", "keep") or "keep")
            scores = predict_row_action_scores(
                row,
                feature_names=list(feature_names or []),
                models=models,
            )
            predicted_reward = float(scores.get(logged_action, np.nan))
            if np.isfinite(predicted_reward):
                y_true.append(float(actual_reward))
                y_pred.append(float(predicted_reward))
            if scores:
                best_action = max(scores.items(), key=lambda kv: float(kv[1]))[0]
                best_action_counts[best_action] = int(best_action_counts.get(best_action, 0)) + 1
                if best_action == logged_action:
                    match_count += 1
                    action_match_counts[best_action] = int(action_match_counts.get(best_action, 0)) + 1
                    match_rewards.append(float(actual_reward))

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return {
        "ok": True,
        "group_key": str(group_key),
        "model_kind": str(model_kind),
        "n_groups": int(len(groups)),
        "n_eval_rows": int(y_true_arr.size),
        "reward_rmse": _metric_rmse(y_true_arr, y_pred_arr),
        "reward_mae": _metric_mae(y_true_arr, y_pred_arr),
        "match_coverage": float(match_count / max(1, len(y_true))),
        "mean_reward_logged": float(np.mean(y_true_arr)) if y_true_arr.size else np.nan,
        "mean_reward_matched": float(np.mean(np.asarray(match_rewards, dtype=float))) if match_rewards else np.nan,
        "best_action_counts": dict(best_action_counts),
        "matched_action_counts": dict(action_match_counts),
    }


def select_model_kind(
    rows: Sequence[dict],
    *,
    alpha: float,
    min_action_samples: int,
    feature_names: Sequence[str],
    requested_kind: str,
) -> dict:
    kind_txt = str(requested_kind or "auto").strip().lower()
    if kind_txt in {"ridge", "huber"}:
        return {"model_kind": str(kind_txt), "selection_eval": {"ok": False, "note": "forced_by_user"}}
    candidates = ["ridge"]
    if HAS_SKLEARN:
        candidates.append("huber")
    best = None
    trials = []
    for cand in candidates:
        payload = evaluate_group_cv(
            rows,
            alpha=float(alpha),
            min_action_samples=int(min_action_samples),
            feature_names=list(feature_names or []),
            model_kind=str(cand),
            group_key="target_group",
        )
        trials.append({"model_kind": str(cand), **dict(payload or {})})
        rmse = float(payload.get("reward_rmse", np.nan))
        if not bool(payload.get("ok", False)) or (not np.isfinite(rmse)):
            continue
        if best is None or rmse < float(best.get("reward_rmse", np.inf)):
            best = {"model_kind": str(cand), **dict(payload)}
    if best is None:
        return {
            "model_kind": "ridge",
            "selection_eval": {"ok": False, "note": "selection_cv_unavailable", "trials": list(trials)},
        }
    return {"model_kind": str(best.get("model_kind", "ridge")), "selection_eval": dict(best), "trials": list(trials)}


def write_report(out_path: Path, summary: dict) -> None:
    model = dict(summary.get("model", {}) or {})
    sel = dict(summary.get("model_selection", {}) or {})
    offline_target = dict(summary.get("offline_eval_target", {}) or {})
    offline_scaffold = dict(summary.get("offline_eval_scaffold", {}) or {})
    offline_trace = dict(summary.get("offline_eval_trace", {}) or {})
    lines = [
        "# ABFE Steering Policy Training Report",
        "",
        "## Data",
        f"- trace_files: {summary.get('n_trace_files', 0)}",
        f"- train_trace_files: {summary.get('n_train_trace_files', 0)}",
        f"- trace_rows: {summary.get('n_trace_rows', 0)}",
        f"- trainable_rows: {summary.get('n_train_rows', 0)}",
        f"- excluded_trained_model_rows: {summary.get('excluded_trained_model_rows', 0)}",
        f"- target_groups: {summary.get('n_target_groups', 0)}",
        f"- scaffold_groups: {summary.get('n_scaffold_groups', 0)}",
        "",
        "## Model",
        f"- version: {model.get('version', '')}",
        f"- model_kind: {model.get('model_kind', '')}",
        f"- alpha: {model.get('alpha', np.nan)}",
        f"- score_blend_alpha: {model.get('score_blend_alpha', np.nan)}",
        f"- ood_enabled: {bool(model.get('ood_enabled', False))}",
        f"- ood_distance_threshold: {model.get('ood_distance_threshold', np.nan)}",
        f"- ood_max_abs_z: {model.get('ood_max_abs_z', np.nan)}",
        "",
        "## Model Selection",
        f"- selected_model_kind: {sel.get('model_kind', model.get('model_kind', ''))}",
        f"- selection_note: {dict(sel.get('selection_eval', {}) or {}).get('note', '')}",
        "",
        "## Offline Eval Target",
        f"- n_groups: {offline_target.get('n_groups', 0)}",
        f"- n_eval_rows: {offline_target.get('n_eval_rows', 0)}",
        f"- reward_rmse: {offline_target.get('reward_rmse', np.nan)}",
        f"- reward_mae: {offline_target.get('reward_mae', np.nan)}",
        f"- match_coverage: {offline_target.get('match_coverage', np.nan)}",
        "",
        "## Offline Eval Scaffold",
        f"- n_groups: {offline_scaffold.get('n_groups', 0)}",
        f"- n_eval_rows: {offline_scaffold.get('n_eval_rows', 0)}",
        f"- reward_rmse: {offline_scaffold.get('reward_rmse', np.nan)}",
        f"- reward_mae: {offline_scaffold.get('reward_mae', np.nan)}",
        f"- match_coverage: {offline_scaffold.get('match_coverage', np.nan)}",
        "",
        "## Offline Eval Trace",
        f"- n_groups: {offline_trace.get('n_groups', 0)}",
        f"- n_eval_rows: {offline_trace.get('n_eval_rows', 0)}",
        f"- reward_rmse: {offline_trace.get('reward_rmse', np.nan)}",
        f"- reward_mae: {offline_trace.get('reward_mae', np.nan)}",
        f"- match_coverage: {offline_trace.get('match_coverage', np.nan)}",
        "",
        "## Per-Action Samples",
    ]
    for action_name, payload in sorted(dict(model.get("actions", {}) or {}).items()):
        lines.append(
            f"- {action_name}: n_train={int(dict(payload or {}).get('n_train', 0))}, "
            f"mean_reward={dict(payload or {}).get('mean_reward', np.nan)}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_paths = resolve_trace_paths(args.trace, args.trace_glob)
    if not trace_paths:
        raise RuntimeError("No policy_trace.jsonl files found")

    rows = load_trace_rows(trace_paths)
    train_rows = [
        dict(r)
        for r in rows
        if bool(r.get("actionable", False))
        and np.isfinite(_safe_float(r.get("reward_estimate", np.nan)))
        and str(r.get("policy_action_name", "keep") or "keep") in set(ABFE_STEERING_ACTION_NAMES)
        and (bool(args.include_trained_model_runs) or (not bool(r.get("policy_model_used", False))))
    ]
    if not train_rows:
        raise RuntimeError("No trainable trace rows found")
    train_trace_files = sorted({str(r.get("trace_path", "")) for r in train_rows if str(r.get("trace_path", ""))})
    excluded_trained_rows = int(
        sum(
            1
            for r in rows
            if bool(r.get("actionable", False))
            and np.isfinite(_safe_float(r.get("reward_estimate", np.nan)))
            and str(r.get("policy_action_name", "keep") or "keep") in set(ABFE_STEERING_ACTION_NAMES)
            and (not bool(args.include_trained_model_runs))
            and bool(r.get("policy_model_used", False))
        )
    )

    feature_names = list(ABFE_STEERING_POLICY_FEATURE_NAMES)
    model_select = select_model_kind(
        train_rows,
        alpha=float(args.alpha),
        min_action_samples=int(args.min_action_samples),
        feature_names=feature_names,
        requested_kind=str(args.model_kind),
    )
    chosen_model_kind = str(model_select.get("model_kind", "ridge") or "ridge")
    models = fit_action_models(
        train_rows,
        alpha=float(args.alpha),
        min_action_samples=int(args.min_action_samples),
        feature_names=feature_names,
        model_kind=str(chosen_model_kind),
    )
    offline_eval_target = evaluate_group_cv(
        train_rows,
        alpha=float(args.alpha),
        min_action_samples=int(args.min_action_samples),
        feature_names=feature_names,
        model_kind=str(chosen_model_kind),
        group_key="target_group",
    )
    offline_eval_scaffold = evaluate_group_cv(
        train_rows,
        alpha=float(args.alpha),
        min_action_samples=int(args.min_action_samples),
        feature_names=feature_names,
        model_kind=str(chosen_model_kind),
        group_key="scaffold_group",
    )
    offline_eval_trace = evaluate_group_cv(
        train_rows,
        alpha=float(args.alpha),
        min_action_samples=int(args.min_action_samples),
        feature_names=feature_names,
        model_kind=str(chosen_model_kind),
        group_key="trace_group",
    )
    ood_profile = build_ood_profile(train_rows, ABFE_STEERING_POLICY_OOD_FEATURE_NAMES)
    target_groups = sorted({str(r.get("target_group", "")) for r in train_rows if str(r.get("target_group", ""))})
    scaffold_groups = sorted({str(r.get("scaffold_group", "")) for r in train_rows if str(r.get("scaffold_group", ""))})

    model_payload = {
        "version": f"safe_bandit_generalized_dm_v1_{time.strftime('%Y%m%d')}",
        "model_type": "linear_action_value",
        "model_kind": str(chosen_model_kind),
        "reward_centering": "per_action_mean",
        "feature_names": list(feature_names),
        "actions": dict(models),
        "score_blend_alpha": float(np.clip(float(args.blend_alpha), 0.0, 2.0)),
        "min_action_score": 0.35,
        "alpha": float(max(0.0, args.alpha)),
        "selection_cv_group": "target_group",
        "ood_enabled": bool(ood_profile.get("ok", False)),
        "ood_feature_names": list(ood_profile.get("ood_feature_names", []) or []),
        "ood_center": list(ood_profile.get("ood_center", []) or []),
        "ood_scale": list(ood_profile.get("ood_scale", []) or []),
        "ood_distance_threshold": float(ood_profile.get("ood_distance_threshold", np.nan)),
        "ood_max_abs_z": float(ood_profile.get("ood_max_abs_z", np.nan)),
        "train_rows": int(len(train_rows)),
        "n_trace_files": int(len(trace_paths)),
    }
    summary = {
        "n_trace_files": int(len(trace_paths)),
        "trace_files": [str(p) for p in trace_paths],
        "n_train_trace_files": int(len(train_trace_files)),
        "train_trace_files": list(train_trace_files),
        "n_trace_rows": int(len(rows)),
        "n_train_rows": int(len(train_rows)),
        "excluded_trained_model_rows": int(excluded_trained_rows),
        "n_target_groups": int(len(target_groups)),
        "n_scaffold_groups": int(len(scaffold_groups)),
        "target_groups": list(target_groups),
        "scaffold_groups": list(scaffold_groups),
        "model": dict(model_payload),
        "model_selection": dict(model_select),
        "offline_eval_target": dict(offline_eval_target),
        "offline_eval_scaffold": dict(offline_eval_scaffold),
        "offline_eval_trace": dict(offline_eval_trace),
    }

    model_json = out_dir / "trained_steering_policy.json"
    metrics_json = out_dir / "training_metrics.json"
    report_md = out_dir / "training_report.md"
    model_json.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_md, summary)

    print(f"Wrote model: {model_json}")
    print(f"Wrote metrics: {metrics_json}")
    print(f"Wrote report: {report_md}")
    print(f"Trace files: {len(trace_paths)}")
    print(f"Train trace files: {len(train_trace_files)}")
    print(f"Train rows: {len(train_rows)}")
    if bool(offline_eval_target.get("ok", False)):
        print(f"Target-group reward RMSE: {offline_eval_target.get('reward_rmse', np.nan):.3f}")
        print(f"Target-group match coverage: {offline_eval_target.get('match_coverage', np.nan):.3f}")
    else:
        print(f"Target-group eval: {offline_eval_target.get('note', 'not_available')}")


if __name__ == "__main__":
    main()
