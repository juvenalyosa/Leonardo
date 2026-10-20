"""
Leonardo's cryptic-pocket method on the CryptoBench benchmark (Skrhak et al., Bioinformatics 2025).

    python scripts/benchmark_cryptobench.py features --split test [--limit 30] [--data ~/leonardo_cryptic_data]
    python scripts/benchmark_cryptobench.py evaluate

Protocol (as CryptoBench): for each apo chain, every observed residue is scored; a residue is positive when it
is in the apo_pocket_selection of any of its holo structures.  AUC / AUPRC are computed over all residues of
the split pooled together (also reported: the mean over structures).  Features are cached per structure
(features/<apo><chain>.npz) so the scoring can be refitted without recomputing the ensembles.
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def entries(data, split):
    d = json.load(open(os.path.join(data, "dataset.json")))
    folds = json.load(open(os.path.join(data, "folds.json")))
    ids = folds[split] if split != "train" else sum((folds[f"train-{i}"] for i in range(4)), [])
    out = []
    for apo in ids:
        holos = d[apo]
        chain = holos[0]["apo_chain"]
        pos = set()
        for h in holos:
            if h["apo_chain"] == chain:
                pos |= set(h["apo_pocket_selection"])
        out.append((apo, chain, sorted(pos)))
    return out


def one(args):
    apo, chain, pos, data, out_dir, params = args
    from leonardo import cryptic_pocket_runtime as cp

    path = os.path.join(out_dir, f"{apo}{chain}.npz")
    if os.path.exists(path):
        return apo, "cached", 0.0
    t0 = time.time()
    try:
        ch = cp.read_chain(os.path.join(data, "cif", f"{apo}.cif.gz"), chains=chain.split("-"))   # "B-C": two chains
        if ch.n_res > 1500:
            return apo, "too large", 0.0
        f, _extra = cp.features(ch, params)
        y = np.array([1 if k in set(pos) else 0 for k in ch.res_keys], dtype=np.int8)
        np.savez_compressed(path, features=f, y=y, keys=np.array(ch.res_keys),
                            names=np.array(ch.res_names), ca=ch.xyz[ch.ca_index])
        return apo, "ok", time.time() - t0
    except Exception as exc:
        return apo, f"error {type(exc).__name__}: {exc}", 0.0


def cmd_features(a):
    out_dir = os.path.join(a.data, "features" + (f"_{a.tag}" if a.tag else ""))
    os.makedirs(out_dir, exist_ok=True)
    todo = entries(a.data, a.split)[: a.limit or None]
    params = json.loads(a.params) if a.params else None
    jobs = [(apo, ch, pos, a.data, out_dir, params) for apo, ch, pos in todo]
    t0 = time.time()
    with cf.ProcessPoolExecutor(a.workers) as ex:
        for k, (apo, status, sec) in enumerate(ex.map(one, jobs), 1):
            if status != "cached":
                print(f"{k}/{len(jobs)} {apo} {status} {sec:.1f}s", flush=True)
    print(f"done in {time.time() - t0:.0f} s")


def load(data, split, tag=""):
    out_dir = os.path.join(data, "features" + (f"_{tag}" if tag else ""))
    X, Y, G = [], [], []
    for apo, ch, _pos in entries(data, split):
        path = os.path.join(out_dir, f"{apo}{ch}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        X.append(z["features"])
        Y.append(z["y"])
        G.append(apo)
    return X, Y, G


def metrics(scores, ys):
    from sklearn.metrics import average_precision_score, roc_auc_score

    s, y = np.concatenate(scores), np.concatenate(ys)
    per = [roc_auc_score(yy, ss) for ss, yy in zip(scores, ys) if 0 < yy.sum() < len(yy)]
    return {"auc": round(float(roc_auc_score(y, s)), 3), "auprc": round(float(average_precision_score(y, s)), 3),
            "auc_per_structure": round(float(np.mean(per)), 3), "structures": len(scores), "residues": int(len(y))}


def cmd_evaluate(a):
    from leonardo import cryptic_pocket_runtime as cp

    X, Y, _ = load(a.data, a.split, a.tag)
    print(f"{a.split}: {len(X)} structures")
    for j, name in enumerate(cp.FEATURES):
        print(f"  {name:14s}", metrics([x[:, j] for x in X], Y))


LOG_FEATURES = {"score", "volume", "ens_score_mean", "ens_volume_mean", "ens_gain", "patch_score", "patch_ens_score",
                "patch_ens_gain"}


def transform(X, names):
    X = np.array(X, dtype=float)
    for j, n in enumerate(names):
        if n in LOG_FEATURES:
            X[:, j] = np.log1p(X[:, j])
    return X


def cmd_fit(a):
    """Forward selection (<= 6 features) by 4-fold CV on the training folds, then a logistic model on all of
    them, written to src/leonardo/cryptic_pocket_model.json.  The test split is not read."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from leonardo import cryptic_pocket_runtime as cp

    names = cp.FEATURES
    folds = []
    for k in range(4):
        X, Y, _ = load(a.data, f"train-{k}", a.tag)
        folds.append((np.vstack([transform(x, names) for x in X]), np.concatenate(Y)))

    def fit(Xtr, ytr):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        return mu, sd, LogisticRegression(max_iter=5000).fit((Xtr - mu) / sd, ytr)

    def cv(cols):
        aucs = []
        for k in range(4):
            Xtr = np.vstack([f[0][:, cols] for j, f in enumerate(folds) if j != k])
            ytr = np.concatenate([f[1] for j, f in enumerate(folds) if j != k])
            mu, sd, m = fit(Xtr, ytr)
            aucs.append(roc_auc_score(folds[k][1], m.predict_proba((folds[k][0][:, cols] - mu) / sd)[:, 1]))
        return float(np.mean(aucs))

    sel, best = [], 0.0
    for _ in range(6):
        score, j = max((cv(sel + [j]), j) for j in range(len(names)) if j not in sel)
        print(f"  + {names[j]:18s} CV AUC {score:.4f}")
        if score < best + 0.002:
            break
        sel.append(j)
        best = score
    X = np.vstack([f[0] for f in folds])
    y = np.concatenate([f[1] for f in folds])
    mu, sd, m = fit(X[:, sel], y)
    model = {"features": [names[j] for j in sel], "log_features": sorted(LOG_FEATURES & {names[j] for j in sel}),
             "mean": mu.tolist(), "std": sd.tolist(), "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]),
             "cv_auc_train": round(best, 4), "params": dict(cp.DEFAULTS, **(json.loads(a.params) if a.params else {})),
             "trained_on": "CryptoBench train folds 0-3 (Skrhak et al., Bioinformatics 2025)"}
    out = os.path.join(os.path.dirname(cp.__file__), "cryptic_pocket_model.json")
    with open(out, "w") as fh:
        json.dump(model, fh, indent=1)
    print("model", {k: model[k] for k in ("features", "coef", "cv_auc_train")}, "->", out)


def cmd_test(a):
    """The test split, once: the physics score alone, the calibrated model, and the static pocket finder."""
    from leonardo import cryptic_pocket_runtime as cp

    X, Y, G = load(a.data, "test", a.tag)
    names = cp.FEATURES
    model = cp.load_model()
    rows = {"static pockets (no dynamics)": [x[:, names.index("score")] for x in X],
            "ensemble pocket score (physics only)": [x[:, names.index("ens_score_mean")] for x in X],
            "Leonardo cryptic score (calibrated)": [cp.apply_model(model, x) for x in X]}
    report = {name: metrics(s, Y) for name, s in rows.items()}
    for name, m in report.items():
        print(f"  {name:38s} {m}")
    with open(os.path.join(a.data, f"test_report{'_' + a.tag if a.tag else ''}.json"), "w") as fh:
        json.dump(report, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["features", "evaluate", "fit", "test"])
    ap.add_argument("--data", default=os.path.expanduser("~/leonardo_cryptic_data"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--params", default="", help="JSON overriding cryptic_pocket_runtime.DEFAULTS")
    ap.add_argument("--tag", default="", help="feature folder suffix (one per parameter set)")
    a = ap.parse_args()
    {"features": cmd_features, "evaluate": cmd_evaluate, "fit": cmd_fit, "test": cmd_test}[a.command](a)


if __name__ == "__main__":
    main()
