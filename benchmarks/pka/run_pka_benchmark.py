"""
PM7/COSMO pKa calibration and benchmark for leonardo.pka_runtime.

Data: IUPAC Digitized pKa Dataset (high-confidence, v2.4), aqueous pKa values from
Perrin (1965, 1972) and Serjeant & Dempsey (1979). It is licensed CC BY-NC 4.0
("Reproduced by permission of International Union of Pure and Applied Chemistry"),
so it is downloaded at run time and NOT redistributed with Leonardo (MIT). Only
derived calibration parameters and aggregate statistics are written to the repo.

Selection: 20-25 C, no cosolvent, assessment Reliable/Approximate, pKa1 (acids) or
pKaH1 (bases), one ionizable site of a QM class (pka_runtime.find_ionizable_sites),
neutral single-fragment molecules with <= 30 heavy atoms of C/H/N/O/S/halogens/P,
replicate spread <= 0.3. Up to --per-class molecules per class (deterministic),
every third one (by InChIKey) held out as the test set.

Usage:
    python benchmarks/pka/run_pka_benchmark.py [--per-class 30] [--iupac-csv PATH]
"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "src"))

# pinned to the v2.4 release so the deterministic selection (and the calibration) is reproducible
IUPAC_URL = "https://raw.githubusercontent.com/IUPAC/Dissociation-Constants/v2.4/iupac_high-confidence_v2_4.csv"
IUPAC_SHA256 = "5cd5ca8bda1952be8b223bb1fbe681780de9335617f9f7c0ee2c8f8c6f01cbf3"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "leonardo", "iupac_pka")
ALLOWED_ELEMENTS = {"C", "H", "N", "O", "S", "F", "Cl", "Br", "I", "P"}


def load_iupac(path):
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {IUPAC_URL}")
        urllib.request.urlretrieve(IUPAC_URL, path)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != IUPAC_SHA256:
        raise SystemExit(f"{path}: SHA-256 {digest} does not match the pinned IUPAC v2.4 file {IUPAC_SHA256}")
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def select_molecules(rows, per_class):
    from rdkit import Chem, RDLogger

    from leonardo import pka_runtime as P

    RDLogger.DisableLog("rdApp.*")
    by_smiles = {}
    for r in rows:
        if r.get("cosolvent") or r.get("pressure") or r.get("assessment") not in ("Reliable", "Approximate"):
            continue
        try:
            t = float(r.get("T") or "nan")
            value = float(r.get("pka_value") or "nan")
        except ValueError:
            continue
        if not (20.0 <= t <= 25.0) or r.get("pka_type") not in ("pKa1", "pKaH1") or value != value:
            continue
        by_smiles.setdefault(r["SMILES"], []).append((r["pka_type"], value, r.get("ref", "")))
    ref_keys = set()
    for c in P.SITE_CLASSES:
        for form in ("acid", "base"):
            ref_keys.add(Chem.MolToInchiKey(P.neutral_parent(Chem.MolFromSmiles(c["reference"][form]))))
    picked = {}
    for smi, vals in by_smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumHeavyAtoms() > 30:
            continue
        if any(a.GetSymbol() not in ALLOWED_ELEMENTS for a in mol.GetAtoms()):
            continue
        if any(a.GetFormalCharge() != 0 for a in mol.GetAtoms()) and "[N+](=O)[O-]" not in smi:
            continue
        try:
            sites = P.find_ionizable_sites(mol)
        except Exception:
            continue
        if len(sites) != 1 or not sites[0]["qm_supported"]:
            continue
        site = sites[0]
        want = "pKa1" if site["kind"] == "acid" else "pKaH1"
        good = [v for kind, v, _ref in vals if kind == want]
        if not good or max(good) - min(good) > 0.3:
            continue
        key = Chem.MolToInchiKey(P.neutral_parent(mol))
        if key in ref_keys:
            continue
        picked.setdefault(site["class"], []).append({
            "smiles": smi, "inchikey": key, "class": site["class"], "flags": site["flags"],
            "exp": statistics.median(good), "refs": sorted({ref for kind, _v, ref in vals if kind == want and ref}),
        })
    out = []
    for cls, items in picked.items():
        items.sort(key=lambda m: hashlib.sha1(m["inchikey"].encode()).hexdigest())
        chosen = items[: int(per_class)]
        chosen.sort(key=lambda m: m["inchikey"])
        for i, m in enumerate(chosen):
            m["split"] = "test" if i % 3 == 2 else "train"
        out.extend(chosen)
    return out


def run_pm7(mols, n_conformers, workers):
    from rdkit import Chem

    from leonardo import pka_runtime as P

    backend = P.MopacBackend(calibration={"global": {"slope": 1.0, "rmse": float("nan")}, "classes": {},
                                          "version": "identity"},
                             n_conformers=n_conformers, workers=workers)
    states, meta = [], []
    for m in mols:
        parent = P.neutral_parent(Chem.MolFromSmiles(m["smiles"]))
        sites = P.find_ionizable_sites(parent)
        for prot in (True, False):
            states.append(P.build_state_molecule(parent, sites, [prot]))
        meta.append((parent, sites))
    for c in P.SITE_CLASSES:
        states += [Chem.AddHs(Chem.MolFromSmiles(c["reference"][f])) for f in ("acid", "base")]
    t0 = time.time()
    failed = 0
    try:
        backend.state_energies(states, progress=lambda msg: None)
    except P.PkaBackendError as exc:        # keep going; failing molecules are dropped below
        print(f"warning: {exc}")
    print(f"PM7 energies for {len(states)} states in {time.time() - t0:.0f}s")
    for m, (parent, sites) in zip(mols, meta):
        try:
            ev = backend.evaluate(parent, sites, [])
        except P.PkaBackendError:
            m["raw"] = None
            failed += 1
            continue
        ref = next(c for c in P.SITE_CLASSES if c["name"] == m["class"])["reference"]["pka"]
        m["ref"] = ref
        m["raw"] = ev["site_pkas"][0]["pka"]
        m["x"] = m["raw"] - ref
        m["y"] = m["exp"] - ref
    print(f"{failed} molecule(s) without converged PM7 energies")
    outliers = [m for m in mols if m.get("raw") is not None and abs(m["x"]) > P.QM_OUTLIER_LIMIT]
    print(f"{len(outliers)} PM7 outlier(s) (|x| > {P.QM_OUTLIER_LIMIT:g}; the runtime falls back to empirical): "
          + ", ".join(m["smiles"] for m in outliers))
    return [m for m in mols if m.get("raw") is not None and abs(m["x"]) <= P.QM_OUTLIER_LIMIT]


def _stats(errs):
    errs = [abs(e) for e in errs]
    if not errs:
        return {"n": 0}
    return {"n": len(errs), "mae": sum(errs) / len(errs), "rmse": (sum(e * e for e in errs) / len(errs)) ** 0.5,
            "within_1": sum(1 for e in errs if e <= 1.0) / len(errs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iupac-csv", default=os.path.join(DEFAULT_CACHE, "iupac_high-confidence_v2_4.csv"))
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--n-conformers", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--calibration-out", default=os.path.join(REPO, "src", "leonardo", "pka_calibration.json"))
    ap.add_argument("--report-out", default=os.path.join(HERE, "PM7_BENCHMARK.md"))
    ap.add_argument("--details-out", default=os.path.join(DEFAULT_CACHE, "pm7_benchmark_details.json"),
                    help="per-molecule results (contain IUPAC values; kept outside the repository)")
    ap.add_argument("--from-details", default="",
                    help="rebuild calibration and report from a previous --details-out file (no PM7 runs)")
    args = ap.parse_args()

    from leonardo import pka_runtime as P

    if args.from_details:
        with open(args.from_details, encoding="utf-8") as fh:
            mols = json.load(fh)["molecules"]
        print(f"loaded {len(mols)} molecules from {args.from_details}")
    else:
        mols = select_molecules(load_iupac(args.iupac_csv), args.per_class)
        print(f"selected {len(mols)} molecules")
        mols = run_pm7(mols, args.n_conformers, args.workers)
    train = [m for m in mols if m["split"] == "train"]
    test = [m for m in mols if m["split"] == "test"]
    def in_domain(m):
        return not (set(m["flags"]) & P.CALIBRATION_EXCLUDED_FLAGS)

    # calibration: robust (Theil-Sen) per-class slopes, shrunk towards the global slope with a
    # pseudo-count chosen by leave-one-out on the in-domain calibration molecules
    rows = [{"class": m["class"], "x": m["x"], "y": m["y"]} for m in train if in_domain(m)]
    lam = P.choose_shrinkage(rows, robust=True)
    cal = P.fit_calibration(rows, shrinkage=lam, robust=True)
    cal_global = P.fit_calibration(rows, shrinkage=1.0e9, robust=True)
    cal_class = P.fit_calibration([{"class": m["class"], "x": m["x"], "y": m["y"]} for m in train], min_class_n=5)
    # PM7 vs class-typical per class, decided on the calibration set only (leave-one-out),
    # never on the test set: classes where empirical wins are marked pm7_recommended = false
    index = {id(m): i for i, m in enumerate(train)}
    for m in train:
        others = [{"class": q["class"], "x": q["x"], "y": q["y"]} for q in train
                  if in_domain(q) and index[id(q)] != index[id(m)]]
        m["loo"] = P.apply_calibration(P.fit_calibration(others, shrinkage=lam, robust=True),
                                       m["class"], m["x"], m["ref"])[0]
    for m in mols:
        cls = next(c for c in P.SITE_CLASSES if c["name"] == m["class"])
        m["empirical"] = cls["empirical"][0]
    recommend = {}
    for name in sorted({m["class"] for m in mols}):
        sub = [m for m in train if m["class"] == name]
        pm7_mae = sum(abs(m["loo"] - m["exp"]) for m in sub) / len(sub) if sub else float("nan")
        emp_mae = sum(abs(m["empirical"] - m["exp"]) for m in sub) / len(sub) if sub else float("nan")
        recommend[name] = {"pm7_loo_mae": pm7_mae, "empirical_mae": emp_mae,
                           "pm7_recommended": bool(not sub or pm7_mae <= emp_mae)}
        if name in cal["classes"]:
            cal["classes"][name]["pm7_recommended"] = recommend[name]["pm7_recommended"]
    for m in mols:
        m["pred"] = P.apply_calibration(cal, m["class"], m["x"], m["ref"])[0]
        m["pred_global"] = P.apply_calibration(cal_global, m["class"], m["x"], m["ref"])[0]
        m["pred_class"] = P.apply_calibration(cal_class, m["class"], m["x"], m["ref"])[0]
        m["shipped"] = m["pred"] if recommend[m["class"]]["pm7_recommended"] else m["empirical"]

    def err(sub, key):
        return [m[key] - m["exp"] for m in sub]

    summary = {
        "test_uncalibrated": _stats(err(test, "raw")),
        "test_global_slope": _stats(err(test, "pred_global")),
        "test_per_class_slope": _stats(err(test, "pred_class")),
        "test_shrinkage": _stats(err(test, "pred")),
        "test_shipped_with_recommendation": _stats(err(test, "shipped")),
        "test_empirical_class_typical": _stats(err(test, "empirical")),
        "test_flagged": _stats(err([m for m in test if m["flags"]], "pred")),
        "test_unflagged": _stats(err([m for m in test if not m["flags"]], "pred")),
        "test_in_domain": _stats(err([m for m in test if in_domain(m)], "pred")),
        "test_out_of_domain": _stats(err([m for m in test if not in_domain(m)], "pred")),
    }
    per_class = {}
    for name in sorted({m["class"] for m in mols}):
        per_class[name] = {
            "n_train": sum(1 for m in train if m["class"] == name),
            "n_cal": int(cal["classes"].get(name, {}).get("n", 0)),
            "test": _stats(err([m for m in test if m["class"] == name], "pred")),
            "test_empirical": _stats(err([m for m in test if m["class"] == name], "empirical")),
            "slope": cal["classes"].get(name, cal["global"])["slope"],
            **recommend[name],
        }
    per_flag = {}
    for flag in sorted({f for m in test for f in m["flags"]}):
        per_flag[flag] = _stats(err([m for m in test if flag in m["flags"]], "pred"))

    version = time.strftime("pm7-%Y.%m.%d")
    calibration = {
        "version": version,
        "methods": {"pm7": {
            "model": "pKa = pKa_ref + slope_class * (dG_deprot - dG_deprot_ref)/(RT ln10); "
                     "PM7/COSMO(EPS=78.4) lowest heat of formation over ETKDG/MMFF conformers",
            "n_conformers": args.n_conformers,
            "shrinkage": lam,
            "robust": True,
            "excluded_flags": sorted(P.CALIBRATION_EXCLUDED_FLAGS),
            "outlier_limit": P.QM_OUTLIER_LIMIT,
            "global": cal["global"],
            "classes": cal["classes"],
            "train_n": len(rows),
            "test": summary["test_shrinkage"],
            "data": "IUPAC Digitized pKa Dataset v2.4 (CC BY-NC 4.0; not redistributed), "
                    "selection in benchmarks/pka/run_pka_benchmark.py",
            "data_sha256": IUPAC_SHA256,
        }},
    }
    with open(args.calibration_out, "w", encoding="utf-8") as fh:
        json.dump(calibration, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.makedirs(os.path.dirname(args.details_out), exist_ok=True)
    with open(args.details_out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "molecules": mols}, fh, indent=1)

    def fmt(st):
        if not st.get("n"):
            return "n=0 | - | - | -"
        return f"n={st['n']} | {st['mae']:.2f} | {st['rmse']:.2f} | {100 * st['within_1']:.0f}%"

    lines = [
        "# PM7/COSMO pKa benchmark",
        "",
        f"Calibration `{version}`, written by `benchmarks/pka/run_pka_benchmark.py`.",
        "",
        "Experimental values: IUPAC Digitized pKa Dataset v2.4 (Perrin 1965/1972, Serjeant & Dempsey",
        "1979), CC BY-NC 4.0 - \"Reproduced by permission of International Union of Pure and Applied",
        "Chemistry\". The dataset is downloaded at run time and not redistributed; this report only",
        "contains aggregate statistics.",
        "",
        f"Molecules: {len(mols)} ({len(train)} calibration, {len(test)} held-out test), "
        f"{args.n_conformers} conformers per state. Calibration: robust (Theil-Sen) per-class slope,",
        f"shrinkage pseudo-count {lam:g} (leave-one-out), fitted on {len(rows)} in-domain calibration molecules;",
        f"out-of-domain flags: {', '.join(sorted(P.CALIBRATION_EXCLUDED_FLAGS))}.",
        "",
        "## Held-out test set",
        "",
        "| Model | n | MAE | RMSE | within 1 unit |",
        "|---|---|---|---|---|",
    ]
    for key, label in (("test_uncalibrated", "PM7 uncalibrated"), ("test_global_slope", "PM7, global robust slope"),
                       ("test_per_class_slope", "PM7, per-class least squares (all molecules)"),
                       ("test_shrinkage", "PM7, robust per-class slope with shrinkage (all classes)"),
                       ("test_shipped_with_recommendation", "**Shipped: PM7, empirical where PM7 is not recommended**"),
                       ("test_empirical_class_typical", "Empirical (class-typical pKa)"),
                       ("test_in_domain", "shipped model, in-domain sites"),
                       ("test_out_of_domain", "shipped model, out-of-domain sites (warned)"),
                       ("test_flagged", "shipped model, any warning flag"),
                       ("test_unflagged", "shipped model, no warning flag")):
        lines.append(f"| {label} | {fmt(summary[key])} |")
    lines += ["", "## Per class", "",
              "`cal n`: in-domain molecules the slope was fitted on (of `train n`). PM7 vs empirical is",
              "decided on the calibration set (leave-one-out MAE), never on the test set. Classes with",
              "fewer than 5 test molecules are not statistically meaningful.", "",
              "| Class | train n | cal n | slope | LOO MAE PM7 / empirical | PM7 recommended | test PM7: n, MAE, RMSE, within 1 | test empirical MAE |",
              "|---|---|---|---|---|---|---|---|"]
    for name, row in per_class.items():
        emp = row["test_empirical"]
        lines.append(f"| {name} | {row['n_train']} | {row['n_cal']} | {row['slope']:.2f} | "
                     f"{row['pm7_loo_mae']:.2f} / {row['empirical_mae']:.2f} | {'yes' if row['pm7_recommended'] else '**no**'} | "
                     f"{fmt(row['test']).replace(' | ', ', ')} | {emp.get('mae', float('nan')):.2f} |")
    lines += ["", "## Per warning flag (test set)", "", "| Flag | n | MAE | RMSE | within 1 |", "|---|---|---|---|---|"]
    for flag, st in per_flag.items():
        lines.append(f"| {flag} | {fmt(st)} |")
    with open(args.report_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
