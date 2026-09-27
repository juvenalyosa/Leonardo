"""
PM7/COSMO and Psi4 pKa calibration and benchmark for leonardo.pka_runtime.

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
    python benchmarks/pka/run_pka_benchmark.py --method psi4 [--theory hf --basis 3-21g]
        [--compare-details ~/.cache/leonardo/iupac_pka/pm7_benchmark_details.json]

Each method has its own entry in src/leonardo/pka_calibration.json; the other entries are kept.
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


def run_qm(mols, args):
    from rdkit import Chem

    from leonardo import pka_runtime as P

    identity = {"global": {"slope": 1.0, "rmse": float("nan")}, "classes": {}, "version": "identity"}
    if args.method == "psi4":
        identity.update(theory=args.theory, basis=args.basis)
        backend = P.Psi4Backend(calibration=identity, theory=args.theory, basis=args.basis,
                                n_conformers=args.n_conformers, workers=args.workers, threads=args.threads)
    else:
        backend = P.MopacBackend(calibration=identity, n_conformers=args.n_conformers, workers=args.workers)
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
    print(f"{backend.label} energies for {len(states)} states in {time.time() - t0:.0f}s")
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
    print(f"{failed} molecule(s) without converged {args.method} energies")
    outliers = [m for m in mols if m.get("raw") is not None and abs(m["x"]) > P.QM_OUTLIER_LIMIT]
    print(f"{len(outliers)} {args.method} outlier(s) (|x| > {P.QM_OUTLIER_LIMIT:g}; the runtime falls back to empirical): "
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
    ap.add_argument("--method", choices=("pm7", "psi4"), default="pm7")
    ap.add_argument("--theory", default="hf", help="Psi4 level of theory (single points on PM7 geometries)")
    ap.add_argument("--basis", default="3-21g", help="Psi4 basis set")
    ap.add_argument("--threads", type=int, default=2, help="threads per Psi4 process")
    ap.add_argument("--n-conformers", type=int, default=10)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--calibration-out", default=os.path.join(REPO, "src", "leonardo", "pka_calibration.json"))
    ap.add_argument("--report-out", default="", help="default: benchmarks/pka/<METHOD>_BENCHMARK.md")
    ap.add_argument("--details-out", default="",
                    help="per-molecule results (contain IUPAC values; kept outside the repository); "
                         "default: <cache>/<method>_benchmark_details.json")
    ap.add_argument("--from-details", default="",
                    help="rebuild calibration and report from a previous --details-out file (no QM runs)")
    ap.add_argument("--compare-details", default="",
                    help="details file of another method (e.g. PM7) to compare on the same test molecules")
    args = ap.parse_args()
    args.theory, args.basis = args.theory.lower(), args.basis.lower()
    qm = "PM7" if args.method == "pm7" else f"Psi4 {args.theory.upper()}/{args.basis}"
    args.report_out = args.report_out or os.path.join(HERE, f"{args.method.upper()}_BENCHMARK.md")
    args.details_out = args.details_out or os.path.join(DEFAULT_CACHE, f"{args.method}_benchmark_details.json")
    rec_key = "pm7_recommended" if args.method == "pm7" else "qm_recommended"

    from leonardo import pka_runtime as P

    if args.from_details:
        with open(args.from_details, encoding="utf-8") as fh:
            mols = json.load(fh)["molecules"]
        print(f"loaded {len(mols)} molecules from {args.from_details}")
    else:
        mols = select_molecules(load_iupac(args.iupac_csv), args.per_class)
        print(f"selected {len(mols)} molecules")
        mols = run_qm(mols, args)
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
    # what the runtime uses where this method is not recommended: the class-typical pKa for PM7,
    # the shipped PM7 model for Psi4 (its leave-one-out value on the calibration set)
    fallback_name = "empirical"
    for m in mols:
        m["fallback"] = m["fallback_loo"] = m["empirical"]
    if args.method == "psi4":
        if not args.compare_details:
            raise SystemExit("--method psi4 needs --compare-details with the PM7 details file "
                             "(Psi4 falls back to PM7 per class)")
        with open(args.compare_details, encoding="utf-8") as fh:
            pm7_by_key = {m["inchikey"]: m for m in json.load(fh)["molecules"]}
        pm7_rec = {c: bool(v.get("pm7_recommended", True))
                   for c, v in ((P.load_calibration("pm7") or {}).get("classes") or {}).items()}
        fallback_name = "PM7"
        for m in mols:
            o = pm7_by_key.get(m["inchikey"])
            if o is None:
                continue
            m["fallback"] = o.get("shipped", m["empirical"])
            m["fallback_loo"] = o["loo"] if ("loo" in o and pm7_rec.get(m["class"], True)) else m["empirical"]
    recommend = {}
    for name in sorted({m["class"] for m in mols}):
        sub = [m for m in train if m["class"] == name]
        qm_mae = sum(abs(m["loo"] - m["exp"]) for m in sub) / len(sub) if sub else float("nan")
        emp_mae = sum(abs(m["empirical"] - m["exp"]) for m in sub) / len(sub) if sub else float("nan")
        fb_mae = sum(abs(m["fallback_loo"] - m["exp"]) for m in sub) / len(sub) if sub else float("nan")
        recommend[name] = {"qm_loo_mae": qm_mae, "empirical_mae": emp_mae, "fallback_mae": fb_mae,
                           "qm_recommended": bool(not sub or qm_mae <= fb_mae)}
        if name in cal["classes"]:
            cal["classes"][name][rec_key] = recommend[name]["qm_recommended"]
    for m in mols:
        m["pred"] = P.apply_calibration(cal, m["class"], m["x"], m["ref"])[0]
        m["pred_global"] = P.apply_calibration(cal_global, m["class"], m["x"], m["ref"])[0]
        m["pred_class"] = P.apply_calibration(cal_class, m["class"], m["x"], m["ref"])[0]
        m["shipped"] = m["pred"] if recommend[m["class"]]["qm_recommended"] else m["fallback"]

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

    version = time.strftime(f"{args.method}-%Y.%m.%d")
    if args.method == "psi4":
        energy_model = (f"Psi4 {args.theory}/{args.basis} single point, {P.PSI4_PCM_SOLVER} water "
                        f"(GePol UFF, area {P.PSI4_PCM_AREA:g}), at the lowest PM7/COSMO conformer "
                        "of ETKDG/MMFF conformers")
    else:
        energy_model = "PM7/COSMO(EPS=78.4) lowest heat of formation over ETKDG/MMFF conformers"
    entry = {
            "model": "pKa = pKa_ref + slope_class * (dG_deprot - dG_deprot_ref)/(RT ln10); " + energy_model,
            "version": version,
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
    }
    if args.method == "psi4":
        entry.update(theory=args.theory, basis=args.basis, pcm_solver=P.PSI4_PCM_SOLVER, pcm_area=P.PSI4_PCM_AREA)
    # one entry per method: keep the others (a PM7 run must not erase the Psi4 calibration and vice versa)
    try:
        with open(args.calibration_out, encoding="utf-8") as fh:
            calibration = json.load(fh)
    except (OSError, ValueError):
        calibration = {}
    calibration.setdefault("methods", {})[args.method] = entry
    calibration["version"] = version if args.method == "pm7" else calibration.get("version", version)
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
        "# PM7/COSMO pKa benchmark" if args.method == "pm7" else f"# {qm} (PCM water, PM7/COSMO geometries) pKa benchmark",
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
    for key, label in (("test_uncalibrated", f"{qm} uncalibrated"), ("test_global_slope", f"{qm}, global robust slope"),
                       ("test_per_class_slope", f"{qm}, per-class least squares (all molecules)"),
                       ("test_shrinkage", f"{qm}, robust per-class slope with shrinkage (all classes)"),
                       ("test_shipped_with_recommendation", f"**Shipped: {qm}, {fallback_name} where {qm} is not recommended**"),
                       ("test_empirical_class_typical", "Empirical (class-typical pKa)"),
                       ("test_in_domain", f"{qm} (all classes), in-domain sites"),
                       ("test_out_of_domain", f"{qm} (all classes), out-of-domain sites (warned)"),
                       ("test_flagged", f"{qm} (all classes), any warning flag"),
                       ("test_unflagged", f"{qm} (all classes), no warning flag")):
        lines.append(f"| {label} | {fmt(summary[key])} |")
    lines += ["", "## Per class", "",
              f"`cal n`: in-domain molecules the slope was fitted on (of `train n`). {qm} vs {fallback_name} is",
              "decided on the calibration set (leave-one-out MAE), never on the test set. Classes with",
              "fewer than 5 test molecules are not statistically meaningful.", "",
              f"| Class | train n | cal n | slope | LOO MAE {qm} / {fallback_name} | {qm} recommended | test {qm}: n, MAE, RMSE, within 1 | test empirical MAE |",
              "|---|---|---|---|---|---|---|---|"]
    for name, row in per_class.items():
        emp = row["test_empirical"]
        lines.append(f"| {name} | {row['n_train']} | {row['n_cal']} | {row['slope']:.2f} | "
                     f"{row['qm_loo_mae']:.2f} / {row['fallback_mae']:.2f} | {'yes' if row['qm_recommended'] else '**no**'} | "
                     f"{fmt(row['test']).replace(' | ', ', ')} | {emp.get('mae', float('nan')):.2f} |")
    lines += ["", "## Per warning flag (test set)", "", "| Flag | n | MAE | RMSE | within 1 |", "|---|---|---|---|---|"]
    for flag, st in per_flag.items():
        lines.append(f"| {flag} | {fmt(st)} |")
    if args.compare_details:
        with open(args.compare_details, encoding="utf-8") as fh:
            other = {m["inchikey"]: m for m in json.load(fh)["molecules"]}
        common = [m for m in test if m["inchikey"] in other and "shipped" in other[m["inchikey"]]]
        lines += ["", f"## Shipped {qm} vs the other method's shipped model (same test molecules)", "",
                  f"Other method: `{os.path.basename(args.compare_details)}`.", "",
                  f"| Class | n | {qm} MAE | other MAE |", "|---|---|---|---|"]
        for name in sorted({m["class"] for m in common}) + ["all"]:
            sub = [m for m in common if name in ("all", m["class"])]
            this_mae = sum(abs(m["shipped"] - m["exp"]) for m in sub) / len(sub)
            other_mae = sum(abs(other[m["inchikey"]]["shipped"] - m["exp"]) for m in sub) / len(sub)
            lines.append(f"| {'**all**' if name == 'all' else name} | {len(sub)} | {this_mae:.2f} | {other_mae:.2f} |")
    with open(args.report_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
