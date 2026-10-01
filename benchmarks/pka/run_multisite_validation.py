"""
Validation of coupled sites: molecules with two ionizable sites (diacids, diamines,
amino acids / zwitterions, ...) against experimental macroscopic pKa values.

None of these molecules is used by the calibration (it is fitted on single-site
molecules only), so this is an external check of the microstate treatment: the two
macroscopic pKa values come from the binding polynomial over the four microstates,
and the net charge at pH 7.4 from the microstate populations.

Data: IUPAC Digitized pKa Dataset v2.4 (CC BY-NC 4.0), downloaded at run time and not
redistributed (see run_pka_benchmark.py); only aggregate statistics are written.

Selection: 20-25 C, no cosolvent, Reliable/Approximate, both stepwise constants reported
(two of pKa1/pKa2/pKaH1/pKaH2, replicate spread <= 0.5 each), neutral single-fragment
molecules with <= 30 heavy atoms, exactly two ionizable sites and both of a QM class, and no
acidic group the module does not model (aromatic NH, amide/sulfonamide NH, enol, lactim, N-OH).

Usage:
    python benchmarks/pka/run_multisite_validation.py [--methods empirical,pm7] [--max-molecules 150]
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

from run_pka_benchmark import ALLOWED_ELEMENTS, DEFAULT_CACHE, load_iupac  # noqa: E402

PAIR_TYPES = ("pKa1", "pKa2", "pKaH1", "pKaH2")
# ionizable groups the module does not model: with one of them, the second experimental pKa may
# belong to it (e.g. a purine NH acid) rather than to the second detected site
UNMODELED_ACIDIC = ("[nH]", "[NX3;H1,H2]-[CX3,c]=[O,S]", "[NX3;H1,H2]-[SX4](=O)=O", "[OH]-[c,C;R]=,:[n,N]",
                    "[#6]=[#6]-[OH]", "[OH]-[#7]")


def select_two_site_molecules(rows, max_molecules):
    from rdkit import Chem, RDLogger

    from leonardo import pka_runtime as P

    RDLogger.DisableLog("rdApp.*")
    unmodeled = [Chem.MolFromSmarts(q) for q in UNMODELED_ACIDIC]
    by_smiles = {}
    for r in rows:
        if r.get("cosolvent") or r.get("pressure") or r.get("assessment") not in ("Reliable", "Approximate"):
            continue
        try:
            t = float(r.get("T") or "nan")
            value = float(r.get("pka_value") or "nan")
        except ValueError:
            continue
        if not (20.0 <= t <= 25.0) or r.get("pka_type") not in PAIR_TYPES or value != value:
            continue
        by_smiles.setdefault(r["SMILES"], {}).setdefault(r["pka_type"], []).append(value)
    picked = {}
    for smi, types in by_smiles.items():
        values = []
        for vals in types.values():
            if max(vals) - min(vals) <= 0.5:
                values.append(statistics.median(vals))
        if len(values) != 2 or len(types) != 2:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumHeavyAtoms() > 30:
            continue
        if any(a.GetSymbol() not in ALLOWED_ELEMENTS or a.GetFormalCharge() != 0 for a in mol.GetAtoms()):
            continue
        try:
            sites = P.find_ionizable_sites(mol)
        except Exception:
            continue
        if any(mol.HasSubstructMatch(q) for q in unmodeled):
            continue
        if len(sites) != 2 or not all(s["qm_supported"] for s in sites):
            continue
        kinds = sorted(s["kind"] for s in sites)
        category = {("acid", "acid"): "diacid", ("base", "base"): "dibase"}.get(tuple(kinds), "acid+base")
        key = Chem.MolToInchiKey(P.neutral_parent(mol))
        picked[key] = {"smiles": smi, "inchikey": key, "exp": sorted(values), "category": category,
                       "classes": sorted(s["class"] for s in sites),
                       "n_bases": sum(1 for s in sites if s["kind"] == "base"),
                       "flags": sorted({f for s in sites for f in s["flags"]})}
    mols = sorted(picked.values(), key=lambda m: hashlib.sha1(m["inchikey"].encode()).hexdigest())
    for m in mols:
        # the site-coupling parameters (pka_runtime.SITE_COUPLING) were fitted on the "fit" half
        m["split"] = "held-out" if int(hashlib.sha1(m["inchikey"].encode()).hexdigest(), 16) % 2 else "fit"
    return sorted(mols[: int(max_molecules)], key=lambda m: m["inchikey"])


def charge_from_pkas(pkas, n_bases, ph):
    """Net charge of the dominant species: +n_bases when fully protonated, -1 per pKa below the pH."""
    return int(n_bases) - sum(1 for p in pkas if p < ph)


def _stats(errs):
    errs = [abs(e) for e in errs]
    if not errs:
        return None
    return {"n": len(errs), "mae": sum(errs) / len(errs), "rmse": (sum(e * e for e in errs) / len(errs)) ** 0.5,
            "within_1": sum(1 for e in errs if e <= 1.0) / len(errs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iupac-csv", default=os.path.join(DEFAULT_CACHE, "iupac_high-confidence_v2_4.csv"))
    ap.add_argument("--methods", default="empirical,pm7")
    ap.add_argument("--max-molecules", type=int, default=100000, help="default: every eligible molecule")
    ap.add_argument("--ph", type=float, default=7.4)
    ap.add_argument("--report-out", default=os.path.join(HERE, "MULTISITE_VALIDATION.md"))
    ap.add_argument("--details-out", default=os.path.join(DEFAULT_CACHE, "multisite_details.json"))
    args = ap.parse_args()

    from rdkit import Chem

    from leonardo import pka_runtime as P

    mols = select_two_site_molecules(load_iupac(args.iupac_csv), args.max_molecules)
    print(f"selected {len(mols)} two-site molecules", flush=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    timings = {}
    for method in methods:
        t0 = time.time()
        for i, m in enumerate(mols):
            try:
                res = P.compute_pka(Chem.MolFromSmiles(m["smiles"]), method=method, focus_ph=args.ph)
            except Exception as exc:      # keep going; reported as failed
                m[method] = {"error": str(exc)}
                continue
            macro = sorted(res["macro_pkas"])
            charge = P.states_at_ph(res, args.ph, min_population=0.0)[0]["charge"]      # dominant net charge
            m[method] = {"macro": macro, "method_used": res["method"], "charge": charge,
                         "sources": sorted({s.get("pka_source", "") for s in res["sites"]})}
            if (i + 1) % 20 == 0:
                print(f"{method}: {i + 1}/{len(mols)}", flush=True)
        timings[method] = time.time() - t0
        print(f"{method}: {len(mols)} molecules in {timings[method]:.0f}s", flush=True)

    ph = args.ph
    lines = [
        "# Two-site molecules: macroscopic pKa and charge state",
        "",
        "Written by `benchmarks/pka/run_multisite_validation.py`. Experimental values: IUPAC Digitized",
        "pKa Dataset v2.4 (Perrin 1965/1972, Serjeant & Dempsey 1979), CC BY-NC 4.0 - \"Reproduced by",
        "permission of International Union of Pure and Applied Chemistry\"; downloaded at run time,",
        "aggregate statistics only.",
        "",
        f"{len(mols)} molecules with two ionizable sites of QM classes, none used for calibration (the",
        "calibration is fitted on single-site molecules). Predicted and experimental macroscopic pKa",
        "values are compared in ascending order. The experimental charge at pH "
        f"{ph:g} is +(number of basic sites) minus the number of experimental pKa values below {ph:g}.",
        "Molecules with an experimental pKa within 0.5 of that pH (mixed charge states) are left out",
        "of the charge column.",
        "",
        "The site-coupling term (`pka_runtime.SITE_COUPLING`, one strength and distance law per pair",
        "type) was fitted on half of these molecules (split by InChIKey hash); **the held-out half is",
        "the validation**, the full set is shown for reference.",
        "",
        "| Method | Molecules | Category | n | macro pKa MAE | RMSE | within 1 unit | charge at pH "
        f"{ph:g} right | time per molecule |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    summary = {}
    for method, split in [(m, sp) for m in methods for sp in ("held-out", "all")]:
        for category in ("diacid", "dibase", "acid+base", "all"):
            sub = [m for m in mols if category in ("all", m["category"]) and "macro" in m.get(method, {})
                   and len(m[method]["macro"]) == 2 and split in ("all", m["split"])]
            if not sub:
                continue
            errs = [p - e for m in sub for p, e in zip(m[method]["macro"], m["exp"])]
            st = _stats(errs)
            clear = [m for m in sub if all(abs(e - ph) > 0.5 for e in m["exp"])]
            right = sum(1 for m in clear if m[method]["charge"] == charge_from_pkas(m["exp"], m["n_bases"], ph))
            summary[f"{method}:{split}:{category}"] = dict(st, charge_right=right, charge_n=len(clear))
            label = f"**{category}**" if category == "all" else category
            lines.append(f"| {method} | {split} | {label} | {st['n'] // 2} | {st['mae']:.2f} | {st['rmse']:.2f} | "
                         f"{100 * st['within_1']:.0f}% | {right}/{len(clear)} "
                         f"({100 * right / max(1, len(clear)):.0f}%) | "
                         f"{timings[method] / max(1, len(mols)):.1f} s |")
        failed = sum(1 for m in mols if "macro" not in m.get(method, {}))
        fallback = sum(1 for m in mols if m.get(method, {}).get("method_used", method) != method)
        if split == "all" and (failed or fallback):
            lines.append(f"| {method} | all | failed / fell back to empirical | {failed} / {fallback} | | | | | |")
    # the pair of sites that matters most for drug-like molecules: amino acids and other zwitterions
    lines += ["", "Categories: diacid (two acidic sites), dibase (two basic sites), acid+base (one of each;",
              "zwitterions such as amino acids when the acid is the stronger one)."]
    with open(args.report_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(args.details_out), exist_ok=True)
    with open(args.details_out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "molecules": mols}, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
