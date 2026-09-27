"""
Fit of the site-site coupling term (leonardo.pka_runtime.SITE_COUPLING).

For each two-site molecule selected by run_multisite_validation.py, the per-site pKa with
the other site neutral comes from the method (class-typical value, or PM7/COSMO); the
coupling W(d) = a / d**power or a * exp(-(d - 1) / decay), d = topological distance
between the charge centres, is fitted per pair type (cation-cation, anion-anion,
cation-anion) on the "fit" half (InChIKey hash) by minimizing the macroscopic pKa MAE.
Evaluate on the held-out half with run_multisite_validation.py.

Usage:
    python benchmarks/pka/fit_site_coupling.py --method empirical|pm7
"""

import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

from run_multisite_validation import select_two_site_molecules  # noqa: E402
from run_pka_benchmark import DEFAULT_CACHE, load_iupac  # noqa: E402

LAWS = [("power", p) for p in (1.0, 1.5, 2.0, 2.5, 3.0)] + [("decay", lam) for lam in (1.0, 1.5, 2.0)]
PAIR = {"dibase": "++", "diacid": "--", "acid+base": "+-"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iupac-csv", default=os.path.join(DEFAULT_CACHE, "iupac_high-confidence_v2_4.csv"))
    ap.add_argument("--method", choices=("empirical", "pm7"), default="empirical")
    args = ap.parse_args()

    from rdkit import Chem

    from leonardo import pka_runtime as P

    rows = []
    backend = P.MopacBackend() if args.method == "pm7" else P.EmpiricalBackend()
    for m in select_two_site_molecules(load_iupac(args.iupac_csv), 100000):
        if m["split"] != "fit":
            continue
        parent = P.neutral_parent(Chem.MolFromSmiles(m["smiles"]))
        enum, fixed, _w = P.select_enumerated_sites(P.find_ionizable_sites(parent))
        try:
            micro = backend.evaluate(parent, enum, fixed)["micro_pka"]
        except P.PkaBackendError:
            continue
        neutral = tuple(s["kind"] == "acid" for s in enum)
        # per-site pKa with the other site neutral, i.e. before the coupling term
        base = [micro[f"{P._state_key(tuple(b if i != k else False for i, b in enumerate(neutral)))}|{k}"]
                for k in range(2)]
        dm = Chem.GetDistanceMatrix(parent)
        d = max(1.0, dm[P._charge_center(parent, enum[0])][P._charge_center(parent, enum[1])])
        rows.append({"pair": PAIR[m["category"]], "base": base, "d": d, "exp": m["exp"],
                     "kinds": [s["kind"] for s in enum]})
    print(f"{len(rows)} fit molecules ({args.method})")

    def mae(sub, w_of_d):
        errs = []
        for r in sub:
            w = w_of_d(r["d"])
            q = lambda k, prot: (1 if prot else 0) if r["kinds"][k] == "base" else (0 if prot else -1)  # noqa: E731

            def micro(part, i):
                return r["base"][i] - w * q(1 - i, part[1 - i])
            macro = P.macroscopic_pkas(P.microstate_free_energies([0, 0], micro_pka_fn=micro))
            errs += [abs(p - e) for p, e in zip(macro, r["exp"])]
        return sum(errs) / max(1, len(errs))

    for pair in ("++", "--", "+-"):
        sub = [r for r in rows if r["pair"] == pair]
        best = None
        for law, par in LAWS:
            for a in (i / 4 for i in range(0, 161)):
                if law == "power":
                    fn = (lambda a, p: lambda d: a / d ** p)(a, par)
                else:
                    fn = (lambda a, lam: lambda d: a * math.exp(-(d - 1) / lam))(a, par)
                score = mae(sub, fn)
                if best is None or score < best[0]:
                    best = (score, a, law, par)
        print(f'"{pair}": {{"a": {best[1]:g}, "{best[2]}": {best[3]:g}}},   # n={len(sub)}, fit MAE {best[0]:.2f}')


if __name__ == "__main__":
    main()
