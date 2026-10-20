"""
Check (and tune) the smoothing of the Ramachandran "allowed" region against the wwPDB validation reports.

    python scripts/calibrate_rama_reference.py [--entries 80] [--cache DIR]

Takes X-ray entries at 1.8-3.0 A (one per 30%-identity cluster, none of them in the reference set), computes
Leonardo's % Ramachandran outliers for several smoothing widths of the allowed contour, and compares them with
percent_ramachandran_outliers of the wwPDB report (MolProbity).  Prints the mean absolute error per width and the
per-entry table for the current setting.
"""

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from leonardo import ramachandran_runtime as rama  # noqa: E402

QUERY = {
    "query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined", "operator": "range",
            "value": {"from": 1.8, "to": 3.0, "include_lower": True, "include_upper": True}}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_entity_polymer_type", "operator": "exact_match", "value": "Protein"}},
    ]},
    "return_type": "entry",
    "request_options": {"group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": 30},
                        "group_by_return_type": "representatives", "paginate": {"start": 0, "rows": 3000}},
}


def candidates():
    q = json.loads(json.dumps(QUERY))
    q["return_type"] = "polymer_entity"
    req = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query", data=json.dumps(q).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as fh:
        return sorted({r["identifier"].split("_")[0] for r in json.load(fh)["result_set"]})


def wwpdb_outliers(entry):
    with urllib.request.urlopen(f"https://data.rcsb.org/rest/v1/core/entry/{entry}", timeout=60) as fh:
        d = json.load(fh)
    g = (d.get("pdbx_vrpt_summary_geometry") or [{}])[0]
    return g.get("percent_ramachandran_outliers")


def fetch(entry, cache):
    path = os.path.join(cache, f"{entry}.cif.gz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{entry}.cif.gz", path + ".part")
        os.replace(path + ".part", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", type=int, default=80)
    ap.add_argument("--offset", type=int, default=0, help="skip this many candidates (a held-out set)")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/leonardo_rama_calibration"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    ref_chains = {str(c).split("_")[0] for c in np.load(rama._REF_PATH)["chains"]}
    pool_ids = [e for e in candidates() if e not in ref_chains]
    random.Random(7).shuffle(pool_ids)

    def one(entry):
        try:
            val = wwpdb_outliers(entry)
            if val is None:
                return None
            return entry, float(val), rama.phi_psi(fetch(entry, args.cache))
        except Exception:
            return None

    data = []
    with cf.ThreadPoolExecutor(8) as ex:
        for got in ex.map(one, pool_ids[args.offset:args.offset + int(args.entries * 1.3)]):
            if got and got[2]:
                data.append(got)
    data = data[:args.entries]
    print(f"{len(data)} entries")
    for scale in (0.8, 1.0, 1.25):
        sa = {k: v * scale for k, v in rama.SIGMA_ALLOWED.items()}
        ref = rama.reference(sigma_allowed=sa)
        errs, ours_all, theirs_all = [], [], []
        for entry, theirs, rows in data:
            rows = rama.evaluate([dict(r) for r in rows], ref)
            ours = rama.summarize(rows)["outliers_pct"]
            errs.append(ours - theirs)
            ours_all.append(ours)
            theirs_all.append(theirs)
        errs = np.array(errs)
        r = np.corrcoef(ours_all, theirs_all)[0, 1]
        print(f"allowed sigma x{scale:<4}: mean error {errs.mean():+.2f}  mean |error| {np.abs(errs).mean():.2f}  "
              f"r = {r:.2f}  (mean wwPDB {np.mean(theirs_all):.2f}%, ours {np.mean(ours_all):.2f}%)")
    ref = rama.reference()
    print("\nentry   wwPDB  Leonardo")
    for entry, theirs, rows in data:
        ours = rama.summarize(rama.evaluate([dict(r) for r in rows], ref))["outliers_pct"]
        print(f"{entry}  {theirs:5.2f}  {ours:5.2f}")


if __name__ == "__main__":
    main()
