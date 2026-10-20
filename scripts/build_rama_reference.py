"""
Build Leonardo's Ramachandran reference (src/leonardo/rama_reference.npz) from high-resolution PDB chains.

    python scripts/build_rama_reference.py [--entries 2500] [--cache DIR]

The set follows the Top8000 recipe (Williams et al. 2018, MolProbity) on a smaller scale:
  * X-ray entries at <= 1.5 A resolution with R-free <= 0.21, one protein chain per 30%-identity cluster
    (RCSB search, representatives sorted by resolution);
  * residues kept only with all backbone atoms of themselves and their neighbours at full occupancy, no
    alternate location, backbone B-factors < 30 A^2, no chain break on either side;
  * six classes, as in MolProbity: general, Gly, trans-Pro, cis-Pro, pre-Pro (a residue before Pro, not Gly or
    Pro), Ile/Val.

For each class the phi/psi histogram (2 deg bins) is stored; ramachandran_runtime smooths it and sets the
"favoured" (98%) and "allowed" (99.95% general, 99.8% others; the MolProbity definitions) levels when it loads
it.  The data come from the PDB (CC0).
"""

import argparse
import concurrent.futures as cf
import json
import math
import os
import random
import urllib.request

import numpy as np

CLASSES = ("general", "glycine", "trans_pro", "cis_pro", "pre_pro", "ile_val")
BIN = 2.0

QUERY = {
    "query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": 1.5}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method", "operator": "exact_match", "value": "X-RAY DIFFRACTION"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "refine.ls_R_factor_R_free", "operator": "less_or_equal", "value": 0.21}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_entity_polymer_type", "operator": "exact_match", "value": "Protein"}},
    ]},
    "return_type": "polymer_entity",
    "request_options": {
        "group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": 30,
                     "ranking_criteria_type": {"sort_by": "rcsb_entry_info.resolution_combined",
                                               "direction": "asc"}},
        "group_by_return_type": "representatives", "paginate": {"start": 0, "rows": 10000}},
}


def representatives():
    req = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query", data=json.dumps(QUERY).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as fh:
        out = json.load(fh)
    return [r["identifier"] for r in out["result_set"]]


def fetch(entry, cache):
    path = os.path.join(cache, f"{entry}.cif.gz")
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{entry}.cif.gz", tmp)
        os.replace(tmp, path)
    return path


def classify(res, nxt, omega):
    name = res.name
    if name == "GLY":
        return "glycine"
    if name == "PRO":
        return "cis_pro" if omega is not None and abs(omega) < 30.0 else "trans_pro"
    if nxt is not None and nxt.name == "PRO":
        return "pre_pro"
    if name in ("ILE", "VAL"):
        return "ile_val"
    return "general"


def _ok_backbone(res):
    for atom_name in ("N", "CA", "C", "O"):
        atoms = [a for a in res if a.name == atom_name]
        if len(atoms) != 1 or atoms[0].has_altloc() or atoms[0].occ < 0.99 or atoms[0].b_iso >= 30.0:
            return False
    return True


def _peptide_bonded(a, b):
    c, n = a.find_atom("C", "*"), b.find_atom("N", "*")
    return c is not None and n is not None and c.pos.dist(n.pos) < 1.5


def angles_from_file(path, entity_id):
    import gemmi

    st = gemmi.read_structure(path)
    st.setup_entities()
    model = st[0]
    wanted = None
    for ent in st.entities:
        if ent.name == str(entity_id):
            wanted = set(ent.subchains)
    rows = []
    for chain in model:
        polymer = chain.get_polymer()
        if not len(polymer) or (wanted and polymer.subchain_id() not in wanted):
            continue
        for i in range(1, len(polymer) - 1):
            prev, res, nxt = polymer[i - 1], polymer[i], polymer[i + 1]
            if not gemmi.find_tabulated_residue(res.name) or not gemmi.find_tabulated_residue(res.name).is_amino_acid():
                continue
            if res.name not in ("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL"
                                .split()):
                continue
            if not (_ok_backbone(prev) and _ok_backbone(res) and _ok_backbone(nxt)):
                continue
            if not (_peptide_bonded(prev, res) and _peptide_bonded(res, nxt)):
                continue
            phi, psi = gemmi.calculate_phi_psi(prev, res, nxt)
            if math.isnan(phi) or math.isnan(psi):
                continue
            omega = math.degrees(gemmi.calculate_omega(prev, res))
            rows.append((classify(res, nxt, omega), math.degrees(phi), math.degrees(psi)))
        break                                                # one chain of the entity
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", type=int, default=2500)
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/leonardo_rama_set"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "src", "leonardo",
                                                  "rama_reference.npz"))
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    reps = representatives()
    random.Random(2018).shuffle(reps)
    reps = reps[:args.entries]
    print(f"{len(reps)} representative chains", flush=True)

    def one(rep):
        entry, entity = rep.split("_")
        try:
            return rep, angles_from_file(fetch(entry, args.cache), entity)
        except Exception as exc:
            return rep, exc

    rows, used, failed = [], [], 0
    with cf.ThreadPoolExecutor(8) as pool:
        for k, (rep, got) in enumerate(pool.map(one, reps)):
            if isinstance(got, Exception):
                failed += 1
            elif got:
                rows.extend(got)
                used.append(rep)
            if k % 200 == 0:
                print(f"  {k}/{len(reps)} chains, {len(rows)} residues", flush=True)
    out = {"classes": np.array(CLASSES), "bin_deg": np.float32(BIN), "chains": np.array(sorted(used))}
    n = int(round(360 / BIN))
    for cls in CLASSES:
        sel = [(p, s) for c, p, s in rows if c == cls]
        hist, _x, _y = np.histogram2d([p for p, _ in sel], [s for _, s in sel], bins=n,
                                      range=[[-180, 180], [-180, 180]])
        out[f"{cls}_hist"] = hist.astype(np.uint32)
        print(f"{cls:10s} {len(sel):7d} residues", flush=True)
    np.savez_compressed(os.path.abspath(args.out), **out)
    print(f"{len(used)} chains used, {failed} failed -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
