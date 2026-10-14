"""
Disulfide design: propose Cys/Cys pairs that can form a disulfide, build and refine each design, filter it by
disulfide geometry, and estimate how much the new cross-link stiffens the protein.

1. Candidates: residue pairs at least ``min_separation`` apart in sequence (same chain) or on different chains,
   not Gly/Pro, with Calpha–Calpha 4.0–7.5 A and Cbeta–Cbeta 3.2–5.2 A, ranked by closeness to the typical
   5.6 A / 3.8 A of natural disulfides. The window is wider than natural disulfides (4.4–6.8 / 3.45–4.5 A) on
   purpose: a pair slightly outside often forms a good disulfide after the restrained relaxation, and the strict
   geometry filters of step 3 decide.
2. Build: both residues become Cys (PDBFixer), the S–S bond is added to the topology (CYX) and the model is
   minimized with Amber14 in implicit solvent while every backbone heavy atom is restrained to its original
   position (side chains are free). Seeded, hydrogens placed on the Reference platform and minimized on the CPU
   platform with deterministic forces: the same input gives the same model.
3. Filters (natural disulfide geometry): S–S 1.95–2.15 A, both C(beta)–S–S angles 90–120 deg, |chi3| 65–115 deg,
   Calpha RMS displacement of the whole protein < 0.7 A. Passing designs are ranked by ||chi3| - 90 deg|.
4. Rigidity: isotropic Gaussian network model on the Calphas (cutoff 7.5 A, unit contact springs); the disulfide
   adds a spring 10x stiffer between the two Calphas. Reported: the change in the relative motion of the two
   residues and in the mean fluctuation of every residue. A structural/rigidity model, not a measured stability.

Run as ``python -m leonardo.disulfide_runtime args.json`` (writes ``result_json``) or call :func:`design`.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

CA_RANGE = (4.0, 7.5)
CB_RANGE = (3.2, 5.2)
CA_IDEAL, CB_IDEAL = 5.6, 3.8
FILTERS = {"ss_A": (1.95, 2.15), "css_deg": (90.0, 120.0), "abs_chi3_deg": (65.0, 115.0), "ca_rmsd_max_A": 0.7}
GNM_CUTOFF_A, SS_SPRING = 7.5, 10.0
_SKIP = {"GLY", "PRO"}
_AA = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
       "SER", "THR", "TRP", "TYR", "VAL"}


def _progress(text):
    print(f"PROGRESS {text}", flush=True)


def _residues(pdb_path):
    """[(chain, resi_str, resn, {atom: xyz})] of standard amino acids, in file order (first altloc only)."""
    out, index = [], {}
    with open(pdb_path) as fh:
        for ln in fh:
            if not ln.startswith(("ATOM", "HETATM")) or ln[16] not in (" ", "A"):
                continue
            resn = ln[17:20].strip()
            if resn not in _AA:
                continue
            key = (ln[21].strip(), ln[22:27].strip())
            if key not in index:
                index[key] = len(out)
                out.append((key[0], key[1], resn, {}))
            out[index[key]][3][ln[12:16].strip()] = np.array([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return out


def find_candidates(pdb_path, chain="", min_separation=3, max_candidates=20):
    res = [r for r in _residues(pdb_path) if not chain or r[0] == chain]
    cands = []
    for a in range(len(res)):
        for b in range(a + 1, len(res)):
            (ca_, ia, na, xa), (cb_, ib, nb, xb) = res[a], res[b]
            if na in _SKIP or nb in _SKIP or not all(k in x for x in (xa, xb) for k in ("CA", "CB")):
                continue
            if na == "CYS" and nb == "CYS":
                continue                                  # already a (possible) disulfide
            if ca_ == cb_ and abs(b - a) < int(min_separation):
                continue
            d_ca = float(np.linalg.norm(xa["CA"] - xb["CA"]))
            d_cb = float(np.linalg.norm(xa["CB"] - xb["CB"]))
            if CA_RANGE[0] <= d_ca <= CA_RANGE[1] and CB_RANGE[0] <= d_cb <= CB_RANGE[1]:
                score = abs(d_cb - CB_IDEAL) / 0.3 + abs(d_ca - CA_IDEAL) / 0.6
                cands.append({"residues": [[ca_, ia, na], [cb_, ib, nb]], "label": f"{na[0]}{ia}C/{nb[0]}{ib}C",
                              "ca_ca_A": round(d_ca, 3), "cb_cb_A": round(d_cb, 3), "geometry_score": round(score, 3)})
    cands.sort(key=lambda c: c["geometry_score"])
    for c in cands:
        c["label"] = _label(c["residues"])
    return cands[:int(max_candidates)]


_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H",
        "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V"}


def _label(residues):
    return "/".join(f"{_ONE.get(n, 'X')}{i}C" for _c, i, n in residues)


def _torsion(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _angle(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def build_design(pdb_path, residues, out_pdb, seed=20260915, k_backbone=1000.0, max_iterations=3000):
    """Both residues -> Cys, S–S bond, restrained Amber14/OBC2 minimization. Returns the refined geometry."""
    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer

    random.seed(int(seed))
    # PDBFixer relaxes the new side-chain atoms on its own platform: Reference keeps that step reproducible
    fixer = PDBFixer(filename=pdb_path, platform=mm.Platform.getPlatformByName("Reference"))
    fixer.removeHeterogens(keepWater=False)
    by_chain = {}
    for ch, resi, resn in residues:
        by_chain.setdefault(ch, []).append(f"{resn}-{resi}-CYS")
    for ch, muts in by_chain.items():
        fixer.applyMutations(muts, ch)
    fixer.findMissingResidues()
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=int(seed))
    top, pos = fixer.topology, fixer.positions
    want = {(str(ch), str(resi)) for ch, resi, _n in residues}
    sg = [a for a in top.atoms() if a.name == "SG" and (a.residue.chain.id, str(a.residue.id)) in want]
    if len(sg) != 2:
        raise ValueError(f"expected 2 SG atoms after the mutation, found {len(sg)}")
    top.addBond(sg[0], sg[1])
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    modeller = app.Modeller(top, pos)
    random.seed(int(seed))
    platform = mm.Platform.getPlatformByName("Reference")
    try:
        modeller.addHydrogens(ff, pH=7.0, platform=platform)
    except TypeError:
        modeller.addHydrogens(ff, pH=7.0)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", float(k_backbone))
    for p in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(p)
    start = modeller.positions
    for atom in modeller.topology.atoms():
        if atom.name in ("N", "CA", "C", "O"):
            restraint.addParticle(atom.index, start[atom.index].value_in_unit(unit.nanometer))
    system.addForce(restraint)
    try:
        cpu = mm.Platform.getPlatformByName("CPU")
        ctx = mm.Context(system, mm.VerletIntegrator(0.001), cpu, {"DeterministicForces": "true", "Threads": "1"})
    except Exception:
        ctx = mm.Context(system, mm.VerletIntegrator(0.001), platform)
    ctx.setPositions(start)
    mm.LocalEnergyMinimizer.minimize(ctx, 1.0, int(max_iterations))
    state = ctx.getState(getPositions=True, getEnergy=True)
    xyz = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    x0 = np.array([[v.value_in_unit(unit.angstrom) for v in p] for p in start])
    with open(out_pdb, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, state.getPositions(), fh, keepIds=True)
    atoms = {(a.residue.chain.id, str(a.residue.id), a.name): a.index for a in modeller.topology.atoms()}
    (ch1, r1, _n1), (ch2, r2, _n2) = residues
    g = {k: xyz[atoms[(c, r, k2)]] for k, (c, r, k2) in {
        "ca1": (ch1, r1, "CA"), "cb1": (ch1, r1, "CB"), "sg1": (ch1, r1, "SG"),
        "ca2": (ch2, r2, "CA"), "cb2": (ch2, r2, "CB"), "sg2": (ch2, r2, "SG")}.items()}
    ca = [a.index for a in modeller.topology.atoms() if a.name == "CA"]
    ca_rmsd = float(np.sqrt(np.mean(np.sum((xyz[ca] - x0[ca]) ** 2, axis=1))))
    geom = {"ss_A": float(np.linalg.norm(g["sg1"] - g["sg2"])),
            "css_deg": [_angle(g["cb1"], g["sg1"], g["sg2"]), _angle(g["sg1"], g["sg2"], g["cb2"])],
            "chi3_deg": _torsion(g["cb1"], g["sg1"], g["sg2"], g["cb2"]),
            "ca_rmsd_A": ca_rmsd, "model_pdb": out_pdb,
            "minimized_energy_kj_mol": float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))}
    return geom


def passes(geom, filters=FILTERS):
    checks = {
        "ss": filters["ss_A"][0] <= geom["ss_A"] <= filters["ss_A"][1],
        "css": all(filters["css_deg"][0] <= v <= filters["css_deg"][1] for v in geom["css_deg"]),
        "chi3": filters["abs_chi3_deg"][0] <= abs(geom["chi3_deg"]) <= filters["abs_chi3_deg"][1],
        "ca_shift": geom["ca_rmsd_A"] < filters["ca_rmsd_max_A"],
    }
    return all(checks.values()), checks


def gnm_rigidity(ca_xyz, pair, cutoff=GNM_CUTOFF_A, ss_spring=SS_SPRING):
    """Gaussian network model with and without an extra spring between residues ``pair`` (indices into ca_xyz).
    Returns the relative motion of the pair, <(dR_i - dR_j)^2>, and per-residue mean-square fluctuations."""
    x = np.asarray(ca_xyz, dtype=float)
    d = np.linalg.norm(x[:, None] - x[None], axis=2)
    contact = ((d < float(cutoff)) & (d > 0)).astype(float)

    def solve(extra):
        k = -contact.copy()
        i, j = pair
        k[i, j] -= extra
        k[j, i] -= extra
        np.fill_diagonal(k, 0.0)
        np.fill_diagonal(k, -k.sum(axis=1))
        g = np.linalg.pinv(k)                         # one zero mode (rigid translation) is projected out
        return g

    g0, g1 = solve(0.0), solve(float(ss_spring))
    i, j = pair
    rel0 = g0[i, i] + g0[j, j] - 2 * g0[i, j]
    rel1 = g1[i, i] + g1[j, j] - 2 * g1[i, j]
    fl0, fl1 = np.diag(g0), np.diag(g1)
    return {"pair_relative_motion_change_pct": float(100.0 * (math.sqrt(rel1) - math.sqrt(rel0)) / math.sqrt(rel0)),
            "mean_fluctuation_change_pct": float(100.0 * (np.mean(np.sqrt(fl1)) - np.mean(np.sqrt(fl0)))
                                                 / np.mean(np.sqrt(fl0))),
            "per_residue_fluctuation_ratio": [float(v) for v in np.sqrt(fl1 / fl0)],
            "cutoff_A": float(cutoff), "ss_spring": float(ss_spring)}


def design(pdb_path, out_dir, chain="", max_candidates=5, min_separation=3, seed=20260915, progress=_progress):
    os.makedirs(out_dir, exist_ok=True)
    cands = find_candidates(pdb_path, chain=chain, min_separation=min_separation, max_candidates=max(20, max_candidates))
    if not cands:
        return {"ok": True, "candidates": [], "designs": [], "best": None,
                "note": "no residue pair has disulfide-compatible backbone geometry"}
    res = _residues(pdb_path)
    ca_keys = [(c, i) for c, i, _n, x in res if "CA" in x and (not chain or c == chain)]
    ca_xyz = [x["CA"] for c, i, _n, x in res if "CA" in x and (not chain or c == chain)]
    designs = []
    for n, cand in enumerate(cands[:int(max_candidates)], start=1):
        progress(f"design {n}/{min(len(cands), int(max_candidates))}: {cand['label']}")
        out_pdb = os.path.join(out_dir, f"design_{n:02d}_{cand['label'].replace('/', '_')}.pdb")
        row = dict(cand)
        try:
            geom = build_design(pdb_path, cand["residues"], out_pdb, seed=seed)
            ok, checks = passes(geom)
            row.update(geom, passes=ok, checks=checks)
            pair = (ca_keys.index((cand["residues"][0][0], cand["residues"][0][1])),
                    ca_keys.index((cand["residues"][1][0], cand["residues"][1][1])))
            row["rigidity"] = {k: v for k, v in gnm_rigidity(ca_xyz, pair).items()
                               if k != "per_residue_fluctuation_ratio"}
        except Exception as exc:                          # one failed build must not hide the others
            row.update(passes=False, error=f"{type(exc).__name__}: {exc}")
        designs.append(row)
    accepted = [d for d in designs if d.get("passes")]
    best = min(accepted, key=lambda d: abs(abs(d["chi3_deg"]) - 90.0)) if accepted else None
    return {"ok": True, "candidates": cands, "designs": designs, "best": best, "filters": FILTERS,
            "n_passing": len(accepted), "seed": int(seed),
            "note": "structural design and rigidity model; not a measured thermal stability"}


def main(args_path):
    args = json.load(open(args_path))
    try:
        out = design(args["pdb"], args["out_dir"], chain=args.get("chain", ""),
                     max_candidates=int(args.get("max_candidates", 5)), min_separation=int(args.get("min_separation", 3)),
                     seed=int(args.get("seed", 20260915)))
    except Exception as exc:
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    with open(args["result_json"], "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
