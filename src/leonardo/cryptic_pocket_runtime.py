"""
Cryptic pockets: where a pocket can open that the structure does not show.

Leonardo's own approach (physics first, a light calibration on top):

  1. Anisotropic network model (ANM) of the Calpha atoms; the lowest non-trivial modes are the cheap,
     collective motions of the fold (Zheng and co-workers showed that sampling along the ~30 lowest modes
     reshapes cryptic sites into detectable pockets).
  2. An ensemble of conformers: random combinations of those modes, weighted by 1/sqrt(eigenvalue) (thermal
     amplitudes) and scaled to a fixed Calpha RMSD; every atom of a residue moves with its Calpha.
  3. Pockets on a grid in every conformer: probe-accessible solvent points (a water-sized probe fits, at
     least ``probe_A`` from every atom centre) whose burial -- the fraction of 30 evenly spread rays that meet
     protein within ``ray_A`` -- is at least ``min_frac``; connected points form pockets, scored by volume x
     mean burial.  Each residue gets the score of the best pocket it lines (an atom within ``contact_A``).
  4. Per-residue features: that score in the structure itself and across the ensemble (how often and how
     much a pocket opens next to the residue), residue burial, hydrophobicity, normalized B-factor, mode
     mobility, and their averages over the surrounding patch.  They are combined by a logistic model with a
     handful of weights fitted on the CryptoBench training folds only (cryptic_pocket_model.json).

The grid work runs in the C++ extension ``_cryptic`` when it is built (``python -m leonardo.cryptic_build``)
and in numpy otherwise; both give the same numbers (tests compare them).
"""

import math

import numpy as np

HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "CYS", "TYR"}
AMINO = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL MSE".split())
DEFAULTS = {
    "grid_A": 1.0,
    "atom_radius_A": 1.8,        # grid cells closer than this to an atom centre are protein (rays stop there)
    "probe_A": 3.0,              # pocket points are at least this far from every atom centre
    "shell_A": 3.5,              # ... and within probe_A + shell_A of one
    "ray_A": 10.0,
    "n_rays": 30,
    "min_frac": 0.7,             # burial: fraction of rays that meet protein
    "min_volume_A3": 30.0,       # smaller clusters are not pockets
    "contact_A": 4.0,            # a residue lines a pocket when one of its atoms is this close to a point
    "anm_cutoff_A": 15.0,
    "n_modes": 30,
    "n_conformers": 24,
    "rmsd_A": 3.0,
    "seed": 7,
}


# ------------------------------------------------------------------------------------------------ structure

class Chain:
    """Heavy atoms of the protein residues of one or more chains (no waters, no hetero groups)."""

    def __init__(self, xyz, atom_res, res_keys, res_names, ca_index, bfac):
        self.xyz = np.asarray(xyz, dtype=float)
        self.atom_res = np.asarray(atom_res, dtype=int)
        self.res_keys = list(res_keys)          # "A_12", "A_12A"
        self.res_names = list(res_names)
        self.ca_index = np.asarray(ca_index, dtype=int)   # atom index of each residue's CA
        self.bfac = np.asarray(bfac, dtype=float)

    @property
    def n_res(self):
        return len(self.res_keys)


def read_chain(path, chains=None, model=0):
    """Protein heavy atoms of ``chains`` (all protein chains if None); first alternate location only."""
    import gemmi

    st = gemmi.read_structure(str(path))
    st.remove_hydrogens()
    st.remove_ligands_and_waters()
    xyz, atom_res, keys, names, ca, bf = [], [], [], [], [], []
    for ch in st[model]:
        if chains and ch.name not in chains:
            continue
        for res in ch:
            if res.name not in AMINO:
                info = gemmi.find_tabulated_residue(res.name)
                if info is None or not info.is_amino_acid():
                    continue
            atoms = []
            seen = set()
            for a in res:
                if a.name in seen:
                    continue                       # keep the first alternate location
                seen.add(a.name)
                atoms.append(a)
            ca_atoms = [a for a in atoms if a.name == "CA"]
            if not ca_atoms:
                continue
            r = len(keys)
            keys.append(f"{ch.name}_{res.seqid.num}{res.seqid.icode.strip()}")
            names.append(res.name)
            for a in atoms:
                if a.name == "CA":
                    ca.append(len(xyz))
                xyz.append((a.pos.x, a.pos.y, a.pos.z))
                atom_res.append(r)
                bf.append(a.b_iso)
    if not keys:
        raise ValueError("no protein residues with a CA atom")
    return Chain(xyz, atom_res, keys, names, ca, bf)


# ------------------------------------------------------------------------------------------------ ANM

def anm_hessian(ca_xyz, cutoff=15.0):
    """Sparse ANM Hessian (3N x 3N, unit spring constant) of the Calpha atoms within ``cutoff``."""
    from scipy import sparse
    from scipy.spatial import cKDTree

    x = np.asarray(ca_xyz, dtype=float)
    n = len(x)
    pairs = cKDTree(x).query_pairs(cutoff, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    d = x[j] - x[i]
    blocks = -d[:, :, None] * d[:, None, :] / np.einsum("ij,ij->i", d, d)[:, None, None]   # off-diagonal i-j
    rows, cols, vals = [], [], []
    a, b = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
    for (u, v) in ((i, j), (j, i)):
        rows.append((3 * u[:, None, None] + a).ravel())
        cols.append((3 * v[:, None, None] + b).ravel())
        vals.append(blocks.ravel())
    diag = np.zeros((n, 3, 3))
    np.add.at(diag, i, -blocks)
    np.add.at(diag, j, -blocks)
    rows.append((3 * np.arange(n)[:, None, None] + a).ravel())
    cols.append((3 * np.arange(n)[:, None, None] + b).ravel())
    vals.append(diag.ravel())
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * n))


def anm_modes(ca_xyz, cutoff=15.0, n_modes=30):
    """Lowest ``n_modes`` non-trivial ANM modes: (eigenvalues, vectors [n_modes, N, 3]).  Dense LAPACK up to
    1200 residues, sparse shift-invert Lanczos above (memory grows with the contacts, not N^2)."""
    hess = anm_hessian(ca_xyz, cutoff)
    n = hess.shape[0] // 3
    k = min(3 * n - 6, int(n_modes))
    if n <= 1200:
        from scipy.linalg import eigh

        vals, vecs = eigh(hess.toarray(), subset_by_index=[6, 6 + k - 1])
    else:
        from scipy.sparse.linalg import eigsh

        vals, vecs = eigsh(hess, k=k + 6, sigma=-1e-3, which="LM")
        order = np.argsort(vals)[6:6 + k]
        vals, vecs = vals[order], vecs[:, order]
    return vals, vecs.T.reshape(k, n, 3)


def conformers(chain, vals, modes, n_conf=40, rmsd=1.5, seed=7):
    """Random thermal combinations of the modes, each scaled to ``rmsd`` (A) over the Calpha atoms; all atoms of
    a residue are moved with its Calpha.  Yields coordinate arrays."""
    rng = np.random.default_rng(seed)
    amp = 1.0 / np.sqrt(np.maximum(vals, 1e-8))
    n = modes.shape[1]
    for _ in range(int(n_conf)):
        c = rng.normal(size=len(vals)) * amp
        disp = np.tensordot(c, modes, axes=1)                  # [N, 3]
        scale = rmsd / max(math.sqrt((disp ** 2).sum() / n), 1e-9)
        yield chain.xyz + (disp * scale)[chain.atom_res]


# ------------------------------------------------------------------------------------------------ pocket grid

def _grid_frame(xyz, spacing, pad):
    lo = xyz.min(axis=0) - pad
    shape = np.ceil((xyz.max(axis=0) + pad - lo) / spacing).astype(int) + 1
    return lo, shape


def _mark(xyz, lo, shape, spacing, radius):
    occ = np.zeros(shape, dtype=bool)
    r = int(math.ceil(radius / spacing))
    offs = np.array([(i, j, k) for i in range(-r, r + 1) for j in range(-r, r + 1) for k in range(-r, r + 1)
                     if (i * i + j * j + k * k) * spacing * spacing <= radius * radius])
    centre = np.floor((xyz - lo) / spacing + 0.5).astype(int)          # round half up, as the C++ lround
    for off in offs:
        idx = centre + off
        ok = np.all((idx >= 0) & (idx < shape), axis=1)
        occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]] = True
    return occ


def fibonacci_directions(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def pocket_grid_numpy(xyz, atom_res, n_res, p):
    """Numpy reference of the C++ pocket_grid (same algorithm, same outputs)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    h = p["grid_A"]
    lo, shape = _grid_frame(xyz, h, p["ray_A"] + 2.0)
    occ = _mark(xyz, lo, shape, h, p["atom_radius_A"])
    cand = _mark(xyz, lo, shape, h, p["probe_A"] + p["shell_A"]) & ~_mark(xyz, lo, shape, h, p["probe_A"])
    pts = np.argwhere(cand)
    hits = np.zeros(len(pts), dtype=int)
    for d in fibonacci_directions(int(p["n_rays"])):
        hit = np.zeros(len(pts), dtype=bool)
        for s in range(1, int(p["ray_A"] / h) + 1):
            q = np.rint(pts + d * s).astype(int)
            ok = np.all((q >= 0) & (q < shape), axis=1)
            hh = np.zeros(len(pts), dtype=bool)
            hh[ok] = occ[q[ok, 0], q[ok, 1], q[ok, 2]]
            hit |= hh
        hits += hit
    keep = hits >= int(math.ceil(p["min_frac"] * p["n_rays"] - 1e-9))
    pts, burial = pts[keep], (hits[keep] / float(p["n_rays"])).astype(np.float32)
    grid = np.zeros(shape, dtype=bool)
    grid[tuple(pts.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    label = lab[tuple(pts.T)] - 1 if len(pts) else np.zeros(0, dtype=int)
    vol = np.bincount(label, minlength=n).astype(float) * h ** 3
    bur = np.bincount(label, weights=burial, minlength=n) / np.maximum(np.bincount(label, minlength=n), 1)
    coords = lo + pts * h
    score, volume = np.zeros(n_res), np.zeros(n_res)
    if len(pts):
        tree = cKDTree(xyz)
        for c in range(n):
            if vol[c] < p["min_volume_A3"]:
                continue
            near = tree.query_ball_point(coords[label == c], p["contact_A"])
            res = {int(atom_res[i]) for lst in near for i in lst}
            for r in res:
                score[r] = max(score[r], vol[c] * bur[c])
                volume[r] += vol[c]
    return {"points": coords, "burial": burial, "label": label.astype(np.int32), "cluster_volume": vol,
            "cluster_burial": bur, "residue_score": score, "residue_volume": volume}


def have_extension():
    try:
        from . import _cryptic  # noqa: F401

        return True
    except ImportError:
        return False


def pocket_grid(xyz, atom_res, n_res, params=None, force_numpy=False):
    """Buried probe-accessible points, pockets and residue scores (C++ when built, numpy otherwise)."""
    p = dict(DEFAULTS, **(params or {}))
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    atom_res = np.ascontiguousarray(atom_res, dtype=np.int32)
    if not force_numpy:
        try:
            from . import _cryptic
        except ImportError:
            _cryptic = None
        if _cryptic is not None:
            out = _cryptic.pocket_grid(xyz, atom_res, int(n_res), spacing=float(p["grid_A"]),
                                       radius=float(p["atom_radius_A"]), probe=float(p["probe_A"]),
                                       shell=float(p["shell_A"]), ray=float(p["ray_A"]), ndir=int(p["n_rays"]),
                                       min_frac=float(p["min_frac"]), min_volume=float(p["min_volume_A3"]),
                                       contact=float(p["contact_A"]))
            return {k: np.asarray(v) for k, v in out.items()}
    return pocket_grid_numpy(xyz, atom_res, int(n_res), p)


# ------------------------------------------------------------------------------------------------ features

FEATURES = ["score", "volume", "ens_score_mean", "ens_volume_mean", "ens_open_fraction", "ens_gain",
            "burial", "hydrophobic", "bfactor_z", "nma_mobility",
            "patch_score", "patch_ens_score", "patch_ens_gain", "patch_hydrophobic"]


def contact_number(chain, radius=10.0):
    from scipy.spatial import cKDTree

    ca = chain.xyz[chain.ca_index]
    return np.array([len(v) - 1 for v in cKDTree(ca).query_ball_point(ca, radius)], dtype=float)


def patch_mean(chain, values, radius=8.0):
    """Mean of a per-residue value over the residues whose Calpha lies within ``radius`` (a pocket is a patch)."""
    from scipy.spatial import cKDTree

    ca = chain.xyz[chain.ca_index]
    return np.array([values[idx].mean() for idx in cKDTree(ca).query_ball_point(ca, radius)])


def features(chain, params=None, progress=None, force_numpy=False):
    """Per-residue features (columns FEATURES) and what the tool shows: the structure's pockets and, per
    residue, how often a pocket opened next to it in the ensemble."""
    p = dict(DEFAULTS, **(params or {}))
    n = chain.n_res
    static = pocket_grid(chain.xyz, chain.atom_res, n, p, force_numpy)
    vals, modes = anm_modes(chain.xyz[chain.ca_index], p["anm_cutoff_A"], p["n_modes"])
    ens_s, ens_v, ens_frames = [], [], []
    for k, x in enumerate(conformers(chain, vals, modes, p["n_conformers"], p["rmsd_A"], p["seed"])):
        g = pocket_grid(x, chain.atom_res, n, p, force_numpy)
        ens_s.append(g["residue_score"])
        ens_v.append(g["residue_volume"])
        keep = np.isin(g["label"], np.flatnonzero(g["cluster_volume"] >= p["min_volume_A3"]))
        ens_frames.append((x, g["points"][keep]))
        if progress:
            progress(k + 1, p["n_conformers"])
    ens_s, ens_v = np.array(ens_s), np.array(ens_v)
    s0 = static["residue_score"]
    gain = np.maximum(ens_s - s0[None, :], 0.0).mean(axis=0)
    b = np.array([chain.bfac[chain.atom_res == r].mean() for r in range(n)])
    bz = (b - b.mean()) / (b.std() + 1e-6)
    mobility = ((modes ** 2).sum(axis=2) / np.maximum(vals[:, None], 1e-8)).sum(axis=0)
    mobility = mobility / (mobility.mean() + 1e-9)
    hyd = np.array([1.0 if nm in HYDROPHOBIC else 0.0 for nm in chain.res_names])
    cols = [s0, static["residue_volume"], ens_s.mean(axis=0), ens_v.mean(axis=0), (ens_s > 0).mean(axis=0), gain,
            contact_number(chain), hyd, bz, mobility]
    cols += [patch_mean(chain, s0), patch_mean(chain, ens_s.mean(axis=0)), patch_mean(chain, gain),
             patch_mean(chain, hyd)]
    return np.column_stack(cols), {"static": static, "open_fraction": (ens_s > 0).mean(axis=0),
                                   "ens_score_mean": ens_s.mean(axis=0), "ens_volume_mean": ens_v.mean(axis=0),
                                   "gain": gain, "frames": ens_frames, "params": p}


# ------------------------------------------------------------------------------------------------ model

_MODEL = {}


def load_model(path=None):
    import json
    import os

    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cryptic_pocket_model.json")
    if path not in _MODEL:
        with open(path) as fh:
            _MODEL[path] = json.load(fh)
    return _MODEL[path]


def apply_model(model, feats):
    """Calibrated probability that each residue lines a cryptic site (logistic model on a few features)."""
    cols = [FEATURES.index(n) for n in model["features"]]
    X = np.array(feats, dtype=float)[:, cols]
    for k, n in enumerate(model["features"]):
        if n in model.get("log_features", []):
            X[:, k] = np.log1p(X[:, k])
    z = ((X - np.array(model["mean"])) / np.array(model["std"])) @ np.array(model["coef"]) + model["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------------------------------------ analysis

# residues at or above this probability are called (chosen on the CryptoBench training folds by MCC: about 10% of
# residues flagged, precision 0.25, recall 0.45 at the residue level)
THRESHOLD = 0.175
FLOOR = 0.05            # never call a residue below this probability (about the base rate)


def site_volume(xyz, atom_res, residues, points, grid_A=1.0, contact_A=4.0):
    """Pocket volume (A^3) next to a group of residues: pocket grid points within ``contact_A`` of their atoms."""
    from scipy.spatial import cKDTree

    if not len(points):
        return 0.0
    atoms = xyz[np.isin(atom_res, residues)]
    if not len(atoms):
        return 0.0
    d, _ = cKDTree(atoms).query(points, distance_upper_bound=contact_A)
    return float(np.isfinite(d).sum()) * grid_A ** 3


def sites_from_scores(chain, prob, static_volume, ensemble_volume, open_fraction, threshold=THRESHOLD,
                      link_A=6.5, radius_A=12.0, min_residues=4, static_points=None, frames=None, grid_A=1.0):
    """Group called residues into sites, best first: seeded at the most probable residue not yet used, a site
    takes the called residues connected to it (Calpha within ``link_A``) that lie within ``radius_A`` of the seed
    (about the extent of a drug-sized pocket).  Its kind compares the pocket volume it lines in the structure
    with the ensemble's: 'forms with motion' (under 60 A^3 in the structure, or at least doubles), 'grows with
    motion' (+30%), 'already open'."""
    from scipy.spatial import cKDTree

    # the global threshold, or the structure's own top 10% (about the share called on the training set) when it
    # calls fewer: probabilities scale with the protein's size, the ranking within a protein does not
    cut = min(threshold, float(np.percentile(prob, 90))) if len(prob) else threshold
    called = set(int(i) for i in np.flatnonzero(prob >= max(cut, FLOOR)))
    ca = chain.xyz[chain.ca_index]
    tree = cKDTree(ca)
    sites = []
    for seed in sorted(called, key=lambda i: -prob[i]):
        if seed not in called:
            continue
        members, frontier = {seed}, [seed]
        while frontier:
            i = frontier.pop()
            for j in tree.query_ball_point(ca[i], link_A):
                if j in called and j not in members and np.linalg.norm(ca[j] - ca[seed]) <= radius_A:
                    members.add(j)
                    frontier.append(j)
        called -= members
        if len(members) < min_residues:
            continue
        idx = np.array(sorted(members))
        if static_points is not None and frames:
            # the site's own pocket: grid points next to its residues, in the structure and in each conformer
            v0 = site_volume(chain.xyz, chain.atom_res, idx, static_points, grid_A)
            ve = float(np.mean([site_volume(x, chain.atom_res, idx, pts, grid_A) for x, pts in frames]))
        else:
            v0, ve = float(static_volume[idx].max()), float(ensemble_volume[idx].max())
        sites.append({
            "residues": [chain.res_keys[i] for i in idx],
            "residue_names": [chain.res_names[i] for i in idx],
            "n_residues": int(len(idx)),
            "score": round(float(prob[idx].sum()), 3),
            "mean_probability": round(float(prob[idx].mean()), 3),
            "pocket_volume_structure_A3": round(v0, 1),
            "pocket_volume_ensemble_A3": round(ve, 1),
            "opens_in_ensemble": round(float(open_fraction[idx].mean()), 2),
            "kind": ("forms with motion" if (v0 < 60.0 or ve >= 2.0 * v0)
                     else "grows with motion" if ve >= 1.3 * v0 else "already open"),
            "centre": [round(float(v), 2) for v in ca[idx].mean(axis=0)],
        })
    sites.sort(key=lambda s: -s["score"])
    for k, s in enumerate(sites, 1):
        s["rank"] = k
    return sites


def analyze(chain, params=None, progress=None, model=None):
    """Residue probabilities, sites and the numbers behind them for one structure (any number of chains)."""
    model = model or load_model()
    p = dict(model.get("params") or {}, **(params or {}))
    feats, extra = features(chain, p, progress=progress)
    prob = apply_model(model, feats)
    st = extra["static"]
    big = np.isin(st["label"], np.flatnonzero(st["cluster_volume"] >= extra["params"]["min_volume_A3"]))
    sites = sites_from_scores(chain, prob, st["residue_volume"], extra["ens_volume_mean"], extra["open_fraction"],
                              static_points=st["points"][big], frames=extra["frames"],
                              grid_A=extra["params"]["grid_A"])
    return {"residues": chain.res_keys, "residue_names": chain.res_names, "probability": prob,
            "open_fraction": extra["open_fraction"], "static_score": extra["static"]["residue_score"],
            "pocket_points": extra["static"]["points"], "sites": sites, "extension": have_extension(),
            "threshold": THRESHOLD}
