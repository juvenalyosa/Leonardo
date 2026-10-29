"""Per-residue physical descriptors added in Crypsis 0.2 (columns DESCRIPTORS and ENSEMBLE).

DESCRIPTORS -- what the pocket ensemble of core.features does not see:

  * solvent exposure: relative SASA of the residue and of its side chain (Shrake-Rupley), half-sphere exposure
    (the protein atoms in front of / behind the Calpha->Cbeta direction), contact numbers at several radii,
    distance to the centroid relative to the radius of gyration, and the hydrophobic fraction of the exposed
    surface around the residue;
  * local backbone geometry: the Calpha virtual bond angle and dihedral (helix / strand / loop) and the distance,
    in residues, to a chain end or break;
  * Gaussian network model: fluctuations in the slowest and in the fastest modes, and the distance to the nearest
    hinge of the two slowest modes (contacts whose two residues move in opposite directions) -- cryptic sites
    often sit where domains or subdomains move against each other;
  * side-chain flexibility: the pocket grid recomputed with the side chains cut back to Cbeta and to Cgamma (a
    pocket that a few side chains hide opens when they move).

ENSEMBLE -- two more normal-mode ensembles: larger motions (5 A Calpha RMSD instead of 3 A) and the 3 A ensemble
with the side chains cut back to Cbeta (backbone and side-chain motion together).

Tried on the CryptoBench training folds and left out because they did not help (cross-validated AUC and AUPRC
within 0.001): the side-chain minus backbone B-factor, a 1.5 A and a 7.5 A ensemble, displacements along single
slow modes, pockets opened by moving one side chain at a time, and hydrophobic / polar probe mapping of the
pockets.  See BENCHMARK.md.
"""

import numpy as np

from . import core

DESCRIPTORS = ["rsasa", "rsasa_sidechain", "hse_up", "hse_down", "cn6", "cn12", "cn16", "centroid_rg",
               "patch_hydrophobic_sasa", "ca_angle", "ca_dihedral_sin", "ca_dihedral_cos", "end_distance",
               "gnm_slow", "gnm_fast", "hinge_distance",
               "trim_cb_score", "trim_cb_volume", "trim_cb_gain", "trim_cg_score", "trim_cg_gain"]
ENSEMBLE = ["amp50_score", "amp50_open", "amp50_gain", "trim_ens_score", "trim_ens_open", "trim_ens_gain"]

RADII = {"C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "SE": 1.9}
MAX_SASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104,
            "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159, "SER": 155,
            "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
BACKBONE = {"N", "CA", "C", "O", "OXT"}
KEEP_CB = BACKBONE | {"CB"}
KEEP_CG = KEEP_CB | {"CG", "CG1", "CG2", "OG", "OG1", "SG", "SD"}


def _element(name):
    name = name.strip().upper()
    return "SE" if name.startswith("SE") else name[:1]


def atom_sasa(xyz, elements, probe=1.4, n_points=96):
    """Shrake-Rupley solvent-accessible surface area of every atom (A^2)."""
    from scipy.spatial import cKDTree

    r = np.array([RADII.get(e, 1.8) for e in elements]) + probe
    sphere = core.fibonacci_directions(n_points)
    tree = cKDTree(xyz)
    neigh = tree.query_ball_point(xyz, r + r.max())
    out = np.zeros(len(xyz))
    for i, nb in enumerate(neigh):
        nb = [j for j in nb if j != i]
        pts = xyz[i] + r[i] * sphere
        if nb:
            d2 = ((pts[:, None, :] - xyz[nb][None, :, :]) ** 2).sum(-1)
            free = ~(d2 < r[nb][None, :] ** 2).any(1)
        else:
            free = np.ones(len(pts), dtype=bool)
        out[i] = 4 * np.pi * r[i] ** 2 * free.mean()
    return out


def _virtual_cb(chain):
    """Cbeta of every residue (built from N, CA, C for glycine and for residues without one)."""
    ca = chain.xyz[chain.ca_index]
    cb = ca.copy()
    for r in range(chain.n_res):
        idx = np.flatnonzero(chain.atom_res == r)
        names = {str(chain.atom_names[i]): i for i in idx}
        if "CB" in names:
            cb[r] = chain.xyz[names["CB"]]
        elif "N" in names and "C" in names:
            b = ca[r] - chain.xyz[names["N"]]
            c = chain.xyz[names["C"]] - ca[r]
            a = np.cross(b, c)
            cb[r] = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca[r]
    return cb


def _backbone_geometry(ca):
    """Calpha virtual angle (i-1, i, i+1), virtual dihedral (i-1, i, i+1, i+2) and distance to a chain end or break."""
    n = len(ca)
    step = np.linalg.norm(np.diff(ca, axis=0), axis=1) if n > 1 else np.zeros(0)
    brk = np.flatnonzero(step > 4.2)                       # a gap between residue k and k+1
    seg_start = np.concatenate([[0], brk + 1])
    seg_end = np.concatenate([brk, [n - 1]])
    end = np.zeros(n)
    for s, e in zip(seg_start, seg_end):
        k = np.arange(s, e + 1)
        end[k] = np.minimum(k - s, e - k)
    connected = np.ones(n + 1, dtype=bool)
    connected[brk + 1] = False                            # connected[k]: residues k-1 and k are bonded
    ang = np.full(n, 120.0)
    dih_s, dih_c = np.zeros(n), np.zeros(n)
    for i in range(1, n - 1):
        if connected[i] and connected[i + 1]:
            u, v = ca[i - 1] - ca[i], ca[i + 1] - ca[i]
            ang[i] = np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
            if i + 2 < n and connected[i + 2]:
                b0, b1, b2 = ca[i] - ca[i - 1], ca[i + 1] - ca[i], ca[i + 2] - ca[i + 1]
                n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
                m1 = np.cross(n1, b1 / np.linalg.norm(b1))
                x, y = n1 @ n2, m1 @ n2
                h = np.hypot(x, y) + 1e-9
                dih_s[i], dih_c[i] = y / h, x / h
    return ang, dih_s, dih_c, np.minimum(end, 20.0)


def gnm(ca, cutoff=7.3, n_slow=3, n_fast=10):
    """Gaussian network model: slow-mode and fast-mode fluctuations (each normalized to mean 1) and the distance
    from every residue to the nearest hinge contact of the slowest modes."""
    from scipy.linalg import eigh
    from scipy.spatial import cKDTree

    n = len(ca)
    pairs = cKDTree(ca).query_pairs(cutoff, output_type="ndarray")
    K = np.zeros((n, n))
    K[pairs[:, 0], pairs[:, 1]] = K[pairs[:, 1], pairs[:, 0]] = -1.0
    K[np.arange(n), np.arange(n)] = -K.sum(1)
    vals, vecs = eigh(K)
    nz = vals > 1e-6 * max(vals.max(), 1.0)
    vals, vecs = vals[nz], vecs[:, nz]
    k = min(n_slow, len(vals))
    slow = (vecs[:, :k] ** 2 / vals[:k]).sum(1)
    f = min(n_fast, len(vals))
    fast = (vecs[:, -f:] ** 2 / vals[-f:]).sum(1)
    hinge = np.zeros(n, dtype=bool)
    for m in range(min(2, len(vals))):
        u = vecs[:, m]
        opp = np.sign(u[pairs[:, 0]]) != np.sign(u[pairs[:, 1]])
        hinge[pairs[opp, 0]] = hinge[pairs[opp, 1]] = True
    if hinge.any():
        dist = cKDTree(ca[hinge]).query(ca)[0]
    else:
        dist = np.full(n, 30.0)
    return slow / (slow.mean() + 1e-12), fast / (fast.mean() + 1e-12), np.minimum(dist, 30.0)


def _trimmed_pockets(chain, keep, params):
    mask = np.array([str(a) in keep for a in chain.atom_names]) | (chain.atom_res < 0)
    g = core.pocket_grid(chain.xyz[mask], chain.atom_res[mask], chain.n_res, params)
    return g["residue_score"], g["residue_volume"]


def descriptors(chain, static_score=None, params=None):
    """[n_res, len(DESCRIPTORS)] descriptors of one structure (``static_score``: core.features column "score",
    recomputed when not given)."""
    from scipy.spatial import cKDTree

    p = dict(core.DEFAULTS, **(params or {}))
    n = chain.n_res
    prot = chain.atom_res >= 0
    xyz, res = chain.xyz[prot], chain.atom_res[prot]
    names = np.array([str(a) for a in chain.atom_names[prot]])
    elements = [_element(a) for a in names]
    sasa = atom_sasa(xyz, elements)
    side = ~np.isin(names, list(BACKBONE))
    res_sasa = np.bincount(res, weights=sasa, minlength=n)
    side_sasa = np.bincount(res, weights=sasa * side, minlength=n)
    maxs = np.array([MAX_SASA.get(r, 200.0) for r in chain.res_names])
    max_side = np.maximum(maxs - 75.0, 20.0)               # ~75 A^2 of a residue's maximum is its backbone
    rsasa, rside = res_sasa / maxs, side_sasa / max_side

    ca = chain.xyz[chain.ca_index]
    cb = _virtual_cb(chain)
    tree = cKDTree(ca)
    up, down = np.zeros(n), np.zeros(n)
    for i, nb in enumerate(tree.query_ball_point(ca, 13.0)):
        nb = [j for j in nb if j != i]
        if not nb:
            continue
        d = ca[nb] - ca[i]
        u = cb[i] - ca[i]
        side_up = d @ u > 0
        up[i], down[i] = side_up.sum(), (~side_up).sum()
    cn = {r: np.array([len(v) - 1 for v in tree.query_ball_point(ca, r)], dtype=float) for r in (6.0, 12.0, 16.0)}
    rg = np.sqrt(((ca - ca.mean(0)) ** 2).sum(1).mean())
    cent = np.linalg.norm(ca - ca.mean(0), axis=1) / (rg + 1e-9)

    carbon = np.array([e == "C" for e in elements])
    atom_tree = cKDTree(xyz)
    patch = np.zeros(n)
    for i, idx in enumerate(atom_tree.query_ball_point(ca, 10.0)):
        s = sasa[idx]
        patch[i] = (s * carbon[idx]).sum() / (s.sum() + 1e-9)

    ang, dsin, dcos, end = _backbone_geometry(ca)
    slow, fast, hinge = gnm(ca)

    if static_score is None:
        static_score = core.pocket_grid(chain.xyz, chain.atom_res, n, p)["residue_score"]
    cb_s, cb_v = _trimmed_pockets(chain, KEEP_CB, p)
    cg_s, _cg_v = _trimmed_pockets(chain, KEEP_CG, p)

    return np.column_stack([rsasa, rside, up, down, cn[6.0], cn[12.0], cn[16.0], cent, patch, ang, dsin, dcos, end,
                            slow, fast, hinge, cb_s, cb_v, np.maximum(cb_s - static_score, 0.0), cg_s,
                            np.maximum(cg_s - static_score, 0.0)])


# ------------------------------------------------------------------------------------------------ ensembles

def ensemble_descriptors(chain, static_score, params=None, anm=None):
    """[n_res, len(ENSEMBLE)]: mean pocket score, opening frequency and mean gain over the structure, across a
    5 A ensemble and across the 3 A ensemble with the side chains cut back to Cbeta (``anm``: the (eigenvalues,
    modes) of core.features, recomputed when not given)."""
    p = dict(core.DEFAULTS, **(params or {}))
    n = chain.n_res
    vals, modes = anm if anm is not None else core.anm_modes(chain.xyz[chain.ca_index], p["anm_cutoff_A"], p["n_modes"])
    keep = np.array([str(a) in KEEP_CB for a in chain.atom_names]) | (chain.atom_res < 0)

    def run(rmsd, mask=None):
        out = []
        for x in core.conformers(chain, vals, modes, p["n_conformers"], rmsd, p["seed"]):
            if mask is None:
                out.append(core.pocket_grid(x, chain.atom_res, n, p)["residue_score"])
            else:
                out.append(core.pocket_grid(x[mask], chain.atom_res[mask], n, p)["residue_score"])
        return np.array(out)

    s0 = np.asarray(static_score, dtype=float)
    a50 = run(5.0)
    t0 = core.pocket_grid(chain.xyz[keep], chain.atom_res[keep], n, p)["residue_score"]
    tr = run(p["rmsd_A"], keep)
    gain = lambda e, ref: np.maximum(e - ref[None, :], 0.0).mean(0)
    return np.column_stack([a50.mean(0), (a50 > 0).mean(0), gain(a50, s0), tr.mean(0), (tr > 0).mean(0),
                            gain(tr, t0)])
