"""
Ramachandran analysis: backbone phi/psi of every residue judged against reference distributions of
high-resolution PDB chains, in the six MolProbity classes (general, Gly, trans-Pro, cis-Pro, pre-Pro,
Ile/Val).

    rows = phi_psi(path)                 # one row per residue (and per model of a multi-model file)
    rows = evaluate(rows)                # adds "region": favoured / allowed / outlier
    stats = summarize(rows)              # % favoured / allowed / outliers, verdict against MolProbity goals
    plot(rows, "rama.png", title=...)    # 2 x 3 panels, one per class, outliers labelled

The reference (rama_reference.npz) is built by scripts/build_rama_reference.py: "favoured" contours hold 98%
of the reference residues, "allowed" 99.95% (general) or 99.8% (other classes), as in MolProbity.  An outlier
is a residue outside the allowed contour: in a well-refined structure fewer than 0.05% (MolProbity goal), and
favoured > 98%.
"""

import math
import os

import numpy as np

CLASSES = ("general", "glycine", "trans_pro", "cis_pro", "pre_pro", "ile_val")
CLASS_TITLES = {"general": "General", "glycine": "Glycine", "trans_pro": "trans-Proline", "cis_pro": "cis-Proline",
                "pre_pro": "pre-Proline", "ile_val": "Ile / Val"}
GOAL_FAVOURED = 98.0
GOAL_OUTLIERS = 0.05

STANDARD = set("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "ASH": "ASP",
           "GLH": "GLU", "LYN": "LYS", "CYX": "CYS", "CYM": "CYS", "MSE": "MET", "SEP": "SER", "TPO": "THR",
           "PTR": "TYR", "NLE": "LEU", "CSO": "CYS", "HYP": "PRO", "MLY": "LYS"}
_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rama_reference.npz")
_REF = {}
FAVOURED_FRACTION = 0.98
ALLOWED_FRACTION = {"general": 0.9995}
DEFAULT_ALLOWED_FRACTION = 0.998
# Gaussian smoothing (degrees) of the reference histograms: narrow for the favoured core, wide for the sparse
# allowed tails (calibrated so the outlier percentages follow the wwPDB/MolProbity validation reports)
SIGMA_FAVOURED = {"general": 3.0, "glycine": 4.0, "trans_pro": 3.0, "cis_pro": 6.0, "pre_pro": 4.0, "ile_val": 3.0}
# x0.8 of (8, 10, 8, 12, 10, 8): on 80 X-ray entries at 1.8-3.0 A (not in the reference) the outlier % follows the
# wwPDB report with mean error +0.03, mean |error| 0.19 percentage points, r = 0.97; on 80 held-out entries -0.04,
# 0.21, r = 0.98 (scripts/calibrate_rama_reference.py)
SIGMA_ALLOWED = {"general": 6.4, "glycine": 8.0, "trans_pro": 6.4, "cis_pro": 9.6, "pre_pro": 8.0, "ile_val": 6.4}


def _smooth(hist, sigma_deg, bin_deg):
    from scipy.ndimage import gaussian_filter

    dens = gaussian_filter(hist.astype(float), sigma_deg / bin_deg, mode="wrap")
    return dens / dens.sum()


def _level(dens, hist, fraction):
    """Density level such that ``fraction`` of the reference residues lie where the density is at least it."""
    w = hist.ravel().astype(float)
    d = dens.ravel()
    order = np.argsort(d)
    cum = np.cumsum(w[order]) / w.sum()
    k = int(np.searchsorted(cum, 1.0 - fraction, side="right"))
    return float(d[order][min(k, len(d) - 1)])


def reference(path=None, sigma_allowed=None, sigma_favoured=None):
    """{class: (favoured-scale density, allowed-scale density, (allowed level, favoured level), count)},
    the bin size and the number of reference chains."""
    path = path or _REF_PATH
    sa = dict(SIGMA_ALLOWED, **(sigma_allowed or {}))
    sf = dict(SIGMA_FAVOURED, **(sigma_favoured or {}))
    key = (path, tuple(sorted(sa.items())), tuple(sorted(sf.items())))
    if key not in _REF:
        data = np.load(path)
        bin_deg = float(data["bin_deg"])
        table = {}
        for cls in CLASSES:
            hist = data[f"{cls}_hist"]
            fine, wide = _smooth(hist, sf[cls], bin_deg), _smooth(hist, sa[cls], bin_deg)
            levels = (_level(wide, hist, ALLOWED_FRACTION.get(cls, DEFAULT_ALLOWED_FRACTION)),
                      _level(fine, hist, FAVOURED_FRACTION))
            table[cls] = (fine, wide, levels, int(hist.sum()))
        _REF[key] = (table, bin_deg, int(len(data["chains"])))
    return _REF[key]


def parent_name(resname):
    name = str(resname).upper()
    return ALIASES.get(name, name)


def classify(resname, next_resname, omega_deg):
    name, nxt = parent_name(resname), parent_name(next_resname) if next_resname else ""
    if name == "GLY":
        return "glycine"
    if name == "PRO":
        return "cis_pro" if omega_deg is not None and abs(omega_deg) < 30.0 else "trans_pro"
    if nxt == "PRO":
        return "pre_pro"
    if name in ("ILE", "VAL"):
        return "ile_val"
    return "general"


def _atom(res, name):
    for a in res:
        if a.name == name:
            return a
    return None


def _bonded(a, b):
    c, n = _atom(a, "C"), _atom(b, "N")
    return c is not None and n is not None and c.pos.dist(n.pos) < 2.0


def phi_psi(path, chains=None, models="first"):
    """Rows {model, chain, resi, resn, class, phi, psi, omega} for residues with both neighbours bonded."""
    import gemmi

    st = gemmi.read_structure(str(path))
    wanted = {str(c) for c in chains} if chains else None
    rows = []
    model_list = list(st) if models == "all" else [st[0]]
    for m_index, model in enumerate(model_list):
        for chain in model:
            if wanted and chain.name not in wanted:
                continue
            residues = [r for r in chain if parent_name(r.name) in STANDARD
                        and all(_atom(r, k) is not None for k in ("N", "CA", "C"))]
            for i in range(1, len(residues) - 1):
                prev, res, nxt = residues[i - 1], residues[i], residues[i + 1]
                if not (_bonded(prev, res) and _bonded(res, nxt)):
                    continue
                phi, psi = gemmi.calculate_phi_psi(prev, res, nxt)
                if math.isnan(phi) or math.isnan(psi):
                    continue
                omega = gemmi.calculate_omega(prev, res)
                omega = None if math.isnan(omega) else math.degrees(omega)
                rows.append({"model": m_index + 1, "chain": chain.name, "resi": f"{res.seqid.num}{res.seqid.icode}".strip(),
                             "resn": res.name, "class": classify(res.name, nxt.name, omega),
                             "phi": round(math.degrees(phi), 1), "psi": round(math.degrees(psi), 1),
                             "omega": None if omega is None else round(omega, 1)})
    return rows


def density_at(cls, phi, psi, ref=None, scale="favoured"):
    """Reference density at (phi, psi), bilinear on the periodic grid (``scale`` favoured or allowed)."""
    table, bin_deg, _n = ref or reference()
    dens = table[cls][0 if scale == "favoured" else 1]
    n = dens.shape[0]
    x = (float(phi) + 180.0) / bin_deg - 0.5
    y = (float(psi) + 180.0) / bin_deg - 0.5
    i0, j0 = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - i0, y - j0
    v = 0.0
    for di, wx in ((0, 1 - fx), (1, fx)):
        for dj, wy in ((0, 1 - fy), (1, fy)):
            v += wx * wy * dens[(i0 + di) % n, (j0 + dj) % n]
    return v


def evaluate(rows, ref=None):
    ref = ref or reference()
    for r in rows:
        allowed, favoured = ref[0][r["class"]][2]
        if density_at(r["class"], r["phi"], r["psi"], ref) >= favoured:
            r["region"] = "favoured"
        elif density_at(r["class"], r["phi"], r["psi"], ref, "allowed") >= allowed:
            r["region"] = "allowed"
        else:
            r["region"] = "outlier"
    return rows


def label(r):
    return f"{r['chain']}:{r['resn']}{r['resi']}" if str(r.get("chain", "")).strip() else f"{r['resn']}{r['resi']}"


def summarize(rows):
    """Percentages over residues (for several models, over all residue-model pairs) and the verdict."""
    n = len(rows)
    if not n:
        return {"residues": 0, "verdict": "no residues with phi/psi"}
    fav = sum(r["region"] == "favoured" for r in rows)
    out = [r for r in rows if r["region"] == "outlier"]
    pf, po = 100.0 * fav / n, 100.0 * len(out) / n
    if pf >= GOAL_FAVOURED and po <= GOAL_OUTLIERS:
        verdict = "excellent (MolProbity goals met)"
    elif pf >= 95.0 and po <= 0.5:
        verdict = "good"
    elif pf >= 90.0 and po <= 2.0:
        verdict = "acceptable; check the outliers"
    else:
        verdict = "poor backbone geometry; refine or rebuild"
    models = sorted({r["model"] for r in rows})
    by_residue = {}
    for r in out:
        by_residue.setdefault(label(r), []).append(r)
    worst = sorted(by_residue.items(), key=lambda kv: -len(kv[1]))
    return {"residues": n if len(models) == 1 else len({label(r) for r in rows}), "models": len(models),
            "favoured_pct": round(pf, 2), "allowed_pct": round(100.0 - pf - po, 2), "outliers_pct": round(po, 2),
            "verdict": verdict,
            "outliers": [{"residue": k, "class": v[0]["class"], "phi": v[0]["phi"], "psi": v[0]["psi"],
                          **({"fraction_of_models": round(len(v) / len(models), 2)} if len(models) > 1 else {})}
                         for k, v in worst[:40]],
            "by_class": {cls: sum(r["class"] == cls for r in rows) for cls in CLASSES}}


def plot(rows, path, title="", ref=None, dpi=200, label_outliers=True):
    """MolProbity-style figure: one panel per class with the favoured / allowed contours of the reference."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ref = ref or reference()
    table, bin_deg, n_chains = ref
    stats = summarize(rows)
    multi = stats.get("models", 1) > 1
    fig = Figure(figsize=(12.0, 8.7), dpi=dpi)
    FigureCanvasAgg(fig)
    centers = np.arange(-180.0 + bin_deg / 2, 180.0, bin_deg)
    for k, cls in enumerate(CLASSES):
        ax = fig.add_subplot(2, 3, k + 1)
        fine, wide, (allowed, favoured), _count = table[cls]
        ax.contourf(centers, centers, wide.T, levels=[allowed, max(float(wide.max()), allowed * 1.0001)],
                    colors=["#dcebf7"])
        ax.contourf(centers, centers, fine.T, levels=[favoured, max(float(fine.max()), favoured * 1.0001)],
                    colors=["#9ec9ea"])
        ax.contour(centers, centers, wide.T, levels=[allowed], colors=["#7fa9cf"], linewidths=0.6)
        ax.contour(centers, centers, fine.T, levels=[favoured], colors=["#3f7fb5"], linewidths=0.8)
        sel = [r for r in rows if r["class"] == cls]
        ok = [r for r in sel if r["region"] != "outlier"]
        bad = [r for r in sel if r["region"] == "outlier"]
        alpha = 0.25 if multi and len(ok) > 2000 else 0.85
        ax.scatter([r["phi"] for r in ok], [r["psi"] for r in ok], s=6 if multi else 9, c="#222222",
                   alpha=alpha, linewidths=0, zorder=3)
        ax.scatter([r["phi"] for r in bad], [r["psi"] for r in bad], s=22, c="#d62728", edgecolors="white",
                   linewidths=0.5, zorder=4)
        if label_outliers:
            seen = set()
            for r in bad[:25]:
                name = label(r)
                if name in seen:
                    continue
                seen.add(name)
                ax.annotate(name, (r["phi"], r["psi"]), xytext=(4, 3), textcoords="offset points", fontsize=7,
                            color="#a01010", zorder=5)
        n_sel = len(sel)
        fav_pct = 100.0 * sum(r["region"] == "favoured" for r in sel) / n_sel if n_sel else 0.0
        ax.set_title(f"{CLASS_TITLES[cls]}  (n = {n_sel}" + (f", {fav_pct:.1f}% fav." if n_sel else "")
                     + (f", {len(bad)} out." if bad else "") + ")", fontsize=10)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xticks(range(-180, 181, 90))
        ax.set_yticks(range(-180, 181, 90))
        ax.axhline(0, color="#999999", lw=0.5, zorder=1)
        ax.axvline(0, color="#999999", lw=0.5, zorder=1)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        if k >= 3:
            ax.set_xlabel("φ (°)", fontsize=9)
        if k % 3 == 0:
            ax.set_ylabel("ψ (°)", fontsize=9)
    head = title or "Ramachandran plot"
    if stats.get("residues"):
        head += (f"\n{stats['residues']} residues" + (f" × {stats['models']} frames" if multi else "")
                 + f" · favoured {stats['favoured_pct']:.1f}% · allowed {stats['allowed_pct']:.1f}% · "
                 f"outliers {stats['outliers_pct']:.2f}% — {stats['verdict']}")
    fig.suptitle(head, fontsize=12)
    fig.text(0.5, 0.01, f"Contours: favoured (98%) and allowed (99.95% general, 99.8% others) regions of "
             f"{n_chains} high-resolution PDB chains (≤1.5 Å, 30% identity clusters). Red: outliers.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.025, 1, 0.955), h_pad=1.6)
    fig.savefig(path, dpi=dpi)
    return path


def analyze(path, out_png=None, chains=None, models="first", title=""):
    rows = evaluate(phi_psi(path, chains=chains, models=models))
    stats = summarize(rows)
    if out_png:
        plot(rows, out_png, title=title)
    return rows, stats
