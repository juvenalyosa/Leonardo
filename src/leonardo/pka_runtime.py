"""
pKa and pH-dependent protonation states of small molecules.

One entry point for the rest of Leonardo: :func:`protonate_molecule` returns the
dominant protonation state at a pH with the heavy-atom coordinates of the input
kept, plus the provenance (method, site pKa values, populations).

Scope: small molecules (ligands, cofactors, non-standard residues). Protein
side chains in their protein environment are not handled here.

The module decides which hydrogens exist; partial charges are still assigned
downstream (Gasteiger, AM1-BCC, ...).

Site pKa values come from a backend:

- ``empirical``: class-typical pKa (fast, no QM), the default for bulk work.
- ``pm7`` / ``psi4`` (later phases): relative (isodesmic) pKa against a reference
  compound of the same class, with a per-class linear calibration.

Populations always come from microstate free energies, never from independent
Henderson-Hasselbalch per site, so coupled sites (zwitterions, diacids) are
handled.
"""

import hashlib
import json
import math
import os
from itertools import product

RT_LN10 = 1.3642  # kcal/mol at 298.15 K
MAX_ENUMERATED_SITES = 4
# Part of the result-cache key: bump whenever SITE_CLASSES, flags or microstate rules change,
# otherwise results computed with the old rules are served from the cache.
PKA_RUNTIME_VERSION = "2"

# --------------------------------------------------------------------------- #
# Site classes
# --------------------------------------------------------------------------- #
# kind: "acid" (neutral HA -> A-) or "base" (neutral B -> BH+).
# reference: compound used for relative (isodesmic) QM pKa, with experimental pKa
# (CRC Handbook of Chemistry and Physics; secondary/tertiary amine references follow
# Kromann et al., PeerJ 2016, 4:e2335).
# empirical: class-typical pKa and spread used by the empirical backend.
# Order matters: an atom is assigned to the first class that matches it.
_NOT_ACYL_N = "!$(N-[#6]=[#7,#8,#16]);!$(N-[#7,#8,#16]);!$(N-S(=O)=O);!$(N-C#N)"
SITE_CLASSES = [
    {
        # exocyclic imine on a ring carbon with two ring/exocyclic N neighbours: cyclic
        # guanidine (2-iminopyrimidine); the ring N is not the basic site
        "name": "guanidine", "kind": "base",
        "smarts": "[NX2;!R;+0;!$(N-C=O);!$(N-S(=O)=O);!$(N-[#8,#7])]=[#6;R]([#7])[#7]",
        "reference": {"name": "guanidine", "acid": "NC(N)=[NH2+]", "base": "NC(N)=N", "pka": 13.6},
        "empirical": (12.0, 1.5),
    },
    {
        "name": "amidine", "kind": "base",
        "smarts": "[NX2;!R;+0;!$(N-[#8,#7]);!$(N-C=O)]=[#6;R]([#7])[#6]",
        "reference": {"name": "acetamidine", "acid": "CC(N)=[NH2+]", "base": "CC(N)=N", "pka": 12.52},
        "empirical": (11.0, 1.5),
    },
    {
        "name": "guanidine", "kind": "base",
        "smarts": "[NX2;+0;!$(N-C=O);!$(N-S(=O)=O);!$(N-[#8,#7])]=[CX3]([NX3])[NX3]",
        "reference": {"name": "guanidine", "acid": "NC(N)=[NH2+]", "base": "NC(N)=N", "pka": 13.6},
        "empirical": (13.0, 1.0),
    },
    {
        "name": "amidine", "kind": "base",
        "smarts": "[NX2;+0;!$(N-[#8,#7]);!$(N-C=O)]=[CX3]([NX3;!$(N-C=O)])[#6]",
        "reference": {"name": "acetamidine", "acid": "CC(N)=[NH2+]", "base": "CC(N)=N", "pka": 12.52},
        "empirical": (11.5, 1.0),
    },
    {
        "name": "carboxylic_aromatic", "kind": "acid",
        "smarts": "[OX2H1]-[CX3;$(C-a)]=O",
        "reference": {"name": "benzoic acid", "acid": "OC(=O)c1ccccc1", "base": "[O-]C(=O)c1ccccc1", "pka": 4.20},
        "empirical": (4.0, 0.6),
    },
    {
        "name": "carboxylic_aliphatic", "kind": "acid",
        "smarts": "[OX2H1]-[CX3;!$(C-a)]=O",
        "reference": {"name": "acetic acid", "acid": "CC(=O)O", "base": "CC(=O)[O-]", "pka": 4.76},
        "empirical": (4.5, 0.8),
    },
    {
        "name": "thiol_aromatic", "kind": "acid",
        "smarts": "[SX2H1]-a",
        "reference": {"name": "thiophenol", "acid": "Sc1ccccc1", "base": "[S-]c1ccccc1", "pka": 6.62},
        "empirical": (6.6, 1.0),
    },
    {
        "name": "thiol_aliphatic", "kind": "acid",
        "smarts": "[SX2H1]-[CX4]",
        "reference": {"name": "ethanethiol", "acid": "CCS", "base": "CC[S-]", "pka": 10.61},
        "empirical": (10.0, 0.8),
    },
    {
        "name": "phenol", "kind": "acid",
        "smarts": "[OX2H1]-c",
        "reference": {"name": "phenol", "acid": "Oc1ccccc1", "base": "[O-]c1ccccc1", "pka": 9.99},
        "empirical": (9.5, 1.3),
    },
    {
        "name": "imidazole", "kind": "base",
        "smarts": "[nX2;+0;r5;$(n:c:[nX3;+0])]",
        "reference": {"name": "imidazole", "acid": "c1c[nH+]c[nH]1", "base": "c1cnc[nH]1", "pka": 6.95},
        "empirical": (6.9, 0.8),
    },
    {
        "name": "pyridine", "kind": "base",
        "smarts": "[nX2;+0;r6]",
        "reference": {"name": "pyridine", "acid": "c1cc[nH+]cc1", "base": "c1ccncc1", "pka": 5.23},
        "empirical": (5.2, 1.5),
    },
    {
        "name": "aniline", "kind": "base",
        "smarts": f"[NX3;+0;{_NOT_ACYL_N}]-c",
        "reference": {"name": "aniline", "acid": "[NH3+]c1ccccc1", "base": "Nc1ccccc1", "pka": 4.60},
        "empirical": (4.6, 1.2),
    },
    {
        "name": "amine1", "kind": "base",
        "smarts": f"[NX3;H2;+0;{_NOT_ACYL_N}]-[CX4]",
        "reference": {"name": "methylamine", "acid": "C[NH3+]", "base": "CN", "pka": 10.63},
        "empirical": (10.2, 0.8),
    },
    {
        "name": "amine2", "kind": "base",
        "smarts": f"[NX3;H1;+0;{_NOT_ACYL_N}]([CX4])[CX4]",
        "reference": {"name": "diethylamine", "acid": "CC[NH2+]CC", "base": "CCNCC", "pka": 10.98},
        "empirical": (10.5, 1.0),
    },
    {
        "name": "amine3", "kind": "base",
        "smarts": f"[NX3;H0;+0;{_NOT_ACYL_N}]([CX4])([CX4])[CX4]",
        "reference": {"name": "triethylamine", "acid": "CC[NH+](CC)CC", "base": "CCN(CC)CC", "pka": 10.75},
        "empirical": (9.8, 1.2),
    },
]
QM_CLASS_NAMES = frozenset(c["name"] for c in SITE_CLASSES)

# Dimorphite-DL site types used as an empirical-only fallback for groups without a
# QM class (sulfonic/phosphoric acids, imides, ...). Types whose Dimorphite mean is
# not a useful aqueous pKa (alcohols, amides, pyrrole NH, ...) are left out.
_DIMORPHITE_FALLBACK_TYPES = {
    "Azide", "Sulfate", "Sulfonate", "Sulfinic_acid", "Thioic_acid", "Phosphate",
    "Internal_phosphate_polyphos_chain", "Initial_phosphate_like_in_ATP_ADP", "Phosphonate",
    "Peroxide1", "O=C-C=C-OH", "Vinyl_alcohol", "N-hydroxyamide", "Ringed_imide1", "Ringed_imide2",
    "Imide", "Amide_electronegative", "Sulfonamide", "Phosphinic_acid", "Phosphate_diester",
    "Phosphonate_ester", "Primary_hydroxyl_amine",
}

# Flags that put a site outside the domain its class calibration describes: such sites are
# left out of calibration fits (they still get a pKa, with a warning). "tertiary_amine" is
# intrinsic to its class and stays in.
CALIBRATION_EXCLUDED_FLAGS = frozenset({
    "resonance_acceptor", "polyaza_ring", "ortho_heteroatom", "acyl_substituted",
    "unusual_group", "tertiary_aniline", "cyclic_imine", "empirical_only",
})

_ACCEPTOR_ON_RING = "[$([N+](=O)[O-]),$(C#N),$([CX3]=O),$(S(=O)=O),$([NX2]=O)]"
_UNUSUAL_GROUPS = "[$([SX6]),$([SX5]),$([PX5]),$([IX3,IX5]),$([BrX3]),$([ClX3,ClX5])]"


class PkaError(Exception):
    """Invalid input for pKa/protonation (unparsable molecule, unknown method, ...)."""


def _chem():
    from rdkit import Chem

    return Chem


# --------------------------------------------------------------------------- #
# Molecule handling
# --------------------------------------------------------------------------- #
def neutral_parent(mol):
    """Heavy-atom molecule in its neutral form (coordinates kept), used for site
    matching: acids as HA, bases as B. Quaternary ammonium and nitro stay charged."""
    Chem = _chem()
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if mol is None:
        raise PkaError("molecule is required")
    heavy = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
    parent = rdMolStandardize.Uncharger().uncharge(heavy)
    Chem.SanitizeMol(parent)
    return parent


def _acceptor_on_same_ring(mol, atom_idx):
    """True when the aromatic ring carrying the site also carries a nitro/cyano/
    carbonyl/sulfonyl substituent (charge delocalization that PM7/COSMO handles poorly)."""
    Chem = _chem()
    atom = mol.GetAtomWithIdx(int(atom_idx))
    ring_atoms = set()
    for nb in atom.GetNeighbors():
        if nb.GetIsAromatic():
            for ring in mol.GetRingInfo().AtomRings():
                if nb.GetIdx() in ring:
                    ring_atoms.update(ring)
    if not ring_atoms:
        return False
    patt = Chem.MolFromSmarts(_ACCEPTOR_ON_RING)
    for match in mol.GetSubstructMatches(patt):
        grp = match[0]
        for nb in mol.GetAtomWithIdx(grp).GetNeighbors():
            if nb.GetIdx() in ring_atoms:
                return True
    return False


def _site_flags(mol, cls_name, atom_idx):
    flags = []
    if cls_name in ("phenol", "aniline", "thiol_aromatic", "carboxylic_aromatic") and _acceptor_on_same_ring(
        mol, atom_idx
    ):
        flags.append("resonance_acceptor")
    if cls_name == "amine3":
        flags.append("tertiary_amine")
    if cls_name == "aniline":
        n_atom = mol.GetAtomWithIdx(int(atom_idx))
        alkyl = [nb for nb in n_atom.GetNeighbors() if not nb.GetIsAromatic()]
        # N,N-dialkyl or tert-alkyl anilines: steric twist breaks N-ring conjugation
        if len(alkyl) >= 2 or any(nb.GetDegree() >= 4 for nb in alkyl):
            flags.append("tertiary_aniline")
    Chem_ = _chem()
    if mol.HasSubstructMatch(Chem_.MolFromSmarts(_UNUSUAL_GROUPS)):
        flags.append("unusual_group")      # hypervalent S/P/halogen: PM7 is unreliable
    if cls_name in ("amidine", "guanidine") and mol.GetAtomWithIdx(int(atom_idx)).IsInRing() is False and any(
        nb.IsInRing() for nb in mol.GetAtomWithIdx(int(atom_idx)).GetNeighbors() if nb.GetAtomicNum() == 6
    ):
        flags.append("cyclic_imine")
    if cls_name in ("pyridine", "imidazole"):
        atom = mol.GetAtomWithIdx(int(atom_idx))
        for ring in mol.GetRingInfo().AtomRings():
            if atom.GetIdx() in ring:
                n_count = sum(1 for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7)
                if n_count > (2 if cls_name == "imidazole" else 1):
                    flags.append("polyaza_ring")
                    break
    if cls_name == "pyridine":
        # electronegative substituent next to the ring N (2-chloro-, 2-methoxypyridine):
        # field and lone-pair effects the reference compound does not share
        atom = mol.GetAtomWithIdx(int(atom_idx))
        for ring_nb in atom.GetNeighbors():
            for sub in ring_nb.GetNeighbors():
                if sub.GetIdx() != atom.GetIdx() and not sub.IsInRing() and sub.GetAtomicNum() in (7, 8, 9, 16, 17, 35, 53):
                    flags.append("ortho_heteroatom")
                    break
            if "ortho_heteroatom" in flags:
                break
    if cls_name in ("amidine", "guanidine"):
        # acyl/sulfonyl on the group: conjugation lowers basicity by several units
        Chem = _chem()
        carbon = next((nb for nb in mol.GetAtomWithIdx(int(atom_idx)).GetNeighbors() if nb.GetAtomicNum() == 6), None)
        group_n = [atom_idx] + ([nb.GetIdx() for nb in carbon.GetNeighbors() if nb.GetAtomicNum() == 7] if carbon else [])
        patt = Chem.MolFromSmarts("[#7]-[$([CX3]=[OX1]),$([SX4](=O)=O)]")
        if any(m[0] in group_n for m in mol.GetSubstructMatches(patt)):
            flags.append("acyl_substituted")
    return flags


def _dimorphite_table():
    try:
        import dimorphite_dl
    except ImportError:
        return []
    path = os.path.join(os.path.dirname(dimorphite_dl.__file__), "smarts", "site_substructures.smarts")
    rows = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5 or line.startswith("#"):
                    continue
                name = parts[0].strip()
                special = name.startswith("*")
                name = name.lstrip("*")
                triples = []
                for k in range(2, len(parts) - 2, 3):
                    try:
                        triples.append((int(parts[k]), float(parts[k + 1]), float(parts[k + 2])))
                    except ValueError:
                        break
                if triples:
                    rows.append((name, special, parts[1].strip(), triples))
    except OSError:
        return []
    return rows


def find_ionizable_sites(mol):
    """
    Ionizable sites of a molecule, each assigned to one class.

    Returns a list of dicts: ``atom`` (heavy-atom index in :func:`neutral_parent`),
    ``class``, ``kind`` ("acid"/"base"), ``empirical_pka``, ``empirical_sd``,
    ``qm_supported`` and ``flags`` (difficult cases).
    """
    Chem = _chem()
    parent = neutral_parent(mol)
    taken = set()
    sites = []
    for cls in SITE_CLASSES:
        patt = Chem.MolFromSmarts(cls["smarts"])
        for match in parent.GetSubstructMatches(patt):
            idx = int(match[0])
            if idx in taken:
                continue
            taken.add(idx)
            pka, sd = cls["empirical"]
            sites.append({
                "atom": idx, "class": cls["name"], "kind": cls["kind"],
                "empirical_pka": float(pka), "empirical_sd": float(sd),
                "qm_supported": True, "flags": _site_flags(parent, cls["name"], idx),
            })
    # empirical-only fallback (Dimorphite site table, matched with explicit H)
    with_h = Chem.AddHs(parent)
    n_heavy = parent.GetNumAtoms()
    for name, special, smarts, triples in _dimorphite_table():
        if name not in _DIMORPHITE_FALLBACK_TYPES:
            continue
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for match in with_h.GetSubstructMatches(patt):
            for pos, pka, sd in triples:
                if pos >= len(match):
                    continue
                idx = int(match[pos])
                if idx >= n_heavy or idx in taken:
                    continue
                atom = parent.GetAtomWithIdx(idx)
                # Dimorphite convention: N sites are bases unless the type is "special" (*)
                kind = "base" if (atom.GetAtomicNum() == 7 and not special) else "acid"
                taken.add(idx)
                sites.append({
                    "atom": idx, "class": f"dimorphite:{name}", "kind": kind,
                    "empirical_pka": float(pka), "empirical_sd": float(sd),
                    "qm_supported": False, "flags": ["empirical_only"],
                })
    sites.sort(key=lambda s: s["atom"])
    return sites


# --------------------------------------------------------------------------- #
# Microstates and populations
# --------------------------------------------------------------------------- #
def select_enumerated_sites(sites, focus_ph=7.4, max_sites=MAX_ENUMERATED_SITES, ph_range=(0.0, 14.0)):
    """
    Split sites into the ones whose protonation is enumerated and the ones fixed in
    their dominant state. Sites that never change within ``ph_range`` are fixed; if
    more than ``max_sites`` remain, the ones farthest from ``focus_ph`` are fixed.
    Returns (enumerated, fixed, warnings); fixed entries carry ``fixed_protonated``.
    """
    lo, hi = float(ph_range[0]), float(ph_range[1])
    enumerated, fixed, warnings = [], [], []
    for s in sites:
        pka = float(s.get("pka", s["empirical_pka"]))
        if pka < lo - 2.0 or pka > hi + 2.0:
            fixed.append(dict(s, fixed_protonated=bool(pka > hi)))
        else:
            enumerated.append(dict(s))
    if len(enumerated) > int(max_sites):
        enumerated.sort(key=lambda s: abs(float(s.get("pka", s["empirical_pka"])) - float(focus_ph)))
        for s in enumerated[int(max_sites):]:
            fixed.append(dict(s, fixed_protonated=bool(float(s.get("pka", s["empirical_pka"])) > float(focus_ph))))
        warnings.append(
            f"{len(enumerated) - int(max_sites)} site(s) fixed in their dominant state at pH {float(focus_ph):.1f} "
            f"(more than {int(max_sites)} ionizable sites)"
        )
        enumerated = sorted(enumerated[: int(max_sites)], key=lambda s: s["atom"])
    return enumerated, fixed, warnings


def enumerate_microstates(n_sites):
    """All protonation patterns of ``n_sites`` sites (True = protonated)."""
    return [tuple(bool(b) for b in bits) for bits in product((False, True), repeat=int(n_sites))]


def microstate_free_energies(site_pkas, micro_pka_fn=None):
    """
    Free energy of each microstate (kcal/mol, relative to all-deprotonated at pH 0).

    With independent sites, protonating site ``s`` lowers G by ``RT ln10 * pKa_s``.
    ``micro_pka_fn(state, s)`` may return the pKa of site ``s`` given the other sites'
    state (coupled sites, from a QM backend); energies are then built along a
    spanning path from the all-deprotonated state, adding sites in index order.
    """
    n = len(site_pkas)
    energies = {}
    for state in enumerate_microstates(n):
        g = 0.0
        partial = [False] * n
        for s in range(n):
            if state[s]:
                pka = float(site_pkas[s]) if micro_pka_fn is None else float(micro_pka_fn(tuple(partial), s))
                g -= RT_LN10 * pka
                partial[s] = True
        energies[state] = g
    return energies


def microstate_populations(energies, ph):
    """Boltzmann populations at ``ph``: G_i(pH) = G_i + n_H,i * RT ln10 * pH."""
    ph = float(ph)
    shifted = {st: g + sum(st) * RT_LN10 * ph for st, g in energies.items()}
    gmin = min(shifted.values())
    w = {st: math.exp(-(g - gmin) / (RT_LN10 / math.log(10.0))) for st, g in shifted.items()}
    z = sum(w.values())
    return {st: v / z for st, v in w.items()}


def _state_charge(sites, state, base_charge=0):
    q = int(base_charge)
    for s, prot in zip(sites, state):
        if s["kind"] == "acid" and not prot:
            q -= 1
        elif s["kind"] == "base" and prot:
            q += 1
    return q


def macroscopic_pkas(energies):
    """
    Macroscopic (stepwise) pKa values, ascending, from the binding polynomial:
    Z_n sums exp(-G_i/RT) over microstates with n protons, and the loss of a proton
    from the n-proton species has pKa = log10(Z_n / Z_(n-1)). Two identical
    independent sites give pK - log10(2) and pK + log10(2).
    """
    if not energies:
        return []
    rt = RT_LN10 / math.log(10.0)
    gmin = min(energies.values())
    z = {}
    for st, g in energies.items():
        n = sum(st)
        z[n] = z.get(n, 0.0) + math.exp(-(g - gmin) / rt)
    n_max = len(next(iter(energies)))
    out = [math.log10(z[n] / z[n - 1]) for n in range(1, n_max + 1) if z.get(n) and z.get(n - 1)]
    return [round(v, 2) for v in sorted(out)]


# --------------------------------------------------------------------------- #
# Calibration (used by the QM backends)
# --------------------------------------------------------------------------- #
def _robust_slope(sub, min_abs_x=0.5):
    """Theil-Sen slope through the origin: median of y/x over points with |x| >= min_abs_x.
    One point, one vote, so a single PM7 failure (large |x|) cannot drag the slope."""
    ratios = sorted(r["y"] / r["x"] for r in sub if abs(r["x"]) >= float(min_abs_x))
    if not ratios:
        return None, 0
    mid = len(ratios) // 2
    med = ratios[mid] if len(ratios) % 2 else 0.5 * (ratios[mid - 1] + ratios[mid])
    return med, len(ratios)


def fit_calibration(rows, min_class_n=5, shrinkage=None, robust=False):
    """
    Linear calibration of relative QM pKa values.

    ``rows``: dicts with ``class``, ``x`` = (dG_deprot(mol) - dG_deprot(ref)) / RT ln10
    and ``y`` = pKa_exp - pKa_ref. The model is ``y = slope * x`` (the reference itself
    maps to its experimental pKa).

    Without ``shrinkage``, classes with at least ``min_class_n`` points get their own
    least-squares slope and the rest use the global slope. With ``shrinkage`` = lambda
    (in units of sum x^2), every class slope is pulled towards the global one:
    ``slope_c = (Sxy_c + lambda * s_global) / (Sxx_c + lambda)``, so a class with few
    points or one outlier (4-nitroaniline in the anilines) cannot run away.
    Returns ``{"global": {...}, "classes": {name: {"slope", "rmse", "n", "own"}},
    "shrinkage": lambda}``.
    """
    rows = [r for r in rows if r.get("x") is not None and r.get("y") is not None]

    def _sums(sub):
        return sum(r["x"] * r["x"] for r in sub), sum(r["x"] * r["y"] for r in sub)

    def _rmse(sub, slope):
        return math.sqrt(sum((r["y"] - slope * r["x"]) ** 2 for r in sub) / len(sub)) if sub else float("nan")

    if robust:
        # Theil-Sen per class, pulled towards the global Theil-Sen slope with a pseudo-count
        # ``shrinkage`` (number of global "votes"); classes without usable points get the global slope
        g_slope, _n = _robust_slope(rows)
        g_slope = 1.0 if g_slope is None else g_slope
        k = float(shrinkage or 0.0)
        classes = {}
        for name in sorted({r["class"] for r in rows}):
            sub = [r for r in rows if r["class"] == name]
            ts, m = _robust_slope(sub)
            if ts is None or m < 3:
                slope, own = g_slope, False
            else:
                slope, own = (m * ts + k * g_slope) / (m + k), True
            classes[name] = {"slope": slope, "rmse": _rmse(sub, slope), "n": len(sub), "own": own}
        return {"global": {"slope": g_slope, "rmse": _rmse(rows, g_slope), "n": len(rows)},
                "classes": classes, "shrinkage": shrinkage, "robust": True}
    sxx, sxy = _sums(rows)
    g_slope = (sxy / sxx) if sxx > 0 else 1.0
    classes = {}
    for name in sorted({r["class"] for r in rows}):
        sub = [r for r in rows if r["class"] == name]
        cxx, cxy = _sums(sub)
        if shrinkage is not None:
            lam = float(shrinkage)
            slope = (cxy + lam * g_slope) / (cxx + lam) if (cxx + lam) > 0 else g_slope
            own = True
        elif len(sub) >= int(min_class_n) and cxx > 0:
            slope, own = cxy / cxx, True
        else:
            slope, own = g_slope, False
        classes[name] = {"slope": slope, "rmse": _rmse(sub, slope), "n": len(sub), "own": own}
    return {"global": {"slope": g_slope, "rmse": _rmse(rows, g_slope), "n": len(rows)},
            "classes": classes, "shrinkage": shrinkage}


def choose_shrinkage(rows, grid=None, robust=False):
    """Shrinkage with the lowest leave-one-out MAE on ``rows`` (1e9 ~ global slope only).
    For ``robust`` fits the shrinkage is a pseudo-count of global votes."""
    if grid is None:
        grid = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1.0e9) if robust else (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1.0e9)
    rows = [r for r in rows if r.get("x") is not None and r.get("y") is not None]
    best = None
    for lam in grid:
        errs = []
        for i, r in enumerate(rows):
            cal = fit_calibration(rows[:i] + rows[i + 1:], shrinkage=lam, robust=robust)
            slope = (cal["classes"].get(r["class"]) or cal["global"])["slope"]
            errs.append(abs(r["y"] - slope * r["x"]))
        mae = sum(errs) / len(errs) if errs else float("inf")
        if best is None or mae < best[1] - 1e-9:
            best = (float(lam), mae)
    return best[0] if best else None


def apply_calibration(calibration, cls_name, x, ref_pka):
    """Calibrated pKa and its uncertainty (class RMSE, else global RMSE)."""
    cal = dict(calibration or {})
    entry = dict((cal.get("classes") or {}).get(cls_name) or {})
    glob = dict(cal.get("global") or {"slope": 1.0, "rmse": float("nan")})
    slope = float(entry.get("slope", glob.get("slope", 1.0)))
    err = float(entry.get("rmse", glob.get("rmse", float("nan"))))
    return float(ref_pka) + slope * float(x), err


# --------------------------------------------------------------------------- #
# Building protonated molecules
# --------------------------------------------------------------------------- #
def build_state_molecule(parent, sites, state, fixed=(), add_hs=True):
    """
    Molecule for one protonation state, on the atoms and coordinates of ``parent``.

    Only charges and hydrogen counts of the site atoms change; with ``add_hs`` the
    hydrogens are added with coordinates, so heavy atoms keep their positions.
    """
    Chem = _chem()
    rw = Chem.RWMol(Chem.Mol(parent))
    todo = [(s, bool(p)) for s, p in zip(sites, state)]
    todo += [(s, bool(s.get("fixed_protonated", False))) for s in fixed]
    for s, protonated in todo:
        atom = rw.GetAtomWithIdx(int(s["atom"]))
        n_h = int(atom.GetTotalNumHs())
        if s["kind"] == "acid" and not protonated:
            if n_h < 1:
                raise PkaError(f"site {s['atom']} ({s['class']}) has no hydrogen to remove")
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
            atom.SetNumExplicitHs(n_h - 1)
            atom.SetNoImplicit(True)
        elif s["kind"] == "base" and protonated:
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
            atom.SetNumExplicitHs(n_h + 1)
            atom.SetNoImplicit(True)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    if add_hs:
        mol = Chem.AddHs(mol, addCoords=mol.GetNumConformers() > 0)
    return mol


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class PkaBackendError(RuntimeError):
    """A QM backend could not produce energies (MOPAC missing, no converged conformer, ...)."""


class PkaCancelled(RuntimeError):
    """The caller's cancel() returned True."""


class EmpiricalBackend:
    """Class-typical pKa per site; no QM. Uncertainty = class spread."""

    name = "empirical"
    calibration_version = "none"

    def evaluate(self, parent, enumerated, fixed, **_kwargs):
        return {
            "site_pkas": [
                {"pka": float(s["empirical_pka"]), "sd": float(s["empirical_sd"]), "source": "class_typical"}
                for s in enumerated
            ],
            "micro_pka": None,
            "warnings": [],
        }

    def site_pkas(self, parent, sites, **kwargs):
        return self.evaluate(parent, sites, (), **kwargs)["site_pkas"]


def _state_key(state):
    return "".join("1" if b else "0" for b in state)


PM7_KEYWORDS = "PM7 EPS=78.4 GNORM=1.0 CYCLES=2000"
# |relative deprotonation energy| beyond this (in pKa units, before calibration) is a PM7
# failure (e.g. SF5-aniline gave -64), not chemistry: the site falls back to empirical.
QM_OUTLIER_LIMIT = 15.0
PM7_RETRY_KEYWORDS = "PM7 EPS=78.4 GNORM=5.0 CYCLES=4000"
CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pka_calibration.json")


def load_calibration(method="pm7", path=None):
    """Calibration shipped with Leonardo for a QM method, or None if absent."""
    target = path or CALIBRATION_FILE
    try:
        with open(target, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    entry = (data.get("methods") or {}).get(str(method))
    if not entry:
        return None
    return dict(entry, version=str(data.get("version", "unknown")))


def parse_mopac_heat_of_formation(out_text):
    """Final heat of formation (kcal/mol) of a MOPAC geometry optimization, or None if the
    run did not converge ("EXCESS NUMBER OF OPTIMIZATION CYCLES", SCF failure, error)."""
    import re

    text = str(out_text or "")
    if "EXCESS NUMBER OF OPTIMIZATION CYCLES" in text or "UNABLE TO ACHIEVE SELF-CONSISTENCE" in text:
        return None
    m = re.search(r"FINAL HEAT OF FORMATION\s*=\s*(-?\d+\.\d+)\s*KCAL/MOL", text)
    return float(m.group(1)) if m else None


def _default_mopac_runner(input_text, timeout=600, exe=None):
    """Run MOPAC on one input in a private directory; returns the .out text ('' on failure)."""
    import shutil
    import subprocess
    import tempfile

    mopac = exe or shutil.which("mopac")
    if not mopac:
        raise PkaBackendError("MOPAC executable not found (install mopac or choose the empirical method)")
    with tempfile.TemporaryDirectory(prefix="leonardo_pka_") as td:
        inp = os.path.join(td, "job.mop")
        with open(inp, "w", encoding="utf-8") as fh:
            fh.write(input_text)
        # one thread per MOPAC process: the backend already runs conformers in parallel, and
        # MOPAC's own OpenMP threads would oversubscribe the CPU
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        try:
            subprocess.run([mopac, inp], cwd=td, capture_output=True, timeout=float(timeout), check=False, env=env)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        try:
            with open(os.path.join(td, "job.out"), encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        except OSError:
            return ""


# reference-compound energies are the same for every molecule of a class: computed once per process
_REFERENCE_DELTA_CACHE = {}


class MopacBackend:
    """
    PM7/COSMO (water) relative pKa.

    For each microstate of the enumerated sites, ``n_conformers`` ETKDG/MMFF conformers
    are optimized with PM7/COSMO and the lowest heat of formation is kept (conformers
    that do not converge are retried with looser settings). The micro pKa of an edge
    (site s protonated vs deprotonated, other sites fixed) is relative to the class
    reference compound, then calibrated per class.
    """

    name = "pm7"

    def __init__(self, calibration=None, n_conformers=10, workers=None, runner=None, seed=20260927,
                 timeout=600):
        self.calibration = calibration if calibration is not None else load_calibration("pm7")
        self.calibration_version = str((self.calibration or {}).get("version", "uncalibrated"))
        self.n_conformers = int(max(1, n_conformers))
        self.workers = int(workers or min(8, os.cpu_count() or 2))
        self.runner = runner or _default_mopac_runner
        self.seed = int(seed)
        self.timeout = float(timeout)
        self._energy_cache = {}

    # -- energies ----------------------------------------------------------- #
    def _conformer_inputs(self, mol_h, keywords, seed):
        Chem = _chem()
        from rdkit.Chem import AllChem

        from .qmmm_mopac_runtime import build_mopac_input_text

        mol = Chem.Mol(mol_h)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=self.n_conformers, params=params))
        if not cids:
            params.useRandomCoords = True
            cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=self.n_conformers, params=params))
        if not cids:
            return []
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except Exception:
            try:
                AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
            except Exception:
                pass
        z = [a.GetAtomicNum() for a in mol.GetAtoms()]
        q = Chem.GetFormalCharge(mol)
        out = []
        for cid in cids:
            xyz = mol.GetConformer(cid).GetPositions()
            out.append(build_mopac_input_text(z, xyz, keywords=keywords, total_charge=q,
                                              title="Leonardo pKa microstate"))
        return out

    def state_energies(self, mols_h, progress=None, cancel=None):
        """Lowest PM7/COSMO heat of formation per molecule (canonical SMILES cached)."""
        from concurrent.futures import ThreadPoolExecutor

        Chem = _chem()
        keys = [Chem.MolToSmiles(m) for m in mols_h]
        todo = [(k, m) for k, m in zip(keys, mols_h) if k not in self._energy_cache]
        todo = list(dict(todo).items())
        for attempt, keywords in enumerate((PM7_KEYWORDS, PM7_RETRY_KEYWORDS)):
            if not todo:
                break
            jobs = []
            for k, m in todo:
                for text in self._conformer_inputs(m, keywords, self.seed + 1000 * attempt):
                    jobs.append((k, text))
            best = {}
            done = 0
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [(k, pool.submit(self.runner, text, self.timeout)) for k, text in jobs]
                for k, fut in futures:
                    if callable(cancel) and cancel():
                        for _k, f in futures:
                            f.cancel()
                        raise PkaCancelled("pKa calculation cancelled")
                    hof = parse_mopac_heat_of_formation(fut.result())
                    done += 1
                    if hof is not None:
                        best[k] = hof if k not in best else min(best[k], hof)
                    if callable(progress) and done % 10 == 0:
                        progress(f"PM7 conformers: {done}/{len(jobs)}")
            self._energy_cache.update(best)
            todo = [(k, m) for k, m in todo if k not in best]
        if todo:
            raise PkaBackendError(f"PM7 did not converge for {len(todo)} protonation state(s): "
                                  + ", ".join(k for k, _m in todo[:3]))
        return [self._energy_cache[k] for k in keys]

    def reference_delta(self, cls_name, progress=None, cancel=None):
        Chem = _chem()
        ref = next(c for c in SITE_CLASSES if c["name"] == cls_name)["reference"]
        key = (cls_name, ref["acid"], ref["base"], PM7_KEYWORDS, self.seed, self.n_conformers,
               getattr(self.runner, "__name__", str(id(self.runner))))
        if self.runner is _default_mopac_runner and key in _REFERENCE_DELTA_CACHE:
            return _REFERENCE_DELTA_CACHE[key]
        acid = Chem.AddHs(Chem.MolFromSmiles(ref["acid"]))
        base = Chem.AddHs(Chem.MolFromSmiles(ref["base"]))
        e_acid, e_base = self.state_energies([acid, base], progress=progress, cancel=cancel)
        value = (e_base - e_acid, float(ref["pka"]))
        if self.runner is _default_mopac_runner:
            _REFERENCE_DELTA_CACHE[key] = value
        return value

    def _pm7_recommended(self, cls_name):
        entry = ((self.calibration or {}).get("classes") or {}).get(cls_name) or {}
        return bool(entry.get("pm7_recommended", True))

    # -- pKa ----------------------------------------------------------------- #
    def evaluate(self, parent, enumerated, fixed, progress=None, cancel=None, **_kwargs):
        n = len(enumerated)
        if n == 0:
            return {"site_pkas": [], "micro_pka": {}, "warnings": []}
        states = enumerate_microstates(n)
        mols = [build_state_molecule(parent, enumerated, st, fixed=fixed, add_hs=True) for st in states]
        if callable(progress):
            progress(f"PM7/COSMO: {len(states)} microstate(s) x {self.n_conformers} conformer(s)")
        energies = dict(zip(states, self.state_energies(mols, progress=progress, cancel=cancel)))
        refs = {}
        micro = {}
        outliers = set()
        for st in states:
            for s_idx, site in enumerate(enumerated):
                if st[s_idx]:
                    continue
                prot = tuple(True if i == s_idx else b for i, b in enumerate(st))
                if not site.get("qm_supported") or not self._pm7_recommended(site["class"]):
                    # no QM class, or the benchmark found the class-typical value more accurate
                    micro[f"{_state_key(st)}|{s_idx}"] = float(site["empirical_pka"])
                    continue
                if site["class"] not in refs:
                    refs[site["class"]] = self.reference_delta(site["class"], progress=progress, cancel=cancel)
                d_ref, ref_pka = refs[site["class"]]
                x = ((energies[st] - energies[prot]) - d_ref) / RT_LN10
                if abs(x) > QM_OUTLIER_LIMIT:
                    outliers.add(s_idx)
                    micro[f"{_state_key(st)}|{s_idx}"] = float(site["empirical_pka"])
                    continue
                pka, _err = apply_calibration(self.calibration, site["class"], x, ref_pka)
                micro[f"{_state_key(st)}|{s_idx}"] = pka
        site_pkas = []
        for s_idx, site in enumerate(enumerated):
            # context: other sites protonated when their empirical pKa is above this one's
            ctx = tuple((i != s_idx) and (float(o["empirical_pka"]) >= float(site["empirical_pka"]))
                        for i, o in enumerate(enumerated))
            pka = micro[f"{_state_key(ctx)}|{s_idx}"]
            if s_idx in outliers:
                sd, source = float(site["empirical_sd"]), "class_typical (PM7 outlier)"
            elif site.get("qm_supported") and not self._pm7_recommended(site["class"]):
                sd, source = float(site["empirical_sd"]), "class_typical (PM7 not recommended for this class)"
            elif site.get("qm_supported"):
                entry = ((self.calibration or {}).get("classes") or {}).get(site["class"]) or {}
                sd = float(entry.get("rmse", ((self.calibration or {}).get("global") or {}).get("rmse", 1.5)))
                source = "pm7_cosmo" + ("" if self.calibration else "_uncalibrated")
            else:
                sd, source = float(site["empirical_sd"]), "class_typical"
            site_pkas.append({"pka": float(pka), "sd": sd, "source": source})
        warnings = [] if self.calibration else ["PM7 pKa values are uncalibrated (no calibration file)"]
        for s_idx, site in enumerate(enumerated):
            if site.get("qm_supported") and not self._pm7_recommended(site["class"]):
                warnings.append(f"site {site['atom']} ({site['class']}): PM7 was less accurate than the "
                                "class-typical pKa for this class in the benchmark; empirical pKa used")
        for s_idx in sorted(outliers):
            warnings.append(f"site {enumerated[s_idx]['atom']} ({enumerated[s_idx]['class']}): PM7 energy is "
                            "implausible for this group; empirical pKa used")
        return {"site_pkas": site_pkas, "micro_pka": micro, "warnings": warnings}


_BACKENDS = {"empirical": EmpiricalBackend, "pm7": MopacBackend}


def register_backend(name, factory):
    """Register a pKa backend (used by the Psi4 phase and by tests)."""
    _BACKENDS[str(name)] = factory


def _backend(method, **kwargs):
    key = str(method or "empirical").strip().lower()
    if key not in _BACKENDS:
        raise PkaError(f"unknown pKa method '{method}' (available: {', '.join(sorted(_BACKENDS))})")
    factory = _BACKENDS[key]
    try:
        return factory(**kwargs) if kwargs else factory()
    except TypeError:
        return factory()


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def _cache_key(parent, method, calibration_version):
    Chem = _chem()
    try:
        ident = Chem.MolToInchiKey(parent) or Chem.MolToSmiles(parent)
    except Exception:
        ident = Chem.MolToSmiles(parent)
    raw = f"{ident}|{method}|{calibration_version}|{PKA_RUNTIME_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _cache_load(cache_dir, key):
    if not cache_dir:
        return None
    path = os.path.join(str(cache_dir), f"{key}.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _cache_store(cache_dir, key, payload):
    if not cache_dir:
        return
    try:
        os.makedirs(str(cache_dir), exist_ok=True)
        tmp = os.path.join(str(cache_dir), f"{key}.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        os.replace(tmp, os.path.join(str(cache_dir), f"{key}.json"))
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def compute_pka(mol, method="empirical", focus_ph=7.4, ph_range=(0.0, 14.0), cache_dir=None,
                progress=None, cancel=None, backend_options=None):
    """
    Site and macroscopic pKa values plus the microstate energies of a molecule.

    Sites to enumerate are chosen with the empirical pKa (see
    :func:`select_enumerated_sites`); the backend then gives the micro pKa of every
    microstate edge, so coupled sites use QM energies. If a QM backend fails the result
    falls back to the empirical values with a warning.

    Returns a JSON-serializable dict: ``sites`` (with ``pka``/``sd``/flags),
    ``fixed_sites``, ``microstates`` (pattern, charge, relative G), ``macro_pkas``,
    ``warnings`` and ``provenance``. Results are cached per molecule + method +
    calibration version.
    """
    parent = neutral_parent(mol)
    backend = _backend(method, **dict(backend_options or {}))
    calibration_version = str(getattr(backend, "calibration_version", "none"))
    key = _cache_key(parent, backend.name, calibration_version)
    cached = _cache_load(cache_dir, key)
    if cached is not None:
        cached.setdefault("provenance", {})["cache"] = "hit"
        return cached
    sites = find_ionizable_sites(parent)
    if callable(progress):
        progress(f"{len(sites)} ionizable site(s) found")
    enumerated, fixed, warnings = select_enumerated_sites(sites, focus_ph=focus_ph, ph_range=ph_range)
    method_used = backend.name
    try:
        if hasattr(backend, "evaluate"):
            evaluation = backend.evaluate(parent, enumerated, fixed, progress=progress, cancel=cancel)
        else:   # minimal backends: independent site pKa values
            evaluation = {"site_pkas": backend.site_pkas(parent, enumerated), "micro_pka": None, "warnings": []}
    except PkaBackendError as exc:
        warnings.append(f"{backend.name} failed ({exc}); empirical pKa values used")
        evaluation = EmpiricalBackend().evaluate(parent, enumerated, fixed)
        method_used, calibration_version = "empirical", "none"
    warnings.extend(evaluation.get("warnings") or [])
    for s, v in zip(enumerated, evaluation["site_pkas"]):
        s["pka"] = float(v["pka"])
        s["sd"] = float(v.get("sd", s["empirical_sd"]))
        s["pka_source"] = str(v.get("source", method_used))
    for s in fixed:
        s.setdefault("pka", float(s["empirical_pka"]))
        s.setdefault("sd", float(s["empirical_sd"]))
        s.setdefault("pka_source", "class_typical (fixed state)")
    micro_table = evaluation.get("micro_pka") or None
    pkas = [s["pka"] for s in enumerated]

    def _micro(partial, s_idx):
        return float(micro_table.get(f"{_state_key(partial)}|{s_idx}", pkas[s_idx]))

    energies = microstate_free_energies(pkas, micro_pka_fn=_micro if micro_table else None)
    base_charge = sum(a.GetFormalCharge() for a in parent.GetAtoms())
    base_charge += sum(1 for s in fixed if s["kind"] == "base" and s["fixed_protonated"])
    base_charge -= sum(1 for s in fixed if s["kind"] == "acid" and not s["fixed_protonated"])
    micro = [
        {"state": [bool(b) for b in st], "charge": _state_charge(enumerated, st, base_charge), "g": round(g, 4)}
        for st, g in energies.items()
    ]
    flag_text = {
        "resonance_acceptor": "acceptor-substituted aromatic ring (charge delocalization; larger QM error expected)",
        "tertiary_amine": "tertiary amine (larger QM error expected, ~1-2.5 pKa units in the literature)",
        "polyaza_ring": "ring with several nitrogens (the reference compound is a poorer match)",
        "ortho_heteroatom": "electronegative substituent next to the ring N (larger QM error expected)",
        "acyl_substituted": "acyl/sulfonyl on the amidine/guanidine (much less basic; larger error expected)",
        "empirical_only": "no QM class for this group; empirical pKa used",
    }
    all_sites = sorted(enumerated + fixed, key=lambda x: x["atom"])
    for s in all_sites:
        for flag in s["flags"]:
            if flag in flag_text and (flag != "empirical_only" or method_used != "empirical"):
                warnings.append(f"site {s['atom']} ({s['class']}): {flag_text[flag]}")
    result = {
        "smiles": _chem().MolToSmiles(parent),
        "method": method_used,
        "sites": all_sites,
        "enumerated_atoms": [s["atom"] for s in enumerated],
        "fixed_sites": fixed,
        "microstates": micro,
        "macro_pkas": macroscopic_pkas(energies) if enumerated else [],
        "warnings": warnings,
        "provenance": {"method": method_used, "calibration_version": calibration_version,
                       "runtime_version": PKA_RUNTIME_VERSION, "cache": "miss"},
    }
    _cache_store(cache_dir, key, result)
    return result


def states_at_ph(result, ph, min_population=0.05):
    """Protonation states of a :func:`compute_pka` result at ``ph``, most populated first.
    Each: ``state`` (per enumerated site), ``population``, ``charge``."""
    energies = {tuple(m["state"]): float(m["g"]) for m in result.get("microstates", [])}
    if not energies:
        return [{"state": [], "population": 1.0, "charge": None}]
    pops = microstate_populations(energies, ph)
    charges = {tuple(m["state"]): m["charge"] for m in result["microstates"]}
    rows = [{"state": list(st), "population": p, "charge": charges[st]} for st, p in pops.items()]
    rows.sort(key=lambda r: -r["population"])
    keep = [r for r in rows if r["population"] >= float(min_population)]
    return keep or rows[:1]


def mean_charge(result, ph):
    energies = {tuple(m["state"]): float(m["g"]) for m in result.get("microstates", [])}
    if not energies:
        return None
    pops = microstate_populations(energies, ph)
    charges = {tuple(m["state"]): m["charge"] for m in result["microstates"]}
    return sum(p * charges[st] for st, p in pops.items())


def population_curves(result, ph_values, min_population=0.05):
    """
    Microstate populations and mean net charge over ``ph_values``, for plotting.

    Returns ``{"ph": [...], "mean_charge": [...], "states": [{"state", "charge",
    "populations"}]}``; microstates that never reach ``min_population`` are left out.
    """
    energies = {tuple(m["state"]): float(m["g"]) for m in result.get("microstates", [])}
    charges = {tuple(m["state"]): m["charge"] for m in result.get("microstates", [])}
    phs = [float(v) for v in ph_values]
    if not energies:
        return {"ph": phs, "mean_charge": [None] * len(phs), "states": []}
    table = {st: [] for st in energies}
    mean_q = []
    for ph in phs:
        pops = microstate_populations(energies, ph)
        for st in energies:
            table[st].append(pops[st])
        mean_q.append(sum(p * charges[st] for st, p in pops.items()))
    states = [{"state": list(st), "charge": charges[st], "populations": vals}
              for st, vals in table.items() if max(vals) >= float(min_population)]
    return {"ph": phs, "mean_charge": mean_q, "states": states}


def molecule_for_state(mol, result, state, embed_if_2d=True, seed=20260927):
    """
    Protonated molecule for one microstate of a :func:`compute_pka` result, with
    hydrogens. Heavy-atom coordinates of ``mol`` are kept; a 2D/coordinate-less input
    is embedded in 3D (ETKDG + MMFF) when ``embed_if_2d``.
    """
    parent = neutral_parent(mol)
    by_atom = {s["atom"]: s for s in result["sites"]}
    enumerated = [by_atom[a] for a in result["enumerated_atoms"]]
    out = build_state_molecule(parent, enumerated, list(state), fixed=result["fixed_sites"], add_hs=True)
    is_3d = out.GetNumConformers() > 0 and any(abs(p[2]) > 1e-4 for p in out.GetConformer().GetPositions())
    if embed_if_2d and not is_3d:
        from rdkit.Chem import AllChem

        out.RemoveAllConformers()
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        if AllChem.EmbedMolecule(out, params) == 0:
            try:
                AllChem.MMFFOptimizeMolecule(out, maxIters=2000)
            except Exception:
                pass
    return out


def protonate_molecule(mol, ph=7.4, method="empirical", cache_dir=None, add_hs=True, **kwargs):
    """
    Dominant protonation state of ``mol`` at ``ph``.

    Heavy atoms keep their indices and coordinates (hydrogens are rebuilt), so the
    result can replace a docking/parametrization input directly. Returns
    ``(molecule, info)``; ``info`` has the method, population of the chosen state,
    net charge, site pKa values and warnings.
    """
    Chem = _chem()
    parent = neutral_parent(mol)
    result = compute_pka(parent, method=method, focus_ph=ph, cache_dir=cache_dir, **kwargs)
    by_atom = {s["atom"]: s for s in result["sites"]}
    enumerated = [by_atom[a] for a in result["enumerated_atoms"]]
    best = states_at_ph(result, ph, min_population=0.0)[0]
    out = build_state_molecule(parent, enumerated, best["state"], fixed=result["fixed_sites"], add_hs=add_hs)
    info = {
        "method": result["method"],
        "ph": float(ph),
        "population": float(best["population"]),
        "charge": int(Chem.GetFormalCharge(out)),
        "input_charge": int(Chem.GetFormalCharge(mol)),
        "sites": [{k: s[k] for k in ("atom", "class", "kind", "pka", "sd", "flags")} for s in result["sites"]],
        "macro_pkas": result["macro_pkas"],
        "warnings": list(result["warnings"]),
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(out)),
    }
    if best["population"] < 0.6 and result["enumerated_atoms"]:
        info["warnings"].append(
            f"dominant state is only {100 * best['population']:.0f}% populated at pH {float(ph):.1f}"
        )
    return out, info


__all__ = [
    "PkaError",
    "SITE_CLASSES",
    "RT_LN10",
    "neutral_parent",
    "find_ionizable_sites",
    "select_enumerated_sites",
    "enumerate_microstates",
    "microstate_free_energies",
    "microstate_populations",
    "macroscopic_pkas",
    "fit_calibration",
    "apply_calibration",
    "build_state_molecule",
    "EmpiricalBackend",
    "MopacBackend",
    "PkaBackendError",
    "PkaCancelled",
    "load_calibration",
    "parse_mopac_heat_of_formation",
    "choose_shrinkage",
    "CALIBRATION_EXCLUDED_FLAGS",
    "register_backend",
    "compute_pka",
    "states_at_ph",
    "mean_charge",
    "protonate_molecule",
    "population_curves",
    "molecule_for_state",
]
