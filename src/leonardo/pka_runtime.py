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
PKA_RUNTIME_VERSION = "1"

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

_ACCEPTOR_ON_RING = "[$([N+](=O)[O-]),$(C#N),$([CX3]=O),$(S(=O)=O)]"


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
    if cls_name in ("pyridine", "imidazole"):
        atom = mol.GetAtomWithIdx(int(atom_idx))
        for ring in mol.GetRingInfo().AtomRings():
            if atom.GetIdx() in ring:
                n_count = sum(1 for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7)
                if n_count > (2 if cls_name == "imidazole" else 1):
                    flags.append("polyaza_ring")
                    break
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
def fit_calibration(rows, min_class_n=5):
    """
    Linear calibration of relative QM pKa values.

    ``rows``: dicts with ``class``, ``x`` = (dG_deprot(mol) - dG_deprot(ref)) / RT ln10
    and ``y`` = pKa_exp - pKa_ref. The model is ``y = slope * x`` (the reference itself
    maps to its experimental pKa). Classes with at least ``min_class_n`` points get
    their own slope, the rest use the global slope. Returns
    ``{"global": {...}, "classes": {name: {"slope", "rmse", "n", "own"}}}``.
    """
    def _fit(sub):
        sxx = sum(r["x"] * r["x"] for r in sub)
        slope = (sum(r["x"] * r["y"] for r in sub) / sxx) if sxx > 0 else 1.0
        rmse = math.sqrt(sum((r["y"] - slope * r["x"]) ** 2 for r in sub) / len(sub)) if sub else float("nan")
        return slope, rmse

    rows = [r for r in rows if r.get("x") is not None and r.get("y") is not None]
    g_slope, g_rmse = _fit(rows) if rows else (1.0, float("nan"))
    classes = {}
    for name in sorted({r["class"] for r in rows}):
        sub = [r for r in rows if r["class"] == name]
        if len(sub) >= int(min_class_n):
            slope, rmse = _fit(sub)
            classes[name] = {"slope": slope, "rmse": rmse, "n": len(sub), "own": True}
        else:
            rmse = math.sqrt(sum((r["y"] - g_slope * r["x"]) ** 2 for r in sub) / len(sub))
            classes[name] = {"slope": g_slope, "rmse": rmse, "n": len(sub), "own": False}
    return {"global": {"slope": g_slope, "rmse": g_rmse, "n": len(rows)}, "classes": classes}


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
class EmpiricalBackend:
    """Class-typical pKa per site; no QM. Uncertainty = class spread."""

    name = "empirical"

    def site_pkas(self, parent, sites, **_kwargs):
        return [
            {"pka": float(s["empirical_pka"]), "sd": float(s["empirical_sd"]), "source": "class_typical"}
            for s in sites
        ]


_BACKENDS = {"empirical": EmpiricalBackend}


def register_backend(name, factory):
    """Register a pKa backend (used by the MOPAC/Psi4 phases and by tests)."""
    _BACKENDS[str(name)] = factory


def _backend(method):
    key = str(method or "empirical").strip().lower()
    if key not in _BACKENDS:
        raise PkaError(f"unknown pKa method '{method}' (available: {', '.join(sorted(_BACKENDS))})")
    return _BACKENDS[key]()


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
                calibration_version="none", progress=None, cancel=None, **backend_kwargs):
    """
    Site and macroscopic pKa values plus the microstate energies of a molecule.

    Returns a JSON-serializable dict: ``sites`` (with ``pka``/``sd``/flags),
    ``fixed_sites``, ``microstates`` (pattern, charge, relative G), ``macro_pkas``,
    ``warnings`` and ``provenance``. Results are cached per molecule + method.
    """
    parent = neutral_parent(mol)
    key = _cache_key(parent, method, calibration_version)
    cached = _cache_load(cache_dir, key)
    if cached is not None:
        cached.setdefault("provenance", {})["cache"] = "hit"
        return cached
    backend = _backend(method)
    sites = find_ionizable_sites(parent)
    if callable(progress):
        progress(f"{len(sites)} ionizable site(s) found")
    values = backend.site_pkas(parent, sites, cancel=cancel, progress=progress, **backend_kwargs)
    for s, v in zip(sites, values):
        s["pka"] = float(v["pka"])
        s["sd"] = float(v.get("sd", s["empirical_sd"]))
        s["pka_source"] = str(v.get("source", backend.name))
    enumerated, fixed, warnings = select_enumerated_sites(sites, focus_ph=focus_ph, ph_range=ph_range)
    energies = microstate_free_energies([s["pka"] for s in enumerated])
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
        "empirical_only": "no QM class for this group; empirical pKa used",
    }
    for s in sites:
        for flag in s["flags"]:
            if flag in flag_text and (flag != "empirical_only" or backend.name != "empirical"):
                warnings.append(f"site {s['atom']} ({s['class']}): {flag_text[flag]}")
    result = {
        "smiles": _chem().MolToSmiles(parent),
        "method": backend.name,
        "sites": sites,
        "enumerated_atoms": [s["atom"] for s in enumerated],
        "fixed_sites": fixed,
        "microstates": micro,
        "macro_pkas": macroscopic_pkas(energies) if enumerated else [],
        "warnings": warnings,
        "provenance": {"method": backend.name, "calibration_version": str(calibration_version),
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
    "register_backend",
    "compute_pka",
    "states_at_ph",
    "mean_charge",
    "protonate_molecule",
]
