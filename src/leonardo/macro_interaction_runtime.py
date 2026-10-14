"""Interface interactions of macromolecular complexes: protein-protein, protein-DNA and protein-RNA.

Pure Python + numpy, no PyMOL needed.  Input is a PDB/mmCIF path or a list of
atom dicts (``name, resn, resi, chain, element, coord``; optional ``hetatm``,
``pymol_index``) plus two selections.  The geometric predicates come from
:mod:`leonardo.interaction_runtime` (the same H-bond, ring, cation-pi and metal
rules Leonardo uses for protein-ligand contacts); this module adds the
macromolecular chemistry: donor/acceptor and charge tables for amino acids and
nucleotides, nucleic-acid part (phosphate / sugar / base) and DNA groove of each
contact, water and metal bridges, a residue-level interface summary and a
Shrake-Rupley buried surface area.

Every interaction record has the same schema::

    {"type", "subtype", "a": {...atom ref...}, "b": {...atom ref...},
     "distance", "distance_def", "angle", "angle_def",
     "na_part", "na_groove", "readout", "via", "details", ...}

``a`` is always the side-A atom and ``b`` the side-B atom.  Like the
protein-ligand rules, the predicates are conservative: a missed borderline
contact is less harmful than a spurious one in an interface table.
"""

from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import numpy as np

from .interaction_runtime import (
    INTERACTION_GEOMETRY_CRITERIA,
    classify_hbond_geometry,
    classify_pi_ring_pair,
    interaction_metadata,
    metal_coordination_cutoff,
    protein_hydrophobic_atom_names,
    validate_metal_coordination,
    validate_ring_charge_interaction,
    vdw_radius,
)

# Ring, cation-pi, hydrophobic and metal thresholds are the protein-ligand ones
# (PLIP-style).  H-bonds between two macromolecules are judged on heavy atoms in
# most crystal structures; the 3.5 A D...A cap is the usual macromolecular one
# (Baker & Hubbard 1984; HBPLUS uses 3.9 A with hydrogens placed).  Salt bridges
# need both the charge centres within 5.5 A (PLIP) and a charged N...O pair
# within 4.0 A (Barlow & Thornton 1983), so long-range ion pairs are not counted.
MACRO_INTERFACE_CRITERIA = dict(INTERACTION_GEOMETRY_CRITERIA)
MACRO_INTERFACE_CRITERIA.update({
    "h_bond_heavy_dist": 3.5,        # heavy-atom fallback: D...A
    "h_bond_fallback_angle": 90.0,   # heavy-atom fallback: D...A-X (acceptor antecedent)
    "h_bond_donor_angle": 90.0,      # heavy-atom fallback: X-D...A (donor antecedent)
    "h_bond_explicit_dist": 3.9,     # explicit H: D...A
    "h_bond_ha_dist": 2.7,           # explicit H: H...A
    "h_bond_angle": 120.0,           # explicit H: D-H...A
    "salt_bridge_dist": 5.5,         # charge centre - charge centre
    "salt_bridge_no_dist": 4.0,      # closest charged N...O
    "water_bridge_dist": 3.5,        # water O ... polar N/O on each side
    "disulfide_dist": 2.5,           # SG-SG
    "interface_dist": 4.5,           # residue is at the interface: any heavy atom this close to the other side
    "contact_dist": 4.0,             # atom-atom contacts counted in the residue summary
    "sasa_probe": 1.4,
    "sasa_points": 192,
})

PROTEIN_RESN = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH",
    "LYN", "MSE", "SEP", "TPO", "PTR", "SEC", "PYL",
}
_NA_BASE = {"DA": "A", "DC": "C", "DG": "G", "DT": "T", "DU": "U", "DI": "I", "A": "A", "C": "C", "G": "G",
            "U": "U", "T": "T", "I": "I", "RA": "A", "RC": "C", "RG": "G", "RU": "U", "ADE": "A", "CYT": "C",
            "GUA": "G", "THY": "T", "URA": "U", "RA5": "A", "RA3": "A", "RC5": "C", "RC3": "C", "RG5": "G",
            "RG3": "G", "RU5": "U", "RU3": "U", "DA5": "A", "DA3": "A", "DC5": "C", "DC3": "C", "DG5": "G",
            "DG3": "G", "DT5": "T", "DT3": "T", "PSU": "U", "5MC": "C", "5MU": "U", "OMG": "G", "OMC": "C"}
WATER_RESN = {"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "TP3", "SOL", "SPC", "T3P"}
METAL_ELEMENTS = {"ZN", "MG", "CA", "MN", "FE", "CU", "NI", "CO", "NA", "K", "CD", "HG", "SR", "BA", "LI",
                  "RB", "CS", "PT", "AU", "AG", "PB", "TB", "EU", "GD", "YB", "W", "MO", "V", "CR"}

_PROTEIN_BACKBONE = {"N", "CA", "C", "O", "OXT", "H", "HA", "HA2", "HA3", "H1", "H2", "H3", "HN"}
_NA_PHOSPHATE = {"P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "O5'", "O3'"}

_PROTEIN_DONORS = {
    "ARG": ("NE", "NH1", "NH2"), "ASN": ("ND2",), "GLN": ("NE2",), "HIS": ("ND1", "NE2"), "HID": ("ND1",),
    "HIE": ("NE2",), "HIP": ("ND1", "NE2"), "HSD": ("ND1",), "HSE": ("NE2",), "HSP": ("ND1", "NE2"),
    "LYS": ("NZ",), "LYN": ("NZ",), "SER": ("OG",), "THR": ("OG1",), "TYR": ("OH",), "TRP": ("NE1",),
    "CYS": ("SG",), "ASH": ("OD2",), "GLH": ("OE2",),
}
_PROTEIN_ACCEPTORS = {
    "ASP": ("OD1", "OD2"), "ASH": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "GLH": ("OE1", "OE2"),
    "ASN": ("OD1",), "GLN": ("OE1",), "HIS": ("ND1", "NE2"), "HID": ("NE2",), "HIE": ("ND1",),
    "HSD": ("NE2",), "HSE": ("ND1",), "SER": ("OG",), "THR": ("OG1",), "TYR": ("OH",), "MET": ("SD",),
    "MSE": ("SE",), "SEP": ("O1P", "O2P", "O3P", "OG"), "TPO": ("O1P", "O2P", "O3P", "OG1"),
    "PTR": ("O1P", "O2P", "O3P", "OH"),
}
_NA_DONORS = {"A": ("N6",), "G": ("N1", "N2"), "C": ("N4",), "T": ("N3",), "U": ("N3",), "I": ("N1",)}
_NA_ACCEPTORS = {"A": ("N1", "N3", "N7"), "G": ("O6", "N3", "N7"), "C": ("O2", "N3"), "T": ("O2", "O4"),
                 "U": ("O2", "O4"), "I": ("O6", "N3", "N7")}
_NA_BACKBONE_ACCEPTORS = ("OP1", "OP2", "OP3", "O1P", "O2P", "O3P", "O3'", "O5'", "O4'", "O2'")

# DNA groove edge of each base atom (B-form); the Watson-Crick pairing atoms and ring interior have none.
DNA_GROOVE_ATOMS = {
    "major": {"A": {"N6", "N7", "C8"}, "G": {"O6", "N7", "C8"}, "C": {"N4", "C5", "C6"},
              "T": {"O4", "C7", "C5M", "C5", "C6"}, "U": {"O4", "C5", "C6"}, "I": {"O6", "N7", "C8"}},
    "minor": {"A": {"N3", "C2"}, "G": {"N2", "N3", "C2"}, "C": {"O2"}, "T": {"O2"}, "U": {"O2"},
              "I": {"N3", "C2"}},
}

_PROTEIN_RINGS = {
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "PTR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TRP": (("CG", "CD1", "NE1", "CE2", "CD2"), ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3")),
    "HIS": (("CG", "ND1", "CD2", "CE1", "NE2"),),
}
for _h in ("HID", "HIE", "HIP", "HSD", "HSE", "HSP"):
    _PROTEIN_RINGS[_h] = _PROTEIN_RINGS["HIS"]
_PURINE_RINGS = (("N1", "C2", "N3", "C4", "C5", "C6"), ("C4", "C5", "N7", "C8", "N9"))
_PYRIMIDINE_RINGS = (("N1", "C2", "N3", "C4", "C5", "C6"),)

# Interface hydrophobic atoms of nucleic acids: the thymine methyl (major groove) and the deoxyribose C2'.
_NA_HYDROPHOBIC = {"T": ("C7", "C5M")}
_DNA_SUGAR_HYDROPHOBIC = ("C2'",)

TYPE_ORDER = ("Salt Bridge", "Hydrogen Bond", "Cation-Pi", "Pi-Pi Stacking", "T-Shaped Pi-Pi", "Metal Bridge",
              "Disulfide", "Water Bridge", "Hydrophobic")
SPECIFIC_TYPES = ("Salt Bridge", "Hydrogen Bond", "Cation-Pi", "Pi-Pi Stacking", "T-Shaped Pi-Pi", "Metal Bridge",
                  "Disulfide")
_EXTRA_METADATA = {
    "Water Bridge": {"map_code": "WB", "physics_family": "water_mediated_polar",
                     "biological_meaning": "Ordered crystal water bridging polar atoms of both partners.",
                     "physical_meaning": "Two hydrogen bonds through a water molecule."},
    "Disulfide": {"map_code": "SS", "physics_family": "covalent",
                  "biological_meaning": "Covalent Cys-Cys cross-link between the partners.",
                  "physical_meaning": "S-S covalent bond."},
    "Metal Bridge": {"map_code": "MB", "physics_family": "coordination",
                     "biological_meaning": "Metal ion coordinated by atoms of both partners.",
                     "physical_meaning": "Shared coordination sphere with partial covalent character."},
}


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _guess_element(name, resn, hetatm):
    n = str(name or "").strip().upper()
    r = str(resn or "").strip().upper()
    if hetatm and n == r and n in METAL_ELEMENTS:
        return n
    letters = re.sub(r"[^A-Z]", "", n)
    if not letters:
        return ""
    if hetatm and letters[:2] in METAL_ELEMENTS and r in METAL_ELEMENTS:
        return letters[:2]
    return letters[0]


def _norm_name(name):
    return str(name or "").strip().replace("*", "'")


def _make_atom(name, resn, resi, chain, element, xyz, hetatm, alt, serial):
    name = _norm_name(name)
    resn = str(resn or "").strip().upper()
    element = str(element or "").strip().upper() or _guess_element(name, resn, hetatm)
    return {"name": name, "resn": resn, "resi": str(resi).strip(), "chain": str(chain or "").strip(),
            "element": element, "coord": np.asarray(xyz, dtype=float), "hetatm": bool(hetatm),
            "alt": str(alt or "").strip(), "serial": serial}


def parse_pdb_lines(lines):
    """ATOM/HETATM records of the first model; alternate locations other than blank/A are dropped."""
    atoms = []
    for raw in lines:
        line = str(raw).rstrip("\n")
        if line.startswith("ENDMDL") and atoms:
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        alt = line[16:17].strip()
        if alt not in ("", "A", "1"):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        resi = (line[22:26].strip() + line[26:27].strip())
        try:
            serial = int(line[6:11])
        except ValueError:
            serial = len(atoms) + 1
        atoms.append(_make_atom(line[12:16], line[17:20], resi, line[21:22], line[76:78] if len(line) > 76 else "",
                                xyz, line.startswith("HETATM"), alt, serial))
    return atoms


_CIF_TOKEN = re.compile(r"'(?:[^']|'(?=\S))*'|\"(?:[^\"]|\"(?=\S))*\"|\S+")


def parse_mmcif_text(text):
    """The ``_atom_site`` loop of an mmCIF file (author chain/residue numbering, first model)."""
    lines = str(text).splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_" and i + 1 < len(lines) and lines[i + 1].strip().startswith("_atom_site."):
            break
        i += 1
    else:
        return []
    i += 1
    fields = []
    while i < len(lines) and lines[i].strip().startswith("_atom_site."):
        fields.append(lines[i].strip().split()[0][len("_atom_site."):])
        i += 1
    tokens = []
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith(("loop_", "_", "#", "data_")):
            break
        tokens.extend(t[1:-1] if t[:1] in "'\"" and len(t) > 1 else t for t in _CIF_TOKEN.findall(s))
        i += 1
    col = {f: k for k, f in enumerate(fields)}
    nf = len(fields)

    def get(row, *names):
        for n in names:
            k = col.get(n)
            if k is not None and row[k] not in ("?", "."):
                return row[k]
        return ""

    atoms, first_model = [], None
    for start in range(0, len(tokens) - nf + 1, nf):
        row = tokens[start:start + nf]
        model = get(row, "pdbx_PDB_model_num") or "1"
        if first_model is None:
            first_model = model
        if model != first_model:
            break
        alt = get(row, "label_alt_id")
        if alt not in ("", "A", "1"):
            continue
        try:
            xyz = (float(get(row, "Cartn_x")), float(get(row, "Cartn_y")), float(get(row, "Cartn_z")))
        except ValueError:
            continue
        resi = get(row, "auth_seq_id", "label_seq_id") + get(row, "pdbx_PDB_ins_code")
        atoms.append(_make_atom(get(row, "auth_atom_id", "label_atom_id"), get(row, "auth_comp_id", "label_comp_id"),
                                resi, get(row, "auth_asym_id", "label_asym_id"), get(row, "type_symbol"), xyz,
                                get(row, "group_PDB") == "HETATM", alt, len(atoms) + 1))
    return atoms


def parse_structure(path):
    """Atoms of a PDB or mmCIF file (by extension, then by content)."""
    with open(os.fspath(path), encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    low = str(path).lower()
    if low.endswith((".cif", ".mmcif")) or ("_atom_site." in text and "\nATOM  " not in text[:200000]):
        return parse_mmcif_text(text)
    return parse_pdb_lines(text.splitlines())


# --------------------------------------------------------------------------- #
# residue chemistry
# --------------------------------------------------------------------------- #
def residue_kind(resn, atom_names=()):
    """protein / dna / rna / water / metal / other for one residue."""
    r = str(resn or "").strip().upper()
    names = set(atom_names or ())
    if r in WATER_RESN:
        return "water"
    if r in PROTEIN_RESN:
        return "protein"
    if r in _NA_BASE or ({"P", "C1'", "C4'"} <= names) or ({"C1'", "C4'", "N9"} <= names) or (
            {"C1'", "C4'", "N1", "C2"} <= names):
        if "O2'" in names or r in ("U", "RA", "RC", "RG", "RU", "PSU") or r.startswith(("RA", "RC", "RG", "RU")):
            return "rna"
        if r.startswith("D") or r in ("T", "THY"):
            return "dna"
        return "nucleic"      # A/C/G with no 2'-oxygen modelled: resolved by the chain (see _atom_kinds)
    if len(names) == 1 and r in METAL_ELEMENTS:
        return "metal"
    if {"N", "CA", "C"} <= names:
        return "protein"
    return "other"


def na_base(resn, atom_names=()):
    b = _NA_BASE.get(str(resn or "").strip().upper())
    if b:
        return b
    names = set(atom_names or ())
    if "N9" in names:
        return "G" if "O6" in names else "A"
    if "C7" in names or "C5M" in names:
        return "T"
    if "N4" in names:
        return "C"
    return "U" if names else ""


def atom_part(atom, kind):
    """backbone / side chain for amino acids, phosphate / sugar / base for nucleotides."""
    name = str(atom.get("name", "")).upper()
    if kind == "protein":
        return "backbone" if name in _PROTEIN_BACKBONE else "side chain"
    if kind in ("dna", "rna"):
        if name in _NA_PHOSPHATE:
            return "phosphate"
        if name.endswith("'") or name.endswith("''"):
            return "sugar"
        return "base"
    return kind


def dna_groove(base, atom_name):
    for groove in ("major", "minor"):
        if str(atom_name).upper() in DNA_GROOVE_ATOMS[groove].get(base, ()):
            return groove
    return None


def _readout(part, groove):
    if part == "phosphate":
        return "phosphate backbone"
    if part == "sugar":
        return "sugar"
    if part == "base":
        return f"base readout ({groove} groove)" if groove else "base"
    return None


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _pairs_within(xa, xb, cutoff, chunk=1024):
    """Index pairs (i, j, d) with |xa_i - xb_j| <= cutoff; chunked so memory stays bounded."""
    xa = np.asarray(xa, dtype=float).reshape(-1, 3)
    xb = np.asarray(xb, dtype=float).reshape(-1, 3)
    if not len(xa) or not len(xb):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    lo, hi = xb.min(axis=0) - cutoff, xb.max(axis=0) + cutoff
    keep = np.where(np.all((xa >= lo) & (xa <= hi), axis=1))[0]
    out_i, out_j, out_d = [], [], []
    c2 = cutoff * cutoff
    for s in range(0, len(keep), chunk):
        idx = keep[s:s + chunk]
        d2 = ((xa[idx, None, :] - xb[None, :, :]) ** 2).sum(-1)
        ii, jj = np.nonzero(d2 <= c2)
        out_i.append(idx[ii])
        out_j.append(jj)
        out_d.append(np.sqrt(d2[ii, jj]))
    if not out_i:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d)


def ring_geometry(coords):
    pts = np.asarray(coords, dtype=float)
    centroid = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - centroid)
    return centroid, vt[2]


def sphere_points(n):
    """Golden-spiral points on the unit sphere (Shrake-Rupley test points)."""
    n = max(12, int(n))
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def shrake_rupley(coords, radii, subset=None, probe=1.4, n_points=192):
    """Solvent-accessible area (A^2) of the atoms in ``subset`` (default: all), with every atom in
    ``coords`` occluding.  Numpy Shrake-Rupley with a golden-spiral test sphere."""
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    rad = np.asarray(radii, dtype=float) + float(probe)
    subset = np.arange(len(xyz)) if subset is None else np.asarray(subset, dtype=int)
    out = np.zeros(len(subset))
    if not len(subset):
        return out
    sphere = sphere_points(n_points)
    rmax = float(rad.max())
    ii, jj, _d = _pairs_within(xyz[subset], xyz, 2.0 * rmax)
    neigh = defaultdict(list)
    for a, b in zip(ii.tolist(), jj.tolist()):
        if subset[a] != b:
            neigh[a].append(b)
    for k, atom in enumerate(subset.tolist()):
        r = rad[atom]
        pts = xyz[atom] + r * sphere
        nb = np.asarray(neigh.get(k, []), dtype=int)
        if len(nb):
            dist = np.linalg.norm(xyz[nb] - xyz[atom], axis=1)
            nb = nb[dist < r + rad[nb]]
        if len(nb):
            d2 = ((pts[:, None, :] - xyz[nb][None, :, :]) ** 2).sum(-1)
            free = np.all(d2 >= rad[nb][None, :] ** 2, axis=1)
            n_free = int(free.sum())
        else:
            n_free = len(sphere)
        out[k] = 4.0 * np.pi * r * r * n_free / len(sphere)
    return out


# --------------------------------------------------------------------------- #
# selections
# --------------------------------------------------------------------------- #
_RES_TOKEN = re.compile(r"^(-?\d+)([A-Za-z]?)(?:-(-?\d+)([A-Za-z]?))?$")


def _resi_number(resi):
    m = re.match(r"^(-?\d+)", str(resi))
    return int(m.group(1)) if m else None


def select_side(atoms, spec, kinds=None):
    """Atom positions of a side.  ``spec`` may be:

    - ``"protein"``, ``"nucleic"``, ``"dna"``, ``"rna"``;
    - chain ids: ``"A"``, ``"A,B"``, ``"A+B"`` (``"chain A"`` also works);
    - residues: ``"A:39"``, ``"A:10-50,60"``, groups joined by ``;`` or ``+``;
    - a list of ``(chain, resi)`` tuples, or a list of atom positions (ints);
    - a callable ``atom -> bool``.
    """
    kinds = kinds if kinds is not None else _atom_kinds(atoms)
    n = len(atoms)
    if callable(spec):
        return [i for i in range(n) if spec(atoms[i])]
    if isinstance(spec, (list, tuple, set, frozenset, np.ndarray)):
        items = list(spec)
        if not items:
            return []
        if all(isinstance(x, (int, np.integer)) for x in items):
            return sorted({int(x) for x in items if 0 <= int(x) < n})
        wanted = {(str(c), str(r)) for c, r in items}
        return [i for i in range(n) if (atoms[i]["chain"], atoms[i]["resi"]) in wanted]
    text = str(spec or "").strip()
    if not text:
        raise ValueError("empty selection")
    low = text.lower()
    keyword = {"protein": ("protein",), "nucleic": ("dna", "rna"), "dna": ("dna",), "rna": ("rna",),
               "polymer": ("protein", "dna", "rna")}
    if low in keyword:
        return [i for i in range(n) if kinds[i] in keyword[low]]
    text = re.sub(r"(?i)\bchain\s+", "", text)
    chains = {a["chain"] for a in atoms}
    picked = set()
    for group in re.split(r"[;+]", text):
        group = group.strip()
        if not group:
            continue
        if ":" in group:
            ch, reslist = group.split(":", 1)
            ch = ch.strip()
            if ch not in chains:
                raise ValueError(f"no chain {ch!r} (chains: {', '.join(sorted(chains))})")
            singles, ranges = set(), []
            for item in reslist.split(","):
                item = item.strip()
                m = _RES_TOKEN.match(item)
                if not m:
                    raise ValueError(f"cannot read residue {item!r} in {spec!r}")
                if m.group(3) is None:
                    singles.add(item.upper())
                else:
                    ranges.append((int(m.group(1)), int(m.group(3))))
            for i, a in enumerate(atoms):
                if a["chain"] != ch:
                    continue
                num = _resi_number(a["resi"])
                if a["resi"].upper() in singles or (num is not None and any(lo <= num <= hi for lo, hi in ranges)):
                    picked.add(i)
        else:
            for ch in group.split(","):
                ch = ch.strip()
                if ch not in chains:
                    raise ValueError(f"no chain {ch!r} (chains: {', '.join(sorted(chains))})")
                picked.update(i for i, a in enumerate(atoms) if a["chain"] == ch)
    return sorted(picked)


def _residue_key(atom):
    return (atom["chain"], atom["resi"], atom["resn"])


def _atom_kinds(atoms):
    names = defaultdict(set)
    for a in atoms:
        names[_residue_key(a)].add(a["name"].upper())
    cache = {}
    for a in atoms:
        key = _residue_key(a)
        if key not in cache:
            kind = residue_kind(a["resn"], names[key])
            if kind == "other" and a.get("element", "").upper() in METAL_ELEMENTS and len(names[key]) == 1:
                kind = "metal"
            cache[key] = kind
    # a nucleotide without its 2'-oxygen modelled takes the type of its chain (RNA with a disordered O2')
    by_chain = defaultdict(Counter)
    for key, kind in cache.items():
        if kind in ("dna", "rna"):
            by_chain[key[0]][kind] += 1
    for key, kind in cache.items():
        if kind == "nucleic":
            c = by_chain.get(key[0])
            cache[key] = c.most_common(1)[0][0] if c else "dna"
    return [cache[_residue_key(a)] for a in atoms]


def default_sides(atoms, kinds=None):
    """(spec_a, spec_b, reason): protein vs nucleic acid when both exist; otherwise the first protein
    chain and the chain that touches it most."""
    kinds = kinds if kinds is not None else _atom_kinds(atoms)
    present = set(kinds)
    if "protein" in present and ({"dna", "rna"} & present):
        return "protein", "nucleic", "protein vs nucleic acid"
    chains = []
    for a, k in zip(atoms, kinds):
        if k in ("protein", "dna", "rna") and a["chain"] not in chains:
            chains.append(a["chain"])
    if len(chains) < 2:
        raise ValueError(f"need two polymer chains or a protein-nucleic acid complex (chains: {', '.join(chains) or 'none'})")
    heavy = [i for i, (a, k) in enumerate(zip(atoms, kinds)) if k in ("protein", "dna", "rna") and a["element"] != "H"]
    xyz = np.array([atoms[i]["coord"] for i in heavy])
    ch = np.array([atoms[i]["chain"] for i in heavy])
    first = chains[0]
    ia = np.where(ch == first)[0]
    ib = np.where(ch != first)[0]
    _i, j, _d = _pairs_within(xyz[ia], xyz[ib], MACRO_INTERFACE_CRITERIA["interface_dist"])
    counts = Counter(ch[ib][j].tolist())
    if not counts:
        raise ValueError(f"chain {first} touches no other chain; give selection_a and selection_b")
    partner = counts.most_common(1)[0][0]
    return first, partner, f"chain {first} vs chain {partner} (its largest contact partner)"


# --------------------------------------------------------------------------- #
# the analysis
# --------------------------------------------------------------------------- #
class _Interface:
    def __init__(self, atoms, side_a, side_b, criteria):
        self.atoms = atoms
        self.crit = criteria
        self.kinds = _atom_kinds(atoms)
        self.xyz = np.array([a["coord"] for a in atoms], dtype=float).reshape(-1, 3)
        self.elem = [str(a.get("element", "")).upper() for a in atoms]
        res_names = defaultdict(set)
        self.res_atoms = defaultdict(dict)
        for i, a in enumerate(atoms):
            res_names[_residue_key(a)].add(a["name"].upper())
            self.res_atoms[_residue_key(a)].setdefault(a["name"].upper(), i)
        self.res_names = res_names
        self.res_has_h = {k: any(self.elem[i] in ("H", "D") for i in v.values()) for k, v in self.res_atoms.items()}
        self.hydrogens_present = any(self.res_has_h.values())

        def usable(i):
            return self.kinds[i] not in ("water", "metal") and self.elem[i] not in ("H", "D")
        self.side_a = [i for i in side_a if usable(i)]
        self.side_b = [i for i in side_b if usable(i)]
        overlap = set(self.side_a) & set(self.side_b)
        if overlap:
            raise ValueError(f"the two selections share {len(overlap)} atoms")
        if not self.side_a or not self.side_b:
            raise ValueError("a selection has no heavy atoms (waters, metal ions and hydrogens are not sides)")
        self.side_of = {}
        for i in self.side_a:
            self.side_of[i] = "A"
        for i in self.side_b:
            self.side_of[i] = "B"
        self._neighbors = {}
        self._hydrogens = None
        self.ia = np.array(self.side_a)
        self.ib = np.array(self.side_b)
        pi, pj, pd = _pairs_within(self.xyz[self.ia], self.xyz[self.ib], 4.6)
        self.close_pairs = list(zip(self.ia[pi].tolist(), self.ib[pj].tolist(), pd.tolist()))

    # --- atom descriptors -------------------------------------------------- #
    def base(self, i):
        a = self.atoms[i]
        return na_base(a["resn"], self.res_names[_residue_key(a)])

    def part(self, i):
        return atom_part(self.atoms[i], self.kinds[i])

    def ref(self, i, atoms_label=None):
        a = self.atoms[i]
        out = {"chain": a["chain"], "resn": a["resn"], "resi": a["resi"], "atom": atoms_label or a["name"],
               "residue": residue_label(a), "part": self.part(i), "kind": self.kinds[i]}
        if a.get("pymol_index") is not None:
            out["pymol_index"] = int(a["pymol_index"])
        return out

    def na_info(self, i):
        kind = self.kinds[i]
        if kind not in ("dna", "rna"):
            return None, None
        part = self.part(i)
        groove = dna_groove(self.base(i), self.atoms[i]["name"]) if (kind == "dna" and part == "base") else None
        return part, groove

    def neighbors(self, i):
        """Covalently bonded heavy atoms (distance-based; same chain)."""
        if i not in self._neighbors:
            d = np.linalg.norm(self.xyz - self.xyz[i], axis=1)
            cand = np.where((d > 0.4) & (d < 2.0))[0]
            ch = self.atoms[i]["chain"]
            self._neighbors[i] = [int(j) for j in cand if self.elem[j] not in ("H", "D") and
                                  self.atoms[j]["chain"] == ch and self.kinds[j] not in ("water", "metal")]
        return self._neighbors[i]

    def bound_hydrogens(self, i):
        if self._hydrogens is None:
            hs = [k for k, e in enumerate(self.elem) if e in ("H", "D")]
            self._hydrogens = (np.array(hs, dtype=int), self.xyz[hs] if hs else np.zeros((0, 3)))
        idx, xyz = self._hydrogens
        if not len(idx):
            return []
        d = np.linalg.norm(xyz - self.xyz[i], axis=1)
        return idx[d < 1.25].tolist()

    def is_donor(self, i):
        a, kind = self.atoms[i], self.kinds[i]
        name, resn = a["name"].upper(), a["resn"]
        if kind == "protein":
            if name == "N":
                return resn != "PRO" or self._is_nterm(i)
            return name in _PROTEIN_DONORS.get(resn, ())
        if kind in ("dna", "rna"):
            return name in _NA_DONORS.get(self.base(i), ()) or name == "O2'"
        return False

    def is_acceptor(self, i):
        a, kind = self.atoms[i], self.kinds[i]
        name = a["name"].upper()
        if kind == "protein":
            return name in ("O", "OXT") or name in _PROTEIN_ACCEPTORS.get(a["resn"], ())
        if kind in ("dna", "rna"):
            return name in _NA_BACKBONE_ACCEPTORS or name in _NA_ACCEPTORS.get(self.base(i), ())
        return False

    def is_polar(self, i):
        return self.elem[i] in ("N", "O") and (self.is_donor(i) or self.is_acceptor(i))

    def _is_nterm(self, i):
        return not any(self.elem[j] == "C" and self.atoms[j]["name"] == "C" for j in self.neighbors(i))

    # --- detectors --------------------------------------------------------- #
    def hbond(self, donor, acceptor):
        """Best H-bond geometry donor -> acceptor, or None."""
        crit = self.crit
        if not (self.is_donor(donor) and self.is_acceptor(acceptor)):
            return None
        da = float(np.linalg.norm(self.xyz[donor] - self.xyz[acceptor]))
        d_atom, a_atom = self.atoms[donor], self.atoms[acceptor]
        if self.res_has_h.get(_residue_key(d_atom)):
            best = None
            for h in self.bound_hydrogens(donor):
                res = classify_hbond_geometry(d_atom, a_atom, hydrogen=self.atoms[h], criteria={
                    "h_bond_dist": crit["h_bond_explicit_dist"], "h_bond_ha_dist": crit["h_bond_ha_dist"],
                    "h_bond_angle": crit["h_bond_angle"]})
                if res.get("ok") and (best is None or res["ha_distance"] < best["ha_distance"]):
                    best = dict(res, hydrogen=self.atoms[h]["name"])
            if best is None:
                return None
            return {"distance": best["ha_distance"], "distance_def": "H···A", "da_distance": da,
                    "angle": best["angle"], "angle_def": "D–H···A", "mode": "explicit_hydrogen",
                    "hydrogen": best["hydrogen"]}
        if da > crit["h_bond_heavy_dist"]:
            return None
        worst_acc = 180.0
        for x in self.neighbors(acceptor):
            res = classify_hbond_geometry(d_atom, a_atom, acceptor_neighbor=self.atoms[x], criteria={
                "h_bond_dist": crit["h_bond_heavy_dist"], "h_bond_fallback_angle": crit["h_bond_fallback_angle"]})
            if not res.get("ok"):
                return None
            worst_acc = min(worst_acc, float(res["angle"]))
        for x in self.neighbors(donor):
            v1 = self.xyz[x] - self.xyz[donor]
            v2 = self.xyz[acceptor] - self.xyz[donor]
            ang = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                                                     -1.0, 1.0))))
            if ang < crit["h_bond_donor_angle"]:
                return None
        return {"distance": da, "distance_def": "D···A", "da_distance": da, "angle": worst_acc,
                "angle_def": "D···A–X (min over acceptor neighbours)", "mode": "heavy_atom"}

    def find_hbonds(self):
        out = []
        for i, j, _d in self.close_pairs:
            if self.elem[i] not in ("N", "O", "S") or self.elem[j] not in ("N", "O", "S"):
                continue
            options = []
            g = self.hbond(i, j)
            if g:
                options.append(("A", g))
            g = self.hbond(j, i)
            if g:
                options.append(("B", g))
            if not options:
                continue
            donor_side, g = min(options, key=lambda o: o[1]["distance"])
            rec = self.record("Hydrogen Bond", i, j, g["distance"], g["distance_def"], g["angle"], g["angle_def"],
                              donor_side=donor_side, mode=g["mode"], da_distance=round(g["da_distance"], 3),
                              details=f"{g['distance_def']} {g['distance']:.2f} Å, {g['angle_def'].split(' (')[0]} "
                                      f"{g['angle']:.0f}°")
            if g.get("hydrogen"):
                rec["hydrogen"] = g["hydrogen"]
            out.append(rec)
        return out

    def charge_groups(self, side):
        """(+/-, label, member atoms, centre, conditional) of one side."""
        groups = []
        seen = set()
        for i in (self.side_a if side == "A" else self.side_b):
            key = _residue_key(self.atoms[i])
            if key in seen:
                continue
            seen.add(key)
            names = self.res_atoms[key]
            kind = self.kinds[i]
            resn = key[2]

            def add(sign, label, members, conditional=None):
                idx = [names[m] for m in members if m in names]
                if idx and len(idx) == len([m for m in members if m in names]):
                    groups.append({"sign": sign, "label": label, "atoms": idx,
                                   "centre": self.xyz[idx].mean(axis=0), "conditional": conditional})
            if kind == "protein":
                if resn in ("LYS",) and "NZ" in names:
                    add(+1, "ammonium", ["NZ"])
                elif resn == "ARG" and {"NH1", "NH2"} <= set(names):
                    add(+1, "guanidinium", [n for n in ("NE", "NH1", "NH2") if n in names])
                elif resn in ("HIS", "HIP", "HSP", "HID", "HIE", "HSD", "HSE") and {"ND1", "NE2"} <= set(names):
                    has_h = self.res_has_h.get(key)
                    if resn in ("HIP", "HSP") or (has_h and {"HD1", "HE2"} <= set(names)):
                        add(+1, "imidazolium", ["ND1", "NE2"])
                    elif resn == "HIS" and not has_h:
                        add(+1, "imidazolium", ["ND1", "NE2"], conditional="His assumed protonated")
                elif resn == "ASP" and {"OD1", "OD2"} <= set(names):
                    add(-1, "carboxylate", ["OD1", "OD2"])
                elif resn == "GLU" and {"OE1", "OE2"} <= set(names):
                    add(-1, "carboxylate", ["OE1", "OE2"])
                if "OXT" in names and "O" in names:
                    add(-1, "C-terminal carboxylate", ["O", "OXT"])
            elif kind in ("dna", "rna"):
                ops = [n for n in ("OP1", "OP2", "O1P", "O2P", "OP3", "O3P") if n in names]
                if len(ops) >= 2:
                    add(-1, "phosphate", ops)
        return groups

    def find_salt_bridges(self):
        out = []
        ga, gb = self.charge_groups("A"), self.charge_groups("B")
        for p in ga:
            for q in gb:
                if p["sign"] * q["sign"] >= 0:
                    continue
                centre = float(np.linalg.norm(p["centre"] - q["centre"]))
                if centre > self.crit["salt_bridge_dist"]:
                    continue
                pairs = [(float(np.linalg.norm(self.xyz[x] - self.xyz[y])), x, y) for x in p["atoms"] for y in q["atoms"]
                         if {self.elem[x], self.elem[y]} == {"N", "O"}]
                if not pairs:
                    continue
                dmin, x, y = min(pairs)
                if dmin > self.crit["salt_bridge_no_dist"]:
                    continue
                pos, neg = (p, q) if p["sign"] > 0 else (q, p)
                rec = self.record("Salt Bridge", x, y, dmin, "N···O (closest charged atoms)", None, None,
                                  centre_distance=round(centre, 3), cation_group=pos["label"],
                                  anion_group=neg["label"],
                                  details=f"N···O {dmin:.2f} Å, charge centres {centre:.2f} Å "
                                          f"({pos['label']}–{neg['label']})")
                cond = p["conditional"] or q["conditional"]
                if cond:
                    rec["conditional"] = cond
                out.append(rec)
        return out

    def hydrophobic_atoms(self, side):
        out = []
        for i in (self.side_a if side == "A" else self.side_b):
            a, kind = self.atoms[i], self.kinds[i]
            name = a["name"].upper()
            if self.elem[i] != "C":
                continue
            if kind == "protein" and name in protein_hydrophobic_atom_names(a["resn"]):
                out.append(i)
            elif kind in ("dna", "rna") and (name in _NA_HYDROPHOBIC.get(self.base(i), ()) or
                                             (kind == "dna" and name in _DNA_SUGAR_HYDROPHOBIC)):
                out.append(i)
        return out

    def find_hydrophobic(self):
        ha, hb = set(self.hydrophobic_atoms("A")), set(self.hydrophobic_atoms("B"))
        best = {}
        npairs = Counter()
        for i, j, d in self.close_pairs:
            if i in ha and j in hb and d <= self.crit["hydrophobic_dist"]:
                key = (_residue_key(self.atoms[i]), _residue_key(self.atoms[j]))
                npairs[key] += 1
                if key not in best or d < best[key][2]:
                    best[key] = (i, j, d)
        return [self.record("Hydrophobic", i, j, d, "C···C", None, None, atom_pairs=npairs[key],
                            details=f"C···C {d:.2f} Å ({npairs[key]} carbon pair(s) ≤ "
                                    f"{self.crit['hydrophobic_dist']:.1f} Å)")
                for key, (i, j, d) in best.items()]

    def rings(self, side):
        out = []
        seen = set()
        for i in (self.side_a if side == "A" else self.side_b):
            key = _residue_key(self.atoms[i])
            if key in seen:
                continue
            seen.add(key)
            kind = self.kinds[i]
            names = self.res_atoms[key]
            if kind == "protein":
                templates = _PROTEIN_RINGS.get(key[2], ())
            elif kind in ("dna", "rna"):
                templates = _PURINE_RINGS if self.base(i) in ("A", "G", "I") else _PYRIMIDINE_RINGS
            else:
                continue
            for tpl in templates:
                if all(n in names for n in tpl):
                    idx = [names[n] for n in tpl]
                    c, nrm = ring_geometry(self.xyz[idx])
                    out.append({"atoms": idx, "centroid": c, "normal": nrm, "size": len(tpl),
                                "label": f"ring({'/'.join(tpl)})", "key": key})
        return out

    def _closest_ring_atom(self, ring, point):
        d = np.linalg.norm(self.xyz[ring["atoms"]] - point, axis=1)
        return ring["atoms"][int(np.argmin(d))]

    def find_pi(self):
        best = {}
        for ra in self.rings("A"):
            for rb in self.rings("B"):
                if np.linalg.norm(ra["centroid"] - rb["centroid"]) > self.crit["pi_pi_dist"]:
                    continue
                res = classify_pi_ring_pair(ra, rb, criteria=self.crit)
                if not res.get("ok"):
                    continue
                key = (ra["key"], rb["key"], res["type"])
                if key not in best or res["distance"] < best[key][2]["distance"]:
                    best[key] = (ra, rb, res)
        out = []
        for ra, rb, res in best.values():
            i = self._closest_ring_atom(ra, rb["centroid"])
            j = self._closest_ring_atom(rb, ra["centroid"])
            rec = self.record(res["type"], i, j, res["distance"], "ring centroid–centroid", res["angle"],
                              "ring normals", offset=round(float(res["offset"]), 3), details=res["details"])
            _ring_face(rec)
            rec["a"]["atom"], rec["b"]["atom"] = ra["label"], rb["label"]
            rec["a"]["representative_atom"] = self.atoms[i]["name"]
            rec["b"]["representative_atom"] = self.atoms[j]["name"]
            out.append(rec)
        return out

    def cations(self, side):
        out = []
        for i in (self.side_a if side == "A" else self.side_b):
            a = self.atoms[i]
            if self.kinds[i] != "protein":
                continue
            name = a["name"].upper()
            if (a["resn"] in ("LYS",) and name == "NZ") or (a["resn"] == "ARG" and name == "CZ"):
                out.append(i)
        return out

    def find_cation_pi(self):
        out = []
        for cat_side, ring_side in (("A", "B"), ("B", "A")):
            rings = self.rings(ring_side)
            best = {}
            for c in self.cations(cat_side):
                for ring in rings:
                    res = validate_ring_charge_interaction(ring, self.xyz[c], criteria=self.crit)
                    if not res.get("ok"):
                        continue
                    key = (_residue_key(self.atoms[c]), ring["key"])
                    if key not in best or res["distance"] < best[key][2]["distance"]:
                        best[key] = (c, ring, res)
            for c, ring, res in best.values():
                r_atom = self._closest_ring_atom(ring, self.xyz[c])
                i, j = (c, r_atom) if cat_side == "A" else (r_atom, c)
                rec = self.record("Cation-Pi", i, j, res["distance"], "cation···ring centroid", None, None,
                                  offset=round(float(res["offset"]), 3),
                                  plane_distance=round(float(res["plane_distance"]), 3), cation_side=cat_side,
                                  details=f"cation···centroid {res['details_suffix']}")
                _ring_face(rec)
                ring_ref = rec["b"] if cat_side == "A" else rec["a"]
                ring_ref["representative_atom"] = ring_ref["atom"]
                ring_ref["atom"] = ring["label"]
                out.append(rec)
        return out

    def find_disulfides(self):
        out = []
        for i, j, d in self.close_pairs:
            a, b = self.atoms[i], self.atoms[j]
            if (a["name"].upper() == "SG" and b["name"].upper() == "SG" and self.kinds[i] == self.kinds[j] == "protein"
                    and d <= self.crit["disulfide_dist"]):
                out.append(self.record("Disulfide", i, j, d, "SG–SG", None, None, details=f"S–S {d:.2f} Å"))
        return out

    def _polar_side_atoms(self, side):
        return [i for i in (self.side_a if side == "A" else self.side_b) if self.is_polar(i)]

    def find_water_bridges(self):
        waters = [i for i, k in enumerate(self.kinds) if k == "water" and self.elem[i] == "O"]
        if not waters:
            return []
        wx = self.xyz[waters]
        cut = self.crit["water_bridge_dist"]
        pa, pb = self._polar_side_atoms("A"), self._polar_side_atoms("B")
        wa_i, wa_j, wa_d = _pairs_within(wx, self.xyz[pa], cut)
        wb_i, wb_j, wb_d = _pairs_within(wx, self.xyz[pb], cut)
        near_a, near_b = {}, {}
        for w, j, d in zip(wa_i.tolist(), wa_j.tolist(), wa_d.tolist()):
            if w not in near_a or d < near_a[w][1]:
                near_a[w] = (pa[j], d)
        for w, j, d in zip(wb_i.tolist(), wb_j.tolist(), wb_d.tolist()):
            if w not in near_b or d < near_b[w][1]:
                near_b[w] = (pb[j], d)
        out = []
        for w in sorted(set(near_a) & set(near_b)):
            (i, da), (j, db) = near_a[w], near_b[w]
            wi = waters[w]
            rec = self.record("Water Bridge", i, j, max(da, db), "A···W and W···B (longer of the two)", None, None,
                              details=f"via {residue_label(self.atoms[wi])}: {da:.2f} / {db:.2f} Å",
                              optional=True)
            rec["via"] = dict(self.ref(wi), distance_a=round(da, 3), distance_b=round(db, 3))
            out.append(rec)
        return out

    def find_metal_bridges(self):
        metals = [i for i, k in enumerate(self.kinds) if k == "metal" or (self.elem[i] in METAL_ELEMENTS and
                                                                          k not in ("protein", "dna", "rna"))]
        out = []
        for m in metals:
            cutoff = metal_coordination_cutoff(self.elem[m], criteria=self.crit)
            best = {}
            for side, idx in (("A", self.ia), ("B", self.ib)):
                d = np.linalg.norm(self.xyz[idx] - self.xyz[m], axis=1)
                for k in np.where(d <= cutoff)[0].tolist():
                    i = int(idx[k])
                    res = validate_metal_coordination(self.atoms[m], self.atoms[i], criteria=self.crit)
                    if res.get("ok") and (side not in best or res["distance"] < best[side][1]):
                        best[side] = (i, res["distance"])
            if "A" in best and "B" in best:
                (i, da), (j, db) = best["A"], best["B"]
                rec = self.record("Metal Bridge", i, j, max(da, db), "M···X (longer of the two)", None, None,
                                  details=f"{self.elem[m]} {residue_label(self.atoms[m])}: {da:.2f} / {db:.2f} Å")
                rec["via"] = dict(self.ref(m), element=self.elem[m], distance_a=round(da, 3), distance_b=round(db, 3))
                out.append(rec)
        return out

    # --- record ------------------------------------------------------------ #
    def record(self, itype, i, j, distance, distance_def, angle, angle_def, details="", **extra):
        rec = {"type": itype, "subtype": f"{self.part(i)}–{self.part(j)}", "a": self.ref(i), "b": self.ref(j),
               "distance": round(float(distance), 3), "distance_def": distance_def,
               "angle": None if angle is None else round(float(angle), 1), "angle_def": angle_def,
               "na_part": None, "na_groove": None, "readout": None, "via": None, "details": details}
        for k in (j, i):
            part, groove = self.na_info(k)
            if part:
                rec.update(na_part=part, na_groove=groove, readout=_readout(part, groove), na_side="B" if k == j else "A")
                break
        meta = interaction_metadata(itype) or _EXTRA_METADATA.get(itype, {})
        if meta.get("map_code"):
            rec["map_code"] = meta["map_code"]
        rec.update(extra)
        return rec


def _ring_face(rec):
    """A ring contact uses the base face, not a groove edge: no groove, readout 'base (stacking)'."""
    if rec.get("na_part") == "base":
        rec["na_groove"] = None
        rec["readout"] = "base (stacking)"


def residue_label(atom):
    return f"{atom.get('resn', '?')}{atom.get('resi', '?')}:{atom.get('chain', '') or '-'}"


def _side_kind(kinds, idx):
    c = Counter(kinds[i] for i in idx if kinds[i] in ("protein", "dna", "rna"))
    if not c:
        return "other"
    if len(c) == 1:
        return next(iter(c))
    if set(c) <= {"dna", "rna"}:
        return "nucleic"
    return "mixed"


def _interface_type(ka, kb):
    names = {"protein": "protein", "dna": "DNA", "rna": "RNA", "nucleic": "nucleic acid", "mixed": "mixed",
             "other": "other"}
    pair = sorted([ka, kb], key=lambda k: k != "protein")
    return f"{names[pair[0]]}–{names[pair[1]]}"


def _sort_key(rec):
    order = TYPE_ORDER.index(rec["type"]) if rec["type"] in TYPE_ORDER else len(TYPE_ORDER)
    return (order, rec["distance"])


def analyze_interface(structure, side_a=None, side_b=None, *, criteria=None, include_waters=True,
                      compute_sasa=True, sasa_points=None):
    """Interface interactions between two selections of a macromolecular complex.

    ``structure`` is a PDB/mmCIF path or a list of atom dicts.  ``side_a``/``side_b`` follow
    :func:`select_side`; when both are omitted, protein vs nucleic acid (if present) or the first
    chain vs its largest contact partner.
    """
    atoms = parse_structure(structure) if isinstance(structure, (str, os.PathLike)) else list(structure or [])
    for a in atoms:
        a["coord"] = np.asarray(a["coord"], dtype=float)
        a.setdefault("chain", "")
        a["resi"] = str(a.get("resi", "")).strip()
        a["resn"] = str(a.get("resn", "")).strip().upper()
        a["name"] = _norm_name(a.get("name", ""))
        if not str(a.get("element", "") or "").strip():
            a["element"] = _guess_element(a["name"], a["resn"], a.get("hetatm", False))
        a["element"] = str(a["element"]).strip().upper()
    if not atoms:
        raise ValueError("no atoms")
    crit = dict(MACRO_INTERFACE_CRITERIA)
    if isinstance(criteria, dict):
        crit.update(criteria)
    kinds = _atom_kinds(atoms)
    notes = []
    if side_a in (None, "") and side_b in (None, ""):
        side_a, side_b, reason = default_sides(atoms, kinds)
        notes.append(f"default selections: {reason}")
    elif side_a in (None, "") or side_b in (None, ""):
        given = side_a if side_a not in (None, "") else side_b
        idx = set(select_side(atoms, given, kinds))
        other = [i for i, k in enumerate(kinds) if k in ("protein", "dna", "rna") and i not in idx]
        if side_a in (None, ""):
            side_a, side_b = other, given
        else:
            side_b = other
        notes.append("the missing selection is every other polymer atom")
    ia = select_side(atoms, side_a, kinds)
    ib = select_side(atoms, side_b, kinds)
    iface = _Interface(atoms, ia, ib, crit)

    records = iface.find_hbonds() + iface.find_salt_bridges() + iface.find_pi() + iface.find_cation_pi() + \
        iface.find_disulfides() + iface.find_metal_bridges() + iface.find_hydrophobic()
    waters_present = any(k == "water" for k in kinds)
    if include_waters:
        records += iface.find_water_bridges()
        if not waters_present:
            notes.append("no crystal waters in the structure: water bridges not evaluated")
    if not iface.hydrogens_present:
        notes.append("no hydrogens: H-bonds judged on heavy atoms (D···A distance and antecedent angles)")
    records.sort(key=_sort_key)
    # salt bridges that are also direct H-bonds: both are reported; flag the H-bond for readers
    sb_pairs = {(r["a"]["residue"], r["b"]["residue"]) for r in records if r["type"] == "Salt Bridge"}
    for r in records:
        if r["type"] == "Hydrogen Bond" and (r["a"]["residue"], r["b"]["residue"]) in sb_pairs:
            r["also_salt_bridge"] = True

    # residue-level interface
    side_kind_a, side_kind_b = _side_kind(kinds, iface.side_a), _side_kind(kinds, iface.side_b)
    res_pair = {}
    for i, j, d in iface.close_pairs:
        if d > crit["interface_dist"]:
            continue
        key = (residue_label(atoms[i]), residue_label(atoms[j]))
        e = res_pair.setdefault(key, {"a": key[0], "b": key[1], "min_distance": d, "n_atom_contacts": 0,
                                      "types": set()})
        e["min_distance"] = min(e["min_distance"], d)
        if d <= crit["contact_dist"]:
            e["n_atom_contacts"] += 1
    for r in records:
        key = (r["a"]["residue"], r["b"]["residue"])
        e = res_pair.setdefault(key, {"a": key[0], "b": key[1], "min_distance": r["distance"], "n_atom_contacts": 0,
                                      "types": set()})
        e["types"].add(r["type"])
    residue_contacts = []
    for e in res_pair.values():
        e = dict(e, types=sorted(e["types"], key=lambda t: TYPE_ORDER.index(t) if t in TYPE_ORDER else 99),
                 min_distance=round(float(e["min_distance"]), 3))
        e["n_specific"] = sum(1 for t in e["types"] if t in SPECIFIC_TYPES)
        residue_contacts.append(e)
    residue_contacts.sort(key=lambda e: (-e["n_specific"], -e["n_atom_contacts"], e["min_distance"]))
    interface_res = {"A": set(), "B": set()}
    for i, j, d in iface.close_pairs:
        if d <= crit["interface_dist"]:
            interface_res["A"].add(i)
            interface_res["B"].add(j)
    res_info = {}
    for side in ("A", "B"):
        for i in interface_res[side]:
            a = atoms[i]
            res_info.setdefault(residue_label(a), {"residue": residue_label(a), "side": side, "chain": a["chain"],
                                                   "resn": a["resn"], "resi": a["resi"], "kind": kinds[i]})

    # buried surface (heavy atoms of the two sides; waters/ions excluded)
    sasa = None
    if compute_sasa:
        sasa = _buried_sasa(iface, crit, int(sasa_points or crit["sasa_points"]))
        per_residue = sasa.pop("_per_residue")
        for label, info in res_info.items():
            info["buried_sasa"] = round(per_residue.get(label, 0.0), 1)
        # residues that lose area without an atom within interface_dist (second shell) are only counted
        sasa["n_residues_buried_beyond_cutoff"] = sum(1 for k, v in per_residue.items()
                                                       if v >= 1.0 and k not in res_info)

    # hotspot-style counts
    per_res = defaultdict(Counter)
    for r in records:
        for side in ("a", "b"):
            per_res[r[side]["residue"]][r["type"]] += 1
    contacts_per_res = Counter()
    partners = defaultdict(set)
    for e in residue_contacts:
        contacts_per_res[e["a"]] += e["n_atom_contacts"]
        contacts_per_res[e["b"]] += e["n_atom_contacts"]
        partners[e["a"]].add(e["b"])
        partners[e["b"]].add(e["a"])
    hotspots = []
    for label, info in res_info.items():
        c = per_res.get(label, Counter())
        hotspots.append(dict(info, counts=dict(c), n_specific=sum(c[t] for t in SPECIFIC_TYPES),
                             n_atom_contacts=contacts_per_res.get(label, 0), n_partners=len(partners.get(label, ()))))
    hotspots.sort(key=lambda h: (-h["n_specific"], -h.get("buried_sasa", 0.0), -h["n_atom_contacts"]))

    nucleic = _nucleic_summary(iface, records, crit) if (side_kind_a in ("dna", "rna", "nucleic") or
                                                         side_kind_b in ("dna", "rna", "nucleic")) else None
    counts = Counter(r["type"] for r in records)
    return {
        "interface_type": _interface_type(side_kind_a, side_kind_b),
        "sides": {
            "A": {"selection": _spec_text(side_a), "kind": side_kind_a, "n_atoms": len(iface.side_a),
                  "chains": sorted({atoms[i]["chain"] for i in iface.side_a})},
            "B": {"selection": _spec_text(side_b), "kind": side_kind_b, "n_atoms": len(iface.side_b),
                  "chains": sorted({atoms[i]["chain"] for i in iface.side_b})},
        },
        "counts": {t: counts[t] for t in TYPE_ORDER if counts.get(t)},
        "interactions": records,
        "interface_residues": {s: sorted([h["residue"] for h in hotspots if h["side"] == s], key=_res_sort)
                               for s in ("A", "B")},
        "residue_contacts": residue_contacts,
        "hotspots": hotspots,
        "sasa": sasa,
        "nucleic_acid": nucleic,
        "hydrogens_present": iface.hydrogens_present,
        "waters_present": waters_present,
        "criteria": {k: crit[k] for k in sorted(MACRO_INTERFACE_CRITERIA)},
        "notes": notes,
        # atom positions of the two sides (heavy atoms, no water/ions), for callers that redo the area
        "_positions": {"A": list(iface.side_a), "B": list(iface.side_b)},
    }


def _spec_text(spec):
    if isinstance(spec, str):
        return spec
    if callable(spec):
        return "<callable>"
    try:
        return f"<{len(spec)} items>"
    except TypeError:
        return str(spec)


def _res_sort(label):
    resn_resi, _, chain = label.rpartition(":")
    m = re.search(r"(-?\d+)([A-Za-z]?)$", resn_resi)
    return (chain, int(m.group(1)) if m else 0, m.group(2) if m else "")


def _buried_sasa(iface, crit, n_points):
    atoms = iface.atoms
    radii = np.array([vdw_radius(e) for e in iface.elem])
    probe = float(crit["sasa_probe"])
    reach = 2.0 * (float(radii[np.concatenate([iface.ia, iface.ib])].max()) + probe)
    ia, ib = iface.ia, iface.ib
    both = np.concatenate([ia, ib])
    per_res = defaultdict(float)
    out = {"method": "shrake_rupley_numpy", "probe_A": probe, "points_per_atom": n_points,
           "radii": "Bondi van der Waals radii of heavy atoms (interaction_runtime)"}
    for side, own, other in (("A", ia, ib), ("B", ib, ia)):
        i, _j, _d = _pairs_within(iface.xyz[own], iface.xyz[other], reach)
        shell = np.unique(i)
        if not len(shell):
            out[f"buried_{side}"] = 0.0
            continue
        own_xyz, own_rad = iface.xyz[own], radii[own]
        alone = shrake_rupley(own_xyz, own_rad, subset=shell, probe=probe, n_points=n_points)
        offset = 0 if side == "A" else len(ia)
        cplx = shrake_rupley(iface.xyz[both], radii[both], subset=shell + offset, probe=probe, n_points=n_points)
        delta = alone - cplx
        out[f"buried_{side}"] = round(float(delta.sum()), 1)
        for k, dv in zip(shell.tolist(), delta.tolist()):
            per_res[residue_label(atoms[int(own[k])])] += dv
    out["buried_total"] = round(out.get("buried_A", 0.0) + out.get("buried_B", 0.0), 1)
    out["_per_residue"] = dict(per_res)
    return out


def _nucleic_summary(iface, records, crit):
    """Counts of protein-nucleic acid contacts by nucleic-acid part and DNA groove."""
    by_part = Counter()
    by_groove = Counter()
    for i, j, d in iface.close_pairs:
        if d > crit["contact_dist"]:
            continue
        for k in (i, j):
            part, groove = iface.na_info(k)
            if part:
                by_part[part] += 1
                if groove:
                    by_groove[groove] += 1
                break
    spec = Counter()
    readout_res = defaultdict(set)
    for r in records:
        if r.get("readout") and r["type"] in SPECIFIC_TYPES:
            spec[r["readout"]] += 1
            other = r["a"] if r.get("na_side") == "B" else r["b"]
            readout_res[r["readout"]].add(other["residue"])
    base_specific = sum(n for k, n in spec.items() if k.startswith("base"))
    backbone_specific = spec.get("phosphate backbone", 0) + spec.get("sugar", 0)
    return {
        "atom_contacts_by_part": dict(by_part),
        "atom_contacts_by_groove": dict(by_groove),
        "specific_by_readout": dict(spec),
        "residues_by_readout": {k: sorted(v, key=_res_sort) for k, v in readout_res.items()},
        "base_readout_specific": base_specific,
        "backbone_specific": backbone_specific,
        "groove_note": "groove assigned for DNA base atoms only (B-form edges)",
    }


def summary_text(result, top=5):
    """One paragraph for a chat card or a log."""
    counts = result.get("counts", {})
    parts = ", ".join(f"{n} {t}" for t, n in counts.items()) or "no interactions"
    res = result.get("interface_residues", {})
    text = (f"{result.get('interface_type')} interface: {len(res.get('A', []))} + {len(res.get('B', []))} residues; "
            f"{parts}.")
    sasa = result.get("sasa")
    if sasa:
        text += f" Buried SASA {sasa['buried_total']:.0f} Å² ({sasa['method']})."
    na = result.get("nucleic_acid")
    if na:
        text += (f" Nucleic acid: {na['base_readout_specific']} base-specific vs {na['backbone_specific']} "
                 "backbone specific contacts.")
    return text
