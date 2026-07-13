"""Interaction chemistry and geometry helpers for protein-ligand contacts.

The PyMOL controller supplies atom/ring dictionaries; this module keeps the
chemistry rules UI-free and directly testable.  Keep predicates conservative:
false positives are more damaging than missing a borderline contact in an
interactive analysis table.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

try:
    from .interaction_typing_runtime import normalize_interaction_residue_name
except Exception:
    from interaction_typing_runtime import normalize_interaction_residue_name


# Aligned to PLIP-style defaults and literature-derived rule sets.  Distances
# are Angstrom, angles are degrees.  These values are intentionally centralized
# so UI workflows, trajectory analysis, and tests use the same physical rules.
INTERACTION_GEOMETRY_CRITERIA = {
    "h_bond_dist": 4.1,
    "h_bond_angle": 100.0,
    "h_bond_ha_dist": 2.8,
    "salt_bridge_dist": 5.5,
    "hydrophobic_dist": 4.0,
    "pi_pi_dist": 5.5,
    "pi_pi_angle_dev": 30.0,
    "pi_stack_offset_max": 2.0,
    "pi_t_angle": 90.0,
    "pi_t_angle_dev": 30.0,
    "cation_pi_dist": 6.0,
    "halogen_dist": 4.0,
    "halogen_don_angle": 165.0,
    "halogen_acc_angle": 120.0,
    "halogen_angle_dev": 30.0,
    "metal_coordination_dist": 3.0,
    "vdw_padding": 0.5,
    "vdw_max_dist": 4.5,
}

# Fluorine is excluded as a default sigma-hole donor.  It can be close to
# acceptors, but classifying C-F...A contacts as halogen bonds causes many false
# positives in protein-ligand screens.
HALOGEN_BOND_DONOR_ELEMENTS = ("CL", "BR", "I")
HBOND_ACCEPTOR_ELEMENTS = ("O", "N", "S", "F")
HBOND_DONOR_ELEMENTS = ("O", "N", "S")
METAL_DONOR_ELEMENTS = ("O", "N", "S", "P", "SE")
LIGAND_CATION_GROUP_TYPES = ("Ammonium", "Aromatic Cation", "Iminium", "Amidinium", "Guanidinium")
DELOCALIZED_LIGAND_CATION_GROUP_TYPES = ("Amidinium", "Guanidinium")
LIGAND_ANION_GROUP_TYPES = ("Carboxylate", "Phosphate", "Sulfonate", "Sulfate")

# Metal coordination distances vary by ion.  The generic criterion remains the
# default cap for unknown metals; common biological ions get element-aware
# cutoffs that avoid rejecting realistic Ca/K coordination shells.
METAL_COORDINATION_DIST_BY_ELEMENT = {
    "MG": 2.8,
    "ZN": 2.8,
    "FE": 2.9,
    "MN": 3.0,
    "CU": 2.8,
    "NI": 2.8,
    "CO": 2.8,
    "CA": 3.2,
    "NA": 3.0,
    "K": 3.4,
}


_COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}
_BOND_TOLERANCE = 0.45


INTERACTION_TYPE_METADATA = {
    "Hydrogen Bond": {
        "map_code": "HB",
        "physics_family": "directional_polar",
        "biological_meaning": "Directional donor-acceptor contact stabilizing binding geometry.",
        "physical_meaning": "Electrostatic and polarization interaction with angular preference.",
    },
    "Salt Bridge": {
        "map_code": "SB",
        "physics_family": "electrostatic",
        "biological_meaning": "Oppositely charged groups forming an ionic contact.",
        "physical_meaning": "Long-range Coulombic attraction between charge centers.",
    },
    "Hydrophobic": {
        "map_code": "HY",
        "physics_family": "hydrophobic",
        "biological_meaning": "Packing of apolar surfaces at the binding interface.",
        "physical_meaning": "Water-release driven association with favorable dispersion contacts.",
    },
    "Pi-Pi Stacking": {
        "map_code": "PI",
        "physics_family": "aromatic",
        "biological_meaning": "Parallel aromatic stacking between ligand and residue rings.",
        "physical_meaning": "Dispersion and quadrupolar interaction between aromatic pi systems.",
    },
    "T-Shaped Pi-Pi": {
        "map_code": "ET",
        "physics_family": "aromatic",
        "biological_meaning": "Edge-to-face aromatic contact between ligand and residue rings.",
        "physical_meaning": "Quadrupolar aromatic interaction in near-orthogonal geometry.",
    },
    "Cation-Pi": {
        "map_code": "CP",
        "physics_family": "electrostatic_aromatic",
        "biological_meaning": "Positive charge engaging an aromatic pi cloud.",
        "physical_meaning": "Electrostatic stabilization of a cation above an aromatic quadrupole.",
    },
    "Anion-Pi": {
        "map_code": "AP",
        "physics_family": "electrostatic_aromatic",
        "biological_meaning": "Anion contacting an electron-poor aromatic system.",
        "physical_meaning": "Electrostatic interaction requiring a pi-acidic aromatic surface.",
    },
    "Halogen Bond": {
        "map_code": "XB",
        "physics_family": "directional_sigma_hole",
        "biological_meaning": "Halogen donor engaging a Lewis-basic protein acceptor.",
        "physical_meaning": "Directional sigma-hole interaction with donor and acceptor angular constraints.",
    },
    "Metal Coordination": {
        "map_code": "MC",
        "physics_family": "coordination",
        "biological_meaning": "Ligand donor atom coordinated to a protein metal center.",
        "physical_meaning": "Coordination interaction with partial covalent character.",
    },
    "Van der Waals": {
        "map_code": "vdW",
        "physics_family": "dispersion",
        "biological_meaning": "Close-shape complementarity contact without stronger specific polarity.",
        "physical_meaning": "Short-range dispersion contact near the sum of van der Waals radii.",
    },
}


_PROTEIN_HYDROPHOBIC_ATOMS = {
    "ALA": ("CB",),
    "VAL": ("CB", "CG1", "CG2"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"),
    "PRO": ("CB", "CG", "CD"),
    "MET": ("CB", "CG", "CE"),
    "MSE": ("CB", "CG", "CE"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "CYS": ("CB",),
}


_PROTEIN_METAL_DONOR_ATOMS = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"),
    "HID": ("ND1", "NE2"),
    "HIE": ("ND1", "NE2"),
    "HIP": ("ND1", "NE2"),
    "ASN": ("OD1",),
    "GLN": ("OE1",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
    "CYS": ("SG",),
    "CYM": ("SG",),
}


_PROTEIN_POSITIVE_GROUPS = {
    "LYS": ({"group_type": "Ammonium", "member_atoms": ("NZ",), "center_atoms": ("NZ",)},),
    "ARG": ({"group_type": "Guanidinium", "member_atoms": ("NE", "CZ", "NH1", "NH2"), "center_atoms": ("NH1", "NH2")},),
    "HIP": ({"group_type": "Imidazolium", "member_atoms": ("CG", "ND1", "CD2", "CE1", "NE2"), "center_atoms": ("ND1", "NE2")},),
}


_PROTEIN_NEGATIVE_GROUPS = {
    "ASP": ({"group_type": "Carboxylate", "member_atoms": ("CG", "OD1", "OD2"), "center_atoms": ("OD1", "OD2")},),
    "GLU": ({"group_type": "Carboxylate", "member_atoms": ("CD", "OE1", "OE2"), "center_atoms": ("OE1", "OE2")},),
}


_VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "ZN": 1.39,
    "MG": 1.73,
    "CA": 1.94,
    "MN": 1.39,
    "FE": 1.32,
    "CO": 1.26,
    "CU": 1.32,
    "NI": 1.63,
}


def interaction_metadata(itype):
    return dict(INTERACTION_TYPE_METADATA.get(str(itype or ""), {}) or {})


def _criteria(criteria=None):
    payload = dict(INTERACTION_GEOMETRY_CRITERIA)
    if isinstance(criteria, dict):
        payload.update(criteria)
    return payload


def _element(atom):
    token = str((atom or {}).get("element", "") or "").strip().upper()
    if token:
        return token
    name = str((atom or {}).get("name", "") or "").strip().upper()
    if name.startswith("CL"):
        return "CL"
    if name.startswith("BR"):
        return "BR"
    if len(name) >= 2 and name[:2] in {"FE", "MG", "ZN", "MN", "CA", "NA"}:
        return name[:2]
    return name[:1]


def _coord(value):
    try:
        arr = np.array(value, dtype=float)
    except Exception:
        return None
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return np.array(arr[:3], dtype=float)


def atom_coord(atom):
    """Return a finite 3-vector for an atom dict, or None."""
    return _coord((atom or {}).get("coord"))


def distance_between(a, b):
    """Euclidean distance between coordinate-like values, or NaN if invalid."""
    ca = _coord(a)
    cb = _coord(b)
    if ca is None or cb is None:
        return float("nan")
    return float(np.linalg.norm(ca - cb))


def atom_distance(atom_a, atom_b):
    """Euclidean distance between two atom dictionaries."""
    ca = atom_coord(atom_a)
    cb = atom_coord(atom_b)
    return distance_between(ca, cb)


def angle_between_vectors(v1, v2):
    """Angle in degrees between two vectors; invalid/zero vectors return NaN."""
    a = _coord(v1)
    b = _coord(v2)
    if a is None or b is None:
        return float("nan")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return float("nan")
    cosang = float(np.dot(a / na, b / nb))
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def ring_plane_offset(center, normal, point):
    """Lateral in-plane offset from a ring center to a point."""
    c = _coord(center)
    n = _coord(normal)
    p = _coord(point)
    if c is None or n is None or p is None:
        return float("nan")
    nn = float(np.linalg.norm(n))
    if nn <= 1.0e-12:
        return float(np.linalg.norm(p - c))
    n = n / nn
    v = p - c
    return float(np.linalg.norm(v - float(np.dot(v, n)) * n))


def ring_plane_distance(center, normal, point):
    """Signed height of a point above a ring plane."""
    c = _coord(center)
    n = _coord(normal)
    p = _coord(point)
    if c is None or n is None or p is None:
        return float("nan")
    nn = float(np.linalg.norm(n))
    if nn <= 1.0e-12:
        return float("nan")
    return float(np.dot(p - c, n / nn))


def classify_hbond_geometry(donor, acceptor, hydrogen=None, acceptor_neighbor=None, criteria=None):
    """Validate H-bond geometry and return a normalized payload.

    Explicit-H mode uses the D-H...A angle and H...A distance.  Fallback mode is
    used when hydrogens are absent; it reports D...A distance and, when an
    acceptor heavy neighbor is available, screens with the D-A-X angle.
    """
    crit = _criteria(criteria)
    donor_el = _element(donor)
    acc_el = _element(acceptor)
    if donor_el not in HBOND_DONOR_ELEMENTS:
        return {"ok": False, "reason": "unsupported_donor_element"}
    if acc_el not in HBOND_ACCEPTOR_ELEMENTS:
        return {"ok": False, "reason": "unsupported_acceptor_element"}

    d_coord = atom_coord(donor)
    a_coord = atom_coord(acceptor)
    if d_coord is None or a_coord is None:
        return {"ok": False, "reason": "missing_coordinates"}
    da_distance = float(np.linalg.norm(d_coord - a_coord))
    if da_distance > float(crit["h_bond_dist"]):
        return {"ok": False, "reason": "donor_acceptor_distance", "da_distance": da_distance}

    if hydrogen is not None:
        h_coord = atom_coord(hydrogen)
        if h_coord is None:
            return {"ok": False, "reason": "missing_hydrogen_coordinates"}
        ha_distance = float(np.linalg.norm(h_coord - a_coord))
        if ha_distance > float(crit["h_bond_ha_dist"]):
            return {"ok": False, "reason": "hydrogen_acceptor_distance", "ha_distance": ha_distance}
        angle = angle_between_vectors(d_coord - h_coord, a_coord - h_coord)
        if not np.isfinite(angle) or angle < float(crit["h_bond_angle"]):
            return {"ok": False, "reason": "donor_h_acceptor_angle", "angle": angle}
        return {
            "ok": True,
            "mode": "explicit_hydrogen",
            "distance": ha_distance,
            "da_distance": da_distance,
            "ha_distance": ha_distance,
            "angle": angle,
            "details": f"D-H...A {ha_distance:.1f}Å, {angle:.0f}°",
        }

    fallback_angle = float(crit.get("h_bond_fallback_angle", 90.0))
    angle = 180.0
    details = f"D...A {da_distance:.1f}Å"
    if acceptor_neighbor is not None:
        x_coord = atom_coord(acceptor_neighbor)
        if x_coord is not None:
            angle = angle_between_vectors(d_coord - a_coord, x_coord - a_coord)
            if not np.isfinite(angle) or angle < fallback_angle:
                return {"ok": False, "reason": "donor_acceptor_neighbor_angle", "angle": angle}
            details = f"D...A {da_distance:.1f}Å, D-A-X {angle:.0f}°"
    return {
        "ok": True,
        "mode": "heavy_atom_fallback",
        "distance": da_distance,
        "da_distance": da_distance,
        "angle": angle,
        "details": details,
    }


def validate_halogen_bond(halogen_atom, donor_neighbor, acceptor_atom, acceptor_neighbor=None, criteria=None):
    """Validate a ligand C-X...A halogen bond candidate.

    The donor atom is the halogen, `donor_neighbor` is normally the carbon bound
    to it, and `acceptor_neighbor` is the heavy atom attached to the acceptor.
    """
    crit = _criteria(criteria)
    hal_el = _element(halogen_atom)
    acc_el = _element(acceptor_atom)
    if hal_el not in HALOGEN_BOND_DONOR_ELEMENTS:
        return {"ok": False, "reason": "unsupported_halogen_donor"}
    if acc_el not in HBOND_ACCEPTOR_ELEMENTS:
        return {"ok": False, "reason": "unsupported_acceptor_element"}
    x_coord = atom_coord(halogen_atom)
    c_coord = atom_coord(donor_neighbor)
    a_coord = atom_coord(acceptor_atom)
    if x_coord is None or c_coord is None or a_coord is None:
        return {"ok": False, "reason": "missing_coordinates"}
    distance = float(np.linalg.norm(x_coord - a_coord))
    if distance > float(crit["halogen_dist"]):
        return {"ok": False, "reason": "distance", "distance": distance}
    donor_angle = angle_between_vectors(c_coord - x_coord, a_coord - x_coord)
    donor_ideal = float(crit["halogen_don_angle"])
    donor_dev = float(crit["halogen_angle_dev"])
    if not np.isfinite(donor_angle) or abs(donor_angle - donor_ideal) > donor_dev:
        return {"ok": False, "reason": "donor_angle", "donor_angle": donor_angle}

    acceptor_angle = float(crit["halogen_acc_angle"])
    if acceptor_neighbor is not None:
        y_coord = atom_coord(acceptor_neighbor)
        if y_coord is not None:
            acceptor_angle = angle_between_vectors(x_coord - a_coord, y_coord - a_coord)
            acc_ideal = float(crit["halogen_acc_angle"])
            if not np.isfinite(acceptor_angle) or abs(acceptor_angle - acc_ideal) > donor_dev:
                return {"ok": False, "reason": "acceptor_angle", "acceptor_angle": acceptor_angle}
    return {
        "ok": True,
        "distance": distance,
        "donor_angle": donor_angle,
        "acceptor_angle": acceptor_angle,
        "details": f"C-X...A {donor_angle:.0f}°, X...A-Y {acceptor_angle:.0f}°",
    }


def metal_coordination_cutoff(metal_element, criteria=None):
    crit = _criteria(criteria)
    generic = float(crit["metal_coordination_dist"])
    return float(METAL_COORDINATION_DIST_BY_ELEMENT.get(str(metal_element or "").strip().upper(), generic))


def validate_metal_coordination(metal_atom, donor_atom, criteria=None):
    donor_el = _element(donor_atom)
    if donor_el not in METAL_DONOR_ELEMENTS:
        return {"ok": False, "reason": "unsupported_donor_element"}
    d = atom_distance(metal_atom, donor_atom)
    cutoff = metal_coordination_cutoff(_element(metal_atom), criteria=criteria)
    if not np.isfinite(d) or d > cutoff:
        return {"ok": False, "reason": "distance", "distance": d, "cutoff": cutoff}
    return {
        "ok": True,
        "distance": float(d),
        "cutoff": float(cutoff),
        "details": f"{donor_el} donor {float(d):.1f}Å",
    }


def classify_pi_ring_pair(protein_ring, ligand_ring, criteria=None):
    """Classify aromatic ring-ring geometry as stacking, T-shaped, or absent."""
    crit = _criteria(criteria)
    p_cent = _coord((protein_ring or {}).get("centroid"))
    l_cent = _coord((ligand_ring or {}).get("centroid"))
    p_norm = _coord((protein_ring or {}).get("normal"))
    l_norm = _coord((ligand_ring or {}).get("normal"))
    if p_cent is None or l_cent is None or p_norm is None or l_norm is None:
        return {"ok": False, "reason": "missing_ring_geometry"}
    d = float(np.linalg.norm(p_cent - l_cent))
    if d > float(crit["pi_pi_dist"]):
        return {"ok": False, "reason": "centroid_distance", "distance": d}
    angle = angle_between_vectors(p_norm, l_norm)
    if not np.isfinite(angle):
        return {"ok": False, "reason": "ring_normal_angle"}
    parallel_dev = min(abs(angle), abs(180.0 - angle))
    ortho_dev = abs(float(crit["pi_t_angle"]) - angle)
    offset = min(
        ring_plane_offset(p_cent, p_norm, l_cent),
        ring_plane_offset(l_cent, l_norm, p_cent),
    )
    if not np.isfinite(offset) or offset > float(crit["pi_stack_offset_max"]):
        return {"ok": False, "reason": "ring_offset", "offset": offset}
    if parallel_dev <= float(crit["pi_pi_angle_dev"]):
        return {
            "ok": True,
            "type": "Pi-Pi Stacking",
            "distance": d,
            "angle": angle,
            "offset": offset,
            "details": f"centroid {d:.1f}Å, angle {angle:.0f}°, offset {offset:.2f}Å",
        }
    if ortho_dev <= float(crit["pi_t_angle_dev"]):
        return {
            "ok": True,
            "type": "T-Shaped Pi-Pi",
            "distance": d,
            "angle": angle,
            "offset": offset,
            "details": f"centroid {d:.1f}Å, angle {angle:.0f}°, offset {offset:.2f}Å",
        }
    return {"ok": False, "reason": "ring_angle", "angle": angle}


def validate_ring_charge_interaction(ring, charge_center, criteria=None, min_plane_distance=1.2):
    """Validate cation/anion placement over an aromatic ring face."""
    crit = _criteria(criteria)
    center = _coord((ring or {}).get("centroid"))
    normal = _coord((ring or {}).get("normal"))
    point = _coord(charge_center)
    if center is None or normal is None or point is None:
        return {"ok": False, "reason": "missing_geometry"}
    d = float(np.linalg.norm(point - center))
    if d > float(crit["cation_pi_dist"]):
        return {"ok": False, "reason": "distance", "distance": d}
    offset = ring_plane_offset(center, normal, point)
    if not np.isfinite(offset) or offset > float(crit["pi_stack_offset_max"]):
        return {"ok": False, "reason": "ring_offset", "offset": offset}
    height = abs(ring_plane_distance(center, normal, point))
    if not np.isfinite(height) or height < float(min_plane_distance):
        return {"ok": False, "reason": "coplanar_or_too_low", "plane_distance": height}
    return {
        "ok": True,
        "distance": d,
        "offset": offset,
        "plane_distance": height,
        "details_suffix": f"{d:.1f}Å, offset {offset:.2f}Å, height {height:.2f}Å",
    }


def assign_ligand_hydrophobic_flags_from_rdkit(ligand_atoms, mol, rd_to_feat_idx):
    """Return ligand atoms with RDKit-derived hydrophobic-carbon flags.

    A carbon is hydrophobic only when all heavy neighbors are carbon.  This
    prevents carbonyl, imine, nitrile, and heteroatom-bound carbons from being
    treated as hydrophobic contacts merely because their element is carbon.
    """
    atoms = [dict(a) for a in list(ligand_atoms or [])]
    for atom in atoms:
        atom["is_hydrophobic"] = False
    if mol is None or not rd_to_feat_idx:
        return atoms
    try:
        rd_to_feat = {int(k): int(v) for k, v in dict(rd_to_feat_idx).items()}
        for rd_atom in mol.GetAtoms():
            if int(rd_atom.GetAtomicNum()) != 6:
                continue
            if any(int(nbr.GetAtomicNum()) not in (1, 6) for nbr in rd_atom.GetNeighbors()):
                continue
            fi = rd_to_feat.get(int(rd_atom.GetIdx()))
            if fi is not None and 0 <= int(fi) < len(atoms):
                atoms[int(fi)]["is_hydrophobic"] = True
    except Exception:
        return atoms
    return atoms


def protein_hydrophobic_atom_names(resn):
    return tuple(_PROTEIN_HYDROPHOBIC_ATOMS.get(normalize_interaction_residue_name(resn), ()))


def protein_metal_donor_atom_names(resn):
    return tuple(_PROTEIN_METAL_DONOR_ATOMS.get(normalize_interaction_residue_name(resn), ()))


def _coords_mean(atoms):
    pts = [np.array(a.get("coord"), dtype=float) for a in list(atoms or []) if a.get("coord") is not None]
    if not pts:
        return None
    return np.mean(np.vstack(pts), axis=0)


def _atom_identity(atom):
    coord = tuple(np.round(np.array(atom.get("coord", (0.0, 0.0, 0.0)), dtype=float), 3))
    return (
        str(atom.get("chain", "") or ""),
        str(atom.get("resi", "") or ""),
        str(atom.get("resn", "") or ""),
        str(atom.get("name", "") or ""),
        coord,
    )


def _formal_charge(atom):
    try:
        return int((atom or {}).get("charge_formal", 0) or 0)
    except Exception:
        return 0


def _declared_neighbor_elements(atom):
    for key in ("neighbor_elements", "bonded_elements"):
        raw = (atom or {}).get(key)
        if raw:
            return [str(x).strip().upper() for x in list(raw or []) if str(x).strip()]
    raw_neighbors = (atom or {}).get("neighbors")
    if raw_neighbors:
        return [_element(x) for x in list(raw_neighbors or []) if _element(x)]
    return []


def _runtime_bond_threshold(elem_a, elem_b):
    ea = str(elem_a or "").strip().upper()
    eb = str(elem_b or "").strip().upper()
    return float(_COVALENT_RADII.get(ea, 0.77) + _COVALENT_RADII.get(eb, 0.77) + _BOND_TOLERANCE)


def ligand_atom_neighbors(atom, ligand_pool=None, neighbor_getter=None):
    """Return bonded ligand neighbors from an explicit getter or geometry fallback."""
    if not atom:
        return []
    if callable(neighbor_getter):
        try:
            return list(neighbor_getter(atom, list(ligand_pool or [])) or [])
        except Exception:
            return []
    declared = _declared_neighbor_elements(atom)
    if declared:
        return [{"element": elem} for elem in declared]
    pool = list(ligand_pool or [])
    if not pool:
        return []
    target_coord = atom_coord(atom)
    if target_coord is None:
        return []
    out = []
    target_key = _atom_identity(atom)
    for cand in pool:
        if cand is atom:
            continue
        if _atom_identity(cand) == target_key:
            continue
        if str(cand.get("chain", "") or "") != str(atom.get("chain", "") or ""):
            continue
        if str(cand.get("resi", "") or "") != str(atom.get("resi", "") or ""):
            continue
        cand_coord = atom_coord(cand)
        if cand_coord is None:
            continue
        if distance_between(target_coord, cand_coord) <= _runtime_bond_threshold(_element(atom), _element(cand)):
            out.append(cand)
    return out


def _neighbor_element_counts(atom, ligand_pool=None, neighbor_getter=None):
    neighbors = ligand_atom_neighbors(atom, ligand_pool=ligand_pool, neighbor_getter=neighbor_getter)
    elems = [_element(n) for n in neighbors if _element(n)]
    return {
        "neighbors": list(neighbors),
        "elements": list(elems),
        "h_count": sum(1 for elem_i in elems if elem_i == "H"),
        "heavy_count": sum(1 for elem_i in elems if elem_i != "H"),
        "degree": len(elems),
        "hetero_oxo_count": sum(1 for elem_i in elems if elem_i in ("O", "S")),
    }


def _is_ammonium_like_topology(atom, ligand_pool=None, neighbor_getter=None):
    """Detect explicit protonated/quaternary ammonium topology.

    This is intentionally topology-based, not pH-prediction-based: a neutral
    tertiary amine with three neighbors stays neutral, while an N with four
    bonded substituents is chemically cationic even if a PDB-style file omitted
    the formal charge.
    """
    if _element(atom) != "N":
        return False
    counts = _neighbor_element_counts(atom, ligand_pool=ligand_pool, neighbor_getter=neighbor_getter)
    if int(counts["degree"]) < 4 or int(counts["heavy_count"]) < 1:
        return False
    # Exclude nitro/N-oxide/sulfoximine-like positive nitrogens from cation-pi
    # and salt-bridge centers. They may be formally positive but are not
    # ammonium-like donor cations.
    if int(counts["hetero_oxo_count"]) > 0:
        return False
    return True


def is_supported_ligand_cation_center(atom, ligand_pool=None, neighbor_getter=None):
    """Return True only for cation centers with explicit formal/group evidence.

    Neutral three-coordinate nitrogens are intentionally not inferred as
    cations from element, hydrogen count, or partial charge alone.  Explicit
    four-coordinate ammonium topology is accepted even when formal charges are
    absent because many PDB/PDBQT-like inputs omit charge fields.  Delocalized
    amidinium/guanidinium groups are accepted when the group annotator marked
    the shared positive group; the interaction code then uses the group center.
    """
    if not atom or bool((atom or {}).get("is_anion")):
        return False
    elem = _element(atom)
    formal = _formal_charge(atom)
    group_type = str((atom or {}).get("group_type", "") or "")
    group_id = str((atom or {}).get("ionic_group_id", "") or "").strip()

    if elem == "P":
        return formal > 0
    if elem != "N":
        return False

    if (
        group_type in DELOCALIZED_LIGAND_CATION_GROUP_TYPES
        and group_id
        and bool((atom or {}).get("is_cation"))
    ):
        return True
    if group_type in LIGAND_CATION_GROUP_TYPES and formal > 0:
        return True
    if _is_ammonium_like_topology(atom, ligand_pool=ligand_pool, neighbor_getter=neighbor_getter):
        return True
    if formal <= 0:
        return False

    counts = _neighbor_element_counts(atom, ligand_pool=ligand_pool, neighbor_getter=neighbor_getter)
    h_count = int(counts["h_count"])
    heavy_count = int(counts["heavy_count"])
    degree = int(counts["degree"])

    # Protonated/quaternary ammonium-like nitrogen.  A positive nitro/N-oxide
    # nitrogen will not pass this topology, so it will not become a cation-pi
    # or salt-bridge center.
    if degree >= 4 and heavy_count >= 1 and int(counts["hetero_oxo_count"]) == 0:
        return True
    # Iminium-like formal cations are planar degree-3 nitrogens with at least
    # one hydrogen or an explicit iminium group annotation.
    if group_type == "Iminium":
        return True
    if degree == 3 and heavy_count >= 2 and h_count >= 1:
        return True
    return False


def is_ligand_anion_center(atom):
    elem = _element(atom)
    if elem not in ("O", "S"):
        return False
    if bool((atom or {}).get("is_anion")):
        return True
    if _formal_charge(atom) <= -1:
        return True
    return str((atom or {}).get("group_type", "") or "") in {
        "Carboxylate",
        "Phosphate",
        "Sulfonate",
        "Sulfate",
        "FormalCharge",
    }


def _same_atom(atom_a, atom_b):
    if atom_a is atom_b:
        return True
    return _atom_identity(atom_a) == _atom_identity(atom_b)


def _atom_name(atom):
    return str((atom or {}).get("name", "") or "").strip().upper()


def _heavy_neighbors_excluding(atom, ligand_pool, excluded_atom=None, neighbor_getter=None):
    out = []
    for nbr in ligand_atom_neighbors(atom, ligand_pool=ligand_pool, neighbor_getter=neighbor_getter):
        if excluded_atom is not None and _same_atom(nbr, excluded_atom):
            continue
        if _element(nbr) != "H":
            out.append(nbr)
    return out


def _has_bound_hydrogen(atom, ligand_pool, neighbor_getter=None):
    return any(
        _element(nbr) == "H"
        for nbr in ligand_atom_neighbors(atom, ligand_pool=ligand_pool, neighbor_getter=neighbor_getter)
    )


def _terminal_oxygen_centers(center_atom, oxygen_atoms, ligand_pool, neighbor_getter=None):
    centers = []
    for oxy in list(oxygen_atoms or []):
        extra_heavy = _heavy_neighbors_excluding(
            oxy,
            ligand_pool=ligand_pool,
            excluded_atom=center_atom,
            neighbor_getter=neighbor_getter,
        )
        # Bridging ester/ether oxygens are not charge centers.  A bound H means
        # the observed state is protonated, so do not use that oxygen for anion
        # center placement.
        if extra_heavy:
            continue
        if _has_bound_hydrogen(oxy, ligand_pool=ligand_pool, neighbor_getter=neighbor_getter):
            continue
        centers.append(oxy)
    return centers


def _mark_ligand_ionic_group(members, center_atoms, group_type, charge_sign, group_id):
    member_list = [m for m in list(members or []) if m]
    center_list = [c for c in list(center_atoms or []) if c]
    if not member_list or not center_list:
        return
    member_names = tuple(_atom_name(a) for a in member_list if _atom_name(a))
    center_names = tuple(_atom_name(a) for a in center_list if _atom_name(a))
    center_ids = {_atom_identity(a) for a in center_list}
    for atom in member_list:
        current_group = str(atom.get("group_type", "") or "")
        if not current_group or current_group == "FormalCharge":
            atom["group_type"] = str(group_type)
        if not str(atom.get("ionic_group_id", "") or "").strip() or current_group == "FormalCharge":
            atom["ionic_group_id"] = str(group_id)
        if member_names and not atom.get("ionic_group_members"):
            atom["ionic_group_members"] = member_names
        if center_names and not atom.get("ionic_group_center_members"):
            atom["ionic_group_center_members"] = center_names
        if _atom_identity(atom) not in center_ids:
            continue
        if int(charge_sign) < 0:
            atom["is_anion"] = True
            if _formal_charge(atom) >= 0:
                try:
                    atom["charge_partial"] = min(float(atom.get("charge_partial", 0.0) or 0.0), -0.35)
                except Exception:
                    atom["charge_partial"] = -0.35
        elif int(charge_sign) > 0:
            atom["is_cation"] = True
            if _formal_charge(atom) <= 0:
                try:
                    atom["charge_partial"] = max(float(atom.get("charge_partial", 0.0) or 0.0), 0.35)
                except Exception:
                    atom["charge_partial"] = 0.35


def assign_ligand_ionic_groups_from_topology(ligand_atoms, neighbor_getter=None):
    """Annotate common ionic ligand groups when formal charges are missing.

    PDB/PDBQT-derived ligands often have coordinates and elements but weak
    formal charge/bond-order metadata.  This fallback marks only high-confidence
    group topologies used for salt bridges: carboxylates, phosphates,
    sulfonates/sulfates, explicit ammonium nitrogens, and simple
    amidinium/guanidinium motifs.
    """
    atoms = list(ligand_atoms or [])
    if not atoms:
        return atoms

    for idx, atom in enumerate(atoms):
        elem = _element(atom)
        neighbors = ligand_atom_neighbors(atom, ligand_pool=atoms, neighbor_getter=neighbor_getter)
        oxy = [n for n in neighbors if _element(n) == "O"]
        nit = [n for n in neighbors if _element(n) == "N"]
        hetero_oxo = [n for n in neighbors if _element(n) in ("O", "S")]

        if elem == "C" and len(oxy) >= 2 and not nit:
            centers = _terminal_oxygen_centers(atom, oxy, atoms, neighbor_getter=neighbor_getter)
            if len(centers) >= 2:
                members = [atom] + list(centers)
                _mark_ligand_ionic_group(
                    members,
                    centers[:2],
                    "Carboxylate",
                    -1,
                    f"topology:Carboxylate:{idx}",
                )
                continue

        if elem == "P" and len(oxy) >= 3:
            centers = _terminal_oxygen_centers(atom, oxy, atoms, neighbor_getter=neighbor_getter)
            if len(centers) >= 2:
                _mark_ligand_ionic_group(
                    [atom] + list(oxy),
                    centers,
                    "Phosphate",
                    -1,
                    f"topology:Phosphate:{idx}",
                )
                continue

        if elem == "S" and len(oxy) >= 3:
            centers = _terminal_oxygen_centers(atom, oxy, atoms, neighbor_getter=neighbor_getter)
            if len(centers) >= 2:
                group = "Sulfate" if len(oxy) >= 4 else "Sulfonate"
                _mark_ligand_ionic_group(
                    [atom] + list(oxy),
                    centers,
                    group,
                    -1,
                    f"topology:{group}:{idx}",
                )
                continue

        if elem == "N" and _is_ammonium_like_topology(atom, ligand_pool=atoms, neighbor_getter=neighbor_getter):
            _mark_ligand_ionic_group(
                [atom],
                [atom],
                "Ammonium",
                +1,
                f"topology:Ammonium:{idx}",
            )
            continue

        if elem == "C" and not hetero_oxo and len(nit) >= 3:
            _mark_ligand_ionic_group(
                [atom] + list(nit),
                nit,
                "Guanidinium",
                +1,
                f"topology:Guanidinium:{idx}",
            )
            continue

        if elem == "C" and not hetero_oxo and len(nit) == 2:
            terminal_nitrogens = all(
                not _heavy_neighbors_excluding(
                    n_atom,
                    atoms,
                    excluded_atom=atom,
                    neighbor_getter=neighbor_getter,
                )
                for n_atom in nit
            )
            # Many PDB ligands omit hydrogens and formal charges.  A carbon
            # bonded to two terminal nitrogens with no oxo heteroatoms is a
            # high-confidence amidinium center, unlike an arbitrary tertiary
            # amine nitrogen.  This recovers benzamidine-like salt bridges
            # without treating every ligand nitrogen as cationic.
            if terminal_nitrogens or any(_has_bound_hydrogen(n_atom, atoms, neighbor_getter=neighbor_getter) for n_atom in nit):
                _mark_ligand_ionic_group(
                    [atom] + list(nit),
                    nit,
                    "Amidinium",
                    +1,
                    f"topology:Amidinium:{idx}",
                )

    return atoms


def build_protein_charge_groups(protein_atoms):
    residues = defaultdict(list)
    for atom in list(protein_atoms or []):
        key = (
            str(atom.get("chain", "") or ""),
            str(atom.get("resi", "") or ""),
            normalize_interaction_residue_name(atom.get("resn", "")),
        )
        residues[key].append(dict(atom))

    positive = []
    negative = []
    for (chain, resi, resn), atoms in residues.items():
        atom_by_name = {str(a.get("name", "") or "").strip().upper(): a for a in atoms}
        for spec in _PROTEIN_POSITIVE_GROUPS.get(resn, ()):
            members = [atom_by_name[nm] for nm in spec["member_atoms"] if nm in atom_by_name]
            centers = [atom_by_name[nm] for nm in spec["center_atoms"] if nm in atom_by_name]
            if len(members) != len(spec["member_atoms"]) or not centers:
                continue
            center = _coords_mean(centers)
            if center is None:
                continue
            positive.append(
                {
                    "charge_sign": +1,
                    "group_type": str(spec["group_type"]),
                    "chain": str(chain),
                    "resi": str(resi),
                    "resn": str(resn),
                    "center": np.array(center, dtype=float),
                    "atoms": list(members),
                    "representative_atom": str(centers[0].get("name", "") or ""),
                    "group_id": f"prot:+:{chain}:{resi}:{resn}:{spec['group_type']}",
                }
            )
        for spec in _PROTEIN_NEGATIVE_GROUPS.get(resn, ()):
            members = [atom_by_name[nm] for nm in spec["member_atoms"] if nm in atom_by_name]
            centers = [atom_by_name[nm] for nm in spec["center_atoms"] if nm in atom_by_name]
            if len(members) != len(spec["member_atoms"]) or not centers:
                continue
            center = _coords_mean(centers)
            if center is None:
                continue
            negative.append(
                {
                    "charge_sign": -1,
                    "group_type": str(spec["group_type"]),
                    "chain": str(chain),
                    "resi": str(resi),
                    "resn": str(resn),
                    "center": np.array(center, dtype=float),
                    "atoms": list(members),
                    "representative_atom": str(centers[0].get("name", "") or ""),
                    "group_id": f"prot:-:{chain}:{resi}:{resn}:{spec['group_type']}",
                }
            )
    return {
        "positive": list(positive),
        "negative": list(negative),
    }


def build_ligand_charge_groups(ligand_atoms):
    atoms = [dict(a) for a in list(ligand_atoms or [])]
    groups = []
    grouped = defaultdict(list)
    for atom in atoms:
        gid = str(atom.get("ionic_group_id", "") or "").strip()
        if gid:
            grouped[gid].append(atom)

    seen_atoms = set()
    for gid, members in grouped.items():
        first = dict(members[0] or {})
        group_is_pos = any(is_supported_ligand_cation_center(a, atoms) for a in members)
        group_is_neg = any(is_ligand_anion_center(a) for a in members)
        if not group_is_pos and not group_is_neg:
            continue
        center_names = {
            str(name).strip().upper()
            for name in list(first.get("ionic_group_center_members", ()) or [])
            if str(name).strip()
        }
        if center_names:
            center_atoms = [a for a in members if str(a.get("name", "")).strip().upper() in center_names]
        else:
            center_atoms = list(members)
        center = _coords_mean(center_atoms)
        if center is None:
            continue
        charge_sign = +1 if group_is_pos else -1
        groups.append(
            {
                "charge_sign": int(charge_sign),
                "group_type": str(first.get("group_type", "FormalCharge") or "FormalCharge"),
                "chain": str(first.get("chain", "") or ""),
                "resi": str(first.get("resi", "") or ""),
                "resn": str(first.get("resn", "") or ""),
                "center": np.array(center, dtype=float),
                "atoms": list(members),
                "representative_atom": str((center_atoms or members)[0].get("name", "") or ""),
                "group_id": str(gid),
            }
        )
        seen_atoms.update(_atom_identity(atom) for atom in members)

    for atom in atoms:
        ident = _atom_identity(atom)
        if ident in seen_atoms:
            continue
        is_pos = bool(is_supported_ligand_cation_center(atom, atoms))
        is_neg = bool(is_ligand_anion_center(atom))
        if not is_pos and not is_neg:
            continue
        center = _coords_mean([atom])
        if center is None:
            continue
        groups.append(
            {
                "charge_sign": +1 if is_pos else -1,
                "group_type": str(atom.get("group_type", "FormalCharge") or "FormalCharge"),
                "chain": str(atom.get("chain", "") or ""),
                "resi": str(atom.get("resi", "") or ""),
                "resn": str(atom.get("resn", "") or ""),
                "center": np.array(center, dtype=float),
                "atoms": [atom],
                "representative_atom": str(atom.get("name", "") or ""),
                "group_id": f"lig:{'+' if is_pos else '-'}:{atom.get('chain', '')}:{atom.get('resi', '')}:{atom.get('name', '')}",
            }
        )
    return list(groups)


def vdw_radius(element):
    key = str(element or "").strip().upper()
    if key in _VDW_RADII:
        return float(_VDW_RADII[key])
    if key and key[0] in _VDW_RADII:
        return float(_VDW_RADII[key[0]])
    return float(_VDW_RADII["C"])


def vdw_contact_cutoff(atom_a, atom_b, padding=0.5, max_cutoff=4.5):
    cutoff = float(vdw_radius(atom_a.get("element")) + vdw_radius(atom_b.get("element")) + float(padding))
    return float(min(float(max_cutoff), cutoff))


def reduce_hydrophobic_pairs(pairs):
    raw = list(pairs or [])
    if not raw:
        return []

    by_lig_atom_and_residue = {}
    for pair in raw:
        lig_atom = dict(pair.get("lig_atom", {}) or {})
        prot_atom = dict(pair.get("prot_atom", {}) or {})
        lig_key = (
            str(lig_atom.get("chain", "") or ""),
            str(lig_atom.get("resi", "") or ""),
            str(lig_atom.get("name", "") or ""),
            int(lig_atom.get("_lig_idx", -1) or -1),
        )
        prot_res_key = (
            str(prot_atom.get("chain", "") or ""),
            str(prot_atom.get("resi", "") or ""),
        )
        key = (lig_key, prot_res_key)
        best = by_lig_atom_and_residue.get(key)
        if best is None or float(pair.get("distance", 999.0)) < float(best.get("distance", 999.0)):
            by_lig_atom_and_residue[key] = dict(pair)

    step1 = list(by_lig_atom_and_residue.values())
    by_prot_atom = {}
    for pair in step1:
        prot_atom = dict(pair.get("prot_atom", {}) or {})
        key = (
            str(prot_atom.get("chain", "") or ""),
            str(prot_atom.get("resi", "") or ""),
            str(prot_atom.get("name", "") or ""),
            int(prot_atom.get("pymol_index", 0) or 0),
        )
        best = by_prot_atom.get(key)
        if best is None or float(pair.get("distance", 999.0)) < float(best.get("distance", 999.0)):
            by_prot_atom[key] = dict(pair)
    return list(by_prot_atom.values())


def build_interaction_record(
    itype,
    prot,
    lig,
    dist,
    details="",
    *,
    extra=None,
    prot_coord=None,
    lig_coord=None,
):
    prot_atom_name = str(prot.get("name", "") or "")
    lig_atom_name = str(lig.get("name", "") or "")
    rec = {
        "type": str(itype or ""),
        "prot_res": f"{prot.get('resn', '?')} {prot.get('chain', '?')}{prot.get('resi', '?')}",
        "prot_atom": str(prot_atom_name),
        "lig_atom": str(lig_atom_name),
        "distance": float(dist),
        "details": str(details or ""),
        "prot_chain": prot.get("chain"),
        "prot_resi": prot.get("resi"),
        "prot_resn": prot.get("resn"),
    }
    meta = interaction_metadata(itype)
    if meta:
        rec.update(meta)
    if prot_coord is not None:
        rec["prot_coord"] = tuple(np.array(prot_coord, dtype=float))
    elif prot.get("coord") is not None:
        rec["prot_coord"] = tuple(np.array(prot.get("coord"), dtype=float))
    if lig_coord is not None:
        rec["lig_coord"] = tuple(np.array(lig_coord, dtype=float))
    elif lig.get("coord") is not None:
        rec["lig_coord"] = tuple(np.array(lig.get("coord"), dtype=float))
    if isinstance(extra, dict):
        rec.update(dict(extra))
    return rec
