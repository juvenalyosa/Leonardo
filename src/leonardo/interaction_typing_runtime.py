"""Protein residue typing helpers for protein-ligand interaction detection."""

from __future__ import annotations

_RESIDUE_ALIASES = {
    "HSD": "HID",
    "HSE": "HIE",
    "HSP": "HIP",
    "MSE": "MET",
}


_PROTEIN_POSITIVE = {
    "LYS": ("NZ",),
    "ARG": ("NH1", "NH2"),
    "HIP": ("ND1", "NE2"),
}


_PROTEIN_NEGATIVE = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
}


_PROTEIN_HBOND_DONORS = {
    "LYS": ("NZ",),
    "LYN": ("NZ",),
    "ARG": ("NE", "NH1", "NH2"),
    "HIS": ("ND1", "NE2"),
    "HID": ("ND1",),
    "HIE": ("NE2",),
    "HIP": ("ND1", "NE2"),
    "TRP": ("NE1",),
    "GLN": ("NE2",),
    "ASN": ("ND2",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
}


# Hydrogens each protein donor atom carries: without explicit hydrogens a donor
# cannot make more heavy-atom H-bonds than this (Thr OG1 donates one).
_PROTEIN_DONOR_H_COUNT = {
    "N": 1,
    "NZ": 3,
    "NE": 1,
    "NH1": 2,
    "NH2": 2,
    "ND1": 1,
    "NE2": 1,
    "NE1": 1,
    "ND2": 2,
    "OG": 1,
    "OG1": 1,
    "OH": 1,
    "SG": 1,
}
_PROTEIN_DONOR_H_COUNT_BY_RESIDUE = {
    ("GLN", "NE2"): 2,
    ("LYN", "NZ"): 2,
    ("PRO", "N"): 0,
}


_PROTEIN_HBOND_ACCEPTORS = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"),
    "HID": ("NE2",),
    "HIE": ("ND1",),
    "HIP": (),
    "GLN": ("OE1",),
    "ASN": ("OD1",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "TYR": ("OH",),
    "CYM": ("SG",),
}


_PROTEIN_AROMATIC_RINGS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HID": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIE": ("CG", "ND1", "CD2", "CE1", "NE2"),
    "HIP": ("CG", "ND1", "CD2", "CE1", "NE2"),
}


_PROTEIN_HYDROPHOBIC = {
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "PRO",
    "PHE",
    "TRP",
    "MET",
}


_PROTEIN_HYDROPHOBIC_ATOMS = {
    "ALA": ("CB",),
    "VAL": ("CB", "CG1", "CG2"),
    "LEU": ("CB", "CG", "CD1", "CD2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"),
    "PRO": ("CB", "CG", "CD"),
    "MET": ("CB", "CG", "CE"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TRP": ("CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "CYS": ("CB",),
}


_PROTEIN_METAL_DONORS = {
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


def normalize_interaction_residue_name(resn):
    token = str(resn or "").strip().upper()
    if not token:
        return ""
    return _RESIDUE_ALIASES.get(token, token)


def _lookup_atom_names(table, resn):
    return table.get(normalize_interaction_residue_name(resn), ())


def protein_positive_atom_names(resn):
    return _lookup_atom_names(_PROTEIN_POSITIVE, resn)


def protein_negative_atom_names(resn):
    return _lookup_atom_names(_PROTEIN_NEGATIVE, resn)


def protein_hbond_donor_atom_names(resn):
    return _lookup_atom_names(_PROTEIN_HBOND_DONORS, resn)


def protein_donor_hydrogen_count(resn, atom_name):
    """Hydrogens a protein donor atom carries (backbone N: 1, Lys NZ: 3, ...); None if unknown."""
    name = str(atom_name or "").strip().upper()
    key = (normalize_interaction_residue_name(resn), name)
    if key in _PROTEIN_DONOR_H_COUNT_BY_RESIDUE:
        return _PROTEIN_DONOR_H_COUNT_BY_RESIDUE[key]
    return _PROTEIN_DONOR_H_COUNT.get(name)


def protein_hbond_acceptor_atom_names(resn):
    return _lookup_atom_names(_PROTEIN_HBOND_ACCEPTORS, resn)


def protein_aromatic_ring_atom_names(resn):
    return _lookup_atom_names(_PROTEIN_AROMATIC_RINGS, resn)


def protein_hydrophobic_atom_names(resn):
    return _lookup_atom_names(_PROTEIN_HYDROPHOBIC_ATOMS, resn)


def protein_metal_donor_atom_names(resn):
    return _lookup_atom_names(_PROTEIN_METAL_DONORS, resn)


def is_protein_hydrophobic_residue(resn):
    return normalize_interaction_residue_name(resn) in _PROTEIN_HYDROPHOBIC
