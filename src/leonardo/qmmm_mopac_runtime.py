import csv
import json
import os
import re
import subprocess
import tempfile

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
EV_TO_HARTREE = 1.0 / 27.211386245988
KCAL_TO_HARTREE = 1.0 / 627.5094740631
# OpenMM-style electrostatics constant for q in e and r in nm.
OPENMM_COULOMB_CONSTANT = 138.935456


# These profiles are intentionally conservative.  MOPAC reads keywords from the
# first line of the input file, so a stale keyword such as 1SCF can silently turn
# an optimization into a single-point job.  Keeping the operation policy here
# makes the GUI and OpenMM worker use the same MOPAC language.
QMMM_MOPAC_OPERATION_PROFILES = {
    "probe": {
        "label": "Single-point QM/MM probe",
        "keywords": ("PM7", "1SCF", "GRAD", "AUX", "QMMM"),
        "required": ("1SCF", "GRAD", "AUX", "QMMM"),
        "remove": ("EF", "TS", "FORCE", "THERMO", "IRC"),
        "region_policy": "ligand_only",
    },
    "electrostatic": {
        "label": "Electrostatic QM/MM report",
        "keywords": ("PM7", "1SCF", "GRAD", "AUX", "QMMM"),
        "required": ("1SCF", "GRAD", "AUX", "QMMM"),
        "remove": ("EF", "TS", "FORCE", "THERMO", "IRC"),
        "region_policy": "ligand_only",
    },
    "optimization": {
        "label": "Embedded QM-region optimization",
        "keywords": ("PM7", "EF", "AUX", "QMMM"),
        "required": ("EF", "AUX", "QMMM"),
        "remove": ("1SCF", "GRAD", "FORCE", "THERMO", "IRC"),
        "region_policy": "ligand_interface",
    },
    "thermochemistry": {
        "label": "QM/MM thermochemistry",
        "keywords": ("PM7", "FORCE", "THERMO", "AUX", "QMMM"),
        "required": ("FORCE", "THERMO", "AUX", "QMMM"),
        "remove": ("1SCF", "GRAD", "EF", "TS", "IRC"),
        "region_policy": "ligand_interface",
    },
    "reaction_scan": {
        "label": "QM/MM reaction scan",
        "keywords": ("PM7", "1SCF", "GRAD", "AUX", "QMMM"),
        "required": ("1SCF", "GRAD", "AUX", "QMMM"),
        "remove": ("EF", "TS", "FORCE", "THERMO", "IRC"),
        "region_policy": "reaction_core",
    },
    "transition_state": {
        "label": "QM/MM transition-state refinement",
        "keywords": ("PM7", "TS", "AUX", "QMMM"),
        "required": ("TS", "AUX", "QMMM"),
        "remove": ("1SCF", "GRAD", "EF", "FORCE", "THERMO", "IRC"),
        "region_policy": "reaction_core",
    },
    "mopac_qst2": {
        "label": "MOPAC QST2/SADDLE then TS",
        "keywords": ("PM7", "SADDLE", "AUX", "XYZ"),
        "required": ("SADDLE", "AUX", "XYZ"),
        "remove": ("1SCF", "GRAD", "EF", "TS", "FORCE", "THERMO", "IRC", "QMMM"),
        "region_policy": "reaction_core",
    },
    "mopac_qst3": {
        "label": "MOPAC QST3-style TS refinement",
        "keywords": ("PM7", "TS", "AUX", "XYZ"),
        "required": ("TS", "AUX", "XYZ"),
        "remove": ("1SCF", "GRAD", "EF", "SADDLE", "FORCE", "THERMO", "IRC", "QMMM"),
        "region_policy": "reaction_core",
    },
    "mopac_ts_frequency": {
        "label": "MOPAC TS frequency validation",
        "keywords": ("PM7", "FORCE", "AUX", "XYZ"),
        "required": ("FORCE", "AUX", "XYZ"),
        "remove": ("1SCF", "GRAD", "EF", "TS", "SADDLE", "THERMO", "IRC", "QMMM"),
        "region_policy": "reaction_core",
    },
    "mopac_irc": {
        "label": "MOPAC IRC from TS",
        "keywords": ("PM7", "FORCE", "IRC=1*", "AUX", "XYZ"),
        "required": ("FORCE", "IRC=1*", "AUX", "XYZ"),
        "remove": ("1SCF", "GRAD", "EF", "TS", "SADDLE", "THERMO", "QMMM"),
        "region_policy": "reaction_core",
    },
    "mopac_reaction_path": {
        "label": "MOPAC QST/TS/Frequency/IRC workflow",
        "keywords": ("PM7", "AUX", "XYZ"),
        "required": ("AUX", "XYZ"),
        "remove": ("1SCF", "GRAD", "QMMM"),
        "region_policy": "reaction_core",
    },
    "md_periodic": {
        "label": "MD with periodic QM/MM probes",
        "keywords": ("PM7", "1SCF", "GRAD", "AUX", "QMMM"),
        "required": ("1SCF", "GRAD", "AUX", "QMMM"),
        "remove": ("EF", "TS", "FORCE", "THERMO", "IRC"),
        "region_policy": "ligand_only",
    },
    "md_coupled": {
        "label": "Force-coupled QM/MM MD",
        "keywords": ("PM7", "1SCF", "GRAD", "AUX", "QMMM"),
        "required": ("1SCF", "GRAD", "AUX", "QMMM"),
        "remove": ("EF", "TS", "FORCE", "THERMO", "IRC"),
        "region_policy": "ligand_interface",
    },
    "redock_rescore": {
        "label": "Docking pose QM/MM energy",
        "keywords": ("PM7", "1SCF", "GRAD", "AUX", "QMMM"),
        "required": ("1SCF", "GRAD", "AUX", "QMMM"),
        "remove": ("EF", "TS", "FORCE", "THERMO", "IRC"),
        "region_policy": "ligand_only",
    },
}

QMMM_MOPAC_OPERATION_ALIASES = {
    "single_point": "probe",
    "sp": "probe",
    "opt": "optimization",
    "thermo": "thermochemistry",
    "ts": "transition_state",
    "qst2": "mopac_qst2",
    "qst3": "mopac_qst3",
    "irc": "mopac_irc",
    "mopac_full": "mopac_reaction_path",
    "reaction_path": "mopac_reaction_path",
    "redocking": "redock_rescore",
    "docking": "redock_rescore",
}

_MOPAC_METHOD_KEYWORDS = {
    "AM1",
    "PM3",
    "PM6",
    "PM7",
    "PM6-D3",
    "PM6-DH2",
    "PM6-DH+",
    "PM6-D3H4",
    "RM1",
    "MNDO",
}

_QMMM_BIOPOLYMER_RESN = {
    "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN", "GLU", "GLH",
    "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "MSE",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "ACE", "NME", "A", "C", "G",
    "T", "U", "DA", "DC", "DG", "DT",
}
_QMMM_WATER_RESN = {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE"}
_QMMM_ION_RESN = {"NA", "K", "CL", "CA", "MG", "ZN", "FE", "MN", "CU", "CO", "NI"}
_QMMM_METAL_ELEMENTS = {"LI", "NA", "K", "MG", "CA", "MN", "FE", "CO", "NI", "CU", "ZN"}
_COVALENT_RADII_A = {
    "H": 0.31,
    "B": 0.85,
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


_ATOMIC_SYMBOLS = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    26: "Fe",
    29: "Cu",
    30: "Zn",
    35: "Br",
    53: "I",
}

_ATOMIC_MASSES_AMU = {
    "H": 1.007825,
    "B": 11.009305,
    "C": 12.0,
    "N": 14.003074,
    "O": 15.994915,
    "F": 18.998403,
    "P": 30.973762,
    "S": 31.972071,
    "CL": 34.968853,
    "BR": 78.918338,
    "I": 126.904468,
    "NA": 22.98977,
    "MG": 23.985042,
    "K": 38.963707,
    "CA": 39.962591,
    "FE": 55.934936,
    "CU": 62.929599,
    "ZN": 63.929145,
}


def _as_xyz_array(xyz):
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Coordinates must have shape (N,3).")
    return arr


def _as_1d(arr, name="array"):
    out = np.asarray(arr, dtype=float).reshape(-1)
    if out.size <= 0:
        raise ValueError(f"{name} is empty.")
    return out


def _symbol_from_atomic_number(z):
    try:
        zi = int(z)
    except Exception:
        zi = 6
    return str(_ATOMIC_SYMBOLS.get(zi, "C"))


def _clean_element_symbol(value):
    txt = str(value or "").strip()
    if not txt:
        return "C"
    letters = "".join(ch for ch in txt if ch.isalpha())
    if not letters:
        return "C"
    letters = letters[:2]
    return letters[:1].upper() if len(letters) == 1 else letters[:1].upper() + letters[1:].lower()


def _mopac_symbol_key(symbol):
    return str(_clean_element_symbol(symbol)).upper()


def _mopac_atomic_mass_amu(symbol):
    return float(_ATOMIC_MASSES_AMU.get(_mopac_symbol_key(symbol), 12.0))


def _mopac_float(token, default=np.nan):
    txt = str(token or "").strip().replace("D", "E").replace("d", "e")
    try:
        val = float(txt)
    except Exception:
        return float(default)
    return float(val) if np.isfinite(val) else float(default)


def _mopac_extract_method_from_keywords(keywords, default="PM7"):
    for tok in _split_mopac_keywords(keywords):
        if _mopac_keyword_name(tok) in _MOPAC_METHOD_KEYWORDS:
            return str(tok).upper()
    return str(default or "PM7").strip().upper()


def _mopac_atoms_from_arrays(atomic_numbers, xyz_ang):
    xyz = _as_xyz_array(xyz_ang)
    z = [int(v) for v in list(np.asarray(atomic_numbers).reshape(-1))]
    if len(z) != int(xyz.shape[0]):
        raise ValueError("atomic_numbers and xyz size mismatch.")
    atoms = []
    for zi, coord in zip(z, xyz):
        atoms.append(
            {
                "element": _symbol_from_atomic_number(int(zi)),
                "x": float(coord[0]),
                "y": float(coord[1]),
                "z": float(coord[2]),
            }
        )
    return atoms


def _mopac_normalize_atoms(atoms=None, atomic_numbers=None, xyz_ang=None):
    if atoms is None:
        return _mopac_atoms_from_arrays(atomic_numbers, xyz_ang)
    out = []
    for raw in list(atoms or []):
        row = dict(raw or {})
        elem = _clean_element_symbol(row.get("element", row.get("symbol", row.get("name", "C"))))
        if "coord" in row:
            coord = np.asarray(row.get("coord"), dtype=float).reshape(3)
            x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        elif "xyz" in row:
            coord = np.asarray(row.get("xyz"), dtype=float).reshape(3)
            x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
        else:
            x = float(row.get("x", 0.0))
            y = float(row.get("y", 0.0))
            z = float(row.get("z", 0.0))
        entry = {
            "element": str(elem),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "name": str(row.get("name", "") or ""),
            "chain": str(row.get("chain", "") or ""),
            "resi": str(row.get("resi", "") or ""),
            "resn": str(row.get("resn", "") or ""),
        }
        for key in ("frozen", "optimize", "coord_flags", "x_opt", "y_opt", "z_opt", "x_flag", "y_flag", "z_flag"):
            if key in row:
                entry[key] = row.get(key)
        out.append(entry)
    if not out:
        raise ValueError("No atoms were provided.")
    return out


def _mopac_atoms_symbols(atoms):
    return [_clean_element_symbol((a or {}).get("element", (a or {}).get("symbol", "C"))) for a in list(atoms or [])]


def _mopac_atoms_xyz(atoms):
    return np.asarray(
        [[float((a or {}).get("x", 0.0)), float((a or {}).get("y", 0.0)), float((a or {}).get("z", 0.0))] for a in list(atoms or [])],
        dtype=float,
    ).reshape((-1, 3))


def _mopac_atoms_from_symbols_xyz(symbols, xyz):
    arr = _as_xyz_array(xyz)
    sy = [_clean_element_symbol(s) for s in list(symbols or [])]
    if len(sy) != int(arr.shape[0]):
        raise ValueError("symbols and xyz size mismatch.")
    return [
        {"element": str(sym), "x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2])}
        for sym, coord in zip(sy, arr)
    ]


def _mopac_validate_atom_compatibility(a_atoms, b_atoms, label_a="reactant", label_b="product"):
    a = _mopac_normalize_atoms(a_atoms)
    b = _mopac_normalize_atoms(b_atoms)
    if len(a) != len(b):
        raise ValueError(f"{label_a} and {label_b} atom counts differ ({len(a)} vs {len(b)}).")
    a_sy = [_mopac_symbol_key(x) for x in _mopac_atoms_symbols(a)]
    b_sy = [_mopac_symbol_key(x) for x in _mopac_atoms_symbols(b)]
    if a_sy != b_sy:
        first = next((i for i, (sa, sb) in enumerate(zip(a_sy, b_sy)) if sa != sb), -1)
        raise ValueError(
            f"{label_a} and {label_b} atom ordering/elements differ"
            + (f" at atom {first + 1}: {a_sy[first]} vs {b_sy[first]}" if first >= 0 else ".")
        )
    return a, b


def _mopac_midpoint_atoms(reactant_atoms, product_atoms):
    r, p = _mopac_validate_atom_compatibility(reactant_atoms, product_atoms)
    symbols = _mopac_atoms_symbols(r)
    xyz = 0.5 * (_mopac_atoms_xyz(r) + _mopac_atoms_xyz(p))
    return _mopac_atoms_from_symbols_xyz(symbols, xyz)


def qmmm_mopac_normalize_operation(operation="probe"):
    """Return the canonical operation key used by the MOPAC QM/MM planner."""
    op = str(operation or "probe").strip().lower()
    op = str(QMMM_MOPAC_OPERATION_ALIASES.get(op, op))
    if op not in QMMM_MOPAC_OPERATION_PROFILES:
        op = "probe"
    return str(op)


def qmmm_mopac_operation_profile(operation="probe"):
    """Return a copy of the operation profile so callers can safely annotate it."""
    op = qmmm_mopac_normalize_operation(operation)
    data = dict(QMMM_MOPAC_OPERATION_PROFILES.get(op, QMMM_MOPAC_OPERATION_PROFILES["probe"]))
    data["operation"] = str(op)
    data["keywords"] = tuple(data.get("keywords", ()))
    data["required"] = tuple(data.get("required", ()))
    data["remove"] = tuple(data.get("remove", ()))
    return data


def _split_mopac_keywords(text):
    return [str(tok).strip().upper() for tok in str(text or "").replace("\n", " ").split() if str(tok).strip()]


def _mopac_keyword_name(token):
    tok = str(token or "").strip().upper()
    if "=" in tok:
        tok = tok.split("=", 1)[0]
    return tok


def qmmm_mopac_keywords_for_operation(
    operation="probe",
    base_keywords="",
    qm_atom_count=0,
    multiplicity=1,
):
    """
    Build operation-specific MOPAC keywords for electrostatic-embedding QM/MM.

    The function preserves a user-chosen semiempirical method when possible, but
    removes operation-conflicting control keywords.  This prevents stale GUI
    values such as `1SCF` from disabling an optimization or thermochemistry job.
    """
    profile = qmmm_mopac_operation_profile(operation)
    default_tokens = list(profile.get("keywords", ()) or ())
    base_tokens = _split_mopac_keywords(base_keywords)

    default_sets = {
        frozenset(_split_mopac_keywords(" ".join(v.get("keywords", ())))): key
        for key, v in QMMM_MOPAC_OPERATION_PROFILES.items()
    }
    base_set = frozenset(base_tokens)
    use_base = bool(base_tokens) and base_set not in default_sets
    tokens = list(base_tokens if use_base else default_tokens)
    if not tokens:
        tokens = list(default_tokens)

    # Keep the first recognizable method from a custom line, otherwise use PM7.
    method = ""
    for tok in tokens:
        if _mopac_keyword_name(tok) in _MOPAC_METHOD_KEYWORDS:
            method = str(tok)
            break
    filtered = []
    seen_names = set()
    remove_names = {_mopac_keyword_name(tok) for tok in tuple(profile.get("remove", ()) or ())}
    for tok in tokens:
        name = _mopac_keyword_name(tok)
        if name in remove_names:
            continue
        if name in _MOPAC_METHOD_KEYWORDS:
            if method and str(tok) != str(method):
                continue
        if name in seen_names and "=" not in str(tok):
            continue
        filtered.append(str(tok))
        seen_names.add(name)
    if method and _mopac_keyword_name(method) not in {_mopac_keyword_name(tok) for tok in filtered}:
        filtered.insert(0, str(method))
    elif not any(_mopac_keyword_name(tok) in _MOPAC_METHOD_KEYWORDS for tok in filtered):
        filtered.insert(0, "PM7")

    seen_names = {_mopac_keyword_name(tok) for tok in filtered}
    for tok in tuple(profile.get("required", ()) or ()):
        name = _mopac_keyword_name(tok)
        if name in seen_names:
            continue
        filtered.append(str(tok).upper())
        seen_names.add(name)

    try:
        qm_count = int(qm_atom_count or 0)
    except Exception:
        qm_count = 0
    try:
        mult_i = int(multiplicity or 1)
    except Exception:
        mult_i = 1
    if qm_count >= 120 and mult_i == 1 and "MOZYME" not in seen_names:
        # MOZYME is useful for large biomolecular semiempirical systems, but it
        # is kept automatic only for closed-shell jobs to avoid surprising users.
        filtered.append("MOZYME")
    return " ".join(str(tok).strip().upper() for tok in filtered if str(tok).strip())


def _qmmm_row_index(row):
    try:
        return int((row or {}).get("index"))
    except Exception:
        return None


def _qmmm_residue_key(row):
    return (
        str((row or {}).get("chain", "") or "").strip(),
        str((row or {}).get("resi", "") or "").strip(),
        str((row or {}).get("resn", "") or "").strip().upper(),
    )


def _qmmm_element(row):
    elem = str((row or {}).get("element", "") or "").strip().upper()
    if elem:
        return elem
    name = str((row or {}).get("name", "") or "").strip().upper()
    letters = "".join(ch for ch in name if ch.isalpha())
    if len(letters) >= 2 and letters[:2] in _QMMM_METAL_ELEMENTS.union({"CL", "BR"}):
        return letters[:2]
    return letters[:1] if letters else "C"


def _qmmm_row_xyz(row):
    try:
        if all(k in row for k in ("x", "y", "z")):
            return np.asarray([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
        if "coord" in row:
            return np.asarray(row.get("coord"), dtype=float).reshape(3)
        if "xyz" in row:
            return np.asarray(row.get("xyz"), dtype=float).reshape(3)
    except Exception:
        return None
    return None


def _qmmm_residue_kind(resn):
    rn = str(resn or "").strip().upper()
    if rn in _QMMM_WATER_RESN:
        return "water"
    if rn in _QMMM_ION_RESN:
        return "ion"
    if rn in _QMMM_BIOPOLYMER_RESN:
        return "biopolymer"
    return "hetero"


def _qmmm_group_atom_rows(atom_rows):
    groups = {}
    for raw in list(atom_rows or []):
        row = dict(raw or {})
        idx = _qmmm_row_index(row)
        if idx is None or idx < 0:
            continue
        key = _qmmm_residue_key(row)
        groups.setdefault(key, []).append(row)
    return groups


def _qmmm_group_heavy_count(rows):
    return int(sum(1 for row in list(rows or []) if _qmmm_element(row) != "H"))


def qmmm_recommend_region(
    atom_rows=None,
    operation="probe",
    ligand_resnames=None,
    qm_pocket_cutoff_ang=4.0,
    include_pocket=True,
    include_metals=True,
    max_qm_atoms=180,
):
    """
    Recommend a QM region from atom metadata.

    The rule is deliberately transparent: choose the largest hetero ligand (or a
    requested residue name), then optionally include nearby non-water residues
    and metals.  The caller still resolves exact indices and may override this.
    """
    rows = [dict(row or {}) for row in list(atom_rows or [])]
    groups = _qmmm_group_atom_rows(rows)
    warnings = []
    notes = []
    if not groups:
        return {"ok": False, "error": "No atom rows available for QM/MM region recommendation."}
    profile = qmmm_mopac_operation_profile(operation)
    policy = str(profile.get("region_policy", "ligand_only") or "ligand_only")
    requested_resn = {str(x or "").strip().upper() for x in list(ligand_resnames or []) if str(x or "").strip()}
    candidates = []
    for key, atoms in groups.items():
        resn = str(key[2] or "").upper()
        kind = _qmmm_residue_kind(resn)
        if requested_resn and resn not in requested_resn:
            continue
        if (not requested_resn) and kind != "hetero":
            continue
        heavy = _qmmm_group_heavy_count(atoms)
        if heavy <= 0:
            continue
        candidates.append((key, atoms, heavy))
    if not candidates:
        return {"ok": False, "error": "No hetero ligand/cofactor residue was found for automatic QM region selection."}
    candidates = sorted(candidates, key=lambda item: (int(item[2]), str(item[0][2])), reverse=True)
    center_key, center_atoms, _heavy = candidates[0]
    qm_keys = {center_key}
    center_xyz = [_qmmm_row_xyz(row) for row in center_atoms]
    center_xyz = [xyz for xyz in center_xyz if xyz is not None]
    has_coords = bool(center_xyz)
    try:
        cutoff = float(qm_pocket_cutoff_ang)
    except Exception:
        cutoff = 4.0
    cutoff = float(max(0.1, min(12.0, cutoff)))
    include_nearby = bool(include_pocket) or policy in {"ligand_interface", "reaction_core"}
    if include_nearby and has_coords:
        for key, atoms in groups.items():
            if key == center_key:
                continue
            resn = str(key[2] or "").upper()
            kind = _qmmm_residue_kind(resn)
            if kind == "water":
                continue
            row_xyz = [_qmmm_row_xyz(row) for row in atoms]
            row_xyz = [xyz for xyz in row_xyz if xyz is not None]
            if not row_xyz:
                continue
            min_dist = min(float(np.linalg.norm(a - b)) for a in center_xyz for b in row_xyz)
            elem_set = {_qmmm_element(row) for row in atoms}
            is_metal = bool(elem_set.intersection(_QMMM_METAL_ELEMENTS))
            if min_dist <= cutoff and (kind != "ion" or (bool(include_metals) and is_metal)):
                qm_keys.add(key)
    elif include_nearby:
        warnings.append("No coordinates were available, so pocket residues could not be added automatically.")
    if bool(include_metals) and has_coords:
        for key, atoms in groups.items():
            if key in qm_keys:
                continue
            elem_set = {_qmmm_element(row) for row in atoms}
            if not elem_set.intersection(_QMMM_METAL_ELEMENTS):
                continue
            row_xyz = [_qmmm_row_xyz(row) for row in atoms]
            row_xyz = [xyz for xyz in row_xyz if xyz is not None]
            if not row_xyz:
                continue
            min_dist = min(float(np.linalg.norm(a - b)) for a in center_xyz for b in row_xyz)
            if min_dist <= max(2.8, cutoff):
                qm_keys.add(key)
                notes.append(f"Included nearby metal/ion {key[2]} {key[0]}:{key[1]}.")

    qm_indices = sorted(
        int(_qmmm_row_index(row))
        for key in qm_keys
        for row in list(groups.get(key, []) or [])
        if _qmmm_row_index(row) is not None
    )
    all_indices = sorted(int(_qmmm_row_index(row)) for row in rows if _qmmm_row_index(row) is not None)
    qset = set(qm_indices)
    mm_indices = [idx for idx in all_indices if idx not in qset]
    try:
        max_atoms = int(max_qm_atoms or 0)
    except Exception:
        max_atoms = 0
    if max_atoms > 0 and len(qm_indices) > max_atoms:
        warnings.append(
            f"Recommended QM region has {len(qm_indices)} atoms, above the requested limit of {max_atoms}."
        )
    if len(qm_indices) < 2:
        warnings.append("Recommended QM region is very small; inspect charge and chemistry before running.")
    return {
        "ok": True,
        "operation": qmmm_mopac_normalize_operation(operation),
        "policy": str(policy),
        "center_residue": {
            "chain": str(center_key[0]),
            "resi": str(center_key[1]),
            "resn": str(center_key[2]),
        },
        "qm_residue_keys": [tuple(key) for key in sorted(qm_keys)],
        "qm_indices": list(qm_indices),
        "mm_indices": list(mm_indices),
        "qm_atom_count": int(len(qm_indices)),
        "mm_atom_count": int(len(mm_indices)),
        "cutoff_ang": float(cutoff),
        "include_pocket": bool(include_nearby),
        "include_metals": bool(include_metals),
        "warnings": list(warnings),
        "notes": list(notes),
        "selection_hint": f"{center_key[2]} {center_key[0]}:{center_key[1]} + {max(0, len(qm_keys) - 1)} nearby residue(s)",
    }


def _qmmm_infer_covalent_cutoff(elem_a, elem_b):
    ea = str(elem_a or "C").strip().upper()
    eb = str(elem_b or "C").strip().upper()
    if ea in _QMMM_METAL_ELEMENTS or eb in _QMMM_METAL_ELEMENTS:
        return 0.0
    ra = float(_COVALENT_RADII_A.get(ea, 0.77))
    rb = float(_COVALENT_RADII_A.get(eb, 0.77))
    return float(min(2.25, ra + rb + 0.45))


def qmmm_assess_interface(atom_rows=None, qm_indices=None, mm_indices=None, bond_pairs=None):
    """
    Diagnose QM/MM boundary quality.

    Link atoms are not generated by this plugin, so a covalent bond crossing the
    QM/MM boundary is treated as a strong warning.  Users should include the
    whole covalent fragment/residue in QM or redesign the region.
    """
    rows = {int(_qmmm_row_index(row)): dict(row or {}) for row in list(atom_rows or []) if _qmmm_row_index(row) is not None}
    qset = {int(i) for i in list(qm_indices or [])}
    mset = {int(i) for i in list(mm_indices or [])}
    warnings = []
    close_contacts = []
    cross_bonds = []
    if not qset:
        warnings.append("QM region is empty.")
    if not mset:
        warnings.append("MM region is empty.")
    if qset.intersection(mset):
        warnings.append("QM and MM regions overlap.")
    explicit_bonds = []
    for pair in list(bond_pairs or []):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            explicit_bonds.append((int(pair[0]), int(pair[1])))
        except Exception:
            continue
    if explicit_bonds:
        for a, b in explicit_bonds:
            if (a in qset and b in mset) or (b in qset and a in mset):
                cross_bonds.append((int(a), int(b), "explicit"))
    else:
        q_rows = [rows[i] for i in qset if i in rows]
        m_rows = [rows[i] for i in mset if i in rows]
        for qa in q_rows:
            qxyz = _qmmm_row_xyz(qa)
            if qxyz is None:
                continue
            qi = int(qa.get("index"))
            qe = _qmmm_element(qa)
            for ma in m_rows:
                mxyz = _qmmm_row_xyz(ma)
                if mxyz is None:
                    continue
                mi = int(ma.get("index"))
                me = _qmmm_element(ma)
                dist = float(np.linalg.norm(qxyz - mxyz))
                cutoff = _qmmm_infer_covalent_cutoff(qe, me)
                if cutoff > 0.0 and dist <= cutoff:
                    cross_bonds.append((int(qi), int(mi), f"inferred:{dist:.2f}A"))
                elif dist <= 2.4:
                    close_contacts.append((int(qi), int(mi), float(dist)))
    if cross_bonds:
        warnings.append(
            "Covalent QM/MM boundary detected. Include the full bonded fragment in QM; link atoms are not inserted automatically."
        )
    return {
        "ok": bool(not warnings),
        "warnings": list(warnings),
        "cross_boundary_bonds": list(cross_bonds),
        "close_contacts": list(close_contacts[:50]),
        "qm_atom_count": int(len(qset)),
        "mm_atom_count": int(len(mset)),
    }


def compute_phi_on_qm_atoms(xyz_qm_ang, xyz_mm_ang, q_mm, dielectric=1.0, min_distance_ang=1.0e-4):
    """
    Electrostatic potential on QM atoms from MM charges in atomic units (Eh/e).
    Coordinates are in Angstrom and charges are in elementary charge units.
    """
    qxyz = _as_xyz_array(xyz_qm_ang)
    mxyz = _as_xyz_array(xyz_mm_ang)
    q_mm_arr = _as_1d(q_mm, name="q_mm")
    if int(mxyz.shape[0]) != int(q_mm_arr.size):
        raise ValueError("xyz_mm and q_mm size mismatch.")
    eps = float(max(1.0e-12, dielectric))
    min_r = float(max(1.0e-8, min_distance_ang))
    phi = np.zeros((int(qxyz.shape[0]),), dtype=float)
    for i in range(int(qxyz.shape[0])):
        dr = mxyz - qxyz[i][None, :]
        r_ang = np.linalg.norm(dr, axis=1)
        r_ang = np.clip(r_ang, min_r, None)
        r_bohr = r_ang * ANGSTROM_TO_BOHR
        phi[i] = float(np.sum(q_mm_arr / (eps * r_bohr)))
    return phi


def electrostatic_backreaction_openmm(
    xyz_qm_ang,
    xyz_mm_ang,
    q_qm,
    q_mm,
    dielectric=1.0,
    min_distance_nm=1.0e-6,
):
    """
    Returns electrostatic QM/MM coupling:
      energy_kj_mol
      forces_qm_kj_mol_nm (Nq,3)
      forces_mm_kj_mol_nm (Nm,3)
    """
    qxyz = _as_xyz_array(xyz_qm_ang) / 10.0
    mxyz = _as_xyz_array(xyz_mm_ang) / 10.0
    q_qm_arr = _as_1d(q_qm, name="q_qm")
    q_mm_arr = _as_1d(q_mm, name="q_mm")
    if int(qxyz.shape[0]) != int(q_qm_arr.size):
        raise ValueError("xyz_qm and q_qm size mismatch.")
    if int(mxyz.shape[0]) != int(q_mm_arr.size):
        raise ValueError("xyz_mm and q_mm size mismatch.")
    eps = float(max(1.0e-12, dielectric))
    min_r = float(max(1.0e-8, min_distance_nm))
    fq = np.zeros_like(qxyz, dtype=float)
    fm = np.zeros_like(mxyz, dtype=float)
    energy = 0.0
    pref_k = float(OPENMM_COULOMB_CONSTANT / eps)
    for i in range(int(qxyz.shape[0])):
        for j in range(int(mxyz.shape[0])):
            rij = mxyz[j] - qxyz[i]
            r = float(np.linalg.norm(rij))
            if r < min_r:
                r = min_r
            qq = float(q_qm_arr[i] * q_mm_arr[j])
            energy += float(pref_k * qq / r)
            inv_r3 = 1.0 / float(r * r * r)
            fij = -float(pref_k * qq) * rij * inv_r3
            fq[i] += fij
            fm[j] -= fij
    return {
        "energy_kj_mol": float(energy),
        "forces_qm_kj_mol_nm": fq,
        "forces_mm_kj_mol_nm": fm,
    }


def assemble_force_correction(
    n_atoms,
    qm_indices,
    mm_indices,
    f_mm_qm_internal=None,
    f_mm_qmmm_coul=None,
    f_qm_internal=None,
    f_qmmm_elec_qm=None,
    f_qmmm_elec_mm=None,
):
    """
    Assemble correction:
      Fcorr = -F_mm,int(QM) - F_mm,Coul(QM-MM) + F_qm,int + F_qmmm,elec
    Arrays must be in OpenMM units (kJ/mol/nm).
    """
    n = int(max(1, int(n_atoms)))
    out = np.zeros((n, 3), dtype=float)
    qidx = [int(i) for i in list(qm_indices or [])]
    midx = [int(i) for i in list(mm_indices or [])]

    def _apply_rows(indices, rows, sign=1.0):
        if rows is None:
            return
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Force block must have shape (N,3).")
        if int(arr.shape[0]) != int(len(indices)):
            raise ValueError("Force block row count mismatch with atom indices.")
        for k, ii in enumerate(indices):
            if ii < 0 or ii >= n:
                raise ValueError("Atom index out of range in force assembly.")
            out[ii, :] += float(sign) * arr[k, :]

    _apply_rows(qidx, f_mm_qm_internal, sign=-1.0)
    _apply_rows(qidx, f_mm_qmmm_coul, sign=-1.0)
    _apply_rows(qidx, f_qm_internal, sign=+1.0)
    _apply_rows(qidx, f_qmmm_elec_qm, sign=+1.0)
    _apply_rows(midx, f_qmmm_elec_mm, sign=+1.0)
    return out


def build_mopac_input_text(
    qm_atomic_numbers,
    xyz_qm_ang,
    keywords="PM7 1SCF GRAD AUX QMMM",
    total_charge=0,
    multiplicity=1,
    title="ProtLigInteract QM/MM step",
):
    xyz = _as_xyz_array(xyz_qm_ang)
    z = [int(v) for v in list(np.asarray(qm_atomic_numbers).reshape(-1))]
    if len(z) != int(xyz.shape[0]):
        raise ValueError("qm_atomic_numbers and xyz_qm size mismatch.")
    kw = str(keywords or "").strip()
    kw_u = kw.upper()
    if "CHARGE=" not in kw_u:
        kw = (kw + f" CHARGE={int(total_charge)}").strip()
    mult_map = {
        1: "SINGLET",
        2: "DOUBLET",
        3: "TRIPLET",
        4: "QUARTET",
        5: "QUINTET",
    }
    mkw = str(mult_map.get(int(max(1, multiplicity)), "")).strip()
    if mkw and mkw not in kw_u:
        kw = (kw + " " + mkw).strip()
    lines = [kw, str(title or "ProtLigInteract QM/MM step"), ""]
    for zi, coord in zip(z, xyz):
        sym = _symbol_from_atomic_number(int(zi))
        x, y, zc = [float(v) for v in coord]
        # Use optimization flags=1 so MOPAC can compute gradients cleanly for this snapshot.
        lines.append(f"{sym:2s} {x:14.8f} 1 {y:14.8f} 1 {zc:14.8f} 1")
    return "\n".join(lines).rstrip() + "\n"


def build_mopac_mol_in_text(phi_qm):
    vals = np.asarray(phi_qm, dtype=float).reshape(-1)
    lines = [str(int(vals.size))]
    for i, v in enumerate(vals, start=1):
        lines.append(f"{int(i):5d} {float(v): .12e}")
    return "\n".join(lines).rstrip() + "\n"


def _extract_first_float(text, patterns):
    src = str(text or "")
    for pat in list(patterns or []):
        m = re.search(str(pat), src, flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        try:
            return _mopac_float(m.group(1), default=np.nan)
        except Exception:
            continue
    return None


def _extract_float_vector_after_key(text, key):
    src = str(text or "")
    k = str(key or "").strip()
    if not k:
        return np.zeros((0,), dtype=float)
    pat = (
        rf"{re.escape(k)}[^\n=]*=\s*(.*?)"
        rf"(?=\n\s*[A-Z][A-Z0-9_./:\-\[\]\(\)]*[^\n=]*=|\Z)"
    )
    m = re.search(pat, src, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    if not m:
        return np.zeros((0,), dtype=float)
    blob = str(m.group(1) or "")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", blob)
    if not nums:
        return np.zeros((0,), dtype=float)
    return np.asarray([_mopac_float(x, default=np.nan) for x in nums], dtype=float).reshape(-1)


def _extract_token_vector_after_key(text, key):
    blob = str("")
    src = str(text or "")
    k = str(key or "").strip()
    if k:
        pat = (
            rf"{re.escape(k)}[^\n=]*=\s*(.*?)"
            rf"(?=\n\s*[A-Z][A-Z0-9_./:\-\[\]\(\)]*[^\n=]*=|\Z)"
        )
        m = re.search(pat, src, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
        if m:
            blob = str(m.group(1) or "")
    return [str(tok).strip() for tok in blob.split() if str(tok).strip()]


def parse_mopac_aux_text(aux_text):
    txt = str(aux_text or "")
    out = {
        "ok": False,
        "energy_hartree": None,
        "heat_of_formation_ev": None,
        "atom_symbols": [],
        "optimized_coords_ang": np.zeros((0, 3), dtype=float),
        "atom_charges": np.zeros((0,), dtype=float),
        "gradients_kcal_mol_ang": np.zeros((0, 3), dtype=float),
        "gradients_kj_mol_nm": np.zeros((0, 3), dtype=float),
        "freq_cm_signed": np.zeros((0,), dtype=float),
        "freq_cm_real": np.zeros((0,), dtype=float),
        "freq_cm_imag": np.zeros((0,), dtype=float),
        "normal_modes_cart": np.zeros((0, 0, 0), dtype=float),
        "thermochemistry": {},
    }
    e_ev = _extract_first_float(
        txt,
        patterns=[
            r"TOTAL_ENERGY[^=\n]*=\s*([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)",
            r"HEAT_OF_FORMATION[^=\n]*=\s*([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)",
        ],
    )
    if e_ev is not None:
        out["energy_hartree"] = float(e_ev) * float(EV_TO_HARTREE)
    hof_ev = _extract_first_float(
        txt,
        patterns=[
            r"HEAT_OF_FORMATION[^=\n]*=\s*([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)",
        ],
    )
    if hof_ev is not None:
        out["heat_of_formation_ev"] = float(hof_ev)
    q = _extract_float_vector_after_key(txt, "ATOM_CHARGES")
    if q.size > 0:
        out["atom_charges"] = np.asarray(q, dtype=float).reshape(-1)
    g = _extract_float_vector_after_key(txt, "GRADIENTS")
    if g.size > 0:
        if int(g.size % 3) != 0:
            # Keep full multiples of 3 only.
            g = g[: int(g.size // 3) * 3]
        if g.size > 0:
            g3 = np.asarray(g, dtype=float).reshape((-1, 3))
            out["gradients_kcal_mol_ang"] = g3
            out["gradients_kj_mol_nm"] = g3 * 41.84
    symbols = [_clean_element_symbol(tok) for tok in _extract_token_vector_after_key(txt, "ATOM_EL")]
    coords = _extract_float_vector_after_key(txt, "ATOM_X_OPT:ANGSTROMS")
    if coords.size <= 0:
        coords = _extract_float_vector_after_key(txt, "ATOM_X:ANGSTROMS")
    nat = int(len(symbols)) if symbols else int(coords.size // 3)
    if nat > 0 and coords.size >= int(3 * nat):
        out["atom_symbols"] = list(symbols or (["C"] * nat))
        out["optimized_coords_ang"] = np.asarray(coords[: int(3 * nat)], dtype=float).reshape((nat, 3))
    raw_freq = _extract_float_vector_after_key(txt, "VIB._FREQ:CM(-1)")
    if raw_freq.size <= 0:
        raw_freq = _extract_float_vector_after_key(txt, "VIB_FREQ:CM(-1)")
    if raw_freq.size > 0:
        finite_freq = np.asarray([float(v) for v in raw_freq if np.isfinite(v)], dtype=float).reshape(-1)
        n_vib = int(max(1, 3 * nat - (5 if nat == 2 else 6))) if nat > 0 else int(finite_freq.size)
        signed = finite_freq[: int(min(finite_freq.size, n_vib))]
        out["freq_cm_signed"] = signed
        out["freq_cm_real"] = np.asarray([max(0.0, float(v)) for v in signed], dtype=float).reshape(-1)
        out["freq_cm_imag"] = np.asarray([abs(float(v)) if float(v) < 0.0 else 0.0 for v in signed], dtype=float).reshape(-1)
        raw_modes = _extract_float_vector_after_key(txt, "NORMAL_MODES")
        if nat > 0 and raw_modes.size >= int(9 * nat * nat):
            ncart = int(3 * nat)
            nmode = int(min(signed.size, raw_modes.size // ncart))
            if nmode > 0:
                mode_mat = np.asarray(raw_modes[: int(ncart * nmode)], dtype=float).reshape((ncart, nmode))
                out["normal_modes_cart"] = np.asarray(
                    [mode_mat[:, i].reshape((nat, 3)) for i in range(nmode)],
                    dtype=float,
                )
    thermo = {}
    for label, patterns in {
        "zero_point_energy": (
            r"ZERO_POINT_ENERGY[^=\n]*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
            r"ZPE[^=\n]*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
        ),
        "enthalpy": (
            r"ENTHALPY[^=\n]*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
            r"HEAT_CONTENT[^=\n]*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
        ),
        "entropy": (
            r"ENTROPY[^=\n]*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
        ),
        "gibbs_free_energy": (
            r"GIBBS[^=\n]*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
            r"FREE_ENERGY[^=\n]*=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)",
        ),
    }.items():
        val = _extract_first_float(txt, patterns=patterns)
        if val is not None:
            thermo[str(label)] = float(val)
    out["thermochemistry"] = dict(thermo)
    out["ok"] = bool(
        (out["energy_hartree"] is not None)
        and (int(out["atom_charges"].size) > 0)
        and (int(out["gradients_kj_mol_nm"].shape[0]) > 0)
    )
    return out


def mopac_validate_ts_frequencies(freq_cm_signed=None, freq_cm_imag=None, threshold_cm=10.0):
    """
    Validate the frequency signature of a first-order saddle point.

    MOPAC reports TS modes as "negative" vibrations in normal output/AUX data.
    Some downstream parsers instead store imaginary magnitudes separately.  This
    helper accepts both conventions and requires exactly one mode above the
    chosen threshold before IRC is considered chemically meaningful.
    """
    try:
        signed = np.asarray(freq_cm_signed, dtype=float).reshape(-1)
    except Exception:
        signed = np.zeros((0,), dtype=float)
    try:
        imag = np.asarray(freq_cm_imag, dtype=float).reshape(-1)
    except Exception:
        imag = np.zeros((0,), dtype=float)
    n = int(max(signed.size, imag.size))
    if signed.size < n:
        signed = np.pad(signed, (0, n - signed.size), mode="constant")
    if imag.size < n:
        imag = np.pad(imag, (0, n - imag.size), mode="constant")
    try:
        threshold = float(abs(float(threshold_cm)))
    except Exception:
        threshold = 10.0
    modes = []
    for idx in range(n):
        s = float(signed[idx])
        im = float(imag[idx])
        if np.isfinite(im) and abs(im) >= threshold:
            modes.append({"mode_index": int(idx), "imag_cm1": float(abs(im)), "source": "imag"})
        elif np.isfinite(s) and s <= -threshold:
            modes.append({"mode_index": int(idx), "imag_cm1": float(abs(s)), "source": "negative"})
    primary = max(modes, key=lambda row: float(row["imag_cm1"])) if modes else {}
    return {
        "ok": bool(len(modes) == 1),
        "imaginary_mode_count": int(len(modes)),
        "imaginary_modes": list(modes),
        "primary_mode_index": int(primary.get("mode_index", -1)) if primary else -1,
        "primary_imag_cm1": float(primary.get("imag_cm1", 0.0)) if primary else 0.0,
        "threshold_cm": float(threshold),
    }


def _mopac_keyword_tokens_for_stage(stage, method="PM7", use_forcets=False, irc_direction="both", x_priority_ang=0.05):
    stage_key = str(stage or "ts").strip().lower()
    meth = str(method or "PM7").strip().upper()
    if _mopac_keyword_name(meth) not in _MOPAC_METHOD_KEYWORDS:
        meth = "PM7"
    if stage_key in {"qst2", "saddle"}:
        return [meth, "SADDLE", "AUX", "XYZ"]
    if stage_key in {"qst3", "ts", "ts_refine", "transition_state"}:
        return [meth, "TS", "AUX", "XYZ"]
    if stage_key in {"frequency", "freq", "force", "ts_frequency"}:
        return [meth, "FORCETS" if bool(use_forcets) else "FORCE", "AUX", "XYZ"]
    if stage_key in {"irc", "reaction_path"}:
        direction = str(irc_direction or "both").strip().lower()
        if direction in {"reverse", "backward", "-1", "negative"}:
            irc_kw = "IRC=-1"
        elif direction in {"forward", "+1", "1", "positive"}:
            irc_kw = "IRC=1"
        else:
            irc_kw = "IRC=1*"
        try:
            xpri = float(x_priority_ang)
        except Exception:
            xpri = 0.05
        xpri = float(max(0.001, min(10.0, xpri)))
        return [meth, "FORCE", irc_kw, f"X-PRIORITY={xpri:.4f}", "AUX", "XYZ"]
    return [meth, "AUX", "XYZ"]


def mopac_reaction_keywords_for_stage(
    stage,
    method="PM7",
    base_keywords="",
    use_forcets=False,
    irc_direction="both",
    x_priority_ang=0.05,
):
    """
    Build a safe MOPAC keyword line for one reaction-search stage.

    QST2/QST3 are Leonardo workflow names.  MOPAC itself provides SADDLE for
    two-endpoint TS location, TS for refinement, FORCE/FORCETS for Hessian
    validation, and IRC=... for reaction-path following.
    """
    method_from_base = _mopac_extract_method_from_keywords(base_keywords, default=method)
    defaults = _mopac_keyword_tokens_for_stage(
        stage,
        method=method_from_base,
        use_forcets=bool(use_forcets),
        irc_direction=irc_direction,
        x_priority_ang=float(x_priority_ang),
    )
    remove = {
        "1SCF",
        "GRAD",
        "QMMM",
        "THERMO",
        "SADDLE",
        "TS",
        "EF",
        "FORCE",
        "FORCETS",
        "IRC",
        "X-PRIORITY",
    }
    keep_names = {_mopac_keyword_name(tok) for tok in defaults}
    tokens = []
    seen = set()
    for tok in _split_mopac_keywords(base_keywords):
        name = _mopac_keyword_name(tok)
        if name in remove and name not in keep_names:
            continue
        if name in remove and name in keep_names:
            continue
        if name in _MOPAC_METHOD_KEYWORDS:
            continue
        if name in seen and "=" not in tok:
            continue
        tokens.append(str(tok).upper())
        seen.add(name)
    for tok in defaults:
        name = _mopac_keyword_name(tok)
        if name in seen and "=" not in str(tok):
            continue
        tokens.append(str(tok).upper())
        seen.add(name)
    return " ".join(tok for tok in tokens if str(tok).strip())


def build_mopac_reaction_input_text(
    reactant_atoms,
    product_atoms=None,
    keywords="PM7 TS AUX XYZ",
    total_charge=0,
    multiplicity=1,
    title="ProtLigInteract MOPAC reaction job",
):
    """
    Build Cartesian MOPAC input for one or two geometries.

    MOPAC SADDLE expects reactant and product geometries with identical atom
    order.  Keeping the atom-order check here prevents a silent, chemically
    meaningless interpolation/search when product atoms were exported in a
    different order.
    """
    reactant = _mopac_normalize_atoms(reactant_atoms)
    product = None
    if product_atoms is not None:
        reactant, product = _mopac_validate_atom_compatibility(reactant, product_atoms)
    kw = str(keywords or "PM7 TS AUX XYZ").strip()
    kw_u = kw.upper()
    if "CHARGE=" not in kw_u:
        kw = (kw + f" CHARGE={int(total_charge)}").strip()
    mult_map = {1: "SINGLET", 2: "DOUBLET", 3: "TRIPLET", 4: "QUARTET", 5: "QUINTET"}
    spin_kw = str(mult_map.get(int(max(1, multiplicity)), "")).strip()
    if spin_kw and spin_kw not in kw_u:
        kw = (kw + " " + spin_kw).strip()

    def _coord_flag(atom, axis):
        if bool(atom.get("frozen", False)) or atom.get("optimize", True) is False:
            return 0
        flags = atom.get("coord_flags", None)
        if isinstance(flags, (list, tuple)) and len(flags) >= 3:
            idx = {"x": 0, "y": 1, "z": 2}.get(str(axis), 0)
            try:
                return 1 if int(flags[idx]) != 0 else 0
            except Exception:
                return 1
        for key in (f"{axis}_opt", f"{axis}_flag"):
            if key in atom:
                val = atom.get(key)
                if isinstance(val, bool):
                    return 1 if val else 0
                try:
                    return 1 if int(val) != 0 else 0
                except Exception:
                    return 1
        return 1

    def _coord_lines(atoms):
        rows = []
        for atom in list(atoms or []):
            sym = _clean_element_symbol(atom.get("element", atom.get("symbol", "C")))
            fx = _coord_flag(atom, "x")
            fy = _coord_flag(atom, "y")
            fz = _coord_flag(atom, "z")
            rows.append(
                f"{sym:2s} {float(atom.get('x', 0.0)):14.8f} {fx:d}"
                f" {float(atom.get('y', 0.0)):14.8f} {fy:d}"
                f" {float(atom.get('z', 0.0)):14.8f} {fz:d}"
            )
        return rows

    lines = [kw, str(title or "ProtLigInteract MOPAC reaction job"), ""]
    lines.extend(_coord_lines(reactant))
    if product is not None:
        lines.append("")
        lines.extend(_coord_lines(product))
    return "\n".join(lines).rstrip() + "\n"


def parse_mopac_arc_geometries(arc_text, atom_symbols=None, expected_atoms=0):
    """
    Recover trajectory-like geometry blocks from a MOPAC ARC file.

    IRC and SADDLE jobs can emit multiple Cartesian blocks.  The parser is kept
    permissive because ARC headers differ across MOPAC versions, but a block is
    accepted only when it has the expected atom count if that count is known.
    """
    txt = str(arc_text or "")
    expected = int(max(0, int(expected_atoms or 0)))
    coord_pat = re.compile(
        r"^\s*([A-Za-z]{1,2})\s+"
        r"([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s+[-+]?\d+\s+"
        r"([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s+[-+]?\d+\s+"
        r"([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s+[-+]?\d+",
        flags=re.IGNORECASE,
    )
    blocks = []
    block_symbols = []
    cur_xyz = []
    cur_symbols = []

    def _flush():
        nonlocal cur_xyz, cur_symbols
        if cur_xyz and (expected <= 0 or len(cur_xyz) == expected):
            blocks.append(np.asarray(cur_xyz, dtype=float).reshape((len(cur_xyz), 3)))
            block_symbols.append([_clean_element_symbol(s) for s in cur_symbols])
        cur_xyz = []
        cur_symbols = []

    for line in txt.splitlines():
        m = coord_pat.match(line)
        if not m:
            _flush()
            continue
        cur_symbols.append(_clean_element_symbol(m.group(1)))
        cur_xyz.append([_mopac_float(m.group(2)), _mopac_float(m.group(3)), _mopac_float(m.group(4))])
    _flush()
    if not blocks:
        return {
            "ok": False,
            "symbols": [_clean_element_symbol(s) for s in list(atom_symbols or [])],
            "coords_ang": np.zeros((0, 0, 3), dtype=float),
            "energy_hartree": np.zeros((0,), dtype=float),
        }
    nat = int(blocks[0].shape[0])
    good_blocks = [b for b in blocks if int(b.shape[0]) == nat]
    sy = [_clean_element_symbol(s) for s in list(atom_symbols or [])]
    if len(sy) != nat:
        sy = list(block_symbols[0] if block_symbols else ["C"] * nat)
    energy_kcal = [
        _mopac_float(m.group(1))
        for m in re.finditer(
            r"HEAT\s+OF\s+FORMATION\s*=\s*([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s*KCAL",
            txt,
            flags=re.IGNORECASE,
        )
    ]
    if len(energy_kcal) >= len(good_blocks):
        e = np.asarray(energy_kcal[-len(good_blocks) :], dtype=float) * float(KCAL_TO_HARTREE)
    else:
        e = np.full((len(good_blocks),), np.nan, dtype=float)
    return {
        "ok": True,
        "symbols": list(sy),
        "coords_ang": np.asarray(good_blocks, dtype=float),
        "energy_hartree": e,
    }


def parse_mopac_xyz_trajectory(xyz_text, expected_atoms=0):
    """
    Parse MOPAC multi-XYZ trajectory files.

    Recent MOPAC builds commonly write IRC trajectories to `job.xyz` instead of
    repeating every frame in the ARC file.  The comment line usually stores the
    heat of formation, so this parser keeps both coordinates and energies for
    downstream plotting.
    """
    lines = str(xyz_text or "").splitlines()
    expected = int(max(0, int(expected_atoms or 0)))
    frames = []
    frame_symbols = []
    comments = []
    energy_kcal = []
    i = 0
    while i < len(lines):
        raw_count = str(lines[i] or "").strip()
        if not raw_count:
            i += 1
            continue
        try:
            nat = int(raw_count.split()[0])
        except Exception:
            i += 1
            continue
        if nat <= 0 or i + nat + 1 >= len(lines):
            i += 1
            continue
        comment = str(lines[i + 1] or "").strip()
        coords = []
        symbols = []
        ok_block = True
        for row in lines[i + 2 : i + 2 + nat]:
            parts = str(row or "").split()
            if len(parts) < 4:
                ok_block = False
                break
            try:
                coords.append([_mopac_float(parts[1]), _mopac_float(parts[2]), _mopac_float(parts[3])])
                symbols.append(_clean_element_symbol(parts[0]))
            except Exception:
                ok_block = False
                break
        if ok_block and (expected <= 0 or nat == expected):
            frames.append(np.asarray(coords, dtype=float).reshape((nat, 3)))
            frame_symbols.append(list(symbols))
            comments.append(comment)
            m = re.search(
                r"HEAT\s+OF\s+FORMATION\s*=\s*([-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?)\s*KCAL",
                comment,
                flags=re.IGNORECASE,
            )
            energy_kcal.append(_mopac_float(m.group(1)) if m else np.nan)
        i += nat + 2

    if not frames:
        return {
            "ok": False,
            "symbols": [],
            "coords_ang": np.zeros((0, 0, 3), dtype=float),
            "energy_hartree": np.zeros((0,), dtype=float),
            "comments": [],
        }
    nat = int(frames[0].shape[0])
    good_idx = [idx for idx, frame in enumerate(frames) if int(frame.shape[0]) == nat]
    good_frames = [frames[idx] for idx in good_idx]
    good_comments = [comments[idx] for idx in good_idx]
    good_energy = [energy_kcal[idx] for idx in good_idx]
    return {
        "ok": True,
        "symbols": list(frame_symbols[good_idx[0]]),
        "coords_ang": np.asarray(good_frames, dtype=float),
        "energy_hartree": np.asarray(good_energy, dtype=float) * float(KCAL_TO_HARTREE),
        "comments": list(good_comments),
    }


def mopac_build_irc_profile_payload(coords_ang, energy_hartree, symbols):
    """
    Build a plotting/export payload from parsed MOPAC IRC frames.

    The coordinate is an approximate cumulative mass-weighted Cartesian arc
    length.  When energies are available, the highest-energy frame is treated as
    the TS frame and set to reaction coordinate 0.
    """
    coords = np.asarray(coords_ang, dtype=float)
    energies = np.asarray(energy_hartree, dtype=float).reshape(-1)
    if coords.ndim != 3 or coords.shape[0] < 1 or coords.shape[2] != 3:
        return {"ok": False, "error": "invalid_irc_frames"}
    n = int(coords.shape[0])
    if energies.size < n:
        energies = np.pad(energies, (0, n - energies.size), constant_values=np.nan)
    energies = energies[:n]
    sy = [_clean_element_symbol(s) for s in list(symbols or [])]
    if len(sy) != int(coords.shape[1]):
        sy = ["C"] * int(coords.shape[1])
    masses = np.asarray([_mopac_atomic_mass_amu(s) for s in sy], dtype=float).reshape((-1, 1))
    bohr = coords * float(ANGSTROM_TO_BOHR)

    def _mw_dist(i, j):
        delta = bohr[int(j), :, :] - bohr[int(i), :, :]
        return float(np.sqrt(np.sum(masses * delta * delta)))

    finite_energy = np.where(np.isfinite(energies))[0]
    ts_idx = int(finite_energy[np.argmax(energies[finite_energy])]) if finite_energy.size else int(n // 2)
    rc = np.zeros((n,), dtype=float)
    for i in range(ts_idx + 1, n):
        rc[i] = rc[i - 1] + _mw_dist(i - 1, i)
    for i in range(ts_idx - 1, -1, -1):
        rc[i] = rc[i + 1] - _mw_dist(i, i + 1)
    if finite_energy.size:
        rel = (energies - float(np.nanmin(energies))) / float(KCAL_TO_HARTREE)
    else:
        rel = np.full((n,), np.nan, dtype=float)
    return {
        "ok": True,
        "point": [int(i) for i in range(n)],
        "coordinate_amu05_bohr": [float(v) for v in rc.tolist()],
        "energy_hartree": [float(v) if np.isfinite(v) else None for v in energies.tolist()],
        "relative_energy_kcal_mol": [float(v) if np.isfinite(v) else None for v in rel.tolist()],
        "branch": ["backward" if i < ts_idx else ("forward" if i > ts_idx else "ts") for i in range(n)],
        "ts_frame_index": int(ts_idx),
        "n_points": int(n),
        "coordinate_unit": "amu^0.5 bohr",
    }


def _mopac_downsample_indices(n_items, max_items):
    n = int(max(0, n_items))
    try:
        m = int(max_items)
    except Exception:
        m = n
    if n <= 0 or m <= 0 or n <= m:
        return list(range(n))
    return [int(round(v)) for v in np.linspace(0, n - 1, m)]


def _write_mopac_xyz_frames(path, symbols, coords_ang, comments=None):
    coords = np.asarray(coords_ang, dtype=float)
    if coords.ndim == 2:
        coords = coords.reshape((1, coords.shape[0], coords.shape[1]))
    sy = [_clean_element_symbol(s) for s in list(symbols or [])]
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("XYZ frame coordinates must have shape (frames,natoms,3).")
    if len(sy) != int(coords.shape[1]):
        sy = ["C"] * int(coords.shape[1])
    cmts = list(comments or [])
    with open(path, "w", encoding="utf-8") as fh:
        for frame_idx in range(int(coords.shape[0])):
            fh.write(f"{int(coords.shape[1])}\n")
            comment = str(cmts[frame_idx]) if frame_idx < len(cmts) else f"MOPAC frame {frame_idx}"
            fh.write(comment + "\n")
            for sym, coord in zip(sy, coords[frame_idx]):
                fh.write(f"{sym:2s} {float(coord[0]): .10f} {float(coord[1]): .10f} {float(coord[2]): .10f}\n")


def _write_mopac_irc_csv(path, profile):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["point", "coordinate_amu05_bohr", "energy_hartree", "relative_energy_kcal_mol", "branch"])
        pts = list((profile or {}).get("point", []) or [])
        rc = list((profile or {}).get("coordinate_amu05_bohr", []) or [])
        en = list((profile or {}).get("energy_hartree", []) or [])
        rel = list((profile or {}).get("relative_energy_kcal_mol", []) or [])
        br = list((profile or {}).get("branch", []) or [])
        for i in range(len(pts)):
            writer.writerow([
                pts[i],
                rc[i] if i < len(rc) else "",
                en[i] if i < len(en) else "",
                rel[i] if i < len(rel) else "",
                br[i] if i < len(br) else "",
            ])


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _run_mopac_reaction_stage(
    stage_name,
    atoms,
    workdir,
    keywords,
    product_atoms=None,
    charge=0,
    multiplicity=1,
    mopac_executable="mopac",
    runner=None,
    timeout_s=3600,
):
    stage_dir = os.path.join(str(workdir), str(stage_name))
    os.makedirs(stage_dir, exist_ok=True)
    mop_path = os.path.join(stage_dir, "job.mop")
    with open(mop_path, "w", encoding="utf-8") as fh:
        fh.write(
            build_mopac_reaction_input_text(
                reactant_atoms=atoms,
                product_atoms=product_atoms,
                keywords=str(keywords),
                total_charge=int(charge),
                multiplicity=int(max(1, multiplicity)),
                title=f"ProtLigInteract {stage_name}",
            )
        )
    run_fn = runner if callable(runner) else subprocess.run
    proc = run_fn(
        [str(mopac_executable), os.path.basename(mop_path)],
        cwd=str(stage_dir),
        capture_output=True,
        text=True,
        timeout=int(max(1, timeout_s)),
    )
    rc = int(getattr(proc, "returncode", -1))
    if rc != 0:
        tail = (str(getattr(proc, "stderr", "") or "") + "\n" + str(getattr(proc, "stdout", "") or ""))[-1500:]
        raise RuntimeError(f"MOPAC {stage_name} failed (rc={rc}): {tail}")
    aux_path = os.path.join(stage_dir, "job.aux")
    if not os.path.isfile(aux_path):
        cand = [f for f in os.listdir(stage_dir) if str(f).lower().endswith(".aux")]
        if cand:
            aux_path = os.path.join(stage_dir, str(cand[0]))
    arc_path = os.path.join(stage_dir, "job.arc")
    if not os.path.isfile(arc_path):
        cand = [f for f in os.listdir(stage_dir) if str(f).lower().endswith(".arc")]
        if cand:
            arc_path = os.path.join(stage_dir, str(cand[0]))
    out_path = os.path.join(stage_dir, "job.out")
    if not os.path.isfile(out_path):
        cand = [f for f in os.listdir(stage_dir) if str(f).lower().endswith(".out")]
        if cand:
            out_path = os.path.join(stage_dir, str(cand[0]))
    xyz_path = os.path.join(stage_dir, "job.xyz")
    if not os.path.isfile(xyz_path):
        cand = [f for f in os.listdir(stage_dir) if str(f).lower().endswith(".xyz")]
        if cand:
            xyz_path = os.path.join(stage_dir, str(cand[0]))
    aux_text = ""
    out_text = ""
    arc_text = ""
    xyz_text = ""
    parsed = {}
    if os.path.isfile(aux_path):
        with open(aux_path, "r", encoding="utf-8", errors="ignore") as fh:
            aux_text = fh.read()
        parsed = parse_mopac_aux_text(aux_text)
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8", errors="ignore") as fh:
            out_text = fh.read()
    if os.path.isfile(arc_path):
        with open(arc_path, "r", encoding="utf-8", errors="ignore") as fh:
            arc_text = fh.read()
    if os.path.isfile(xyz_path):
        with open(xyz_path, "r", encoding="utf-8", errors="ignore") as fh:
            xyz_text = fh.read()
    critical_text = (str(aux_text or "") + "\n" + str(out_text or "")).upper()
    critical_markers = (
        "ERRORS DETECTED IN KEYWORDS",
        "MORE THAN ONE GEOMETRY OPTION",
        "UNRECOGNIZED KEY-WORD",
        "UNRECOGNIZED KEYWORD",
        "KEYWORD COMBINATION",
    )
    if any(marker in critical_text for marker in critical_markers):
        detail = ""
        m = re.search(r"ERROR_MESSAGE\s*=\s*\"([^\"]+)\"", str(aux_text or ""), flags=re.IGNORECASE)
        if m:
            detail = str(m.group(1))
        if not detail:
            for line in str(out_text or "").splitlines():
                up = line.upper()
                if "ERROR" in up or "KEYWORD" in up or "GEOMETRY OPTION" in up:
                    detail = str(line).strip()
                    break
        raise RuntimeError(f"MOPAC {stage_name} keyword/input error: {detail or 'see output file'}")
    error_message = ""
    m_err = re.search(r"ERROR_MESSAGE\s*=\s*\"([^\"]+)\"", str(aux_text or ""), flags=re.IGNORECASE)
    if m_err:
        error_message = str(m_err.group(1)).strip()
    elif "ERROR AND NORMAL TERMINATION MESSAGES" in critical_text:
        for line in str(out_text or "").splitlines():
            up = line.upper()
            if "ERROR" in up or "TOO MANY VARIABLES" in up or "GRADIENT IS TOO LARGE" in up:
                error_message = str(line).strip(" *")
                if error_message:
                    break
    symbols = list(parsed.get("atom_symbols", []) or _mopac_atoms_symbols(atoms))
    arc = parse_mopac_arc_geometries(arc_text, atom_symbols=symbols, expected_atoms=len(symbols)) if arc_text else {}
    xyz_traj = parse_mopac_xyz_trajectory(xyz_text, expected_atoms=len(symbols)) if xyz_text else {}
    return {
        "ok": True,
        "stage": str(stage_name),
        "stage_dir": str(stage_dir),
        "mop_file": str(mop_path),
        "aux_file": str(aux_path) if os.path.isfile(aux_path) else "",
        "arc_file": str(arc_path) if os.path.isfile(arc_path) else "",
        "out_file": str(out_path) if os.path.isfile(out_path) else "",
        "xyz_file": str(xyz_path) if os.path.isfile(xyz_path) else "",
        "stdout": str(getattr(proc, "stdout", "") or ""),
        "stderr": str(getattr(proc, "stderr", "") or ""),
        "keywords": str(keywords),
        "mopac_error_message": str(error_message),
        "parsed": parsed,
        "arc": arc,
        "xyz_trajectory": xyz_traj,
    }


def _mopac_stage_atoms_from_result(stage_result, fallback_atoms):
    parsed = dict((stage_result or {}).get("parsed", {}) or {})
    symbols = list(parsed.get("atom_symbols", []) or [])
    coords = np.asarray(parsed.get("optimized_coords_ang", []), dtype=float)
    if symbols and coords.ndim == 2 and coords.shape[0] == len(symbols):
        return _mopac_atoms_from_symbols_xyz(symbols, coords)
    arc = dict((stage_result or {}).get("arc", {}) or {})
    arc_coords = np.asarray(arc.get("coords_ang", []), dtype=float)
    arc_symbols = list(arc.get("symbols", []) or symbols)
    if arc_coords.ndim == 3 and arc_coords.shape[0] > 0 and arc_symbols:
        return _mopac_atoms_from_symbols_xyz(arc_symbols, arc_coords[-1])
    xyz_traj = dict((stage_result or {}).get("xyz_trajectory", {}) or {})
    xyz_coords = np.asarray(xyz_traj.get("coords_ang", []), dtype=float)
    xyz_symbols = list(xyz_traj.get("symbols", []) or symbols)
    if xyz_coords.ndim == 3 and xyz_coords.shape[0] > 0 and xyz_symbols:
        return _mopac_atoms_from_symbols_xyz(xyz_symbols, xyz_coords[-1])
    return _mopac_normalize_atoms(fallback_atoms)


def run_mopac_reaction_workflow(
    reactant_atoms,
    product_atoms=None,
    ts_guess_atoms=None,
    operation="mopac_reaction_path",
    method="PM7",
    base_keywords="",
    charge=0,
    multiplicity=1,
    mopac_executable="mopac",
    workdir="",
    runner=None,
    timeout_s=3600,
    use_forcets=False,
    irc_direction="both",
    irc_x_priority_ang=0.05,
    irc_max_points=25,
    frequency_threshold_cm=10.0,
):
    """
    Run a MOPAC reaction-search workflow.

    The user-facing names QST2/QST3 are mapped onto MOPAC-native operations:
    SADDLE uses reactant/product endpoints (QST2-like), TS refines a supplied
    transition-state guess (QST3-like), FORCE/FORCETS validates the Hessian, and
    IRC follows the validated first-order saddle point.
    """
    op = qmmm_mopac_normalize_operation(operation)
    if op not in {"mopac_qst2", "mopac_qst3", "mopac_ts_frequency", "mopac_irc", "mopac_reaction_path"}:
        op = "mopac_reaction_path"
    reactant = _mopac_normalize_atoms(reactant_atoms)
    product = _mopac_normalize_atoms(product_atoms) if product_atoms is not None else None
    ts_guess = _mopac_normalize_atoms(ts_guess_atoms) if ts_guess_atoms is not None else None
    if product is not None:
        reactant, product = _mopac_validate_atom_compatibility(reactant, product, "reactant", "product")
    if ts_guess is not None:
        reactant, ts_guess = _mopac_validate_atom_compatibility(reactant, ts_guess, "reactant", "TS guess")
    if op in {"mopac_qst2", "mopac_reaction_path"} and product is None:
        raise ValueError("QST2/SADDLE workflow requires product_atoms with the same atom order as reactant_atoms.")
    if op == "mopac_qst3" and ts_guess is None:
        raise ValueError("QST3-style workflow requires ts_guess_atoms.")
    meth = _mopac_extract_method_from_keywords(base_keywords, default=method)
    run_dir = str(workdir or "").strip() or tempfile.mkdtemp(prefix="mopac_reaction_")
    os.makedirs(run_dir, exist_ok=True)
    result = {
        "ok": False,
        "operation": str(op),
        "method": str(meth),
        "workdir": str(run_dir),
        "stages": [],
        "warnings": [],
        "notes": [],
        "ts_validation": {},
        "irc_profile": {},
    }

    try:
        current = list(ts_guess or reactant)
        if op in {"mopac_qst2", "mopac_reaction_path"}:
            kw = mopac_reaction_keywords_for_stage("qst2", method=meth, base_keywords=base_keywords)
            stage = _run_mopac_reaction_stage(
                "01_qst2_saddle",
                atoms=reactant,
                product_atoms=product,
                workdir=run_dir,
                keywords=kw,
                charge=charge,
                multiplicity=multiplicity,
                mopac_executable=mopac_executable,
                runner=runner,
                timeout_s=timeout_s,
            )
            result["stages"].append(stage)
            if str(stage.get("mopac_error_message", "") or "").strip():
                raise RuntimeError(f"MOPAC SADDLE/QST2 failed: {stage.get('mopac_error_message')}")
            current = _mopac_stage_atoms_from_result(stage, fallback_atoms=_mopac_midpoint_atoms(reactant, product))
        elif op == "mopac_qst3":
            current = list(ts_guess)
        elif op in {"mopac_ts_frequency", "mopac_irc"} and ts_guess is not None:
            current = list(ts_guess)

        run_ts_refine = op in {"mopac_qst2", "mopac_qst3", "mopac_reaction_path"}
        if run_ts_refine:
            kw = mopac_reaction_keywords_for_stage("ts", method=meth, base_keywords=base_keywords)
            stage = _run_mopac_reaction_stage(
                "02_ts_refine",
                atoms=current,
                workdir=run_dir,
                keywords=kw,
                charge=charge,
                multiplicity=multiplicity,
                mopac_executable=mopac_executable,
                runner=runner,
                timeout_s=timeout_s,
            )
            result["stages"].append(stage)
            if str(stage.get("mopac_error_message", "") or "").strip():
                raise RuntimeError(f"MOPAC TS refinement failed: {stage.get('mopac_error_message')}")
            current = _mopac_stage_atoms_from_result(stage, fallback_atoms=current)

        run_frequency = op in {"mopac_qst2", "mopac_qst3", "mopac_ts_frequency", "mopac_irc", "mopac_reaction_path"}
        freq_stage = None
        if run_frequency:
            kw = mopac_reaction_keywords_for_stage(
                "frequency",
                method=meth,
                base_keywords=base_keywords,
                use_forcets=bool(use_forcets),
            )
            freq_stage = _run_mopac_reaction_stage(
                "03_ts_frequency",
                atoms=current,
                workdir=run_dir,
                keywords=kw,
                charge=charge,
                multiplicity=multiplicity,
                mopac_executable=mopac_executable,
                runner=runner,
                timeout_s=timeout_s,
            )
            result["stages"].append(freq_stage)
            parsed = dict(freq_stage.get("parsed", {}) or {})
            validation = mopac_validate_ts_frequencies(
                freq_cm_signed=parsed.get("freq_cm_signed", []),
                freq_cm_imag=parsed.get("freq_cm_imag", []),
                threshold_cm=float(frequency_threshold_cm),
            )
            result["ts_validation"] = dict(validation)
            if not bool(validation.get("ok", False)):
                result["warnings"].append(
                    "TS frequency validation did not find exactly one imaginary mode; IRC was not launched."
                )

        run_irc = op in {"mopac_irc", "mopac_reaction_path"}
        if run_irc and bool((result.get("ts_validation") or {}).get("ok", False)):
            kw = mopac_reaction_keywords_for_stage(
                "irc",
                method=meth,
                base_keywords=base_keywords,
                irc_direction=irc_direction,
                x_priority_ang=float(irc_x_priority_ang),
            )
            irc_stage = _run_mopac_reaction_stage(
                "04_irc",
                atoms=current,
                workdir=run_dir,
                keywords=kw,
                charge=charge,
                multiplicity=multiplicity,
                mopac_executable=mopac_executable,
                runner=runner,
                timeout_s=timeout_s,
            )
            result["stages"].append(irc_stage)
            arc = dict(irc_stage.get("arc", {}) or {})
            coords = np.asarray(arc.get("coords_ang", []), dtype=float)
            energies = np.asarray(arc.get("energy_hartree", []), dtype=float)
            symbols = list(arc.get("symbols", []) or _mopac_atoms_symbols(current))
            if coords.ndim != 3 or coords.shape[0] <= 0:
                xyz_traj = dict(irc_stage.get("xyz_trajectory", {}) or {})
                coords = np.asarray(xyz_traj.get("coords_ang", []), dtype=float)
                energies = np.asarray(xyz_traj.get("energy_hartree", []), dtype=float)
                symbols = list(xyz_traj.get("symbols", []) or symbols)
            if coords.ndim == 3 and coords.shape[0] > 0:
                keep = _mopac_downsample_indices(int(coords.shape[0]), int(irc_max_points))
                coords_keep = coords[keep, :, :]
                energies_keep = energies[keep] if energies.size >= coords.shape[0] else energies
                profile = mopac_build_irc_profile_payload(coords_keep, energies_keep, symbols)
                result["irc_profile"] = dict(profile)
                xyz_path = os.path.join(run_dir, "mopac_irc_path.xyz")
                csv_path = os.path.join(run_dir, "mopac_irc_profile.csv")
                _write_mopac_xyz_frames(
                    xyz_path,
                    symbols,
                    coords_keep,
                    comments=[f"MOPAC IRC point {i}" for i in range(int(coords_keep.shape[0]))],
                )
                _write_mopac_irc_csv(csv_path, profile)
                result["irc_xyz"] = str(xyz_path)
                result["irc_csv"] = str(csv_path)
            else:
                result["warnings"].append("MOPAC IRC finished but no ARC/XYZ geometry frames were recovered.")
        elif run_irc:
            result["notes"].append("IRC skipped because TS frequency validation failed or was unavailable.")

        symbols = _mopac_atoms_symbols(current)
        coords = _mopac_atoms_xyz(current)
        ts_xyz = os.path.join(run_dir, "mopac_ts_candidate.xyz")
        _write_mopac_xyz_frames(ts_xyz, symbols, coords, comments=["MOPAC TS candidate"])
        result["ts_xyz"] = str(ts_xyz)
        result["ts_atoms"] = list(current)
        result["ok"] = bool(
            (not run_irc or bool((result.get("irc_profile") or {}).get("ok", False)))
            and (not run_frequency or bool((result.get("ts_validation") or {}).get("ok", False)))
        )
    except Exception as exc:
        result["error"] = str(exc)
        result["ok"] = False

    summary_path = os.path.join(run_dir, "mopac_reaction_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(result), fh, indent=2)
        result["summary_json"] = str(summary_path)
    except Exception as exc:
        result.setdefault("warnings", []).append(f"Summary JSON write failed: {exc}")
    return result


def qmmm_step_file_based(
    xyz_full_ang,
    atomic_numbers_full,
    qm_indices,
    mm_indices,
    q_mm,
    mopac_executable="mopac",
    mopac_keywords="PM7 1SCF GRAD AUX QMMM",
    qm_total_charge=0,
    qm_multiplicity=1,
    workdir="",
    dielectric=1.0,
    runner=None,
    timeout_s=120,
):
    """
    Single QM/MM step with file-based MOPAC execution.
    Returns QM energy/forces and electrostatic coupling terms.
    """
    xyz = _as_xyz_array(xyz_full_ang)
    zfull = [int(v) for v in list(np.asarray(atomic_numbers_full).reshape(-1))]
    if len(zfull) != int(xyz.shape[0]):
        raise ValueError("atomic_numbers_full and xyz_full size mismatch.")
    qidx = [int(i) for i in list(qm_indices or [])]
    midx = [int(i) for i in list(mm_indices or [])]
    if not qidx:
        raise ValueError("qm_indices cannot be empty.")
    if not midx:
        raise ValueError("mm_indices cannot be empty.")
    if set(qidx).intersection(set(midx)):
        raise ValueError("qm_indices and mm_indices must be disjoint.")
    xyz_qm = xyz[qidx, :]
    xyz_mm = xyz[midx, :]
    q_mm_arr = _as_1d(q_mm, name="q_mm")
    if int(q_mm_arr.size) != int(len(midx)):
        raise ValueError("q_mm length must match mm_indices.")
    z_qm = [int(zfull[i]) for i in qidx]
    # The high-level planner chooses operation-specific controls (1SCF, EF,
    # FORCE/THERMO, TS).  This low-level executor only enforces the keywords
    # required to produce structured electrostatic-embedding files.
    kw_tokens = _split_mopac_keywords(str(mopac_keywords or "PM7 1SCF GRAD AUX QMMM"))
    kw_names = {_mopac_keyword_name(tok) for tok in kw_tokens}
    for required in ("AUX", "QMMM"):
        if required not in kw_names:
            kw_tokens.append(required)
            kw_names.add(required)
    mopac_keywords = " ".join(kw_tokens)
    phi = compute_phi_on_qm_atoms(
        xyz_qm_ang=xyz_qm,
        xyz_mm_ang=xyz_mm,
        q_mm=q_mm_arr,
        dielectric=dielectric,
    )
    own_tmp = False
    run_dir = str(workdir or "").strip()
    if not run_dir:
        run_dir = tempfile.mkdtemp(prefix="qmmm_mopac_")
        own_tmp = True
    os.makedirs(run_dir, exist_ok=True)
    mop_file = os.path.join(run_dir, "step.mop")
    mol_in_file = os.path.join(run_dir, "mol.in")
    with open(mop_file, "w", encoding="utf-8") as fh:
        fh.write(
            build_mopac_input_text(
                qm_atomic_numbers=z_qm,
                xyz_qm_ang=xyz_qm,
                keywords=mopac_keywords,
                total_charge=qm_total_charge,
                multiplicity=qm_multiplicity,
                title="ProtLigInteract QM/MM file-based step",
            )
        )
    with open(mol_in_file, "w", encoding="utf-8") as fh:
        fh.write(build_mopac_mol_in_text(phi))
    run_fn = runner if callable(runner) else subprocess.run
    cmdline = [str(mopac_executable), os.path.basename(mop_file)]
    cp = run_fn(
        cmdline,
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=max(1, int(timeout_s)),
    )
    rc = int(getattr(cp, "returncode", -1))
    if rc != 0:
        raise RuntimeError(
            f"MOPAC execution failed (rc={rc}): "
            f"{str(getattr(cp, 'stderr', '') or '')[:500]}"
        )
    aux_file = os.path.join(run_dir, "step.aux")
    if not os.path.isfile(aux_file):
        # fallback to first .aux
        aux_candidates = [f for f in os.listdir(run_dir) if str(f).lower().endswith(".aux")]
        if aux_candidates:
            aux_file = os.path.join(run_dir, str(aux_candidates[0]))
    if not os.path.isfile(aux_file):
        raise RuntimeError("MOPAC run finished but no .aux file was produced.")
    with open(aux_file, "r", encoding="utf-8", errors="ignore") as fh:
        aux_txt = fh.read()
    parsed = parse_mopac_aux_text(aux_txt)
    if not bool(parsed.get("ok", False)):
        raise RuntimeError("Could not parse required TOTAL_ENERGY/ATOM_CHARGES/GRADIENTS from MOPAC .aux.")
    q_qm = np.asarray(parsed.get("atom_charges", []), dtype=float).reshape(-1)
    if int(q_qm.size) != int(len(qidx)):
        raise RuntimeError(f"ATOM_CHARGES size mismatch: expected {len(qidx)}, got {q_qm.size}")
    g_qm_kj_nm = np.asarray(parsed.get("gradients_kj_mol_nm", []), dtype=float).reshape((-1, 3))
    if int(g_qm_kj_nm.shape[0]) != int(len(qidx)):
        raise RuntimeError(f"GRADIENTS size mismatch: expected {len(qidx)} rows, got {g_qm_kj_nm.shape[0]}")
    f_qm_internal = -g_qm_kj_nm
    electro = electrostatic_backreaction_openmm(
        xyz_qm_ang=xyz_qm,
        xyz_mm_ang=xyz_mm,
        q_qm=q_qm,
        q_mm=q_mm_arr,
        dielectric=dielectric,
    )
    result = {
        "ok": True,
        "workdir": str(run_dir),
        "mop_file": str(mop_file),
        "mol_in_file": str(mol_in_file),
        "aux_file": str(aux_file),
        "stdout": str(getattr(cp, "stdout", "") or ""),
        "stderr": str(getattr(cp, "stderr", "") or ""),
        "qm_indices": list(qidx),
        "mm_indices": list(midx),
        "phi_qm_au": np.asarray(phi, dtype=float),
        "q_qm": np.asarray(q_qm, dtype=float),
        "mopac_keywords": str(mopac_keywords),
        "energy_qm_hartree": float(parsed.get("energy_hartree")),
        "heat_of_formation_ev": parsed.get("heat_of_formation_ev"),
        "thermochemistry": dict(parsed.get("thermochemistry", {}) or {}),
        "forces_qm_internal_kj_mol_nm": np.asarray(f_qm_internal, dtype=float),
        "energy_qmmm_elec_kj_mol": float(electro.get("energy_kj_mol", 0.0)),
        "forces_qmmm_elec_qm_kj_mol_nm": np.asarray(electro.get("forces_qm_kj_mol_nm"), dtype=float),
        "forces_qmmm_elec_mm_kj_mol_nm": np.asarray(electro.get("forces_mm_kj_mol_nm"), dtype=float),
    }
    if own_tmp:
        result["temporary_workdir"] = True
    return result


__all__ = [
    "BOHR_TO_ANGSTROM",
    "ANGSTROM_TO_BOHR",
    "EV_TO_HARTREE",
    "KCAL_TO_HARTREE",
    "OPENMM_COULOMB_CONSTANT",
    "QMMM_MOPAC_OPERATION_PROFILES",
    "qmmm_mopac_normalize_operation",
    "qmmm_mopac_operation_profile",
    "qmmm_mopac_keywords_for_operation",
    "qmmm_recommend_region",
    "qmmm_assess_interface",
    "compute_phi_on_qm_atoms",
    "electrostatic_backreaction_openmm",
    "assemble_force_correction",
    "build_mopac_input_text",
    "build_mopac_mol_in_text",
    "build_mopac_reaction_input_text",
    "parse_mopac_aux_text",
    "parse_mopac_arc_geometries",
    "parse_mopac_xyz_trajectory",
    "mopac_validate_ts_frequencies",
    "mopac_reaction_keywords_for_stage",
    "mopac_build_irc_profile_payload",
    "run_mopac_reaction_workflow",
    "qmmm_step_file_based",
]
