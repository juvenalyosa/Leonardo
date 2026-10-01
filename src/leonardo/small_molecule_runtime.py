"""Shared RDKit helpers for small-molecule preparation."""

from __future__ import annotations

import math
import os
from typing import Any, Callable


def _clone_and_sanitize_mol(mol):
    from rdkit import Chem

    if mol is None:
        raise ValueError("molecule is required")
    work = Chem.Mol(mol)
    Chem.SanitizeMol(work)
    return work


def small_molecule_explicit_hydrogen_count(mol) -> int:
    from rdkit import Chem

    if mol is None:
        return 0
    try:
        return int(sum(1 for atom in Chem.Mol(mol).GetAtoms() if int(atom.GetAtomicNum()) == 1))
    except Exception:
        return 0


def small_molecule_formal_charge(mol) -> int:
    from rdkit import Chem

    if mol is None:
        return 0
    try:
        return int(sum(int(atom.GetFormalCharge()) for atom in Chem.Mol(mol).GetAtoms()))
    except Exception:
        return 0


def summarize_small_molecule_state(mol) -> dict[str, Any]:
    if mol is None:
        return {
            "ok": False,
            "atoms": 0,
            "heavy_atoms": 0,
            "explicit_hydrogens": 0,
            "formal_charge": 0,
        }
    try:
        work = _clone_and_sanitize_mol(mol)
    except Exception:
        return {
            "ok": False,
            "atoms": 0,
            "heavy_atoms": 0,
            "explicit_hydrogens": 0,
            "formal_charge": 0,
        }
    return {
        "ok": True,
        "atoms": int(work.GetNumAtoms()),
        "heavy_atoms": int(work.GetNumHeavyAtoms()),
        "explicit_hydrogens": int(small_molecule_explicit_hydrogen_count(work)),
        "formal_charge": int(small_molecule_formal_charge(work)),
    }


def summarize_pdbqt_ligand_file(path: str) -> dict[str, Any]:
    """
    Read enough of a ligand PDBQT to audit docking-critical properties.

    PDBQT parsing is deliberately conservative here: the function checks the first
    ligand model, atom records, partial-charge tokens, atom types, and TORSDOF.
    It does not try to rebuild chemistry from coordinates.
    """

    p = str(path or "").strip()
    out: dict[str, Any] = {
        "ok": False,
        "path": p,
        "atoms": 0,
        "heavy_atoms": 0,
        "hydrogens": 0,
        "charge_sum": 0.0,
        "has_partial_charges": False,
        "atom_types": {},
        "unknown_atom_types": [],
        "missing_atom_types": 0,
        "torsdof": None,
    }
    if (not p) or (not os.path.isfile(p)):
        out["error"] = "pdbqt_missing"
        return out

    allowed_types = {
        "C",
        "A",
        "N",
        "NA",
        "NS",
        "OA",
        "OS",
        "S",
        "SA",
        "P",
        "HD",
        "H",
        "F",
        "Cl",
        "CL",
        "Br",
        "BR",
        "I",
        "Mg",
        "MG",
        "Mn",
        "MN",
        "Zn",
        "ZN",
        "Ca",
        "CA",
        "Fe",
        "FE",
        "Cu",
        "CU",
        "Ni",
        "NI",
        "Co",
        "CO",
        "Na",
        "K",
    }
    atom_types: dict[str, int] = {}
    unknown: set[str] = set()
    charge_sum = 0.0
    charges_seen = 0
    missing_types = 0
    in_first_model = False
    saw_model = False
    finished_first_model = False

    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.startswith("MODEL"):
                    if saw_model:
                        finished_first_model = True
                        break
                    saw_model = True
                    in_first_model = True
                    continue
                if line.startswith("ENDMDL") and saw_model:
                    finished_first_model = True
                    break
                if finished_first_model:
                    break
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    if line.startswith("TORSDOF"):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                out["torsdof"] = int(float(parts[1]))
                            except Exception:
                                out["torsdof"] = None
                    continue
                if saw_model and not in_first_model:
                    continue
                parts = line.split()
                atype = ""
                charge = None
                if len(parts) >= 12:
                    atype = str(parts[-1]).strip()
                    try:
                        charge = float(parts[-2])
                    except Exception:
                        charge = None
                else:
                    atype = str(line[77:79] if len(line) >= 79 else "").strip()
                    try:
                        charge_txt = str(line[70:76] if len(line) >= 76 else "").strip()
                        charge = float(charge_txt) if charge_txt else None
                    except Exception:
                        charge = None
                elem = _pdbqt_atom_type_to_element(atype, atom_name=(line[12:16] if len(line) >= 16 else ""))
                out["atoms"] = int(out["atoms"]) + 1
                if str(elem).upper() == "H":
                    out["hydrogens"] = int(out["hydrogens"]) + 1
                else:
                    out["heavy_atoms"] = int(out["heavy_atoms"]) + 1
                if charge is not None and math.isfinite(float(charge)):
                    charge_sum += float(charge)
                    charges_seen += 1
                if atype:
                    atom_types[atype] = int(atom_types.get(atype, 0)) + 1
                    if atype not in allowed_types:
                        unknown.add(atype)
                else:
                    missing_types += 1
    except Exception as exc:
        out["error"] = str(exc)
        return out

    out["charge_sum"] = float(charge_sum)
    out["has_partial_charges"] = bool(charges_seen > 0 and charges_seen == int(out["atoms"]))
    out["atom_types"] = dict(sorted(atom_types.items()))
    out["unknown_atom_types"] = sorted(unknown)
    out["missing_atom_types"] = int(missing_types)
    out["ok"] = int(out["atoms"]) > 0
    if not bool(out["ok"]):
        out["error"] = "no_atoms"
    return out


def _pdbqt_atom_type_to_element(atom_type: str, atom_name: str = "") -> str:
    at = str(atom_type or "").strip()
    lut = {
        "C": "C",
        "A": "C",
        "N": "N",
        "NA": "N",
        "NS": "N",
        "O": "O",
        "OA": "O",
        "OS": "O",
        "S": "S",
        "SA": "S",
        "P": "P",
        "HD": "H",
        "H": "H",
        "F": "F",
        "Cl": "Cl",
        "CL": "Cl",
        "Br": "Br",
        "BR": "Br",
        "I": "I",
        "Mg": "Mg",
        "MG": "Mg",
        "Mn": "Mn",
        "MN": "Mn",
        "Zn": "Zn",
        "ZN": "Zn",
        "Ca": "Ca",
        "CA": "Ca",
        "Fe": "Fe",
        "FE": "Fe",
        "Cu": "Cu",
        "CU": "Cu",
        "Ni": "Ni",
        "NI": "Ni",
        "Co": "Co",
        "CO": "Co",
        "Na": "Na",
        "K": "K",
    }
    if at in lut:
        return lut[at]
    name = str(atom_name or "").strip()
    letters = "".join(ch for ch in name if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2].upper() in {"CL", "BR", "MG", "MN", "ZN", "CA", "FE", "CU", "NI", "CO", "NA"}:
        return letters[0].upper() + letters[1].lower()
    return letters[0].upper()


def build_small_molecule_prep_qc(
    *,
    input_mol=None,
    prepared_mol=None,
    input_file: str = "",
    prepared_file: str = "",
    prep_info: dict[str, Any] | None = None,
    target_ph: float = 7.4,
    preserve_input_charges: bool = False,
    smiles_hint: str = "",
    expected_heavy_atoms: int | None = None,
    charge_tolerance: float = 0.65,
) -> dict[str, Any]:
    """
    Build a red/yellow/green preparation QC report for docking ligands.

    RED means the prepared ligand should not be docked. YELLOW means docking can
    proceed, but the result should be interpreted with the listed caveats.
    """

    info = dict(prep_info or {})
    warnings: list[str] = []
    errors: list[str] = []
    notes: list[str] = []
    input_summary = summarize_small_molecule_state(input_mol) if input_mol is not None else {"ok": False}
    prepared_summary = summarize_small_molecule_state(prepared_mol) if prepared_mol is not None else {"ok": False}
    pdbqt_summary = {}

    in_file = str(input_file or "").strip()
    out_file = str(prepared_file or "").strip()
    if out_file and out_file.lower().endswith(".pdbqt"):
        pdbqt_summary = summarize_pdbqt_ligand_file(out_file)
        if not bool(pdbqt_summary.get("ok", False)):
            errors.append("prepared PDBQT has no readable ligand atoms")
        if int(pdbqt_summary.get("missing_atom_types", 0) or 0) > 0:
            errors.append("prepared PDBQT has atoms without AutoDock atom types")
        unknown_types = list(pdbqt_summary.get("unknown_atom_types", []) or [])
        if unknown_types:
            errors.append("prepared PDBQT has unsupported atom types: " + ",".join(str(x) for x in unknown_types[:8]))
        if not bool(pdbqt_summary.get("has_partial_charges", False)):
            warnings.append("prepared PDBQT does not expose a partial charge on every atom")
    elif out_file and os.path.isfile(out_file):
        if not bool(prepared_summary.get("ok", False)):
            warnings.append("prepared ligand is not PDBQT and could not be audited with RDKit")
    else:
        errors.append("prepared ligand output is missing")

    expected_heavy = 0
    try:
        expected_heavy = int(expected_heavy_atoms or 0)
    except Exception:
        expected_heavy = 0
    if expected_heavy <= 0 and bool(input_summary.get("ok", False)):
        expected_heavy = int(input_summary.get("heavy_atoms", 0) or 0)
    observed_heavy = 0
    if pdbqt_summary:
        observed_heavy = int(pdbqt_summary.get("heavy_atoms", 0) or 0)
    elif bool(prepared_summary.get("ok", False)):
        observed_heavy = int(prepared_summary.get("heavy_atoms", 0) or 0)
    if expected_heavy > 0 and observed_heavy > 0 and int(expected_heavy) != int(observed_heavy):
        errors.append(f"heavy atom count changed during ligand preparation ({expected_heavy} -> {observed_heavy})")

    formal_charge = None
    if bool(input_summary.get("ok", False)):
        formal_charge = int(input_summary.get("formal_charge", 0) or 0)
        if int(input_summary.get("explicit_hydrogens", 0) or 0) <= 0:
            warnings.append("input ligand had no explicit hydrogens before preparation")
    elif in_file and not in_file.lower().endswith(".pdbqt"):
        warnings.append("input ligand could not be parsed by RDKit for charge/topology QC")

    if bool(prepared_summary.get("ok", False)):
        # the pH step may change the net charge (neutral amidine -> amidinium)
        formal_charge = int(prepared_summary.get("formal_charge", 0) or 0)
    if pdbqt_summary and formal_charge is not None:
        qsum = float(pdbqt_summary.get("charge_sum", 0.0) or 0.0)
        q_delta = abs(qsum - float(formal_charge))
        if q_delta > max(1.25, float(charge_tolerance) * 2.0):
            errors.append(f"PDBQT charge sum differs strongly from formal charge ({qsum:.2f} vs {formal_charge:+d})")
        elif q_delta > float(charge_tolerance):
            warnings.append(f"PDBQT charge sum differs from formal charge ({qsum:.2f} vs {formal_charge:+d})")

    if bool(info.get("fragment_selected", False)):
        notes.append("main fragment selected before docking prep")
    if bool(info.get("used_repair", False)):
        notes.append("chemistry-aware protonation/repair was used")
    if bool(info.get("hydrogens_added", False)):
        notes.append("explicit hydrogens were added")
    if bool(preserve_input_charges):
        notes.append("input charges were preserved by policy")
    if smiles_hint:
        notes.append("SMILES/bond-order hint was available")

    status = "green"
    if errors:
        status = "red"
    elif warnings:
        status = "yellow"
    return {
        "ok": status != "red",
        "status": status,
        "target_ph": float(target_ph),
        "preserve_input_charges": bool(preserve_input_charges),
        "input_file": str(in_file),
        "prepared_file": str(out_file),
        "input_summary": dict(input_summary or {}),
        "prepared_summary": dict(prepared_summary or {}),
        "pdbqt_summary": dict(pdbqt_summary or {}),
        "warnings": list(warnings),
        "errors": list(errors),
        "notes": list(notes),
    }


def prepare_small_molecule_working_mol(
    mol,
    include_h: bool = True,
    target_ph: float = 7.4,
    repair_fn: Callable[..., Any] | None = None,
    select_fragment_fn: Callable[..., Any] | None = None,
    reprotonate: bool = False,
):
    """
    Build a sanitized working molecule for small-molecule editing/preparation.

    When hydrogens are requested and the source currently has no explicit hydrogens,
    an optional `repair_fn` can provide a chemistry-aware protonated model before
    falling back to plain RDKit `AddHs`. With ``reprotonate`` the repair also runs on
    a source that already has hydrogens (docking at a chosen pH: a neutral benzamidine
    SDF must become the amidinium).
    """

    from rdkit import Chem

    if mol is None:
        raise ValueError("molecule is required")
    info: dict[str, Any] = {
        "include_h": bool(include_h),
        "target_ph": float(target_ph),
        "used_repair": False,
        "repair_error": "",
        "fragment_selected": False,
        "hydrogens_added": False,
    }
    work = _clone_and_sanitize_mol(mol)
    if callable(select_fragment_fn):
        try:
            picked = select_fragment_fn(Chem.Mol(work))
            if picked is not None:
                work = _clone_and_sanitize_mol(picked)
                info["fragment_selected"] = True
        except Exception:
            pass
    info["input_hydrogens"] = int(small_molecule_explicit_hydrogen_count(work))
    if include_h:
        if (info["input_hydrogens"] <= 0 or bool(reprotonate)) and callable(repair_fn):
            try:
                repaired = repair_fn(Chem.Mol(work), target_ph=float(target_ph))
                if repaired is not None:
                    work = _clone_and_sanitize_mol(repaired)
                    info["used_repair"] = True
            except TypeError:
                try:
                    repaired = repair_fn(Chem.Mol(work), float(target_ph))
                    if repaired is not None:
                        work = _clone_and_sanitize_mol(repaired)
                        info["used_repair"] = True
                except Exception as exc:
                    info["repair_error"] = str(exc)
            except Exception as exc:
                info["repair_error"] = str(exc)
        if int(small_molecule_explicit_hydrogen_count(work)) <= 0:
            work = Chem.AddHs(Chem.Mol(work), addCoords=True)
            work = _clone_and_sanitize_mol(work)
            info["hydrogens_added"] = True
    else:
        work = Chem.RemoveHs(Chem.Mol(work))
        work = _clone_and_sanitize_mol(work)
        if callable(select_fragment_fn):
            try:
                picked = select_fragment_fn(Chem.Mol(work))
                if picked is not None:
                    work = _clone_and_sanitize_mol(picked)
                    info["fragment_selected"] = True
            except Exception:
                pass
    info.update(summarize_small_molecule_state(work))
    return work, info
