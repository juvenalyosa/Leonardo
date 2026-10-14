import json
import os
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np

try:
    from .qmmm_mopac_runtime import (
        assemble_force_correction,
        qmmm_assess_interface,
        qmmm_mopac_keywords_for_operation,
        qmmm_recommend_region,
        qmmm_step_file_based,
    )
except Exception:
    from .qmmm_mopac_runtime import (
        assemble_force_correction,
        qmmm_assess_interface,
        qmmm_mopac_keywords_for_operation,
        qmmm_recommend_region,
        qmmm_step_file_based,
    )


_AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "ASH",
    "ACE",
    "CYS",
    "CYM",
    "CYX",
    "GLN",
    "GLU",
    "GLH",
    "GLY",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "HSD",
    "HSE",
    "HSP",
    "ILE",
    "LEU",
    "LYS",
    "LYN",
    "MET",
    "MSE",
    "NME",
    "PHE",
    "PRO",
    "PYL",
    "SEC",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}

_NA3 = {
    "A",
    "C",
    "G",
    "T",
    "U",
    "DA",
    "DC",
    "DG",
    "DT",
    "DU",
    "ADE",
    "CYT",
    "GUA",
    "THY",
    "URI",
}

_WATER = {"HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE"}
_IONS = {"NA", "K", "CL", "CA", "MG", "ZN", "FE", "MN", "CU", "CO", "NI"}
_OPENMM_STANDARD_SMALL_MOLECULE_RESN = {
    # peptide caps: part of the protein force fields, never ligands to parameterize
    "ACE", "NME", "NMA", "NH2", "NHE",
    "HEM", "HEC", "HEA", "HEB", "HEO", "HEV",
    "NAD", "NAI", "NDP", "NAP", "NADH", "NADP",
    "FAD", "FMN", "COA", "SAM", "SAH", "PLP", "THF", "MTH",
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "CTP", "CDP", "CMP", "UTP", "UDP", "UMP",
    "TTP", "TDP", "TMP",
    "SO4", "PO4", "HPO", "PI", "CO3",
    "BMA", "NAG", "MAN", "GAL", "GLC",
    "DMS", "GOL", "EDO", "PEG",
}
_BACKBONE_HEAVY_ATOM_NAMES = {
    "N",
    "CA",
    "C",
    "O",
    "OXT",
    "P",
    "OP1",
    "OP2",
    "OP3",
    "O3'",
    "O5'",
    "C3'",
    "C4'",
    "C5'",
}


def _as_xyz_array(xyz):
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Coordinates must have shape (N,3).")
    return arr


def _as_1d_float(arr, name="array"):
    out = np.asarray(arr, dtype=float).reshape(-1)
    if out.size <= 0:
        raise ValueError(f"{name} is empty.")
    return out


def _safe_float(value, default=float("nan")):
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _as_indices(indices, n_atoms, name):
    out = []
    seen = set()
    for v in list(indices or []):
        try:
            idx = int(v)
        except Exception:
            continue
        if idx < 0 or idx >= int(n_atoms):
            raise ValueError(f"{name} contains out-of-range atom index: {idx}")
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def _normalize_residue_key(chain, resi, resn):
    return (str(chain or "").strip(), str(resi or "").strip(), str(resn or "").strip().upper())


def _normalize_residue_key_list(keys):
    out = []
    seen = set()
    for item in list(keys or []):
        if isinstance(item, dict):
            key = _normalize_residue_key(item.get("chain", ""), item.get("resi", ""), item.get("resn", ""))
        elif isinstance(item, (list, tuple)) and len(item) >= 3:
            key = _normalize_residue_key(item[0], item[1], item[2])
        else:
            txt = str(item or "").strip()
            if not txt:
                continue
            parts = [p.strip() for p in txt.replace("|", ":").split(":")]
            if len(parts) >= 3:
                key = _normalize_residue_key(parts[0], parts[1], parts[2])
            else:
                continue
        if not key[2]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _normalize_atom_descriptor(item):
    if isinstance(item, dict):
        out = {
            "chain": str(item.get("chain", "") or "").strip(),
            "resi": str(item.get("resi", "") or "").strip(),
            "resn": str(item.get("resn", "") or "").strip().upper(),
            "name": str(item.get("name", "") or "").strip(),
        }
    elif isinstance(item, (list, tuple)) and len(item) >= 4:
        out = {
            "chain": str(item[0] or "").strip(),
            "resi": str(item[1] or "").strip(),
            "resn": str(item[2] or "").strip().upper(),
            "name": str(item[3] or "").strip(),
        }
    else:
        txt = str(item or "").strip()
        if not txt:
            return {}
        parts = [p.strip() for p in txt.replace("|", ":").split(":")]
        if len(parts) < 4:
            return {}
        out = {
            "chain": str(parts[0] or "").strip(),
            "resi": str(parts[1] or "").strip(),
            "resn": str(parts[2] or "").strip().upper(),
            "name": str(parts[3] or "").strip(),
        }
    if not out.get("name"):
        return {}
    return out


def _normalize_atom_descriptor_list(items):
    out = []
    seen = set()
    for item in list(items or []):
        desc = _normalize_atom_descriptor(item)
        if not desc:
            continue
        key = (
            str(desc.get("chain", "") or "").strip(),
            str(desc.get("resi", "") or "").strip(),
            str(desc.get("resn", "") or "").strip().upper(),
            str(desc.get("name", "") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(desc))
    return out


def _resolve_atom_descriptors(atom_rows, descriptors, label):
    idxs = []
    missing = []
    seen = set()
    for desc in list(_normalize_atom_descriptor_list(descriptors) or []):
        matched = None
        d_chain = str(desc.get("chain", "") or "").strip()
        d_resi = str(desc.get("resi", "") or "").strip()
        d_resn = str(desc.get("resn", "") or "").strip().upper()
        d_name = str(desc.get("name", "") or "").strip()
        for row in list(atom_rows or []):
            if d_chain and str(row.get("chain", "") or "").strip() != d_chain:
                continue
            if d_resi and str(row.get("resi", "") or "").strip() != d_resi:
                continue
            if d_resn and str(row.get("resn", "") or "").strip().upper() != d_resn:
                continue
            if d_name and str(row.get("name", "") or "").strip() != d_name:
                continue
            try:
                matched = int(row.get("index"))
            except Exception:
                matched = None
            if matched is not None:
                break
        if matched is None:
            missing.append(
                f"{d_name}:{d_resn}:{d_chain}:{d_resi}".strip(":")
            )
            continue
        if matched in seen:
            continue
        seen.add(matched)
        idxs.append(int(matched))
    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview += ", ..."
        raise ValueError(f"No atoms matched {label}: {preview}")
    return idxs


def classify_residue_kind(resn):
    rn = str(resn or "").strip().upper()
    if rn in _WATER:
        return "water"
    if rn in _IONS:
        return "ion"
    if rn in _AA3 or rn in _NA3:
        return "biopolymer"
    return "hetero"


def is_standard_openmm_small_molecule_resn(resn):
    rn = str(resn or "").strip().upper()
    if not rn:
        return False
    return bool(rn in _OPENMM_STANDARD_SMALL_MOLECULE_RESN)


def _guess_atom_element_symbol(row):
    sym = str((row or {}).get("element", "") or "").strip().upper()
    if sym:
        return sym
    name = str((row or {}).get("name", "") or "").strip().upper()
    letters = "".join(ch for ch in name if ch.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2] in {"BR", "CA", "CL", "CO", "CU", "FE", "MG", "MN", "NA", "NI", "ZN"}:
        return letters[:2]
    return letters[:1]


def plan_openmm_staged_minimization(atom_rows, min_iters=0, hydrogens_added=0):
    total_iters = int(max(0, int(min_iters or 0)))
    restrained = []
    counts = OrderedDict(
        [
            ("polymer_backbone", 0),
            ("polymer_sidechain", 0),
            ("ligand_heavy", 0),
        ]
    )
    has_ligand = False
    for row in list(atom_rows or []):
        one = dict(row or {})
        try:
            idx = int(one.get("index"))
        except Exception:
            continue
        if idx < 0:
            continue
        elem = _guess_atom_element_symbol(one)
        if elem == "H":
            continue
        resn = str(one.get("resn", "") or "").strip().upper()
        atom_name = str(one.get("name", "") or "").strip().upper()
        kind = classify_residue_kind(resn)
        if kind in {"water", "ion"}:
            continue
        if kind == "biopolymer":
            if atom_name in _BACKBONE_HEAVY_ATOM_NAMES:
                group = "polymer_backbone"
                k_ref = 2600.0 if int(hydrogens_added) > 0 else 1900.0
            else:
                group = "polymer_sidechain"
                k_ref = 1200.0 if int(hydrogens_added) > 0 else 850.0
        else:
            group = "ligand_heavy"
            has_ligand = True
            k_ref = 900.0 if int(hydrogens_added) > 0 else 650.0
        counts[group] += 1
        restrained.append(
            {
                "index": int(idx),
                "group": str(group),
                "k_ref": float(k_ref),
            }
        )
    if total_iters <= 0 or not restrained:
        stages = [{"label": "unrestrained", "iterations": int(total_iters), "scale": 0.0}] if total_iters > 0 else []
        return {
            "restrained_atoms": list(restrained),
            "counts": dict(counts),
            "restrained_atom_count": int(len(restrained)),
            "hydrogens_added": int(max(0, int(hydrogens_added or 0))),
            "stages": list(stages),
        }
    if total_iters < 12:
        stage1 = max(1, int(total_iters // 3))
        stage2 = max(0, int(total_iters // 4)) if (has_ligand or int(hydrogens_added) > 0) else 0
    else:
        stage1_frac = 0.22 if int(hydrogens_added) > 0 else 0.14
        stage2_frac = 0.14 if has_ligand else 0.10
        stage1_min = 25 if total_iters >= 120 else 10
        stage2_min = 15 if total_iters >= 120 else 5
        stage1 = max(stage1_min, int(round(float(total_iters) * float(stage1_frac))))
        stage2 = max(stage2_min, int(round(float(total_iters) * float(stage2_frac)))) if (has_ligand or int(hydrogens_added) > 0) else 0
        if stage1 >= total_iters:
            stage1 = max(1, total_iters - 1)
        if stage1 + stage2 >= total_iters:
            stage2 = max(0, total_iters - stage1 - 1)
    remaining = int(total_iters - stage1 - stage2)
    if remaining < 0:
        remaining = 0
    stages = []
    if stage1 > 0:
        stages.append(
            {
                "label": "restrained_prep",
                "iterations": int(stage1),
                "scale": 1.0,
            }
        )
    if stage2 > 0:
        stages.append(
            {
                "label": "restrained_release",
                "iterations": int(stage2),
                "scale": 0.25 if int(hydrogens_added) > 0 else 0.35,
            }
        )
    if remaining > 0:
        stages.append(
            {
                "label": "unrestrained",
                "iterations": int(remaining),
                "scale": 0.0,
            }
        )
    return {
        "restrained_atoms": list(restrained),
        "counts": dict(counts),
        "restrained_atom_count": int(len(restrained)),
        "hydrogens_added": int(max(0, int(hydrogens_added or 0))),
        "stages": list(stages),
    }


def openmm_component_uid(comp):
    if not isinstance(comp, dict):
        return ""
    uid = str(comp.get("uid", "") or "").strip()
    if uid:
        return uid
    return str(comp.get("key", "") or "").strip()


def classify_openmm_component_ff_class(comp):
    c = dict(comp or {})
    resn = str(c.get("resn", "") or "").strip().upper()
    cat = str(c.get("category", "") or "").strip().lower()
    if cat in {"ion", "water"}:
        return "standard"
    if is_standard_openmm_small_molecule_resn(resn):
        return "standard"
    return "needs_parameterization"


def default_openmm_xml_selected_keys(components):
    out = []
    seen = set()
    for comp in list(components or []):
        if classify_openmm_component_ff_class(comp) == "standard":
            continue
        key = openmm_component_uid(comp)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(key))
    return out


def split_openmm_parameterization_targets(components):
    todo = []
    skipped = []
    for comp in list(components or []):
        one = dict(comp or {})
        if classify_openmm_component_ff_class(one) == "standard":
            skipped.append(one)
        else:
            todo.append(one)
    return {
        "parameterize": list(todo),
        "skipped_standard": list(skipped),
    }


def read_openmm_forcefield_residue_names(xml_path):
    xp = str(xml_path or "").strip()
    if (not xp) or (not os.path.isfile(xp)):
        return []
    try:
        root = ET.parse(xp).getroot()
    except Exception:
        return []
    out = []
    seen = set()
    for node in root.iter():
        tag = str(getattr(node, "tag", "") or "")
        if "}" in tag:
            tag = tag.rsplit("}", 1)[-1]
        if str(tag).strip().lower() != "residue":
            continue
        name = str(node.attrib.get("name", "") or "").strip().upper()
        if (not name) or (name in seen):
            continue
        seen.add(name)
        out.append(name)
    return out


def dedupe_openmm_forcefield_xml_paths(xml_paths, residue_name_fn=None):
    reader = residue_name_fn if callable(residue_name_fn) else read_openmm_forcefield_residue_names
    paths = []
    skipped = []
    seen_paths = set()
    template_to_path = {}
    for raw in list(xml_paths or []):
        xp = str(raw or "").strip()
        if (not xp) or (xp in seen_paths):
            continue
        seen_paths.add(xp)
        residues = []
        try:
            residues = [str(x or "").strip().upper() for x in list(reader(xp) or []) if str(x or "").strip()]
        except Exception:
            residues = []
        overlap = sorted({name for name in residues if name in template_to_path})
        if overlap:
            skipped.append(
                {
                    "path": xp,
                    "residue_templates": list(overlap),
                    "kept_from": [str(template_to_path[name]) for name in overlap],
                }
            )
            continue
        paths.append(xp)
        for name in residues:
            template_to_path[str(name)] = str(xp)
    return {
        "paths": list(paths),
        "skipped": list(skipped),
    }


def resolve_openmm_component_template_plan(
    components,
    xml_map,
    file_exists_fn=None,
    component_uid_fn=None,
    ff_class_fn=None,
    xml_kind_fn=None,
    guess_forcefield_xml_from_system_fn=None,
):
    exists = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    uid_fn = component_uid_fn if callable(component_uid_fn) else openmm_component_uid
    ff_fn = ff_class_fn if callable(ff_class_fn) else classify_openmm_component_ff_class
    kind_fn = xml_kind_fn if callable(xml_kind_fn) else (lambda _p: "")
    guess_ff_fn = (
        guess_forcefield_xml_from_system_fn
        if callable(guess_forcefield_xml_from_system_fn)
        else (lambda _p: "")
    )
    xmap = dict(xml_map or {})
    paths = []
    missing = []
    skipped_standard = []
    seen = set()
    for comp in list(components or []):
        one = dict(comp or {})
        if ff_fn(one) == "standard":
            skipped_standard.append(one)
            continue
        key = str(uid_fn(one) or "").strip()
        xp = str(xmap.get(key, "") or "").strip()
        if (not xp) or (not exists(xp)):
            missing.append(one)
            continue
        kind = str(kind_fn(xp) or "").strip().lower()
        if kind == "system":
            alt = str(guess_ff_fn(xp) or "").strip()
            if alt and exists(alt):
                xp = alt
                kind = str(kind_fn(xp) or "").strip().lower()
        if kind != "forcefield" or (not exists(xp)):
            missing.append(one)
            continue
        if xp in seen:
            continue
        seen.add(xp)
        paths.append(xp)
    return {
        "paths": list(paths),
        "missing": list(missing),
        "skipped_standard": list(skipped_standard),
    }


def sanitize_pdb_conect_for_standalone_components(pdb_lines, component_keys):
    keep_keys = set(_normalize_residue_key_list(component_keys))
    lines = [str(line or "").rstrip("\n") for line in list(pdb_lines or [])]
    if not keep_keys or not lines:
        return {
            "lines": list(lines),
            "removed_links": 0,
        }
    serial_to_key = {}
    for line in lines:
        rec = str(line[:6]).strip().upper()
        if rec not in {"ATOM", "HETATM"}:
            continue
        try:
            serial = int(str(line[6:11]).strip())
        except Exception:
            continue
        resn = str(line[17:20]).strip().upper()
        chain = str(line[21:22]).strip()
        resi = str(line[22:26]).strip()
        key = _normalize_residue_key(chain, resi, resn)
        if key in keep_keys:
            serial_to_key[int(serial)] = key
    if not serial_to_key:
        return {
            "lines": list(lines),
            "removed_links": 0,
        }
    cleaned = []
    removed_links = 0
    for line in lines:
        rec = str(line[:6]).strip().upper()
        if rec != "CONECT":
            cleaned.append(line)
            continue
        fields = []
        for idx in range(6, len(line), 5):
            chunk = str(line[idx:idx + 5]).strip()
            if not chunk:
                continue
            try:
                fields.append(int(chunk))
            except Exception:
                continue
        if not fields:
            cleaned.append(line)
            continue
        src = int(fields[0])
        src_key = serial_to_key.get(src)
        kept_targets = []
        for tgt in list(fields[1:]):
            tgt_key = serial_to_key.get(int(tgt))
            if src_key is not None:
                if tgt_key != src_key:
                    removed_links += 1
                    continue
            elif tgt_key is not None:
                removed_links += 1
                continue
            kept_targets.append(int(tgt))
        if kept_targets:
            rebuilt = "CONECT" + f"{src:5d}" + "".join(f"{int(tgt):5d}" for tgt in kept_targets)
            cleaned.append(rebuilt)
    return {
        "lines": list(cleaned),
        "removed_links": int(removed_links),
    }


def rewrite_pdb_component_identity(pdb_lines, resn="", chain="", resi="", hetero=True):
    resn_txt = str(resn or "").strip().upper() or "LIG"
    resn_txt = resn_txt[:3]
    chain_txt = str(chain or "").strip()[:1]
    resi_txt = str(resi or "").strip()
    try:
        resi_field = f"{int(resi_txt):4d}"
    except Exception:
        resi_field = f"{resi_txt[:4]:>4s}" if resi_txt else "   1"
    out = []
    record = "HETATM" if bool(hetero) else "ATOM  "
    for raw in list(pdb_lines or []):
        line = str(raw or "").rstrip("\n")
        rec = str(line[:6]).strip().upper()
        if rec in {"ATOM", "HETATM"}:
            padded = line if len(line) >= 80 else line.ljust(80)
            patched = (
                f"{record:<6s}"
                f"{padded[6:17]}"
                f"{resn_txt:>3s}"
                f"{padded[20:21]}"
                f"{chain_txt:1s}"
                f"{resi_field}"
                f"{padded[26:]}"
            )
            out.append(patched.rstrip())
        else:
            out.append(line)
    return list(out)


def normalize_pdb_conect_records(pdb_lines):
    lines = [str(line or "").rstrip("\n") for line in list(pdb_lines or [])]
    out = []
    for line in lines:
        rec = str(line[:6]).strip().upper()
        if rec != "CONECT":
            out.append(line)
            continue
        fields = []
        for idx in range(6, len(line), 5):
            chunk = str(line[idx:idx + 5]).strip()
            if not chunk:
                continue
            try:
                fields.append(int(chunk))
            except Exception:
                continue
        if not fields:
            continue
        src = int(fields[0])
        seen = set()
        targets = []
        for tgt in list(fields[1:]):
            it = int(tgt)
            if it in seen or it == src:
                continue
            seen.add(it)
            targets.append(int(it))
        if not targets:
            continue
        out.append("CONECT" + f"{src:5d}" + "".join(f"{int(t):5d}" for t in targets))
    return list(out)


def compose_openmm_input_pdb(polymer_lines, component_line_blocks):
    serial_next = 1
    merged_atom_lines = []
    merged_conect_lines = []

    def _remap_block(lines, start_serial, keep_conect):
        mapping = {}
        atom_lines = []
        conect_lines = []
        next_serial = int(start_serial)
        for raw in list(lines or []):
            line = str(raw or "").rstrip("\n")
            rec = str(line[:6]).strip().upper()
            if rec not in {"ATOM", "HETATM"}:
                continue
            try:
                old_serial = int(str(line[6:11]).strip())
            except Exception:
                old_serial = None
            new_serial = int(next_serial)
            next_serial += 1
            if old_serial is not None:
                mapping[int(old_serial)] = int(new_serial)
            atom_lines.append(f"{line[:6]}{new_serial:5d}{line[11:]}")
        if bool(keep_conect):
            for raw in list(lines or []):
                line = str(raw or "").rstrip("\n")
                rec = str(line[:6]).strip().upper()
                if rec != "CONECT":
                    continue
                fields = []
                for idx in range(6, len(line), 5):
                    chunk = str(line[idx:idx + 5]).strip()
                    if not chunk:
                        continue
                    try:
                        fields.append(int(chunk))
                    except Exception:
                        continue
                if not fields:
                    continue
                src_old = int(fields[0])
                if src_old not in mapping:
                    continue
                targets = [mapping[int(t)] for t in list(fields[1:]) if int(t) in mapping]
                if not targets:
                    continue
                src_new = mapping[int(src_old)]
                conect_lines.append("CONECT" + f"{src_new:5d}" + "".join(f"{int(t):5d}" for t in targets))
        return {
            "atom_lines": list(atom_lines),
            "conect_lines": list(conect_lines),
            "next_serial": int(next_serial),
        }

    poly = _remap_block(polymer_lines, serial_next, keep_conect=False)
    serial_next = int(poly["next_serial"])
    if poly["atom_lines"]:
        merged_atom_lines.extend(list(poly["atom_lines"]))
        merged_atom_lines.append("TER")

    for block in list(component_line_blocks or []):
        one = _remap_block(block, serial_next, keep_conect=True)
        serial_next = int(one["next_serial"])
        if not one["atom_lines"]:
            continue
        merged_atom_lines.extend(list(one["atom_lines"]))
        merged_atom_lines.append("TER")
        merged_conect_lines.extend(list(one["conect_lines"]))

    while merged_atom_lines and str(merged_atom_lines[-1]).strip().upper() == "TER":
        merged_atom_lines.pop()
    out = list(merged_atom_lines)
    out.extend(list(merged_conect_lines))
    out.append("END")
    return {
        "lines": list(out),
        "atom_count": int(sum(1 for line in out if str(line[:6]).strip().upper() in {"ATOM", "HETATM"})),
        "conect_count": int(sum(1 for line in out if str(line[:6]).strip().upper() == "CONECT")),
    }


def merge_openmm_periodic_torsion_rows(rows):
    grouped = OrderedDict()
    for row in list(rows or []):
        tag = str(row.get("tag", "Proper") or "Proper").strip() or "Proper"
        key = (
            tag,
            str(row.get("type1", "") or ""),
            str(row.get("type2", "") or ""),
            str(row.get("type3", "") or ""),
            str(row.get("type4", "") or ""),
        )
        bucket = grouped.get(key)
        if bucket is None:
            bucket = {
                "tag": str(tag),
                "type1": str(key[1]),
                "type2": str(key[2]),
                "type3": str(key[3]),
                "type4": str(key[4]),
                "_n_terms": 0,
            }
            grouped[key] = bucket
        term_idx = int(bucket.get("_n_terms", 0)) + 1
        bucket["_n_terms"] = int(term_idx)
        bucket[f"periodicity{term_idx}"] = str(row.get("periodicity1", row.get("periodicity", 1)))
        bucket[f"phase{term_idx}"] = str(row.get("phase1", row.get("phase", 0.0)))
        bucket[f"k{term_idx}"] = str(row.get("k1", row.get("k", 0.0)))
    out = []
    for bucket in grouped.values():
        row = {k: v for k, v in bucket.items() if not str(k).startswith("_")}
        out.append(row)
    return out


def build_openmm_parameterization_audit(
    atom_records,
    residue_bond_pairs,
    bond_terms,
    angle_terms,
    torsion_terms,
    torsion_rows=None,
):
    atoms = list(atom_records or [])
    residue_bonds = list(residue_bond_pairs or [])
    bond_term_rows = list(bond_terms or [])
    angle_term_rows = list(angle_terms or [])
    torsion_term_rows = list(torsion_terms or [])
    merged_torsion_rows = (
        list(torsion_rows)
        if torsion_rows is not None
        else merge_openmm_periodic_torsion_rows(torsion_term_rows)
    )
    charges = []
    sigma_vals = []
    epsilon_vals = []
    for row in atoms:
        try:
            charges.append(float(row.get("charge_e", 0.0)))
        except Exception:
            charges.append(0.0)
        try:
            sigma_vals.append(float(row.get("sigma_nm", 0.0)))
        except Exception:
            sigma_vals.append(0.0)
        try:
            epsilon_vals.append(float(row.get("epsilon_kjmol", 0.0)))
        except Exception:
            epsilon_vals.append(0.0)
    proper_terms = sum(1 for row in torsion_term_rows if not bool(row.get("improper", False)))
    improper_terms = sum(1 for row in torsion_term_rows if bool(row.get("improper", False)))
    return {
        "counts": {
            "atoms": int(len(atoms)),
            "residue_bonds": int(len(residue_bonds)),
            "bond_terms": int(len(bond_term_rows)),
            "angle_terms": int(len(angle_term_rows)),
            "torsion_terms": int(len(torsion_term_rows)),
            "torsion_rows_merged": int(len(merged_torsion_rows)),
            "proper_torsion_terms": int(proper_terms),
            "improper_torsion_terms": int(improper_terms),
        },
        "charge": {
            "sum_e": float(sum(charges)) if charges else 0.0,
            "min_e": float(min(charges)) if charges else 0.0,
            "max_e": float(max(charges)) if charges else 0.0,
            "abs_sum_e": float(sum(abs(q) for q in charges)) if charges else 0.0,
        },
        "lj": {
            "sigma_min_nm": float(min(sigma_vals)) if sigma_vals else 0.0,
            "sigma_max_nm": float(max(sigma_vals)) if sigma_vals else 0.0,
            "epsilon_min_kj_mol": float(min(epsilon_vals)) if epsilon_vals else 0.0,
            "epsilon_max_kj_mol": float(max(epsilon_vals)) if epsilon_vals else 0.0,
            "zero_epsilon_atoms": int(sum(1 for v in epsilon_vals if abs(float(v)) <= 1.0e-12)),
        },
        "flags": {
            "has_bonded_terms": bool(len(bond_term_rows) > 0 or len(angle_term_rows) > 0 or len(torsion_term_rows) > 0),
            "has_lj_terms": bool(len(sigma_vals) == len(atoms) and len(epsilon_vals) == len(atoms)),
            "has_nonzero_charge": bool(any(abs(float(q)) > 1.0e-10 for q in charges)),
        },
    }


def _safe_openmm_residue_name(name):
    nm = re.sub(r"[^A-Za-z0-9]+", "", str(name or "").strip().upper())
    if not nm:
        nm = "LIG"
    return str(nm[:8])


def _normalize_openmm_element_symbol(symbol, atom_name=""):
    sym = re.sub(r"[^A-Za-z]", "", str(symbol or "").strip())
    if not sym:
        name = re.sub(r"[^A-Za-z]", "", str(atom_name or "").strip().upper())
        if len(name) >= 2 and name[:2] in {"BR", "CA", "CL", "CO", "CU", "FE", "MG", "MN", "NA", "NI", "ZN"}:
            sym = name[:2]
        else:
            sym = name[:1] or "C"
    if len(sym) == 1:
        return str(sym.upper())
    return str(sym[0].upper() + sym[1:].lower())


def export_forcefield_xml_from_amber_prmtop(prmtop_file, out_xml, residue_name_hint=""):
    prm = str(prmtop_file or "").strip()
    outp = str(out_xml or "").strip()
    if (not prm) or (not os.path.isfile(prm)):
        return {"ok": False, "error": "Ligand prmtop was not found."}
    if not outp:
        return {"ok": False, "error": "Output ForceField XML path is empty."}
    try:
        import openmm as OMM
        from openmm import app as OMM_APP
        from openmm import unit as OMM_UNIT
    except Exception as e:
        return {"ok": False, "error": f"OpenMM is not available: {e}"}
    try:
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    except Exception:
        pass
    def _fmt(value):
        try:
            return f"{float(value):.12g}"
        except Exception:
            return "0.0"
    try:
        prmtop = OMM_APP.AmberPrmtopFile(prm)
        lig_sys = prmtop.createSystem(
            nonbondedMethod=OMM_APP.NoCutoff,
            constraints=None,
            rigidWater=False,
            removeCMMotion=False,
        )
        top = prmtop.topology
        residues = list(top.residues())
        if not residues:
            return {"ok": False, "error": "Amber prmtop topology has no residues."}
        if len(residues) != 1:
            return {"ok": False, "error": f"Expected single ligand residue, found {len(residues)}."}
        res_name = _safe_openmm_residue_name(str(residue_name_hint or residues[0].name or "LIG"))
        atoms = sorted(list(top.atoms()), key=lambda atom: int(getattr(atom, "index", 0)))
        if not atoms:
            return {"ok": False, "error": "Amber prmtop topology has no atoms."}
        g2l = {}
        for li, atom in enumerate(atoms):
            g2l[int(getattr(atom, "index", li))] = int(li)
        nb_force = None
        hb_forces = []
        ha_forces = []
        pt_forces = []
        for force in lig_sys.getForces():
            if isinstance(force, OMM.NonbondedForce) and nb_force is None:
                nb_force = force
            elif isinstance(force, OMM.HarmonicBondForce):
                hb_forces.append(force)
            elif isinstance(force, OMM.HarmonicAngleForce):
                ha_forces.append(force)
            elif isinstance(force, OMM.PeriodicTorsionForce):
                pt_forces.append(force)
        if nb_force is None:
            return {"ok": False, "error": "Ligand system has no NonbondedForce."}
        used_names = {}
        atom_records = []
        for li, atom in enumerate(atoms):
            gi = int(getattr(atom, "index", li))
            name = str(getattr(atom, "name", "") or "").strip() or f"A{li+1}"
            name = re.sub(r"\s+", "", str(name))
            if name in used_names:
                used_names[name] += 1
                name = f"{name}{used_names[name]}"
            else:
                used_names[name] = 1
            element = ""
            try:
                if getattr(atom, "element", None) is not None:
                    element = str(atom.element.symbol or "").strip()
            except Exception:
                element = ""
            element = _normalize_openmm_element_symbol(element, atom_name=name)
            q, sig, eps = nb_force.getParticleParameters(int(gi))
            atom_records.append(
                {
                    "local_index": int(li),
                    "global_index": int(gi),
                    "name": str(name),
                    "element": str(element),
                    "type": f"{res_name}_{li+1}",
                    "mass_da": float(_omm_quantity_to_float(lig_sys.getParticleMass(gi), OMM_UNIT.dalton, default=12.011)),
                    "charge_e": float(_omm_quantity_to_float(q, OMM_UNIT.elementary_charge, default=0.0)),
                    "sigma_nm": float(max(1.0e-8, _omm_quantity_to_float(sig, OMM_UNIT.nanometer, default=0.3))),
                    "epsilon_kjmol": float(max(0.0, _omm_quantity_to_float(eps, OMM_UNIT.kilojoule_per_mole, default=0.0))),
                }
            )
        bond_pairs = []
        bond_seen = set()
        for bond in list(top.bonds()):
            a = int(getattr(bond[0], "index", -1))
            b = int(getattr(bond[1], "index", -1))
            if (a not in g2l) or (b not in g2l):
                continue
            ia = int(g2l[a])
            ib = int(g2l[b])
            if ia == ib:
                continue
            key = (min(ia, ib), max(ia, ib))
            if key in bond_seen:
                continue
            bond_seen.add(key)
            bond_pairs.append(key)
        if (not bond_pairs) and hb_forces:
            for force in hb_forces:
                for ii in range(force.getNumBonds()):
                    a, b, _length, _kval = force.getBondParameters(ii)
                    ai = int(a)
                    bi = int(b)
                    if (ai not in g2l) or (bi not in g2l):
                        continue
                    ia = int(g2l[ai])
                    ib = int(g2l[bi])
                    key = (min(ia, ib), max(ia, ib))
                    if key in bond_seen:
                        continue
                    bond_seen.add(key)
                    bond_pairs.append(key)
        if (len(atom_records) > 1) and (not bond_pairs):
            return {"ok": False, "error": "Ligand topology has multiple atoms but no residue bonds were recovered from the Amber topology."}
        bond_set = set(bond_pairs)
        def _is_bond(i, j):
            return (min(int(i), int(j)), max(int(i), int(j))) in bond_set
        bond_terms = []
        for force in hb_forces:
            for ii in range(force.getNumBonds()):
                a, b, length, kval = force.getBondParameters(ii)
                ai = int(a)
                bi = int(b)
                if (ai not in g2l) or (bi not in g2l):
                    continue
                bond_terms.append(
                    {
                        "i": int(g2l[ai]),
                        "j": int(g2l[bi]),
                        "length_nm": float(_omm_quantity_to_float(length, OMM_UNIT.nanometer, default=0.14)),
                        "k_kjmol_nm2": float(
                            _omm_quantity_to_float(
                                kval,
                                OMM_UNIT.kilojoule_per_mole / (OMM_UNIT.nanometer * OMM_UNIT.nanometer),
                                default=300000.0,
                            )
                        ),
                    }
                )
        angle_terms = []
        for force in ha_forces:
            for ii in range(force.getNumAngles()):
                a, b, c, theta, kval = force.getAngleParameters(ii)
                ai = int(a)
                bi = int(b)
                ci = int(c)
                if (ai not in g2l) or (bi not in g2l) or (ci not in g2l):
                    continue
                angle_terms.append(
                    {
                        "i": int(g2l[ai]),
                        "j": int(g2l[bi]),
                        "k": int(g2l[ci]),
                        "angle_rad": float(_omm_quantity_to_float(theta, OMM_UNIT.radian, default=1.91)),
                        "k_kjmol_rad2": float(
                            _omm_quantity_to_float(
                                kval,
                                OMM_UNIT.kilojoule_per_mole / (OMM_UNIT.radian * OMM_UNIT.radian),
                                default=350.0,
                            )
                        ),
                    }
                )
        torsion_terms = []
        for force in pt_forces:
            for ii in range(force.getNumTorsions()):
                a, b, c, d, per, phase, kval = force.getTorsionParameters(ii)
                ai = int(a)
                bi = int(b)
                ci = int(c)
                di = int(d)
                if (ai not in g2l) or (bi not in g2l) or (ci not in g2l) or (di not in g2l):
                    continue
                i1 = int(g2l[ai])
                i2 = int(g2l[bi])
                i3 = int(g2l[ci])
                i4 = int(g2l[di])
                is_proper = bool(_is_bond(i1, i2) and _is_bond(i2, i3) and _is_bond(i3, i4))
                is_improper = False
                if not is_proper:
                    hubs = []
                    for hub, others in (
                        (i1, (i2, i3, i4)),
                        (i2, (i1, i3, i4)),
                        (i3, (i1, i2, i4)),
                        (i4, (i1, i2, i3)),
                    ):
                        deg = int(sum(1 for other in others if _is_bond(hub, other)))
                        hubs.append(deg)
                    is_improper = bool(max(hubs) >= 3)
                torsion_terms.append(
                    {
                        "i": int(i1),
                        "j": int(i2),
                        "k": int(i3),
                        "l": int(i4),
                        "periodicity": int(per),
                        "phase_rad": float(_omm_quantity_to_float(phase, OMM_UNIT.radian, default=0.0)),
                        "k_kjmol": float(_omm_quantity_to_float(kval, OMM_UNIT.kilojoule_per_mole, default=0.0)),
                        "improper": bool(is_improper and (not is_proper)),
                    }
                )
        ff_root = ET.Element("ForceField")
        atom_types = ET.SubElement(ff_root, "AtomTypes")
        for atom in atom_records:
            ET.SubElement(
                atom_types,
                "Type",
                {
                    "name": str(atom["type"]),
                    "class": str(atom["type"]),
                    "element": str(atom["element"]),
                    "mass": _fmt(atom["mass_da"]),
                },
            )
        residues_node = ET.SubElement(ff_root, "Residues")
        res_node = ET.SubElement(residues_node, "Residue", {"name": str(res_name)})
        for atom in atom_records:
            ET.SubElement(
                res_node,
                "Atom",
                {
                    "name": str(atom["name"]),
                    "type": str(atom["type"]),
                    "charge": _fmt(atom["charge_e"]),
                },
            )
        for i, j in bond_pairs:
            ET.SubElement(
                res_node,
                "Bond",
                {
                    "atomName1": str(atom_records[int(i)]["name"]),
                    "atomName2": str(atom_records[int(j)]["name"]),
                },
            )
        if bond_terms:
            bond_node = ET.SubElement(ff_root, "HarmonicBondForce")
            for term in bond_terms:
                ET.SubElement(
                    bond_node,
                    "Bond",
                    {
                        "type1": str(atom_records[int(term["i"])]["type"]),
                        "type2": str(atom_records[int(term["j"])]["type"]),
                        "length": _fmt(term["length_nm"]),
                        "k": _fmt(term["k_kjmol_nm2"]),
                    },
                )
        if angle_terms:
            angle_node = ET.SubElement(ff_root, "HarmonicAngleForce")
            for term in angle_terms:
                ET.SubElement(
                    angle_node,
                    "Angle",
                    {
                        "type1": str(atom_records[int(term["i"])]["type"]),
                        "type2": str(atom_records[int(term["j"])]["type"]),
                        "type3": str(atom_records[int(term["k"])]["type"]),
                        "angle": _fmt(term["angle_rad"]),
                        "k": _fmt(term["k_kjmol_rad2"]),
                    },
                )
        torsion_rows = []
        if torsion_terms:
            any_improper = any(bool(term.get("improper", False)) for term in torsion_terms)
            torsion_node = ET.SubElement(
                ff_root,
                "PeriodicTorsionForce",
                ({"ordering": "amber"} if any_improper else {}),
            )
            for term in torsion_terms:
                order = [int(term["i"]), int(term["j"]), int(term["k"]), int(term["l"])]
                if bool(term.get("improper", False)):
                    central = None
                    for hub in list(order):
                        deg = int(sum(1 for other in list(order) if other != hub and _is_bond(hub, other)))
                        if deg >= 3:
                            central = int(hub)
                            break
                    if central is not None:
                        outer = [int(idx) for idx in list(order) if int(idx) != int(central)]
                        order = [int(central)] + list(outer)
                torsion_rows.append(
                    {
                        "tag": ("Improper" if bool(term.get("improper", False)) else "Proper"),
                        "type1": str(atom_records[int(order[0])]["type"]),
                        "type2": str(atom_records[int(order[1])]["type"]),
                        "type3": str(atom_records[int(order[2])]["type"]),
                        "type4": str(atom_records[int(order[3])]["type"]),
                        "periodicity1": str(int(term["periodicity"])),
                        "phase1": _fmt(term["phase_rad"]),
                        "k1": _fmt(term["k_kjmol"]),
                    }
                )
            torsion_rows = merge_openmm_periodic_torsion_rows(torsion_rows)
            for row in torsion_rows:
                ET.SubElement(
                    torsion_node,
                    str(row.get("tag", "Proper")),
                    {str(k): str(v) for k, v in dict(row).items() if str(k) != "tag"},
                )
        nb_node = ET.SubElement(
            ff_root,
            "NonbondedForce",
            {"coulomb14scale": "0.833333333333", "lj14scale": "0.5"},
        )
        ET.SubElement(nb_node, "UseAttributeFromResidue", {"name": "charge"})
        for atom in atom_records:
            ET.SubElement(
                nb_node,
                "Atom",
                {
                    "type": str(atom["type"]),
                    "sigma": _fmt(atom["sigma_nm"]),
                    "epsilon": _fmt(atom["epsilon_kjmol"]),
                },
            )
        source_summary = summarize_openmm_system_terms(lig_sys, unit_module=OMM_UNIT)
        tree = ET.ElementTree(ff_root)
        try:
            ET.indent(tree, space="  ")
        except Exception:
            pass
        tree.write(outp, encoding="utf-8", xml_declaration=True)
        meta = {
            "source_prmtop": str(prm),
            "residue_name": str(res_name),
            "atoms": [{"name": str(atom["name"]), "element": str(atom["element"])} for atom in atom_records],
            "counts": {
                "atoms": int(len(atom_records)),
                "bonds": int(len(bond_terms)),
                "angles": int(len(angle_terms)),
                "torsions": int(len(torsion_terms)),
            },
        }
        with open(outp + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        audit = build_openmm_parameterization_audit(
            atom_records=atom_records,
            residue_bond_pairs=bond_pairs,
            bond_terms=bond_terms,
            angle_terms=angle_terms,
            torsion_terms=torsion_terms,
            torsion_rows=torsion_rows,
        )
        audit["source_prmtop"] = str(prm)
        audit["residue_name"] = str(res_name)
        audit["source_system"] = {
            "particles": int(source_summary.get("particles", 0) or 0),
            "force_counts": dict(source_summary.get("force_counts", {}) or {}),
            "nonbonded_exceptions": int(len(list(source_summary.get("nonbonded_exceptions", []) or []))),
        }
        try:
            ff_rt = OMM_APP.ForceField(outp)
            rt_sys = ff_rt.createSystem(
                top,
                nonbondedMethod=OMM_APP.NoCutoff,
                constraints=None,
                rigidWater=False,
                removeCMMotion=False,
            )
            rt_summary = summarize_openmm_system_terms(rt_sys, unit_module=OMM_UNIT)
            roundtrip = compare_openmm_system_term_summaries(
                source_summary,
                rt_summary,
                tol=1.0e-8,
            )
        except Exception as e:
            roundtrip = {
                "ok": False,
                "issues": [f"ffxml_roundtrip_build_failed:{e}"],
                "error": str(e),
            }
        audit["roundtrip"] = dict(roundtrip or {})
        with open(outp + ".audit.json", "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2)
        if not bool((roundtrip or {}).get("ok", False)):
            issues = list((roundtrip or {}).get("issues", []) or [])
            detail = str(issues[0] if issues else (roundtrip or {}).get("error", "parameter mismatch"))
            return {"ok": False, "error": f"ForceField XML round-trip validation failed: {detail}"}
        return {
            "ok": True,
            "message": "ok",
            "xml": str(outp),
            "meta_json": str(outp + ".meta.json"),
            "audit_json": str(outp + ".audit.json"),
            "residue_name": str(res_name),
            "counts": dict(audit.get("counts", {}) or {}),
            "charge": dict(audit.get("charge", {}) or {}),
            "roundtrip": dict(roundtrip or {}),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _omm_quantity_to_float(value, unit=None, default=0.0):
    try:
        if unit is not None and hasattr(value, "value_in_unit"):
            return float(value.value_in_unit(unit))
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        try:
            return float(getattr(value, "_value", default))
        except Exception:
            return float(default)


def summarize_openmm_system_terms(system, unit_module=None):
    unit = unit_module
    summary = {
        "particles": int(getattr(system, "getNumParticles", lambda: 0)()),
        "force_counts": OrderedDict(),
        "masses_da": [],
        "nonbonded_particles": [],
        "nonbonded_exceptions": [],
        "bonds": [],
        "angles": [],
        "torsions": [],
    }
    dalton = getattr(unit, "dalton", None)
    elementary_charge = getattr(unit, "elementary_charge", None)
    nanometer = getattr(unit, "nanometer", None)
    kilojoule_per_mole = getattr(unit, "kilojoule_per_mole", None)
    radian = getattr(unit, "radian", None)
    for idx in range(int(summary["particles"])):
        try:
            mass_val = _omm_quantity_to_float(system.getParticleMass(idx), dalton, default=0.0)
        except Exception:
            mass_val = 0.0
        summary["masses_da"].append(float(mass_val))
    for force in list(getattr(system, "getForces", lambda: [])()):
        name = str(getattr(force, "__class__", type(force)).__name__ or "")
        summary["force_counts"][name] = int(summary["force_counts"].get(name, 0)) + 1
        if name == "NonbondedForce":
            for idx in range(int(summary["particles"])):
                q, sig, eps = force.getParticleParameters(idx)
                summary["nonbonded_particles"].append(
                    {
                        "index": int(idx),
                        "charge_e": float(_omm_quantity_to_float(q, elementary_charge, default=0.0)),
                        "sigma_nm": float(_omm_quantity_to_float(sig, nanometer, default=0.0)),
                        "epsilon_kjmol": float(_omm_quantity_to_float(eps, kilojoule_per_mole, default=0.0)),
                    }
                )
            charge_sq_unit = None
            try:
                if elementary_charge is not None:
                    charge_sq_unit = elementary_charge * elementary_charge
            except Exception:
                charge_sq_unit = None
            for ii in range(int(force.getNumExceptions())):
                a, b, qprod, sig, eps = force.getExceptionParameters(ii)
                ia = int(a)
                ib = int(b)
                if ia > ib:
                    ia, ib = ib, ia
                summary["nonbonded_exceptions"].append(
                    {
                        "a": int(ia),
                        "b": int(ib),
                        "chargeprod_e2": float(_omm_quantity_to_float(qprod, charge_sq_unit, default=0.0)),
                        "sigma_nm": float(_omm_quantity_to_float(sig, nanometer, default=0.0)),
                        "epsilon_kjmol": float(_omm_quantity_to_float(eps, kilojoule_per_mole, default=0.0)),
                    }
                )
        elif name == "HarmonicBondForce":
            k_unit = None
            try:
                if kilojoule_per_mole is not None and nanometer is not None:
                    k_unit = kilojoule_per_mole / (nanometer * nanometer)
            except Exception:
                k_unit = None
            for ii in range(int(force.getNumBonds())):
                a, b, length, kval = force.getBondParameters(ii)
                ia = int(a)
                ib = int(b)
                if ia > ib:
                    ia, ib = ib, ia
                summary["bonds"].append(
                    {
                        "a": int(ia),
                        "b": int(ib),
                        "length_nm": float(_omm_quantity_to_float(length, nanometer, default=0.0)),
                        "k_kjmol_nm2": float(_omm_quantity_to_float(kval, k_unit, default=0.0)),
                    }
                )
        elif name == "HarmonicAngleForce":
            k_unit = None
            try:
                if kilojoule_per_mole is not None and radian is not None:
                    k_unit = kilojoule_per_mole / (radian * radian)
            except Exception:
                k_unit = None
            for ii in range(int(force.getNumAngles())):
                a, b, c, theta, kval = force.getAngleParameters(ii)
                ia = int(a)
                ib = int(b)
                ic = int(c)
                if ia > ic:
                    ia, ic = ic, ia
                summary["angles"].append(
                    {
                        "a": int(ia),
                        "b": int(ib),
                        "c": int(ic),
                        "angle_rad": float(_omm_quantity_to_float(theta, radian, default=0.0)),
                        "k_kjmol_rad2": float(_omm_quantity_to_float(kval, k_unit, default=0.0)),
                    }
                )
        elif name == "PeriodicTorsionForce":
            for ii in range(int(force.getNumTorsions())):
                a, b, c, d, per, phase, kval = force.getTorsionParameters(ii)
                forward = (int(a), int(b), int(c), int(d))
                reverse = (int(d), int(c), int(b), int(a))
                quad = min(forward, reverse)
                summary["torsions"].append(
                    {
                        "a": int(quad[0]),
                        "b": int(quad[1]),
                        "c": int(quad[2]),
                        "d": int(quad[3]),
                        "periodicity": int(per),
                        "phase_rad": float(_omm_quantity_to_float(phase, radian, default=0.0)),
                        "k_kjmol": float(_omm_quantity_to_float(kval, kilojoule_per_mole, default=0.0)),
                    }
                )
    summary["masses_da"] = [float(v) for v in list(summary["masses_da"])]
    summary["nonbonded_particles"] = sorted(
        list(summary["nonbonded_particles"]),
        key=lambda row: int(row.get("index", 0)),
    )
    summary["nonbonded_exceptions"] = sorted(
        list(summary["nonbonded_exceptions"]),
        key=lambda row: (
            int(row.get("a", 0)),
            int(row.get("b", 0)),
            float(row.get("chargeprod_e2", 0.0)),
            float(row.get("sigma_nm", 0.0)),
            float(row.get("epsilon_kjmol", 0.0)),
        ),
    )
    summary["bonds"] = sorted(
        list(summary["bonds"]),
        key=lambda row: (
            int(row.get("a", 0)),
            int(row.get("b", 0)),
            float(row.get("length_nm", 0.0)),
            float(row.get("k_kjmol_nm2", 0.0)),
        ),
    )
    summary["angles"] = sorted(
        list(summary["angles"]),
        key=lambda row: (
            int(row.get("a", 0)),
            int(row.get("b", 0)),
            int(row.get("c", 0)),
            float(row.get("angle_rad", 0.0)),
            float(row.get("k_kjmol_rad2", 0.0)),
        ),
    )
    summary["torsions"] = sorted(
        list(summary["torsions"]),
        key=lambda row: (
            int(row.get("a", 0)),
            int(row.get("b", 0)),
            int(row.get("c", 0)),
            int(row.get("d", 0)),
            int(row.get("periodicity", 0)),
            float(row.get("phase_rad", 0.0)),
            float(row.get("k_kjmol", 0.0)),
        ),
    )
    return summary


def compare_openmm_system_term_summaries(source_summary, target_summary, tol=1.0e-8):
    src = dict(source_summary or {})
    dst = dict(target_summary or {})
    tol_val = abs(float(tol))
    result = {
        "ok": True,
        "tolerance": float(tol_val),
        "counts": {
            "particles_source": int(src.get("particles", 0) or 0),
            "particles_target": int(dst.get("particles", 0) or 0),
            "nonbonded_particles_source": int(len(list(src.get("nonbonded_particles", []) or []))),
            "nonbonded_particles_target": int(len(list(dst.get("nonbonded_particles", []) or []))),
            "nonbonded_exceptions_source": int(len(list(src.get("nonbonded_exceptions", []) or []))),
            "nonbonded_exceptions_target": int(len(list(dst.get("nonbonded_exceptions", []) or []))),
            "bonds_source": int(len(list(src.get("bonds", []) or []))),
            "bonds_target": int(len(list(dst.get("bonds", []) or []))),
            "angles_source": int(len(list(src.get("angles", []) or []))),
            "angles_target": int(len(list(dst.get("angles", []) or []))),
            "torsions_source": int(len(list(src.get("torsions", []) or []))),
            "torsions_target": int(len(list(dst.get("torsions", []) or []))),
        },
        "force_counts_source": dict(src.get("force_counts", {}) or {}),
        "force_counts_target": dict(dst.get("force_counts", {}) or {}),
        "force_counts_match": bool(dict(src.get("force_counts", {}) or {}) == dict(dst.get("force_counts", {}) or {})),
        "max_abs_diff": OrderedDict(),
        "issues": [],
    }

    def _record_issue(message):
        result["ok"] = False
        result["issues"].append(str(message))

    if int(src.get("particles", 0) or 0) != int(dst.get("particles", 0) or 0):
        _record_issue(
            f"particle_count_mismatch:{int(src.get('particles', 0) or 0)}!={int(dst.get('particles', 0) or 0)}"
        )

    def _compare_float_lists(label, lhs, rhs):
        left = [float(v) for v in list(lhs or [])]
        right = [float(v) for v in list(rhs or [])]
        if len(left) != len(right):
            _record_issue(f"{label}_count_mismatch:{len(left)}!={len(right)}")
            result["max_abs_diff"][label] = None
            return
        max_diff = 0.0
        for lv, rv in zip(left, right):
            max_diff = max(max_diff, abs(float(lv) - float(rv)))
        result["max_abs_diff"][label] = float(max_diff)
        if max_diff > tol_val:
            _record_issue(f"{label}_mismatch:max_abs_diff={max_diff:.3e}")

    def _compare_rows(label, lhs, rhs, key_fields, float_fields):
        sort_fields = tuple(list(key_fields or []) + list(float_fields or []))
        left = sorted(
            [dict(row or {}) for row in list(lhs or [])],
            key=lambda row: tuple(row.get(name) for name in sort_fields),
        )
        right = sorted(
            [dict(row or {}) for row in list(rhs or [])],
            key=lambda row: tuple(row.get(name) for name in sort_fields),
        )
        if len(left) != len(right):
            _record_issue(f"{label}_count_mismatch:{len(left)}!={len(right)}")
            result["max_abs_diff"][label] = None
            return
        max_diff = 0.0
        for row_l, row_r in zip(left, right):
            key_l = tuple(row_l.get(name) for name in key_fields)
            key_r = tuple(row_r.get(name) for name in key_fields)
            if key_l != key_r:
                _record_issue(f"{label}_key_mismatch:{key_l}!={key_r}")
                result["max_abs_diff"][label] = None
                return
            for field in list(float_fields or []):
                diff = abs(float(row_l.get(field, 0.0) or 0.0) - float(row_r.get(field, 0.0) or 0.0))
                max_diff = max(max_diff, float(diff))
        result["max_abs_diff"][label] = float(max_diff)
        if max_diff > tol_val:
            _record_issue(f"{label}_mismatch:max_abs_diff={max_diff:.3e}")

    def _normalize_exception_rows(rows):
        out = []
        for row in list(rows or []):
            one = dict(row or {})
            qprod = abs(float(one.get("chargeprod_e2", 0.0) or 0.0))
            epsilon = abs(float(one.get("epsilon_kjmol", 0.0) or 0.0))
            if qprod <= tol_val and epsilon <= tol_val:
                # OpenMM may store arbitrary sigma for fully excluded pairs.  It has no physical effect.
                one["sigma_nm"] = 0.0
            out.append(one)
        return list(out)

    def _bond_pairs(rows):
        out = set()
        for row in list(rows or []):
            try:
                a = int(row.get("a", 0))
                b = int(row.get("b", 0))
            except Exception:
                continue
            out.add((min(a, b), max(a, b)))
        return out

    def _normalize_torsion_rows(rows, bonds):
        out = []
        for row in list(rows or []):
            ids = [
                int(row.get("a", 0) or 0),
                int(row.get("b", 0) or 0),
                int(row.get("c", 0) or 0),
                int(row.get("d", 0) or 0),
            ]
            central = None
            for hub in list(ids):
                deg = int(sum(1 for other in list(ids) if other != hub and (min(hub, other), max(hub, other)) in bonds))
                if deg >= 3:
                    central = int(hub)
                    break
            if central is not None:
                outer = sorted(int(v) for v in list(ids) if int(v) != int(central))
                out.append(
                    {
                        "kind": "improper",
                        "c0": int(central),
                        "c1": int(outer[0]),
                        "c2": int(outer[1]),
                        "c3": int(outer[2]),
                        "periodicity": int(row.get("periodicity", 0) or 0),
                        "phase_rad": float(row.get("phase_rad", 0.0) or 0.0),
                        "k_kjmol": float(row.get("k_kjmol", 0.0) or 0.0),
                    }
                )
                continue
            forward = tuple(int(v) for v in ids)
            reverse = tuple(reversed(forward))
            quad = min(forward, reverse)
            out.append(
                {
                    "kind": "proper",
                    "c0": int(quad[0]),
                    "c1": int(quad[1]),
                    "c2": int(quad[2]),
                    "c3": int(quad[3]),
                    "periodicity": int(row.get("periodicity", 0) or 0),
                    "phase_rad": float(row.get("phase_rad", 0.0) or 0.0),
                    "k_kjmol": float(row.get("k_kjmol", 0.0) or 0.0),
                }
            )
        return list(out)

    _compare_float_lists("particle_mass_da", src.get("masses_da", []), dst.get("masses_da", []))
    _compare_rows(
        "nonbonded_particles",
        src.get("nonbonded_particles", []),
        dst.get("nonbonded_particles", []),
        key_fields=("index",),
        float_fields=("charge_e", "sigma_nm", "epsilon_kjmol"),
    )
    _compare_rows(
        "nonbonded_exceptions",
        _normalize_exception_rows(src.get("nonbonded_exceptions", [])),
        _normalize_exception_rows(dst.get("nonbonded_exceptions", [])),
        key_fields=("a", "b"),
        float_fields=("chargeprod_e2", "sigma_nm", "epsilon_kjmol"),
    )
    _compare_rows(
        "bonds",
        src.get("bonds", []),
        dst.get("bonds", []),
        key_fields=("a", "b"),
        float_fields=("length_nm", "k_kjmol_nm2"),
    )
    _compare_rows(
        "angles",
        src.get("angles", []),
        dst.get("angles", []),
        key_fields=("a", "b", "c"),
        float_fields=("angle_rad", "k_kjmol_rad2"),
    )
    _compare_rows(
        "torsions",
        _normalize_torsion_rows(src.get("torsions", []), _bond_pairs(src.get("bonds", []))),
        _normalize_torsion_rows(dst.get("torsions", []), _bond_pairs(dst.get("bonds", []))),
        key_fields=("kind", "c0", "c1", "c2", "c3", "periodicity"),
        float_fields=("phase_rad", "k_kjmol"),
    )
    return result


def assess_ligand_topology_signature_compatibility(expected, observed):
    exp = dict(expected or {})
    obs = dict(observed or {})
    def _as_int(value, default=-1):
        try:
            if value is None:
                return int(default)
            return int(value)
        except Exception:
            return int(default)
    try:
        exp_rot = _as_int(exp.get("rotb", -1), default=-1)
    except Exception:
        exp_rot = -1
    try:
        exp_aro = _as_int(exp.get("aromatic_bonds", -1), default=-1)
    except Exception:
        exp_aro = -1
    try:
        obs_rot = _as_int(obs.get("rotb", -1), default=-1)
    except Exception:
        obs_rot = -1
    try:
        obs_aro = _as_int(obs.get("aromatic_bonds", -1), default=-1)
    except Exception:
        obs_aro = -1
    result = {
        "ok": True,
        "expected": {"rotb": int(exp_rot), "aromatic_bonds": int(exp_aro)},
        "observed": {"rotb": int(obs_rot), "aromatic_bonds": int(obs_aro)},
        "issues": [],
    }

    def _issue(message):
        result["ok"] = False
        result["issues"].append(str(message))

    if exp_rot >= 0 and obs_rot >= 0 and obs_rot > int(exp_rot + 2):
        _issue(f"rotatable_bonds_exceeds_expected:{obs_rot}>{exp_rot + 2}")
    if exp_aro >= 4 and obs_aro >= 0:
        min_aro = max(4, int(exp_aro - 2))
        if obs_aro < min_aro:
            _issue(f"aromatic_bonds_too_low:{obs_aro}<{min_aro}")
        if exp_rot == 0 and obs_rot >= 0 and obs_rot > 1:
            _issue(f"aromatic_ligand_unexpected_flexibility:{obs_rot}>1")
    return result


def build_mopac_keywords_for_operation(operation, base_keywords=""):
    return qmmm_mopac_keywords_for_operation(operation=operation, base_keywords=base_keywords)


def _hydrogens_bonded_to(atom_rows, qm_set, n_atoms, max_bond_ang=1.25):
    """Indices of H atoms (not in qm_set) whose nearest heavy atom within max_bond_ang is in qm_set."""
    rows = []
    for row in list(atom_rows or []):
        try:
            idx = int(row.get("index"))
        except Exception:
            continue
        xyz = row.get("xyz")
        if xyz is None and all(k in row for k in ("x", "y", "z")):
            xyz = (row["x"], row["y"], row["z"])
        if xyz is None or idx < 0 or idx >= n_atoms:
            continue
        el = str(row.get("element", "") or "").strip().upper()
        rows.append((idx, el, np.asarray(xyz, dtype=float).reshape(3)))
    if not rows:
        return []
    heavy = [(i, p) for i, el, p in rows if el != "H"]
    if not heavy:
        return []
    hidx = np.array([i for i, _p in heavy])
    hxyz = np.array([p for _i, p in heavy])
    out = []
    for i, el, p in rows:
        if el != "H" or i in qm_set:
            continue
        d = np.linalg.norm(hxyz - p, axis=1)
        j = int(np.argmin(d))
        if float(d[j]) <= max_bond_ang and int(hidx[j]) in qm_set:
            out.append(int(i))
    return out


def resolve_qmmm_regions(atom_rows, n_atoms, config=None):
    """
    Resolve QM/MM atom indices from explicit settings or an automatic hetero-ligand rule.
    atom_rows entries may include:
      - index (0-based)
      - resn, chain, resi
    """
    cfg = dict(config or {})
    n = int(max(0, int(n_atoms)))
    if n <= 0:
        raise ValueError("n_atoms must be > 0.")
    explicit_qm = _as_indices(cfg.get("qm_indices", []), n, "qm_indices")
    explicit_mm = _as_indices(cfg.get("mm_indices", []), n, "mm_indices")
    qm_res_keys = set(_normalize_residue_key_list(cfg.get("qm_residue_keys", [])))
    mm_res_keys = set(_normalize_residue_key_list(cfg.get("mm_residue_keys", [])))
    qm_atom_desc = list(_normalize_atom_descriptor_list(cfg.get("qm_atom_descriptors", [])))
    mm_atom_desc = list(_normalize_atom_descriptor_list(cfg.get("mm_atom_descriptors", [])))
    force_qm_atom_desc = list(_normalize_atom_descriptor_list(cfg.get("force_qm_atom_descriptors", [])))
    force_qm_indices = _as_indices(cfg.get("force_qm_indices", []), n, "force_qm_indices")
    force_qm_res_keys = set(_normalize_residue_key_list(cfg.get("force_qm_residue_keys", [])))
    info = {"mode": "", "selection_hint": "", "qm_group_key": None}

    if explicit_qm:
        qm_idx = list(explicit_qm)
        info["mode"] = "explicit_qm_indices"
    elif qm_atom_desc:
        qm_idx = sorted(set(int(i) for i in _resolve_atom_descriptors(atom_rows, qm_atom_desc, "qm_atom_descriptors")))
        info["mode"] = "atom_descriptor_selection"
        info["selection_hint"] = f"QM atom selection (n={len(qm_idx)})"
    elif bool(cfg.get("auto_include_pocket", False)):
        rec = qmmm_recommend_region(
            atom_rows=atom_rows,
            operation=str(cfg.get("operation", "probe") or "probe"),
            ligand_resnames=list(cfg.get("qm_resnames", []) or []),
            qm_pocket_cutoff_ang=float(cfg.get("qm_pocket_cutoff_ang", 4.0) or 4.0),
            include_pocket=True,
            include_metals=bool(cfg.get("include_metals", True)),
            max_qm_atoms=int(cfg.get("max_qm_atoms", 180) or 180),
        )
        if not bool(rec.get("ok", False)):
            raise ValueError(str(rec.get("error", "Automatic ligand-pocket QM region failed.")))
        qm_idx = [int(i) for i in list(rec.get("qm_indices", []) or [])]
        info["mode"] = "auto_ligand_pocket"
        info["qm_group_key"] = tuple((rec.get("center_residue", {}) or {}).get(k, "") for k in ("chain", "resi", "resn"))
        info["selection_hint"] = str(rec.get("selection_hint", "") or f"Auto ligand pocket (n={len(qm_idx)})")
        info["recommendation"] = dict(rec)
    else:
        pref_resn = {str(x).strip().upper() for x in list(cfg.get("qm_resnames", []) or []) if str(x).strip()}
        groups = {}
        for row in list(atom_rows or []):
            try:
                idx = int(row.get("index"))
            except Exception:
                continue
            if idx < 0 or idx >= n:
                continue
            resn = str(row.get("resn", "") or "").strip().upper()
            chain = str(row.get("chain", "") or "").strip()
            resi = str(row.get("resi", "") or "").strip()
            key = (chain, resi, resn)
            if qm_res_keys:
                if key not in qm_res_keys:
                    continue
            elif pref_resn:
                if resn not in pref_resn:
                    continue
            else:
                kind = classify_residue_kind(resn)
                if kind != "hetero":
                    continue
            groups.setdefault(key, []).append(idx)
        if not groups:
            if qm_res_keys:
                raise ValueError("No atoms matched qm_residue_keys in current topology.")
            if pref_resn:
                raise ValueError("No atoms matched qm_resnames in current topology.")
            raise ValueError("Automatic QM region detection failed: no hetero residue found.")
        ranked = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
        if qm_res_keys:
            qm_atoms = []
            chosen = []
            for key, atoms in ranked:
                if key in qm_res_keys:
                    qm_atoms.extend(list(atoms))
                    chosen.append(key)
            qm_idx = sorted(set(int(i) for i in qm_atoms))
            info["mode"] = "residue_key_selection"
            if chosen:
                head = chosen[0]
                info["qm_group_key"] = tuple(head)
                info["selection_hint"] = f"{head[2]} {head[0]}:{head[1]} (+{max(0, len(chosen)-1)} residues)"
        else:
            best_key, best_atoms = ranked[0]
            qm_idx = sorted(set(int(i) for i in best_atoms))
            info["mode"] = "auto_largest_hetero_residue" if not pref_resn else "auto_qm_resnames"
            info["qm_group_key"] = tuple(best_key)
            info["selection_hint"] = f"{best_key[2]} {best_key[0]}:{best_key[1]} (n={len(qm_idx)})"

    if not qm_idx:
        raise ValueError("QM region is empty.")
    forced_qm = []
    if force_qm_atom_desc:
        forced_qm.extend(_resolve_atom_descriptors(atom_rows, force_qm_atom_desc, "force_qm_atom_descriptors"))
    if force_qm_indices:
        forced_qm.extend(list(force_qm_indices))
    if force_qm_res_keys:
        for row in list(atom_rows or []):
            try:
                idx = int(row.get("index"))
            except Exception:
                continue
            if idx < 0 or idx >= n:
                continue
            if _normalize_residue_key(row.get("chain", ""), row.get("resi", ""), row.get("resn", "")) in force_qm_res_keys:
                forced_qm.append(int(idx))
    if forced_qm:
        before = len(set(qm_idx))
        qm_idx = sorted(set(int(i) for i in list(qm_idx) + list(forced_qm)))
        info["forced_qm_atom_count"] = int(max(0, len(set(qm_idx)) - before))
    added = []
    if bool(cfg.get("include_bonded_hydrogens", True)) and not explicit_qm:
        # selections made on a crystal structure (no H) name heavy atoms only: their hydrogens must be
        # QM too, otherwise MOPAC sees radicals and the H stay MM behind 1-A "link atoms".
        # Explicit qm_indices are taken as given (the caller also fixes the matching MM set).
        added = _hydrogens_bonded_to(atom_rows, set(qm_idx), n)
        if added:
            qm_idx = sorted(set(qm_idx) | set(added))
            info["bonded_hydrogens_added"] = int(len(added))
    if info.get("selection_hint"):
        info["selection_hint"] = re.sub(r"\(n=\d+\)", f"(n={len(set(qm_idx))})", str(info["selection_hint"]))
    if explicit_mm:
        mm_idx = list(explicit_mm)
        info["mm_mode"] = "explicit_mm_indices"
    elif mm_atom_desc:
        mm_idx = sorted(set(int(i) for i in _resolve_atom_descriptors(atom_rows, mm_atom_desc, "mm_atom_descriptors")))
        info["mm_mode"] = "atom_descriptor_selection"
        info["mm_selection_hint"] = f"MM atom selection (n={len(mm_idx)})"
    elif mm_res_keys:
        mm_atoms = []
        for row in list(atom_rows or []):
            try:
                idx = int(row.get("index"))
            except Exception:
                continue
            if idx < 0 or idx >= n:
                continue
            key = _normalize_residue_key(row.get("chain", ""), row.get("resi", ""), row.get("resn", ""))
            if key in mm_res_keys:
                mm_atoms.append(idx)
        mm_idx = sorted(set(int(i) for i in mm_atoms))
        info["mm_mode"] = "residue_key_selection"
    else:
        qset = set(qm_idx)
        mm_idx = [i for i in range(n) if i not in qset]
        info["mm_mode"] = "complement"
    if forced_qm and not explicit_mm:
        qset = set(qm_idx)
        mm_idx = [i for i in mm_idx if int(i) not in qset]
    if added:
        # hydrogens moved to QM leave any MM selection that still lists them
        aset = set(int(i) for i in added)
        mm_idx = [i for i in mm_idx if int(i) not in aset]
    if not mm_idx:
        raise ValueError("MM region is empty.")
    if set(qm_idx).intersection(set(mm_idx)):
        raise ValueError("QM and MM regions overlap.")
    info["qm_atom_count"] = int(len(qm_idx))
    info["mm_atom_count"] = int(len(mm_idx))
    try:
        iface = qmmm_assess_interface(atom_rows=atom_rows, qm_indices=qm_idx, mm_indices=mm_idx)
        info["interface"] = dict(iface)
    except Exception:
        pass
    if not str(info.get("selection_hint", "") or "").strip():
        mode = str(info.get("mode", "") or "").strip()
        if mode == "explicit_qm_indices":
            info["selection_hint"] = f"QM explicit indices (n={len(qm_idx)})"
        elif mode == "residue_key_selection":
            info["selection_hint"] = f"QM residue selection (n={len(qm_idx)})"
        elif mode == "auto_qm_resnames":
            info["selection_hint"] = f"QM residue-name selection (n={len(qm_idx)})"
        elif mode == "auto_largest_hetero_residue":
            info["selection_hint"] = f"Auto hetero residue (n={len(qm_idx)})"
        else:
            info["selection_hint"] = f"QM region (n={len(qm_idx)})"
    return {"qm_indices": list(qm_idx), "mm_indices": list(mm_idx), "info": dict(info)}


def prepare_openmm_system_for_qmmm_coupling(system, qm_indices):
    """
    Remove the classical description of the QM region from an OpenMM ``System`` before force-coupled QM/MM MD.

    In force-coupled MD the QM atoms receive MOPAC forces (internal QM + electrostatic embedding) and the MM
    atoms the reaction of the QM charges.  If OpenMM still applies its own bonded terms and Coulomb forces to
    the QM atoms, those interactions are counted twice.  This function edits ``system`` in place:

    * NonbondedForce: charge 0 on QM atoms (removes classical QM–QM and QM–MM Coulomb, also under PME); QM–QM
      pairs become exceptions with no Coulomb and no Lennard-Jones (MOPAC describes them); existing QM–MM
      exceptions (1-4 pairs across the boundary) keep their Lennard-Jones but lose their charge product.
      QM–MM Lennard-Jones is kept on purpose: it is the steric repulsion MOPAC does not provide.
    * HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce: terms whose atoms are all QM get zero force
      constant.  Terms that cross the QM/MM boundary are kept and counted (there are no link atoms).
    * GBSAOBCForce: charge 0 on QM atoms.
    * Any other force type that touches QM atoms is reported in ``unhandled_forces``.

    Call it after ``createSystem`` and before building the ``Simulation``/``Context``.  Returns a report dict.
    """
    qset = sorted({int(i) for i in list(qm_indices or [])})
    if system is None:
        raise ValueError("An OpenMM System is required.")
    n = int(system.getNumParticles())
    if not qset:
        raise ValueError("qm_indices cannot be empty.")
    if qset[0] < 0 or qset[-1] >= n:
        raise ValueError("qm_indices out of range for this System.")
    qm = set(qset)
    report = {
        "qm_atom_count": len(qset),
        "qm_charges_zeroed": 0,
        "qm_total_classical_charge_e": 0.0,
        "qm_qm_exceptions": 0,
        "boundary_exceptions_uncharged": 0,
        "bonded_terms_removed": {},
        "boundary_bonded_terms_kept": {},
        "unhandled_forces": [],
    }
    for fi in range(int(system.getNumForces())):
        force = system.getForce(fi)
        name = str(force.__class__.__name__)
        if name == "NonbondedForce":
            for i in qset:
                q, sig, eps = force.getParticleParameters(i)
                report["qm_total_classical_charge_e"] += float(_omm_quantity_to_float(q))
                force.setParticleParameters(i, 0.0, sig, eps)
                report["qm_charges_zeroed"] += 1
            existing = {}
            for k in range(int(force.getNumExceptions())):
                a, b, qq, sig, eps = force.getExceptionParameters(k)
                existing[(min(a, b), max(a, b))] = k
                if (a in qm) != (b in qm):
                    force.setExceptionParameters(k, a, b, 0.0, sig, eps)
                    report["boundary_exceptions_uncharged"] += 1
            for x, i in enumerate(qset):
                for j in qset[x + 1:]:
                    key = (i, j)
                    if key in existing:
                        force.setExceptionParameters(existing[key], i, j, 0.0, 1.0, 0.0)
                    else:
                        force.addException(i, j, 0.0, 1.0, 0.0)
                    report["qm_qm_exceptions"] += 1
        elif name in ("HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"):
            removed = kept = 0
            if name == "HarmonicBondForce":
                for k in range(int(force.getNumBonds())):
                    a, b, r0, kb = force.getBondParameters(k)
                    inside = [a in qm, b in qm]
                    if all(inside):
                        force.setBondParameters(k, a, b, r0, 0.0)
                        removed += 1
                    elif any(inside):
                        kept += 1
            elif name == "HarmonicAngleForce":
                for k in range(int(force.getNumAngles())):
                    a, b, c, th0, ka = force.getAngleParameters(k)
                    inside = [a in qm, b in qm, c in qm]
                    if all(inside):
                        force.setAngleParameters(k, a, b, c, th0, 0.0)
                        removed += 1
                    elif any(inside):
                        kept += 1
            else:
                for k in range(int(force.getNumTorsions())):
                    a, b, c, d, per, ph, kt = force.getTorsionParameters(k)
                    inside = [a in qm, b in qm, c in qm, d in qm]
                    if all(inside):
                        force.setTorsionParameters(k, a, b, c, d, per, ph, 0.0)
                        removed += 1
                    elif any(inside):
                        kept += 1
            report["bonded_terms_removed"][name] = int(removed)
            report["boundary_bonded_terms_kept"][name] = int(kept)
        elif name == "GBSAOBCForce":
            for i in qset:
                _q, radius, scale = force.getParticleParameters(i)
                force.setParticleParameters(i, 0.0, radius, scale)
        elif name in ("CMMotionRemover", "MonteCarloBarostat", "MonteCarloAnisotropicBarostat",
                      "MonteCarloMembraneBarostat", "CustomExternalForce"):
            continue
        else:
            touches = True
            try:
                if hasattr(force, "getNumParticles") and int(force.getNumParticles()) == 0:
                    touches = False
            except Exception:
                pass
            if touches:
                report["unhandled_forces"].append(name)
    return report


def run_openmm_qmmm_single_point_probe(
    xyz_full_ang,
    atomic_numbers_full,
    mm_charges_e,
    atom_rows,
    config=None,
    workdir="",
    runner=None,
    qmmm_step_fn=None,
    bond_pairs=None,
    lj_sigma_nm=None,
    lj_epsilon_kj=None,
):
    """
    Run a single QM/MM evaluation via MOPAC and return correction force diagnostics.
    This function is backend-only and can be called by OpenMM workers.

    ``bond_pairs`` (topology bonds) makes covalent QM/MM boundaries exact for the hydrogen link atoms.
    ``lj_sigma_nm``/``lj_epsilon_kj`` (all atoms, from the OpenMM NonbondedForce) enable the QM–MM Lennard-Jones
    term; config ``include_lj`` (default: on when parameters are given) must be False in force-coupled MD, where
    OpenMM already evaluates it.  Config ``link_atoms`` (default True) switches link atoms off.
    """
    cfg = dict(config or {})
    xyz = _as_xyz_array(xyz_full_ang)
    z = [int(v) for v in list(np.asarray(atomic_numbers_full).reshape(-1))]
    if len(z) != int(xyz.shape[0]):
        raise ValueError("atomic_numbers_full and coordinates size mismatch.")
    mm_q_all = _as_1d_float(mm_charges_e, name="mm_charges_e")
    if int(mm_q_all.size) != int(xyz.shape[0]):
        raise ValueError("mm_charges_e length must match number of atoms.")

    atom_rows_xyz = []
    for row in list(atom_rows or []):
        one = dict(row or {})
        try:
            idx = int(one.get("index"))
        except Exception:
            idx = -1
        if 0 <= idx < int(xyz.shape[0]):
            one["x"] = float(xyz[idx, 0])
            one["y"] = float(xyz[idx, 1])
            one["z"] = float(xyz[idx, 2])
        atom_rows_xyz.append(one)
    reg = resolve_qmmm_regions(atom_rows=atom_rows_xyz, n_atoms=int(xyz.shape[0]), config=cfg)
    qm_idx = list(reg.get("qm_indices", []) or [])
    mm_idx = list(reg.get("mm_indices", []) or [])
    q_mm = np.asarray(mm_q_all[mm_idx], dtype=float).reshape(-1)

    operation = str(cfg.get("operation", "probe") or "probe").strip().lower()
    mopac_exe = str(cfg.get("mopac_executable", "mopac") or "mopac").strip()
    mopac_keywords = build_mopac_keywords_for_operation(
        operation=operation,
        base_keywords=str(cfg.get("mopac_keywords", "") or "").strip(),
    )
    qm_charge = int(cfg.get("qm_total_charge", 0))
    qm_mult = int(max(1, int(cfg.get("qm_multiplicity", 1))))
    dielectric = float(max(1.0e-12, float(cfg.get("dielectric", 1.0))))
    timeout_s = int(max(1, int(cfg.get("timeout_s", 120))))

    probe_dir = str(workdir or "").strip()
    if probe_dir:
        os.makedirs(probe_dir, exist_ok=True)
    step_fn = qmmm_step_fn if callable(qmmm_step_fn) else qmmm_step_file_based
    include_lj = bool(cfg.get("include_lj", lj_sigma_nm is not None and lj_epsilon_kj is not None))
    step_kwargs = dict(
        xyz_full_ang=xyz,
        atomic_numbers_full=z,
        qm_indices=qm_idx,
        mm_indices=mm_idx,
        q_mm=q_mm,
        mopac_executable=mopac_exe,
        mopac_keywords=mopac_keywords,
        qm_total_charge=qm_charge,
        qm_multiplicity=qm_mult,
        workdir=probe_dir,
        dielectric=dielectric,
        runner=runner,
        timeout_s=timeout_s,
        bond_pairs=bond_pairs,
        link_atoms=bool(cfg.get("link_atoms", True)),
        lj_sigma_nm=lj_sigma_nm,
        lj_epsilon_kj=lj_epsilon_kj,
        include_lj=include_lj,
    )
    try:
        import inspect

        accepted = inspect.signature(step_fn).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in accepted.values()):
            step_kwargs = {k: v for k, v in step_kwargs.items() if k in accepted}
    except (TypeError, ValueError):
        pass
    try:
        qmmm = step_fn(**step_kwargs)
    except Exception as exc:
        return {
            "ok": False,
            "operation": str(operation),
            "region": dict(reg.get("info", {}) or {}),
            "qm_indices": list(qm_idx),
            "mm_indices": list(mm_idx),
            "error": f"QM/MM backend failed: {exc}",
        }
    if not isinstance(qmmm, dict):
        return {
            "ok": False,
            "operation": str(operation),
            "region": dict(reg.get("info", {}) or {}),
            "qm_indices": list(qm_idx),
            "mm_indices": list(mm_idx),
            "error": "QM/MM backend returned a non-dict payload.",
        }
    if not bool(qmmm.get("ok", True)):
        reason = str(qmmm.get("error", "") or qmmm.get("reason", "") or "QM/MM backend reported failure.")
        return {
            "ok": False,
            "operation": str(operation),
            "region": dict(reg.get("info", {}) or {}),
            "qm_indices": list(qm_idx),
            "mm_indices": list(mm_idx),
            "qmmm": dict(qmmm),
            "error": str(reason),
        }

    try:
        f_corr = assemble_force_correction(
            n_atoms=int(xyz.shape[0]),
            qm_indices=qm_idx,
            mm_indices=mm_idx,
            f_mm_qm_internal=cfg.get("f_mm_qm_internal"),
            f_mm_qmmm_coul=cfg.get("f_mm_qmmm_coul"),
            f_qm_internal=qmmm.get("forces_qm_internal_kj_mol_nm"),
            f_qmmm_elec_qm=qmmm.get("forces_qmmm_elec_qm_kj_mol_nm"),
            f_qmmm_elec_mm=qmmm.get("forces_qmmm_elec_mm_kj_mol_nm"),
            f_link_mm=qmmm.get("forces_link_mm_kj_mol_nm"),
            f_lj_qm=qmmm.get("forces_lj_qm_kj_mol_nm") if qmmm.get("include_lj") else None,
            f_lj_mm=qmmm.get("forces_lj_mm_kj_mol_nm") if qmmm.get("include_lj") else None,
        )
        f_qm = np.asarray(qmmm.get("forces_qmmm_elec_qm_kj_mol_nm"), dtype=float)
        f_mm = np.asarray(qmmm.get("forces_qmmm_elec_mm_kj_mol_nm"), dtype=float)
    except Exception as exc:
        return {
            "ok": False,
            "operation": str(operation),
            "region": dict(reg.get("info", {}) or {}),
            "qm_indices": list(qm_idx),
            "mm_indices": list(mm_idx),
            "qmmm": dict(qmmm),
            "error": f"QM/MM force assembly failed: {exc}",
        }
    net_elec = np.sum(f_qm, axis=0) + np.sum(f_mm, axis=0)
    net_corr = np.sum(np.asarray(f_corr, dtype=float), axis=0)
    tol = float(max(1.0e-10, float(cfg.get("third_law_tolerance", 1.0e-6))))
    diag = {
        "qm_atom_count": int(len(qm_idx)),
        "mm_atom_count": int(len(mm_idx)),
        "net_electrostatic_force_norm": float(np.linalg.norm(net_elec)),
        "net_correction_force_norm": float(np.linalg.norm(net_corr)),
        "max_force_correction_norm": float(np.max(np.linalg.norm(f_corr, axis=1))) if int(f_corr.shape[0]) > 0 else 0.0,
        "third_law_tolerance": float(tol),
        "third_law_ok": bool(float(np.linalg.norm(net_elec)) <= tol),
    }
    return {
        "ok": True,
        "operation": str(operation),
        "region": dict(reg.get("info", {}) or {}),
        "qm_indices": list(qm_idx),
        "mm_indices": list(mm_idx),
        "qmmm": dict(qmmm or {}),
        "force_correction_kj_mol_nm": np.asarray(f_corr, dtype=float),
        "diagnostics": dict(diag),
    }


def run_openmm_qmmm_distance_scan_probe(
    xyz_full_ang,
    atomic_numbers_full,
    mm_charges_e,
    atom_rows,
    scan_pair,
    scan_start_ang,
    scan_end_ang,
    n_points=7,
    config=None,
    workdir="",
    runner=None,
    qmmm_step_fn=None,
    bond_pairs=None,
    lj_sigma_nm=None,
    lj_epsilon_kj=None,
):
    cfg = dict(config or {})
    xyz = _as_xyz_array(xyz_full_ang)
    if not isinstance(scan_pair, (list, tuple)) or len(scan_pair) < 2:
        raise ValueError("scan_pair must contain two atom indices.")
    i = int(scan_pair[0])
    j = int(scan_pair[1])
    n = int(xyz.shape[0])
    if i < 0 or i >= n or j < 0 or j >= n or i == j:
        raise ValueError("scan_pair atom indices are invalid.")
    start = float(scan_start_ang)
    end = float(scan_end_ang)
    npts = int(max(2, int(n_points)))
    dvals = np.linspace(start, end, npts)
    rows = []
    best = None
    base_vec = np.asarray(xyz[j, :] - xyz[i, :], dtype=float)
    base_norm = float(np.linalg.norm(base_vec))
    if base_norm <= 1.0e-12:
        base_vec = np.asarray([1.0, 0.0, 0.0], dtype=float)
        base_norm = 1.0
    direction = base_vec / base_norm
    op_cfg = dict(cfg)
    op_cfg["operation"] = "reaction_scan"
    for idx, dist in enumerate(list(dvals)):
        xyz_mod = np.asarray(xyz, dtype=float).copy()
        xyz_mod[j, :] = xyz_mod[i, :] + (direction * float(dist))
        pt_dir = str(workdir or "").strip()
        if pt_dir:
            pt_dir = os.path.join(pt_dir, f"scan_{idx+1:03d}")
        one = run_openmm_qmmm_single_point_probe(
            xyz_full_ang=xyz_mod,
            atomic_numbers_full=atomic_numbers_full,
            mm_charges_e=mm_charges_e,
            atom_rows=atom_rows,
            config=op_cfg,
            workdir=pt_dir,
            runner=runner,
            qmmm_step_fn=qmmm_step_fn,
            bond_pairs=bond_pairs,
            lj_sigma_nm=lj_sigma_nm,
            lj_epsilon_kj=lj_epsilon_kj,
        )
        one_ok = bool(one.get("ok", False))
        e_qm = _safe_float((one.get("qmmm", {}) or {}).get("energy_qm_hartree", np.nan))
        e_qmmm = _safe_float((one.get("qmmm", {}) or {}).get("energy_qmmm_elec_kj_mol", np.nan))
        row = {
            "point": int(idx + 1),
            "distance_ang": float(dist),
            "energy_qm_hartree": float(e_qm),
            "energy_qmmm_elec_kj_mol": float(e_qmmm),
            "ok": bool(one_ok),
            "error": str(one.get("error", "") or ""),
            "result": dict(one),
        }
        rows.append(row)
        if (not bool(one_ok)) or (not np.isfinite(e_qm)):
            continue
        if best is None or float(e_qm) < float(best.get("energy_qm_hartree", np.inf)):
            best = row
    if best is None:
        return {
            "ok": False,
            "scan_pair": [int(i), int(j)],
            "scan_start_ang": float(start),
            "scan_end_ang": float(end),
            "n_points": int(npts),
            "n_successful": int(sum(1 for row in rows if bool(row.get("ok", False)))),
            "n_failed": int(sum(1 for row in rows if not bool(row.get("ok", False)))),
            "rows": list(rows),
            "best_point": {},
            "error": "Reaction scan produced no successful QM/MM points.",
        }
    return {
        "ok": True,
        "scan_pair": [int(i), int(j)],
        "scan_start_ang": float(start),
        "scan_end_ang": float(end),
        "n_points": int(npts),
        "n_successful": int(sum(1 for row in rows if bool(row.get("ok", False)))),
        "n_failed": int(sum(1 for row in rows if not bool(row.get("ok", False)))),
        "rows": list(rows),
        "best_point": dict(best or {}),
    }


def _qmmm_embedded_model(xyz_full_ang, atomic_numbers_full, mm_charges_e, atom_rows, config, workdir,
                         bond_pairs=None, lj_sigma_nm=None, lj_epsilon_kj=None):
    """Resolve the QM/MM partition like the single-point probe and build an EmbeddedQMMMModel."""
    from .qmmm_mopac_runtime import _mopac_extract_method_from_keywords
    from .qmmm_reaction_runtime import EmbeddedQMMMModel

    cfg = dict(config or {})
    xyz = _as_xyz_array(xyz_full_ang)
    rows = []
    for row in list(atom_rows or []):
        one = dict(row or {})
        idx = int(one.get("index", -1))
        if 0 <= idx < int(xyz.shape[0]):
            one["x"], one["y"], one["z"] = (float(v) for v in xyz[idx])
        rows.append(one)
    reg = resolve_qmmm_regions(atom_rows=rows, n_atoms=int(xyz.shape[0]), config=cfg)
    keywords = str(cfg.get("mopac_keywords", "") or "")
    method = _mopac_extract_method_from_keywords(keywords, default="PM7")
    # The user's other keywords (UHF, MOZYME, CUTOFF=...) apply to every step, as in the single-point probe;
    # job-type keywords are set by the model and charge/spin by its own arguments.
    controls = {"1SCF", "GRAD", "AUX", "QMMM", "EF", "TS", "SADDLE", "FORCE", "FORCETS", "THERMO", "IRC",
                "X-PRIORITY", "XYZ", "CHARGE", "SINGLET", "DOUBLET", "TRIPLET", "QUARTET", "QUINTET"}
    extra = [tok for tok in keywords.split()
             if tok.upper() != method and tok.upper().split("=", 1)[0].split("(", 1)[0] not in controls]
    model = EmbeddedQMMMModel(
        xyz, atomic_numbers_full, reg["qm_indices"], reg["mm_indices"], mm_charges_e, workdir,
        mopac_executable=str(cfg.get("mopac_executable", "mopac") or "mopac"),
        method=method, extra_keywords=" ".join(extra),
        qm_total_charge=int(cfg.get("qm_total_charge", 0)), qm_multiplicity=int(max(1, int(cfg.get("qm_multiplicity", 1)))),
        bond_pairs=bond_pairs, lj_sigma_nm=lj_sigma_nm, lj_epsilon_kj=lj_epsilon_kj,
        dielectric=float(cfg.get("dielectric", 1.0)), timeout_s=int(max(1, int(cfg.get("timeout_s", 600)))),
    )
    return model, reg


def _write_qm_path(workdir, model, frames, comments, name):
    from .qmmm_mopac_runtime import _symbol_from_atomic_number

    path = os.path.join(workdir, name)
    with open(path, "w", encoding="utf-8") as fh:
        for xyz, c in zip(frames, comments):
            fh.write(f"{len(model.qm)}\n{c}\n")
            for i, p in zip(model.qm, np.asarray(xyz).reshape((-1, 3))):
                fh.write(f"{_symbol_from_atomic_number(model.z[i]):2s} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
    return path


def run_openmm_qmmm_relaxed_scan(
    xyz_full_ang,
    atomic_numbers_full,
    mm_charges_e,
    atom_rows,
    scan_pair,
    scan_start_ang,
    scan_end_ang,
    n_points=9,
    config=None,
    workdir="",
    bond_pairs=None,
    lj_sigma_nm=None,
    lj_epsilon_kj=None,
    k_kj_mol_ang2=None,
    transfer_atom=None,
):
    """
    Relaxed QM/MM scan inside the enzyme (see ``qmmm_reaction_runtime``): at every target value the whole QM region
    is optimized with the embedding recomputed at each step; the MM environment stays fixed.

    * Distance scan of d(i, j) with ``scan_pair`` = (i, j) and ``scan_start/end`` in Angstrom.
    * With ``transfer_atom`` = c, a **transfer-coordinate** scan xi = d(i, j) − d(j, c): atom/group j moves from i
      to c (recommended for proton/phosphoryl transfers; a one-distance scan shows false maxima there).  Then
      ``scan_start/end`` are xi values, e.g. −1.2 → +1.2 Å.

    Returns the single-point QM/MM probe at the **highest** point of the scan (the transition-state estimate)
    with ``scan`` holding every point; ``scan_path.xyz`` and ``scan_profile.csv`` are written in ``workdir``.
    """
    cfg = dict(config or {})
    run_dir = str(workdir or "").strip() or os.path.join(os.getcwd(), "qmmm_relaxed_scan")
    os.makedirs(run_dir, exist_ok=True)
    i, j = int(scan_pair[0]), int(scan_pair[1])
    try:
        model, reg = _qmmm_embedded_model(xyz_full_ang, atomic_numbers_full, mm_charges_e, atom_rows, cfg, run_dir,
                                          bond_pairs, lj_sigma_nm, lj_epsilon_kj)
        targets = np.linspace(float(scan_start_ang), float(scan_end_ang), int(max(2, int(n_points))))
        common = dict(fmax_kj_mol_ang=float(cfg.get("fmax_kj_mol_ang", 2.0)), steps=int(cfg.get("max_opt_steps", 300)))
        if transfer_atom is not None:
            k_r = float(cfg.get("scan_restraint_kj_mol_ang2", k_kj_mol_ang2 or 5000.0))
            scan = model.relaxed_transfer_scan(i, j, int(transfer_atom), targets, k_kj_mol_ang2=k_r, **common)
        else:
            k_r = float(cfg.get("scan_restraint_kj_mol_ang2", k_kj_mol_ang2 or 20000.0))
            scan = model.relaxed_distance_scan(i, j, targets, k_kj_mol_ang2=k_r, **common)
    except Exception as exc:
        return {"ok": False, "operation": "reaction_scan", "error": f"Relaxed QM/MM scan failed: {exc}"}
    rows = scan["rows"]
    coord = "xi_ang" if transfer_atom is not None else "distance_ang"
    target = "target_xi_ang" if transfer_atom is not None else "target_distance_ang"
    _write_qm_path(run_dir, model, [r["qm_xyz"] for r in rows],
                   [f"point={r['point']} {coord}={r[coord]:.4f} E_rel_kJ_mol={r['energy_rel_kj_mol']:.3f}" for r in rows],
                   "scan_path.xyz")
    with open(os.path.join(run_dir, "scan_profile.csv"), "w", encoding="utf-8") as fh:
        fh.write(f"point,{target},{coord},energy_kj_mol,energy_rel_kj_mol,converged,steps\n")
        for r in rows:
            fh.write(f"{r['point']},{r[target]:.4f},{r[coord]:.4f},{r['energy_kj_mol']:.4f},"
                     f"{r['energy_rel_kj_mol']:.4f},{int(r['converged'])},{r['steps']}\n")
    top = scan["highest_point"]
    probe_cfg = dict(cfg)
    probe_cfg.update({"operation": "probe", "qm_indices": list(model.qm), "mm_indices": list(model.mm)})
    one = run_openmm_qmmm_single_point_probe(
        model.full_coordinates(top["qm_xyz"]), atomic_numbers_full, mm_charges_e, atom_rows, config=probe_cfg,
        workdir=os.path.join(run_dir, "highest_point"), bond_pairs=bond_pairs, lj_sigma_nm=lj_sigma_nm,
        lj_epsilon_kj=lj_epsilon_kj)
    from .qmmm_reaction_runtime import path_tangent_mode

    k_top = int(top["point"]) - 1
    tangent = path_tangent_mode([r["qm_xyz"] for r in rows], k_top)
    ts_block = None
    if bool(cfg.get("scan_refine_ts", True)) and bool(scan["interior_maximum"]):
        # The highest point of a relaxed scan and the path tangent there are the best dimer start (the tangent,
        # not a single bond direction, points along the reaction mode of all QM atoms).
        try:
            ts = model.refine_ts_dimer(top["qm_xyz"], mode_guess=tangent,
                                       fmax_kj_mol_ang=float(cfg.get("fmax_kj_mol_ang", 2.0)),
                                       steps=int(cfg.get("max_opt_steps", 300)))
            freq = model.vibrations(ts["qm_xyz"]) if (ts["converged"] and bool(cfg.get("ts_frequencies", True))) else None
            _write_qm_path(run_dir, model, [ts["qm_xyz"]], [f"TS E_kJ_mol={ts['energy_kj_mol']:.3f}"], "ts.xyz")
            ts_block = {
                "converged": bool(ts["converged"]), "stopped_reason": ts["stopped_reason"], "steps": int(ts["steps"]),
                "energy_kj_mol": float(ts["energy_kj_mol"]),
                # from the reactant minimum of the scan (not its first point)
                "barrier_kj_mol": (float(ts["energy_kj_mol"] - scan["reactant_energy_kj_mol"]) if ts["converged"]
                                   else None),
                "curvature_ev_ang2": float(ts["curvature_ev_ang2"]),
                "frequencies_cm1": None if freq is None else [float(v) for v in np.sort(freq["freq_cm_signed"])[:8]],
                "frequency_validation": None if freq is None else dict(freq["validation"]),
                "full_xyz_ang": model.full_coordinates(ts["qm_xyz"]), "ts_xyz": os.path.join(run_dir, "ts.xyz"),
            }
        except Exception as exc:
            ts_block = {"converged": False, "stopped_reason": f"TS refinement failed: {exc}"}
    public_rows = [{k: v for k, v in r.items() if k != "qm_xyz"} for r in rows]
    one["operation"] = "reaction_scan"
    if ts_block is not None:
        one["ts"] = ts_block
    one["scan"] = {
        "relaxed": True, "coordinate": ("transfer_xi" if transfer_atom is not None else "distance"),
        "scan_pair": [i, j], "transfer_atom": (None if transfer_atom is None else int(transfer_atom)),
        "scan_start_ang": float(scan_start_ang), "scan_end_ang": float(scan_end_ang),
        "n_points": len(rows), "n_successful": int(sum(1 for r in rows if r["converged"])),
        "rows": public_rows, "barrier_estimate_kj_mol": float(scan["barrier_estimate_kj_mol"]),
        "barrier_from_first_point_kj_mol": float(scan["barrier_from_first_point_kj_mol"]),
        "reaction_energy_kj_mol": float(scan["reaction_energy_kj_mol"]),
        "reactant_point": int(scan["reactant_point"]["point"]),
        "highest_point": {k: v for k, v in top.items() if k != "qm_xyz"},
        "interior_maximum": bool(scan["interior_maximum"]),
        "n_energy_evaluations": int(scan["n_energy_evaluations"]),
        "path_xyz": os.path.join(run_dir, "scan_path.xyz"), "profile_csv": os.path.join(run_dir, "scan_profile.csv"),
        "ts_guess_full_xyz_ang": model.full_coordinates(top["qm_xyz"]),
        "ts_mode_guess_qm": np.asarray(tangent, dtype=float),
    }
    return one


def run_openmm_qmmm_ts_search(
    xyz_full_ang,
    atomic_numbers_full,
    mm_charges_e,
    atom_rows,
    config=None,
    workdir="",
    mode_pair=None,
    run_frequencies=True,
    bond_pairs=None,
    lj_sigma_nm=None,
    lj_epsilon_kj=None,
    mode_guess_qm=None,
):
    """
    Transition-state search inside the enzyme with the dimer method (embedding recomputed at every step), started
    from ``xyz_full_ang`` (e.g. the highest point of a relaxed scan).  The reaction direction is ``mode_guess_qm``
    (n_QM × 3, best: the ``ts_mode_guess_qm`` tangent returned by the relaxed scan) or, as a weaker fallback,
    ``mode_pair`` = (i, j), the bond being formed/broken.  Numerical frequencies then check for exactly one imaginary
    mode.  If the dimer finds no negative curvature it stops and reports why (``ts.stopped_reason``).
    """
    from .qmmm_reaction_runtime import bond_direction_mode

    cfg = dict(config or {})
    run_dir = str(workdir or "").strip() or os.path.join(os.getcwd(), "qmmm_ts_search")
    os.makedirs(run_dir, exist_ok=True)
    try:
        model, reg = _qmmm_embedded_model(xyz_full_ang, atomic_numbers_full, mm_charges_e, atom_rows, cfg, run_dir,
                                          bond_pairs, lj_sigma_nm, lj_epsilon_kj)
        if mode_guess_qm is not None:
            mode = np.asarray(mode_guess_qm, dtype=float).reshape((len(model.qm), 3))
        else:
            mode = bond_direction_mode(model, *mode_pair) if mode_pair else None
        ts = model.refine_ts_dimer(model.xyz_ref[model.qm], mode_guess=mode,
                                   fmax_kj_mol_ang=float(cfg.get("fmax_kj_mol_ang", 2.0)),
                                   steps=int(cfg.get("max_opt_steps", 300)))
        freq = model.vibrations(ts["qm_xyz"]) if (bool(run_frequencies) and ts["converged"]) else None
    except Exception as exc:
        return {"ok": False, "operation": "transition_state", "error": f"QM/MM transition-state search failed: {exc}"}
    _write_qm_path(run_dir, model, [ts["qm_xyz"]], [f"TS E_kJ_mol={ts['energy_kj_mol']:.3f}"], "ts.xyz")
    probe_cfg = dict(cfg)
    probe_cfg.update({"operation": "probe", "qm_indices": list(model.qm), "mm_indices": list(model.mm)})
    one = run_openmm_qmmm_single_point_probe(
        model.full_coordinates(ts["qm_xyz"]), atomic_numbers_full, mm_charges_e, atom_rows, config=probe_cfg,
        workdir=os.path.join(run_dir, "ts_point"), bond_pairs=bond_pairs, lj_sigma_nm=lj_sigma_nm,
        lj_epsilon_kj=lj_epsilon_kj)
    one["operation"] = "transition_state"
    one["ts"] = {
        "converged": bool(ts["converged"]), "stopped_reason": ts["stopped_reason"], "steps": int(ts["steps"]),
        "energy_kj_mol": float(ts["energy_kj_mol"]),
        "curvature_ev_ang2": float(ts["curvature_ev_ang2"]), "max_force_kj_mol_ang": float(ts["max_force_kj_mol_ang"]),
        "frequencies_cm1": None if freq is None else [float(v) for v in np.sort(freq["freq_cm_signed"])[:8]],
        "frequency_validation": None if freq is None else dict(freq["validation"]),
        "ts_xyz": os.path.join(run_dir, "ts.xyz"), "full_xyz_ang": model.full_coordinates(ts["qm_xyz"]),
        "n_energy_evaluations": int(model.n_calls),
    }
    if not ts["converged"]:
        one.setdefault("warnings", []).append(f"Transition-state search did not converge: {ts['stopped_reason'] or 'step limit reached'}")
    if freq is not None and not bool(freq["validation"].get("ok", False)):
        one.setdefault("warnings", []).append(
            f"The stationary point has {freq['validation'].get('imaginary_mode_count')} imaginary modes (a TS needs exactly one).")
    return one


__all__ = [
    "classify_residue_kind",
    "is_standard_openmm_small_molecule_resn",
    "openmm_component_uid",
    "classify_openmm_component_ff_class",
    "default_openmm_xml_selected_keys",
    "split_openmm_parameterization_targets",
    "read_openmm_forcefield_residue_names",
    "dedupe_openmm_forcefield_xml_paths",
    "resolve_openmm_component_template_plan",
    "sanitize_pdb_conect_for_standalone_components",
    "rewrite_pdb_component_identity",
    "normalize_pdb_conect_records",
    "compose_openmm_input_pdb",
    "merge_openmm_periodic_torsion_rows",
    "build_openmm_parameterization_audit",
    "export_forcefield_xml_from_amber_prmtop",
    "summarize_openmm_system_terms",
    "compare_openmm_system_term_summaries",
    "assess_ligand_topology_signature_compatibility",
    "build_mopac_keywords_for_operation",
    "resolve_qmmm_regions",
    "prepare_openmm_system_for_qmmm_coupling",
    "run_openmm_qmmm_single_point_probe",
    "run_openmm_qmmm_distance_scan_probe",
    "run_openmm_qmmm_relaxed_scan",
    "run_openmm_qmmm_ts_search",
]
