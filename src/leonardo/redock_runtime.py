import csv
import datetime
import inspect
import os
import re
from math import ceil, isfinite


def _safe_float(value, default=float("nan")):
    try:
        out = float(value)
    except Exception:
        return float(default)
    return float(out) if isfinite(out) else float(default)


def _has_explicit_redocking_box(center, size):
    try:
        cx = float(center[0])
        cy = float(center[1])
        cz = float(center[2])
        sx = float(size[0])
        sy = float(size[1])
        sz = float(size[2])
    except Exception:
        return False
    return (
        isfinite(cx)
        and isfinite(cy)
        and isfinite(cz)
        and isfinite(sx)
        and isfinite(sy)
        and isfinite(sz)
        and sx > 0.0
        and sy > 0.0
        and sz > 0.0
    )


def pick_best_pose_row(rows):
    rows_local = list(rows or [])
    if not rows_local:
        return None
    with_rmsd = []
    for row in rows_local:
        rv = _safe_float((row or {}).get("rmsd_ref", float("nan")))
        if isfinite(rv):
            with_rmsd.append(row)
    if with_rmsd:
        return min(
            with_rmsd,
            key=lambda r: (
                _safe_float((r or {}).get("rmsd_ref", float("inf")), default=float("inf")),
                _safe_float((r or {}).get("affinity", float("inf")), default=float("inf")),
            ),
        )
    return min(
        rows_local,
        key=lambda r: _safe_float((r or {}).get("affinity", float("inf")), default=float("inf")),
    )


REDOCK_SUCCESS_RMSD_A = 2.0


def pick_top_scored_pose_row(rows):
    """
    The pose a redocking run actually predicts: best docking score (lowest affinity),
    ties by mode rank. Pose recovery is judged on this pose (top-1 RMSD <= 2 A); the
    lowest-RMSD pose among all modes (pick_best_pose_row) needs the crystal answer to be
    picked and is only a sampling measure.
    """
    rows_local = [r for r in list(rows or []) if isinstance(r, dict)]
    if not rows_local:
        return None

    def _key(r):
        aff = _safe_float(r.get("affinity", float("inf")), default=float("inf"))
        mode = _safe_float(r.get("mode", float("inf")), default=float("inf"))
        return (aff if isfinite(aff) else float("inf"), mode if isfinite(mode) else float("inf"))

    return min(rows_local, key=_key)


def is_better_repeat(new_best, cur_best, eps=1.0e-6):
    """Repeat comparison on the top-scored pose: better docking score, then lower RMSD."""
    if new_best is None:
        return False
    if cur_best is None:
        return True
    na = _safe_float((new_best or {}).get("affinity", float("nan")))
    ca = _safe_float((cur_best or {}).get("affinity", float("nan")))
    if isfinite(na) and not isfinite(ca):
        return True
    if isfinite(ca) and not isfinite(na):
        return False
    if isfinite(na) and isfinite(ca) and abs(na - ca) > float(eps):
        return na < ca
    nr = _safe_float((new_best or {}).get("rmsd_ref", float("nan")))
    cr = _safe_float((cur_best or {}).get("rmsd_ref", float("nan")))
    if isfinite(nr) and isfinite(cr):
        return nr < (cr - float(eps))
    return isfinite(nr) and not isfinite(cr)


def format_failed_repeats_tail(repeat_rows, max_rows=3, max_chars=220):
    tail = ""
    rows = list(repeat_rows or [])
    nrows = int(max(1, max_rows))
    nchars = int(max(20, max_chars))
    for rr in rows[-nrows:]:
        err_txt = str((rr or {}).get("error", "") or "")
        if err_txt:
            shown = str(err_txt)
            if len(shown) > nchars:
                shown = "..." + shown[-(nchars - 3):]
            tail += f" | repeat {int((rr or {}).get('repeat', 0) or 0)}: {shown}"
    return tail


def build_redocking_pose_object_name(receptor_obj, tag="", safe_name_fn=None, prefix="redock"):
    rec = str(receptor_obj or "")
    rec_safe = str(rec)
    if callable(safe_name_fn):
        try:
            rec_safe = str(safe_name_fn(rec) or rec)
        except Exception:
            rec_safe = str(rec)
    return f"{str(prefix or 'redock')}_{rec_safe}_{str(tag or '')}"


def resolve_redocking_pose_view_file(
    best_pose_file,
    convert_pdbqt_to_sdf_fn=None,
    convert_pdbqt_to_pdb_fn=None,
):
    pose_file = str(best_pose_file or "")
    pose_view = str(pose_file)
    if str(pose_file).lower().endswith(".pdbqt"):
        base_dir = os.path.dirname(str(pose_file))
        view_sdf = os.path.join(base_dir, "docked_poses_view.sdf")
        ok_sdf = False
        if callable(convert_pdbqt_to_sdf_fn):
            try:
                ok_sdf = bool(convert_pdbqt_to_sdf_fn(pose_file, view_sdf))
            except Exception:
                ok_sdf = False
        if ok_sdf:
            pose_view = str(view_sdf)
        else:
            view_pdb = os.path.join(base_dir, "docked_poses_view.pdb")
            ok_pdb = False
            if callable(convert_pdbqt_to_pdb_fn):
                try:
                    ok_pdb = bool(convert_pdbqt_to_pdb_fn(pose_file, view_pdb))
                except Exception:
                    ok_pdb = False
            if ok_pdb:
                pose_view = str(view_pdb)
    return str(pose_view)


def _parse_redocking_pose_rows_from_text(text):
    rows = []
    seen_modes = set()
    for raw in str(text or "").splitlines():
        m = re.match(
            r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$",
            str(raw).strip(),
        )
        if not m:
            continue
        mode = int(m.group(1))
        if mode in seen_modes:
            continue
        seen_modes.add(mode)
        rows.append(
            {
                "mode": int(mode),
                "affinity": float(m.group(2)),
                "rmsd_lb": float(m.group(3)),
                "rmsd_ub": float(m.group(4)),
            }
        )
    return rows


def _collect_redocking_output_text(log_file, out="", err=""):
    chunks = []
    try:
        if log_file and os.path.isfile(log_file):
            with open(log_file, "r", encoding="utf-8", errors="ignore") as fh:
                txt = str(fh.read() or "")
            if txt.strip():
                chunks.append(txt)
    except Exception:
        pass
    for txt in (out, err):
        st = str(txt or "")
        if st.strip():
            chunks.append(st)
    return "\n".join(chunks)


def _redocking_needs_flex_search_space_retry(text):
    low = str(text or "").lower()
    return (
        "could not find any conformations completely within the search space" in low
        or (
            "search space" in low
            and "flexible side chains" in low
        )
    )


def _redocking_extract_cmd_arg(cmdline, flag):
    cmd = list(cmdline or [])
    try:
        idx = cmd.index(str(flag))
    except Exception:
        return ""
    try:
        return str(cmd[idx + 1] or "").strip()
    except Exception:
        return ""


def _read_redocking_structure_atoms(path):
    atoms = []
    sp = str(path or "").strip()
    if (not sp) or (not os.path.isfile(sp)):
        return atoms
    try:
        with open(sp, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                ln = str(raw or "")
                if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
                    continue
                try:
                    x = float(ln[30:38])
                    y = float(ln[38:46])
                    z = float(ln[46:54])
                except Exception:
                    continue
                atoms.append(
                    {
                        "chain": str(ln[21:22] or "").strip(),
                        "resi": str(ln[22:27] or "").strip(),
                        "resn": str(ln[17:20] or "").strip().upper(),
                        "coord": (float(x), float(y), float(z)),
                    }
                )
    except Exception:
        return []
    return list(atoms)


def _parse_redocking_flexres_spec(native_flexres):
    rows = []
    seen = set()
    for raw in str(native_flexres or "").split(","):
        tok = str(raw or "").strip()
        if not tok:
            continue
        parts = tok.split(":")
        chain = ""
        resi = ""
        if len(parts) >= 2:
            chain = str(parts[0] or "").strip()
            resi = "".join(str(part or "").strip() for part in parts[1:])
        else:
            resi = str(parts[0] or "").strip()
        if not resi:
            continue
        key = (str(chain), str(resi))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"chain": str(chain), "resi": str(resi)})
    return list(rows)


def _collect_redocking_flex_coords(
    *,
    receptor_structure="",
    native_flexres="",
    flex_file="",
):
    if str(flex_file or "").strip():
        return [tuple(atom.get("coord", ())) for atom in list(_read_redocking_structure_atoms(flex_file) or []) if tuple(atom.get("coord", ()))]
    receptor_path = str(receptor_structure or "").strip()
    spec_rows = list(_parse_redocking_flexres_spec(native_flexres) or [])
    if (not receptor_path) or (not spec_rows):
        return []
    spec_keys = set((str(r.get("chain", "") or "").strip(), str(r.get("resi", "") or "").strip()) for r in list(spec_rows or []))
    out = []
    for atom in list(_read_redocking_structure_atoms(receptor_path) or []):
        key = (str(atom.get("chain", "") or "").strip(), str(atom.get("resi", "") or "").strip())
        if key in spec_keys:
            out.append(tuple(atom.get("coord", ())))
    return list(out)


def _estimate_redocking_geometry_autobox_add(
    *,
    autobox_ligand="",
    receptor_structure="",
    native_flexres="",
    flex_file="",
    margin_a=2.5,
    cap=40.0,
):
    lig_atoms = list(_read_redocking_structure_atoms(autobox_ligand) or [])
    lig_coords = [tuple(atom.get("coord", ())) for atom in lig_atoms if tuple(atom.get("coord", ()))]
    if not lig_coords:
        return float("nan")
    flex_coords = list(
        _collect_redocking_flex_coords(
            receptor_structure=str(receptor_structure or ""),
            native_flexres=str(native_flexres or ""),
            flex_file=str(flex_file or ""),
        )
        or []
    )
    if not flex_coords:
        return float("nan")
    try:
        margin = float(margin_a)
    except Exception:
        margin = 2.5
    if (not isfinite(margin)) or margin < 0.0:
        margin = 2.5
    try:
        cap_val = float(cap)
    except Exception:
        cap_val = 40.0
    if (not isfinite(cap_val)) or cap_val <= 0.0:
        cap_val = 40.0

    lig_min = [min(float(c[i]) for c in list(lig_coords or [])) for i in range(3)]
    lig_max = [max(float(c[i]) for c in list(lig_coords or [])) for i in range(3)]
    need = 0.0
    for coord in list(flex_coords or []):
        if len(coord) < 3:
            continue
        for axis in range(3):
            try:
                vv = float(coord[axis])
            except Exception:
                continue
            low_over = max(0.0, float(lig_min[axis]) - float(vv))
            high_over = max(0.0, float(vv) - float(lig_max[axis]))
            need = max(float(need), float(low_over), float(high_over))
    if need <= 0.0:
        return float(margin)
    return float(min(float(cap_val), float(need) + float(margin)))


def _estimate_redocking_flex_autobox_min(cmdline, default_min=10.0, cap=40.0):
    cmd = list(cmdline or [])
    target = float(default_min)
    if "--flexres" in cmd:
        try:
            spec = str(cmd[cmd.index("--flexres") + 1] or "").strip()
        except Exception:
            spec = ""
        n_res = len([tok for tok in spec.split(",") if str(tok).strip()])
        if n_res > 0:
            target = max(float(target), 6.0 + (1.5 * float(n_res)))
    geom_target = _estimate_redocking_geometry_autobox_add(
        autobox_ligand=_redocking_extract_cmd_arg(cmd, "--autobox_ligand"),
        receptor_structure=(
            _redocking_extract_cmd_arg(cmd, "-r")
            or _redocking_extract_cmd_arg(cmd, "--receptor")
        ),
        native_flexres=_redocking_extract_cmd_arg(cmd, "--flexres"),
        flex_file=_redocking_extract_cmd_arg(cmd, "--flex"),
        cap=float(cap),
    )
    if isfinite(float(geom_target)):
        target = max(float(target), float(geom_target))
    return float(min(float(cap), float(target)))


def estimate_redocking_initial_autobox_add(
    base_autobox_add=4.0,
    *,
    native_flexres="",
    flex_file="",
    autobox_ligand="",
    receptor_structure="",
    min_autobox_add=10.0,
    per_residue_add=1.5,
    max_autobox_add=40.0,
    geometry_margin_a=2.5,
):
    try:
        base_add = float(base_autobox_add)
    except Exception:
        base_add = 4.0
    try:
        min_add = float(min_autobox_add)
    except Exception:
        min_add = 10.0
    try:
        per_res = float(per_residue_add)
    except Exception:
        per_res = 1.5
    try:
        max_add = float(max_autobox_add)
    except Exception:
        max_add = 26.0
    target = float(base_add)
    native_spec = str(native_flexres or "").strip()
    if native_spec:
        n_res = len([tok for tok in native_spec.split(",") if str(tok).strip()])
        if n_res > 0:
            target = max(float(target), min_add, 6.0 + (per_res * float(n_res)))
    elif str(flex_file or "").strip():
        target = max(float(target), min_add)
    geom_target = _estimate_redocking_geometry_autobox_add(
        autobox_ligand=str(autobox_ligand or ""),
        receptor_structure=str(receptor_structure or ""),
        native_flexres=str(native_spec),
        flex_file=str(flex_file or ""),
        margin_a=float(geometry_margin_a),
        cap=float(max_add),
    )
    if isfinite(float(geom_target)):
        target = max(float(target), float(geom_target))
    return float(min(max_add, target))


def _expand_redocking_flex_autobox_cmdline(cmdline, min_autobox_add=10.0, delta=6.0, max_autobox_add=26.0):
    cmd = list(cmdline or [])
    if ("--flex" not in cmd) and ("--flexres" not in cmd):
        return None, None
    if "--autobox_ligand" not in cmd:
        return None, None
    out = list(cmd)
    target_min = _estimate_redocking_flex_autobox_min(out, default_min=float(min_autobox_add), cap=float(max_autobox_add))
    new_add = float(target_min)
    if "--autobox_add" in out:
        idx = out.index("--autobox_add")
        try:
            cur_add = float(out[idx + 1])
        except Exception:
            cur_add = 0.0
        new_add = float(min(float(max_autobox_add), max(float(target_min), float(cur_add) + float(delta))))
        if new_add <= float(cur_add) + 1.0e-6:
            return None, None
        out[idx + 1] = f"{new_add:.3f}"
    else:
        out.extend(["--autobox_add", f"{new_add:.3f}"])
    return out, float(new_add)


def strip_redocking_residue_from_structure_file(in_file, out_file, chain, resi, resn, fallback_by_resn=True):
    """
    Remove a residue (typically reference ligand) from prepared receptor files (PDB/PDBQT).
    Returns number of removed ATOM/HETATM records.

    If no record matches chain/resi/resn (receptor preparation renumbered or re-chained the
    ligand) but records of that residue name are present, they are removed by name, so the
    ligand is never docked into a site that still holds the crystal ligand.
    """
    removed = _strip_redocking_residue_records(in_file, out_file, chain, resi, resn)
    if removed == 0 and bool(fallback_by_resn) and str(resn or "").strip():
        removed = _strip_redocking_residue_records(in_file, out_file, "", None, resn)
    return int(removed)


def _strip_redocking_residue_records(in_file, out_file, chain, resi, resn):
    """resi=None matches any residue number (by residue name, and chain if given)."""
    inp = str(in_file or "").strip()
    outp = str(out_file or "").strip()
    if (not inp) or (not os.path.isfile(inp)) or (not outp):
        return 0
    resn_u = str(resn or "").strip().upper()
    resi_s = str(resi or "").strip() if resi is not None else ""
    chain_s = str(chain or "").strip()
    removed = 0
    lines_out = []
    try:
        with open(inp, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        l_resn = str(line[17:20]).strip().upper()
                        l_chain = str(line[21:22]).strip()
                        l_resi = str(line[22:27]).strip()
                    except Exception:
                        parts = line.split()
                        l_resn = str(parts[3]).strip().upper() if len(parts) > 3 else ""
                        l_chain = str(parts[4]).strip() if len(parts) > 4 else ""
                        l_resi = str(parts[5]).strip() if len(parts) > 5 else ""
                    same = bool(
                        l_resn == resn_u
                        and (resi is None or l_resi == resi_s)
                        and ((not chain_s) or (l_chain == chain_s))
                    )
                    if same:
                        removed += 1
                        continue
                lines_out.append(line)
        with open(outp, "w", encoding="utf-8") as fh:
            for line in lines_out:
                fh.write(str(line) + "\n")
    except Exception:
        return 0
    return int(removed)


def _redocking_pdb_atom_coord(line):
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except Exception:
        parts = str(line or "").split()
        if len(parts) >= 9:
            try:
                return (float(parts[6]), float(parts[7]), float(parts[8]))
            except Exception:
                return None
    return None


def _redocking_reference_coords_from_pdb(path):
    coords = []
    p = str(path or "").strip()
    if (not p) or (not os.path.isfile(p)):
        return coords
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                if not (raw.startswith("ATOM") or raw.startswith("HETATM")):
                    continue
                c = _redocking_pdb_atom_coord(raw)
                if c is not None:
                    coords.append(c)
    except Exception:
        return []
    return coords


def build_redocking_receptor_structure_qc(
    structure_file,
    *,
    reference_ligand_file="",
    site_cutoff_a=8.0,
):
    """
    Lightweight receptor-input QC for docking/redocking.

    It detects PDB-level issues that commonly corrupt docking interpretation:
    missing residue/atom REMARKs, alternate locations, and retained HETATM
    classes (metals, cofactors, waters). When a reference ligand PDB is supplied,
    altloc atoms near the binding site are counted separately.
    """

    path = str(structure_file or "").strip()
    warnings = []
    errors = []
    notes = []
    atom_count = 0
    hetatm_count = 0
    water_count = 0
    metal_count = 0
    cofactor_count = 0
    altloc_count = 0
    altloc_residues = set()
    altloc_near_site = 0
    missing_residue_lines = []
    missing_atom_lines = []
    metal_resns = {
        "LI",
        "NA",
        "K",
        "RB",
        "CS",
        "MG",
        "CA",
        "SR",
        "BA",
        "MN",
        "FE",
        "CO",
        "NI",
        "CU",
        "ZN",
        "CD",
        "HG",
    }
    water_resns = {"HOH", "WAT", "H2O", "DOD"}
    ref_coords = _redocking_reference_coords_from_pdb(reference_ligand_file)
    cutoff2 = float(site_cutoff_a) * float(site_cutoff_a)

    if (not path) or (not os.path.isfile(path)):
        return {
            "ok": False,
            "status": "red",
            "errors": ["receptor structure file is missing"],
            "warnings": [],
            "notes": [],
            "structure_file": path,
        }

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if line.startswith("REMARK 465"):
                    if len(line.strip()) > 10:
                        missing_residue_lines.append(line.strip())
                    continue
                if line.startswith("REMARK 470"):
                    if len(line.strip()) > 10:
                        missing_atom_lines.append(line.strip())
                    continue
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                atom_count += 1
                resn = str(line[17:20]).strip().upper()
                chain = str(line[21:22]).strip()
                resi = str(line[22:27]).strip()
                alt = str(line[16:17]).strip()
                if alt:
                    altloc_count += 1
                    altloc_residues.add(f"{chain}:{resi}:{resn}")
                    if ref_coords:
                        c = _redocking_pdb_atom_coord(line)
                        if c is not None:
                            x, y, z = c
                            for rx, ry, rz in ref_coords:
                                dx = x - rx
                                dy = y - ry
                                dz = z - rz
                                if (dx * dx + dy * dy + dz * dz) <= cutoff2:
                                    altloc_near_site += 1
                                    break
                if line.startswith("HETATM"):
                    hetatm_count += 1
                    if resn in water_resns:
                        water_count += 1
                    elif resn in metal_resns:
                        metal_count += 1
                    else:
                        cofactor_count += 1
    except Exception as exc:
        return {
            "ok": False,
            "status": "red",
            "errors": [str(exc)],
            "warnings": [],
            "notes": [],
            "structure_file": path,
        }

    if atom_count <= 0:
        errors.append("receptor structure has no ATOM/HETATM records")
    if missing_residue_lines:
        warnings.append(f"PDB reports missing residues ({len(missing_residue_lines)} REMARK 465 lines)")
    if missing_atom_lines:
        warnings.append(f"PDB reports missing atoms ({len(missing_atom_lines)} REMARK 470 lines)")
    if altloc_count > 0:
        warnings.append(f"alternate locations present ({altloc_count} atoms across {len(altloc_residues)} residues)")
    if altloc_near_site > 0:
        warnings.append(f"alternate locations near reference ligand ({altloc_near_site} atoms within {float(site_cutoff_a):.1f} A)")
    if metal_count > 0:
        notes.append(f"metals retained: {metal_count}")
    if cofactor_count > 0:
        notes.append(f"non-water cofactors/ligands retained: {cofactor_count}")
    if water_count > 0:
        notes.append(f"waters retained: {water_count}")

    status = "green"
    if errors:
        status = "red"
    elif warnings:
        status = "yellow"
    return {
        "ok": status != "red",
        "status": status,
        "structure_file": path,
        "reference_ligand_file": str(reference_ligand_file or ""),
        "site_cutoff_a": float(site_cutoff_a),
        "atom_count": int(atom_count),
        "hetatm_count": int(hetatm_count),
        "water_count": int(water_count),
        "metal_count": int(metal_count),
        "cofactor_count": int(cofactor_count),
        "altloc_count": int(altloc_count),
        "altloc_near_site": int(altloc_near_site),
        "altloc_residues": sorted(altloc_residues),
        "missing_residue_remark_count": int(len(missing_residue_lines)),
        "missing_atom_remark_count": int(len(missing_atom_lines)),
        "missing_residue_examples": missing_residue_lines[:5],
        "missing_atom_examples": missing_atom_lines[:5],
        "warnings": list(warnings),
        "errors": list(errors),
        "notes": list(notes),
    }


def invoke_prepare_flexible_receptor(
    prepare_flexible_receptor_fn,
    receptor_for_dock,
    job_dir,
    flex_rows,
    *,
    prepared_pdb="",
    rec_info=None,
):
    if not callable(prepare_flexible_receptor_fn):
        raise TypeError("prepare_flexible_receptor_fn is not callable")
    kwargs = {}
    prepared = str(prepared_pdb or "").strip()
    try:
        sig = inspect.signature(prepare_flexible_receptor_fn)
    except Exception:
        sig = None
    if sig is not None:
        params = sig.parameters
        supports_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if supports_kwargs or ("prepared_pdb" in params):
            kwargs["prepared_pdb"] = str(prepared)
        if supports_kwargs or ("rec_info" in params):
            kwargs["rec_info"] = dict(rec_info or {}) if isinstance(rec_info, dict) else rec_info
    return prepare_flexible_receptor_fn(
        receptor_for_dock,
        job_dir,
        flex_rows,
        **kwargs,
    )


def build_redocking_chainwise_ca_alignment(
    ref_trace,
    rec_trace,
    *,
    sequence_pairs_fn=None,
    min_chain_matches=8,
    min_total_matches=30,
    min_chain_coverage=0.35,
    min_contiguous_block=6,
):
    payload = {
        "ok": False,
        "pairs": [],
        "chain_pairs": [],
        "matched_ca": 0,
        "identity_count": 0,
        "mismatch_count": 0,
        "coverage": 0.0,
        "error": "",
    }
    if not callable(sequence_pairs_fn):
        payload["error"] = "sequence_pairs_fn is not callable"
        return payload

    ref_rows = [dict(r or {}) for r in list(ref_trace or []) if isinstance(r, dict)]
    rec_rows = [dict(r or {}) for r in list(rec_trace or []) if isinstance(r, dict)]
    if not ref_rows:
        payload["error"] = "reference trace is empty"
        return payload
    if not rec_rows:
        payload["error"] = "receptor trace is empty"
        return payload

    try:
        min_chain_matches_i = int(min_chain_matches)
    except Exception:
        min_chain_matches_i = 8
    try:
        min_total_matches_i = int(min_total_matches)
    except Exception:
        min_total_matches_i = 30
    try:
        min_chain_coverage_f = float(min_chain_coverage)
    except Exception:
        min_chain_coverage_f = 0.35
    try:
        min_contiguous_block_i = int(min_contiguous_block)
    except Exception:
        min_contiguous_block_i = 6
    min_chain_matches_i = max(1, int(min_chain_matches_i))
    min_total_matches_i = max(1, int(min_total_matches_i))
    min_contiguous_block_i = max(1, int(min_contiguous_block_i))
    if (not isfinite(min_chain_coverage_f)) or min_chain_coverage_f <= 0.0:
        min_chain_coverage_f = 0.35

    def _group_by_chain(rows):
        out = {}
        for idx, row in enumerate(list(rows or [])):
            chain = str((row or {}).get("chain", "") or "").strip()
            chain_key = chain if chain else "_"
            item = dict(row or {})
            item["global_index"] = int(idx)
            out.setdefault(chain_key, []).append(item)
        return out

    ref_by_chain = _group_by_chain(ref_rows)
    rec_by_chain = _group_by_chain(rec_rows)

    def _chain_summary(ref_chain, rec_chain):
        ref_items = list(ref_by_chain.get(ref_chain, []) or [])
        rec_items = list(rec_by_chain.get(rec_chain, []) or [])
        seq_ref = "".join(str((item or {}).get("aa1", "X") or "X") for item in ref_items)
        seq_rec = "".join(str((item or {}).get("aa1", "X") or "X") for item in rec_items)
        try:
            pairs_all, score = sequence_pairs_fn(seq_ref, seq_rec)
        except Exception:
            pairs_all, score = [], 0
        kept = []
        identity_count = 0
        mismatch_count = 0
        for pair in list(pairs_all or []):
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            try:
                idx_ref = int(pair[0])
                idx_rec = int(pair[1])
            except Exception:
                continue
            if idx_ref < 0 or idx_rec < 0 or idx_ref >= len(ref_items) or idx_rec >= len(rec_items):
                continue
            aa_ref = str(ref_items[idx_ref].get("aa1", "X") or "X")
            aa_rec = str(rec_items[idx_rec].get("aa1", "X") or "X")
            if aa_ref == "X" or aa_rec == "X":
                continue
            kept.append(
                (
                    int(ref_items[idx_ref].get("global_index", idx_ref)),
                    int(rec_items[idx_rec].get("global_index", idx_rec)),
                )
            )
            if aa_ref == aa_rec:
                identity_count += 1
            else:
                mismatch_count += 1
        block_lengths = []
        run_len = 0
        prev_ref = None
        prev_rec = None
        for pair in list(kept or []):
            ref_idx = int(pair[0])
            rec_idx = int(pair[1])
            if prev_ref is None or (ref_idx == (prev_ref + 1) and rec_idx == (prev_rec + 1)):
                run_len += 1
            else:
                block_lengths.append(int(run_len))
                run_len = 1
            prev_ref = int(ref_idx)
            prev_rec = int(rec_idx)
        if run_len > 0:
            block_lengths.append(int(run_len))
        shorter = min(len(ref_items), len(rec_items))
        required = max(3, min(int(min_chain_matches_i), int(shorter))) if shorter > 0 else int(min_chain_matches_i)
        required_block = max(3, min(int(min_contiguous_block_i), max(3, int(shorter // 2) if shorter > 0 else 0)))
        coverage = (float(len(kept)) / float(shorter)) if shorter > 0 else 0.0
        identity_fraction = (float(identity_count) / float(len(kept))) if kept else 0.0
        max_block_len = max(block_lengths) if block_lengths else 0
        ok = bool(
            len(kept) >= int(required)
            and coverage >= float(min_chain_coverage_f)
            and int(max_block_len) >= int(required_block)
        )
        return {
            "ref_chain": str(ref_chain),
            "rec_chain": str(rec_chain),
            "pairs": list(kept),
            "score": int(score or 0),
            "matched_ca": int(len(kept)),
            "identity_count": int(identity_count),
            "mismatch_count": int(mismatch_count),
            "identity_fraction": float(identity_fraction),
            "coverage": float(coverage),
            "shorter_len": int(shorter),
            "required_matches": int(required),
            "required_block_len": int(required_block),
            "block_count": int(len(block_lengths)),
            "max_block_len": int(max_block_len),
            "same_chain_id": bool(str(ref_chain) == str(rec_chain) and str(ref_chain) != "_"),
            "ok": bool(ok),
        }

    chain_pairs = []
    used_ref = set()
    used_rec = set()

    shared_chains = [cid for cid in list(ref_by_chain.keys()) if cid in rec_by_chain]
    for cid in list(shared_chains):
        stat = _chain_summary(cid, cid)
        if bool(stat.get("ok", False)):
            chain_pairs.append(stat)
            used_ref.add(str(cid))
            used_rec.add(str(cid))

    candidates = []
    for ref_chain in list(ref_by_chain.keys()):
        if ref_chain in used_ref:
            continue
        for rec_chain in list(rec_by_chain.keys()):
            if rec_chain in used_rec:
                continue
            stat = _chain_summary(ref_chain, rec_chain)
            if bool(stat.get("ok", False)):
                candidates.append(stat)
    candidates.sort(
        key=lambda item: (
            1 if bool((item or {}).get("same_chain_id", False)) else 0,
            int((item or {}).get("max_block_len", 0) or 0),
            int((item or {}).get("matched_ca", 0) or 0),
            float((item or {}).get("coverage", 0.0) or 0.0),
            float((item or {}).get("identity_fraction", 0.0) or 0.0),
            int((item or {}).get("score", 0) or 0),
        ),
        reverse=True,
    )
    for stat in list(candidates):
        ref_chain = str((stat or {}).get("ref_chain", "") or "")
        rec_chain = str((stat or {}).get("rec_chain", "") or "")
        if ref_chain in used_ref or rec_chain in used_rec:
            continue
        chain_pairs.append(stat)
        used_ref.add(ref_chain)
        used_rec.add(rec_chain)

    if not chain_pairs:
        fallback = _chain_summary("_", "_") if ("_" in ref_by_chain and "_" in rec_by_chain) else None
        if fallback and bool(fallback.get("ok", False)):
            chain_pairs = [fallback]

    total_pairs = []
    identity_total = 0
    mismatch_total = 0
    for stat in list(chain_pairs or []):
        total_pairs.extend(list((stat or {}).get("pairs", []) or []))
        identity_total += int((stat or {}).get("identity_count", 0) or 0)
        mismatch_total += int((stat or {}).get("mismatch_count", 0) or 0)

    required_total = max(8, min(int(min_total_matches_i), min(len(ref_rows), len(rec_rows))))
    if len(total_pairs) < int(required_total):
        payload["error"] = (
            f"only {len(total_pairs)} chain-aware CA matches "
            f"(required >= {int(required_total)})"
        )
        payload["chain_pairs"] = list(chain_pairs)
        payload["matched_ca"] = int(len(total_pairs))
        payload["identity_count"] = int(identity_total)
        payload["mismatch_count"] = int(mismatch_total)
        return payload

    payload["ok"] = True
    payload["pairs"] = list(total_pairs)
    payload["chain_pairs"] = list(chain_pairs)
    payload["matched_ca"] = int(len(total_pairs))
    payload["identity_count"] = int(identity_total)
    payload["mismatch_count"] = int(mismatch_total)
    payload["coverage"] = float(len(total_pairs)) / float(max(1, min(len(ref_rows), len(rec_rows))))
    return payload


def recover_redocking_reference_mol_from_pdb(
    pdb_path,
    *,
    smiles_hint="",
    chem_module=None,
    all_chem_module=None,
    select_best_fragment_fn=None,
):
    payload = {
        "ok": False,
        "mol": None,
        "source": "",
        "message": "",
    }
    src = str(pdb_path or "").strip()
    if (not src) or (not os.path.isfile(src)):
        payload["message"] = "reference_ligand_pdb_missing"
        return payload
    if chem_module is None:
        payload["message"] = "chem_module_missing"
        return payload

    mol_from_pdb_fn = getattr(chem_module, "MolFromPDBFile", None)
    mol_from_smiles_fn = getattr(chem_module, "MolFromSmiles", None)
    mol_clone_fn = getattr(chem_module, "Mol", None)
    remove_hs_fn = getattr(chem_module, "RemoveHs", None)
    sanitize_mol_fn = getattr(chem_module, "SanitizeMol", None)
    if not callable(mol_from_pdb_fn) or not callable(mol_clone_fn):
        payload["message"] = "chem_module_incomplete"
        return payload

    crystal = None
    try:
        crystal = mol_from_pdb_fn(src, removeHs=False, sanitize=False)
    except Exception:
        crystal = None
    if crystal is None:
        payload["message"] = "rdkit_parse_failed"
        return payload
    try:
        crystal = mol_clone_fn(crystal)
    except Exception:
        pass
    if callable(select_best_fragment_fn):
        try:
            best = select_best_fragment_fn(crystal)
            if best is not None:
                crystal = best
        except Exception:
            pass

    mapped = None
    smi = str(smiles_hint or "").strip()
    assign_bond_orders_fn = getattr(all_chem_module, "AssignBondOrdersFromTemplate", None)
    if smi and callable(mol_from_smiles_fn) and callable(assign_bond_orders_fn):
        template = None
        try:
            template = mol_from_smiles_fn(smi)
        except Exception:
            template = None
        if template is not None:
            try:
                template = mol_clone_fn(template)
            except Exception:
                pass
            if callable(remove_hs_fn):
                try:
                    template = remove_hs_fn(template)
                except Exception:
                    pass
                try:
                    crystal_core = remove_hs_fn(mol_clone_fn(crystal))
                except Exception:
                    crystal_core = crystal
            else:
                crystal_core = crystal
            if callable(select_best_fragment_fn):
                try:
                    best = select_best_fragment_fn(template)
                    if best is not None:
                        template = best
                except Exception:
                    pass
                try:
                    best = select_best_fragment_fn(crystal_core)
                    if best is not None:
                        crystal_core = best
                except Exception:
                    pass
            try:
                mapped = assign_bond_orders_fn(template, crystal_core)
            except Exception:
                mapped = None
            if mapped is not None and callable(sanitize_mol_fn):
                try:
                    sanitize_mol_fn(mapped)
                except Exception:
                    pass

    work_mol = mapped if mapped is not None else crystal
    if work_mol is not None and callable(sanitize_mol_fn):
        try:
            sanitize_mol_fn(work_mol)
        except Exception:
            pass
    if work_mol is not None and callable(select_best_fragment_fn):
        try:
            best = select_best_fragment_fn(work_mol)
            if best is not None:
                work_mol = best
        except Exception:
            pass
    try:
        atom_count = int(work_mol.GetNumAtoms())
        conformer_count = int(work_mol.GetNumConformers())
    except Exception:
        atom_count = 0
        conformer_count = 0
    if atom_count <= 0 or conformer_count <= 0:
        payload["message"] = "reference_mol_invalid"
        return payload

    payload["ok"] = True
    payload["mol"] = work_mol
    payload["source"] = "smiles_template" if mapped is not None else "pdb_fallback"
    payload["message"] = "ok"
    return payload


def attach_pose_object_to_rows(rows, pose_object):
    out = []
    po = str(pose_object or "")
    for row in list(rows or []):
        rr = dict(row or {})
        rr["pose_object"] = po
        out.append(rr)
    return out


def load_redocking_pose_object_safe(pose_object, pose_view_file, load_pose_object_fn):
    if not callable(load_pose_object_fn):
        return {
            "ok": False,
            "error": "load_pose_object_fn is not callable",
            "pose_object": str(pose_object or ""),
            "pose_view_file": str(pose_view_file or ""),
        }
    po = str(pose_object or "")
    pv = str(pose_view_file or "")
    try:
        load_pose_object_fn(pv, po)
        return {
            "ok": True,
            "pose_object": po,
            "pose_view_file": pv,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "pose_object": po,
            "pose_view_file": pv,
        }


def redocking_pose_load_warning_text(system_label, error):
    return f"Redocking pose load warning ({system_label}): {error}"


def prepare_redocking_pose_outputs(
    *,
    best_rows,
    best_pose_file,
    receptor_obj,
    system_label="",
    tag="",
    safe_name_fn=None,
    convert_pdbqt_to_sdf_fn=None,
    convert_pdbqt_to_pdb_fn=None,
    load_pose_object_fn=None,
    prefix="redock",
):
    pose_object = build_redocking_pose_object_name(
        receptor_obj=receptor_obj,
        tag=tag,
        safe_name_fn=safe_name_fn,
        prefix=prefix,
    )
    pose_view_file = resolve_redocking_pose_view_file(
        best_pose_file,
        convert_pdbqt_to_sdf_fn=convert_pdbqt_to_sdf_fn,
        convert_pdbqt_to_pdb_fn=convert_pdbqt_to_pdb_fn,
    )
    pose_load_payload = load_redocking_pose_object_safe(
        pose_object=pose_object,
        pose_view_file=pose_view_file,
        load_pose_object_fn=load_pose_object_fn,
    )
    warning_text = ""
    if not bool(pose_load_payload.get("ok", False)):
        warning_text = redocking_pose_load_warning_text(
            system_label=system_label,
            error=str(pose_load_payload.get("error", "unknown error") or "unknown error"),
        )
    return {
        "rows": list(attach_pose_object_to_rows(best_rows, pose_object)),
        "pose_object": str(pose_object),
        "pose_view_file": str(pose_view_file),
        "pose_load": dict(pose_load_payload or {}),
        "load_warning": str(warning_text),
    }


def resolve_redocking_ligand_file(ligand_override_file="", picked_ligand_file="", file_exists_fn=None):
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    lig_file = str(ligand_override_file or "").strip() or str(picked_ligand_file or "").strip()
    if (not lig_file) or (not bool(exists_fn(lig_file))):
        return {
            "ok": False,
            "ligand_file": "",
            "error": "No ligand input is prepared. Use 'Ligand Source & Prep'.",
        }
    return {
        "ok": True,
        "ligand_file": str(lig_file),
    }


def prepare_redocking_single_run_context(
    *,
    smina_bin,
    ligand_override_file="",
    picked_ligand_file="",
    workspace_dir="",
    system_label="",
    receptor_obj="",
    params=None,
    safe_name_fn=None,
    file_exists_fn=None,
    makedirs_fn=None,
    timestamp_tag="",
):
    sbin = str(smina_bin or "").strip()
    if not sbin:
        return {
            "ok": False,
            "error": "smina was not found in PATH.",
        }
    lig_src_payload = resolve_redocking_ligand_file(
        ligand_override_file=str(ligand_override_file or ""),
        picked_ligand_file=str(picked_ligand_file or ""),
        file_exists_fn=file_exists_fn,
    )
    if not bool(lig_src_payload.get("ok", False)):
        return {
            "ok": False,
            "error": str(lig_src_payload.get("error", "No ligand input is prepared. Use 'Ligand Source & Prep'.")),
        }

    def _safe_name(v):
        txt = str(v or "")
        if callable(safe_name_fn):
            try:
                txt = str(safe_name_fn(v) or v)
            except Exception:
                txt = str(v or "")
        return txt

    tag = str(timestamp_tag or "").strip()
    if not tag:
        tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = os.path.join(
        str(workspace_dir or ""),
        f"redock_{_safe_name(system_label)}_{_safe_name(receptor_obj)}_{tag}",
    )
    if callable(makedirs_fn):
        makedirs_fn(job_dir, exist_ok=True)
    else:
        os.makedirs(job_dir, exist_ok=True)
    target_ph = float((params or {}).get("ph", 7.4))
    return {
        "ok": True,
        "smina_bin": str(sbin),
        "ligand_file": str(lig_src_payload.get("ligand_file", "") or ""),
        "tag": str(tag),
        "job_dir": str(job_dir),
        "target_ph": float(target_ph),
    }


def prepare_redocking_base_payload(
    *,
    ref_obj="",
    wt_obj="",
    ref_fields=None,
    run_mutant=False,
    mut_obj="",
    mutant_systems=None,
    params=None,
    use_modeler_wt=False,
    modeler_wt_obj="",
    grid_from_reference_fn=None,
):
    ref_name = str(ref_obj or "").strip()
    wt_name = str(wt_obj or "").strip()
    modeler_name = str(modeler_wt_obj or "").strip()
    used_modeler_wt = False
    if bool(use_modeler_wt) and modeler_name:
        wt_name = str(modeler_name)
        used_modeler_wt = True

    if not ref_name:
        return {
            "ok": False,
            "error": "Select a reference complex object.",
            "used_modeler_wt": bool(used_modeler_wt),
            "payload": {},
        }
    if not wt_name:
        return {
            "ok": False,
            "error": "Select a WT receptor object.",
            "used_modeler_wt": bool(used_modeler_wt),
            "payload": {},
        }
    fields = tuple(ref_fields or ())
    if not fields:
        return {
            "ok": False,
            "error": "Set crystal ligand Chain/Resi/Resn (or click Auto-detect).",
            "used_modeler_wt": bool(used_modeler_wt),
            "payload": {},
        }
    if not callable(grid_from_reference_fn):
        return {
            "ok": False,
            "error": "grid_from_reference_fn is not callable",
            "used_modeler_wt": bool(used_modeler_wt),
            "payload": {},
        }

    try:
        center, size, nref = grid_from_reference_fn(ref_name, fields)
    except Exception as e:
        return {
            "ok": False,
            "error": f"Failed to compute reference grid from selected ligand: {e}",
            "used_modeler_wt": bool(used_modeler_wt),
            "payload": {},
        }
    if center is None:
        return {
            "ok": False,
            "error": "Reference ligand was not found. Verify chain/resi/resn against reference object.",
            "used_modeler_wt": bool(used_modeler_wt),
            "payload": {},
        }

    systems = [("WT", str(wt_name))]

    def _normalize_mutant_systems(entries):
        out = []
        idx = 0
        seen = set()
        for ent in list(entries or []):
            idx += 1
            label = ""
            obj = ""
            if isinstance(ent, dict):
                label = str(ent.get("label", "") or "").strip()
                obj = str(ent.get("object", "") or ent.get("receptor", "") or "").strip()
            elif isinstance(ent, (list, tuple)):
                if len(ent) > 0:
                    label = str(ent[0] or "").strip()
                if len(ent) > 1:
                    obj = str(ent[1] or "").strip()
            else:
                obj = str(ent or "").strip()
            if not obj:
                continue
            if not label:
                if idx == 1:
                    label = "Mutant"
                else:
                    label = f"Mutant_{idx:03d}"
            key = (str(label), str(obj))
            if key in seen:
                continue
            seen.add(key)
            out.append((str(label), str(obj)))
        return out

    if bool(run_mutant):
        mut_entries = _normalize_mutant_systems(mutant_systems)
        if not mut_entries:
            mut_name = str(mut_obj or "").strip()
            if mut_name:
                mut_entries = [("Mutant", str(mut_name))]
        if not mut_entries:
            return {
                "ok": False,
                "error": "Mutant comparison is enabled, but mutant receptor is empty.",
                "used_modeler_wt": bool(used_modeler_wt),
                "payload": {},
            }
        filtered_mut_entries = []
        dropped_identical = []
        for mut_label, mut_name in list(mut_entries):
            if str(mut_name) == str(wt_name):
                dropped_identical.append((str(mut_label), str(mut_name)))
                continue
            filtered_mut_entries.append((str(mut_label), str(mut_name)))
        if not filtered_mut_entries:
            return {
                "ok": False,
                "error": "WT and mutant receptor objects are identical.",
                "used_modeler_wt": bool(used_modeler_wt),
                "payload": {},
            }
        for mut_label, mut_name in list(filtered_mut_entries):
            systems.append((str(mut_label), str(mut_name)))

    return {
        "ok": True,
        "error": "",
        "used_modeler_wt": bool(used_modeler_wt),
        "payload": {
            "ref_obj": str(ref_name),
            "wt_obj": str(wt_name),
            "ref_fields": tuple(fields),
            "center": tuple(center) if center is not None else None,
            "size": tuple(size) if size is not None else None,
            "nref": int(nref or 0),
            "systems": list(systems),
            "params": dict(params or {}),
            "dropped_mutant_identical_to_wt": list(dropped_identical) if bool(run_mutant) else [],
        },
    }


def prepare_redocking_base_runtime(
    *,
    reference_obj_fn=None,
    wt_obj_fn=None,
    use_modeler_wt_fn=None,
    modeler_wt_obj_fn=None,
    ref_fields_fn=None,
    run_mutant_fn=None,
    mutant_obj_fn=None,
    mutant_systems_fn=None,
    params_fn=None,
    grid_from_reference_fn=None,
    base_payload_builder_fn=None,
):
    def _read_str(fn):
        if not callable(fn):
            return ""
        try:
            return str(fn() or "").strip()
        except Exception:
            return ""

    def _read_bool(fn):
        if not callable(fn):
            return False
        try:
            return bool(fn())
        except Exception:
            return False

    def _read_tuple(fn):
        if not callable(fn):
            return ()
        try:
            return tuple(fn() or ())
        except Exception:
            return ()

    def _read_dict(fn):
        if not callable(fn):
            return {}
        try:
            return dict(fn() or {})
        except Exception:
            return {}

    def _read_list(fn):
        if not callable(fn):
            return []
        try:
            return list(fn() or [])
        except Exception:
            return []

    ref_obj = _read_str(reference_obj_fn)
    wt_obj = _read_str(wt_obj_fn)
    use_modeler_wt = _read_bool(use_modeler_wt_fn)
    model_obj = _read_str(modeler_wt_obj_fn)
    ref_fields = _read_tuple(ref_fields_fn)
    run_mut = _read_bool(run_mutant_fn)
    mut_obj = _read_str(mutant_obj_fn)
    mut_systems = _read_list(mutant_systems_fn)
    params = _read_dict(params_fn)

    build_payload = (
        base_payload_builder_fn if callable(base_payload_builder_fn) else prepare_redocking_base_payload
    )
    try:
        payload = dict(
            build_payload(
                ref_obj=str(ref_obj),
                wt_obj=str(wt_obj),
                ref_fields=tuple(ref_fields or ()),
                run_mutant=bool(run_mut),
                mut_obj=str(mut_obj),
                mutant_systems=list(mut_systems or []),
                params=dict(params or {}),
                use_modeler_wt=bool(use_modeler_wt),
                modeler_wt_obj=str(model_obj),
                grid_from_reference_fn=grid_from_reference_fn,
            )
            or {}
        )
    except Exception as e:
        payload = {
            "ok": False,
            "error": str(e),
            "used_modeler_wt": bool(use_modeler_wt and model_obj),
            "payload": {},
        }
    payload_out = dict(payload or {})
    payload_out["modeler_wt_obj"] = str(model_obj)
    return payload_out


def prepare_redocking_ligand_override_context(
    *,
    use_reference_ligand=False,
    ref_obj="",
    ref_fields=None,
    workspace_dir="",
    export_reference_ligand_fn=None,
    pick_ligand_input_fn=None,
    file_exists_fn=None,
    makedirs_fn=None,
    reference_prep_dir="",
):
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    use_ref = bool(use_reference_ligand)
    prep_dir = str(reference_prep_dir or "").strip()
    if not prep_dir:
        prep_dir = os.path.join(str(workspace_dir or ""), "redock_refprep")

    if use_ref:
        if not callable(export_reference_ligand_fn):
            return {
                "ok": False,
                "error": "export_reference_ligand_fn is not callable",
                "ligand_override_file": "",
                "use_reference_ligand": True,
                "prep_dir": str(prep_dir),
            }
        try:
            if callable(makedirs_fn):
                makedirs_fn(prep_dir, exist_ok=True)
            else:
                os.makedirs(prep_dir, exist_ok=True)
        except Exception:
            pass
        ligand_override_file = ""
        lig_msg = ""
        try:
            ligand_override_file, lig_msg = export_reference_ligand_fn(ref_obj, ref_fields, prep_dir)
        except Exception as e:
            ligand_override_file = ""
            lig_msg = str(e)
        if (not ligand_override_file) or (not bool(exists_fn(ligand_override_file))):
            return {
                "ok": False,
                "error": f"Failed to export crystal reference ligand for redocking: {lig_msg}",
                "ligand_override_file": "",
                "use_reference_ligand": True,
                "prep_dir": str(prep_dir),
            }
        return {
            "ok": True,
            "error": "",
            "ligand_override_file": str(ligand_override_file),
            "use_reference_ligand": True,
            "prep_dir": str(prep_dir),
        }

    if not callable(pick_ligand_input_fn):
        return {
            "ok": False,
            "error": "pick_ligand_input_fn is not callable",
            "ligand_override_file": "",
            "use_reference_ligand": False,
            "prep_dir": str(prep_dir),
        }
    lig_file = str(pick_ligand_input_fn() or "").strip()
    if (not lig_file) or (not bool(exists_fn(lig_file))):
        return {
            "ok": False,
            "error": "No ligand is prepared for docking. Open 'Ligand Source & Prep' and generate PDBQT or charged MOL2.",
            "ligand_override_file": "",
            "use_reference_ligand": False,
            "prep_dir": str(prep_dir),
        }
    return {
        "ok": True,
        "error": "",
        "ligand_override_file": str(lig_file),
        "use_reference_ligand": False,
        "prep_dir": str(prep_dir),
    }


def prepare_redocking_ligand_override_runtime(
    *,
    use_reference_ligand=False,
    ref_obj="",
    ref_fields=None,
    workspace_dir_fn=None,
    export_reference_ligand_fn=None,
    pick_ligand_input_fn=None,
    file_exists_fn=None,
    makedirs_fn=None,
    reference_prep_dir="",
    context_builder_fn=None,
):
    workspace_dir = ""
    if callable(workspace_dir_fn):
        try:
            workspace_dir = str(workspace_dir_fn() or "")
        except Exception:
            workspace_dir = ""

    build_context = (
        context_builder_fn
        if callable(context_builder_fn)
        else prepare_redocking_ligand_override_context
    )
    try:
        payload = dict(
            build_context(
                use_reference_ligand=bool(use_reference_ligand),
                ref_obj=str(ref_obj or ""),
                ref_fields=tuple(ref_fields or ()),
                workspace_dir=str(workspace_dir or ""),
                export_reference_ligand_fn=export_reference_ligand_fn,
                pick_ligand_input_fn=pick_ligand_input_fn,
                file_exists_fn=file_exists_fn,
                makedirs_fn=makedirs_fn,
                reference_prep_dir=str(reference_prep_dir or ""),
            )
            or {}
        )
    except Exception as e:
        payload = {
            "ok": False,
            "error": str(e),
            "ligand_override_file": "",
            "use_reference_ligand": bool(use_reference_ligand),
            "prep_dir": str(reference_prep_dir or ""),
        }
    return dict(payload or {})


def prepare_redocking_reference_ligand_export_plan(
    *,
    selection="",
    job_dir="",
    output_name="reference_ligand_redock.pdb",
):
    sel = str(selection or "").strip()
    out_name = str(output_name or "").strip() or "reference_ligand_redock.pdb"
    jdir = str(job_dir or "").strip()
    out_path = os.path.join(str(jdir), str(out_name)) if jdir else str(out_name)
    payload = {
        "ok": False,
        "reason": "",
        "error": "",
        "selection": str(sel),
        "job_dir": str(jdir),
        "output_name": str(out_name),
        "output_path": str(out_path),
    }
    if not sel:
        payload["reason"] = "empty_selection"
        payload["error"] = "Reference ligand selection is empty"
        return payload
    payload["ok"] = True
    return payload


def finalize_redocking_reference_ligand_export_result(
    *,
    output_path="",
    save_error="",
    file_exists=False,
    atom_count=None,
):
    out_path = str(output_path or "").strip()
    err = str(save_error or "").strip()
    atom_count_i = None
    if atom_count is not None:
        try:
            atom_count_i = int(atom_count)
        except Exception:
            atom_count_i = None
    payload = {
        "ok": False,
        "reason": "",
        "message": "",
        "output_path": str(out_path),
        "file_exists": bool(file_exists),
        "atom_count": atom_count_i,
    }
    if err:
        payload["reason"] = "save_failed"
        payload["message"] = f"Failed to export reference ligand: {err}"
        return payload
    if not bool(file_exists):
        payload["reason"] = "output_missing"
        payload["message"] = "Reference ligand export failed"
        return payload
    if atom_count_i is not None and int(atom_count_i) <= 0:
        payload["reason"] = "zero_atoms"
        payload["message"] = "Reference ligand has zero atoms"
        return payload
    payload["ok"] = True
    payload["reason"] = "ok"
    payload["message"] = "ok"
    return payload


def resolve_redocking_reference_ligand_export_response(
    *,
    export_plan=None,
    export_result=None,
    fallback_output_path="",
):
    plan = dict(export_plan or {}) if isinstance(export_plan, dict) else {}
    result = dict(export_result or {}) if isinstance(export_result, dict) else {}
    fallback_path = str(fallback_output_path or "").strip()

    if plan and not bool(plan.get("ok", False)):
        return {
            "ok": False,
            "reason": str(plan.get("reason", "empty_selection") or "empty_selection"),
            "output_path": "",
            "message": str(plan.get("error", "Reference ligand selection is empty") or "Reference ligand selection is empty"),
        }

    if result:
        if bool(result.get("ok", False)):
            out_path = str(result.get("output_path", fallback_path) or fallback_path)
            return {
                "ok": True,
                "reason": str(result.get("reason", "ok") or "ok"),
                "output_path": str(out_path),
                "message": str(result.get("message", "ok") or "ok"),
            }
        return {
            "ok": False,
            "reason": str(result.get("reason", "export_failed") or "export_failed"),
            "output_path": "",
            "message": str(result.get("message", "Reference ligand export failed") or "Reference ligand export failed"),
        }

    return {
        "ok": False,
        "reason": "missing_export_result",
        "output_path": "",
        "message": "Reference ligand export failed",
    }


def prepare_redocking_reference_mol_plan(
    *,
    has_rdkit=False,
    selection="",
    job_dir="",
    output_name="reference_ligand.pdb",
):
    payload = {
        "ok": False,
        "reason": "",
        "error": "",
        "has_rdkit": bool(has_rdkit),
        "selection": str(selection or ""),
        "job_dir": str(job_dir or ""),
        "output_name": str(output_name or "reference_ligand.pdb"),
        "output_path": "",
    }
    if not bool(has_rdkit):
        payload["reason"] = "rdkit_unavailable"
        payload["error"] = "RDKit not available"
        return payload
    export_plan = prepare_redocking_reference_ligand_export_plan(
        selection=str(selection or ""),
        job_dir=str(job_dir or ""),
        output_name=str(output_name or "reference_ligand.pdb"),
    )
    payload["reason"] = str(export_plan.get("reason", "") or "")
    payload["error"] = str(export_plan.get("error", "") or "")
    payload["selection"] = str(export_plan.get("selection", selection) or selection)
    payload["job_dir"] = str(export_plan.get("job_dir", job_dir) or job_dir)
    payload["output_name"] = str(export_plan.get("output_name", output_name) or output_name)
    payload["output_path"] = str(export_plan.get("output_path", "") or "")
    payload["ok"] = bool(export_plan.get("ok", False))
    return payload


def finalize_redocking_reference_mol_result(
    *,
    save_error="",
    has_mol=False,
    atom_count=None,
    conformer_count=None,
):
    err = str(save_error or "").strip()
    atoms_i = None
    if atom_count is not None:
        try:
            atoms_i = int(atom_count)
        except Exception:
            atoms_i = None
    conformers_i = None
    if conformer_count is not None:
        try:
            conformers_i = int(conformer_count)
        except Exception:
            conformers_i = None
    payload = {
        "ok": False,
        "reason": "",
        "message": "",
        "has_mol": bool(has_mol),
        "atom_count": atoms_i,
        "conformer_count": conformers_i,
    }
    if err:
        payload["reason"] = "save_failed"
        payload["message"] = f"Failed to export reference ligand: {err}"
        return payload
    if not bool(has_mol):
        payload["reason"] = "rdkit_read_failed"
        payload["message"] = "Failed to read reference ligand into RDKit"
        return payload
    if atoms_i is not None and int(atoms_i) <= 0:
        payload["reason"] = "no_atoms"
        payload["message"] = "Reference ligand contains no atoms"
        return payload
    if conformers_i is not None and int(conformers_i) <= 0:
        payload["reason"] = "no_conformer"
        payload["message"] = "Reference ligand has no 3D conformer"
        return payload
    payload["ok"] = True
    payload["reason"] = "ok"
    payload["message"] = "ok"
    return payload


def prepare_redocking_rmsd_mol_plan(
    *,
    has_rdkit=False,
    has_input_mol=False,
    min_atoms=3,
):
    try:
        min_atoms_i = int(min_atoms)
    except Exception:
        min_atoms_i = 3
    min_atoms_i = max(1, int(min_atoms_i))
    payload = {
        "ok": False,
        "reason": "",
        "has_rdkit": bool(has_rdkit),
        "has_input_mol": bool(has_input_mol),
        "min_atoms": int(min_atoms_i),
    }
    if not bool(has_rdkit):
        payload["reason"] = "rdkit_unavailable"
        return payload
    if not bool(has_input_mol):
        payload["reason"] = "missing_input_mol"
        return payload
    payload["ok"] = True
    payload["reason"] = "ok"
    return payload


def finalize_redocking_rmsd_mol_result(
    *,
    has_work_mol=False,
    atom_count=None,
    conformer_count=None,
    min_atoms=3,
):
    try:
        min_atoms_i = int(min_atoms)
    except Exception:
        min_atoms_i = 3
    min_atoms_i = max(1, int(min_atoms_i))
    atoms_i = None
    if atom_count is not None:
        try:
            atoms_i = int(atom_count)
        except Exception:
            atoms_i = None
    conformers_i = None
    if conformer_count is not None:
        try:
            conformers_i = int(conformer_count)
        except Exception:
            conformers_i = None
    payload = {
        "ok": False,
        "reason": "",
        "has_work_mol": bool(has_work_mol),
        "atom_count": atoms_i,
        "conformer_count": conformers_i,
        "min_atoms": int(min_atoms_i),
    }
    if not bool(has_work_mol):
        payload["reason"] = "missing_work_mol"
        return payload
    if atoms_i is None or int(atoms_i) < int(min_atoms_i):
        payload["reason"] = "insufficient_atoms"
        return payload
    if conformers_i is None or int(conformers_i) <= 0:
        payload["reason"] = "missing_conformer"
        return payload
    payload["ok"] = True
    payload["reason"] = "ok"
    return payload


def build_redocking_rmsd_mol_config(
    *,
    overrides=None,
    min_atoms=3,
):
    try:
        default_min_atoms = int(min_atoms)
    except Exception:
        default_min_atoms = 3
    default_min_atoms = max(1, int(default_min_atoms))

    over = {}
    if isinstance(overrides, dict):
        over = dict(overrides)
    elif overrides is not None:
        try:
            over = dict(overrides)
        except Exception:
            over = {}

    try:
        min_atoms_i = int(over.get("min_atoms", default_min_atoms))
    except Exception:
        min_atoms_i = int(default_min_atoms)
    min_atoms_i = max(1, int(min_atoms_i))
    return {
        "min_atoms": int(min_atoms_i),
    }


def resolve_redocking_rmsd_mol(
    *,
    has_rdkit=False,
    input_mol=None,
    min_atoms=3,
    clone_mol_fn=None,
    select_best_fragment_fn=None,
    remove_hs_fn=None,
    sanitize_mol_fn=None,
    get_atom_count_fn=None,
    get_conformer_count_fn=None,
):
    prep_plan = prepare_redocking_rmsd_mol_plan(
        has_rdkit=bool(has_rdkit),
        has_input_mol=bool(input_mol is not None),
        min_atoms=min_atoms,
    )
    payload = {
        "ok": False,
        "reason": str(prep_plan.get("reason", "") or ""),
        "prep_plan": dict(prep_plan or {}),
        "result": {},
        "work_mol": None,
    }
    if not bool(prep_plan.get("ok", False)):
        return payload

    work = None
    if callable(clone_mol_fn):
        try:
            work = clone_mol_fn(input_mol)
        except Exception:
            work = None
    else:
        work = input_mol
    if work is not None and callable(select_best_fragment_fn):
        try:
            best = select_best_fragment_fn(work)
            if best is not None:
                work = best
        except Exception:
            pass
    if work is not None and callable(remove_hs_fn):
        try:
            work = remove_hs_fn(work)
        except Exception:
            pass
    if work is not None and callable(sanitize_mol_fn):
        try:
            sanitize_mol_fn(work)
        except Exception:
            pass

    atom_count = None
    conformer_count = None
    if work is not None:
        if callable(get_atom_count_fn):
            try:
                atom_count = int(get_atom_count_fn(work))
            except Exception:
                atom_count = None
        if atom_count is None:
            try:
                atom_count = int(work.GetNumAtoms())
            except Exception:
                atom_count = None

        if callable(get_conformer_count_fn):
            try:
                conformer_count = int(get_conformer_count_fn(work))
            except Exception:
                conformer_count = None
        if conformer_count is None:
            try:
                conformer_count = int(work.GetNumConformers())
            except Exception:
                conformer_count = None

    result = finalize_redocking_rmsd_mol_result(
        has_work_mol=bool(work is not None),
        atom_count=atom_count,
        conformer_count=conformer_count,
        min_atoms=int(prep_plan.get("min_atoms", 3) or 3),
    )
    payload["result"] = dict(result or {})
    payload["ok"] = bool(result.get("ok", False))
    payload["reason"] = str(result.get("reason", "") or payload["reason"])
    if bool(payload["ok"]):
        payload["work_mol"] = work
    return payload


def resolve_redocking_rdkit_rmsd_mol(
    *,
    has_rdkit=False,
    input_mol=None,
    min_atoms=3,
    chem_module=None,
    select_best_fragment_fn=None,
):
    clone_mol_fn = getattr(chem_module, "Mol", None)
    if not callable(clone_mol_fn):
        clone_mol_fn = None

    remove_hs_fn = getattr(chem_module, "RemoveHs", None)
    if not callable(remove_hs_fn):
        remove_hs_fn = None

    sanitize_mol_fn = getattr(chem_module, "SanitizeMol", None)
    if not callable(sanitize_mol_fn):
        sanitize_mol_fn = None

    return dict(
        resolve_redocking_rmsd_mol(
            has_rdkit=bool(has_rdkit),
            input_mol=input_mol,
            min_atoms=min_atoms,
            clone_mol_fn=clone_mol_fn,
            select_best_fragment_fn=select_best_fragment_fn,
            remove_hs_fn=remove_hs_fn,
            sanitize_mol_fn=sanitize_mol_fn,
        )
        or {}
    )


def resolve_redocking_rdkit_rmsd_mol_with_config(
    *,
    has_rdkit=False,
    input_mol=None,
    rmsd_mol_config=None,
    chem_module=None,
    select_best_fragment_fn=None,
    default_min_atoms=3,
):
    try:
        default_min_atoms_i = int(default_min_atoms)
    except Exception:
        default_min_atoms_i = 3
    default_min_atoms_i = max(1, int(default_min_atoms_i))
    cfg = dict(
        build_redocking_rmsd_mol_config(
            overrides=rmsd_mol_config,
            min_atoms=default_min_atoms_i,
        )
        or {}
    )
    return dict(
        resolve_redocking_rdkit_rmsd_mol(
            has_rdkit=bool(has_rdkit),
            input_mol=input_mol,
            min_atoms=cfg.get("min_atoms", int(default_min_atoms_i)),
            chem_module=chem_module,
            select_best_fragment_fn=select_best_fragment_fn,
        )
        or {}
    )


def build_redocking_unique_atom_maps(maps, max_maps=256):
    try:
        max_maps_i = int(max_maps)
    except Exception:
        max_maps_i = 256
    max_maps_i = max(1, int(max_maps_i))

    uniq = []
    seen = set()
    for amap in list(maps or []):
        if not amap:
            continue
        norm = []
        bad_map = False
        for pair in list(amap or []):
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                bad_map = True
                break
            try:
                a = int(pair[0])
                b = int(pair[1])
            except Exception:
                bad_map = True
                break
            norm.append((int(a), int(b)))
        if bad_map or (not norm):
            continue
        key = tuple(norm)
        if key in seen:
            continue
        seen.add(key)
        uniq.append([(int(a), int(b)) for a, b in key])
        if len(uniq) >= int(max_maps_i):
            break
    return list(uniq)


def compute_redocking_rmsd_no_fit(ref_mol, pose_mol, atom_map, *, min_pairs=3):
    if (ref_mol is None) or (pose_mol is None):
        return float("nan")
    try:
        min_pairs_i = int(min_pairs)
    except Exception:
        min_pairs_i = 3
    min_pairs_i = max(1, int(min_pairs_i))
    amap = list(atom_map or [])
    if len(amap) < int(min_pairs_i):
        return float("nan")
    try:
        cref = ref_mol.GetConformer()
        cpose = pose_mol.GetConformer()
    except Exception:
        return float("nan")
    sse = 0.0
    n = 0
    for pair in amap:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            p_idx = int(pair[0])
            r_idx = int(pair[1])
        except Exception:
            continue
        try:
            p = cpose.GetAtomPosition(int(p_idx))
            r = cref.GetAtomPosition(int(r_idx))
        except Exception:
            continue
        dx = float(p.x) - float(r.x)
        dy = float(p.y) - float(r.y)
        dz = float(p.z) - float(r.z)
        sse += (dx * dx + dy * dy + dz * dz)
        n += 1
    if n < int(min_pairs_i):
        return float("nan")
    return float((float(sse) / float(n)) ** 0.5)


def redocking_heavy_atom_skeleton(mol):
    """
    Heavy-atom graph of a molecule for pose RMSD: elements and connectivity only (every
    bond single, no charges, no aromaticity, no sanitization), with its coordinates.

    Docked poses come back through PDBQT, which carries no bond orders; Open Babel's
    re-perception then gives wrong valences (benzamidinium became a cyclohexane with a
    five-valent N) and a sanitized molecule could not be built, so every RMSD was NaN.
    Matching on the skeleton does not depend on bond orders or protonation.
    Returns (skeleton_mol, coordinates) or (None, None).
    """
    try:
        from rdkit import Chem
    except Exception:
        return None, None
    if mol is None:
        return None, None
    try:
        conf = mol.GetConformer()
    except Exception:
        return None, None
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) < 1:
        return None, None
    new_index = {old: i for i, old in enumerate(heavy)}
    rw = Chem.RWMol()
    for old in heavy:
        atom = Chem.Atom(int(mol.GetAtomWithIdx(old).GetAtomicNum()))
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in new_index and b in new_index:
            rw.AddBond(new_index[a], new_index[b], Chem.BondType.SINGLE)
    skel = rw.GetMol()
    try:
        frags = Chem.GetMolFrags(skel)
    except Exception:
        frags = ()
    if len(frags) > 1:
        # largest heavy-atom fragment only (counter-ions, stray atoms of a pose file)
        keep = sorted(max(frags, key=len))
        heavy = [heavy[i] for i in keep]
        return redocking_heavy_atom_skeleton(_redocking_submol(mol, heavy))
    try:
        skel.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(skel)
    except Exception:
        return None, None
    coords = []
    for old in heavy:
        pos = conf.GetAtomPosition(int(old))
        coords.append((float(pos.x), float(pos.y), float(pos.z)))
    return skel, coords


def _redocking_submol(mol, atom_indices):
    """Copy of ``mol`` restricted to ``atom_indices`` (with their bonds and coordinates)."""
    from rdkit import Chem

    keep = set(int(i) for i in atom_indices)
    rw = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in keep), reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


def compute_redocking_graph_rmsd(ref_mol, pose_mol, *, max_maps=1_000_000):
    """
    Symmetry-corrected heavy-atom RMSD without superposition, over every graph
    isomorphism between the two heavy-atom skeletons (see redocking_heavy_atom_skeleton).
    NaN when the skeletons differ (different atom count or connectivity).
    """
    ref_skel, ref_xyz = redocking_heavy_atom_skeleton(ref_mol)
    pose_skel, pose_xyz = redocking_heavy_atom_skeleton(pose_mol)
    if ref_skel is None or pose_skel is None or len(ref_xyz) != len(pose_xyz):
        return float("nan")
    if ref_skel.GetNumBonds() != pose_skel.GetNumBonds():
        return float("nan")
    try:
        matches = ref_skel.GetSubstructMatches(
            pose_skel, uniquify=False, useChirality=False, maxMatches=int(max(1, max_maps))
        )
    except Exception:
        return float("nan")
    best = float("inf")
    n = len(pose_xyz)
    for match in matches:
        if len(match) != n:
            continue
        sse = 0.0
        for p_idx, r_idx in enumerate(match):
            px, py, pz = pose_xyz[p_idx]
            rx, ry, rz = ref_xyz[int(r_idx)]
            sse += (px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2
        best = min(best, (sse / float(n)) ** 0.5)
    return float(best) if isfinite(best) else float("nan")


def resolve_redocking_best_rmsd(
    *,
    has_rdkit=True,
    ref_mol=None,
    pose_mol=None,
    prepare_rmsd_mol_fn=None,
    candidate_atom_maps_fn=None,
    compute_rmsd_no_fit_fn=None,
    max_maps=256,
    min_pairs=3,
):
    nan_value = float("nan")
    if (not bool(has_rdkit)) or (ref_mol is None) or (pose_mol is None):
        return nan_value
    graph_rmsd = nan_value
    try:
        graph_rmsd = float(compute_redocking_graph_rmsd(ref_mol, pose_mol))
    except Exception:
        graph_rmsd = nan_value
    if isfinite(graph_rmsd):
        return float(graph_rmsd)
    if (not callable(prepare_rmsd_mol_fn)) or (not callable(candidate_atom_maps_fn)):
        return nan_value
    if not callable(compute_rmsd_no_fit_fn):
        return nan_value
    try:
        max_maps_i = int(max_maps)
    except Exception:
        max_maps_i = 256
    max_maps_i = max(1, int(max_maps_i))
    try:
        min_pairs_i = int(min_pairs)
    except Exception:
        min_pairs_i = 3
    min_pairs_i = max(1, int(min_pairs_i))

    try:
        ref_work = prepare_rmsd_mol_fn(ref_mol)
    except Exception:
        ref_work = None
    try:
        pose_work = prepare_rmsd_mol_fn(pose_mol)
    except Exception:
        pose_work = None
    if (ref_work is None) or (pose_work is None):
        return nan_value

    atom_maps = []
    try:
        atom_maps = list(candidate_atom_maps_fn(ref_work, pose_work, max_maps=int(max_maps_i)) or [])
    except TypeError:
        try:
            atom_maps = list(candidate_atom_maps_fn(ref_work, pose_work, int(max_maps_i)) or [])
        except Exception:
            atom_maps = []
    except Exception:
        atom_maps = []
    if not atom_maps:
        return nan_value

    best = float("inf")
    for amap in list(atom_maps):
        try:
            rmsd = float(
                compute_rmsd_no_fit_fn(
                    ref_work,
                    pose_work,
                    amap,
                    min_pairs=int(min_pairs_i),
                )
            )
        except TypeError:
            try:
                rmsd = float(
                    compute_rmsd_no_fit_fn(
                        ref_work,
                        pose_work,
                        amap,
                        int(min_pairs_i),
                    )
                )
            except Exception:
                rmsd = float("nan")
        except Exception:
            rmsd = float("nan")
        if isfinite(rmsd):
            best = min(float(best), float(rmsd))
    if isfinite(best):
        return float(best)
    return nan_value


def build_redocking_best_rmsd_config(
    *,
    overrides=None,
    max_maps=256,
    min_pairs=3,
):
    defaults = {
        "max_maps": max_maps,
        "min_pairs": min_pairs,
    }

    over = {}
    if isinstance(overrides, dict):
        over = dict(overrides)
    elif overrides is not None:
        try:
            over = dict(overrides)
        except Exception:
            over = {}

    def _norm_int(raw_val, fallback):
        try:
            fb = int(fallback)
        except Exception:
            fb = 1
        fb = max(1, int(fb))
        try:
            vv = int(raw_val)
        except Exception:
            vv = int(fb)
        return max(1, int(vv))

    return {
        "max_maps": _norm_int(over.get("max_maps", defaults["max_maps"]), defaults["max_maps"]),
        "min_pairs": _norm_int(over.get("min_pairs", defaults["min_pairs"]), defaults["min_pairs"]),
    }


def build_redocking_directional_match_functions(
    *,
    ref_mol=None,
    pose_mol=None,
    get_substruct_matches_fn=None,
    uniquify=False,
    use_chirality=False,
):
    def _empty(_limit):
        return []

    if (ref_mol is None) or (pose_mol is None) or (not callable(get_substruct_matches_fn)):
        return {
            "forward_matches_fn": _empty,
            "reverse_matches_fn": _empty,
        }

    def _build(query_mol, pattern_mol):
        def _run(limit):
            try:
                lim = int(limit)
            except Exception:
                lim = 1
            lim = max(1, int(lim))
            try:
                matches = list(
                    get_substruct_matches_fn(
                        query_mol,
                        pattern_mol,
                        bool(uniquify),
                        bool(use_chirality),
                    )
                    or []
                )
            except Exception:
                matches = []
            return list(matches[: int(lim)])

        return _run

    return {
        "forward_matches_fn": _build(ref_mol, pose_mol),
        "reverse_matches_fn": _build(pose_mol, ref_mol),
    }


def build_redocking_bond_order_matches(
    *,
    ref_mol=None,
    pose_mol=None,
    nref=0,
    npose=0,
    max_maps=256,
    assign_bond_orders_fn=None,
    sanitize_mol_fn=None,
    forward_matches_for_mol_fn=None,
):
    try:
        nref_i = int(nref)
    except Exception:
        nref_i = 0
    try:
        npose_i = int(npose)
    except Exception:
        npose_i = 0
    if nref_i <= 0 or npose_i <= 0 or nref_i != npose_i:
        return []
    if (not callable(assign_bond_orders_fn)) or (not callable(forward_matches_for_mol_fn)):
        return []
    try:
        max_maps_i = int(max_maps)
    except Exception:
        max_maps_i = 256
    max_maps_i = max(1, int(max_maps_i))

    try:
        pose_bo = assign_bond_orders_fn(ref_mol, pose_mol)
    except Exception:
        pose_bo = None
    if pose_bo is None:
        return []
    if callable(sanitize_mol_fn):
        try:
            sanitize_mol_fn(pose_bo)
        except Exception:
            pass
    try:
        matches = list(forward_matches_for_mol_fn(ref_mol, pose_bo, int(max_maps_i)) or [])
    except Exception:
        matches = []
    return list(matches[: int(max_maps_i)])


def build_redocking_mcs_payload(
    *,
    ref_mol=None,
    pose_mol=None,
    min_atoms=4,
    mcs_match_limit=32,
    find_mcs_fn=None,
    mcs_num_atoms_fn=None,
    mcs_smarts_fn=None,
    mol_from_smarts_fn=None,
    get_substruct_matches_fn=None,
):
    if not callable(find_mcs_fn):
        return {}
    try:
        min_atoms_i = int(min_atoms)
    except Exception:
        min_atoms_i = 4
    min_atoms_i = max(1, int(min_atoms_i))
    try:
        match_limit_i = int(mcs_match_limit)
    except Exception:
        match_limit_i = 32
    match_limit_i = max(1, int(match_limit_i))

    try:
        mcs = find_mcs_fn(ref_mol, pose_mol)
    except Exception:
        mcs = None
    if not mcs:
        return {}

    try:
        mcs_atoms = int(mcs_num_atoms_fn(mcs)) if callable(mcs_num_atoms_fn) else int(getattr(mcs, "numAtoms", 0) or 0)
    except Exception:
        mcs_atoms = 0
    try:
        mcs_smarts = str(mcs_smarts_fn(mcs) if callable(mcs_smarts_fn) else getattr(mcs, "smartsString", "") or "")
    except Exception:
        mcs_smarts = ""
    payload = {
        "num_atoms": int(mcs_atoms),
        "smarts": str(mcs_smarts),
        "ref_matches": [],
        "pose_matches": [],
    }
    if int(mcs_atoms) < int(min_atoms_i) or (not str(mcs_smarts)):
        return payload
    if (not callable(mol_from_smarts_fn)) or (not callable(get_substruct_matches_fn)):
        return payload
    try:
        patt = mol_from_smarts_fn(str(mcs_smarts))
    except Exception:
        patt = None
    if patt is None:
        return payload
    try:
        ref_matches = list(get_substruct_matches_fn(ref_mol, patt, int(match_limit_i), True) or [])
    except Exception:
        ref_matches = []
    try:
        pose_matches = list(get_substruct_matches_fn(pose_mol, patt, int(match_limit_i), True) or [])
    except Exception:
        pose_matches = []
    payload["ref_matches"] = list(ref_matches[: int(match_limit_i)])
    payload["pose_matches"] = list(pose_matches[: int(match_limit_i)])
    return payload


def build_redocking_candidate_map_config(
    *,
    overrides=None,
    min_atoms=4,
    mcs_match_limit=32,
    mcs_timeout=8,
    mcs_coverage_ratio=0.85,
):
    defaults = {
        "min_atoms": min_atoms,
        "mcs_match_limit": mcs_match_limit,
        "mcs_timeout": mcs_timeout,
        "mcs_coverage_ratio": mcs_coverage_ratio,
    }

    over = {}
    if isinstance(overrides, dict):
        over = dict(overrides)
    elif overrides is not None:
        try:
            over = dict(overrides)
        except Exception:
            over = {}

    def _norm_int(raw_val, fallback):
        try:
            fb = int(fallback)
        except Exception:
            fb = 1
        fb = max(1, int(fb))
        try:
            v = int(raw_val)
        except Exception:
            v = int(fb)
        return max(1, int(v))

    try:
        default_ratio = float(defaults["mcs_coverage_ratio"])
    except Exception:
        default_ratio = 0.85
    if (not isfinite(default_ratio)) or (default_ratio <= 0.0):
        default_ratio = 0.85

    raw_ratio = over.get("mcs_coverage_ratio", defaults["mcs_coverage_ratio"])
    try:
        mcs_coverage_ratio_f = float(raw_ratio)
    except Exception:
        mcs_coverage_ratio_f = float(default_ratio)
    if (not isfinite(mcs_coverage_ratio_f)) or (mcs_coverage_ratio_f <= 0.0):
        mcs_coverage_ratio_f = float(default_ratio)

    return {
        "min_atoms": _norm_int(over.get("min_atoms", defaults["min_atoms"]), defaults["min_atoms"]),
        "mcs_match_limit": _norm_int(
            over.get("mcs_match_limit", defaults["mcs_match_limit"]),
            defaults["mcs_match_limit"],
        ),
        "mcs_timeout": _norm_int(over.get("mcs_timeout", defaults["mcs_timeout"]), defaults["mcs_timeout"]),
        "mcs_coverage_ratio": float(mcs_coverage_ratio_f),
    }


def build_redocking_candidate_map_callback_bundle(
    *,
    ref_mol=None,
    pose_mol=None,
    nref=0,
    npose=0,
    min_atoms=4,
    mcs_match_limit=32,
    directional_get_substruct_matches_fn=None,
    directional_uniquify=False,
    directional_use_chirality=False,
    assign_bond_orders_fn=None,
    sanitize_mol_fn=None,
    find_mcs_fn=None,
    mcs_num_atoms_fn=None,
    mcs_smarts_fn=None,
    mcs_mol_from_smarts_fn=None,
    mcs_get_substruct_matches_fn=None,
):
    directional_match_fns = dict(
        build_redocking_directional_match_functions(
            ref_mol=ref_mol,
            pose_mol=pose_mol,
            get_substruct_matches_fn=directional_get_substruct_matches_fn,
            uniquify=bool(directional_uniquify),
            use_chirality=bool(directional_use_chirality),
        )
        or {}
    )

    def _bond_order_forward_matches(ref_obj, probe_obj, limit):
        if not callable(directional_get_substruct_matches_fn):
            return []
        try:
            lim_i = int(limit)
        except Exception:
            lim_i = 1
        lim_i = max(1, int(lim_i))
        try:
            matches = list(
                directional_get_substruct_matches_fn(
                    ref_obj,
                    probe_obj,
                    False,
                    False,
                )
                or []
            )
        except Exception:
            matches = []
        return list(matches[: int(lim_i)])

    def _bond_order_matches(limit):
        return list(
            build_redocking_bond_order_matches(
                ref_mol=ref_mol,
                pose_mol=pose_mol,
                nref=nref,
                npose=npose,
                max_maps=limit,
                assign_bond_orders_fn=assign_bond_orders_fn,
                sanitize_mol_fn=sanitize_mol_fn,
                forward_matches_for_mol_fn=_bond_order_forward_matches,
            )
            or []
        )

    def _mcs_get_substruct_matches(mol_obj, patt, limit, uniq):
        try:
            lim_i = int(limit)
        except Exception:
            lim_i = 1
        lim_i = max(1, int(lim_i))
        if callable(mcs_get_substruct_matches_fn):
            try:
                matches = list(mcs_get_substruct_matches_fn(mol_obj, patt, int(lim_i), bool(uniq)) or [])
            except Exception:
                matches = []
            return list(matches[: int(lim_i)])
        if not callable(directional_get_substruct_matches_fn):
            return []
        try:
            matches = list(
                directional_get_substruct_matches_fn(
                    mol_obj,
                    patt,
                    bool(uniq),
                    False,
                )
                or []
            )
        except Exception:
            matches = []
        return list(matches[: int(lim_i)])

    def _mcs_payload():
        return dict(
            build_redocking_mcs_payload(
                ref_mol=ref_mol,
                pose_mol=pose_mol,
                min_atoms=min_atoms,
                mcs_match_limit=mcs_match_limit,
                find_mcs_fn=find_mcs_fn,
                mcs_num_atoms_fn=mcs_num_atoms_fn,
                mcs_smarts_fn=mcs_smarts_fn,
                mol_from_smarts_fn=mcs_mol_from_smarts_fn,
                get_substruct_matches_fn=_mcs_get_substruct_matches,
            )
            or {}
        )

    return {
        "forward_matches_fn": directional_match_fns.get("forward_matches_fn"),
        "reverse_matches_fn": directional_match_fns.get("reverse_matches_fn"),
        "bond_order_matches_fn": _bond_order_matches,
        "mcs_payload_fn": _mcs_payload,
    }


def build_redocking_rdkit_candidate_map_callback_bundle(
    *,
    ref_mol=None,
    pose_mol=None,
    nref=0,
    npose=0,
    chem_module=None,
    all_chem_module=None,
    mcs_module=None,
    min_atoms=4,
    mcs_match_limit=32,
    mcs_timeout=8,
):
    map_cfg = dict(
        build_redocking_candidate_map_config(
            min_atoms=min_atoms,
            mcs_match_limit=mcs_match_limit,
            mcs_timeout=mcs_timeout,
        )
        or {}
    )

    def _directional_get_substruct_matches(query_obj, pattern_obj, uniq, use_chiral):
        if query_obj is None or pattern_obj is None:
            return []
        try:
            matches = list(
                query_obj.GetSubstructMatches(
                    pattern_obj,
                    uniquify=bool(uniq),
                    useChirality=bool(use_chiral),
                )
                or []
            )
        except Exception:
            matches = []
        return list(matches)

    assign_bond_orders_fn = None
    assign_bond_orders_raw_fn = getattr(all_chem_module, "AssignBondOrdersFromTemplate", None)
    if callable(assign_bond_orders_raw_fn):
        mol_clone_fn = getattr(chem_module, "Mol", None)

        def _assign_bond_orders(ref_obj, pose_obj):
            probe = pose_obj
            if callable(mol_clone_fn):
                try:
                    probe = mol_clone_fn(pose_obj)
                except Exception:
                    probe = pose_obj
            return assign_bond_orders_raw_fn(ref_obj, probe)

        assign_bond_orders_fn = _assign_bond_orders

    sanitize_mol_fn = None
    sanitize_raw_fn = getattr(chem_module, "SanitizeMol", None)
    if callable(sanitize_raw_fn):
        sanitize_mol_fn = sanitize_raw_fn

    find_mcs_fn = None
    find_mcs_raw_fn = getattr(mcs_module, "FindMCS", None)
    if callable(find_mcs_raw_fn):
        atom_compare = getattr(getattr(mcs_module, "AtomCompare", None), "CompareElements", None)
        bond_compare = getattr(getattr(mcs_module, "BondCompare", None), "CompareOrder", None)

        def _find_mcs(ref_obj, pose_obj):
            kwargs = {
                "ringMatchesRingOnly": True,
                "completeRingsOnly": True,
                "timeout": int(map_cfg.get("mcs_timeout", 8) or 8),
            }
            if atom_compare is not None:
                kwargs["atomCompare"] = atom_compare
            if bond_compare is not None:
                kwargs["bondCompare"] = bond_compare
            return find_mcs_raw_fn([ref_obj, pose_obj], **kwargs)

        find_mcs_fn = _find_mcs

    mcs_num_atoms_fn = lambda mcs_obj: int(getattr(mcs_obj, "numAtoms", 0) or 0)
    mcs_smarts_fn = lambda mcs_obj: str(getattr(mcs_obj, "smartsString", "") or "")

    mcs_mol_from_smarts_fn = None
    mol_from_smarts_raw_fn = getattr(chem_module, "MolFromSmarts", None)
    if callable(mol_from_smarts_raw_fn):
        mcs_mol_from_smarts_fn = lambda smarts: mol_from_smarts_raw_fn(str(smarts or ""))

    def _mcs_get_substruct_matches(mol_obj, patt, limit, uniq):
        if mol_obj is None or patt is None:
            return []
        try:
            lim_i = int(limit)
        except Exception:
            lim_i = 1
        lim_i = max(1, int(lim_i))
        try:
            matches = list(
                mol_obj.GetSubstructMatches(
                    patt,
                    uniquify=bool(uniq),
                )
                or []
            )
        except Exception:
            matches = []
        return list(matches[: int(lim_i)])

    return dict(
        build_redocking_candidate_map_callback_bundle(
            ref_mol=ref_mol,
            pose_mol=pose_mol,
            nref=nref,
            npose=npose,
            min_atoms=map_cfg.get("min_atoms", 4),
            mcs_match_limit=map_cfg.get("mcs_match_limit", 32),
            directional_get_substruct_matches_fn=_directional_get_substruct_matches,
            directional_uniquify=False,
            directional_use_chirality=False,
            assign_bond_orders_fn=assign_bond_orders_fn,
            sanitize_mol_fn=sanitize_mol_fn,
            find_mcs_fn=find_mcs_fn,
            mcs_num_atoms_fn=mcs_num_atoms_fn,
            mcs_smarts_fn=mcs_smarts_fn,
            mcs_mol_from_smarts_fn=mcs_mol_from_smarts_fn,
            mcs_get_substruct_matches_fn=_mcs_get_substruct_matches,
        )
        or {}
    )


def build_redocking_rdkit_callback_bundle_for_config(
    ref_mol,
    pose_mol,
    nref,
    npose,
    candidate_map_config,
    *,
    chem_module=None,
    all_chem_module=None,
    mcs_module=None,
):
    cfg = {}
    try:
        cfg = dict(candidate_map_config or {})
    except Exception:
        cfg = {}
    return dict(
        build_redocking_rdkit_candidate_map_callback_bundle(
            ref_mol=ref_mol,
            pose_mol=pose_mol,
            nref=nref,
            npose=npose,
            chem_module=chem_module,
            all_chem_module=all_chem_module,
            mcs_module=mcs_module,
            min_atoms=cfg.get("min_atoms", 4),
            mcs_match_limit=cfg.get("mcs_match_limit", 32),
            mcs_timeout=cfg.get("mcs_timeout", 8),
        )
        or {}
    )


def resolve_redocking_candidate_atom_maps(
    *,
    ref_mol=None,
    pose_mol=None,
    max_maps=256,
    candidate_map_config=None,
    build_callback_bundle_fn=None,
    unique_atom_maps_fn=None,
    get_atom_count_fn=None,
):
    if (ref_mol is None) or (pose_mol is None):
        return []

    def _atom_count(mol_obj):
        if callable(get_atom_count_fn):
            try:
                return int(get_atom_count_fn(mol_obj))
            except Exception:
                pass
        try:
            return int(mol_obj.GetNumAtoms())
        except Exception:
            return 0

    nref_i = max(0, int(_atom_count(ref_mol)))
    npose_i = max(0, int(_atom_count(pose_mol)))

    map_cfg = dict(
        build_redocking_candidate_map_config(
            overrides=candidate_map_config,
        )
        or {}
    )

    callback_bundle = {}
    if callable(build_callback_bundle_fn):
        try:
            callback_bundle = dict(
                build_callback_bundle_fn(
                    ref_mol,
                    pose_mol,
                    int(nref_i),
                    int(npose_i),
                    dict(map_cfg),
                )
                or {}
            )
        except Exception:
            callback_bundle = {}

    payload = build_redocking_candidate_atom_maps(
        nref=int(nref_i),
        npose=int(npose_i),
        max_maps=max_maps,
        forward_matches_fn=callback_bundle.get("forward_matches_fn"),
        reverse_matches_fn=callback_bundle.get("reverse_matches_fn"),
        bond_order_matches_fn=callback_bundle.get("bond_order_matches_fn"),
        mcs_payload_fn=callback_bundle.get("mcs_payload_fn"),
        unique_atom_maps_fn=unique_atom_maps_fn,
        min_atoms=map_cfg.get("min_atoms", 4),
        mcs_coverage_ratio=map_cfg.get("mcs_coverage_ratio", 0.85),
        mcs_match_limit=map_cfg.get("mcs_match_limit", 32),
    )
    return list(payload.get("atom_maps", []) or [])


def resolve_redocking_rdkit_candidate_atom_maps(
    *,
    ref_mol=None,
    pose_mol=None,
    max_maps=256,
    candidate_map_config=None,
    chem_module=None,
    all_chem_module=None,
    mcs_module=None,
    unique_atom_maps_fn=None,
    get_atom_count_fn=None,
):
    def _build_callback_bundle(ref_obj, pose_obj, nref_i, npose_i, cfg):
        return dict(
            build_redocking_rdkit_callback_bundle_for_config(
                ref_obj,
                pose_obj,
                int(nref_i),
                int(npose_i),
                cfg,
                chem_module=chem_module,
                all_chem_module=all_chem_module,
                mcs_module=mcs_module,
            )
            or {}
        )

    return list(
        resolve_redocking_candidate_atom_maps(
            ref_mol=ref_mol,
            pose_mol=pose_mol,
            max_maps=max_maps,
            candidate_map_config=candidate_map_config,
            build_callback_bundle_fn=_build_callback_bundle,
            unique_atom_maps_fn=unique_atom_maps_fn,
            get_atom_count_fn=get_atom_count_fn,
        )
        or []
    )


def build_redocking_candidate_atom_maps(
    *,
    nref=0,
    npose=0,
    max_maps=256,
    forward_matches_fn=None,
    reverse_matches_fn=None,
    bond_order_matches_fn=None,
    mcs_payload_fn=None,
    unique_atom_maps_fn=None,
    min_atoms=4,
    mcs_coverage_ratio=0.85,
    mcs_match_limit=32,
):
    try:
        nref_i = int(nref)
    except Exception:
        nref_i = 0
    try:
        npose_i = int(npose)
    except Exception:
        npose_i = 0
    try:
        max_maps_i = int(max_maps)
    except Exception:
        max_maps_i = 256
    max_maps_i = max(1, int(max_maps_i))
    try:
        min_atoms_i = int(min_atoms)
    except Exception:
        min_atoms_i = 4
    min_atoms_i = max(1, int(min_atoms_i))
    try:
        mcs_ratio = float(mcs_coverage_ratio)
    except Exception:
        mcs_ratio = 0.85
    if (not isfinite(mcs_ratio)) or (mcs_ratio <= 0.0):
        mcs_ratio = 0.85
    try:
        mcs_limit_i = int(mcs_match_limit)
    except Exception:
        mcs_limit_i = 32
    mcs_limit_i = max(1, int(mcs_limit_i))

    payload = {
        "ok": True,
        "nref": int(nref_i),
        "npose": int(npose_i),
        "max_maps": int(max_maps_i),
        "min_atoms": int(min_atoms_i),
        "atom_maps": [],
        "raw_map_count": 0,
        "strategy": "none",
        "used_mcs": False,
    }
    if nref_i < 3 or npose_i < 3:
        return payload

    maps = []

    def _append_forward(mt):
        if len(mt) != int(npose_i):
            return
        out = []
        for i in range(int(npose_i)):
            out.append((int(i), int(mt[i])))
        maps.append(list(out))

    def _append_reverse(mt):
        if len(mt) != int(nref_i):
            return
        out = []
        for i in range(int(nref_i)):
            out.append((int(mt[i]), int(i)))
        maps.append(list(out))

    if nref_i == npose_i and callable(forward_matches_fn):
        try:
            for mt in list(forward_matches_fn(int(max_maps_i)) or [])[: int(max_maps_i)]:
                _append_forward(list(mt or []))
        except Exception:
            pass
        if maps:
            payload["strategy"] = "forward"

    if (not maps) and nref_i == npose_i and callable(reverse_matches_fn):
        try:
            for mt in list(reverse_matches_fn(int(max_maps_i)) or [])[: int(max_maps_i)]:
                _append_reverse(list(mt or []))
        except Exception:
            pass
        if maps:
            payload["strategy"] = "reverse"

    if (not maps) and nref_i == npose_i and callable(bond_order_matches_fn):
        try:
            for mt in list(bond_order_matches_fn(int(max_maps_i)) or [])[: int(max_maps_i)]:
                _append_forward(list(mt or []))
        except Exception:
            pass
        if maps:
            payload["strategy"] = "bond_order"

    if (not maps) and callable(mcs_payload_fn):
        mcs_payload = {}
        try:
            mcs_payload = dict(mcs_payload_fn() or {})
        except Exception:
            mcs_payload = {}
        try:
            mcs_atoms = int(mcs_payload.get("num_atoms", 0) or 0)
        except Exception:
            mcs_atoms = 0
        has_smarts = bool(str(mcs_payload.get("smarts", "") or "").strip())
        if mcs_atoms >= int(min_atoms_i) and has_smarts:
            ref_matches = list(mcs_payload.get("ref_matches", []) or [])[: int(mcs_limit_i)]
            pose_matches = list(mcs_payload.get("pose_matches", []) or [])[: int(mcs_limit_i)]
            need = max(int(min_atoms_i), int(ceil(float(mcs_ratio) * float(min(nref_i, npose_i)))))
            tries = 0
            for pm in pose_matches:
                if tries >= int(max_maps_i):
                    break
                pm_list = list(pm or [])
                for rm in ref_matches:
                    if tries >= int(max_maps_i):
                        break
                    rm_list = list(rm or [])
                    nmap = min(len(pm_list), len(rm_list))
                    if nmap < int(need):
                        continue
                    amap = []
                    bad = False
                    for i in range(int(nmap)):
                        try:
                            amap.append((int(pm_list[i]), int(rm_list[i])))
                        except Exception:
                            bad = True
                            break
                    if bad or (not amap):
                        continue
                    maps.append(list(amap))
                    tries += 1
            if maps:
                payload["strategy"] = "mcs"
                payload["used_mcs"] = True

    uniq_fn = unique_atom_maps_fn if callable(unique_atom_maps_fn) else build_redocking_unique_atom_maps
    try:
        uniq = list(uniq_fn(maps, max_maps=int(max_maps_i)) or [])
    except Exception:
        uniq = list(build_redocking_unique_atom_maps(maps, max_maps=int(max_maps_i)) or [])
    payload["raw_map_count"] = int(len(maps))
    payload["atom_maps"] = list(uniq)
    return payload


def prepare_redocking_reference_flex_context(
    *,
    has_rdkit=False,
    ref_obj="",
    ref_fields=None,
    workspace_dir="",
    reference_mol_from_object_fn=None,
    flex_enabled=False,
    flex_rows=None,
    makedirs_fn=None,
):
    debug_messages = []
    ref_mol = None
    ref_msg = ""
    prep_dir = os.path.join(str(workspace_dir or ""), "redock_refprep")

    if bool(has_rdkit):
        try:
            if callable(makedirs_fn):
                makedirs_fn(prep_dir, exist_ok=True)
            else:
                os.makedirs(prep_dir, exist_ok=True)
        except Exception:
            pass
        if callable(reference_mol_from_object_fn):
            try:
                ref_mol, ref_msg = reference_mol_from_object_fn(ref_obj, ref_fields, prep_dir)
            except Exception as e:
                ref_mol = None
                ref_msg = str(e)
        if ref_mol is None:
            debug_messages.append(
                f"Redocking RMSD note: reference ligand RDKit conversion failed ({ref_msg}); affinity-only fallback if needed."
            )

    rows = list(flex_rows or [])
    flex_payload = validate_redocking_flex_selection(
        flex_enabled=bool(flex_enabled),
        flex_rows=rows,
        job_dir=str(workspace_dir or ""),
    )
    if not bool(flex_payload.get("ok", False)):
        return {
            "ok": False,
            "error": (
                "Flexible redocking is enabled, but no flexible residues are selected.\n"
                "Click 'Choose Flexible Residues...' first."
            ),
            "ref_mol": ref_mol,
            "ref_msg": str(ref_msg),
            "flex_enabled": bool(flex_enabled),
            "flex_rows": [],
            "debug_messages": list(debug_messages),
        }

    return {
        "ok": True,
        "error": "",
        "ref_mol": ref_mol,
        "ref_msg": str(ref_msg),
        "flex_enabled": bool(flex_enabled),
        "flex_rows": list(flex_payload.get("flex_rows", []) or []),
        "debug_messages": list(debug_messages),
    }


def prepare_redocking_reference_flex_runtime(
    *,
    has_rdkit=False,
    ref_obj="",
    ref_fields=None,
    workspace_dir_fn=None,
    reference_mol_from_object_fn=None,
    flex_enabled=False,
    flex_rows=None,
    makedirs_fn=None,
    append_debug_log_fn=None,
    context_builder_fn=None,
):
    workspace_dir = ""
    if callable(workspace_dir_fn):
        try:
            workspace_dir = str(workspace_dir_fn() or "")
        except Exception:
            workspace_dir = ""
    build_context = (
        context_builder_fn
        if callable(context_builder_fn)
        else prepare_redocking_reference_flex_context
    )
    try:
        payload = dict(
            build_context(
                has_rdkit=bool(has_rdkit),
                ref_obj=str(ref_obj or ""),
                ref_fields=tuple(ref_fields or ()),
                workspace_dir=str(workspace_dir or ""),
                reference_mol_from_object_fn=reference_mol_from_object_fn,
                flex_enabled=bool(flex_enabled),
                flex_rows=list(flex_rows or []),
                makedirs_fn=makedirs_fn,
            )
            or {}
        )
    except Exception as e:
        payload = {
            "ok": False,
            "error": str(e),
            "ref_mol": None,
            "ref_msg": "",
            "flex_enabled": bool(flex_enabled),
            "flex_rows": list(flex_rows or []),
            "debug_messages": [],
        }
    for msg in list(payload.get("debug_messages", []) or []):
        if str(msg or "").strip() and callable(append_debug_log_fn):
            try:
                append_debug_log_fn(str(msg))
            except Exception:
                pass
    return dict(payload or {})


def validate_redocking_receptor_prepare_result(
    receptor_input,
    rec_info,
    job_dir="",
    file_exists_fn=None,
):
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    if isinstance(rec_info, dict) and str(rec_info.get("error", "")) == "cancelled":
        return {
            "ok": False,
            "cancelled": True,
            "error": "Cancelled by user during receptor preparation.",
            "job_dir": str(job_dir or ""),
        }
    if (not receptor_input) or (not bool(exists_fn(receptor_input))):
        return {
            "ok": False,
            "cancelled": False,
            "error": f"Failed receptor preparation: {rec_info}",
            "job_dir": str(job_dir or ""),
        }
    return {
        "ok": True,
        "cancelled": False,
        "receptor_input": str(receptor_input),
        "rec_info": rec_info,
        "job_dir": str(job_dir or ""),
    }


def resolve_redocking_receptor_for_docking(
    receptor_input,
    stripped_file,
    removed_reference_atoms=0,
    file_exists_fn=None,
):
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    removed = int(removed_reference_atoms or 0)
    receptor_for_dock = str(receptor_input or "")
    if removed > 0 and bool(stripped_file) and bool(exists_fn(stripped_file)):
        receptor_for_dock = str(stripped_file)
    return {
        "receptor_for_dock": str(receptor_for_dock),
        "removed_reference_atoms": int(removed),
    }


def validate_redocking_flex_selection(flex_enabled=False, flex_rows=None, job_dir=""):
    rows = list(flex_rows or [])
    if bool(flex_enabled) and (not rows):
        return {
            "ok": False,
            "error": "Flexible redocking is enabled but no residues were selected.",
            "job_dir": str(job_dir or ""),
            "flex_rows": [],
        }
    return {
        "ok": True,
        "flex_rows": list(rows),
    }


def validate_redocking_flexible_prepare_result(rigid_in, flex_in, flex_info, job_dir=""):
    info = dict(flex_info or {}) if isinstance(flex_info, dict) else {"error": str(flex_info or "")}
    if (not rigid_in) or (not flex_in):
        return {
            "ok": False,
            "error": f"Flexible receptor preparation failed: {info.get('message', info.get('error', 'unknown'))}",
            "job_dir": str(job_dir or ""),
        }
    return {
        "ok": True,
        "receptor_for_dock": str(rigid_in),
        "flex_file": str(flex_in),
        "flex_info": dict(info),
    }


def prepare_redocking_receptor_flex_context(
    *,
    receptor_obj,
    job_dir,
    target_ph=7.4,
    ref_fields=None,
    flex_enabled=False,
    flex_residues=None,
    flow_label="redocking",
    auto_use_cached_hetatm=False,
    prepare_receptor_fn=None,
    strip_reference_residue_fn=None,
    prepare_flexible_receptor_fn=None,
    native_flexres_supported=False,
    build_native_flexres_spec_fn=None,
    file_exists_fn=None,
):
    if not callable(prepare_receptor_fn):
        return {
            "ok": False,
            "error": "prepare_receptor_fn is not callable",
            "job_dir": str(job_dir or ""),
        }
    receptor_input, rec_info = prepare_receptor_fn(
        receptor_obj,
        job_dir,
        target_ph=float(target_ph),
        flow_label=str(flow_label or "redocking"),
        auto_use_cached_hetatm=bool(auto_use_cached_hetatm),
    )
    receptor_prep_payload = validate_redocking_receptor_prepare_result(
        receptor_input=receptor_input,
        rec_info=rec_info,
        job_dir=job_dir,
        file_exists_fn=file_exists_fn,
    )
    if not bool(receptor_prep_payload.get("ok", False)):
        return dict(receptor_prep_payload)
    receptor_input = str(receptor_prep_payload.get("receptor_input", "") or "")
    rec_info = receptor_prep_payload.get("rec_info", rec_info)

    ext = os.path.splitext(str(receptor_input))[1].lower() or ".pdb"
    stripped_file = os.path.join(str(job_dir or ""), f"receptor_stripped{ext}")
    ch = ""
    ri = ""
    rn = ""
    try:
        if isinstance(ref_fields, (list, tuple)) and len(ref_fields) >= 3:
            ch = ref_fields[0]
            ri = ref_fields[1]
            rn = ref_fields[2]
    except Exception:
        ch = ""
        ri = ""
        rn = ""
    removed_reference_atoms = 0
    if callable(strip_reference_residue_fn):
        try:
            removed_reference_atoms = int(strip_reference_residue_fn(receptor_input, stripped_file, ch, ri, rn) or 0)
        except Exception:
            removed_reference_atoms = 0
    receptor_payload = resolve_redocking_receptor_for_docking(
        receptor_input=receptor_input,
        stripped_file=stripped_file,
        removed_reference_atoms=removed_reference_atoms,
        file_exists_fn=file_exists_fn,
    )
    receptor_for_dock = str(receptor_payload.get("receptor_for_dock", receptor_input) or receptor_input)
    removed_reference_atoms = int(receptor_payload.get("removed_reference_atoms", removed_reference_atoms) or removed_reference_atoms)
    prepared_pdb_for_flex = ""
    if isinstance(rec_info, dict):
        prepared_pdb_for_flex = str(rec_info.get("prepared_pdb", "") or "").strip()
    if prepared_pdb_for_flex and callable(strip_reference_residue_fn):
        prep_ext = os.path.splitext(str(prepared_pdb_for_flex))[1].lower() or ".pdb"
        stripped_prepared = os.path.join(str(job_dir or ""), f"receptor_prepared_stripped{prep_ext}")
        try:
            removed_prepared_atoms = int(
                strip_reference_residue_fn(prepared_pdb_for_flex, stripped_prepared, ch, ri, rn) or 0
            )
        except Exception:
            removed_prepared_atoms = 0
        if removed_prepared_atoms > 0 and bool(file_exists_fn(stripped_prepared) if callable(file_exists_fn) else os.path.isfile(stripped_prepared)):
            prepared_pdb_for_flex = str(stripped_prepared)

    flex_rows = list(flex_residues or [])
    flex_file = ""
    flex_info = {"ok": False, "used": False}
    flex_select_payload = validate_redocking_flex_selection(
        flex_enabled=bool(flex_enabled),
        flex_rows=flex_rows,
        job_dir=job_dir,
    )
    if not bool(flex_select_payload.get("ok", False)):
        return dict(flex_select_payload)
    flex_rows = list(flex_select_payload.get("flex_rows", []) or [])

    prepared_flex_error = None
    if bool(flex_enabled):
        if callable(prepare_flexible_receptor_fn):
            rigid_in, flex_in, flex_info = invoke_prepare_flexible_receptor(
                prepare_flexible_receptor_fn,
                receptor_for_dock,
                job_dir,
                flex_rows,
                prepared_pdb=prepared_pdb_for_flex,
                rec_info=rec_info,
            )
            flex_prep_payload = validate_redocking_flexible_prepare_result(
                rigid_in=rigid_in,
                flex_in=flex_in,
                flex_info=flex_info,
                job_dir=job_dir,
            )
            if bool(flex_prep_payload.get("ok", False)):
                receptor_for_dock = str(
                    flex_prep_payload.get("receptor_for_dock", receptor_for_dock) or receptor_for_dock
                )
                flex_file = str(flex_prep_payload.get("flex_file", "") or "")
                flex_info = dict(flex_prep_payload.get("flex_info", flex_info) or {})
            else:
                prepared_flex_error = dict(flex_prep_payload)
        elif not bool(native_flexres_supported):
            return {
                "ok": False,
                "error": "prepare_flexible_receptor_fn is not callable",
                "job_dir": str(job_dir or ""),
            }

    if bool(flex_enabled) and (not flex_file) and bool(native_flexres_supported):
        native_spec = ""
        if callable(build_native_flexres_spec_fn):
            try:
                native_spec = str(build_native_flexres_spec_fn(flex_rows) or "").strip()
            except Exception:
                native_spec = ""
        if not native_spec:
            return {
                "ok": False,
                "error": "Failed to build native smina flexible-residue specification.",
                "job_dir": str(job_dir or ""),
            }
        return {
            "ok": True,
            "job_dir": str(job_dir or ""),
            "receptor_input": str(receptor_input),
            "receptor_for_dock": str(receptor_for_dock),
            "removed_reference_atoms": int(removed_reference_atoms),
            "stripped_file": str(stripped_file),
            "rec_info": rec_info,
            "flex_rows": list(flex_rows),
            "flex_file": "",
            "flex_info": {
                "ok": True,
                "used": True,
                "native": True,
                "tool": "smina_native_flexres",
                "spec": str(native_spec),
            },
        }

    if bool(flex_enabled) and (not flex_file) and prepared_flex_error:
        return dict(prepared_flex_error)

    return {
        "ok": True,
        "job_dir": str(job_dir or ""),
        "receptor_input": str(receptor_input),
        "receptor_for_dock": str(receptor_for_dock),
        "removed_reference_atoms": int(removed_reference_atoms),
        "stripped_file": str(stripped_file),
        "rec_info": rec_info,
        "flex_rows": list(flex_rows),
        "flex_file": str(flex_file),
        "flex_info": dict(flex_info or {}),
    }


def validate_redocking_ligand_prepare_result(
    ligand_input,
    lig_info,
    job_dir="",
    file_exists_fn=None,
):
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    if (not ligand_input) or (not bool(exists_fn(ligand_input))):
        return {
            "ok": False,
            "error": f"Failed ligand preparation: {lig_info}",
            "job_dir": str(job_dir or ""),
        }
    info = dict(lig_info or {}) if isinstance(lig_info, dict) else {"raw": lig_info}
    qc = dict(info.get("qc", {}) or {}) if isinstance(info.get("qc", {}), dict) else {}
    if str(qc.get("status", "") or "").lower() == "red":
        errs = "; ".join(str(x) for x in list(qc.get("errors", []) or [])[:3])
        return {
            "ok": False,
            "error": f"Ligand preparation QC failed: {errs or lig_info}",
            "ligand_input": str(ligand_input),
            "lig_info": lig_info,
            "job_dir": str(job_dir or ""),
        }
    return {
        "ok": True,
        "ligand_input": str(ligand_input),
        "lig_info": lig_info,
        "job_dir": str(job_dir or ""),
    }


def redocking_ligand_qc_messages(lig_info):
    if not isinstance(lig_info, dict):
        return []
    qc = lig_info.get("qc", {})
    if not isinstance(qc, dict):
        return []
    status = str(qc.get("status", "") or "").strip().upper()
    if not status:
        return []
    parts = []
    errors = [str(x) for x in list(qc.get("errors", []) or []) if str(x).strip()]
    warnings = [str(x) for x in list(qc.get("warnings", []) or []) if str(x).strip()]
    if errors:
        parts.append("errors=" + " | ".join(errors[:2]))
    if warnings:
        parts.append("warnings=" + " | ".join(warnings[:2]))
    report = str(lig_info.get("qc_report", "") or "").strip()
    if report:
        parts.append(f"report={report}")
    suffix = "; ".join(parts)
    return [f"Redocking ligand QC: {status}" + (f" ({suffix})" if suffix else "")]


def resolve_redocking_ligand_charge_policy(
    ligand_file,
    *,
    preserve_input_charges=False,
    charge_mode="",
    parameterized_ligand_mol2="",
    file_exists_fn=None,
    samefile_or_equal_fn=None,
    chemically_equivalent_fn=None,
):
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    lig_file = str(ligand_file or "").strip()
    preserve = bool(preserve_input_charges)
    mode = str(charge_mode or "").strip().lower()
    pm = str(parameterized_ligand_mol2 or "").strip()
    used_template = False
    debug_message = ""

    lig_ext = os.path.splitext(lig_file)[1].lower()

    if mode == "preserve" and pm and bool(exists_fn(pm)):
        if lig_ext == ".pdbqt" and preserve:
            debug_message = (
                "Redocking ligand charge policy: using prepared ligand PDBQT from Ligand "
                "Source Prep as the redocking input."
            )
        elif not preserve:
            same = False
            if callable(samefile_or_equal_fn):
                try:
                    same = bool(samefile_or_equal_fn(lig_file, pm))
                except Exception:
                    same = False
            chem_eq = False
            if (not same) and callable(chemically_equivalent_fn):
                try:
                    chem_eq = bool(chemically_equivalent_fn(lig_file, pm))
                except Exception:
                    chem_eq = False
            if same or chem_eq:
                lig_file = str(pm)
                preserve = True
                used_template = True
                debug_message = (
                    "Redocking ligand charge policy: using charged MOL2 template "
                    f"for preserved charges ({pm})."
                )

    return {
        "ligand_file": str(lig_file),
        "preserve_input_charges": bool(preserve),
        "used_parameterized_template": bool(used_template),
        "parameterized_ligand_mol2": str(pm),
        "debug_message": str(debug_message),
    }


def resolve_redocking_ligand_smiles_hint(
    *,
    ligand_override_file="",
    ref_fields=None,
    pubchem_smiles_hint_fn=None,
    rcsb_smiles_by_het_fn=None,
):
    smiles_hint = ""
    hint_source = "none"
    residue_name = ""

    if callable(pubchem_smiles_hint_fn):
        try:
            smiles_hint = str(pubchem_smiles_hint_fn() or "").strip()
        except Exception:
            smiles_hint = ""
    if smiles_hint:
        hint_source = "pubchem"
        return {
            "smiles_hint": str(smiles_hint),
            "hint_source": str(hint_source),
            "residue_name": str(residue_name),
        }

    if str(ligand_override_file or "").strip():
        try:
            residue_name = (
                str(ref_fields[2] if isinstance(ref_fields, (list, tuple)) and len(ref_fields) >= 3 else "")
                .strip()
                .upper()
            )
        except Exception:
            residue_name = ""
        if residue_name and callable(rcsb_smiles_by_het_fn):
            try:
                smiles_hint = str(rcsb_smiles_by_het_fn(residue_name) or "").strip()
            except Exception:
                smiles_hint = ""
            if smiles_hint:
                hint_source = "rcsb"

    return {
        "smiles_hint": str(smiles_hint),
        "hint_source": str(hint_source),
        "residue_name": str(residue_name),
    }


def prepare_redocking_ligand_context(
    *,
    ligand_file,
    job_dir,
    target_ph=7.4,
    ligand_override_file="",
    ref_fields=None,
    preserve_input_charges=False,
    charge_mode="",
    parameterized_ligand_mol2="",
    file_exists_fn=None,
    samefile_or_equal_fn=None,
    chemically_equivalent_fn=None,
    pubchem_smiles_hint_fn=None,
    rcsb_smiles_by_het_fn=None,
    prepare_ligand_fn=None,
):
    debug_messages = []
    charge_policy_payload = resolve_redocking_ligand_charge_policy(
        ligand_file,
        preserve_input_charges=bool(preserve_input_charges),
        charge_mode=charge_mode,
        parameterized_ligand_mol2=parameterized_ligand_mol2,
        file_exists_fn=file_exists_fn,
        samefile_or_equal_fn=samefile_or_equal_fn,
        chemically_equivalent_fn=chemically_equivalent_fn,
    )
    lig_file = str(charge_policy_payload.get("ligand_file", ligand_file) or ligand_file)
    preserve = bool(charge_policy_payload.get("preserve_input_charges", False))
    charge_note = str(charge_policy_payload.get("debug_message", "") or "")
    if charge_note:
        debug_messages.append(charge_note)

    smiles_payload = resolve_redocking_ligand_smiles_hint(
        ligand_override_file=ligand_override_file,
        ref_fields=ref_fields,
        pubchem_smiles_hint_fn=pubchem_smiles_hint_fn,
        rcsb_smiles_by_het_fn=rcsb_smiles_by_het_fn,
    )
    smiles_hint = str(smiles_payload.get("smiles_hint", "") or "").strip()
    hint_source = str(smiles_payload.get("hint_source", "") or "")
    residue_name = str(smiles_payload.get("residue_name", "") or "").strip()
    if hint_source == "rcsb" and smiles_hint and residue_name:
        debug_messages.append(f"Redocking ligand bond-order hint from RCSB CCD: {residue_name}")

    if not callable(prepare_ligand_fn):
        return {
            "ok": False,
            "error": "prepare_ligand_fn is not callable",
            "job_dir": str(job_dir or ""),
            "debug_messages": list(debug_messages),
        }

    ligand_input, lig_info = prepare_ligand_fn(
        lig_file,
        job_dir,
        target_ph=float(target_ph),
        smiles_hint=smiles_hint,
        preserve_input_charges=preserve,
    )
    ligand_prep_payload = validate_redocking_ligand_prepare_result(
        ligand_input=ligand_input,
        lig_info=lig_info,
        job_dir=job_dir,
        file_exists_fn=file_exists_fn,
    )
    debug_messages.extend(redocking_ligand_qc_messages(lig_info))
    if not bool(ligand_prep_payload.get("ok", False)):
        out = dict(ligand_prep_payload)
        out["debug_messages"] = list(debug_messages)
        return out

    return {
        "ok": True,
        "job_dir": str(job_dir or ""),
        "ligand_file": str(lig_file),
        "preserve_input_charges": bool(preserve),
        "smiles_hint": str(smiles_hint),
        "hint_source": str(hint_source),
        "residue_name": str(residue_name),
        "ligand_input": str(ligand_prep_payload.get("ligand_input", "") or ""),
        "lig_info": lig_info,
        "debug_messages": list(debug_messages),
    }


def build_redocking_smina_plan(
    smina_bin,
    receptor_for_dock,
    ligand_input,
    center,
    size,
    params=None,
    flex_file="",
    native_flexres="",
    autobox_ligand="",
):
    pp = dict(params or {})

    def _int_param(key, default):
        try:
            value = pp.get(key, default)
            if value is None or str(value).strip() == "":
                return int(default)
            return int(value)
        except Exception:
            return int(default)

    def _float_param(key, default):
        try:
            value = pp.get(key, default)
            if value is None or str(value).strip() == "":
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    n_repeats = int(max(1, min(100, _int_param("repeats", 1))))
    seed0 = int(_int_param("seed0", 0))
    ex = int(_int_param("exhaustiveness", 48))
    modes = int(_int_param("num_modes", 40))
    energy_range = float(_float_param("energy_range", 6.0))
    min_rmsd = float(_float_param("min_rmsd", 0.5))
    auto_box_file = str(autobox_ligand or "").strip()
    native_spec = str(native_flexres or "").strip()
    use_flex_file = str(flex_file or "").strip()
    autobox_add = float(
        estimate_redocking_initial_autobox_add(
            base_autobox_add=_float_param("autobox_add", 4.0),
            native_flexres=native_spec,
            flex_file=use_flex_file,
            autobox_ligand=auto_box_file,
            receptor_structure=str(receptor_for_dock or ""),
        )
    )

    cx = float(center[0] if isinstance(center, (list, tuple)) and len(center) > 0 else 0.0)
    cy = float(center[1] if isinstance(center, (list, tuple)) and len(center) > 1 else 0.0)
    cz = float(center[2] if isinstance(center, (list, tuple)) and len(center) > 2 else 0.0)
    sx = float(size[0] if isinstance(size, (list, tuple)) and len(size) > 0 else 18.0)
    sy = float(size[1] if isinstance(size, (list, tuple)) and len(size) > 1 else 18.0)
    sz = float(size[2] if isinstance(size, (list, tuple)) and len(size) > 2 else 18.0)

    base_cmd = [
        str(smina_bin or ""),
        "-r",
        str(receptor_for_dock or ""),
        "-l",
        str(ligand_input or ""),
        "--exhaustiveness",
        str(int(ex)),
        "--num_modes",
        str(int(modes)),
        "--energy_range",
        f"{energy_range:.3f}",
        "--min_rmsd_filter",
        f"{min_rmsd:.3f}",
    ]
    if auto_box_file:
        base_cmd.extend(
            [
                "--autobox_ligand",
                str(auto_box_file),
                "--autobox_add",
                f"{autobox_add:.3f}",
            ]
        )
    else:
        base_cmd.extend(
            [
                "--center_x",
                f"{cx:.4f}",
                "--center_y",
                f"{cy:.4f}",
                "--center_z",
                f"{cz:.4f}",
                "--size_x",
                f"{sx:.4f}",
                "--size_y",
                f"{sy:.4f}",
                "--size_z",
                f"{sz:.4f}",
            ]
        )
    if native_spec:
        base_cmd.extend(["--flexres", str(native_spec)])
    elif use_flex_file:
        base_cmd.extend(["--flex", str(use_flex_file)])
    return {
        "n_repeats": int(n_repeats),
        "seed0": int(seed0),
        "base_cmd": list(base_cmd),
        "autobox_ligand": str(auto_box_file),
        "native_flexres": str(native_spec),
    }


def compute_redocking_rmsd_values_for_repeat(
    out_pose,
    poses_count,
    *,
    has_rdkit=False,
    ref_mol=None,
    convert_pdbqt_to_sdf_fn=None,
    load_pose_mols_fn=None,
    best_rmsd_fn=None,
    log_warning_fn=None,
):
    n = int(max(0, int(poses_count or 0)))
    rmsd_vals = [float("nan") for _ in range(n)]
    if (not bool(has_rdkit)) or (ref_mol is None):
        return rmsd_vals
    pose_file = str(out_pose or "")
    pose_sdf = ""
    can_eval = False
    if pose_file.lower().endswith(".pdbqt"):
        pose_sdf = os.path.join(os.path.dirname(pose_file), "docked_poses_rmsd.sdf")
        if callable(convert_pdbqt_to_sdf_fn):
            try:
                can_eval = bool(convert_pdbqt_to_sdf_fn(pose_file, pose_sdf))
            except Exception:
                can_eval = False
            # The converter rejects files RDKit cannot sanitize (fine for display), but PDBQT has no
            # bond orders, so a charged ligand re-perceived by Open Babel often cannot be (the 3PTB
            # benzamidinium came back with a four-bonded neutral N): every RMSD was NaN. The RMSD only
            # needs elements, connectivity and coordinates (compute_redocking_graph_rmsd).
            if (not can_eval) and os.path.isfile(pose_sdf) and os.path.getsize(pose_sdf) > 0:
                can_eval = True
    elif pose_file.lower().endswith(".sdf"):
        pose_sdf = pose_file
        can_eval = True
    if (not can_eval) or (not pose_sdf) or (not os.path.isfile(pose_sdf)):
        return rmsd_vals
    if (not callable(load_pose_mols_fn)) or (not callable(best_rmsd_fn)):
        return rmsd_vals
    try:
        p_mols = list(load_pose_mols_fn(str(pose_sdf)) or [])
        for i in range(min(len(rmsd_vals), len(p_mols))):
            pm = p_mols[i]
            if pm is None:
                continue
            try:
                rmsd_vals[i] = float(best_rmsd_fn(ref_mol, pm))
            except Exception:
                rmsd_vals[i] = float("nan")
    except Exception as e:
        if callable(log_warning_fn):
            try:
                log_warning_fn(str(e))
            except Exception:
                pass
    return rmsd_vals


def build_redocking_rows_for_repeat(
    rep,
    poses,
    rmsd_vals,
    *,
    system_label,
    receptor_obj,
    source_receptor_obj,
    job_dir,
):
    rows_i = []
    pose_list = list(poses or [])
    rmsd_list = list(rmsd_vals or [])
    for i, p in enumerate(pose_list):
        pp = dict(p or {})
        try:
            mode = int(pp.get("mode", i + 1))
        except Exception:
            mode = int(i + 1)
        try:
            aff = float(pp.get("affinity", float("nan")))
        except Exception:
            aff = float("nan")
        try:
            rms = float(rmsd_list[i]) if i < len(rmsd_list) else float("nan")
        except Exception:
            rms = float("nan")
        rows_i.append(
            {
                "system": str(system_label),
                "receptor": str(receptor_obj),
                "receptor_source": str(source_receptor_obj or receptor_obj),
                "mode": int(mode),
                "affinity": float(aff),
                "rmsd_ref": float(rms),
                "pose_object": "",
                "state": int(i + 1),
                "job_dir": str(job_dir),
                "repeat": int(rep),
            }
        )
    return rows_i


def _normalize_redocking_runtime_system_entry(entry):
    label = ""
    rec_obj = ""
    rec_src = ""
    member_tag = ""
    if isinstance(entry, (list, tuple)):
        if len(entry) > 0:
            label = str(entry[0] or "")
        if len(entry) > 1:
            rec_obj = str(entry[1] or "")
        if len(entry) > 2:
            rec_src = str(entry[2] or "")
        if len(entry) > 3:
            member_tag = str(entry[3] or "")
    if not rec_src:
        rec_src = str(rec_obj)
    if not member_tag:
        member_tag = f"{label}_model_001" if str(label).strip() else "model_001"
    return (str(label), str(rec_obj), str(rec_src), str(member_tag))


def _redocking_label_sort_key(label):
    lab = str(label or "").strip().lower()
    if lab.startswith("wt"):
        return (0, lab)
    if lab.startswith("mut"):
        return (1, lab)
    return (2, lab)


def _redocking_member_sort_key(member_tag):
    tag = str(member_tag or "").strip()
    if not tag:
        return ("", 10**9, "", "")
    low = str(tag).lower()
    match = re.search(r"(\d+)(?!.*\d)", low)
    if not match:
        return (low, 10**9, "", low)
    try:
        idx = int(match.group(1))
    except Exception:
        idx = 10**9
    prefix = str(low[: match.start()] or "")
    suffix = str(low[match.end() :] or "")
    return (prefix, int(idx), suffix, low)


def sort_redocking_runtime_systems(runtime_systems):
    items = []
    for ent in list(runtime_systems or []):
        label, rec_obj, rec_src, member_tag = _normalize_redocking_runtime_system_entry(ent)
        if not str(label or "").strip():
            continue
        items.append((str(label), str(rec_obj), str(rec_src), str(member_tag)))
    return sorted(
        items,
        key=lambda ent: (
            _redocking_label_sort_key(ent[0]),
            _redocking_member_sort_key(ent[3]),
            str(ent[1] or "").lower(),
        ),
    )


def dedupe_redocking_runtime_systems_by_signature(
    runtime_systems,
    *,
    signature_for_entry_fn=None,
):
    systems = [
        _normalize_redocking_runtime_system_entry(ent)
        for ent in list(runtime_systems or [])
        if str(_normalize_redocking_runtime_system_entry(ent)[0] or "").strip()
    ]
    if not callable(signature_for_entry_fn):
        return {
            "systems": list(systems),
            "dropped": [],
            "duplicate_count": 0,
        }

    kept = []
    dropped = []
    seen = {}
    for label, rec_obj, rec_src, member_tag in list(systems or []):
        sig = ""
        try:
            sig = str(signature_for_entry_fn(label, rec_obj, rec_src, member_tag) or "").strip()
        except Exception:
            sig = ""
        rec = {
            "label": str(label),
            "receptor_obj": str(rec_obj),
            "source_receptor": str(rec_src),
            "member_tag": str(member_tag),
            "signature": str(sig),
        }
        if (not sig) or (sig not in seen):
            kept.append((str(label), str(rec_obj), str(rec_src), str(member_tag)))
            if sig:
                seen[str(sig)] = dict(rec)
            continue
        first = dict(seen.get(str(sig), {}) or {})
        rec["duplicate_of"] = str(first.get("member_tag", "") or "")
        rec["duplicate_object"] = str(first.get("receptor_obj", "") or "")
        dropped.append(dict(rec))

    return {
        "systems": list(kept),
        "dropped": list(dropped),
        "duplicate_count": int(len(dropped)),
    }


def _median_finite(values):
    finite_vals = []
    for v in list(values or []):
        try:
            fv = float(v)
        except Exception:
            continue
        if isfinite(fv):
            finite_vals.append(float(fv))
    if not finite_vals:
        return float("nan")
    finite_vals = sorted(finite_vals)
    n = int(len(finite_vals))
    mid = n // 2
    if (n % 2) == 1:
        return float(finite_vals[mid])
    return float(0.5 * (finite_vals[mid - 1] + finite_vals[mid]))


def build_redocking_site_qc_payload(
    pair_records,
    anchor_keys,
    *,
    min_anchor_matches=4,
):
    payload = {
        "ok": False,
        "error": "",
        "reason": "",
        "site_rmsd_a": float("nan"),
        "site_anchor_matches": 0,
        "site_anchor_total": 0,
    }
    anchor_set = set()
    for item in list(anchor_keys or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        chain = str(item[0] or "").strip()
        resi = str(item[1] or "").strip()
        if resi:
            anchor_set.add((chain, resi))
    payload["site_anchor_total"] = int(len(anchor_set))
    if not anchor_set:
        payload["reason"] = "empty_anchor_keys"
        payload["error"] = "Reference site anchor set is empty."
        return payload

    min_matches = max(1, int(min_anchor_matches or 1))
    sq_sum = 0.0
    matched = 0
    for row in list(pair_records or []):
        if not isinstance(row, dict):
            continue
        key = (
            str((row or {}).get("ref_chain", "") or "").strip(),
            str((row or {}).get("ref_resi", "") or "").strip(),
        )
        if key not in anchor_set:
            continue
        ref_xyz = list((row or {}).get("ref_coord", []) or [])
        fit_xyz = list(
            (row or {}).get("aligned_coord", [])
            or (row or {}).get("fit_coord", [])
            or (row or {}).get("mobile_fit_coord", [])
            or []
        )
        try:
            ref_vals = [float(ref_xyz[0]), float(ref_xyz[1]), float(ref_xyz[2])]
            fit_vals = [float(fit_xyz[0]), float(fit_xyz[1]), float(fit_xyz[2])]
        except Exception:
            continue
        if not all(isfinite(v) for v in list(ref_vals + fit_vals)):
            continue
        sq_sum += (
            ((fit_vals[0] - ref_vals[0]) ** 2)
            + ((fit_vals[1] - ref_vals[1]) ** 2)
            + ((fit_vals[2] - ref_vals[2]) ** 2)
        )
        matched += 1

    payload["site_anchor_matches"] = int(matched)
    if matched < min_matches:
        payload["reason"] = "insufficient_anchor_matches"
        payload["error"] = (
            f"Only {int(matched)} matched reference-site anchor residues were found "
            f"(need at least {int(min_matches)})."
        )
        return payload

    rmsd = float((sq_sum / float(matched)) ** 0.5)
    payload["ok"] = bool(isfinite(rmsd))
    payload["site_rmsd_a"] = float(rmsd)
    if not bool(payload.get("ok", False)):
        payload["reason"] = "nonfinite_site_rmsd"
        payload["error"] = "Reference-site anchor RMSD is not finite."
    return payload


def resolve_redocking_site_qc_result(
    site_payload,
    *,
    rmsd_max_a=2.0,
    min_anchor_matches=4,
    system_label="",
    member_tag="",
):
    data = dict(site_payload or {}) if isinstance(site_payload, dict) else {"error": str(site_payload or "")}
    payload = {
        "ok": False,
        "error": "",
        "note": "",
        "site_rmsd_a": _safe_float(data.get("site_rmsd_a", data.get("rmsd", float("nan")))),
        "site_anchor_matches": 0,
        "site_anchor_total": 0,
        "site_anchor_cutoff_a": _safe_float(
            data.get("site_anchor_cutoff_a", data.get("anchor_cutoff_a", float("nan")))
        ),
    }
    try:
        payload["site_anchor_matches"] = int(
            data.get("site_anchor_matches", data.get("matched_pairs", 0)) or 0
        )
    except Exception:
        payload["site_anchor_matches"] = 0
    try:
        payload["site_anchor_total"] = int(
            data.get("site_anchor_total", data.get("anchor_total", payload["site_anchor_matches"]))
            or payload["site_anchor_matches"]
        )
    except Exception:
        payload["site_anchor_total"] = int(payload["site_anchor_matches"])

    min_matches = max(1, int(min_anchor_matches or 1))
    if int(payload["site_anchor_matches"]) < int(min_matches):
        payload["error"] = (
            f"local site QC has too few matched anchor residues "
            f"({int(payload['site_anchor_matches'])} < {int(min_matches)})"
        )
        return payload
    if not isfinite(float(payload["site_rmsd_a"])):
        payload["error"] = "local site QC RMSD is unavailable"
        return payload
    rmsd_limit = _safe_float(rmsd_max_a, default=2.0)
    if not isfinite(rmsd_limit):
        rmsd_limit = 2.0
    if float(payload["site_rmsd_a"]) > float(rmsd_limit):
        payload["error"] = (
            f"local site RMSD too high ({float(payload['site_rmsd_a']):.3f} A > {float(rmsd_limit):.3f} A); "
            "docking box would not cover the same pocket atoms consistently"
        )
        return payload

    payload["ok"] = True
    prefix = " ".join(
        [str(x).strip() for x in (system_label, member_tag) if str(x or "").strip()]
    ).strip()
    if not prefix:
        prefix = "Local site QC"
    cutoff = float(payload.get("site_anchor_cutoff_a", float("nan")))
    cutoff_txt = f" within {cutoff:.1f} A of reference ligand" if isfinite(cutoff) else ""
    payload["note"] = (
        f"{prefix}: anchor CA RMSD={float(payload['site_rmsd_a']):.3f} A over "
        f"{int(payload['site_anchor_matches'])}/{int(payload['site_anchor_total'])} residues{cutoff_txt}"
    )
    return payload


def prepare_redocking_runtime_systems_context(
    *,
    systems,
    params=None,
    use_modeler_ensemble=False,
    align_to_ref=False,
    ref_obj="",
    collect_modeler_ensemble_fn=None,
    align_receptor_to_reference_fn=None,
    force_align_when_ensemble=True,
    align_rmsd_max_a=4.0,
    site_qc_enabled=False,
    site_rmsd_max_a=2.0,
    site_qc_min_anchor_matches=4,
    site_anchor_cutoff_a=6.0,
):
    params_runtime = dict(params or {})
    modeler_mode = bool(use_modeler_ensemble)
    align_requested = bool(align_to_ref)
    site_qc_requested = bool(site_qc_enabled)
    ensemble_notes = []
    ensemble_counts = {}
    input_systems = []
    source_systems = list(systems or [])

    if modeler_mode:
        # one docking run per ensemble member; a requested seed is kept so the run is repeatable (it used to
        # be reset to 0 = a random seed every time)
        params_runtime["repeats"] = 1
        params_runtime["seed0"] = max(0, int(params_runtime.get("seed0", 0) or 0))
        if not callable(collect_modeler_ensemble_fn):
            return {
                "ok": False,
                "error": "collect_modeler_ensemble_fn is not callable",
            }
        for item in source_systems:
            label, rec_obj, _rec_src, _member = _normalize_redocking_runtime_system_entry(item)
            ens = dict(collect_modeler_ensemble_fn(label, rec_obj) or {})
            if not bool(ens.get("ok", False)):
                return {
                    "ok": False,
                    "error": str(ens.get("note", "Failed to collect MODELLER ensemble systems.")),
                }
            s_list = list(ens.get("systems", []) or [])
            if not s_list:
                s_list = [(str(label), str(rec_obj), str(rec_obj), f"{label}_model_001")]
            for s in s_list:
                input_systems.append(_normalize_redocking_runtime_system_entry(s))
            ensemble_counts[str(label)] = int(ens.get("model_count", len(s_list)) or len(s_list))
            note = str(ens.get("note", "") or "").strip()
            if note:
                ensemble_notes.append(note)
    else:
        for item in source_systems:
            label, rec_obj, _rec_src, _member = _normalize_redocking_runtime_system_entry(item)
            input_systems.append((str(label), str(rec_obj), str(rec_obj), f"{label}_model_001"))
    input_systems = list(sort_redocking_runtime_systems(input_systems) or [])

    align_notes = []
    if modeler_mode and (not align_requested) and bool(force_align_when_ensemble):
        align_requested = True
        align_notes.append("Alignment step was forced ON for MODELLER ensemble redocking.")
    if site_qc_requested and (not align_requested):
        align_requested = True
        align_notes.append("Alignment step was forced ON for local pocket consistency QC.")
    if (not align_requested) and str(ref_obj or "").strip() and any(
        str(rec) != str(ref_obj) for _lab, rec, _src, _mem in list(input_systems or [])
    ):
        # the search box and the crystal reference are in the reference frame: a receptor from
        # another object docked in its own frame would give a wrong box and a meaningless RMSD
        align_requested = True
        align_notes.append(
            "Alignment step was forced ON: some receptors are not the reference object, and the search box "
            "and crystal reference are in the reference frame."
        )

    runtime_systems = []
    site_qc_notes = []
    site_summary = {}
    if align_requested:
        if not callable(align_receptor_to_reference_fn):
            return {
                "ok": False,
                "error": "align_receptor_to_reference_fn is not callable",
            }
        align_summary = {}
        rmsd_limit = float(align_rmsd_max_a)
        for label, rec_obj, rec_src, member_tag in list(input_systems or []):
            ali = dict(align_receptor_to_reference_fn(ref_obj, rec_obj, system_label=label) or {})
            ali_ok = bool(ali.get("ok", False))
            detail = str(ali.get("details") or ali.get("error") or "unknown")
            try:
                ar = float(ali.get("rmsd", float("nan")))
            except Exception:
                ar = float("nan")
            if ali_ok and isfinite(ar) and ar > rmsd_limit:
                ali_ok = False
                detail = (
                    f"alignment RMSD too high ({ar:.3f} A > {rmsd_limit:.3f} A); "
                    "model likely incompatible with reference frame"
                )
            site_payload = {}
            if ali_ok and bool(site_qc_requested):
                site_payload = dict(
                    resolve_redocking_site_qc_result(
                        dict(ali or {}),
                        rmsd_max_a=float(site_rmsd_max_a),
                        min_anchor_matches=int(site_qc_min_anchor_matches or 4),
                        system_label=str(label),
                        member_tag=(str(member_tag) if bool(modeler_mode) else ""),
                    )
                    or {}
                )
                if not bool(site_payload.get("ok", False)):
                    ali_ok = False
                    detail = str(site_payload.get("error", "local site QC failed") or "local site QC failed")
            if ali_ok:
                run_obj = str(ali.get("object", rec_obj) or rec_obj)
                runtime_systems.append((str(label), str(run_obj), str(rec_src), str(member_tag)))
                if modeler_mode:
                    st = align_summary.setdefault(str(label), {"ok": 0, "fail": 0, "rmsd": []})
                    st["ok"] = int(st.get("ok", 0) or 0) + 1
                    if isfinite(ar):
                        st.setdefault("rmsd", []).append(float(ar))
                else:
                    if bool(ali.get("created", False)):
                        ali_detail = str(ali.get("details", "") or "").strip()
                        if isfinite(ar):
                            note = (
                                f"{label} alignment: {rec_obj} -> {run_obj} ({ali.get('method', 'align')}, RMSD={ar:.3f} A)"
                            )
                        else:
                            note = (
                                f"{label} alignment: {rec_obj} -> {run_obj} ({ali.get('method', 'align')})"
                            )
                        if ali_detail:
                            note += f" [{ali_detail}]"
                        align_notes.append(note)
                    else:
                        align_notes.append(f"{label} alignment: already in reference frame ({rec_obj}).")
                if bool(site_qc_requested):
                    st_site = site_summary.setdefault(str(label), {"ok": 0, "rmsd": [], "anchors": []})
                    st_site["ok"] = int(st_site.get("ok", 0) or 0) + 1
                    sr = _safe_float(site_payload.get("site_rmsd_a", float("nan")))
                    sa = _safe_float(site_payload.get("site_anchor_matches", float("nan")))
                    if isfinite(sr):
                        st_site.setdefault("rmsd", []).append(float(sr))
                    if isfinite(sa):
                        st_site.setdefault("anchors", []).append(float(sa))
                    if not bool(modeler_mode):
                        site_note = str(site_payload.get("note", "") or "").strip()
                        if site_note:
                            site_qc_notes.append(site_note)
            else:
                if modeler_mode:
                    return {
                        "ok": False,
                        "error": (
                            f"{label} ensemble member {member_tag} failed alignment; "
                            f"all ensemble models must align before redocking ({detail})."
                        ),
                    }
                return {
                    "ok": False,
                    "error": f"{label} alignment failed; cannot continue redocking safely ({detail}).",
                }
        if modeler_mode:
            for label in ("WT", "Mutant"):
                if label not in align_summary:
                    continue
                st = dict(align_summary.get(label, {}) or {})
                ok_n = int(st.get("ok", 0) or 0)
                fail_n = int(st.get("fail", 0) or 0)
                if ok_n <= 0:
                    return {
                        "ok": False,
                        "error": f"{label} ensemble has zero align-compatible members after QC.",
                    }
                rms_list = list(st.get("rmsd", []) or [])
                med_rms = float(_median_finite(rms_list))
                if isfinite(med_rms):
                    align_notes.append(
                        f"{label} ensemble alignment: ok={ok_n}, failed={fail_n}, median RMSD={med_rms:.3f} A"
                    )
                else:
                    align_notes.append(f"{label} ensemble alignment: ok={ok_n}, failed={fail_n}")
            if bool(site_qc_requested):
                for label in ("WT", "Mutant"):
                    if label not in site_summary:
                        continue
                    st_site = dict(site_summary.get(label, {}) or {})
                    ok_n = int(st_site.get("ok", 0) or 0)
                    rms_vals = list(st_site.get("rmsd", []) or [])
                    anchor_vals = list(st_site.get("anchors", []) or [])
                    med_site = float(_median_finite(rms_vals))
                    med_anchor = float(_median_finite(anchor_vals))
                    note = f"{label} ensemble local-site QC: ok={ok_n}"
                    if isfinite(med_site):
                        note += f", median anchor CA RMSD={med_site:.3f} A"
                    if isfinite(med_anchor):
                        note += f", median matched anchors={int(round(med_anchor))}"
                    site_qc_notes.append(note)
    else:
        runtime_systems = list(input_systems)
        align_notes.append("Alignment step: disabled.")
    runtime_systems = list(sort_redocking_runtime_systems(runtime_systems) or [])

    return {
        "ok": True,
        "error": "",
        "params": dict(params_runtime),
        "use_modeler_ensemble": bool(modeler_mode),
        "ensemble_notes": list(ensemble_notes),
        "ensemble_counts": dict(ensemble_counts),
        "input_systems": list(input_systems),
        "runtime_systems": list(runtime_systems),
        "align_notes": list(align_notes),
        "align_to_ref": bool(align_requested),
        "site_qc_enabled": bool(site_qc_requested),
        "site_qc_rmsd_max_a": float(site_rmsd_max_a),
        "site_qc_min_anchor_matches": int(site_qc_min_anchor_matches or 4),
        "site_anchor_cutoff_a": float(site_anchor_cutoff_a),
        "site_qc_notes": list(site_qc_notes),
    }


def prepare_redocking_runtime_systems_runtime(
    *,
    systems,
    params=None,
    use_modeler_ensemble=False,
    align_to_ref=False,
    ref_obj="",
    collect_modeler_ensemble_fn=None,
    align_receptor_to_reference_fn=None,
    force_align_when_ensemble=True,
    align_rmsd_max_a=6.0,
    site_qc_enabled=False,
    site_rmsd_max_a=2.0,
    site_qc_min_anchor_matches=4,
    site_anchor_cutoff_a=6.0,
    context_builder_fn=None,
):
    build_context = (
        context_builder_fn
        if callable(context_builder_fn)
        else prepare_redocking_runtime_systems_context
    )
    try:
        payload = dict(
            build_context(
                systems=list(systems or []),
                params=dict(params or {}),
                use_modeler_ensemble=bool(use_modeler_ensemble),
                align_to_ref=bool(align_to_ref),
                ref_obj=str(ref_obj or ""),
                collect_modeler_ensemble_fn=collect_modeler_ensemble_fn,
                align_receptor_to_reference_fn=align_receptor_to_reference_fn,
                force_align_when_ensemble=bool(force_align_when_ensemble),
                align_rmsd_max_a=float(align_rmsd_max_a),
                site_qc_enabled=bool(site_qc_enabled),
                site_rmsd_max_a=float(site_rmsd_max_a),
                site_qc_min_anchor_matches=int(site_qc_min_anchor_matches or 4),
                site_anchor_cutoff_a=float(site_anchor_cutoff_a),
            )
            or {}
        )
    except Exception as e:
        payload = {
            "ok": False,
            "error": str(e),
        }
    return dict(payload or {})


def run_redocking_hetatm_preflight(
    *,
    runtime_systems,
    selection_cache=None,
    collect_components_fn=None,
    prompt_selection_fn=None,
    component_cache_key_fn=None,
    line_builder_fn=None,
):
    if not callable(collect_components_fn):
        return {
            "ok": False,
            "cancelled": False,
            "error": "collect_components_fn is not callable",
            "lines": [],
            "selection_cache": dict(selection_cache or {}),
        }
    if not callable(prompt_selection_fn):
        return {
            "ok": False,
            "cancelled": False,
            "error": "prompt_selection_fn is not callable",
            "lines": [],
            "selection_cache": dict(selection_cache or {}),
        }
    if not callable(component_cache_key_fn):
        return {
            "ok": False,
            "cancelled": False,
            "error": "component_cache_key_fn is not callable",
            "lines": [],
            "selection_cache": dict(selection_cache or {}),
        }

    cache = selection_cache if isinstance(selection_cache, dict) else {}
    preflight_lines = []
    preflight_done = set()
    entries = list(runtime_systems or [])

    for ent in entries:
        label, rec_obj, rec_src, _member_tag = _normalize_redocking_runtime_system_entry(ent)
        prompt_obj = str(rec_src or rec_obj or "").strip()
        if (not prompt_obj) or (prompt_obj in preflight_done):
            continue
        preflight_done.add(prompt_obj)

        try:
            comps = list(collect_components_fn(prompt_obj) or [])
        except Exception as e:
            return {
                "ok": False,
                "cancelled": False,
                "error": str(e),
                "lines": list(preflight_lines),
                "selection_cache": dict(cache or {}),
            }
        if not comps:
            continue

        flow_label = f"redocking_{str(label or '').lower()}"
        try:
            pick = dict(prompt_selection_fn(prompt_obj, comps, flow_label=flow_label) or {})
        except Exception as e:
            return {
                "ok": False,
                "cancelled": False,
                "error": str(e),
                "lines": list(preflight_lines),
                "selection_cache": dict(cache or {}),
            }
        if bool(pick.get("cancelled", False)):
            return {
                "ok": False,
                "cancelled": True,
                "lines": list(preflight_lines),
                "selection_cache": dict(cache or {}),
            }

        selected = list(pick.get("selected", []) or [])
        sel_keys = []
        for comp in selected:
            try:
                ckey = str(component_cache_key_fn(comp) or "").strip()
            except Exception:
                ckey = ""
            if ckey:
                sel_keys.append(str(ckey))
        sel_keys = sorted(list(sel_keys))
        cache[str(prompt_obj)] = list(sel_keys)

        for ent2 in entries:
            _label2, obj2, src2, _member2 = _normalize_redocking_runtime_system_entry(ent2)
            if str(src2 or "").strip() == prompt_obj or str(obj2 or "").strip() == prompt_obj:
                if str(obj2 or "").strip():
                    cache[str(obj2)] = list(sel_keys)
                if str(src2 or "").strip():
                    cache[str(src2)] = list(sel_keys)

        counts = dict(pick.get("selected_counts", {}) or {})
        if callable(line_builder_fn):
            try:
                line = str(
                    line_builder_fn(
                        label=str(label),
                        prompt_obj=str(prompt_obj),
                        selected=list(selected),
                        selected_counts=dict(counts),
                        pick=dict(pick),
                    )
                    or ""
                ).strip()
            except Exception:
                line = ""
        else:
            line = (
                f"{label} HETATM keep ({prompt_obj}): "
                f"cof={int(counts.get('cofactor', 0))}, "
                f"ion={int(counts.get('ion', 0))}, "
                f"wat={int(counts.get('water', 0))}"
            )
        if line:
            preflight_lines.append(str(line))

    return {
        "ok": True,
        "cancelled": False,
        "error": "",
        "lines": list(preflight_lines),
        "selection_cache": dict(cache or {}),
    }


def run_redocking_hetatm_preflight_runtime(
    *,
    runtime_systems,
    selection_cache=None,
    collect_components_fn=None,
    prompt_selection_adapter_fn=None,
    component_cache_key_fn=None,
    allow_cached_auto=False,
    runtime_runner_fn=None,
):
    cache = selection_cache if isinstance(selection_cache, dict) else {}
    if not callable(prompt_selection_adapter_fn):
        return {
            "ok": False,
            "cancelled": False,
            "error": "prompt_selection_adapter_fn is not callable",
            "lines": [],
            "selection_cache": dict(cache or {}),
        }
    runner = runtime_runner_fn if callable(runtime_runner_fn) else run_redocking_hetatm_preflight

    def _prompt_selection(prompt_obj, comps, flow_label=""):
        return prompt_selection_adapter_fn(
            prompt_obj,
            comps,
            flow_label=flow_label,
            cache_dict=cache,
            allow_cached_auto=bool(allow_cached_auto),
        )

    try:
        payload = dict(
            runner(
                runtime_systems=list(runtime_systems or []),
                selection_cache=cache,
                collect_components_fn=collect_components_fn,
                prompt_selection_fn=_prompt_selection,
                component_cache_key_fn=component_cache_key_fn,
            )
            or {}
        )
    except Exception as e:
        payload = {
            "ok": False,
            "cancelled": False,
            "error": str(e),
            "lines": [],
            "selection_cache": dict(cache or {}),
        }
    return dict(payload or {})


def finalize_redock_member_run(run_result, system_label, member_tag, ensemble_mode=False):
    rr = dict(run_result or {})
    rr["ensemble_member"] = str(member_tag)
    rr["ensemble_mode"] = bool(ensemble_mode)
    rows_out = []
    for row in list(rr.get("rows", []) or []):
        row_i = dict(row or {})
        row_i["system"] = str(system_label)
        row_i["system_member"] = str(member_tag)
        rows_out.append(row_i)
    rr["rows"] = rows_out
    return rr


def run_redocking_runtime_systems(
    *,
    runtime_systems,
    run_single_system_fn,
    ref_obj,
    ref_fields,
    ref_mol,
    center,
    size,
    params=None,
    use_modeler_ensemble=False,
    flex_enabled=False,
    flex_residues=None,
    ligand_override_file="",
    auto_use_cached_hetatm=True,
    finalize_member_run_fn=None,
):
    if not callable(run_single_system_fn):
        raise TypeError("run_single_system_fn must be callable")
    finalize_fn = finalize_member_run_fn if callable(finalize_member_run_fn) else finalize_redock_member_run

    all_rows = []
    run_groups = {}
    for ent in list(runtime_systems or []):
        label = str(ent[0] if isinstance(ent, (list, tuple)) and len(ent) > 0 else "")
        rec_obj = str(ent[1] if isinstance(ent, (list, tuple)) and len(ent) > 1 else "")
        rec_src = str(ent[2] if isinstance(ent, (list, tuple)) and len(ent) > 2 else rec_obj)
        member_tag = str(
            ent[3] if isinstance(ent, (list, tuple)) and len(ent) > 3 else f"{label}_model_001"
        )
        try:
            rr = dict(
                run_single_system_fn(
                    label,
                    rec_obj,
                    ref_obj,
                    ref_fields,
                    ref_mol,
                    center,
                    size,
                    dict(params or {}),
                    source_receptor_obj=rec_src,
                    flex_enabled=bool(flex_enabled),
                    flex_residues=list(flex_residues or []),
                    auto_use_cached_hetatm=bool(auto_use_cached_hetatm),
                    ligand_override_file=str(ligand_override_file or ""),
                )
                or {}
            )
        except Exception as e:
            return {
                "ok": False,
                "cancelled": False,
                "error": str(e),
                "rows": list(all_rows),
                "run_groups": {str(k): list(v) for k, v in dict(run_groups).items()},
            }
        if bool(rr.get("cancelled", False)):
            return {
                "ok": False,
                "cancelled": True,
                "rows": list(all_rows),
                "run_groups": {str(k): list(v) for k, v in dict(run_groups).items()},
            }
        if not bool(rr.get("ok", False)):
            return {
                "ok": False,
                "cancelled": False,
                "error": f"{label} run failed ({member_tag}): {rr.get('error', 'unknown error')}",
                "rows": list(all_rows),
                "run_groups": {str(k): list(v) for k, v in dict(run_groups).items()},
            }
        try:
            rr_final = dict(
                finalize_fn(
                    rr,
                    system_label=str(label),
                    member_tag=str(member_tag),
                    ensemble_mode=bool(use_modeler_ensemble),
                )
                or {}
            )
        except Exception as e:
            return {
                "ok": False,
                "cancelled": False,
                "error": str(e),
                "rows": list(all_rows),
                "run_groups": {str(k): list(v) for k, v in dict(run_groups).items()},
            }
        rows_i = list(rr_final.get("rows", []) or [])
        run_groups.setdefault(str(label), []).append(rr_final)
        all_rows.extend(rows_i)

    return {
        "ok": True,
        "cancelled": False,
        "rows": list(all_rows),
        "run_groups": {str(k): list(v) for k, v in dict(run_groups).items()},
    }


def run_redocking_runtime_systems_runtime(
    *,
    runtime_systems,
    run_single_system_fn,
    ref_obj,
    ref_fields,
    ref_mol,
    center,
    size,
    params=None,
    use_modeler_ensemble=False,
    flex_enabled=False,
    flex_residues=None,
    ligand_override_file="",
    auto_use_cached_hetatm=True,
    runtime_runner_fn=None,
):
    run_payload = {
        "ok": False,
        "cancelled": False,
        "error": "unknown error",
        "rows": [],
        "run_groups": {},
    }
    runner = runtime_runner_fn if callable(runtime_runner_fn) else run_redocking_runtime_systems
    try:
        run_payload = dict(
            runner(
                runtime_systems=list(runtime_systems or []),
                run_single_system_fn=run_single_system_fn,
                ref_obj=ref_obj,
                ref_fields=ref_fields,
                ref_mol=ref_mol,
                center=center,
                size=size,
                params=dict(params or {}),
                use_modeler_ensemble=bool(use_modeler_ensemble),
                flex_enabled=bool(flex_enabled),
                flex_residues=list(flex_residues or []),
                ligand_override_file=str(ligand_override_file or ""),
                auto_use_cached_hetatm=bool(auto_use_cached_hetatm),
            )
            or {}
        )
    except Exception as e:
        run_payload = {
            "ok": False,
            "cancelled": False,
            "error": str(e),
            "rows": [],
            "run_groups": {},
        }
    return dict(run_payload or {})


def build_redock_run_map(ordered_labels, run_groups, aggregate_label_runs):
    if not callable(aggregate_label_runs):
        raise TypeError("aggregate_label_runs must be callable")
    labels = [str(lbl) for lbl in list(ordered_labels or [])]
    groups = dict(run_groups or {})
    run_map = {}
    for label in labels:
        grp = list(groups.get(label, []) or [])
        if not grp:
            continue
        if len(grp) == 1:
            run_map[label] = grp[0]
            continue
        agg = dict(aggregate_label_runs(label, grp) or {})
        if not bool(agg.get("ok", False)):
            raise RuntimeError(f"{label} aggregation failed: {agg.get('error', 'unknown error')}")
        run_map[label] = agg
    return run_map


def build_redocking_run_map_payload(
    *,
    systems,
    run_groups,
    aggregate_label_runs_fn,
    build_run_map_fn=None,
):
    labels = []
    for item in list(systems or []):
        try:
            labels.append(str(item[0]))
        except Exception:
            continue
    build_fn = build_run_map_fn if callable(build_run_map_fn) else build_redock_run_map
    if not callable(aggregate_label_runs_fn):
        return {
            "ok": False,
            "error": "aggregate_label_runs_fn is not callable",
            "run_map": {},
        }
    try:
        run_map = build_fn(
            ordered_labels=list(labels),
            run_groups=dict(run_groups or {}),
            aggregate_label_runs=aggregate_label_runs_fn,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "run_map": {},
        }
    return {
        "ok": True,
        "run_map": dict(run_map or {}),
    }


def build_redocking_run_map_runtime(
    *,
    systems,
    run_groups,
    aggregate_label_runs_fn,
    run_map_payload_builder_fn=None,
):
    build_payload = (
        run_map_payload_builder_fn
        if callable(run_map_payload_builder_fn)
        else build_redocking_run_map_payload
    )
    try:
        payload = dict(
            build_payload(
                systems=list(systems or []),
                run_groups=dict(run_groups or {}),
                aggregate_label_runs_fn=aggregate_label_runs_fn,
            )
            or {}
        )
    except Exception as e:
        payload = {
            "ok": False,
            "error": str(e),
            "run_map": {},
        }
    return dict(payload or {})


def collect_redocking_rows_from_run_map(run_map, preferred_labels=None):
    rm = dict(run_map or {})
    labels = list(preferred_labels or ["WT", "Mutant"])
    out_rows = []
    seen = set()
    for lbl in labels:
        key = str(lbl)
        seen.add(key)
        rr = rm.get(key)
        if rr:
            out_rows.extend(list((rr or {}).get("rows", []) or []))
    for key, rr in rm.items():
        if str(key) in seen:
            continue
        if rr:
            out_rows.extend(list((rr or {}).get("rows", []) or []))
    return list(out_rows)


def build_redocking_summary_payload(
    *,
    run_map,
    nref=0,
    center=(0.0, 0.0, 0.0),
    size=(0.0, 0.0, 0.0),
    use_reference_ligand=False,
    use_modeler_ensemble=False,
    ensemble_counts=None,
    repeats=1,
    site_qc_enabled=False,
    site_qc_rmsd_max_a=2.0,
    site_qc_min_anchor_matches=4,
    site_anchor_cutoff_a=6.0,
    site_qc_notes=None,
    summary_csv_path="",
    summary_lines_fn=None,
    comparison_rows_builder_fn=None,
    comparison_csv_writer_fn=None,
):
    lines_fn = summary_lines_fn if callable(summary_lines_fn) else build_redocking_summary_lines
    build_comparison_rows = (
        comparison_rows_builder_fn
        if callable(comparison_rows_builder_fn)
        else build_redocking_comparison_rows
    )
    write_comparison_csv = (
        comparison_csv_writer_fn
        if callable(comparison_csv_writer_fn)
        else write_redocking_comparison_summary_csv
    )
    rm = dict(run_map or {})
    lines = list(
        lines_fn(
            rm,
            int(nref or 0),
            tuple(center or (0.0, 0.0, 0.0)),
            tuple(size or (0.0, 0.0, 0.0)),
            use_reference_ligand=bool(use_reference_ligand),
            use_modeler_ensemble=bool(use_modeler_ensemble),
            ensemble_counts=dict(ensemble_counts or {}),
            repeats=int(repeats or 1),
            site_qc_enabled=bool(site_qc_enabled),
            site_qc_rmsd_max_a=float(site_qc_rmsd_max_a),
            site_qc_min_anchor_matches=int(site_qc_min_anchor_matches or 4),
            site_anchor_cutoff_a=float(site_anchor_cutoff_a),
            site_qc_notes=list(site_qc_notes or []),
        )
        or []
    )
    comparison_rows = list(build_comparison_rows(rm) or [])
    summary_csv = ""
    out_csv = str(summary_csv_path or "").strip()
    if out_csv and comparison_rows:
        summary_csv = str(write_comparison_csv(out_csv, comparison_rows) or "")
        if summary_csv:
            lines.append(f"Summary CSV: {summary_csv}")
    return {
        "summary_lines": list(lines),
        "summary_text": "\n".join(list(lines)),
        "comparison_rows": list(comparison_rows),
        "summary_csv": str(summary_csv or ""),
    }


def build_redocking_post_transform_runtime(
    *,
    context,
    preferred_labels=None,
    row_collector_fn=None,
):
    ctx = dict(context or {})
    run_map = dict(ctx.get("run_map", {}) or {})
    labels_pref = list(preferred_labels or [])
    if not labels_pref:
        labels_pref = [str(x) for x in list(ctx.get("system_labels", []) or []) if str(x or "").strip()]
    if not labels_pref:
        labels_pref = ["WT", "Mutant"]
    collect_fn = (
        row_collector_fn
        if callable(row_collector_fn)
        else collect_redocking_rows_from_run_map
    )
    rows = list(
        collect_fn(
            run_map=dict(run_map or {}),
            preferred_labels=list(labels_pref or []),
        )
        or []
    )
    return {
        "results": list(rows),
    }


def build_redocking_post_summary_runtime(
    *,
    context,
    summary_builder_fn=None,
):
    ctx = dict(context or {})
    build_summary = (
        summary_builder_fn
        if callable(summary_builder_fn)
        else build_redocking_summary_payload
    )
    summary_payload = dict(
        build_summary(
            run_map=dict(ctx.get("run_map", {}) or {}),
            nref=int(ctx.get("nref", 0) or 0),
            center=tuple(ctx.get("center", (0.0, 0.0, 0.0))),
            size=tuple(ctx.get("size", (0.0, 0.0, 0.0))),
            use_reference_ligand=bool(ctx.get("use_reference_ligand", False)),
            use_modeler_ensemble=bool(ctx.get("use_modeler_ensemble", False)),
            ensemble_counts=dict(ctx.get("ensemble_counts", {}) or {}),
            repeats=int(ctx.get("repeats", 1) or 1),
            site_qc_enabled=bool(ctx.get("site_qc_enabled", False)),
            site_qc_rmsd_max_a=float(ctx.get("site_qc_rmsd_max_a", 2.0) or 2.0),
            site_qc_min_anchor_matches=int(ctx.get("site_qc_min_anchor_matches", 4) or 4),
            site_anchor_cutoff_a=float(ctx.get("site_anchor_cutoff_a", 6.0) or 6.0),
            site_qc_notes=list(ctx.get("site_qc_notes", []) or []),
            summary_csv_path=str(ctx.get("summary_csv_path", "") or ""),
        )
        or {}
    )
    return {
        "summary": {
            "summary_lines": list(summary_payload.get("summary_lines", []) or []),
            "summary_text": str(summary_payload.get("summary_text", "") or ""),
            "comparison_rows": list(summary_payload.get("comparison_rows", []) or []),
            "summary_csv": str(summary_payload.get("summary_csv", "") or ""),
        }
    }


def build_redocking_post_apply_state_runtime(
    *,
    context,
    results,
):
    ctx = dict(context or {})
    return {
        "run_map": dict(ctx.get("run_map", {}) or {}),
        "results": list(results or []),
    }


def run_redocking_post_status_runtime(
    *,
    build_bestpose_combo_trajectories_fn=None,
    hide_all_pose_objects_fn=None,
    hide_all_runtime_receptors_fn=None,
    clear_selected_preview_objects_fn=None,
    update_ligand_status_label_fn=None,
    on_traj_visibility_changed_fn=None,
    load_selected_pose_fn=None,
    auto_load_selected_pose=False,
    append_debug_log_fn=None,
):
    if callable(build_bestpose_combo_trajectories_fn):
        try:
            build_bestpose_combo_trajectories_fn()
        except Exception as e:
            if callable(append_debug_log_fn):
                try:
                    append_debug_log_fn(f"Redocking combo-trajectory warning: {e}")
                except Exception:
                    pass

    for cleanup_fn in (
        hide_all_pose_objects_fn,
        hide_all_runtime_receptors_fn,
        clear_selected_preview_objects_fn,
    ):
        if not callable(cleanup_fn):
            continue
        try:
            cleanup_fn()
        except Exception:
            pass

    if callable(update_ligand_status_label_fn):
        update_ligand_status_label_fn()
    if callable(on_traj_visibility_changed_fn):
        on_traj_visibility_changed_fn()

    if bool(auto_load_selected_pose) and callable(load_selected_pose_fn):
        try:
            load_selected_pose_fn(silent=True)
        except TypeError:
            try:
                load_selected_pose_fn()
            except Exception:
                pass
        except Exception:
            pass

    return {}


def build_redocking_post_dialog_runtime(
    *,
    summary,
):
    summ = dict(summary or {}) if isinstance(summary, dict) else {}
    return {
        "summary_text": str(summ.get("summary_text", "") or ""),
        "summary_csv": str(summ.get("summary_csv", "") or ""),
    }


def build_redocking_post_pipeline_context(
    *,
    run_map,
    nref=0,
    center=(0.0, 0.0, 0.0),
    size=(0.0, 0.0, 0.0),
    use_reference_ligand=False,
    use_modeler_ensemble=False,
    ensemble_counts=None,
    repeats=1,
    system_labels=None,
    align_notes=None,
    site_qc_enabled=False,
    site_qc_rmsd_max_a=2.0,
    site_qc_min_anchor_matches=4,
    site_anchor_cutoff_a=6.0,
    site_qc_notes=None,
    summary_csv_path="",
):
    return {
        "run_map": dict(run_map or {}),
        "nref": int(nref or 0),
        "center": tuple(center or (0.0, 0.0, 0.0)),
        "size": tuple(size or (0.0, 0.0, 0.0)),
        "use_reference_ligand": bool(use_reference_ligand),
        "use_modeler_ensemble": bool(use_modeler_ensemble),
        "ensemble_counts": dict(ensemble_counts or {}),
        "repeats": int(repeats or 1),
        "system_labels": [str(x) for x in list(system_labels or []) if str(x or "").strip()],
        "align_notes": [str(x) for x in list(align_notes or []) if str(x or "").strip()],
        "site_qc_enabled": bool(site_qc_enabled),
        "site_qc_rmsd_max_a": float(site_qc_rmsd_max_a),
        "site_qc_min_anchor_matches": int(site_qc_min_anchor_matches or 4),
        "site_anchor_cutoff_a": float(site_anchor_cutoff_a),
        "site_qc_notes": [str(x) for x in list(site_qc_notes or []) if str(x or "").strip()],
        "summary_csv_path": str(summary_csv_path or ""),
    }


def best_pose_for_member(rows, label, member_tag):
    rows_local = list(rows or [])
    lab = str(label or "").strip()
    tag = str(member_tag or "").strip()

    candidates = []
    for row in rows_local:
        if not isinstance(row, dict):
            continue
        if str((row or {}).get("system", "") or "").strip() != lab:
            continue
        if tag and str((row or {}).get("system_member", "") or "").strip() != tag:
            continue
        candidates.append(dict(row))

    if (not candidates) and rows_local:
        for row in rows_local:
            if not isinstance(row, dict):
                continue
            if str((row or {}).get("system", "") or "").strip() == lab:
                candidates.append(dict(row))
    if not candidates:
        return None
    best = pick_top_scored_pose_row(candidates)
    return dict(best) if isinstance(best, dict) else None


def _is_better_state_row(new_row, cur_row):
    """Row shown for a trajectory state: the top-scored pose (see is_better_repeat)."""
    return bool(is_better_repeat(new_row, cur_row))

def build_redocking_state_row_map(
    *,
    rows,
    label,
    state_map=None,
    n_states=0,
    pick_best_pose_row_fn=None,
    resolve_state_for_member_fn=None,
    best_pose_for_member_fn=None,
    fill_missing_states_with_fallback=True,
    synthesize_missing_state_rows=False,
):
    all_rows = list(rows or [])
    system_label = str(label or "")
    smap = dict(state_map or {})
    n_states_i = int(n_states or 0)

    rows_label = [
        dict(r)
        for r in all_rows
        if isinstance(r, dict) and str((r or {}).get("system", "") or "").strip() == system_label
    ]
    if not rows_label:
        return {
            "rows_label": [],
            "fallback": None,
            "row_by_state": {},
        }

    pick_fn = pick_best_pose_row_fn if callable(pick_best_pose_row_fn) else pick_top_scored_pose_row
    resolve_fn = resolve_state_for_member_fn if callable(resolve_state_for_member_fn) else resolve_state_for_member
    best_member_fn = best_pose_for_member_fn if callable(best_pose_for_member_fn) else best_pose_for_member

    fallback = pick_fn(rows_label)
    fallback = dict(fallback) if isinstance(fallback, dict) else None
    row_by_state = {}
    fill_missing = bool(fill_missing_states_with_fallback)
    synthesize_missing = bool(synthesize_missing_state_rows)

    # First pass: explicit traj_state, then member-state mapping.
    for rr in rows_label:
        st_i = 0
        try:
            st_i = int((rr or {}).get("traj_state", 0) or 0)
        except Exception:
            st_i = 0
        if st_i <= 0:
            try:
                st_i = int(resolve_fn(smap, (rr or {}).get("system_member", ""), default_state=0))
            except Exception:
                st_i = 0
        if st_i <= 0:
            continue
        if n_states_i > 0:
            st_i = int(max(1, min(int(st_i), int(n_states_i))))
        prev = row_by_state.get(int(st_i))
        if _is_better_state_row(rr, prev):
            row_by_state[int(st_i)] = dict(rr)

    # Second pass: ensure known member->state mappings are represented.
    for member_tag, st in list(smap.items()):
        try:
            st_i = int(st)
        except Exception:
            st_i = 0
        if st_i <= 0:
            continue
        if n_states_i > 0:
            st_i = int(max(1, min(int(st_i), int(n_states_i))))
        if int(st_i) in row_by_state:
            continue
        rr = None
        try:
            rr = best_member_fn(all_rows, system_label, member_tag)
        except Exception:
            rr = None
        if isinstance(rr, dict) and str(member_tag or "").strip():
            rr_member = str((rr or {}).get("system_member", "") or "").strip()
            if rr_member.lower() != str(member_tag).strip().lower():
                rr = None
        if rr is None and synthesize_missing:
            rr = {
                "system": str(system_label),
                "system_member": str(member_tag),
                "traj_state": int(st_i),
                "pose_object": "",
                "state": 1,
            }
        elif rr is None:
            rr = fallback
        if isinstance(rr, dict):
            row_by_state[int(st_i)] = dict(rr)

    # Final safety: one row per trajectory frame.
    if n_states_i > 0:
        for st_i in range(1, int(n_states_i) + 1):
            if int(st_i) not in row_by_state:
                if fill_missing and (fallback is not None):
                    row_by_state[int(st_i)] = dict(fallback)
                elif synthesize_missing:
                    row_by_state[int(st_i)] = {
                        "system": str(system_label),
                        "system_member": "",
                        "traj_state": int(st_i),
                        "pose_object": "",
                        "state": 1,
                    }

    return {
        "rows_label": list(rows_label),
        "fallback": dict(fallback) if isinstance(fallback, dict) else None,
        "row_by_state": {int(k): dict(v) for k, v in dict(row_by_state).items()},
    }


def run_redocking_workflow_orchestrator(
    *,
    prepare_base_payload_fn,
    resolve_ligand_override_fn,
    prepare_refmol_flex_fn,
    prepare_runtime_systems_fn,
    prepare_runtime_views_fn=None,
    run_hetatm_preflight_fn=None,
    run_runtime_systems_fn=None,
    build_run_map_fn=None,
    build_post_context_fn=None,
):
    if not callable(prepare_base_payload_fn):
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "prepare_base", "error": "prepare_base_payload_fn is not callable"}
    if not callable(resolve_ligand_override_fn):
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "resolve_ligand_override", "error": "resolve_ligand_override_fn is not callable"}
    if not callable(prepare_refmol_flex_fn):
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "prepare_refmol_flex", "error": "prepare_refmol_flex_fn is not callable"}
    if not callable(prepare_runtime_systems_fn):
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "prepare_runtime_systems", "error": "prepare_runtime_systems_fn is not callable"}
    if not callable(run_hetatm_preflight_fn):
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "hetatm_preflight", "error": "run_hetatm_preflight_fn is not callable"}
    if not callable(run_runtime_systems_fn):
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "runtime_runs", "error": "run_runtime_systems_fn is not callable"}
    if not callable(build_run_map_fn):
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "aggregate_run_map", "error": "build_run_map_fn is not callable"}

    try:
        base_payload = dict(prepare_base_payload_fn() or {})
    except Exception as e:
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "prepare_base", "error": str(e)}
    if not base_payload:
        return {"ok": False, "aborted": True, "cancelled": False, "stage": "prepare_base", "error": ""}

    ref_obj = str(base_payload.get("ref_obj", "") or "")
    ref_fields = tuple(base_payload.get("ref_fields", ()) or ())
    center = tuple(base_payload.get("center", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0))
    size = tuple(base_payload.get("size", (18.0, 18.0, 18.0)) or (18.0, 18.0, 18.0))
    nref = int(base_payload.get("nref", 0) or 0)
    systems = list(base_payload.get("systems", []) or [])
    params = dict(base_payload.get("params", {}) or {})

    try:
        ligand_payload = dict(resolve_ligand_override_fn(ref_obj=ref_obj, ref_fields=ref_fields) or {})
    except Exception as e:
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "resolve_ligand_override", "error": str(e)}
    if not ligand_payload:
        return {"ok": False, "aborted": True, "cancelled": False, "stage": "resolve_ligand_override", "error": "", "base_payload": dict(base_payload)}
    ligand_override_file = str(ligand_payload.get("ligand_override_file", "") or "")
    use_ref_redock_ligand = bool(ligand_payload.get("use_reference_ligand", False))

    try:
        ref_flex_payload = dict(prepare_refmol_flex_fn(ref_obj=ref_obj, ref_fields=ref_fields) or {})
    except Exception as e:
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "prepare_refmol_flex", "error": str(e), "base_payload": dict(base_payload), "ligand_payload": dict(ligand_payload)}
    if not ref_flex_payload:
        return {"ok": False, "aborted": True, "cancelled": False, "stage": "prepare_refmol_flex", "error": "", "base_payload": dict(base_payload), "ligand_payload": dict(ligand_payload)}
    ref_mol = ref_flex_payload.get("ref_mol", None)
    redock_flex_enabled = bool(ref_flex_payload.get("flex_enabled", False))
    redock_flex_rows = list(ref_flex_payload.get("flex_rows", []) or [])

    try:
        runtime_payload = dict(prepare_runtime_systems_fn(ref_obj=ref_obj, systems=systems, params=params) or {})
    except Exception as e:
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "prepare_runtime_systems", "error": str(e)}
    if not runtime_payload:
        return {
            "ok": False,
            "aborted": True,
            "cancelled": False,
            "stage": "prepare_runtime_systems",
            "error": "",
            "base_payload": dict(base_payload),
            "ligand_payload": dict(ligand_payload),
            "ref_flex_payload": dict(ref_flex_payload),
        }

    params = dict(runtime_payload.get("params", {}) or {})
    use_modeler_ensemble = bool(runtime_payload.get("use_modeler_ensemble", False))
    ensemble_counts = dict(runtime_payload.get("ensemble_counts", {}) or {})
    input_systems = list(runtime_payload.get("input_systems", []) or [])
    runtime_systems = list(runtime_payload.get("runtime_systems", []) or [])

    if callable(prepare_runtime_views_fn):
        try:
            prepare_runtime_views_fn(runtime_systems=runtime_systems, input_systems=input_systems)
        except Exception as e:
            return {
                "ok": False,
                "aborted": False,
                "cancelled": False,
                "stage": "prepare_runtime_views",
                "error": str(e),
            }

    try:
        hetatm_payload = dict(run_hetatm_preflight_fn(runtime_systems=runtime_systems) or {})
    except Exception as e:
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "hetatm_preflight", "error": str(e)}
    if bool(hetatm_payload.get("cancelled", False)):
        return {
            "ok": False,
            "aborted": True,
            "cancelled": True,
            "stage": "hetatm_preflight",
            "error": "",
            "base_payload": dict(base_payload),
            "runtime_payload": dict(runtime_payload),
        }
    if not bool(hetatm_payload.get("ok", False)):
        return {
            "ok": False,
            "aborted": True,
            "cancelled": False,
            "stage": "hetatm_preflight",
            "error": str(hetatm_payload.get("error", "") or ""),
            "base_payload": dict(base_payload),
            "runtime_payload": dict(runtime_payload),
        }

    try:
        run_payload = dict(
            run_runtime_systems_fn(
                runtime_systems=runtime_systems,
                ref_obj=ref_obj,
                ref_fields=ref_fields,
                ref_mol=ref_mol,
                center=center,
                size=size,
                params=params,
                use_modeler_ensemble=bool(use_modeler_ensemble),
                flex_enabled=bool(redock_flex_enabled),
                flex_residues=list(redock_flex_rows),
                ligand_override_file=str(ligand_override_file or ""),
            )
            or {}
        )
    except Exception as e:
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "runtime_runs", "error": str(e)}
    if bool(run_payload.get("cancelled", False)):
        return {
            "ok": False,
            "aborted": True,
            "cancelled": True,
            "stage": "runtime_runs",
            "error": "",
            "base_payload": dict(base_payload),
            "runtime_payload": dict(runtime_payload),
            "hetatm_payload": dict(hetatm_payload),
        }
    if not bool(run_payload.get("ok", False)):
        return {
            "ok": False,
            "aborted": True,
            "cancelled": False,
            "stage": "runtime_runs",
            "error": str(run_payload.get("error", "") or ""),
            "base_payload": dict(base_payload),
            "runtime_payload": dict(runtime_payload),
            "hetatm_payload": dict(hetatm_payload),
        }

    try:
        aggregate_payload = dict(
            build_run_map_fn(
                systems=systems,
                run_groups=dict(run_payload.get("run_groups", {}) or {}),
            )
            or {}
        )
    except Exception as e:
        return {"ok": False, "aborted": False, "cancelled": False, "stage": "aggregate_run_map", "error": str(e)}
    if not bool(aggregate_payload.get("ok", False)):
        return {
            "ok": False,
            "aborted": True,
            "cancelled": False,
            "stage": "aggregate_run_map",
            "error": str(aggregate_payload.get("error", "") or ""),
            "base_payload": dict(base_payload),
            "run_payload": dict(run_payload),
        }

    run_map = dict(aggregate_payload.get("run_map", {}) or {})
    system_labels = []
    for ent in list(systems or []):
        try:
            lab = str(ent[0] or "").strip()
        except Exception:
            lab = ""
        if lab and lab not in system_labels:
            system_labels.append(str(lab))
    context_builder = build_post_context_fn if callable(build_post_context_fn) else build_redocking_post_pipeline_context
    pipeline_context = dict(
        context_builder(
            run_map=dict(run_map or {}),
            nref=int(nref),
            center=tuple(center),
            size=tuple(size),
            use_reference_ligand=bool(use_ref_redock_ligand),
            use_modeler_ensemble=bool(use_modeler_ensemble),
            ensemble_counts=dict(ensemble_counts or {}),
            repeats=int(params.get("repeats", 1) or 1),
            system_labels=list(system_labels or []),
            align_notes=list(runtime_payload.get("align_notes", []) or []),
            site_qc_enabled=bool(runtime_payload.get("site_qc_enabled", False)),
            site_qc_rmsd_max_a=float(runtime_payload.get("site_qc_rmsd_max_a", 2.0) or 2.0),
            site_qc_min_anchor_matches=int(runtime_payload.get("site_qc_min_anchor_matches", 4) or 4),
            site_anchor_cutoff_a=float(runtime_payload.get("site_anchor_cutoff_a", 6.0) or 6.0),
            site_qc_notes=list(runtime_payload.get("site_qc_notes", []) or []),
        )
        or {}
    )

    return {
        "ok": True,
        "aborted": False,
        "cancelled": False,
        "stage": "done",
        "pipeline_context": dict(pipeline_context or {}),
        "base_payload": dict(base_payload),
        "ligand_payload": dict(ligand_payload),
        "ref_flex_payload": dict(ref_flex_payload),
        "runtime_payload": dict(runtime_payload),
        "hetatm_payload": dict(hetatm_payload),
        "run_payload": dict(run_payload),
        "aggregate_payload": dict(aggregate_payload),
    }


def rows_best_per_member(rows):
    grouped = {}
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        rr = dict(row)
        sys_name = str(rr.get("system", "") or "").strip()
        if not sys_name:
            continue
        member = str(rr.get("system_member", "") or "").strip()
        if not member:
            member = f"{sys_name}_model_001"
        rr["system"] = str(sys_name)
        rr["system_member"] = str(member)
        grouped.setdefault((str(sys_name), str(member)), []).append(rr)
    out = []
    for _key, grp in grouped.items():
        best = pick_top_scored_pose_row(grp)
        if isinstance(best, dict):
            out.append(dict(best))

    def _sys_ord(v):
        s = str(v or "").strip().lower()
        if s.startswith("wt"):
            return 0
        if s.startswith("mut"):
            return 1
        return 2

    def _member_idx(v):
        s = str(v or "")
        m = re.search(r"(\d+)$", s)
        if not m:
            return 10**9
        try:
            return int(m.group(1))
        except Exception:
            return 10**9

    def _flt(v, default):
        try:
            return float(v)
        except Exception:
            return float(default)

    out = sorted(
        out,
        key=lambda r: (
            _sys_ord((r or {}).get("system", "")),
            _member_idx((r or {}).get("system_member", "")),
            (
                _flt((r or {}).get("rmsd_ref", float("nan")), float("inf"))
                if isfinite(_flt((r or {}).get("rmsd_ref", float("nan")), float("nan")))
                else float("inf")
            ),
            _flt((r or {}).get("affinity", float("inf")), float("inf")),
        ),
    )
    return out


def build_redocking_table_rows(
    rows,
    *,
    views=None,
    rows_best_per_member_fn=None,
    resolve_state_for_member_fn=None,
):
    best_rows_fn = rows_best_per_member_fn if callable(rows_best_per_member_fn) else rows_best_per_member
    resolve_fn = resolve_state_for_member_fn if callable(resolve_state_for_member_fn) else resolve_state_for_member

    sorted_rows = list(best_rows_fn(rows) or [])
    views_map = dict(views or {})
    rank_by_system = {}
    out_rows = []

    for rr in sorted_rows:
        if not isinstance(rr, dict):
            continue
        row = dict(rr)
        sys_name = str((row or {}).get("system", "") or "")
        rank = int(rank_by_system.get(sys_name, 0) or 0) + 1
        rank_by_system[sys_name] = int(rank)

        member = str((row or {}).get("system_member", "") or "").strip()
        if member and member not in ("WT_model_001", "Mutant_model_001"):
            system_display = f"{sys_name} [{member}]"
        else:
            system_display = str(sys_name)

        vinfo = dict((views_map.get(str(sys_name), {}) or {}))
        st_map = dict(vinfo.get("state_map", {}) or {})
        try:
            n_states = int(vinfo.get("n_states", 0) or 0)
        except Exception:
            n_states = 0
        try:
            traj_state = int(resolve_fn(st_map, member, default_state=rank))
        except Exception:
            traj_state = int(rank)
        if n_states > 0:
            traj_state = int(max(1, min(int(traj_state), int(n_states))))

        row["rank"] = int(rank)
        row["system_display"] = str(system_display)
        row["traj_state"] = int(traj_state)
        out_rows.append(row)

    return list(out_rows)


def redocking_traj_visible_for_label(label, *, show_wt=True, show_mut=False, default_visible=True):
    lab = str(label or "").strip().lower()
    if lab.startswith("wt"):
        return bool(show_wt)
    if lab.startswith("mut"):
        return bool(show_mut)
    return bool(default_visible)


def resolve_redocking_traj_state(
    *,
    explicit_state=0,
    state_map=None,
    member_tag="",
    n_states=0,
    resolve_state_for_member_fn=None,
    default_state=1,
):
    resolve_fn = resolve_state_for_member_fn if callable(resolve_state_for_member_fn) else resolve_state_for_member
    try:
        state = int(explicit_state or 0)
    except Exception:
        state = 0
    if state <= 0:
        try:
            state = int(resolve_fn(dict(state_map or {}), member_tag, default_state=0))
        except Exception:
            state = 0
    if state <= 0:
        state = int(default_state or 1)
    try:
        n_states_i = int(n_states or 0)
    except Exception:
        n_states_i = 0
    if n_states_i > 0:
        state = int(max(1, min(int(state), int(n_states_i))))
    return int(state)


def build_redocking_traj_focus_visibility_actions(
    *,
    traj_object="",
    frame_object="",
    pose_frame_object="",
    combo_object="",
    visible=True,
    known_objects=None,
    prefer_traj_object=False,
):
    traj_obj = str(traj_object or "").strip()
    frame_obj = str(frame_object or "").strip()
    pose_frame_obj = str(pose_frame_object or "").strip()
    combo_obj = str(combo_object or "").strip()
    known = set(known_objects or [])
    vis = bool(visible)

    actions = []
    if combo_obj and combo_obj in known:
        actions.append({"action": ("enable" if vis else "disable"), "object": str(combo_obj)})
        for obj in (frame_obj, pose_frame_obj):
            if obj and obj in known:
                actions.append({"action": "disable", "object": str(obj)})
        return list(actions)

    if bool(prefer_traj_object) and traj_obj and traj_obj in known:
        actions.append({"action": ("enable" if vis else "disable"), "object": str(traj_obj)})
        if frame_obj and frame_obj in known:
            actions.append({"action": "disable", "object": str(frame_obj)})
        if pose_frame_obj and pose_frame_obj in known:
            actions.append({"action": ("enable" if vis else "disable"), "object": str(pose_frame_obj)})
        return list(actions)

    for obj in (frame_obj, pose_frame_obj):
        if obj and obj in known:
            actions.append({"action": ("enable" if vis else "disable"), "object": str(obj)})
    return list(actions)


def prepare_redocking_traj_frame_payload(
    *,
    label,
    info,
    member_tag="",
    explicit_state=0,
    known_objects=None,
    resolve_state_for_member_fn=None,
    best_pose_for_member_fn=None,
    visible_for_label_fn=None,
    default_visible=True,
):
    info_local = dict(info or {})
    traj_obj = str(info_local.get("traj_object", "") or "").strip()
    frame_obj = str(info_local.get("frame_object", "") or "").strip()
    pose_frame_obj = str(info_local.get("pose_frame_object", "") or "").strip()
    combo_obj = str(info_local.get("combo_object", "") or "").strip()
    state_map = dict(info_local.get("state_map", {}) or {})
    try:
        n_states = int(info_local.get("n_states", 0) or 0)
    except Exception:
        n_states = 0

    payload = {
        "ok": False,
        "reason": "",
        "label": str(label or ""),
        "member_tag": str(member_tag or ""),
        "state": 0,
        "traj_object": str(traj_obj),
        "frame_object": str(frame_obj),
        "pose_frame_object": str(pose_frame_obj),
        "combo_object": str(combo_obj),
        "clear_frame_object": False,
        "clear_pose_frame_object": False,
        "prefer_traj_object": False,
        "sync_combo_state": False,
        "sync_traj_state": False,
        "overlay_pose_object": "",
        "overlay_pose_state": 1,
        "overlay_available": False,
        "visible": bool(default_visible),
        "visibility_actions": [],
    }

    if (not traj_obj) or (not frame_obj):
        payload["reason"] = "missing_traj_or_frame"
        return payload

    known = set(known_objects or [])
    if traj_obj not in known:
        payload["reason"] = "traj_not_loaded"
        return payload

    state = resolve_redocking_traj_state(
        explicit_state=explicit_state,
        state_map=dict(state_map or {}),
        member_tag=str(member_tag or ""),
        n_states=int(n_states),
        resolve_state_for_member_fn=resolve_state_for_member_fn,
        default_state=1,
    )
    payload["state"] = int(state)
    payload["clear_frame_object"] = bool(frame_obj and frame_obj in known)
    payload["clear_pose_frame_object"] = bool(pose_frame_obj and pose_frame_obj in known)
    payload["prefer_traj_object"] = bool((not combo_obj) and traj_obj and (traj_obj in known) and (int(n_states) > 1))
    payload["sync_combo_state"] = bool(combo_obj and combo_obj in known)
    payload["sync_traj_state"] = bool(payload["prefer_traj_object"])

    best_rec = None
    if callable(best_pose_for_member_fn):
        try:
            best_rec = best_pose_for_member_fn(str(label or ""), str(member_tag or ""))
        except Exception:
            best_rec = None
    if isinstance(best_rec, dict):
        bp_obj = str(best_rec.get("pose_object", "") or "").strip()
        try:
            bp_state = int(best_rec.get("state", 1) or 1)
        except Exception:
            bp_state = 1
        bp_state = max(1, int(bp_state))
        payload["overlay_pose_object"] = str(bp_obj)
        payload["overlay_pose_state"] = int(bp_state)
        payload["overlay_available"] = bool(bp_obj and bp_obj in known)

    vis = bool(default_visible)
    if callable(visible_for_label_fn):
        try:
            vis = bool(visible_for_label_fn(str(label or "")))
        except Exception:
            vis = bool(default_visible)
    payload["visible"] = bool(vis)
    payload["visibility_actions"] = list(
        build_redocking_traj_focus_visibility_actions(
            traj_object=str(traj_obj),
            frame_object=str(frame_obj),
            pose_frame_object=str(pose_frame_obj),
            combo_object=str(combo_obj),
            visible=bool(vis),
            known_objects=set(known or set()),
            prefer_traj_object=bool(payload["prefer_traj_object"]),
        )
    )
    payload["ok"] = True
    return payload


def run_redocking_traj_frame_runtime(
    *,
    label,
    info,
    member_tag="",
    explicit_state=0,
    known_objects=None,
    resolve_state_for_member_fn=None,
    best_pose_for_member_fn=None,
    visible_for_label_fn=None,
    default_visible=True,
    prepare_traj_frame_payload_fn=None,
    model_selection_fn=None,
    sync_frame_fn=None,
    set_state_fn=None,
    delete_object_fn=None,
    create_object_fn=None,
    hide_fn=None,
    show_fn=None,
    color_fn=None,
    group_fn=None,
    apply_ball_and_stick_style_fn=None,
    apply_element_colors_fn=None,
    get_known_objects_fn=None,
    build_focus_actions_fn=None,
    apply_object_actions_fn=None,
):
    prepare_fn = (
        prepare_traj_frame_payload_fn
        if callable(prepare_traj_frame_payload_fn)
        else prepare_redocking_traj_frame_payload
    )
    payload = dict(
        prepare_fn(
            label=str(label or ""),
            info=dict(info or {}),
            member_tag=member_tag,
            explicit_state=explicit_state,
            known_objects=set(known_objects or set()),
            resolve_state_for_member_fn=resolve_state_for_member_fn,
            best_pose_for_member_fn=best_pose_for_member_fn,
            visible_for_label_fn=visible_for_label_fn,
            default_visible=bool(default_visible),
        )
        or {}
    )
    if not bool(payload.get("ok", False)):
        return {
            "ok": False,
            "frame_object": "",
            "state": 0,
            "payload": dict(payload or {}),
            "actions": [],
            "error": str(payload.get("reason", "") or ""),
        }

    traj_obj = str(payload.get("traj_object", "") or "").strip()
    frame_obj = str(payload.get("frame_object", "") or "").strip()
    pose_frame_obj = str(payload.get("pose_frame_object", "") or "").strip()
    combo_obj = str(payload.get("combo_object", "") or "").strip()
    try:
        state = int(payload.get("state", 0) or 0)
    except Exception:
        state = 0
    if (not traj_obj) or (not frame_obj) or state <= 0:
        return {
            "ok": False,
            "frame_object": "",
            "state": 0,
            "payload": dict(payload or {}),
            "actions": [],
            "error": "invalid_frame_payload",
        }

    try:
        if bool(payload.get("sync_combo_state", False)) and combo_obj:
            synced = False
            if callable(sync_frame_fn):
                try:
                    sync_frame_fn(int(state))
                    synced = True
                except Exception:
                    synced = False
            # Do not pin a multi-state combo object to one state when frame sync is available.
            # PyMOL playback should continue to advance the visible trajectory after focusing.
            if (not synced) and callable(set_state_fn):
                try:
                    set_state_fn(int(state), combo_obj)
                except Exception:
                    try:
                        set_state_fn(int(state), None)
                    except Exception:
                        pass
        elif bool(payload.get("sync_traj_state", False)) and traj_obj:
            synced = False
            if callable(sync_frame_fn):
                try:
                    sync_frame_fn(int(state))
                    synced = True
                except Exception:
                    synced = False
            # Same rationale as combo objects: prefer frame sync over object-level state pinning.
            if (not synced) and callable(set_state_fn):
                try:
                    set_state_fn(int(state), traj_obj)
                except Exception:
                    try:
                        set_state_fn(int(state), None)
                    except Exception:
                        pass

        if bool(payload.get("clear_frame_object", False)) and callable(delete_object_fn):
            try:
                delete_object_fn(frame_obj)
            except Exception:
                pass

        if callable(create_object_fn):
            selection = str(traj_obj)
            if callable(model_selection_fn):
                selection = str(model_selection_fn(traj_obj) or traj_obj)
            create_object_fn(frame_obj, selection, int(state), 1)

        if callable(hide_fn):
            hide_fn("everything", f"({frame_obj}) and not polymer")
        if callable(show_fn):
            show_fn("cartoon", f"({frame_obj}) and polymer")
        if callable(color_fn):
            color_fn("white", f"({frame_obj}) and polymer")
        if callable(group_fn):
            group_fn("Redocking.Trajectory", frame_obj)

        if pose_frame_obj:
            if bool(payload.get("clear_pose_frame_object", False)) and callable(delete_object_fn):
                try:
                    delete_object_fn(pose_frame_obj)
                except Exception:
                    pass
            if bool(payload.get("overlay_available", False)):
                bp_obj = str(payload.get("overlay_pose_object", "") or "").strip()
                try:
                    bp_state = int(payload.get("overlay_pose_state", 1) or 1)
                except Exception:
                    bp_state = 1
                if bp_obj and callable(create_object_fn):
                    try:
                        bp_sel = str(bp_obj)
                        if callable(model_selection_fn):
                            bp_sel = str(model_selection_fn(bp_obj) or bp_obj)
                        create_object_fn(pose_frame_obj, bp_sel, int(bp_state), 1)
                        if callable(apply_ball_and_stick_style_fn):
                            apply_ball_and_stick_style_fn(pose_frame_obj, sphere_scale=0.22, stick_radius=0.16)
                        if callable(apply_element_colors_fn):
                            apply_element_colors_fn(pose_frame_obj)
                        if callable(group_fn):
                            group_fn("Redocking.Trajectory", pose_frame_obj)
                    except Exception:
                        pass

        known_after = set(known_objects or set())
        if callable(get_known_objects_fn):
            try:
                known_after = set(get_known_objects_fn() or [])
            except Exception:
                known_after = set(known_objects or set())
        build_actions = (
            build_focus_actions_fn
            if callable(build_focus_actions_fn)
            else build_redocking_traj_focus_visibility_actions
        )
        actions = list(
            build_actions(
                traj_object=str(traj_obj),
                frame_object=str(frame_obj),
                pose_frame_object=str(pose_frame_obj),
                combo_object=str(combo_obj),
                visible=bool(payload.get("visible", True)),
                known_objects=set(known_after or set()),
                prefer_traj_object=bool(payload.get("prefer_traj_object", False)),
            )
            or []
        )
        if not actions:
            actions = list(payload.get("visibility_actions", []) or [])
        if callable(apply_object_actions_fn):
            apply_object_actions_fn(
                list(actions or []),
                known_objects=set(known_after or set()),
            )
        return {
            "ok": True,
            "frame_object": str(frame_obj),
            "state": int(state),
            "payload": dict(payload or {}),
            "actions": list(actions or []),
            "error": "",
        }
    except Exception as e:
        return {
            "ok": False,
            "frame_object": "",
            "state": 0,
            "payload": dict(payload or {}),
            "actions": [],
            "error": str(e),
        }


def prepare_redocking_selected_pose_context(
    rows,
    *,
    row_index=-1,
    known_objects=None,
    preview_object="redock_selected_pose_preview",
    require_pose_in_scene=True,
):
    rows_local = list(rows or [])
    try:
        idx = int(row_index)
    except Exception:
        idx = -1
    known = set(known_objects or [])
    preview_obj = str(preview_object or "").strip() or "redock_selected_pose_preview"

    payload = {
        "ok": False,
        "reason": "",
        "warning": "",
        "row_index": int(idx),
        "record": {},
        "pose_object": "",
        "pose_state": 1,
        "receptor_object": "",
        "system_label": "",
        "member_tag": "",
        "traj_state": 0,
        "preview_object": str(preview_obj),
    }

    if not rows_local:
        payload["reason"] = "no_results"
        return payload
    if idx < 0 or idx >= len(rows_local):
        payload["reason"] = "row_out_of_range"
        payload["warning"] = "Select a row in the redocking table first."
        return payload

    rec = rows_local[int(idx)] if isinstance(rows_local[int(idx)], dict) else {}
    pose_obj = str(rec.get("pose_object", "") or "").strip()
    try:
        pose_state = int(rec.get("state", 1) or 1)
    except Exception:
        pose_state = 1
    pose_state = max(1, int(pose_state))
    receptor_obj = str(rec.get("receptor", "") or "").strip()
    system_label = str(rec.get("system", "") or "").strip()
    member_tag = str(rec.get("system_member", "") or "").strip()
    try:
        traj_state = int(rec.get("traj_state", 0) or 0)
    except Exception:
        traj_state = 0
    traj_state = max(0, int(traj_state))

    payload["record"] = dict(rec)
    payload["pose_object"] = str(pose_obj)
    payload["pose_state"] = int(pose_state)
    payload["receptor_object"] = str(receptor_obj)
    payload["system_label"] = str(system_label)
    payload["member_tag"] = str(member_tag)
    payload["traj_state"] = int(traj_state)

    if bool(require_pose_in_scene) and ((not pose_obj) or (pose_obj not in known)):
        payload["reason"] = "pose_missing"
        payload["warning"] = "Pose object is not available in the scene."
        return payload

    payload["ok"] = True
    return payload


def build_redocking_selected_pose_view_plan(
    *,
    preview_object="redock_selected_pose_preview",
    traj_frame_object="",
    receptor_object="",
    known_objects=None,
):
    known = set(known_objects or [])
    preview_obj = str(preview_object or "").strip() or "redock_selected_pose_preview"
    traj_frame_obj = str(traj_frame_object or "").strip()
    receptor_obj = str(receptor_object or "").strip()

    receptor_candidate = str(traj_frame_obj or receptor_obj)
    receptor_view_obj = str(receptor_candidate)
    if receptor_view_obj and receptor_view_obj not in known:
        receptor_view_obj = ""

    focus_parts = [f"({preview_obj})"]
    if traj_frame_obj:
        focus_parts.append(f"({traj_frame_obj}) and polymer")
    focus_sel = " or ".join([str(x) for x in list(focus_parts or []) if str(x or "").strip()])

    return {
        "preview_object": str(preview_obj),
        "traj_frame_object": str(traj_frame_obj),
        "receptor_object": str(receptor_obj),
        "receptor_candidate_object": str(receptor_candidate),
        "receptor_view_object": str(receptor_view_obj),
        "focus_selection": str(focus_sel),
    }


def build_redocking_selected_pose_execution_plan(
    *,
    receptor_view_object="",
    focus_selection="",
    default_action="disable",
    focus_zoom_buffer_a=6.0,
):
    receptor_view_obj = str(receptor_view_object or "").strip()
    focus_sel = str(focus_selection or "").strip()
    dflt = str(default_action or "").strip().lower()
    if dflt not in ("enable", "disable"):
        dflt = "disable"
    try:
        zoom_buf = float(focus_zoom_buffer_a)
    except Exception:
        zoom_buf = 6.0
    if (not isfinite(zoom_buf)) or (zoom_buf <= 0.0):
        zoom_buf = 6.0

    receptor_poly_sel = ""
    scene_actions = []
    if receptor_view_obj:
        receptor_poly_sel = f"({receptor_view_obj}) and polymer"
        scene_actions = [
            {"action": "enable", "object": str(receptor_view_obj)},
            {"action": "show", "representation": "cartoon", "selection": str(receptor_poly_sel)},
            {"action": "color", "color": "white", "selection": str(receptor_poly_sel)},
            {
                "action": "set",
                "setting": "cartoon_transparency",
                "value": 0.0,
                "selection": str(receptor_poly_sel),
            },
        ]

    operations = []
    if scene_actions:
        operations.append(
            {
                "kind": "scene_actions",
                "actions": list(scene_actions),
                "default_action": str(dflt),
            }
        )
    if focus_sel:
        operations.append({"kind": "orient", "selection": str(focus_sel)})
        operations.append(
            {
                "kind": "zoom",
                "selection": str(focus_sel),
                "buffer_a": float(zoom_buf),
            }
        )

    return {
        "ok": True,
        "default_action": str(dflt),
        "focus_zoom_buffer_a": float(zoom_buf),
        "receptor_view_object": str(receptor_view_obj),
        "receptor_polymer_selection": str(receptor_poly_sel),
        "focus_selection": str(focus_sel),
        "scene_actions": list(scene_actions),
        "has_scene_actions": bool(scene_actions),
        "should_orient": bool(focus_sel),
        "should_zoom": bool(focus_sel),
        "operations": list(operations),
    }


def run_redocking_selected_pose_runtime(
    *,
    rows,
    row_index=-1,
    known_objects=None,
    preview_object="redock_selected_pose_preview",
    default_action="disable",
    focus_zoom_buffer_a=6.0,
    prepare_selected_pose_context_fn=None,
    hide_all_pose_objects_fn=None,
    hide_all_runtime_receptors_fn=None,
    delete_object_fn=None,
    create_object_fn=None,
    model_selection_fn=None,
    apply_ball_and_stick_style_fn=None,
    apply_element_colors_fn=None,
    group_fn=None,
    set_traj_frame_fn=None,
    get_known_objects_fn=None,
    build_selected_pose_view_plan_fn=None,
    build_selected_pose_execution_plan_fn=None,
    apply_scene_actions_fn=None,
    render_reference_context_fn=None,
    orient_fn=None,
    zoom_fn=None,
):
    known_before = set(known_objects or set())
    preview_obj = str(preview_object or "").strip() or "redock_selected_pose_preview"
    payload = {
        "ok": False,
        "reason": "",
        "warning": "",
        "error": "",
        "context": {},
        "view_plan": {},
        "execution_plan": {},
        "preview_object": str(preview_obj),
        "traj_frame_object": "",
        "traj_state": 0,
        "known_objects_before": set(known_before or set()),
        "known_objects_after": set(known_before or set()),
    }

    prepare_fn = (
        prepare_selected_pose_context_fn
        if callable(prepare_selected_pose_context_fn)
        else prepare_redocking_selected_pose_context
    )
    try:
        idx = int(row_index)
    except Exception:
        idx = -1

    ctx = dict(
        prepare_fn(
            rows=list(rows or []),
            row_index=int(idx),
            known_objects=set(known_before or set()),
            preview_object=str(preview_obj),
        )
        or {}
    )
    payload["context"] = dict(ctx or {})
    payload["preview_object"] = str(ctx.get("preview_object", preview_obj) or preview_obj)

    if not bool(ctx.get("ok", False)):
        payload["reason"] = str(ctx.get("reason", "") or "")
        payload["warning"] = str(ctx.get("warning", "") or "")
        return payload

    pose_obj = str(ctx.get("pose_object", "") or "").strip()
    if not pose_obj:
        payload["reason"] = "pose_missing"
        payload["warning"] = "Pose object is not available in the scene."
        return payload

    try:
        if callable(hide_all_pose_objects_fn):
            hide_all_pose_objects_fn()
        if callable(hide_all_runtime_receptors_fn):
            hide_all_runtime_receptors_fn()

        preview_obj = str(payload.get("preview_object", preview_obj) or preview_obj)
        if preview_obj and preview_obj in known_before and callable(delete_object_fn):
            delete_object_fn(preview_obj)

        try:
            pose_state = int(ctx.get("pose_state", 1) or 1)
        except Exception:
            pose_state = 1
        pose_state = max(1, int(pose_state))

        if callable(create_object_fn):
            pose_sel = str(pose_obj)
            if callable(model_selection_fn):
                pose_sel = str(model_selection_fn(pose_obj) or pose_obj)
            create_object_fn(preview_obj, pose_sel, int(pose_state), 1)
        if callable(apply_ball_and_stick_style_fn):
            apply_ball_and_stick_style_fn(preview_obj, sphere_scale=0.22, stick_radius=0.16)
        if callable(apply_element_colors_fn):
            apply_element_colors_fn(preview_obj)
        if callable(group_fn):
            group_fn("Redocking.Results", preview_obj)

        system_label = str(ctx.get("system_label", "") or "")
        member_tag = str(ctx.get("member_tag", "") or "")
        try:
            traj_state = int(ctx.get("traj_state", 0) or 0)
        except Exception:
            traj_state = 0
        traj_state = max(0, int(traj_state))

        frame_out = ("", 0)
        if callable(set_traj_frame_fn):
            try:
                frame_out = set_traj_frame_fn(system_label, member_tag, explicit_state=int(traj_state))
            except TypeError:
                frame_out = set_traj_frame_fn(system_label, member_tag, int(traj_state))

        traj_frame_obj = ""
        traj_state_applied = 0
        if isinstance(frame_out, dict):
            traj_frame_obj = str(frame_out.get("frame_object", "") or "").strip()
            try:
                traj_state_applied = int(frame_out.get("state", 0) or 0)
            except Exception:
                traj_state_applied = 0
        elif isinstance(frame_out, (list, tuple)):
            if len(frame_out) >= 1:
                traj_frame_obj = str(frame_out[0] or "").strip()
            if len(frame_out) >= 2:
                try:
                    traj_state_applied = int(frame_out[1] or 0)
                except Exception:
                    traj_state_applied = 0
        elif isinstance(frame_out, str):
            traj_frame_obj = str(frame_out).strip()

        payload["traj_frame_object"] = str(traj_frame_obj)
        payload["traj_state"] = int(max(0, int(traj_state_applied)))

        known_after = set(known_before or set())
        if callable(get_known_objects_fn):
            try:
                known_after = set(get_known_objects_fn() or [])
            except Exception:
                known_after = set(known_before or set())
        payload["known_objects_after"] = set(known_after or set())

        view_plan_fn = (
            build_selected_pose_view_plan_fn
            if callable(build_selected_pose_view_plan_fn)
            else build_redocking_selected_pose_view_plan
        )
        view_plan = dict(
            view_plan_fn(
                preview_object=str(preview_obj),
                traj_frame_object=str(traj_frame_obj),
                receptor_object=str(ctx.get("receptor_object", "") or ""),
                known_objects=set(known_after or set()),
            )
            or {}
        )
        payload["view_plan"] = dict(view_plan or {})

        exec_plan_fn = (
            build_selected_pose_execution_plan_fn
            if callable(build_selected_pose_execution_plan_fn)
            else build_redocking_selected_pose_execution_plan
        )
        execution_plan = dict(
            exec_plan_fn(
                receptor_view_object=str(view_plan.get("receptor_view_object", "") or ""),
                focus_selection=str(view_plan.get("focus_selection", "") or ""),
                default_action=str(default_action or "disable"),
                focus_zoom_buffer_a=float(focus_zoom_buffer_a),
            )
            or {}
        )
        payload["execution_plan"] = dict(execution_plan or {})

        operations = list(execution_plan.get("operations", []) or [])
        for op in operations:
            if str((op or {}).get("kind", "") or "").strip().lower() != "scene_actions":
                continue
            if callable(apply_scene_actions_fn):
                apply_scene_actions_fn(
                    actions=list((op or {}).get("actions", []) or []),
                    known_objects=set(known_after or set()),
                    default_action=str((op or {}).get("default_action", "disable") or "disable"),
                )

        if callable(render_reference_context_fn):
            render_reference_context_fn()

        for op in operations:
            kind = str((op or {}).get("kind", "") or "").strip().lower()
            sel = str((op or {}).get("selection", "") or "").strip()
            if (not sel) or (kind not in ("orient", "zoom")):
                continue
            if kind == "orient":
                if callable(orient_fn):
                    orient_fn(sel)
                continue
            try:
                zoom_buf = float((op or {}).get("buffer_a", 6.0) or 6.0)
            except Exception:
                zoom_buf = 6.0
            if callable(zoom_fn):
                zoom_fn(sel, float(zoom_buf))

        payload["ok"] = True
        return payload
    except Exception as e:
        payload["reason"] = "runtime_error"
        payload["error"] = str(e)
        payload["warning"] = f"Failed to load selected pose: {e}"
        return payload


def resolve_redocking_selected_pose_warning_message(payload, *, silent=False):
    data = dict(payload or {}) if isinstance(payload, dict) else {}
    reason = str(data.get("reason", "") or "")
    warning = str(data.get("warning", "") or "")
    show_warning = (not bool(silent)) and bool(warning) and (reason != "no_results")
    return {
        "show_warning": bool(show_warning),
        "warning": (str(warning) if bool(show_warning) else ""),
        "reason": str(reason),
    }


def resolve_redocking_reference_grid_warning_message(
    *,
    ref_object="",
    ref_fields=None,
    center=None,
    missing_setup_warning="Select reference object and crystal ligand (chain/resi/resn).",
    selection_missing_warning="Reference ligand selection was not found in reference object.",
):
    ref_obj = str(ref_object or "").strip()
    fields = tuple(ref_fields or ()) if isinstance(ref_fields, (list, tuple)) else tuple()
    if (not ref_obj) or (not fields):
        return {
            "show_warning": True,
            "warning": str(missing_setup_warning),
            "reason": "missing_reference_or_fields",
        }
    if center is None:
        return {
            "show_warning": True,
            "warning": str(selection_missing_warning),
            "reason": "selection_not_found",
        }
    return {
        "show_warning": False,
        "warning": "",
        "reason": "",
    }


REDOCK_BOX_MIN_MARGIN_A = 1.0


def build_redocking_reference_grid_size(
    span,
    *,
    padding=3.0,
    min_edge=10.0,
    max_edge=24.0,
    size_delta=None,
    final_min_edge=2.0,
    final_max_edge=120.0,
):
    payload = {
        "ok": False,
        "reason": "",
        "base_size": [],
        "size": [],
        "size_delta": [],
    }
    try:
        span_vals = [float(span[0]), float(span[1]), float(span[2])]
    except Exception:
        payload["reason"] = "invalid_span"
        return payload

    delta_vals = [0.0, 0.0, 0.0]
    if isinstance(size_delta, (list, tuple)) and len(size_delta) >= 3:
        for idx in range(3):
            try:
                delta_vals[idx] = float(size_delta[idx])
            except Exception:
                delta_vals[idx] = 0.0

    pad = _safe_float(padding, default=0.0)
    base_lo = _safe_float(min_edge, default=10.0)
    base_hi = _safe_float(max_edge, default=24.0)
    final_lo = _safe_float(final_min_edge, default=2.0)
    final_hi = _safe_float(final_max_edge, default=120.0)
    if not isfinite(pad):
        pad = 0.0
    if not isfinite(base_lo):
        base_lo = 10.0
    if not isfinite(base_hi):
        base_hi = max(base_lo, 24.0)
    if base_hi < base_lo:
        base_hi = float(base_lo)
    if not isfinite(final_lo):
        final_lo = 2.0
    if not isfinite(final_hi):
        final_hi = max(final_lo, 120.0)
    if final_hi < final_lo:
        final_hi = float(final_lo)

    def _clip(value, lo, hi):
        vv = float(value)
        if vv < float(lo):
            return float(lo)
        if vv > float(hi):
            return float(hi)
        return float(vv)

    base_size = []
    out_size = []
    for idx in range(3):
        span_i = max(0.0, float(span_vals[idx]))
        # padding on each side of the ligand (the box is centred on the ligand's bounding box);
        # no edge limit may leave less than 1 A around the crystal ligand
        cover = span_i + 2.0 * REDOCK_BOX_MIN_MARGIN_A
        raw_edge = span_i + 2.0 * float(pad)
        base_edge = _clip(raw_edge, base_lo, max(base_hi, cover))
        final_edge = _clip(float(base_edge) + float(delta_vals[idx]), max(final_lo, cover), max(final_hi, cover))
        base_size.append(float(base_edge))
        out_size.append(float(final_edge))

    payload["ok"] = True
    payload["base_size"] = list(base_size)
    payload["size"] = list(out_size)
    payload["size_delta"] = list(delta_vals)
    return payload


def adjust_redocking_grid_size(
    size,
    *,
    axis="",
    delta=0.0,
    min_edge=2.0,
    max_edge=120.0,
):
    payload = {
        "ok": False,
        "reason": "",
        "axis": str(axis or "").strip().lower(),
        "delta": 0.0,
        "size": [],
    }
    try:
        cur = [float(size[0]), float(size[1]), float(size[2])]
    except Exception:
        payload["reason"] = "invalid_size"
        return payload
    axis_key = str(axis or "").strip().lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = axis_map.get(axis_key, None)
    if idx is None:
        payload["reason"] = "invalid_axis"
        return payload
    step = _safe_float(delta, default=0.0)
    if not isfinite(step):
        step = 0.0
    lo = _safe_float(min_edge, default=2.0)
    hi = _safe_float(max_edge, default=120.0)
    if not isfinite(lo):
        lo = 2.0
    if not isfinite(hi):
        hi = max(lo, 120.0)
    if hi < lo:
        hi = float(lo)
    cur[idx] = max(float(lo), min(float(hi), float(cur[idx]) + float(step)))
    payload["ok"] = True
    payload["delta"] = float(step)
    payload["size"] = list(cur)
    return payload


def build_redocking_reference_grid_info_message(center, size, nref):
    try:
        cx = float(center[0])
        cy = float(center[1])
        cz = float(center[2])
        sx = float(size[0])
        sy = float(size[1])
        sz = float(size[2])
        nref_i = int(nref or 0)
    except Exception:
        return {
            "ok": False,
            "message": "",
        }
    return {
        "ok": True,
        "message": (
            f"Grid centered on reference ligand ({nref_i} atoms).\n"
            f"Center=({cx:.3f}, {cy:.3f}, {cz:.3f})\n"
            f"Size=({sx:.2f}, {sy:.2f}, {sz:.2f}) A"
        ),
    }


def build_redocking_reference_context_plan(
    *,
    ref_object="",
    ref_fields=None,
    known_objects=None,
    reference_selection_fn=None,
):
    ref_obj = str(ref_object or "").strip()
    known = set(known_objects or [])
    payload = {
        "ok": False,
        "reason": "",
        "ref_object": str(ref_obj),
        "polymer_selection": "",
        "highlight_selection": "",
        "actions": [],
    }
    if not ref_obj:
        payload["reason"] = "missing_reference"
        return payload
    if ref_obj not in known:
        payload["reason"] = "reference_not_loaded"
        return payload

    poly_sel = f"({ref_obj}) and polymer"
    actions = [
        {"action": "enable", "object": str(ref_obj)},
        {"action": "show", "representation": "cartoon", "selection": str(poly_sel)},
        {"action": "color", "color": "gray70", "selection": str(poly_sel)},
        {"action": "set", "setting": "cartoon_transparency", "value": 0.55, "selection": str(poly_sel)},
    ]

    highlight_sel = ""
    fields = tuple(ref_fields or ()) if isinstance(ref_fields, (list, tuple)) else tuple()
    if len(fields) >= 3 and callable(reference_selection_fn):
        try:
            highlight_sel = str(reference_selection_fn(ref_obj, tuple(fields[:3])) or "").strip()
        except Exception:
            highlight_sel = ""
    if highlight_sel:
        actions.append({"action": "show", "representation": "sticks", "selection": str(highlight_sel)})
        actions.append({"action": "color", "color": "yellow", "selection": str(highlight_sel)})

    payload["ok"] = True
    payload["polymer_selection"] = str(poly_sel)
    payload["highlight_selection"] = str(highlight_sel)
    payload["actions"] = list(actions)
    return payload


def run_redocking_reference_context_runtime(
    *,
    ref_object="",
    ref_fields=None,
    known_objects=None,
    get_known_objects_fn=None,
    reference_selection_fn=None,
    build_reference_context_plan_fn=None,
    apply_scene_actions_fn=None,
    default_action="disable",
):
    known = set(known_objects or set())
    if callable(get_known_objects_fn):
        try:
            known = set(get_known_objects_fn() or [])
        except Exception:
            known = set(known_objects or set())

    build_plan = (
        build_reference_context_plan_fn
        if callable(build_reference_context_plan_fn)
        else build_redocking_reference_context_plan
    )
    try:
        plan = dict(
            build_plan(
                ref_object=str(ref_object or ""),
                ref_fields=ref_fields,
                known_objects=set(known or set()),
                reference_selection_fn=reference_selection_fn,
            )
            or {}
        )
    except Exception as e:
        return {
            "ok": False,
            "reason": "plan_error",
            "error": str(e),
            "known_objects": set(known or set()),
            "plan": {},
            "actions": [],
            "actions_applied": False,
        }

    if not bool(plan.get("ok", False)):
        return {
            "ok": False,
            "reason": str(plan.get("reason", "") or ""),
            "error": "",
            "known_objects": set(known or set()),
            "plan": dict(plan or {}),
            "actions": list(plan.get("actions", []) or []),
            "actions_applied": False,
        }

    actions = list(plan.get("actions", []) or [])
    actions_applied = False
    if callable(apply_scene_actions_fn):
        try:
            apply_scene_actions_fn(
                actions=list(actions or []),
                known_objects=set(known or set()),
                default_action=str(default_action or "disable"),
            )
            actions_applied = True
        except Exception as e:
            return {
                "ok": False,
                "reason": "apply_error",
                "error": str(e),
                "known_objects": set(known or set()),
                "plan": dict(plan or {}),
                "actions": list(actions or []),
                "actions_applied": False,
            }

    return {
        "ok": True,
        "reason": "",
        "error": "",
        "known_objects": set(known or set()),
        "plan": dict(plan or {}),
        "actions": list(actions or []),
        "actions_applied": bool(actions_applied),
    }


def prepare_redocking_selected_interaction_context(
    selected_context,
    *,
    traj_frame_object="",
    known_objects=None,
    preview_object="redock_selected_pose_preview",
    complex_object="redock_selected_complex",
    require_preview_in_scene=True,
    require_receptor_in_scene=False,
):
    known = set(known_objects or [])
    preview_obj = str(preview_object or "").strip() or "redock_selected_pose_preview"
    complex_obj = str(complex_object or "").strip() or "redock_selected_complex"

    payload = {
        "ok": False,
        "reason": "",
        "warning": "Could not build selected pose preview for interactions.",
        "preview_object": str(preview_obj),
        "complex_object": str(complex_obj),
        "receptor_object": "",
        "receptor_context_object": "",
        "complex_selection": "",
        "system_label": "",
        "member_tag": "",
        "traj_state": 0,
    }

    ctx = dict(selected_context or {}) if isinstance(selected_context, dict) else {}
    receptor_obj = str(ctx.get("receptor_object", "") or "").strip()
    system_label = str(ctx.get("system_label", "") or "").strip()
    member_tag = str(ctx.get("member_tag", "") or "").strip()
    try:
        traj_state = int(ctx.get("traj_state", 0) or 0)
    except Exception:
        traj_state = 0
    traj_state = max(0, int(traj_state))
    traj_frame_obj = str(traj_frame_object or "").strip()
    receptor_ctx_obj = str(traj_frame_obj or receptor_obj)

    payload["receptor_object"] = str(receptor_obj)
    payload["receptor_context_object"] = str(receptor_ctx_obj)
    payload["system_label"] = str(system_label)
    payload["member_tag"] = str(member_tag)
    payload["traj_state"] = int(traj_state)

    if bool(require_preview_in_scene) and (preview_obj not in known):
        payload["reason"] = "preview_missing"
        return payload
    if not receptor_ctx_obj:
        payload["reason"] = "missing_receptor_context"
        return payload
    if bool(require_receptor_in_scene) and (receptor_ctx_obj not in known):
        payload["reason"] = "receptor_missing"
        return payload

    payload["complex_selection"] = f"(({receptor_ctx_obj}) and polymer) or ({preview_obj})"
    payload["ok"] = True
    return payload


def prepare_redocking_selected_interaction_post_payload(
    *,
    complex_object="redock_selected_complex",
    ligand_info=None,
):
    complex_obj = str(complex_object or "").strip()
    payload = {
        "ok": bool(complex_obj),
        "error": ("" if complex_obj else "complex_object is empty"),
        "loaded_object": str(complex_obj),
        "traj_topology_object": str(complex_obj),
        "set_ligand_info": False,
        "ligand_info": None,
        "ligand_fields": {
            "chain": "",
            "resi": "",
            "resn": "",
        },
    }
    if not complex_obj:
        return payload

    chain = ""
    resi = ""
    resn = ""

    if isinstance(ligand_info, dict):
        chain = str(ligand_info.get("chain", "") or "")
        resi = str(ligand_info.get("resi", "") or "")
        resn = str(ligand_info.get("resn", "") or "")
    elif isinstance(ligand_info, (list, tuple)) and len(ligand_info) >= 3:
        chain = str(ligand_info[0] or "")
        resi = str(ligand_info[1] or "")
        resn = str(ligand_info[2] or "")

    if chain or resi or resn:
        payload["set_ligand_info"] = True
        payload["ligand_info"] = (str(chain), str(resi), str(resn))
        payload["ligand_fields"] = {
            "chain": str(chain),
            "resi": str(resi),
            "resn": str(resn),
        }
    return payload


def build_redocking_selected_interaction_post_apply_plan(
    *,
    post_payload,
    default_complex_object="redock_selected_complex",
):
    post = dict(post_payload or {}) if isinstance(post_payload, dict) else {}
    default_obj = str(default_complex_object or "").strip() or "redock_selected_complex"
    loaded_obj = str(post.get("loaded_object", default_obj) or default_obj)
    topo_obj = str(post.get("traj_topology_object", loaded_obj) or loaded_obj)

    set_ligand_info = bool(post.get("set_ligand_info", False))
    ligand_info = tuple(post.get("ligand_info", ()) or ())
    lig_fields = dict(post.get("ligand_fields", {}) or {})
    chain = str(lig_fields.get("chain", "") or "")
    resi = str(lig_fields.get("resi", "") or "")
    resn = str(lig_fields.get("resn", "") or "")

    operations = [
        {"kind": "set_attr", "name": "loaded_object", "value": str(loaded_obj)},
        {"kind": "set_attr", "name": "_traj_topology_object", "value": str(topo_obj)},
    ]
    if set_ligand_info:
        operations.append(
            {
                "kind": "set_attr",
                "name": "ligand_info",
                "value": tuple(ligand_info),
            }
        )
        operations.extend(
            [
                {"kind": "set_text", "widget": "ligand_chain", "value": str(chain)},
                {"kind": "set_text", "widget": "ligand_resi", "value": str(resi)},
                {"kind": "set_text", "widget": "ligand_resn", "value": str(resn)},
            ]
        )

    return {
        "ok": True,
        "loaded_object": str(loaded_obj),
        "traj_topology_object": str(topo_obj),
        "set_ligand_info": bool(set_ligand_info),
        "ligand_info": tuple(ligand_info),
        "ligand_fields": {
            "chain": str(chain),
            "resi": str(resi),
            "resn": str(resn),
        },
        "operations": list(operations),
    }


def run_redocking_selected_interaction_runtime(
    *,
    rows,
    row_index=-1,
    known_objects=None,
    selected_context_known_objects=None,
    preview_object="redock_selected_pose_preview",
    complex_object="redock_selected_complex",
    prepare_selected_pose_context_fn=None,
    load_selected_pose_fn=None,
    set_traj_frame_fn=None,
    get_known_objects_fn=None,
    prepare_selected_interaction_context_fn=None,
    delete_object_fn=None,
    create_object_fn=None,
    auto_detect_ligand_fn=None,
    prepare_selected_interaction_post_payload_fn=None,
    apply_post_payload_fn=None,
    populate_limit_combo_fn=None,
    calc_interactions_fn=None,
):
    payload = {
        "ok": False,
        "reason": "",
        "warning": "",
        "error": "",
        "selected_context": {},
        "interaction_context": {},
        "post_payload": {},
        "complex_object": str(complex_object or "redock_selected_complex"),
        "complex_selection": "",
        "traj_frame_object": "",
        "traj_state": 0,
        "known_objects": set(known_objects or set()),
    }

    prepare_selected_fn = (
        prepare_selected_pose_context_fn
        if callable(prepare_selected_pose_context_fn)
        else prepare_redocking_selected_pose_context
    )
    try:
        idx = int(row_index)
    except Exception:
        idx = -1

    selected_ctx = dict(
        prepare_selected_fn(
            rows=list(rows or []),
            row_index=int(idx),
            known_objects=set(selected_context_known_objects or set()),
            require_pose_in_scene=False,
            preview_object=str(preview_object or "redock_selected_pose_preview"),
        )
        or {}
    )
    payload["selected_context"] = dict(selected_ctx or {})
    if not bool(selected_ctx.get("ok", False)):
        payload["reason"] = str(selected_ctx.get("reason", "") or "")
        payload["warning"] = "Select a redocking row first."
        return payload

    if callable(load_selected_pose_fn):
        try:
            load_selected_pose_fn(silent=True)
        except TypeError:
            try:
                load_selected_pose_fn()
            except Exception:
                pass
        except Exception:
            pass

    system_label = str(selected_ctx.get("system_label", "") or "")
    member_tag = str(selected_ctx.get("member_tag", "") or "")
    try:
        traj_state = int(selected_ctx.get("traj_state", 0) or 0)
    except Exception:
        traj_state = 0
    traj_state = max(0, int(traj_state))

    frame_out = ("", 0)
    if callable(set_traj_frame_fn):
        try:
            frame_out = set_traj_frame_fn(system_label, member_tag, explicit_state=int(traj_state))
        except TypeError:
            frame_out = set_traj_frame_fn(system_label, member_tag, int(traj_state))
        except Exception:
            frame_out = ("", 0)

    traj_frame_obj = ""
    traj_state_applied = 0
    if isinstance(frame_out, dict):
        traj_frame_obj = str(frame_out.get("frame_object", "") or "").strip()
        try:
            traj_state_applied = int(frame_out.get("state", 0) or 0)
        except Exception:
            traj_state_applied = 0
    elif isinstance(frame_out, (list, tuple)):
        if len(frame_out) >= 1:
            traj_frame_obj = str(frame_out[0] or "").strip()
        if len(frame_out) >= 2:
            try:
                traj_state_applied = int(frame_out[1] or 0)
            except Exception:
                traj_state_applied = 0
    elif isinstance(frame_out, str):
        traj_frame_obj = str(frame_out).strip()

    payload["traj_frame_object"] = str(traj_frame_obj)
    payload["traj_state"] = int(max(0, int(traj_state_applied)))

    known = set(known_objects or set())
    if callable(get_known_objects_fn):
        try:
            known = set(get_known_objects_fn() or [])
        except Exception:
            known = set(known_objects or set())
    payload["known_objects"] = set(known or set())

    prepare_interaction_fn = (
        prepare_selected_interaction_context_fn
        if callable(prepare_selected_interaction_context_fn)
        else prepare_redocking_selected_interaction_context
    )
    interaction_ctx = dict(
        prepare_interaction_fn(
            dict(selected_ctx or {}),
            traj_frame_object=str(traj_frame_obj or ""),
            known_objects=set(known or set()),
            preview_object=str(preview_object or "redock_selected_pose_preview"),
            complex_object=str(complex_object or "redock_selected_complex"),
            require_preview_in_scene=True,
            require_receptor_in_scene=False,
        )
        or {}
    )
    payload["interaction_context"] = dict(interaction_ctx or {})
    if not bool(interaction_ctx.get("ok", False)):
        payload["reason"] = str(interaction_ctx.get("reason", "") or "interaction_context_invalid")
        payload["warning"] = str(
            interaction_ctx.get(
                "warning",
                "Could not build selected pose preview for interactions.",
            )
            or "Could not build selected pose preview for interactions."
        )
        return payload

    complex_obj = str(
        interaction_ctx.get("complex_object", complex_object or "redock_selected_complex")
        or (complex_object or "redock_selected_complex")
    ).strip()
    complex_sel = str(interaction_ctx.get("complex_selection", "") or "").strip()
    payload["complex_object"] = str(complex_obj)
    payload["complex_selection"] = str(complex_sel)
    if not complex_sel:
        payload["reason"] = "missing_complex_selection"
        payload["warning"] = "Could not build selected pose preview for interactions."
        return payload

    post_builder = (
        prepare_selected_interaction_post_payload_fn
        if callable(prepare_selected_interaction_post_payload_fn)
        else prepare_redocking_selected_interaction_post_payload
    )
    try:
        known_now = set(known or set())
        if callable(get_known_objects_fn):
            try:
                known_now = set(get_known_objects_fn() or [])
            except Exception:
                known_now = set(known or set())
        if complex_obj and complex_obj in known_now and callable(delete_object_fn):
            delete_object_fn(complex_obj)
        if callable(create_object_fn):
            create_object_fn(complex_obj, complex_sel, 1, 1)

        pre_post_payload = dict(
            post_builder(
                complex_object=str(complex_obj),
                ligand_info=None,
            )
            or {}
        )
        if callable(apply_post_payload_fn):
            apply_post_payload_fn(dict(pre_post_payload or {}))

        ligand_info = None
        if callable(auto_detect_ligand_fn):
            try:
                ligand_info = auto_detect_ligand_fn()
            except TypeError:
                try:
                    ligand_info = auto_detect_ligand_fn(str(complex_obj))
                except Exception:
                    ligand_info = None
            except Exception:
                ligand_info = None
        post_payload = dict(
            post_builder(
                complex_object=str(complex_obj),
                ligand_info=ligand_info,
            )
            or {}
        )
        payload["post_payload"] = dict(post_payload or {})

        if callable(apply_post_payload_fn):
            apply_post_payload_fn(dict(post_payload or {}))
        if callable(populate_limit_combo_fn):
            try:
                populate_limit_combo_fn()
            except Exception:
                pass
        if callable(calc_interactions_fn):
            calc_interactions_fn()
        payload["ok"] = True
        return payload
    except Exception as e:
        payload["reason"] = "runtime_error"
        payload["error"] = str(e)
        payload["warning"] = f"Failed to compute interactions for selected redock pose: {e}"
        return payload


def resolve_redocking_selected_interaction_warning_message(
    payload,
    *,
    default_warning="Could not build selected pose preview for interactions.",
):
    data = dict(payload or {}) if isinstance(payload, dict) else {}
    warning = str(data.get("warning", default_warning) or default_warning)
    return {
        "show_warning": bool(warning),
        "warning": str(warning),
        "reason": str(data.get("reason", "") or ""),
    }


def build_redocking_traj_visibility_actions(
    *,
    views,
    known_objects=None,
    visible_for_label_fn=None,
    show_wt=True,
    show_mut=False,
    default_visible=True,
):
    known = set(known_objects or [])

    if callable(visible_for_label_fn):
        vis_fn = visible_for_label_fn
    else:
        vis_fn = lambda lbl: redocking_traj_visible_for_label(
            lbl,
            show_wt=bool(show_wt),
            show_mut=bool(show_mut),
            default_visible=bool(default_visible),
        )

    actions = []
    for label, info in dict(views or {}).items():
        if not isinstance(info, dict):
            continue
        frame_obj = str(info.get("frame_object", "") or "").strip()
        pose_frame_obj = str(info.get("pose_frame_object", "") or "").strip()
        combo_obj = str(info.get("combo_object", "") or "").strip()
        traj_obj = str(info.get("traj_object", "") or "").strip()
        try:
            n_states = int(info.get("n_states", 0) or 0)
        except Exception:
            n_states = 0
        prefer_traj_object = bool((not combo_obj) and traj_obj and (traj_obj in known) and (int(n_states) > 1))
        vis = bool(vis_fn(label))

        if combo_obj and combo_obj in known:
            if traj_obj and traj_obj in known:
                actions.append({"action": "disable", "object": str(traj_obj), "label": str(label)})
            actions.append(
                {
                    "action": ("enable" if vis else "disable"),
                    "object": str(combo_obj),
                    "label": str(label),
                }
            )
            for obj in (frame_obj, pose_frame_obj):
                if obj and obj in known:
                    actions.append({"action": "disable", "object": str(obj), "label": str(label)})
            continue

        if prefer_traj_object:
            actions.append(
                {
                    "action": ("enable" if vis else "disable"),
                    "object": str(traj_obj),
                    "label": str(label),
                }
            )
            if frame_obj and frame_obj in known:
                actions.append({"action": "disable", "object": str(frame_obj), "label": str(label)})
            if pose_frame_obj and pose_frame_obj in known:
                actions.append(
                    {
                        "action": ("enable" if vis else "disable"),
                        "object": str(pose_frame_obj),
                        "label": str(label),
                    }
                )
            continue

        if traj_obj and traj_obj in known:
            actions.append({"action": "disable", "object": str(traj_obj), "label": str(label)})

        for obj in (frame_obj, pose_frame_obj):
            if not obj or obj not in known:
                continue
            actions.append(
                {
                    "action": ("enable" if vis else "disable"),
                    "object": str(obj),
                    "label": str(label),
                }
            )

    return list(actions)


def run_redocking_traj_visibility_runtime(
    *,
    views,
    known_objects=None,
    visible_for_label_fn=None,
    show_wt=True,
    show_mut=False,
    default_visible=True,
    get_known_objects_fn=None,
    build_traj_visibility_actions_fn=None,
    apply_object_actions_fn=None,
):
    known = set(known_objects or set())
    if callable(get_known_objects_fn):
        try:
            known = set(get_known_objects_fn() or [])
        except Exception:
            known = set(known_objects or set())

    build_actions = (
        build_traj_visibility_actions_fn
        if callable(build_traj_visibility_actions_fn)
        else build_redocking_traj_visibility_actions
    )
    try:
        actions = list(
            build_actions(
                views=dict(views or {}),
                known_objects=set(known or set()),
                visible_for_label_fn=visible_for_label_fn,
                show_wt=bool(show_wt),
                show_mut=bool(show_mut),
                default_visible=bool(default_visible),
            )
            or []
        )
    except Exception as e:
        return {
            "ok": False,
            "known_objects": set(known or set()),
            "actions": [],
            "actions_applied": False,
            "error": str(e),
        }

    actions_applied = False
    if callable(apply_object_actions_fn):
        try:
            apply_object_actions_fn(
                list(actions or []),
                known_objects=set(known or set()),
            )
            actions_applied = True
        except Exception as e:
            return {
                "ok": False,
                "known_objects": set(known or set()),
                "actions": list(actions or []),
                "actions_applied": False,
                "error": str(e),
            }

    return {
        "ok": True,
        "known_objects": set(known or set()),
        "actions": list(actions or []),
        "actions_applied": bool(actions_applied),
        "error": "",
    }


def build_redocking_row_object_actions(
    rows,
    *,
    object_key="",
    action="disable",
    known_objects=None,
):
    key = str(object_key or "").strip()
    if not key:
        return []
    act = str(action or "disable").strip().lower() or "disable"
    known = None if known_objects is None else set(known_objects or [])

    objects = set()
    for row in list(rows or []):
        if not isinstance(row, dict):
            continue
        obj = str((row or {}).get(key, "") or "").strip()
        if not obj:
            continue
        if known is not None and obj not in known:
            continue
        objects.add(str(obj))

    return [
        {
            "action": str(act),
            "object": str(obj),
            "source_key": str(key),
        }
        for obj in sorted(objects)
    ]


def run_redocking_row_object_actions_runtime(
    *,
    rows,
    object_key="",
    action="disable",
    known_objects=None,
    get_known_objects_fn=None,
    build_row_object_actions_fn=None,
    apply_object_actions_fn=None,
):
    known = set(known_objects or set())
    if callable(get_known_objects_fn):
        try:
            known = set(get_known_objects_fn() or [])
        except Exception:
            known = set(known_objects or set())

    build_actions = (
        build_row_object_actions_fn
        if callable(build_row_object_actions_fn)
        else build_redocking_row_object_actions
    )
    try:
        actions = list(
            build_actions(
                rows=list(rows or []),
                object_key=str(object_key or ""),
                action=str(action or "disable"),
                known_objects=set(known or set()),
            )
            or []
        )
    except Exception as e:
        return {
            "ok": False,
            "reason": "build_error",
            "error": str(e),
            "known_objects": set(known or set()),
            "actions": [],
            "actions_applied": False,
        }

    actions_applied = False
    if callable(apply_object_actions_fn):
        try:
            apply_object_actions_fn(
                list(actions or []),
                known_objects=set(known or set()),
            )
            actions_applied = True
        except Exception as e:
            return {
                "ok": False,
                "reason": "apply_error",
                "error": str(e),
                "known_objects": set(known or set()),
                "actions": list(actions or []),
                "actions_applied": False,
            }

    return {
        "ok": True,
        "reason": "",
        "error": "",
        "known_objects": set(known or set()),
        "actions": list(actions or []),
        "actions_applied": bool(actions_applied),
    }


def build_redocking_runtime_input_actions(
    runtime_systems,
    *,
    action="disable",
    known_objects=None,
):
    act = str(action or "disable").strip().lower()
    if act not in ("enable", "disable"):
        act = "disable"
    known = None if known_objects is None else set(known_objects or [])

    actions = []
    for ent in list(runtime_systems or []):
        label = ""
        rec_obj = ""
        member_tag = ""
        if isinstance(ent, dict):
            label = str(ent.get("label", ent.get("system", "")) or "")
            rec_obj = str(
                ent.get(
                    "receptor_object",
                    ent.get("receptor", ent.get("object", ent.get("obj", ""))),
                )
                or ""
            )
            member_tag = str(ent.get("member_tag", ent.get("system_member", "")) or "")
            if not member_tag:
                member_tag = f"{label}_model_001" if str(label).strip() else "model_001"
        else:
            label, rec_obj, _rec_src, member_tag = _normalize_redocking_runtime_system_entry(ent)

        obj = str(rec_obj or "").strip()
        if not obj:
            continue
        if (known is not None) and (obj not in known):
            continue
        actions.append(
            {
                "action": str(act),
                "object": str(obj),
                "label": str(label),
                "member_tag": str(member_tag),
            }
        )
    return list(actions)


def prepare_redocking_runtime_views_runtime(
    *,
    runtime_systems,
    input_systems,
    build_traj_views_fn=None,
    get_known_objects_fn=None,
    build_runtime_input_actions_fn=None,
    apply_scene_actions_fn=None,
    append_debug_log_fn=None,
    input_action="disable",
    default_action="disable",
):
    traj_views_ok = True
    traj_views_error = ""
    if callable(build_traj_views_fn):
        try:
            build_traj_views_fn(runtime_systems)
        except Exception as e:
            traj_views_ok = False
            traj_views_error = str(e)
            if callable(append_debug_log_fn):
                try:
                    append_debug_log_fn(f"Redocking trajectory-view warning: {e}")
                except Exception:
                    pass

    known_objects = set()
    actions = []
    actions_applied = False
    try:
        if callable(get_known_objects_fn):
            known_objects = set(get_known_objects_fn() or [])
        build_actions = (
            build_runtime_input_actions_fn
            if callable(build_runtime_input_actions_fn)
            else build_redocking_runtime_input_actions
        )
        actions = list(
            build_actions(
                runtime_systems=list(input_systems or []),
                action=str(input_action or "disable"),
                known_objects=set(known_objects or set()),
            )
            or []
        )
        if callable(apply_scene_actions_fn):
            apply_scene_actions_fn(
                actions=list(actions or []),
                known_objects=set(known_objects or set()),
                default_action=str(default_action or "disable"),
            )
            actions_applied = True
    except Exception:
        actions = []
        actions_applied = False

    return {
        "ok": True,
        "traj_views_ok": bool(traj_views_ok),
        "traj_views_error": str(traj_views_error or ""),
        "known_objects": set(known_objects or set()),
        "actions": list(actions or []),
        "actions_applied": bool(actions_applied),
    }


def build_redocking_traj_view_plan(
    runtime_systems,
    *,
    labels=None,
    timestamp_tag="",
    safe_name_fn=None,
    max_name_len=220,
):
    labels_use = list(labels or ["WT", "Mutant"])
    grouped = {}
    for ent in list(runtime_systems or []):
        label, rec_obj, _rec_src, member_tag = _normalize_redocking_runtime_system_entry(ent)
        lab = str(label or "").strip()
        rec = str(rec_obj or "").strip()
        mem = str(member_tag or "").strip()
        if not lab:
            continue
        grouped.setdefault(lab, []).append((str(rec), str(mem)))

    ts = str(timestamp_tag or "").strip()
    if not ts:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _safe(v):
        txt = str(v or "")
        if callable(safe_name_fn):
            try:
                txt = str(safe_name_fn(txt) or txt)
            except Exception:
                txt = str(v or "")
        max_len = int(max_name_len or 0)
        if max_len > 0 and len(txt) > max_len:
            txt = txt[:max_len]
        return str(txt)

    views_plan = {}
    for label in labels_use:
        members = list(
            sort_redocking_runtime_systems(
                [
                    (str(label), str(rec_obj), str(rec_obj), str(member_tag))
                    for (rec_obj, member_tag) in list(grouped.get(str(label), []) or [])
                    if str(rec_obj or "").strip()
                ]
            )
            or []
        )
        members = [(str(rec_obj), str(member_tag)) for _lab, rec_obj, _rec_src, member_tag in list(members or [])]
        if not members:
            continue
        lab_txt = str(label).lower()
        traj_obj = _safe(f"redock_{lab_txt}_ensemble_traj_{ts}")
        frame_obj = _safe(f"redock_{lab_txt}_traj_frame")
        pose_frame_obj = _safe(f"redock_{lab_txt}_traj_best_pose")
        views_plan[str(label)] = {
            "members": list(members),
            "traj_object": str(traj_obj),
            "frame_object": str(frame_obj),
            "pose_frame_object": str(pose_frame_obj),
            "combo_object": "",
        }

    return {
        "timestamp_tag": str(ts),
        "views_plan": dict(views_plan or {}),
    }


def load_redocking_traj_members(
    members,
    *,
    count_atoms_fn=None,
    create_state_fn=None,
):
    if not callable(count_atoms_fn):
        return {
            "ok": False,
            "error": "count_atoms_fn is not callable",
            "state_map": {},
            "n_states": 0,
            "loaded_members": [],
            "skipped_members": [],
        }
    if not callable(create_state_fn):
        return {
            "ok": False,
            "error": "create_state_fn is not callable",
            "state_map": {},
            "n_states": 0,
            "loaded_members": [],
            "skipped_members": [],
        }

    state_map = {}
    loaded_states = 0
    loaded_members = []
    skipped_members = []

    for item in list(members or []):
        rec_obj = ""
        member_tag = ""
        if isinstance(item, (list, tuple)):
            if len(item) > 0:
                rec_obj = str(item[0] or "").strip()
            if len(item) > 1:
                member_tag = str(item[1] or "").strip()
        if (not rec_obj) and isinstance(item, dict):
            rec_obj = str(item.get("receptor_obj", "") or "").strip()
            member_tag = str(item.get("member_tag", "") or "").strip()
        if not rec_obj:
            skipped_members.append(
                {"receptor_obj": str(rec_obj), "member_tag": str(member_tag), "reason": "empty_receptor"}
            )
            continue

        atom_count = 0
        try:
            atom_count = int(count_atoms_fn(rec_obj) or 0)
        except Exception:
            atom_count = 0
        if atom_count <= 0:
            skipped_members.append(
                {
                    "receptor_obj": str(rec_obj),
                    "member_tag": str(member_tag),
                    "reason": "no_atoms",
                }
            )
            continue

        next_state = int(loaded_states) + 1
        state_map[str(member_tag)] = int(next_state)
        created_ok = False
        try:
            created_ok = bool(create_state_fn(rec_obj, int(next_state), str(member_tag)))
        except Exception:
            created_ok = False
        if not created_ok:
            state_map.pop(str(member_tag), None)
            skipped_members.append(
                {
                    "receptor_obj": str(rec_obj),
                    "member_tag": str(member_tag),
                    "reason": "create_failed",
                }
            )
            continue

        loaded_states = int(next_state)
        loaded_members.append(
            {
                "receptor_obj": str(rec_obj),
                "member_tag": str(member_tag),
                "state": int(next_state),
            }
        )

    return {
        "ok": True,
        "error": "",
        "state_map": dict(state_map),
        "n_states": int(loaded_states),
        "loaded_members": list(loaded_members),
        "skipped_members": list(skipped_members),
    }


def build_redocking_combo_object_name(
    label,
    *,
    timestamp_tag="",
    safe_name_fn=None,
    max_name_len=220,
):
    lab_txt = str(label or "").strip().lower() or "system"
    ts = str(timestamp_tag or "").strip()
    if not ts:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"redock_{lab_txt}_traj_combo_{ts}"
    if callable(safe_name_fn):
        try:
            name = str(safe_name_fn(name) or name)
        except Exception:
            name = str(name)
    max_len = int(max_name_len or 0)
    if max_len > 0 and len(name) > max_len:
        name = name[:max_len]
    return str(name)


def build_redocking_combo_states(
    n_states,
    *,
    row_by_state=None,
    fallback_row=None,
    create_state_fn=None,
):
    if not callable(create_state_fn):
        return {
            "ok": False,
            "error": "create_state_fn is not callable",
            "n_states": 0,
            "built_any": False,
            "built_states": [],
            "skipped_states": [],
        }

    try:
        n_states_i = int(n_states or 0)
    except Exception:
        n_states_i = 0
    if n_states_i <= 0:
        return {
            "ok": True,
            "error": "",
            "n_states": 0,
            "built_any": False,
            "built_states": [],
            "skipped_states": [],
        }

    row_map = {}
    for k, v in list(dict(row_by_state or {}).items()):
        try:
            st = int(k)
        except Exception:
            st = 0
        if st <= 0:
            continue
        row_map[int(st)] = dict(v) if isinstance(v, dict) else v

    fallback = dict(fallback_row) if isinstance(fallback_row, dict) else fallback_row
    built_states = []
    skipped_states = []

    for st in range(1, int(n_states_i) + 1):
        has_direct = int(st) in row_map
        row = row_map.get(int(st), fallback)
        source = ("state_map" if has_direct else "fallback")
        if row is None:
            skipped_states.append({"state": int(st), "source": str(source), "reason": "no_row"})
            continue
        if not isinstance(row, dict):
            skipped_states.append({"state": int(st), "source": str(source), "reason": "invalid_row"})
            continue

        pose_obj = str(row.get("pose_object", "") or "").strip()
        try:
            pose_state = int(row.get("state", 1) or 1)
        except Exception:
            pose_state = 1
        pose_state = max(1, int(pose_state))

        created_ok = False
        try:
            created_ok = bool(
                create_state_fn(
                    int(st),
                    dict(row),
                    str(pose_obj),
                    int(pose_state),
                    str(source),
                )
            )
        except Exception:
            created_ok = False

        if created_ok:
            built_states.append(
                {
                    "state": int(st),
                    "pose_object": str(pose_obj),
                    "pose_state": int(pose_state),
                    "source": str(source),
                }
            )
        else:
            skipped_states.append(
                {
                    "state": int(st),
                    "pose_object": str(pose_obj),
                    "pose_state": int(pose_state),
                    "source": str(source),
                    "reason": "create_failed",
                }
            )

    return {
        "ok": True,
        "error": "",
        "n_states": int(n_states_i),
        "built_any": bool(built_states),
        "built_states": list(built_states),
        "skipped_states": list(skipped_states),
    }


def collect_redocking_traj_view_objects(views, object_keys=None):
    keys = list(object_keys or ["frame_object", "pose_frame_object", "combo_object", "traj_object"])
    seen = set()
    out = []
    for info in list(dict(views or {}).values()):
        if not isinstance(info, dict):
            continue
        for key in keys:
            obj = str((info or {}).get(str(key), "") or "").strip()
            if (not obj) or (obj in seen):
                continue
            seen.add(str(obj))
            out.append(str(obj))
    return list(out)


def build_redocking_traj_view_entry(
    *,
    traj_object="",
    frame_object="",
    pose_frame_object="",
    combo_object="",
    state_map=None,
    n_states=0,
):
    return {
        "traj_object": str(traj_object or ""),
        "frame_object": str(frame_object or ""),
        "pose_frame_object": str(pose_frame_object or ""),
        "combo_object": str(combo_object or ""),
        "state_map": dict(state_map or {}),
        "n_states": int(n_states or 0),
    }


def resolve_state_for_member(state_map, member_tag, default_state=0):
    smap = dict(state_map or {})
    tag = str(member_tag or "").strip()
    state = 0
    if tag:
        try:
            state = int(smap.get(tag, 0) or 0)
        except Exception:
            state = 0
    if state <= 0 and tag:
        tag_l = tag.lower()
        for k, v in list(smap.items()):
            if tag_l == str(k or "").strip().lower():
                try:
                    state = int(v or 0)
                except Exception:
                    state = 0
                if state > 0:
                    break
    if state <= 0 and tag:
        m = re.search(r"(\d+)$", tag)
        if m:
            try:
                state = int(m.group(1))
            except Exception:
                state = 0
    if state <= 0:
        try:
            state = int(default_state or 0)
        except Exception:
            state = 0
    return int(max(0, int(state)))


def write_redocking_repeat_stats_csv(csv_path, repeat_rows_meta, include_member_fields=False):
    out_csv = str(csv_path or "")
    if not out_csv:
        return ""
    rows = list(repeat_rows_meta or [])
    try:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    except Exception:
        pass
    try:
        if bool(include_member_fields):
            fieldnames = [
                "repeat",
                "model_member",
                "receptor_object",
                "ok",
                "top1_affinity_kcal",
                "top1_rmsd_a",
                "min_rmsd_all_poses_a",
                "n_poses",
                "job_dir",
                "pose_file",
                "log_file",
                "error",
            ]
        else:
            fieldnames = [
                "repeat",
                "seed",
                "ok",
                "top1_affinity_kcal",
                "top1_rmsd_a",
                "min_rmsd_all_poses_a",
                "n_poses",
                "pose_file",
                "log_file",
                "error",
            ]
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=fieldnames)
            wr.writeheader()
            for rec in rows:
                rr = dict(rec or {})
                try:
                    ba = float(rr.get("best_affinity", float("nan")))
                except Exception:
                    ba = float("nan")
                try:
                    br = float(rr.get("best_rmsd", float("nan")))
                except Exception:
                    br = float("nan")
                row_out = {
                    "repeat": int(rr.get("repeat", 0) or 0),
                    "ok": bool(rr.get("ok", False)),
                    "top1_affinity_kcal": (f"{ba:.6f}" if isfinite(ba) else ""),
                    "top1_rmsd_a": (f"{br:.6f}" if isfinite(br) else ""),
                    "min_rmsd_all_poses_a": (
                        f"{_safe_float(rr.get('min_rmsd_all_poses', float('nan'))):.6f}"
                        if isfinite(_safe_float(rr.get("min_rmsd_all_poses", float("nan")))) else ""
                    ),
                    "n_poses": int(rr.get("n_poses", 0) or 0),
                    "pose_file": str(rr.get("pose_file", "") or ""),
                    "log_file": str(rr.get("log_file", "") or ""),
                    "error": str(rr.get("error", "") or ""),
                }
                if bool(include_member_fields):
                    row_out["model_member"] = str(rr.get("model_member", "") or "")
                    row_out["receptor_object"] = str(rr.get("receptor_object", "") or "")
                    row_out["job_dir"] = str(rr.get("job_dir", "") or "")
                else:
                    row_out["seed"] = int(rr.get("seed", 0) or 0)
                wr.writerow(row_out)
    except Exception:
        return ""
    return out_csv


def summarize_redocking_repeat_rows(
    repeat_rows_meta,
    repeat_statistics_fn=None,
    seed=12345,
    bootstrap_n=2000,
):
    rows = [dict(r) for r in list(repeat_rows_meta or []) if isinstance(r, dict)]
    aff_samples = []
    rms_samples = []
    for rr in rows:
        if not bool(rr.get("ok", False)):
            continue
        try:
            ba = float(rr.get("best_affinity", float("nan")))
        except Exception:
            ba = float("nan")
        try:
            br = float(rr.get("best_rmsd", float("nan")))
        except Exception:
            br = float("nan")
        if isfinite(ba):
            aff_samples.append(float(ba))
        if isfinite(br):
            rms_samples.append(float(br))

    def _repeat_stats(values):
        vv = [float(v) for v in list(values or []) if isfinite(float(v))]
        if callable(repeat_statistics_fn):
            try:
                return dict(
                    repeat_statistics_fn(
                        vv,
                        bootstrap_n=int(bootstrap_n),
                        seed=int(seed),
                    )
                    or {}
                )
            except Exception:
                pass
        n = int(len(vv))
        mean_v = float(sum(vv) / float(n)) if n > 0 else float("nan")
        median_v = float(_median_finite(vv)) if n > 0 else float("nan")
        return {
            "n": int(n),
            "mean": float(mean_v) if isfinite(mean_v) else float("nan"),
            "median": float(median_v) if isfinite(median_v) else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "confidence": ("low" if n > 0 else "undefined"),
        }

    aff_stats = _repeat_stats(aff_samples)
    rms_stats = _repeat_stats(rms_samples)
    finite_aff = [float(v) for v in list(aff_samples or []) if isfinite(float(v))]
    affinity_zero_plateau = bool(len(finite_aff) >= 2 and all(abs(float(v)) <= 1.0e-6 for v in finite_aff))
    # best_affinity/best_rmsd of a repeat are its top-scored pose: success = top-1 RMSD <= 2 A
    n_success = int(sum(1 for v in rms_samples if float(v) <= REDOCK_SUCCESS_RMSD_A))
    min_samples = []
    for rr in rows:
        if bool(rr.get("ok", False)):
            mv = _safe_float(rr.get("min_rmsd_all_poses", float("nan")))
            if isfinite(mv):
                min_samples.append(float(mv))
    return {
        "aff_samples": list(aff_samples),
        "rms_samples": list(rms_samples),
        "aff_stats": dict(aff_stats or {}),
        "rms_stats": dict(rms_stats or {}),
        "affinity_zero_plateau": bool(affinity_zero_plateau),
        "top1_success": {
            "threshold_a": float(REDOCK_SUCCESS_RMSD_A),
            "n_success": int(n_success),
            "n": int(len(rms_samples)),
            "rate": (float(n_success) / float(len(rms_samples))) if rms_samples else float("nan"),
        },
        "min_rmsd_all_poses_median": float(_median_finite(min_samples)) if min_samples else float("nan"),
    }


def prepare_redocking_repeat_outputs(
    *,
    repeat_rows_meta,
    repeat_statistics_fn=None,
    seed=12345,
    bootstrap_n=2000,
    csv_path="",
    include_member_fields=False,
):
    repeat_summary = summarize_redocking_repeat_rows(
        repeat_rows_meta=repeat_rows_meta,
        repeat_statistics_fn=repeat_statistics_fn,
        seed=int(seed),
        bootstrap_n=int(bootstrap_n),
    )
    aff_stats = dict(repeat_summary.get("aff_stats", {}) or {})
    rms_stats = dict(repeat_summary.get("rms_stats", {}) or {})
    affinity_zero_plateau = bool(repeat_summary.get("affinity_zero_plateau", False))
    repeat_csv = ""
    out_csv = str(csv_path or "").strip()
    if out_csv:
        repeat_csv = write_redocking_repeat_stats_csv(
            out_csv,
            repeat_rows_meta,
            include_member_fields=bool(include_member_fields),
        )
    return {
        "repeat_stats_affinity": dict(aff_stats or {}),
        "repeat_stats_rmsd": dict(rms_stats or {}),
        "affinity_zero_plateau": bool(affinity_zero_plateau),
        "repeat_csv": str(repeat_csv or ""),
        "top1_success": dict(repeat_summary.get("top1_success", {}) or {}),
        "min_rmsd_all_poses_median": float(repeat_summary.get("min_rmsd_all_poses_median", float("nan"))),
    }


def build_redocking_run_debug_line(
    *,
    system_label,
    receptor_obj,
    source_receptor_obj,
    removed_ref_atoms,
    repeats,
    aff_stats,
    best_repeat_id,
    pose_count,
    best_row,
    rms_stats,
    flex_enabled=False,
    flex_count=0,
    flex_file="",
    rec_info=None,
    lig_info=None,
    flex_info=None,
):
    best = dict(best_row or {}) if isinstance(best_row, dict) else {}
    try:
        best_aff = float(best.get("affinity", float("nan")))
    except Exception:
        best_aff = float("nan")
    try:
        best_rms = float(best.get("rmsd_ref", float("nan")))
    except Exception:
        best_rms = float("nan")
    best_rms_txt = best.get("rmsd_ref", "n/a") if isfinite(best_rms) else "n/a"
    rec_notes = rec_info.get("notes") if isinstance(rec_info, dict) else rec_info
    lig_notes = lig_info.get("notes") if isinstance(lig_info, dict) else lig_info
    return (
        f"Redocking {system_label}: receptor={receptor_obj}, source={source_receptor_obj or receptor_obj}, "
        f"removed_ref_atoms={int(removed_ref_atoms or 0)}, repeats={int(repeats or 0)}, "
        f"repeat_success={int((aff_stats or {}).get('n', 0) or 0)}, "
        f"best_repeat={best_repeat_id if best_repeat_id is not None else 'n/a'}, poses={int(pose_count or 0)}, "
        f"top1_mode={best.get('mode', 'n/a') if best else 'n/a'}, "
        f"top1_aff={best_aff:.3f}, "
        f"top1_rmsd={best_rms_txt}, "
        f"aff_ci=({(aff_stats or {}).get('ci_low')},{(aff_stats or {}).get('ci_high')}), "
        f"aff_conf={(aff_stats or {}).get('confidence')}, "
        f"rms_ci=({(rms_stats or {}).get('ci_low')},{(rms_stats or {}).get('ci_high')}), "
        f"rms_conf={(rms_stats or {}).get('confidence')}, "
        f"flex_enabled={bool(flex_enabled)}, flex_count={int(flex_count or 0)}, flex_file={flex_file or 'none'}, "
        f"receptor_prep_notes={rec_notes}, ligand_prep_notes={lig_notes}, flex_info={flex_info}"
    )


def build_redocking_single_run_payload(
    *,
    rows,
    best_row,
    job_dir,
    receptor_object,
    source_receptor,
    receptor_input,
    ligand_input,
    removed_reference_atoms=0,
    flex_enabled=False,
    flex_file="",
    flex_info=None,
    repeat_rows=None,
    repeat_stats_affinity=None,
    repeat_stats_rmsd=None,
    repeat_csv="",
    repeat_best=None,
    repeat_count=1,
    affinity_zero_plateau=False,
    top1_success=None,
    min_rmsd_all_poses_median=float("nan"),
):
    return {
        "ok": True,
        "top1_success": dict(top1_success or {}),
        "min_rmsd_all_poses_median": float(min_rmsd_all_poses_median),
        "rows": list(rows or []),
        "best": (dict(best_row or {}) if isinstance(best_row, dict) else best_row),
        "job_dir": str(job_dir or ""),
        "receptor_object": str(receptor_object or ""),
        "source_receptor": str(source_receptor or ""),
        "receptor_input": str(receptor_input or ""),
        "ligand_input": str(ligand_input or ""),
        "removed_reference_atoms": int(removed_reference_atoms or 0),
        "flex_enabled": bool(flex_enabled),
        "flex_file": str(flex_file or ""),
        "flex_info": dict(flex_info or {}),
        "repeat_rows": list(repeat_rows or []),
        "repeat_stats_affinity": dict(repeat_stats_affinity or {}),
        "repeat_stats_rmsd": dict(repeat_stats_rmsd or {}),
        "repeat_csv": str(repeat_csv or ""),
        "repeat_best": (int(repeat_best) if repeat_best is not None else None),
        "repeat_count": int(repeat_count or 0),
        "affinity_zero_plateau": bool(affinity_zero_plateau),
    }


def compose_redocking_single_run_outcome(
    *,
    best_rows,
    repeat_rows_meta,
    job_dir="",
    repeat_statistics_fn=None,
    repeat_seed=12345,
    repeat_bootstrap_n=2000,
    repeat_csv_path="",
    system_label,
    receptor_obj,
    source_receptor_obj,
    receptor_input,
    ligand_input,
    removed_reference_atoms=0,
    n_repeats=1,
    best_repeat_id=None,
    flex_enabled=False,
    flex_count=0,
    flex_file="",
    rec_info=None,
    lig_info=None,
    flex_info=None,
):
    rows = list(best_rows or [])
    best = pick_top_scored_pose_row(rows)
    repeat_outputs = prepare_redocking_repeat_outputs(
        repeat_rows_meta=repeat_rows_meta,
        repeat_statistics_fn=repeat_statistics_fn,
        seed=int(repeat_seed),
        bootstrap_n=int(repeat_bootstrap_n),
        csv_path=str(repeat_csv_path or ""),
        include_member_fields=False,
    )
    aff_stats = dict(repeat_outputs.get("repeat_stats_affinity", {}) or {})
    rms_stats = dict(repeat_outputs.get("repeat_stats_rmsd", {}) or {})
    affinity_zero_plateau = bool(repeat_outputs.get("affinity_zero_plateau", False))
    repeat_csv = str(repeat_outputs.get("repeat_csv", "") or "")
    debug_line = build_redocking_run_debug_line(
        system_label=system_label,
        receptor_obj=receptor_obj,
        source_receptor_obj=(source_receptor_obj or receptor_obj),
        removed_ref_atoms=removed_reference_atoms,
        repeats=n_repeats,
        aff_stats=aff_stats,
        best_repeat_id=best_repeat_id,
        pose_count=len(rows),
        best_row=best,
        rms_stats=rms_stats,
        flex_enabled=bool(flex_enabled),
        flex_count=int(flex_count or 0),
        flex_file=(flex_file or "none"),
        rec_info=rec_info,
        lig_info=lig_info,
        flex_info=flex_info,
    )
    run_payload = build_redocking_single_run_payload(
        rows=rows,
        best_row=best,
        job_dir=str(job_dir or ""),
        receptor_object=receptor_obj,
        source_receptor=(source_receptor_obj or receptor_obj),
        receptor_input=receptor_input,
        ligand_input=ligand_input,
        removed_reference_atoms=removed_reference_atoms,
        flex_enabled=bool(flex_enabled),
        flex_file=(flex_file or ""),
        flex_info=dict(flex_info or {}),
        repeat_rows=repeat_rows_meta,
        repeat_stats_affinity=aff_stats,
        repeat_stats_rmsd=rms_stats,
        repeat_csv=repeat_csv,
        repeat_best=best_repeat_id,
        repeat_count=n_repeats,
        affinity_zero_plateau=bool(affinity_zero_plateau),
        top1_success=dict(repeat_outputs.get("top1_success", {}) or {}),
        min_rmsd_all_poses_median=float(repeat_outputs.get("min_rmsd_all_poses_median", float("nan"))),
    )
    return {
        "debug_line": str(debug_line),
        "run_payload": dict(run_payload or {}),
        "best_row": best,
        "repeat_stats_affinity": dict(aff_stats or {}),
        "repeat_stats_rmsd": dict(rms_stats or {}),
        "repeat_csv": str(repeat_csv or ""),
        "affinity_zero_plateau": bool(affinity_zero_plateau),
    }


def aggregate_label_runs(
    system_label,
    run_list,
    repeat_statistics_fn=None,
    workspace_dir="",
    safe_name_fn=None,
    timestamp_tag="",
):
    label = str(system_label or "System")
    runs = [dict(r) for r in list(run_list or []) if isinstance(r, dict) and bool(r.get("ok", False))]
    if not runs:
        return {"ok": False, "error": f"{label}: no successful runs to aggregate."}
    if len(runs) == 1:
        return runs[0]

    all_rows = []
    repeat_rows_meta = []
    best_repeat = None
    best_repeat_id = None
    for i, rr in enumerate(runs, start=1):
        rows_i = list(rr.get("rows", []) or [])
        member = str(rr.get("ensemble_member", "") or f"{label}_model_{i:03d}")
        for row in rows_i:
            row["system"] = str(label)
            row["system_member"] = str(member)
        all_rows.extend(rows_i)
        best_i = rr.get("best")
        if not isinstance(best_i, dict):
            best_i = pick_top_scored_pose_row(rows_i)
        if is_better_repeat(best_i, best_repeat):
            best_repeat = dict(best_i or {})
            best_repeat_id = int(i)
        seed_val = 0
        rep_rows = list(rr.get("repeat_rows", []) or [])
        if rep_rows:
            try:
                seed_val = int(rep_rows[0].get("seed", 0) or 0)
            except Exception:
                seed_val = 0
        best_aff = float("nan")
        best_rms = float("nan")
        if isinstance(best_i, dict):
            try:
                best_aff = float(best_i.get("affinity", float("nan")))
            except Exception:
                best_aff = float("nan")
            try:
                best_rms = float(best_i.get("rmsd_ref", float("nan")))
            except Exception:
                best_rms = float("nan")
        repeat_rows_meta.append(
            {
                "repeat": int(i),
                "seed": int(seed_val),
                "ok": True,
                "best_affinity": float(best_aff),
                "best_rmsd": float(best_rms),
                "min_rmsd_all_poses": float(
                    _safe_float((pick_best_pose_row(rows_i) or {}).get("rmsd_ref", float("nan")))
                ),
                "n_poses": int(len(rows_i)),
                "pose_file": str(rep_rows[0].get("pose_file", "") if rep_rows else ""),
                "log_file": str(rep_rows[0].get("log_file", "") if rep_rows else ""),
                "error": "",
                "model_member": str(member),
                "job_dir": str(rr.get("job_dir", "") or ""),
                "receptor_object": str(rr.get("receptor_object", "") or ""),
            }
        )

    best_row = pick_top_scored_pose_row(all_rows)
    out_csv = ""
    wk = str(workspace_dir or "").strip()
    if wk:
        ts = str(timestamp_tag or "").strip()
        if not ts:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = str(label)
        if callable(safe_name_fn):
            try:
                safe_label = str(safe_name_fn(label) or label)
            except Exception:
                safe_label = str(label)
        agg_dir = os.path.join(wk, f"redock_ensemble_stats_{safe_label}_{ts}")
        out_csv = os.path.join(agg_dir, "redocking_repeat_stats.csv")
    repeat_outputs = prepare_redocking_repeat_outputs(
        repeat_rows_meta=repeat_rows_meta,
        repeat_statistics_fn=repeat_statistics_fn,
        seed=12345,
        bootstrap_n=2000,
        csv_path=out_csv,
        include_member_fields=True,
    )
    aff_stats = dict(repeat_outputs.get("repeat_stats_affinity", {}) or {})
    rms_stats = dict(repeat_outputs.get("repeat_stats_rmsd", {}) or {})
    affinity_zero_plateau = bool(repeat_outputs.get("affinity_zero_plateau", False))
    repeat_csv = str(repeat_outputs.get("repeat_csv", "") or "")

    return {
        "ok": True,
        "rows": list(all_rows),
        "best": best_row,
        "job_dir": str(repeat_rows_meta[0].get("job_dir", "") if repeat_rows_meta else ""),
        "receptor_object": str(best_row.get("receptor", "") if isinstance(best_row, dict) else ""),
        "source_receptor": str(best_row.get("receptor_source", "") if isinstance(best_row, dict) else ""),
        "receptor_input": "",
        "ligand_input": "",
        "removed_reference_atoms": int(0),
        "flex_enabled": bool(any(bool(r.get("flex_enabled", False)) for r in runs)),
        "flex_file": "",
        "flex_info": {},
        "repeat_rows": list(repeat_rows_meta),
        "repeat_stats_affinity": dict(aff_stats or {}),
        "repeat_stats_rmsd": dict(rms_stats or {}),
        "repeat_csv": str(repeat_csv or ""),
        "repeat_best": int(best_repeat_id) if best_repeat_id is not None else None,
        "repeat_count": int(len(repeat_rows_meta)),
        "affinity_zero_plateau": bool(affinity_zero_plateau),
        "top1_success": dict(repeat_outputs.get("top1_success", {}) or {}),
        "min_rmsd_all_poses_median": float(repeat_outputs.get("min_rmsd_all_poses_median", float("nan"))),
        "ensemble_mode": True,
        "ensemble_members": int(len(runs)),
        "member_runs": list(runs),
    }


def aggregate_label_runs_runtime(
    *,
    system_label,
    run_list,
    repeat_statistics_fn=None,
    workspace_dir_fn=None,
    safe_name_fn=None,
    timestamp_tag_fn=None,
    aggregate_fn=None,
):
    workspace_dir = ""
    if callable(workspace_dir_fn):
        try:
            workspace_dir = str(workspace_dir_fn() or "")
        except Exception:
            workspace_dir = ""
    timestamp_tag = ""
    if callable(timestamp_tag_fn):
        try:
            timestamp_tag = str(timestamp_tag_fn() or "")
        except Exception:
            timestamp_tag = ""
    run_aggregate = aggregate_fn if callable(aggregate_fn) else aggregate_label_runs
    try:
        payload = dict(
            run_aggregate(
                system_label=system_label,
                run_list=list(run_list or []),
                repeat_statistics_fn=repeat_statistics_fn,
                workspace_dir=str(workspace_dir or ""),
                safe_name_fn=safe_name_fn,
                timestamp_tag=str(timestamp_tag or ""),
            )
            or {}
        )
    except Exception as e:
        payload = {
            "ok": False,
            "error": str(e),
        }
    return dict(payload or {})


def pick_comparison_metrics(run_info):
    rr = dict(run_info or {})
    best = dict(rr.get("best", {}) or {})
    aff = float("nan")
    rms = float("nan")
    aff_src = "best pose"
    rms_src = "best pose"
    try:
        aff = float(best.get("affinity", float("nan")))
    except Exception:
        aff = float("nan")
    try:
        rms = float(best.get("rmsd_ref", float("nan")))
    except Exception:
        rms = float("nan")
    affs = dict(rr.get("repeat_stats_affinity", {}) or {})
    rmss = dict(rr.get("repeat_stats_rmsd", {}) or {})
    try:
        n_aff = int(affs.get("n", 0) or 0)
    except Exception:
        n_aff = 0
    try:
        n_rms = int(rmss.get("n", 0) or 0)
    except Exception:
        n_rms = 0
    try:
        median_aff = float(affs.get("median", float("nan"))) if affs else float("nan")
    except Exception:
        median_aff = float("nan")
    try:
        mean_aff = float(affs.get("mean", float("nan"))) if affs else float("nan")
    except Exception:
        mean_aff = float("nan")
    try:
        median_rms = float(rmss.get("median", float("nan"))) if rmss else float("nan")
    except Exception:
        median_rms = float("nan")
    try:
        mean_rms = float(rmss.get("mean", float("nan"))) if rmss else float("nan")
    except Exception:
        mean_rms = float("nan")
    if n_aff >= 2 and isfinite(median_aff):
        aff = float(median_aff)
        aff_src = f"ensemble median (n={n_aff})"
    elif n_aff >= 2 and isfinite(mean_aff):
        aff = float(mean_aff)
        aff_src = f"ensemble mean (n={n_aff})"
    elif n_aff == 1 and (not isfinite(aff)):
        if isfinite(median_aff):
            aff = float(median_aff)
            aff_src = "single-sample median"
        elif isfinite(mean_aff):
            aff = float(mean_aff)
            aff_src = "single-sample mean"
    if n_rms >= 2 and isfinite(median_rms):
        rms = float(median_rms)
        rms_src = f"ensemble median (n={n_rms})"
    elif n_rms >= 2 and isfinite(mean_rms):
        rms = float(mean_rms)
        rms_src = f"ensemble mean (n={n_rms})"
    elif n_rms == 1 and (not isfinite(rms)):
        if isfinite(median_rms):
            rms = float(median_rms)
            rms_src = "single-sample median"
        elif isfinite(mean_rms):
            rms = float(mean_rms)
            rms_src = "single-sample mean"
    return aff, rms, aff_src, rms_src, best


def classify_delta(d_aff, d_rms, aff_hi=1.20, aff_lo=0.60, rms_hi=1.00, rms_lo=0.45):
    cls = "Impacted (mixed effect)"
    rationale = "Mixed/ambiguous shift across affinity and RMSD."
    if isfinite(float(d_aff)) and isfinite(float(d_rms)):
        if d_aff >= aff_hi or d_rms >= rms_hi:
            cls = "Resistant-like"
            rationale = "Mutant weakens docking affinity and/or increases crystal-pose deviation."
        elif d_aff <= -aff_hi and d_rms <= rms_lo:
            cls = "Sensitive-like"
            rationale = "Mutant improves affinity while keeping pose close to crystal reference."
        elif abs(d_aff) <= aff_lo and abs(d_rms) <= rms_lo:
            cls = "Neutral-like"
            rationale = "Small changes in both affinity and pose."
    elif isfinite(float(d_aff)):
        if d_aff >= aff_hi:
            cls = "Resistant-like"
            rationale = "Affinity-only evidence indicates weaker mutant binding."
        elif d_aff <= -aff_hi:
            cls = "Sensitive-like"
            rationale = "Affinity-only evidence indicates stronger mutant binding."
        elif abs(d_aff) <= aff_lo:
            cls = "Neutral-like"
            rationale = "Affinity-only evidence indicates minimal change."
    elif isfinite(float(d_rms)):
        if d_rms >= rms_hi:
            cls = "Resistant-like"
            rationale = "RMSD-only evidence indicates larger mutant pose displacement."
        elif abs(d_rms) <= rms_lo:
            cls = "Neutral-like"
            rationale = "RMSD-only evidence indicates similar binding pose."
    evidence = 0.0
    if isfinite(float(d_aff)):
        evidence += min(abs(float(d_aff)) / float(aff_hi), 1.0)
    if isfinite(float(d_rms)):
        evidence += min(abs(float(d_rms)) / float(rms_hi), 1.0)
    conf = "high" if evidence >= 1.4 else ("medium" if evidence >= 0.8 else "low")
    return cls, rationale, conf, float(evidence)


def sampling_confidence_from_runs(wt_run, mut_run):
    def _score(lbl):
        s = str(lbl or "").strip().lower()
        if not s:
            return 0.0
        if "high" in s:
            return 3.0
        if ("moderate" in s) or ("medium" in s):
            return 2.0
        if "low" in s:
            return 1.0
        return 0.0

    scores = []
    counts = []
    for rr in (wt_run, mut_run):
        if not isinstance(rr, dict):
            continue
        for key in ("repeat_stats_affinity", "repeat_stats_rmsd"):
            st = dict(rr.get(key, {}) or {})
            sc = _score(st.get("confidence", ""))
            if sc > 0.0:
                scores.append(float(sc))
            try:
                nn = int(st.get("n", 0) or 0)
            except Exception:
                nn = 0
            if nn > 0:
                counts.append(int(nn))
    if not scores:
        return "undefined", "No repeat statistics available."
    min_n = int(min(counts)) if counts else 0
    mean_score = float(sum(scores) / max(1, len(scores)))
    if min_n < 2:
        return "low", f"Very few repeats (min n={min_n})."
    if mean_score >= 2.5 and min_n >= 5:
        return "high", f"Stable repeats (min n={min_n})."
    if mean_score >= 1.75 and min_n >= 3:
        return "medium", f"Moderate repeat stability (min n={min_n})."
    return "low", f"Noisy and/or limited repeats (min n={min_n})."


def wt_quality_from_run(wt_run, wt_mean_rms=float("nan"), wt_best_rms=float("nan")):
    rr = dict(wt_run or {})
    rmss = dict(rr.get("repeat_stats_rmsd", {}) or {})
    try:
        n_rms = int(rmss.get("n", 0) or 0)
    except Exception:
        n_rms = 0
    try:
        ci_half = float(rmss.get("ci_half", float("nan")))
    except Exception:
        ci_half = float("nan")
    mean_rms = float(wt_mean_rms) if isfinite(float(wt_mean_rms)) else float("nan")
    if not isfinite(mean_rms):
        try:
            mean_rms = float(rmss.get("mean", float("nan")))
        except Exception:
            mean_rms = float("nan")
    best_rms = float(wt_best_rms) if isfinite(float(wt_best_rms)) else float("nan")
    if not isfinite(best_rms):
        try:
            best_rms = float(dict(rr.get("best", {}) or {}).get("rmsd_ref", float("nan")))
        except Exception:
            best_rms = float("nan")
    ci_tight = (not isfinite(ci_half)) or (float(ci_half) <= 0.8)
    if isfinite(mean_rms) and isfinite(best_rms):
        if mean_rms <= 2.0 and best_rms <= 1.5 and n_rms >= 5 and ci_tight:
            return "excellent", "WT recapitulates crystal pose robustly."
        if mean_rms <= 2.5 and best_rms <= 2.0 and n_rms >= 3:
            return "good", "WT redocking is acceptable."
        if mean_rms <= 3.0:
            return "borderline", "WT redocking is usable but uncertain."
        return "poor", "WT pose reproduction is weak."
    if isfinite(mean_rms):
        if mean_rms <= 2.5:
            return "good", "WT redocking is acceptable (limited best-pose data)."
        if mean_rms <= 3.0:
            return "borderline", "WT redocking is borderline."
        return "poor", "WT redocking is poor."
    return "unknown", "WT ligand RMSD is unavailable."


def _fmt_value(v, nd=3, unit=""):
    try:
        vv = float(v)
    except Exception:
        return "n/a"
    if isfinite(vv):
        return f"{vv:.{int(nd)}f}{unit}"
    return "n/a"


def stats_for_label(run_info):
    rr = run_info if isinstance(run_info, dict) else {}
    if not rr:
        return {
            "aff": float("nan"),
            "rms": float("nan"),
            "aff_metric_label": "best-pose",
            "rms_metric_label": "best-pose",
            "aff_mean": float("nan"),
            "rms_mean": float("nan"),
            "aff_median": float("nan"),
            "rms_median": float("nan"),
            "best_aff": float("nan"),
            "best_rms": float("nan"),
            "n_aff": 0,
            "n_rms": 0,
        }
    aff, rms, _aff_src, _rms_src, best = pick_comparison_metrics(rr)
    affs = dict(rr.get("repeat_stats_affinity", {}) or {})
    rmss = dict(rr.get("repeat_stats_rmsd", {}) or {})
    try:
        n_aff = int(affs.get("n", 0) or 0)
    except Exception:
        n_aff = 0
    try:
        n_rms = int(rmss.get("n", 0) or 0)
    except Exception:
        n_rms = 0
    try:
        mean_aff = float(affs.get("mean", float("nan"))) if n_aff > 0 else float("nan")
    except Exception:
        mean_aff = float("nan")
    try:
        median_aff = float(affs.get("median", float("nan"))) if n_aff > 0 else float("nan")
    except Exception:
        median_aff = float("nan")
    try:
        mean_rms = float(rmss.get("mean", float("nan"))) if n_rms > 0 else float("nan")
    except Exception:
        mean_rms = float("nan")
    if not isfinite(mean_aff):
        mean_aff = float(aff) if isfinite(float(aff)) else float("nan")
    try:
        median_rms = float(rmss.get("median", float("nan"))) if n_rms > 0 else float("nan")
    except Exception:
        median_rms = float("nan")
    if not isfinite(mean_rms):
        mean_rms = float(rms) if isfinite(float(rms)) else float("nan")
    try:
        b_aff = float(best.get("affinity", float("nan"))) if isinstance(best, dict) else float("nan")
    except Exception:
        b_aff = float("nan")
    try:
        b_rms = float(best.get("rmsd_ref", float("nan"))) if isinstance(best, dict) else float("nan")
    except Exception:
        b_rms = float("nan")
    aff_metric_label = "best-pose"
    if n_aff >= 2 and isfinite(median_aff):
        aff_metric_label = "median"
    elif n_aff >= 2 and isfinite(mean_aff):
        aff_metric_label = "mean"
    elif n_aff == 1:
        aff_metric_label = "single-sample"
    rms_metric_label = "best-pose"
    if n_rms >= 2 and isfinite(median_rms):
        rms_metric_label = "median"
    elif n_rms >= 2 and isfinite(mean_rms):
        rms_metric_label = "mean"
    elif n_rms == 1:
        rms_metric_label = "single-sample"
    return {
        "aff": float(aff) if isfinite(aff) else float("nan"),
        "rms": float(rms) if isfinite(rms) else float("nan"),
        "aff_metric_label": str(aff_metric_label),
        "rms_metric_label": str(rms_metric_label),
        "aff_mean": float(mean_aff) if isfinite(mean_aff) else float("nan"),
        "rms_mean": float(mean_rms) if isfinite(mean_rms) else float("nan"),
        "aff_median": float(median_aff) if isfinite(median_aff) else float("nan"),
        "rms_median": float(median_rms) if isfinite(median_rms) else float("nan"),
        "best_aff": float(b_aff) if isfinite(b_aff) else float("nan"),
        "best_rms": float(b_rms) if isfinite(b_rms) else float("nan"),
        "n_aff": int(n_aff),
        "n_rms": int(n_rms),
        "top1_success": dict(rr.get("top1_success", {}) or {}),
        "min_rmsd_all_poses_median": float(_safe_float(rr.get("min_rmsd_all_poses_median", float("nan")))),
    }


def _format_top1_success(stats):
    succ = dict((stats or {}).get("top1_success", {}) or {})
    n = int(succ.get("n", 0) or 0)
    txt = (f"top-1 success (RMSD <= {float(succ.get('threshold_a', REDOCK_SUCCESS_RMSD_A)):.1f} A) "
           f"{int(succ.get('n_success', 0) or 0)}/{n}") if n else "top-1 success n/a"
    low = _safe_float((stats or {}).get("min_rmsd_all_poses_median", float("nan")))
    if isfinite(low):
        txt += f"; lowest RMSD among all modes (sampling, median) {low:.2f} A"
    return txt


def compare_wt_mutant(wt_run, mut_run):
    if (not wt_run) or (not mut_run):
        return "Insufficient WT/mutant results for comparison."
    wt_aff, wt_rms, wt_aff_src, wt_rms_src, wt_best = pick_comparison_metrics(wt_run)
    mut_aff, mut_rms, mut_aff_src, mut_rms_src, mut_best = pick_comparison_metrics(mut_run)
    d_aff = mut_aff - wt_aff if (isfinite(wt_aff) and isfinite(mut_aff)) else float("nan")
    d_rms = mut_rms - wt_rms if (isfinite(wt_rms) and isfinite(mut_rms)) else float("nan")
    cls, rationale, conf_effect, _effect_evidence = classify_delta(d_aff, d_rms)
    conf_sampling, sampling_note = sampling_confidence_from_runs(wt_run, mut_run)
    wt_quality, wt_quality_note = wt_quality_from_run(
        wt_run,
        wt_mean_rms=wt_rms,
        wt_best_rms=(float(wt_best.get("rmsd_ref", float("nan"))) if isinstance(wt_best, dict) else float("nan")),
    )
    wt_mode = int(wt_best.get("mode", 0)) if isinstance(wt_best, dict) else 0
    mut_mode = int(mut_best.get("mode", 0)) if isinstance(mut_best, dict) else 0
    wt_mode_txt = f"best mode={wt_mode}" if wt_mode > 0 else "best mode=n/a"
    mut_mode_txt = f"best mode={mut_mode}" if mut_mode > 0 else "best mode=n/a"
    return (
        f"WT: affinity={_fmt_value(wt_aff, 3, ' kcal/mol')} ({wt_aff_src}), "
        f"ligand RMSD to crystal={_fmt_value(wt_rms, 3, ' A')} ({wt_rms_src}), {wt_mode_txt}\n"
        f"Mutant: affinity={_fmt_value(mut_aff, 3, ' kcal/mol')} ({mut_aff_src}), "
        f"ligand RMSD to crystal={_fmt_value(mut_rms, 3, ' A')} ({mut_rms_src}), {mut_mode_txt}\n"
        f"Delta (mut-wt): affinity={_fmt_value(d_aff, 3, ' kcal/mol')}, ligand RMSD={_fmt_value(d_rms, 3, ' A')}\n"
        f"Classification: {cls}\n"
        f"confidence_effect={conf_effect}, confidence_sampling={conf_sampling}, redocking_quality={wt_quality}\n"
        f"Rationale: {rationale}\n"
        f"Sampling note: {sampling_note}\n"
        f"WT quality note: {wt_quality_note}"
    )


def _ordered_mutant_labels(run_map, wt_label="WT"):
    rm = dict(run_map or {})
    wt_key = str(wt_label or "WT")
    out = []
    for key in list(rm.keys()):
        lab = str(key or "").strip()
        if (not lab) or (lab == wt_key):
            continue
        low = lab.lower()
        if low.startswith("mut"):
            out.append(lab)
    if out:
        return list(out)
    # Fallback: if no "Mut*" labels exist, compare all non-WT labels.
    for key in list(rm.keys()):
        lab = str(key or "").strip()
        if (not lab) or (lab == wt_key):
            continue
        out.append(lab)
    return list(out)


def build_redocking_comparison_rows(run_map, wt_label="WT", mutant_labels=None):
    rm = dict(run_map or {})
    wt_key = str(wt_label or "WT")
    wt_run = dict(rm.get(wt_key, {}) or {})
    if not wt_run:
        return []

    wt_stats = stats_for_label(wt_run)
    wt_aff = float(wt_stats.get("aff", float("nan")))
    wt_rms = float(wt_stats.get("rms", float("nan")))
    wt_quality, wt_quality_note = wt_quality_from_run(
        wt_run,
        wt_mean_rms=wt_rms,
        wt_best_rms=wt_stats.get("best_rms", float("nan")),
    )
    labels = list(mutant_labels or _ordered_mutant_labels(rm, wt_label=wt_key))
    out = []
    for lab in labels:
        mut_run = dict(rm.get(str(lab), {}) or {})
        if not mut_run:
            continue
        mut_stats = stats_for_label(mut_run)
        mut_aff = float(mut_stats.get("aff", float("nan")))
        mut_rms = float(mut_stats.get("rms", float("nan")))
        d_aff = mut_aff - wt_aff if (isfinite(mut_aff) and isfinite(wt_aff)) else float("nan")
        d_rms = mut_rms - wt_rms if (isfinite(mut_rms) and isfinite(wt_rms)) else float("nan")
        cls, rationale, conf_effect, _effect_evidence = classify_delta(d_aff, d_rms)
        conf_sampling, sampling_note = sampling_confidence_from_runs(wt_run, mut_run)
        out.append(
            {
                "wt_label": str(wt_key),
                "mutant_label": str(lab),
                "mutations": "",
                "wt_affinity_label": str(wt_stats.get("aff_metric_label", "") or ""),
                "wt_rmsd_label": str(wt_stats.get("rms_metric_label", "") or ""),
                "wt_affinity_kcal": float(wt_aff) if isfinite(wt_aff) else float("nan"),
                "wt_rmsd_a": float(wt_rms) if isfinite(wt_rms) else float("nan"),
                "mut_affinity_label": str(mut_stats.get("aff_metric_label", "") or ""),
                "mut_rmsd_label": str(mut_stats.get("rms_metric_label", "") or ""),
                "mut_affinity_kcal": float(mut_aff) if isfinite(mut_aff) else float("nan"),
                "mut_rmsd_a": float(mut_rms) if isfinite(mut_rms) else float("nan"),
                "delta_affinity_kcal": float(d_aff) if isfinite(d_aff) else float("nan"),
                "delta_rmsd_a": float(d_rms) if isfinite(d_rms) else float("nan"),
                "classification": str(cls),
                "confidence_effect": str(conf_effect),
                "confidence_sampling": str(conf_sampling),
                "redocking_quality": str(wt_quality),
                "rationale": str(rationale),
                "sampling_note": str(sampling_note),
                "wt_quality_note": str(wt_quality_note),
                "wt_repeat_csv": str(wt_run.get("repeat_csv", "") or ""),
                "mut_repeat_csv": str(mut_run.get("repeat_csv", "") or ""),
            }
        )
    return list(out)


def write_redocking_comparison_summary_csv(csv_path, comparison_rows):
    out_csv = str(csv_path or "").strip()
    if not out_csv:
        return ""
    rows = [dict(r or {}) for r in list(comparison_rows or []) if isinstance(r, dict)]
    if not rows:
        return ""
    try:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    except Exception:
        pass
    fieldnames = [
        "wt_label",
        "mutant_label",
        "mutations",
        "wt_affinity_kcal",
        "wt_rmsd_a",
        "mut_affinity_kcal",
        "mut_rmsd_a",
        "delta_affinity_kcal",
        "delta_rmsd_a",
        "classification",
        "confidence_effect",
        "confidence_sampling",
        "redocking_quality",
        "rationale",
        "sampling_note",
        "wt_quality_note",
        "wt_repeat_csv",
        "mut_repeat_csv",
    ]
    try:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=fieldnames)
            wr.writeheader()
            for rr in rows:
                row_out = {}
                for key in fieldnames:
                    val = rr.get(key, "")
                    if isinstance(val, float):
                        row_out[key] = f"{float(val):.6f}" if isfinite(float(val)) else ""
                    else:
                        row_out[key] = str(val or "")
                wr.writerow(row_out)
    except Exception:
        return ""
    return out_csv


def build_redocking_summary_lines(
    run_map,
    nref,
    center,
    size,
    *,
    use_reference_ligand=False,
    use_modeler_ensemble=False,
    ensemble_counts=None,
    repeats=1,
    site_qc_enabled=False,
    site_qc_rmsd_max_a=2.0,
    site_qc_min_anchor_matches=4,
    site_anchor_cutoff_a=6.0,
    site_qc_notes=None,
):
    rm = dict(run_map or {})
    ec = dict(ensemble_counts or {})
    wt_run = dict(rm.get("WT", {}) or {})
    wt_stats = stats_for_label(wt_run)
    mut_labels = _ordered_mutant_labels(rm, wt_label="WT")
    lines = [
        f"Reference ligand atoms: {int(nref)}",
        (
            "Grid: center=("
            f"{float(center[0]):.3f}, {float(center[1]):.3f}, {float(center[2]):.3f}), "
            f"size=({float(size[0]):.2f}, {float(size[1]):.2f}, {float(size[2]):.2f}) A"
        ),
    ]
    if bool(use_reference_ligand):
        lines.append("Ligand mode: crystal reference ligand (true redocking).")
    else:
        lines.append("Ligand mode: external ligand source.")
    if bool(site_qc_enabled):
        cutoff_txt = (
            f" within {float(site_anchor_cutoff_a):.1f} A of crystal ligand"
            if isfinite(_safe_float(site_anchor_cutoff_a, default=float("nan")))
            else ""
        )
        lines.append(
            "Local site QC: "
            f"anchor CA RMSD <= {float(site_qc_rmsd_max_a):.2f} A "
            f"with at least {int(site_qc_min_anchor_matches or 4)} matched residues{cutoff_txt}."
        )
        for note in [str(x) for x in list(site_qc_notes or []) if str(x or "").strip()]:
            lines.append(f"Site QC: {note}")
    lines.append(
        "Reported RMSD values are heavy-atom, symmetry-corrected ligand RMSD of the top-scored pose to the "
        "crystal reference ligand (no superposition); success = top-1 RMSD <= 2 A. Site QC RMSD is only a "
        "preparation check."
    )
    if bool(use_modeler_ensemble):
        mut_total = int(sum(int(ec.get(lbl, 0) or 0) for lbl in list(mut_labels or [])) or 0)
        if mut_total > 0:
            lines.append(
                "Sampling: MODELLER ensemble "
                f"(WT={int(ec.get('WT', 0))}, Mutants total={int(mut_total)} models)."
            )
        else:
            lines.append(
                "Sampling: MODELLER ensemble "
                f"(WT={int(ec.get('WT', 0))} models)."
            )
    else:
        lines.append(f"Sampling: repeats={int(repeats)}.")
    if wt_run:
        lines.append(
            f"WT: {str(wt_stats.get('aff_metric_label', 'best-pose'))} affinity={_fmt_value(wt_stats.get('aff'), 3, ' kcal/mol')} (n={int(wt_stats.get('n_aff', 0))}), "
            f"{str(wt_stats.get('rms_metric_label', 'best-pose'))} top-1 ligand RMSD={_fmt_value(wt_stats.get('rms'), 3, ' A')} (n={int(wt_stats.get('n_rms', 0))}), "
            f"best-scored pose RMSD={_fmt_value(wt_stats.get('best_rms'), 3, ' A')}; "
            f"{_format_top1_success(wt_stats)}"
        )
    if mut_labels:
        for lab in list(mut_labels or []):
            mut_run = dict(rm.get(str(lab), {}) or {})
            if not mut_run:
                continue
            mut_stats = stats_for_label(mut_run)
            lines.append(
                f"{lab}: {str(mut_stats.get('aff_metric_label', 'best-pose'))} affinity={_fmt_value(mut_stats.get('aff'), 3, ' kcal/mol')} (n={int(mut_stats.get('n_aff', 0))}), "
                f"{str(mut_stats.get('rms_metric_label', 'best-pose'))} top-1 ligand RMSD={_fmt_value(mut_stats.get('rms'), 3, ' A')} (n={int(mut_stats.get('n_rms', 0))}), "
                f"best-scored pose RMSD={_fmt_value(mut_stats.get('best_rms'), 3, ' A')}; "
                f"{_format_top1_success(mut_stats)}"
            )
            if wt_run:
                d_aff = float(mut_stats.get("aff", float("nan"))) - float(wt_stats.get("aff", float("nan")))
                d_rms = float(mut_stats.get("rms", float("nan"))) - float(wt_stats.get("rms", float("nan")))
                cls, rationale, conf_effect, _effect_evidence = classify_delta(d_aff, d_rms)
                conf_sampling, _sampling_note = sampling_confidence_from_runs(
                    wt_run,
                    mut_run,
                )
                wt_quality, _wt_quality_note = wt_quality_from_run(
                    wt_run,
                    wt_mean_rms=wt_stats.get("rms", float("nan")),
                    wt_best_rms=wt_stats.get("best_rms", float("nan")),
                )
                lines.append(
                    f"Verdict ({lab} vs WT): {cls} | confidence_effect={conf_effect}, confidence_sampling={conf_sampling}, redocking_quality={wt_quality}"
                )
                lines.append(
                    f"Delta ({lab}-WT): affinity={_fmt_value(d_aff, 3, ' kcal/mol')}, ligand RMSD={_fmt_value(d_rms, 3, ' A')}"
                )
                lines.append(f"Rationale ({lab}): {rationale}")
    return lines


def run_redocking_repeats(
    base_cmd,
    job_dir,
    n_repeats,
    seed0,
    run_command,
    parse_pose_scores,
    build_rows_for_repeat,
    timeout=7200,
):
    if not callable(run_command):
        raise TypeError("run_command must be callable")
    if not callable(parse_pose_scores):
        raise TypeError("parse_pose_scores must be callable")
    if not callable(build_rows_for_repeat):
        raise TypeError("build_rows_for_repeat must be callable")

    repeats = int(max(1, n_repeats))
    base_seed = int(seed0 or 0)
    root_dir = str(job_dir or "")
    cmd_base = list(base_cmd or [])

    repeat_rows_meta = []
    best_rows = []
    best_repeat = None
    best_repeat_id = None
    best_pose_file = ""

    for rep in range(1, repeats + 1):
        rep_dir = root_dir if repeats == 1 else os.path.join(root_dir, f"repeat_{rep:03d}")
        os.makedirs(rep_dir, exist_ok=True)
        out_pose = os.path.join(rep_dir, "docked_poses.pdbqt")
        out_log = os.path.join(rep_dir, "docking.log")
        seed_val = int(base_seed + rep - 1) if base_seed > 0 else 0
        cmdline = list(cmd_base)
        if seed_val > 0:
            cmdline.extend(["--seed", str(seed_val)])
        if ("--flex" in cmdline or "--flexres" in cmdline) and ("--out_flex" not in cmdline):
            cmdline.extend(["--out_flex", os.path.join(rep_dir, "flex_residues_out.pdbqt")])
        cmdline.extend(["--out", out_pose, "--log", out_log])
        retry_steps = []
        attempt_idx = 0
        while True:
            ok, out, err, rc = run_command(cmdline, cwd=rep_dir, timeout=int(timeout))
            combined_text = _collect_redocking_output_text(out_log, out=out, err=err)
            if (
                attempt_idx < 4
                and _redocking_needs_flex_search_space_retry(combined_text)
            ):
                retry_cmdline, retry_add = _expand_redocking_flex_autobox_cmdline(cmdline)
                if retry_cmdline:
                    attempt_idx += 1
                    cmdline = list(retry_cmdline)
                    retry_steps.append(f"{float(retry_add):.3f}")
                    continue
            break
        retry_note = ""
        if retry_steps:
            retry_note = f" Auto-retried with autobox_add={','.join(retry_steps)}."
        rec_meta = {
            "repeat": int(rep),
            "seed": int(seed_val),
            "ok": False,
            "best_affinity": float("nan"),
            "best_rmsd": float("nan"),
            "n_poses": 0,
            "pose_file": str(out_pose),
            "log_file": str(out_log),
            "error": "",
        }
        if (not bool(ok)) or (not os.path.isfile(out_pose)):
            rec_meta["error"] = f"smina rc={int(rc)}:{retry_note} {str((err or out) or '')[-1200:]}".strip()
            repeat_rows_meta.append(rec_meta)
            continue

        poses = list(parse_pose_scores(out_pose, out_log) or [])
        if not poses:
            poses = list(_parse_redocking_pose_rows_from_text("\n".join([str(out or ""), str(err or "")])) or [])
        if not poses:
            log_tail = ""
            try:
                if os.path.isfile(out_log):
                    with open(out_log, "r", encoding="utf-8", errors="ignore") as fh:
                        txt = str(fh.read() or "")
                    log_tail = txt[-800:].strip()
            except Exception:
                log_tail = ""
            std_tail = "\n".join(
                [str(x or "").strip() for x in [out, err] if str(x or "").strip()]
            )[-800:].strip()
            rec_meta["error"] = ("No poses were parsed from docking output." + str(retry_note or "")).strip()
            if log_tail:
                rec_meta["error"] += f" Log tail: {log_tail}"
            if std_tail:
                rec_meta["error"] += f" Output tail: {std_tail}"
            repeat_rows_meta.append(rec_meta)
            continue

        rows_i = list(build_rows_for_repeat(rep, poses, out_pose, out_log) or [])
        if not rows_i:
            rec_meta["error"] = "No pose rows were produced for repeat."
            repeat_rows_meta.append(rec_meta)
            continue

        rep_best = pick_top_scored_pose_row(rows_i)
        rec_meta["ok"] = True
        rec_meta["n_poses"] = int(len(rows_i))
        lowest = pick_best_pose_row(rows_i)
        rec_meta["min_rmsd_all_poses"] = float(_safe_float((lowest or {}).get("rmsd_ref", float("nan"))))
        if rep_best is not None:
            try:
                rec_meta["best_affinity"] = float((rep_best or {}).get("affinity", float("nan")))
            except Exception:
                rec_meta["best_affinity"] = float("nan")
            try:
                rec_meta["best_rmsd"] = float((rep_best or {}).get("rmsd_ref", float("nan")))
            except Exception:
                rec_meta["best_rmsd"] = float("nan")
        repeat_rows_meta.append(rec_meta)

        if is_better_repeat(rep_best, best_repeat):
            best_repeat = dict(rep_best or {})
            best_rows = list(rows_i)
            best_repeat_id = int(rep)
            best_pose_file = str(out_pose)

    if (not best_rows) or (not best_pose_file) or (not os.path.isfile(best_pose_file)):
        tail = format_failed_repeats_tail(repeat_rows_meta, max_rows=3, max_chars=220)
        return {
            "ok": False,
            "error": ("All redocking repeats failed." + (tail if tail else "")),
            "repeat_rows": repeat_rows_meta,
        }

    return {
        "ok": True,
        "repeat_rows": repeat_rows_meta,
        "best_rows": best_rows,
        "best_repeat": best_repeat,
        "best_repeat_id": int(best_repeat_id) if best_repeat_id is not None else None,
        "best_pose_file": str(best_pose_file),
    }


def run_redocking_repeat_pipeline(
    *,
    base_cmd,
    job_dir,
    n_repeats,
    seed0,
    system_label,
    receptor_obj,
    source_receptor_obj="",
    run_command=None,
    parse_pose_scores_fn=None,
    compute_rmsd_values_fn=None,
    build_rows_for_repeat_fn=None,
    run_repeats_fn=None,
    timeout=7200,
):
    if not callable(run_command):
        raise TypeError("run_command must be callable")
    if not callable(parse_pose_scores_fn):
        raise TypeError("parse_pose_scores_fn must be callable")
    if not callable(compute_rmsd_values_fn):
        raise TypeError("compute_rmsd_values_fn must be callable")
    rows_builder = build_rows_for_repeat_fn if callable(build_rows_for_repeat_fn) else build_redocking_rows_for_repeat
    if not callable(rows_builder):
        raise TypeError("build_rows_for_repeat_fn must be callable")
    repeat_runner = run_repeats_fn if callable(run_repeats_fn) else run_redocking_repeats

    src_obj = str(source_receptor_obj or receptor_obj)
    rec_obj = str(receptor_obj or "")
    job_root = str(job_dir or "")

    def _build_rows_for_repeat(rep, poses, out_pose, out_log):
        rmsd_vals = list(
            compute_rmsd_values_fn(
                rep=rep,
                poses=list(poses or []),
                out_pose=out_pose,
                out_log=out_log,
            )
            or []
        )
        return list(
            rows_builder(
                rep=rep,
                poses=poses,
                rmsd_vals=rmsd_vals,
                system_label=system_label,
                receptor_obj=rec_obj,
                source_receptor_obj=src_obj,
                job_dir=job_root,
            )
            or []
        )

    rep_payload = repeat_runner(
        base_cmd=list(base_cmd or []),
        job_dir=job_root,
        n_repeats=int(n_repeats),
        seed0=int(seed0),
        run_command=run_command,
        parse_pose_scores=parse_pose_scores_fn,
        build_rows_for_repeat=_build_rows_for_repeat,
        timeout=int(timeout),
    )
    if not bool(rep_payload.get("ok", False)):
        return {
            "ok": False,
            "error": str(rep_payload.get("error", "All redocking repeats failed.")),
            "job_dir": str(job_root),
            "repeat_rows": list(rep_payload.get("repeat_rows", []) or []),
        }
    out = dict(rep_payload or {})
    out["ok"] = True
    out["job_dir"] = str(job_root)
    return out


def run_redocking_single_run_pipeline(
    *,
    smina_bin,
    receptor_for_dock,
    ligand_input,
    autobox_ligand="",
    center,
    size,
    params=None,
    flex_file="",
    job_dir="",
    system_label="",
    receptor_obj="",
    source_receptor_obj="",
    tag="",
    safe_name_fn=None,
    convert_pdbqt_to_sdf_fn=None,
    convert_pdbqt_to_pdb_fn=None,
    load_pose_object_fn=None,
    run_command=None,
    parse_pose_scores_fn=None,
    compute_rmsd_values_fn=None,
    build_rows_for_repeat_fn=None,
    run_repeats_fn=None,
    repeat_statistics_fn=None,
    repeat_seed=12345,
    repeat_bootstrap_n=2000,
    repeat_csv_path="",
    receptor_input="",
    removed_reference_atoms=0,
    flex_enabled=False,
    flex_rows=None,
    rec_info=None,
    lig_info=None,
    flex_info=None,
    timeout=7200,
    pose_prefix="redock",
):
    plan_payload = build_redocking_smina_plan(
        smina_bin=smina_bin,
        receptor_for_dock=receptor_for_dock,
        ligand_input=ligand_input,
        autobox_ligand=autobox_ligand,
        center=center,
        size=size,
        params=params,
        flex_file=flex_file,
        native_flexres=(dict(flex_info or {}).get("spec", "") if bool(dict(flex_info or {}).get("native", False)) else ""),
    )
    n_repeats = int(plan_payload.get("n_repeats", 1) or 1)
    seed0 = int(plan_payload.get("seed0", 0) or 0)
    base_cmd = list(plan_payload.get("base_cmd", []) or [])
    rep_payload = run_redocking_repeat_pipeline(
        base_cmd=base_cmd,
        job_dir=job_dir,
        n_repeats=n_repeats,
        seed0=seed0,
        system_label=system_label,
        receptor_obj=receptor_obj,
        source_receptor_obj=(source_receptor_obj or receptor_obj),
        run_command=run_command,
        parse_pose_scores_fn=parse_pose_scores_fn,
        compute_rmsd_values_fn=compute_rmsd_values_fn,
        build_rows_for_repeat_fn=build_rows_for_repeat_fn,
        run_repeats_fn=run_repeats_fn,
        timeout=int(timeout),
    )
    if not bool(rep_payload.get("ok", False)):
        out = dict(rep_payload)
        out["debug_messages"] = []
        return out

    repeat_rows_meta = list(rep_payload.get("repeat_rows", []) or [])
    best_rows = list(rep_payload.get("best_rows", []) or [])
    best_repeat_id = rep_payload.get("best_repeat_id", None)
    best_pose_file = str(rep_payload.get("best_pose_file", "") or "")

    pose_outputs = prepare_redocking_pose_outputs(
        best_rows=best_rows,
        best_pose_file=best_pose_file,
        receptor_obj=receptor_obj,
        system_label=system_label,
        tag=tag,
        safe_name_fn=safe_name_fn,
        convert_pdbqt_to_sdf_fn=convert_pdbqt_to_sdf_fn,
        convert_pdbqt_to_pdb_fn=convert_pdbqt_to_pdb_fn,
        load_pose_object_fn=load_pose_object_fn,
        prefix=pose_prefix,
    )
    best_rows = list(pose_outputs.get("rows", []) or [])

    stats_seed = int(seed0) if int(seed0) > 0 else int(repeat_seed)
    outcome_payload = compose_redocking_single_run_outcome(
        best_rows=best_rows,
        repeat_rows_meta=repeat_rows_meta,
        job_dir=job_dir,
        repeat_statistics_fn=repeat_statistics_fn,
        repeat_seed=stats_seed,
        repeat_bootstrap_n=int(repeat_bootstrap_n),
        repeat_csv_path=repeat_csv_path,
        system_label=system_label,
        receptor_obj=receptor_obj,
        source_receptor_obj=(source_receptor_obj or receptor_obj),
        receptor_input=receptor_input,
        ligand_input=ligand_input,
        removed_reference_atoms=removed_reference_atoms,
        n_repeats=n_repeats,
        best_repeat_id=best_repeat_id,
        flex_enabled=bool(flex_enabled),
        flex_count=len(list(flex_rows or [])),
        flex_file=(flex_file or ""),
        rec_info=rec_info,
        lig_info=lig_info,
        flex_info=flex_info,
    )

    debug_messages = []
    pose_warning = str(pose_outputs.get("load_warning", "") or "").strip()
    if pose_warning:
        debug_messages.append(str(pose_warning))
    debug_line = str(outcome_payload.get("debug_line", "") or "").strip()
    if debug_line:
        debug_messages.append(str(debug_line))

    return {
        "ok": True,
        "job_dir": str(job_dir or ""),
        "run_payload": dict(outcome_payload.get("run_payload", {}) or {}),
        "debug_messages": list(debug_messages),
        "plan_payload": dict(plan_payload or {}),
        "repeat_payload": dict(rep_payload or {}),
    }


def run_redocking_single_system_orchestrator(
    *,
    system_label,
    receptor_obj,
    reference_obj=None,
    ref_fields,
    ref_mol,
    center,
    size,
    params=None,
    source_receptor_obj="",
    flex_enabled=False,
    flex_residues=None,
    auto_use_cached_hetatm=False,
    ligand_override_file="",
    smina_bin="",
    picked_ligand_file="",
    workspace_dir="",
    safe_name_fn=None,
    file_exists_fn=None,
    makedirs_fn=None,
    prepare_receptor_fn=None,
    strip_reference_residue_fn=None,
    prepare_flexible_receptor_fn=None,
    native_flexres_supported_fn=None,
    build_native_flexres_spec_fn=None,
    should_preserve_charges_fn=None,
    current_charge_mode_fn=None,
    parameterized_ligand_mol2="",
    samefile_or_equal_fn=None,
    chemically_equivalent_fn=None,
    pubchem_smiles_hint_fn=None,
    rcsb_smiles_by_het_fn=None,
    prepare_ligand_fn=None,
    has_rdkit=False,
    convert_pdbqt_to_sdf_fn=None,
    convert_pdbqt_to_pdb_fn=None,
    load_pose_molecules_fn=None,
    best_rmsd_fn=None,
    load_pose_object_fn=None,
    run_command=None,
    parse_pose_scores_fn=None,
    repeat_statistics_fn=None,
    timeout=7200,
    flow_label="redocking",
    pose_prefix="redock",
    repeat_seed=12345,
    repeat_bootstrap_n=2000,
    repeat_csv_filename="redocking_repeat_stats.csv",
    compute_rmsd_values_for_repeat_fn=None,
    single_run_pipeline_fn=None,
):
    debug_messages = []
    pp = dict(params or {})
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    preflight_payload = prepare_redocking_single_run_context(
        smina_bin=smina_bin,
        ligand_override_file=str(ligand_override_file or ""),
        picked_ligand_file=str(picked_ligand_file or ""),
        workspace_dir=str(workspace_dir or ""),
        system_label=system_label,
        receptor_obj=receptor_obj,
        params=pp,
        safe_name_fn=safe_name_fn,
        file_exists_fn=exists_fn,
        makedirs_fn=makedirs_fn,
    )
    if not bool(preflight_payload.get("ok", False)):
        out = dict(preflight_payload or {})
        out["debug_messages"] = list(debug_messages)
        return out

    smina_bin = str(preflight_payload.get("smina_bin", "") or "")
    lig_file = str(preflight_payload.get("ligand_file", "") or "")
    tag = str(preflight_payload.get("tag", "") or "")
    job_dir = str(preflight_payload.get("job_dir", "") or "")
    target_ph = float(preflight_payload.get("target_ph", 7.4))
    native_flexres_supported = False
    if bool(flex_enabled) and callable(native_flexres_supported_fn):
        try:
            native_flexres_supported = bool(native_flexres_supported_fn(smina_bin, run_command))
        except Exception:
            native_flexres_supported = False

    receptor_flex_payload = prepare_redocking_receptor_flex_context(
        receptor_obj=receptor_obj,
        job_dir=job_dir,
        target_ph=target_ph,
        ref_fields=ref_fields,
        flex_enabled=bool(flex_enabled),
        flex_residues=flex_residues,
        flow_label=str(flow_label or "redocking"),
        auto_use_cached_hetatm=bool(auto_use_cached_hetatm),
        prepare_receptor_fn=prepare_receptor_fn,
        strip_reference_residue_fn=strip_reference_residue_fn,
        prepare_flexible_receptor_fn=prepare_flexible_receptor_fn,
        native_flexres_supported=bool(native_flexres_supported),
        build_native_flexres_spec_fn=build_native_flexres_spec_fn,
        file_exists_fn=exists_fn,
    )
    if not bool(receptor_flex_payload.get("ok", False)):
        out = dict(receptor_flex_payload or {})
        out["debug_messages"] = list(debug_messages)
        return out

    receptor_input = str(receptor_flex_payload.get("receptor_input", "") or "")
    rec_info = receptor_flex_payload.get("rec_info", {})
    receptor_for_dock = str(receptor_flex_payload.get("receptor_for_dock", receptor_input) or receptor_input)
    removed = int(receptor_flex_payload.get("removed_reference_atoms", 0) or 0)
    flex_rows = list(receptor_flex_payload.get("flex_rows", []) or [])
    flex_file = str(receptor_flex_payload.get("flex_file", "") or "")
    flex_info = dict(receptor_flex_payload.get("flex_info", {}) or {})

    preserve_input_charges = False
    if callable(should_preserve_charges_fn):
        try:
            preserve_input_charges = bool(should_preserve_charges_fn(lig_file))
        except Exception:
            preserve_input_charges = False
    charge_mode = ""
    if callable(current_charge_mode_fn):
        try:
            charge_mode = str(current_charge_mode_fn() or "")
        except Exception:
            charge_mode = ""

    ligand_payload = prepare_redocking_ligand_context(
        ligand_file=lig_file,
        job_dir=job_dir,
        target_ph=target_ph,
        ligand_override_file=ligand_override_file,
        ref_fields=ref_fields,
        preserve_input_charges=bool(preserve_input_charges),
        charge_mode=charge_mode,
        parameterized_ligand_mol2=str(parameterized_ligand_mol2 or ""),
        file_exists_fn=exists_fn,
        samefile_or_equal_fn=samefile_or_equal_fn,
        chemically_equivalent_fn=chemically_equivalent_fn,
        pubchem_smiles_hint_fn=pubchem_smiles_hint_fn,
        rcsb_smiles_by_het_fn=rcsb_smiles_by_het_fn,
        prepare_ligand_fn=prepare_ligand_fn,
    )
    debug_messages.extend(list(ligand_payload.get("debug_messages", []) or []))
    if not bool(ligand_payload.get("ok", False)):
        out = dict(ligand_payload or {})
        out["debug_messages"] = list(debug_messages)
        return out

    ligand_input = str(ligand_payload.get("ligand_input", "") or "")
    lig_info = ligand_payload.get("lig_info", {})
    autobox_ligand = ""
    lig_override = str(ligand_override_file or "").strip()
    ref_obj = str(reference_obj or "").strip()
    explicit_grid_available = bool(_has_explicit_redocking_box(center, size))
    if lig_override and bool(exists_fn(lig_override)) and (not explicit_grid_available):
        receptor_name = str(receptor_obj or "").strip()
        source_name = str(source_receptor_obj or receptor_obj or "").strip()
        if receptor_name != source_name:
            autobox_ligand = str(lig_override)
        elif ref_obj and receptor_name == ref_obj:
            autobox_ligand = str(lig_override)
    if autobox_ligand:
        debug_messages.append(
            f"Redocking search box: ligand-guided autobox using crystal reference ({autobox_ligand})."
        )
    elif lig_override and explicit_grid_available:
        debug_messages.append(
            "Redocking search box: using explicit aligned grid center/size; crystal-reference autobox disabled."
        )

    rmsd_builder = (
        compute_rmsd_values_for_repeat_fn
        if callable(compute_rmsd_values_for_repeat_fn)
        else compute_redocking_rmsd_values_for_repeat
    )

    def _compute_rmsd_values_for_repeat(rep, poses, out_pose, out_log=None, **_kwargs):
        _ = out_log
        return list(
            rmsd_builder(
                out_pose=out_pose,
                poses_count=len(list(poses or [])),
                has_rdkit=bool(has_rdkit),
                ref_mol=ref_mol,
                convert_pdbqt_to_sdf_fn=convert_pdbqt_to_sdf_fn,
                load_pose_mols_fn=load_pose_molecules_fn,
                best_rmsd_fn=best_rmsd_fn,
                log_warning_fn=lambda msg: debug_messages.append(
                    f"Redocking RMSD evaluation warning ({system_label}, repeat {int(rep)}): {msg}"
                ),
            )
            or []
        )

    single_runner = single_run_pipeline_fn if callable(single_run_pipeline_fn) else run_redocking_single_run_pipeline
    run_payload = single_runner(
        smina_bin=smina_bin,
        receptor_for_dock=receptor_for_dock,
        ligand_input=ligand_input,
        autobox_ligand=autobox_ligand,
        center=center,
        size=size,
        params=pp,
        flex_file=flex_file,
        job_dir=job_dir,
        system_label=system_label,
        receptor_obj=receptor_obj,
        source_receptor_obj=(source_receptor_obj or receptor_obj),
        tag=tag,
        safe_name_fn=safe_name_fn,
        convert_pdbqt_to_sdf_fn=convert_pdbqt_to_sdf_fn,
        convert_pdbqt_to_pdb_fn=convert_pdbqt_to_pdb_fn,
        load_pose_object_fn=load_pose_object_fn,
        run_command=run_command,
        parse_pose_scores_fn=parse_pose_scores_fn,
        compute_rmsd_values_fn=_compute_rmsd_values_for_repeat,
        repeat_statistics_fn=repeat_statistics_fn,
        repeat_seed=int(repeat_seed),
        repeat_bootstrap_n=int(repeat_bootstrap_n),
        repeat_csv_path=os.path.join(job_dir, str(repeat_csv_filename or "redocking_repeat_stats.csv")),
        receptor_input=receptor_for_dock,
        removed_reference_atoms=removed,
        flex_enabled=bool(flex_enabled),
        flex_rows=flex_rows,
        rec_info=rec_info,
        lig_info=lig_info,
        flex_info=flex_info,
        timeout=int(timeout),
        pose_prefix=pose_prefix,
    )
    debug_messages.extend(list(run_payload.get("debug_messages", []) or []))
    out = dict(run_payload or {})
    out["debug_messages"] = list(debug_messages)
    return out


def run_redocking_single_system_runtime(
    *,
    system_label,
    receptor_obj,
    reference_obj=None,
    ref_fields=None,
    ref_mol=None,
    center=None,
    size=None,
    params=None,
    source_receptor_obj=None,
    flex_enabled=False,
    flex_residues=None,
    auto_use_cached_hetatm=False,
    ligand_override_file="",
    tool_path_fn=None,
    pick_ligand_input_fn=None,
    workspace_dir_fn=None,
    safe_name_fn=None,
    file_exists_fn=None,
    makedirs_fn=None,
    prepare_receptor_fn=None,
    strip_reference_residue_fn=None,
    prepare_flexible_receptor_fn=None,
    native_flexres_supported_fn=None,
    build_native_flexres_spec_fn=None,
    should_preserve_charges_fn=None,
    current_charge_mode_fn=None,
    parameterized_ligand_mol2="",
    samefile_or_equal_fn=None,
    chemically_equivalent_fn=None,
    pubchem_smiles_hint_fn=None,
    rcsb_smiles_by_het_fn=None,
    prepare_ligand_fn=None,
    has_rdkit=False,
    convert_pdbqt_to_sdf_fn=None,
    convert_pdbqt_to_pdb_fn=None,
    load_pose_molecules_fn=None,
    best_rmsd_fn=None,
    load_pose_object_fn=None,
    run_command_fn=None,
    parse_pose_scores_fn=None,
    repeat_statistics_fn=None,
    timeout=7200,
    flow_label="redocking",
    pose_prefix="redock",
    repeat_seed=12345,
    repeat_bootstrap_n=2000,
    repeat_csv_filename="redocking_repeat_stats.csv",
    append_debug_log_fn=None,
    orchestrator_fn=None,
):
    _ = reference_obj
    orchestrate = orchestrator_fn if callable(orchestrator_fn) else run_redocking_single_system_orchestrator
    smina_bin = ""
    if callable(tool_path_fn):
        try:
            smina_bin = str(tool_path_fn("smina") or "")
        except Exception:
            smina_bin = ""
    picked_ligand_file = ""
    if callable(pick_ligand_input_fn):
        try:
            picked_ligand_file = str(pick_ligand_input_fn() or "")
        except Exception:
            picked_ligand_file = ""
    workspace_dir = ""
    if callable(workspace_dir_fn):
        try:
            workspace_dir = str(workspace_dir_fn() or "")
        except Exception:
            workspace_dir = ""
    src_receptor = str(source_receptor_obj or receptor_obj or "")
    run_payload = dict(
        orchestrate(
            system_label=system_label,
            receptor_obj=receptor_obj,
            reference_obj=reference_obj,
            ref_fields=ref_fields,
            ref_mol=ref_mol,
            center=center,
            size=size,
            params=params,
            source_receptor_obj=src_receptor,
            flex_enabled=bool(flex_enabled),
            flex_residues=flex_residues,
            auto_use_cached_hetatm=bool(auto_use_cached_hetatm),
            ligand_override_file=ligand_override_file,
            smina_bin=smina_bin,
            picked_ligand_file=picked_ligand_file,
            workspace_dir=workspace_dir,
            safe_name_fn=safe_name_fn,
            file_exists_fn=file_exists_fn,
            makedirs_fn=makedirs_fn,
            prepare_receptor_fn=prepare_receptor_fn,
            strip_reference_residue_fn=strip_reference_residue_fn,
            prepare_flexible_receptor_fn=prepare_flexible_receptor_fn,
            native_flexres_supported_fn=native_flexres_supported_fn,
            build_native_flexres_spec_fn=build_native_flexres_spec_fn,
            should_preserve_charges_fn=should_preserve_charges_fn,
            current_charge_mode_fn=current_charge_mode_fn,
            parameterized_ligand_mol2=str(parameterized_ligand_mol2 or ""),
            samefile_or_equal_fn=samefile_or_equal_fn,
            chemically_equivalent_fn=chemically_equivalent_fn,
            pubchem_smiles_hint_fn=pubchem_smiles_hint_fn,
            rcsb_smiles_by_het_fn=rcsb_smiles_by_het_fn,
            prepare_ligand_fn=prepare_ligand_fn,
            has_rdkit=bool(has_rdkit),
            convert_pdbqt_to_sdf_fn=convert_pdbqt_to_sdf_fn,
            convert_pdbqt_to_pdb_fn=convert_pdbqt_to_pdb_fn,
            load_pose_molecules_fn=load_pose_molecules_fn,
            best_rmsd_fn=best_rmsd_fn,
            load_pose_object_fn=load_pose_object_fn,
            run_command=run_command_fn,
            parse_pose_scores_fn=parse_pose_scores_fn,
            repeat_statistics_fn=repeat_statistics_fn,
            timeout=int(timeout),
            flow_label=str(flow_label or "redocking"),
            pose_prefix=str(pose_prefix or "redock"),
            repeat_seed=int(repeat_seed),
            repeat_bootstrap_n=int(repeat_bootstrap_n),
            repeat_csv_filename=str(repeat_csv_filename or "redocking_repeat_stats.csv"),
        )
        or {}
    )
    for msg in list(run_payload.get("debug_messages", []) or []):
        if str(msg or "").strip() and callable(append_debug_log_fn):
            try:
                append_debug_log_fn(str(msg))
            except Exception:
                pass
    if not bool(run_payload.get("ok", False)):
        return dict(run_payload or {})
    return dict(run_payload.get("run_payload", {}) or {})
