import datetime
import os
from math import isfinite


def _coerce_finite_triplet(value, default=None):
    """Return a 3-float list only when all coordinates are finite."""
    try:
        vals = [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        if default is None:
            return None
        try:
            vals = [float(default[0]), float(default[1]), float(default[2])]
        except Exception:
            return None
    if any((not isfinite(v)) for v in vals):
        if default is None:
            return None
        try:
            vals = [float(default[0]), float(default[1]), float(default[2])]
        except Exception:
            return None
        if any((not isfinite(v)) for v in vals):
            return None
    return vals


def _clip_float(value, default, lower, upper):
    try:
        val = float(value)
    except Exception:
        val = float(default)
    if not isfinite(val):
        val = float(default)
    return float(min(max(val, float(lower)), float(upper)))


def _clip_int(value, default, lower, upper):
    try:
        val = int(value)
    except Exception:
        val = int(default)
    return int(min(max(val, int(lower)), int(upper)))


def normalize_virtual_screening_grid(
    center,
    size,
    *,
    padding=0.0,
    min_edge=6.0,
    max_edge=120.0,
    fallback_center=(0.0, 0.0, 0.0),
    fallback_size=(18.0, 18.0, 18.0),
):
    """
    Normalize a virtual-screening smina box before docking.

    This is intentionally GUI-free so the docking grid policy can be unit
    tested.  fpocket occasionally produces narrow pocket atom clouds; padding
    and min_edge keep the resulting box physically usable while preserving the
    pocket center.
    """
    warnings = []
    center_out = _coerce_finite_triplet(center, default=fallback_center)
    if center_out is None:
        return {"ok": False, "error": "Grid center is incomplete.", "warnings": ["bad_center"]}
    if _coerce_finite_triplet(center, default=None) is None:
        warnings.append("center_fallback")

    size_in = _coerce_finite_triplet(size, default=fallback_size)
    if size_in is None:
        return {"ok": False, "error": "Grid size is incomplete.", "warnings": warnings + ["bad_size"]}
    if _coerce_finite_triplet(size, default=None) is None:
        warnings.append("size_fallback")

    pad = _clip_float(padding, 0.0, 0.0, 20.0)
    min_e = _clip_float(min_edge, 6.0, 1.0, 60.0)
    max_e = _clip_float(max_edge, 120.0, min_e, 240.0)
    size_out = []
    for raw in size_in:
        edge = max(float(min_e), float(raw) + (2.0 * float(pad)))
        size_out.append(float(min(float(max_e), edge)))
    return {
        "ok": True,
        "center": tuple(float(v) for v in center_out),
        "size": tuple(float(v) for v in size_out),
        "padding": float(pad),
        "min_edge": float(min_e),
        "max_edge": float(max_e),
        "warnings": warnings,
    }


def normalize_virtual_screening_controls(raw_controls=None, pocket=None):
    """
    Validate and clip the run controls used by the Virtual Screening dialog.

    The UI supplies already-visible center/size controls, so padding is not
    applied here.  Padding is only applied when syncing a grid from a pocket.
    """
    raw = dict(raw_controls or {})
    pkt = dict(pocket or {}) if isinstance(pocket, dict) else {}
    fallback_center = pkt.get("center", (0.0, 0.0, 0.0))
    fallback_size = pkt.get("size", (18.0, 18.0, 18.0))
    min_edge = _clip_float(raw.get("grid_min_edge", 6.0), 6.0, 1.0, 60.0)
    grid = normalize_virtual_screening_grid(
        raw.get("center", fallback_center),
        raw.get("size", fallback_size),
        padding=0.0,
        min_edge=min_edge,
        max_edge=120.0,
        fallback_center=fallback_center,
        fallback_size=fallback_size,
    )
    if not bool(grid.get("ok", False)):
        return {
            "ok": False,
            "error": str(grid.get("error", "Invalid grid controls.")),
            "warnings": list(grid.get("warnings", []) or []),
        }
    return {
        "ok": True,
        "target_ph": _clip_float(raw.get("target_ph", 7.4), 7.4, 0.0, 14.0),
        "max_ligs": _clip_int(raw.get("max_ligs", 250), 250, 1, 200000),
        "center": tuple(grid.get("center", (0.0, 0.0, 0.0))),
        "size": tuple(grid.get("size", (18.0, 18.0, 18.0))),
        "exhaustiveness": _clip_int(raw.get("exhaustiveness", 8), 8, 1, 256),
        "timeout_lig": _clip_int(raw.get("timeout_lig", 300), 300, 30, 7200),
        "grid_padding": _clip_float(raw.get("grid_padding", 0.0), 0.0, 0.0, 20.0),
        "grid_min_edge": float(min_edge),
        "warnings": list(grid.get("warnings", []) or []),
    }


def format_virtual_screening_library_status(library_path="", info=None, preview=False):
    """Build a compact, stable label for SDF library parsing results."""
    data = dict(info or {})
    name = os.path.basename(str(library_path or "").strip()) if str(library_path or "").strip() else "not selected"
    try:
        loaded = int(data.get("loaded", 0) or 0)
    except Exception:
        loaded = 0
    try:
        skipped = int(data.get("skipped", 0) or 0)
    except Exception:
        skipped = 0
    try:
        total = int(data.get("total", 0) or 0)
    except Exception:
        total = 0
    reason = str(data.get("reason", "not_loaded") or "not_loaded")
    prefix = "Library preview" if bool(preview) else "Library"
    if loaded <= 0:
        return f"{prefix}: {name} | no valid ligands ({reason})"
    extra = ""
    try:
        limit = int(data.get("limit", 0) or 0)
        if limit > 0:
            extra = f" | limit={limit}"
    except Exception:
        extra = ""
    if bool(data.get("truncated", False)):
        extra += " | more ligands not parsed"
    return f"{prefix}: {name} | valid={loaded} | skipped={skipped} | read={total}{extra}"


def _finite_affinity_value(pose):
    try:
        affinity = float((pose or {}).get("affinity", float("nan")))
    except Exception:
        return float("nan")
    return float(affinity) if isfinite(affinity) else float("nan")


def _pick_best_affinity_pose(poses):
    best_pose = None
    best_affinity = float("inf")
    for pose in list(poses or []):
        affinity = _finite_affinity_value(pose)
        if (not isfinite(affinity)) or affinity >= float(best_affinity):
            continue
        best_pose = pose
        best_affinity = float(affinity)
    if best_pose is None:
        return None, float("nan")
    return best_pose, float(best_affinity)


def process_virtual_screening_entry(
    item,
    idx,
    *,
    smina_bin,
    receptor_input,
    center,
    size,
    exhaustiveness,
    timeout_lig,
    lig_dir,
    pose_dir,
    target_ph,
    safe_name_fn,
    write_sdf,
    prepare_with_meeko,
    fallback_prepare_ligand,
    run_command,
    parse_pose_scores,
    convert_pose_for_view,
):
    name = str((item or {}).get("name", f"Lig_{int(idx)+1}"))
    ident = str((item or {}).get("identifier", ""))
    safe = str(safe_name_fn(name) if callable(safe_name_fn) else name).strip()[:48]
    stem = f"{int(idx+1):05d}_{safe}" if safe else f"{int(idx+1):05d}_lig"
    lig_sdf = os.path.join(str(lig_dir), f"{stem}.sdf")
    out_pose = os.path.join(str(pose_dir), f"{stem}_pose.pdbqt")
    out_log = os.path.join(str(pose_dir), f"{stem}.log")

    prep_mode = "meeko"
    prep_note = ""
    ligand_input = ""
    status = "ok"
    err_tail = ""
    affinity = float("nan")
    pose_view = ""

    ok_w, msg_w = write_sdf((item or {}).get("mol"), lig_sdf, name)
    if not bool(ok_w):
        status = "failed"
        prep_note = f"SDF write failed: {msg_w}"

    if status == "ok":
        meeko_out = os.path.join(str(lig_dir), f"{stem}.pdbqt")
        ok_meeko, msg_meeko = prepare_with_meeko((item or {}).get("mol"), meeko_out)
        if ok_meeko:
            ligand_input = str(meeko_out)
            prep_mode = "meeko"
            prep_note = str(msg_meeko or "")
        else:
            prep_mode = "fallback"
            prep_note = str(msg_meeko or "")
            fallback_job = os.path.join(str(lig_dir), f"job_{stem}")
            fallback_in = ""
            lig_info = {}
            try:
                fallback_in, lig_info = fallback_prepare_ligand(lig_sdf, fallback_job, item, float(target_ph))
            except TypeError:
                fallback_in, lig_info = fallback_prepare_ligand(lig_sdf, fallback_job, float(target_ph))
            if fallback_in and os.path.isfile(str(fallback_in)):
                ligand_input = str(fallback_in)
                try:
                    extra = lig_info.get("notes") if isinstance(lig_info, dict) else lig_info
                except Exception:
                    extra = ""
                prep_note = f"{prep_note}; fallback={extra}"
            else:
                status = "failed"
                prep_note = f"{prep_note}; fallback ligand prep failed"

    if status == "ok":
        cmdline = [
            str(smina_bin),
            "-r",
            str(receptor_input),
            "-l",
            str(ligand_input),
            "--center_x",
            f"{float(center[0]):.4f}",
            "--center_y",
            f"{float(center[1]):.4f}",
            "--center_z",
            f"{float(center[2]):.4f}",
            "--size_x",
            f"{float(size[0]):.4f}",
            "--size_y",
            f"{float(size[1]):.4f}",
            "--size_z",
            f"{float(size[2]):.4f}",
            "--exhaustiveness",
            str(int(exhaustiveness)),
            "--num_modes",
            "1",
            "--energy_range",
            "4.0",
            "--min_rmsd_filter",
            "1.0",
            "--out",
            str(out_pose),
            "--log",
            str(out_log),
        ]
        ok, out, err, rc = run_command(cmdline, cwd=os.path.dirname(str(pose_dir)), timeout=max(30, int(timeout_lig)))
        err_tail = str((err or "")[-1200:])
        if (not bool(ok)) or (not os.path.isfile(out_pose)):
            status = "failed"
            prep_note = f"{prep_note}; smina rc={int(rc)}"
        else:
            poses = list(parse_pose_scores(out_pose, out_log) or [])
            if poses:
                _best_pose, affinity = _pick_best_affinity_pose(poses)
            atom_rows = 0
            try:
                with open(out_pose, "r", encoding="utf-8", errors="ignore") as fh_pose:
                    for ln in fh_pose:
                        if ln.startswith("ATOM") or ln.startswith("HETATM"):
                            atom_rows += 1
            except Exception:
                atom_rows = 0
            if (not poses) or (not isfinite(float(affinity))) or int(atom_rows) <= 0:
                status = "failed"
                prep_note = f"{prep_note}; no_scored_pose_parsed"
            else:
                pose_view = str(convert_pose_for_view(out_pose, pose_dir, stem) or out_pose)

    return {
        "index": int((item or {}).get("index", int(idx) + 1)),
        "name": str(name),
        "identifier": str(ident),
        "affinity": float(affinity) if isfinite(float(affinity)) else float("nan"),
        "prep": str(prep_mode),
        "status": str(status),
        "notes": str(prep_note),
        "pose_file": str(out_pose),
        "pose_view_file": str(pose_view),
        "stderr_tail": str(err_tail),
        "mol": (item or {}).get("mol"),
    }


def run_virtual_screening_entries(
    entries,
    *,
    smina_bin,
    receptor_input,
    center,
    size,
    exhaustiveness,
    timeout_lig,
    lig_dir,
    pose_dir,
    target_ph,
    safe_name_fn,
    write_sdf,
    prepare_with_meeko,
    fallback_prepare_ligand,
    run_command,
    parse_pose_scores,
    convert_pose_for_view,
    on_progress=None,
):
    entries_list = list(entries or [])
    controls = normalize_virtual_screening_controls(
        {
            "center": center,
            "size": size,
            "exhaustiveness": exhaustiveness,
            "timeout_lig": timeout_lig,
            "target_ph": target_ph,
        }
    )
    if bool(controls.get("ok", False)):
        center = tuple(controls.get("center", center))
        size = tuple(controls.get("size", size))
        exhaustiveness = int(controls.get("exhaustiveness", exhaustiveness))
        timeout_lig = int(controls.get("timeout_lig", timeout_lig))
        target_ph = float(controls.get("target_ph", target_ph))
    results = []
    cancelled = False
    total = int(len(entries_list))
    for i, item in enumerate(entries_list):
        if callable(on_progress):
            try:
                keep_running = bool(on_progress(int(i), int(total), item))
            except Exception:
                keep_running = True
            if not keep_running:
                cancelled = True
                break
        rec = process_virtual_screening_entry(
            item,
            i,
            smina_bin=smina_bin,
            receptor_input=receptor_input,
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
            timeout_lig=timeout_lig,
            lig_dir=lig_dir,
            pose_dir=pose_dir,
            target_ph=target_ph,
            safe_name_fn=safe_name_fn,
            write_sdf=write_sdf,
            prepare_with_meeko=prepare_with_meeko,
            fallback_prepare_ligand=fallback_prepare_ligand,
            run_command=run_command,
            parse_pose_scores=parse_pose_scores,
            convert_pose_for_view=convert_pose_for_view,
        )
        results.append(rec)
    return results, bool(cancelled)


def vs_prepare_job_paths(workspace_dir, receptor_obj, safe_name_fn=None, tag=None, makedirs_fn=None):
    safe_fn = safe_name_fn if callable(safe_name_fn) else (lambda x: str(x))
    mk_fn = makedirs_fn if callable(makedirs_fn) else os.makedirs
    run_tag = str(tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    job_dir = os.path.join(str(workspace_dir), f"virtual_screen_{safe_fn(receptor_obj)}_{run_tag}")
    lig_dir = os.path.join(job_dir, "ligands")
    pose_dir = os.path.join(job_dir, "poses")
    mk_fn(job_dir, exist_ok=True)
    mk_fn(lig_dir, exist_ok=True)
    mk_fn(pose_dir, exist_ok=True)
    return {
        "tag": str(run_tag),
        "job_dir": str(job_dir),
        "lig_dir": str(lig_dir),
        "pose_dir": str(pose_dir),
    }


def _vs_sort_key(rec):
    try:
        ok = str((rec or {}).get("status", "")).lower() == "ok"
        affinity = float((rec or {}).get("affinity", float("nan")))
        good = ok and isfinite(affinity)
        return (0 if good else 1, float(affinity) if good else float("inf"), int((rec or {}).get("index", 0) or 0))
    except Exception:
        return (2, float("inf"), 0)


def rank_virtual_screening_results(results):
    arr = sorted(list(results or []), key=_vs_sort_key)
    rank = 1
    for rec in arr:
        try:
            ok = str((rec or {}).get("status", "")).lower() == "ok"
            affinity = float((rec or {}).get("affinity", float("nan")))
            if ok and isfinite(affinity):
                rec["rank"] = int(rank)
                rank += 1
            else:
                rec["rank"] = None
        except Exception:
            rec["rank"] = None
    return arr


def summarize_virtual_screening_results(results):
    arr = list(results or [])
    n_total = int(len(arr))
    n_ok = 0
    best = None
    for rec in arr:
        try:
            ok = str((rec or {}).get("status", "")).lower() == "ok"
            affinity = float((rec or {}).get("affinity", float("nan")))
            if ok and isfinite(affinity):
                n_ok += 1
                if best is None:
                    best = rec
        except Exception:
            continue
    return {
        "n_total": int(n_total),
        "n_ok": int(n_ok),
        "n_fail": int(max(0, n_total - n_ok)),
        "best": best,
    }


def vs_status_text(best, n_ok, n_total):
    try:
        if isinstance(best, dict):
            return (
                f"Status: done ({int(n_ok)}/{int(n_total)} scored). Best: "
                f"{best.get('name')} = {float(best.get('affinity', float('nan'))):.3f} kcal/mol"
            )
    except Exception:
        pass
    return f"Status: done ({int(n_ok)}/{int(n_total)} scored)."


def build_virtual_screening_summary_lines(
    *,
    receptor_obj,
    library_path,
    loaded_count,
    skipped_count,
    n_ok,
    n_fail,
    output_dir,
    best=None,
    cancelled=False,
):
    lines = [
        "Virtual screening completed.",
        f"Receptor: {str(receptor_obj)}",
        f"Library: {str(library_path)}",
        f"Loaded ligands: {int(loaded_count)} (skipped: {int(skipped_count)})",
        f"Scored: {int(n_ok)}",
        f"Failed: {int(n_fail)}",
        f"Output: {str(output_dir)}",
    ]
    if isinstance(best, dict):
        try:
            lines.append(
                f"Best hit: {best.get('name')} | affinity={float(best.get('affinity', float('nan'))):.3f} kcal/mol"
            )
        except Exception:
            lines.append(f"Best hit: {best.get('name')}")
    if bool(cancelled):
        lines.append("Run was cancelled before processing all ligands.")
    return lines
