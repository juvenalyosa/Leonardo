import os
import re
from math import isfinite


def _finite_affinity_value(pose):
    try:
        affinity = float((pose or {}).get("affinity", float("nan")))
    except Exception:
        return float("nan")
    return float(affinity) if isfinite(affinity) else float("nan")


def _finite_nonnegative_rmsd(value):
    try:
        rmsd = float(value)
    except Exception:
        return float("nan")
    if (not isfinite(rmsd)) or rmsd < 0.0:
        return float("nan")
    return float(rmsd)


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


def _parse_pose_rows_from_text(text):
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
                "rmsd_lb": _finite_nonnegative_rmsd(m.group(3)),
                "rmsd_ub": _finite_nonnegative_rmsd(m.group(4)),
            }
        )
    return rows


def _merge_pose_rows_with_log_table(poses, log_rows):
    """
    Merge pose remarks with the smina score table.

    smina PDBQT/SDF pose remarks often contain `minimizedRMSD -1` when no
    reference ligand is available.  That is a sentinel, not a physical RMSD.
    The log score table is the authoritative source for the displayed lower
    and upper RMSD bounds when it exists.
    """
    pose_rows = [dict(p or {}) for p in list(poses or [])]
    table_rows = [dict(r or {}) for r in list(log_rows or [])]
    if not pose_rows:
        return table_rows
    by_mode = {}
    for row in table_rows:
        try:
            mode = int(row.get("mode", 0) or 0)
        except Exception:
            continue
        if mode > 0 and mode not in by_mode:
            by_mode[mode] = row
    for idx, pose in enumerate(pose_rows):
        try:
            mode = int(pose.get("mode", idx + 1) or (idx + 1))
        except Exception:
            mode = int(idx + 1)
        pose["mode"] = int(mode)
        table = by_mode.get(int(mode), {})
        if table:
            table_aff = _finite_affinity_value(table)
            pose_aff = _finite_affinity_value(pose)
            if isfinite(table_aff) and not isfinite(pose_aff):
                pose["affinity"] = float(table_aff)
        for key in ("rmsd_lb", "rmsd_ub"):
            table_val = _finite_nonnegative_rmsd(table.get(key)) if table else float("nan")
            pose_val = _finite_nonnegative_rmsd(pose.get(key))
            pose[key] = float(table_val if isfinite(table_val) else pose_val)
    return pose_rows


def _collect_smina_output_text(log_file, out="", err=""):
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


def _needs_flex_search_space_retry(text):
    low = str(text or "").lower()
    return (
        "could not find any conformations completely within the search space" in low
        or (
            "search space" in low
            and "flexible side chains" in low
        )
    )


def _estimate_flex_autobox_min(cmdline, default_min=10.0, cap=26.0):
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
    return float(min(float(cap), float(target)))


def _expand_flex_size_cmdline(cmdline, delta=6.0, max_edge=60.0):
    """Grow an explicit --size_x/y/z box so flexible side chains fit (Leonardo passes
    center/size, not --autobox_ligand). Returns (cmdline, new_sizes) or (None, None)."""
    out = list(cmdline or [])
    sizes = []
    for key in ("--size_x", "--size_y", "--size_z"):
        if key not in out:
            return None, None
        idx = out.index(key)
        try:
            cur = float(out[idx + 1])
        except (IndexError, ValueError):
            return None, None
        new = min(float(max_edge), cur + float(delta))
        sizes.append((idx, cur, new))
    if all(new <= cur + 1.0e-6 for _idx, cur, new in sizes):
        return None, None
    for idx, _cur, new in sizes:
        out[idx + 1] = f"{new:.4f}"
    return out, [new for _idx, _cur, new in sizes]


def _expand_flex_autobox_cmdline(cmdline, min_autobox_add=10.0, delta=6.0, max_autobox_add=26.0):
    cmd = list(cmdline or [])
    if ("--flex" not in cmd) and ("--flexres" not in cmd):
        return None, None
    if "--autobox_ligand" not in cmd:
        retry, sizes = _expand_flex_size_cmdline(cmd, delta=float(delta))
        if retry is None:
            return None, None
        return retry, "box " + "x".join(f"{v:.1f}" for v in sizes)
    out = list(cmd)
    target_min = _estimate_flex_autobox_min(out, default_min=float(min_autobox_add), cap=float(max_autobox_add))
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


def parse_smina_pose_scores(pose_file, log_file=None):
    poses = []
    mode = None
    try:
        with open(pose_file, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = str(raw).strip()
                if line.startswith("MODEL"):
                    parts = line.split()
                    try:
                        mode = int(parts[-1])
                    except Exception:
                        mode = len(poses) + 1
                    continue
                if line.startswith("REMARK VINA RESULT:"):
                    vals = line.replace("REMARK VINA RESULT:", "").split()
                    if len(vals) >= 3:
                        try:
                            poses.append(
                                {
                                    "mode": int(mode if mode is not None else (len(poses) + 1)),
                                    "affinity": float(vals[0]),
                                    "rmsd_lb": _finite_nonnegative_rmsd(vals[1]),
                                    "rmsd_ub": _finite_nonnegative_rmsd(vals[2]),
                                }
                            )
                        except Exception:
                            continue
                    continue
                if line.startswith("REMARK minimizedAffinity"):
                    vals = line.split()
                    if len(vals) >= 3:
                        try:
                            poses.append(
                                {
                                    "mode": int(mode if mode is not None else (len(poses) + 1)),
                                    "affinity": float(vals[2]),
                                    "rmsd_lb": float("nan"),
                                    "rmsd_ub": float("nan"),
                                }
                            )
                        except Exception:
                            continue
                    continue
                if line.startswith("REMARK minimizedRMSD") and poses:
                    vals = line.split()
                    if len(vals) >= 3:
                        try:
                            rmsd_val = float(vals[2])
                        except Exception:
                            continue
                        clean_rmsd = _finite_nonnegative_rmsd(rmsd_val)
                        poses[-1]["rmsd_lb"] = float(clean_rmsd)
                        poses[-1]["rmsd_ub"] = float(clean_rmsd)
    except Exception:
        poses = []
    log_rows = []
    if log_file and os.path.isfile(log_file):
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as fh:
                log_rows = list(_parse_pose_rows_from_text(fh.read()) or [])
        except Exception:
            log_rows = []
    if poses:
        return _merge_pose_rows_with_log_table(poses, log_rows)
    if not log_rows:
        return []
    return log_rows


def run_smina_repeats(
    base_cmd,
    job_dir,
    n_repeats,
    seed0,
    run_command,
    parse_pose_scores,
    timeout=7200,
):
    """Run repeated smina docking executions and keep the best repeat."""
    if not callable(run_command):
        raise TypeError("run_command must be callable")
    if not callable(parse_pose_scores):
        raise TypeError("parse_pose_scores must be callable")

    repeats = int(max(1, n_repeats))
    base_seed = int(seed0 or 0)
    cmd_base = list(base_cmd or [])
    # smina runs with cwd=rep_dir; relative --out/--log paths would resolve twice
    root_dir = os.path.abspath(str(job_dir or "."))

    repeat_rows = []
    best_repeat = None
    best_poses = []
    best_affinity = float("inf")

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
            combined_text = _collect_smina_output_text(out_log, out=out, err=err)
            if (
                attempt_idx < 4
                and _needs_flex_search_space_retry(combined_text)
            ):
                retry_cmdline, retry_add = _expand_flex_autobox_cmdline(cmdline)
                if retry_cmdline:
                    attempt_idx += 1
                    cmdline = list(retry_cmdline)
                    retry_steps.append(retry_add if isinstance(retry_add, str) else f"autobox_add={float(retry_add):.3f}")
                    continue
            break
        retry_note = ""
        if retry_steps:
            retry_note = f" Auto-retried with {', '.join(retry_steps)}."
        used_seed = re.search(r"Using random seed:\s*(-?\d+)", combined_text or "")
        rec = {
            "repeat": int(rep),
            # the seed smina really used (it draws a random one when none is given)
            "seed": int(used_seed.group(1)) if used_seed else int(seed_val),
            "ok": False,
            "best_affinity": float("nan"),
            "n_poses": 0,
            "pose_file": str(out_pose),
            "log_file": str(out_log),
            "error": "",
        }
        if (not bool(ok)) or (not os.path.isfile(out_pose)):
            rec["error"] = f"smina rc={int(rc)}:{retry_note} {str((err or out) or '')[-1200:]}".strip()
            repeat_rows.append(rec)
            continue

        poses_i = list(parse_pose_scores(out_pose, out_log) or [])
        if not poses_i:
            poses_i = list(_parse_pose_rows_from_text("\n".join([str(out or ""), str(err or "")])) or [])
        if not poses_i:
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
            rec["error"] = ("No poses parsed from docking output." + str(retry_note or "")).strip()
            if log_tail:
                rec["error"] += f" Log tail: {log_tail}"
            if std_tail:
                rec["error"] += f" Output tail: {std_tail}"
            repeat_rows.append(rec)
            continue

        rec["ok"] = True
        rec["n_poses"] = int(len(poses_i))
        rep_best, rep_best_aff = _pick_best_affinity_pose(poses_i)
        if rep_best is None:
            rec["ok"] = False
            rec["error"] = "No finite pose affinity parsed from docking output."
            repeat_rows.append(rec)
            continue
        rec["best_affinity"] = float(rep_best_aff) if isfinite(rep_best_aff) else float("nan")
        repeat_rows.append(rec)
        if isfinite(rep_best_aff) and float(rep_best_aff) < float(best_affinity):
            best_affinity = float(rep_best_aff)
            best_repeat = dict(rec)
            best_poses = list(poses_i)

    return {
        "repeat_rows": repeat_rows,
        "best_repeat": best_repeat,
        "best_poses": best_poses,
    }


def format_recent_errors(repeat_rows, max_rows=3, max_chars=400):
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
            tail += f"\n- repeat {int((rr or {}).get('repeat', 0) or 0)}: {shown}"
    return tail
