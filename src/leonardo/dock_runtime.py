import glob
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict


def _runtime_float(value, default=None):
    try:
        val = float(value)
    except Exception:
        return default
    return val if val == val and val not in (float("inf"), float("-inf")) else default


def _first_existing(candidates):
    for cand in candidates:
        if not cand:
            continue
        if os.path.isabs(cand):
            try:
                if os.path.isfile(cand) and (os.access(cand, os.X_OK) or str(cand).lower().endswith(".py")):
                    return cand
            except Exception:
                continue
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


def resolve_tool_path(name, sys_prefix=None, sys_executable=None):
    target = str(name or "").strip()
    if not target:
        return None

    prefix = str(sys_prefix or sys.prefix)
    executable = str(sys_executable or sys.executable)
    conda_prefix = str(os.environ.get("CONDA_PREFIX", "") or "").strip()

    conda_env_roots = (
        "/opt/anaconda3/envs",
        os.path.expanduser("~/miniconda3/envs"),
        os.path.expanduser("~/anaconda3/envs"),
    )
    preferred_env_bins = []
    for root in conda_env_roots:
        preferred_env_bins.extend(
            [
                os.path.join(root, "protliginteract-full", "bin"),
                os.path.join(root, "protliginteract-optional", "bin"),
            ]
        )
    conda_env_bins = []
    for root in conda_env_roots:
        try:
            conda_env_bins.extend(glob.glob(os.path.join(root, "protliginteract*", "bin")))
            conda_env_bins.extend(glob.glob(os.path.join(root, "*", "bin")))
        except Exception:
            pass
    active_protlig_bins = []
    for env_root in (conda_prefix, prefix):
        env_root = str(env_root or "").strip()
        if env_root and os.path.basename(env_root).startswith("protliginteract"):
            active_protlig_bins.append(os.path.join(env_root, "bin"))
    common_bins = (
        active_protlig_bins
        + preferred_env_bins
        + [
            os.path.join(conda_prefix, "bin") if conda_prefix else "",
            os.path.join(prefix, "bin"),
            os.path.dirname(executable),
        ]
        + conda_env_bins
        + [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/opt/anaconda3/bin",
        ]
    )
    deduped_bins = []
    seen_bins = set()
    for b in common_bins:
        b = str(b or "").strip()
        if not b or b in seen_bins:
            continue
        seen_bins.add(b)
        deduped_bins.append(b)
    common_bins = deduped_bins
    common_roots = [conda_prefix, prefix] + [os.path.dirname(b) for b in common_bins] + [
        "/opt/anaconda3",
        "/usr/local",
        "/opt/homebrew",
    ]
    common_roots = [r for r in common_roots if r]

    adt_dirs = []
    for root in common_roots:
        adt_dirs.extend(
            [
                os.path.join(root, "MGLToolsPckgs", "AutoDockTools", "Utilities24"),
                os.path.join(root, "share", "mgltools", "AutoDockTools", "Utilities24"),
            ]
        )
        try:
            adt_dirs.extend(
                glob.glob(
                    os.path.join(root, "lib", "python*", "site-packages", "AutoDockTools", "Utilities24")
                )
            )
        except Exception:
            pass
    adt_dirs = [d for d in adt_dirs if d]

    if target in ("obabel", "openbabel"):
        cands = [os.path.join(b, "obabel") for b in common_bins] + [
            os.path.join(b, "babel") for b in common_bins
        ] + ["obabel", "babel"]
        return _first_existing(cands)

    if target in ("prepare_receptor4.py", "prepare_receptor4"):
        cands = [os.path.join(b, "prepare_receptor4.py") for b in common_bins] + [
            os.path.join(b, "prepare_receptor4") for b in common_bins
        ] + [
            os.path.join(d, "prepare_receptor4.py") for d in adt_dirs
        ] + ["prepare_receptor4.py", "prepare_receptor4"]
        return _first_existing(cands)

    if target in ("prepare_ligand4.py", "prepare_ligand4"):
        cands = [os.path.join(b, "prepare_ligand4.py") for b in common_bins] + [
            os.path.join(b, "prepare_ligand4") for b in common_bins
        ] + [
            os.path.join(d, "prepare_ligand4.py") for d in adt_dirs
        ] + ["prepare_ligand4.py", "prepare_ligand4"]
        return _first_existing(cands)

    if target in ("prepare_flexreceptor4.py", "prepare_flexreceptor4"):
        cands = [os.path.join(b, "prepare_flexreceptor4.py") for b in common_bins] + [
            os.path.join(b, "prepare_flexreceptor4") for b in common_bins
        ] + [
            os.path.join(d, "prepare_flexreceptor4.py") for d in adt_dirs
        ] + ["prepare_flexreceptor4.py", "prepare_flexreceptor4"]
        return _first_existing(cands)

    if target in ("mk_prepare_receptor.py", "mk_prepare_receptor"):
        cands = [os.path.join(b, "mk_prepare_receptor.py") for b in common_bins] + [
            os.path.join(b, "mk_prepare_receptor") for b in common_bins
        ] + ["mk_prepare_receptor.py", "mk_prepare_receptor"]
        return _first_existing(cands)

    if target in ("smina", "reduce", "antechamber", "parmchk2", "fpocket", "tleap", "psi4"):
        cands = [os.path.join(b, target) for b in common_bins] + [target]
        return _first_existing(cands)

    return shutil.which(target)


def run_command(argv, cwd=None, timeout=1800):
    try:
        proc = subprocess.run(
            list(argv),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=int(timeout),
        )
        ok = int(proc.returncode) == 0
        return ok, proc.stdout or "", proc.stderr or "", int(proc.returncode)
    except Exception as exc:
        return False, "", str(exc), -1


def build_legacy_python_tool_command(
    tool_path,
    args=None,
    *,
    which_fn=None,
    isfile_fn=None,
    access_fn=None,
    open_fn=None,
    sys_executable=None,
):
    tool = str(tool_path or "").strip()
    argv = [str(x) for x in list(args or [])]
    if not tool:
        return list(argv)
    if not str(tool).lower().endswith(".py"):
        return [str(tool)] + list(argv)

    which = which_fn if callable(which_fn) else shutil.which
    isfile = isfile_fn if callable(isfile_fn) else os.path.isfile
    access = access_fn if callable(access_fn) else os.access
    opener = open_fn if callable(open_fn) else open
    py_exec = str(sys_executable or sys.executable or "").strip()

    shebang = ""
    if bool(isfile(tool)):
        try:
            with opener(tool, "r", encoding="utf-8", errors="ignore") as fh:
                shebang = str(fh.readline() or "").strip().lower()
        except Exception:
            shebang = ""

    if shebang.startswith("#!"):
        try:
            if bool(access(tool, os.X_OK)):
                return [str(tool)] + list(argv)
        except Exception:
            pass
        if "pythonsh" in shebang:
            interp = which("pythonsh") or which("python2")
            if interp:
                return [str(interp), str(tool)] + list(argv)
        if "python2" in shebang:
            interp = which("python2") or which("pythonsh")
            if interp:
                return [str(interp), str(tool)] + list(argv)

    if py_exec:
        return [str(py_exec), str(tool)] + list(argv)
    return [str(tool)] + list(argv)


def build_meeko_flexres_spec(residues):
    rows = [dict(r or {}) for r in list(residues or []) if isinstance(r, dict)]
    if not rows:
        return ""

    def _tokenize(chain, resi):
        ch = str(chain or "").strip()
        rs = str(resi or "").strip()
        if not rs:
            return ""
        rs = "".join(part.strip() for part in rs.split(":"))
        if not rs:
            return ""
        return f"{ch}:{rs}" if ch else f":{rs}"

    parts = []
    seen = set()
    for row in rows:
        tok = _tokenize((row or {}).get("chain", ""), (row or {}).get("resi", ""))
        if (not tok) or (tok in seen):
            continue
        seen.add(tok)
        parts.append(tok)
    return ",".join(parts)


def build_meeko_receptor_input_args(structure_path):
    path = str(structure_path or "").strip()
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pdb", ".ent"):
        return ["--read_pdb", str(path)]
    if ext == ".pqr":
        return ["--read_pqr", str(path)]
    return ["-i", str(path)]


def build_smina_flexres_spec(residues):
    return str(build_meeko_flexres_spec(residues) or "")


def smina_supports_native_flexres(smina_bin, run_command=None, timeout=60):
    tool = str(smina_bin or "").strip()
    if not tool:
        return False
    runner = run_command if callable(run_command) else run_command_default
    try:
        ok, out, err, _rc = runner([tool, "--help"], timeout=int(timeout))
    except Exception:
        return False
    text = "\n".join([str(out or ""), str(err or "")]).lower()
    if "--flexres" in text:
        return True
    return bool(ok and "--flexdist" in text)


def run_command_default(argv, cwd=None, timeout=1800):
    return run_command(argv, cwd=cwd, timeout=timeout)


def parse_fpocket_info_text(info_text):
    """
    Parse fpocket `*_info.txt` content into pocket dictionaries.

    fpocket has changed small formatting details across versions, so this parser
    accepts `key: value`, `key = value`, and whitespace-separated key/value
    lines inside each `Pocket N` block. Numeric values are normalized to floats
    while non-numeric values are preserved for diagnostics.
    """
    pockets = []
    current = None
    key_value_pat = re.compile(r"^([^:=]+?)\s*(?::|=|\t+|\s{2,})\s*(.+)$")
    for raw in str(info_text or "").splitlines():
        line = str(raw or "").strip()
        if not line:
            continue
        m = re.match(r"^Pocket\s+(\d+)\b", line, flags=re.IGNORECASE)
        if m:
            if current is not None:
                pockets.append(current)
            current = {"id": int(m.group(1))}
            continue
        if current is None:
            continue
        km = key_value_pat.match(line)
        if not km:
            continue
        key = re.sub(r"[^a-z0-9]+", "_", km.group(1).strip().lower()).strip("_")
        if not key:
            continue
        val = km.group(2).strip()
        mv = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
        if mv:
            parsed = _runtime_float(mv.group(0), None)
            current[key] = parsed if parsed is not None else val
        else:
            current[key] = val
    if current is not None:
        pockets.append(current)
    return pockets


def parse_pdb_atom_coordinates_from_lines(lines):
    """
    Extract Cartesian coordinates from PDB ATOM/HETATM records.

    Fixed-column parsing is attempted first, then a split-field fallback handles
    relaxed PDB-like files written by some pocket tools.
    """
    coords = []
    for raw in list(lines or []):
        line = str(raw or "")
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        xyz = None
        try:
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except Exception:
            parts = line.split()
            if len(parts) >= 9:
                try:
                    xyz = [float(parts[6]), float(parts[7]), float(parts[8])]
                except Exception:
                    xyz = None
        if xyz is not None:
            coords.append(xyz)
    return coords


def pocket_geometry_from_coordinates(coords):
    """Return center, extent, and atom count for a pocket atom cloud."""
    rows = []
    for xyz in list(coords or []):
        try:
            row = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        except Exception:
            continue
        if all(v == v and v not in (float("inf"), float("-inf")) for v in row):
            rows.append(row)
    if not rows:
        return None, None, 0
    n = float(len(rows))
    center = [sum(row[i] for row in rows) / n for i in range(3)]
    mins = [min(row[i] for row in rows) for i in range(3)]
    maxs = [max(row[i] for row in rows) for i in range(3)]
    size = [maxs[i] - mins[i] for i in range(3)]
    return center, size, int(len(rows))


def normalize_docking_grid(center, size, padding=0.0, min_edge=1.0, max_edge=120.0):
    """
    Normalize a pocket-derived docking grid.

    fpocket pocket atom clouds can be very thin in one dimension. smina needs a
    physically usable search box, so the UI applies user-visible padding and a
    minimum edge while keeping the pocket center unchanged.
    """
    try:
        c = [float(center[0]), float(center[1]), float(center[2])]
        s0 = [float(size[0]), float(size[1]), float(size[2])]
    except Exception:
        return {"ok": False, "error": "Pocket grid geometry is incomplete."}
    vals = c + s0
    if any((v != v) or v in (float("inf"), float("-inf")) for v in vals):
        return {"ok": False, "error": "Pocket grid geometry contains non-finite values."}
    try:
        pad = max(0.0, float(padding))
    except Exception:
        pad = 0.0
    try:
        min_e = max(1.0, float(min_edge))
    except Exception:
        min_e = 1.0
    try:
        max_e = max(min_e, float(max_edge))
    except Exception:
        max_e = max(120.0, min_e)
    size_out = []
    for raw in s0:
        edge = max(min_e, float(raw) + (2.0 * pad))
        edge = min(max_e, edge)
        size_out.append(float(edge))
    return {"ok": True, "center": c, "size": size_out, "padding": float(pad), "min_edge": float(min_e)}


def format_docking_grid_summary(center, size, source="", pocket_id=None):
    """Build a compact status string for the active docking grid."""
    try:
        c = [float(center[0]), float(center[1]), float(center[2])]
        s = [float(size[0]), float(size[1]), float(size[2])]
    except Exception:
        return "Grid: not defined"
    src = str(source or "manual").strip()
    pid_txt = ""
    try:
        if pocket_id is not None and int(pocket_id) > 0:
            pid_txt = f" | pocket={int(pocket_id)}"
    except Exception:
        pid_txt = ""
    return (
        f"Grid: center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) A | "
        f"size=({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}) A | source={src}{pid_txt}"
    )


def run_post_docking_pipeline(
    results,
    context=None,
    rank_fn=None,
    transform_fn=None,
    summary_fn=None,
    apply_state_fn=None,
    log_fn=None,
    status_fn=None,
    dialog_fn=None,
):
    """Run an ordered callback pipeline over post-docking state payloads."""
    state_results = list(results or [])
    state_context = dict(context or {})
    state_summary = {}

    def _apply_payload(payload):
        nonlocal state_results, state_context, state_summary
        if not isinstance(payload, dict):
            return
        if "results" in payload:
            state_results = list(payload.get("results", []) or [])
        if "context" in payload:
            state_context.update(dict(payload.get("context", {}) or {}))
        if "summary" in payload:
            state_summary.update(dict(payload.get("summary", {}) or {}))

    for fn in (
        rank_fn,
        transform_fn,
        summary_fn,
        apply_state_fn,
        log_fn,
        status_fn,
        dialog_fn,
    ):
        if callable(fn):
            _apply_payload(fn(state_results, state_context, state_summary))

    return {
        "results": list(state_results),
        "context": dict(state_context),
        "summary": dict(state_summary),
    }
