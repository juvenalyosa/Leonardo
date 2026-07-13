import csv
import os
import re


def _task_monitor_read_tail(path, max_bytes=65536):
    p = str(path or "").strip()
    if (not p) or (not os.path.isfile(p)):
        return ""
    try:
        with open(p, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = int(fh.tell())
            take = int(max(1024, min(int(max_bytes), size)))
            fh.seek(max(0, size - take), os.SEEK_SET)
            data = fh.read()
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _task_monitor_read_head_line(path, max_bytes=4096):
    p = str(path or "").strip()
    if (not p) or (not os.path.isfile(p)):
        return ""
    try:
        with open(p, "rb") as fh:
            data = fh.read(int(max(128, max_bytes)))
        text = data.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if str(line or "").strip():
                return str(line)
    except Exception:
        return ""
    return ""


def _task_monitor_last_nonempty_line(text, skip_prefixes=()):
    lines = [str(x) for x in str(text or "").splitlines()]
    for raw in reversed(lines):
        line = str(raw or "").strip()
        if not line:
            continue
        if any(line.startswith(str(pfx)) for pfx in tuple(skip_prefixes or ())):
            continue
        return line
    return ""


def _task_monitor_clean_psi4_line(line):
    txt = str(line or "").strip()
    if not txt:
        return ""
    txt = re.sub(r"^\d{4}-\d{2}-\d{2}:\d{2}:\d{2}:\d{2}(?:,\d+)?\s+[A-Z]+\s+\[[^\]]+\]\s*", "", txt)
    txt = re.sub(r"^\*+ ?", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    skip_exact = {
        "Computation Completed",
        "Threads set to 1 by Python driver.",
    }
    if txt in skip_exact:
        return ""
    if set(txt) <= {"-", "="}:
        return ""
    return txt[:260]


def _task_monitor_format_number(text, decimals=2):
    raw = str(text or "").strip()
    if not raw:
        return ""
    try:
        val = float(raw)
    except Exception:
        return raw
    fmt = "{:." + str(int(max(0, decimals))) + "f}"
    return fmt.format(val)


def _task_monitor_parse_openmm_state_csv(path):
    head = _task_monitor_read_head_line(path)
    tail = _task_monitor_read_tail(path, max_bytes=32768)
    if (not head) or (not tail):
        return ""
    head = str(head).lstrip("#").strip()
    header = next(csv.reader([head]), [])
    if not header:
        return ""
    last_line = _task_monitor_last_nonempty_line(tail, skip_prefixes=("#",))
    if (not last_line) or (last_line.lstrip("#").strip() == head):
        return ""
    row = next(csv.reader([last_line.lstrip("#").strip()]), [])
    if len(row) != len(header):
        return ""
    rec = {str(k).strip(): str(v).strip() for k, v in zip(header, row)}

    def _pick(name_fragment):
        frag = str(name_fragment or "").strip().lower()
        for key, val in rec.items():
            if frag in str(key).strip().lower():
                return val
        return ""

    parts = []
    step = _pick("step")
    if step:
        parts.append(f"Step {step}")
    temp = _pick("temperature")
    if temp:
        parts.append(f"T={_task_monitor_format_number(temp, 1)} K")
    pot = _pick("potential energy")
    if pot:
        parts.append(f"Epot={_task_monitor_format_number(pot, 1)} kJ/mol")
    speed = _pick("speed")
    if speed:
        parts.append(f"Speed={_task_monitor_format_number(speed, 2)} ns/day")
    if not parts:
        return ""
    return " | ".join(parts)[:260]


def _task_monitor_parse_openmm_min_csv(path):
    tail = _task_monitor_read_tail(path, max_bytes=16384)
    last_line = _task_monitor_last_nonempty_line(tail, skip_prefixes=("iteration,",))
    if not last_line:
        return ""
    row = next(csv.reader([last_line]), [])
    if len(row) < 2:
        return ""
    return (
        f"Minimization iter {str(row[0]).strip()} | "
        f"E={_task_monitor_format_number(row[1], 1)} kJ/mol"
    )[:260]


def _task_monitor_parse_qmmm_probe_csv(path):
    tail = _task_monitor_read_tail(path, max_bytes=16384)
    last_line = _task_monitor_last_nonempty_line(tail, skip_prefixes=("step,",))
    if not last_line:
        return ""
    row = next(csv.reader([last_line]), [])
    if len(row) < 5:
        return ""
    step = str(row[0]).strip()
    ok = str(row[2]).strip()
    e_qm = _task_monitor_format_number(row[3], 6)
    e_qmmm = _task_monitor_format_number(row[4], 2)
    return f"QM/MM probe step {step} | ok={ok} | E_qm={e_qm} Eh | E_qmmm={e_qmmm} kJ/mol"[:260]


def task_monitor_probe_psi4(job_dir):
    jdir = str(job_dir or "").strip()
    if (not jdir) or (not os.path.isdir(jdir)):
        return {}
    candidates = []
    for name in ("psi4_output.log", "psi4_output.out", "psi4_run.out"):
        path = os.path.join(jdir, name)
        if os.path.isfile(path):
            try:
                mtime = float(os.path.getmtime(path))
            except Exception:
                mtime = 0.0
            candidates.append((mtime, path))
    for _mtime, path in sorted(candidates, key=lambda x: x[0], reverse=True):
        tail = _task_monitor_read_tail(path, max_bytes=65536)
        for raw in reversed(str(tail or "").splitlines()):
            msg = _task_monitor_clean_psi4_line(raw)
            if msg:
                try:
                    mtime = float(os.path.getmtime(path))
                except Exception:
                    mtime = 0.0
                return {
                    "message": msg,
                    "source": os.path.basename(path),
                    "mtime": float(mtime),
                    "signature": f"{os.path.basename(path)}:{mtime:.6f}:{msg}",
                }
    for name, msg in (
        ("ir_spectrum.csv", "Writing IR spectrum"),
        ("vib_modes.npz", "Writing vibrational modes"),
        ("optimized.xyz", "Geometry output ready"),
    ):
        path = os.path.join(jdir, name)
        if os.path.isfile(path):
            try:
                mtime = float(os.path.getmtime(path))
            except Exception:
                mtime = 0.0
            return {
                "message": str(msg),
                "source": str(name),
                "mtime": float(mtime),
                "signature": f"{name}:{mtime:.6f}:{msg}",
            }
    return {}


def task_monitor_probe_openmm(job_dir):
    jdir = str(job_dir or "").strip()
    if (not jdir) or (not os.path.isdir(jdir)):
        return {}
    patterns = (
        ("*_openmm_state.csv", _task_monitor_parse_openmm_state_csv),
        ("*_openmm_minimization.csv", _task_monitor_parse_openmm_min_csv),
        ("*_qmmm_probe_series.csv", _task_monitor_parse_qmmm_probe_csv),
    )
    for pattern, parser in patterns:
        matches = []
        try:
            import glob

            matches = sorted(glob.glob(os.path.join(jdir, pattern)))
        except Exception:
            matches = []
        for path in sorted(matches, key=lambda p: os.path.getmtime(p), reverse=True):
            msg = str(parser(path) or "").strip()
            if msg:
                try:
                    mtime = float(os.path.getmtime(path))
                except Exception:
                    mtime = 0.0
                return {
                    "message": msg,
                    "source": os.path.basename(path),
                    "mtime": float(mtime),
                    "signature": f"{os.path.basename(path)}:{mtime:.6f}:{msg}",
                }
    return {}


def task_monitor_probe_job(tool_label, job_dir):
    tool = str(tool_label or "").strip().lower()
    if "psi4" in tool:
        return task_monitor_probe_psi4(job_dir)
    if ("openmm" in tool) or ("qm/mm" in tool) or ("qmmm" in tool):
        return task_monitor_probe_openmm(job_dir)
    return {}
