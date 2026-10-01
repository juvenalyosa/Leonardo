"""Diagnostics and output: Leonardo's debug log, every running task, installation readiness,
publication figures, and the QM-driven vibrational refinement of the ligand force field."""

import json
import os
import subprocess
import sys
import time

from .. import presentation
from ..jobs import POPEN_GROUP, stop_process
from ..registry import ToolError

READINESS_WORKFLOWS = ["bioseq", "alphafold", "docking", "redocking", "virtual_screening", "lead_optimization",
                       "abfe", "openmm_qmmm", "psi4_qm"]


def _cmd():
    from pymol import cmd

    return cmd


def _python_executable():
    for cand in (os.path.join(sys.prefix, "bin", "python"), os.path.join(sys.prefix, "python.exe")):
        if os.path.isfile(cand):
            return cand
    return sys.executable


def debug_log_tail(lines, n=60, contains=""):
    rows = [str(x) for x in (lines or [])]
    if contains:
        key = str(contains).lower()
        rows = [x for x in rows if key in x.lower()]
    return rows[-max(1, int(n)):]


def task_rows(state_map, now=None):
    now = time.time() if now is None else float(now)
    rows = []
    for jid, meta in sorted(dict(state_map or {}).items(), key=lambda kv: float((kv[1] or {}).get("started_at", 0))):
        meta = dict(meta or {})
        started = float(meta.get("started_at", 0) or 0)
        rows.append({
            "id": str(jid),
            "tool": str(meta.get("tool_label", "") or ""),
            "task": str(meta.get("task_label", "") or ""),
            "status": str(meta.get("status", "") or ""),
            "progress": meta.get("progress"),
            "message": str(meta.get("message", "") or "")[:300],
            "job_dir": str(meta.get("job_dir", "") or ""),
            "running_s": round(now - started, 1) if started else None,
        })
    return rows


def readiness_summary(report):
    out = {"ok": bool(report.get("ok")), "status_counts": dict(report.get("status_counts") or {}), "workflows": []}
    for item in report.get("workflow_reports", []):
        row = {"id": item.get("workflow_id"), "title": item.get("title"), "status": item.get("status")}
        for key in ("missing_python_required", "missing_executables_required", "missing_python_recommended",
                    "missing_executables_recommended"):
            if item.get(key):
                row[key] = list(item[key])
        if item.get("install_hints"):
            row["install_hints"] = list(item["install_hints"])[:4]
        out["workflows"].append(row)
    return out


def vib_refine_command(python, inputs, out_dir, qm_engine="mopac", method="wb97x-d", basis="def2-SVP",
                       threads=2, memory_gb=4, swarm_size=8, n_iter=8, fit_mode="typed_hybrid", seed=0,
                       platform="CPU"):
    """argv for ``python -m leonardo.run_ligand_vib_refine`` from Leonardo's ligand files."""
    argv = [python, "-m", "leonardo.run_ligand_vib_refine", "--out-dir", out_dir, "--qm-engine", qm_engine,
            "--method", method, "--basis", basis, "--threads", str(int(threads)), "--memory-gb", str(int(memory_gb)),
            "--swarm-size", str(int(swarm_size)), "--n-iter", str(int(n_iter)), "--fit-mode", fit_mode,
            "--seed", str(int(seed)), "--platform", platform]
    if inputs.get("mol2") and inputs.get("prmtop") and inputs.get("inpcrd"):
        argv += ["--mol2", inputs["mol2"], "--prmtop", inputs["prmtop"], "--inpcrd", inputs["inpcrd"]]
        if inputs.get("frcmod"):
            argv += ["--frcmod", inputs["frcmod"]]
        if inputs.get("sdf"):
            argv += ["--qm-start-sdf", inputs["sdf"]]
    elif inputs.get("sdf"):
        argv += ["--bootstrap-template-sdf", inputs["sdf"]]
    else:
        raise ToolError("no ligand to refine: parametrize the ligand first (parametrize_ligand) or set an SDF ligand")
    return argv


def _ligand_inputs(dlg):
    get = lambda attr: str(getattr(dlg, attr, "") or "")  # noqa: E731
    ins = {"mol2": get("_parameterized_ligand_mol2"), "prmtop": get("_parameterized_ligand_prmtop"),
           "inpcrd": get("_parameterized_ligand_inpcrd"), "frcmod": get("_parameterized_ligand_frcmod")}
    lig = get("_docking_ligand_file")
    if lig.lower().endswith(".sdf"):
        ins["sdf"] = lig
    return {k: v for k, v in ins.items() if v and os.path.isfile(v)}


def register(reg):
    @reg.tool("read_diagnostics_log", "Last lines of Leonardo's diagnostics/debug log (what the modules printed: "
              "commands run, warnings, tracebacks). Use it to understand why a step failed. Optional text filter.",
              {"properties": {"lines": {"type": "integer", "minimum": 1, "maximum": 400},
                              "contains": {"type": "string"}}},
              permission="read", title="Diagnostics log")
    def read_diagnostics_log(ctx, lines=60, contains=""):
        rows = debug_log_tail(getattr(ctx.dlg, "_debug_log_lines", []), lines, contains)
        return {"lines": rows, "count": len(rows)}

    @reg.tool("list_tasks", "Every background task in Leonardo's Task Monitor (also the ones the user started "
              "from the GUI: docking, MD, Psi4, ...) with status, progress, message and job folder, plus the "
              "assistant's own jobs.",
              permission="read", title="Task Monitor")
    def list_tasks(ctx):
        rows = task_rows(getattr(ctx.dlg, "_task_monitor_state", {}) or {})
        jobs = ctx.jobs.list() if hasattr(ctx.jobs, "list") else []
        running = sum(1 for r in rows if r["status"].lower() not in ("completed", "failed", "canceled", "cancelled"))
        return {"leonardo_tasks": rows, "assistant_jobs": jobs, "running": running,
                "_card": f"**{len(rows)} Leonardo task(s)**, {running} running; {len(jobs)} assistant job(s)."}

    @reg.tool("readiness_report", "Which Leonardo workflows this installation can run (bioseq, alphafold, docking, "
              "redocking, virtual_screening, lead_optimization, abfe, openmm_qmmm, psi4_qm): missing Python "
              "packages or programs and install hints. Use it for 'can I run X here?' questions.",
              {"properties": {"workflows": {"type": "array", "items": {"type": "string", "enum": READINESS_WORKFLOWS}}}},
              permission="read", title="Installation readiness")
    def readiness_report(ctx, workflows=None):
        from ...scientific_readiness_runtime import build_suite_readiness_report

        out = readiness_summary(build_suite_readiness_report(workflow_ids=list(workflows or [])))
        counts = out["status_counts"]
        out["_card"] = (f"Readiness: {counts.get('ready', 0)} ready, {counts.get('ready_with_gaps', 0)} with gaps, "
                        f"{counts.get('missing_required', 0)} missing requirements.")
        return out

    @reg.tool("export_figure", "Publication image of the current PyMOL view (ray-traced PNG) with Leonardo's "
              "interaction legend, a scale bar and a title, as the Render/Export panel does.",
              {"properties": {
                  "name": {"type": "string"},
                  "width": {"type": "integer", "minimum": 400, "maximum": 6000},
                  "height": {"type": "integer", "minimum": 400, "maximum": 6000},
                  "dpi": {"type": "integer", "minimum": 72, "maximum": 1200},
                  "legend": {"type": "boolean"},
                  "scale_bar": {"type": "boolean"},
                  "scale_length_angstrom": {"type": "number", "minimum": 1, "maximum": 100},
                  "title": {"type": "string"},
                  "transparent_background": {"type": "boolean"},
              }, "required": ["name"]},
              permission="compute_light", title="Export figure")
    def export_figure(ctx, name, width=2000, height=2000, dpi=300, legend=True, scale_bar=True,
                      scale_length_angstrom=10.0, title="", transparent_background=False):
        dlg = ctx.dlg
        cmd = _cmd()
        path = os.path.join(ctx.workspace("figures"), presentation.safe_name(name, "figure") + ".png")
        had_settings = isinstance(getattr(dlg, "_settings", None), dict)
        if not had_settings:
            dlg._settings = {}          # the export helpers read scale length and title from here
        settings = dlg._settings
        saved = dict(settings)
        notes = []
        try:
            settings["scale_length"] = float(scale_length_angstrom)
            settings["title_text"] = str(title or "")
            cmd.set("ray_trace_mode", 1)
            cmd.set("ray_opaque_background", 0 if transparent_background else 1)
            for flag, obj, fn in ((legend, "Interactions.Legend", lambda: dlg._update_legend(
                    for_export=True, anchor=dlg._legend_anchor())),
                    (scale_bar, "Interactions.ScaleBar", dlg._update_scale_bar_for_export),
                    (bool(title), "Interactions.Title", dlg._update_title_for_export)):
                try:
                    if flag:
                        fn()
                    else:
                        cmd.delete(obj)
                except Exception as exc:
                    notes.append(f"{obj.split('.')[-1]} skipped: {exc}")
            cmd.png(path, width=int(width), height=int(height), dpi=int(dpi), ray=1)
        finally:
            settings.clear()
            settings.update(saved)
            if not had_settings:
                del dlg._settings
            cmd.set("ray_opaque_background", 1)
        if not os.path.isfile(path):
            raise ToolError("PyMOL did not write the image")
        out = {"path": path, "width": int(width), "height": int(height), "dpi": int(dpi),
               "_card": f"Figure saved: `{path}`"}
        if notes:
            out["notes"] = notes
        return out

    @reg.tool("refine_ligand_vibrational_ff", "Refine the active ligand's GAFF force field against QM vibrations: "
              "QM optimization + frequencies (MOPAC PM7 in minutes, or Psi4 DFT in hours), then a particle-swarm "
              "fit of bond/angle/torsion/LJ scale factors so MM normal modes match QM. Writes a refined frcmod, "
              "OpenMM XML, IR comparison plot and report. Uses the parametrized ligand (parametrize_ligand) or "
              "bootstraps it from the active SDF with AmberTools. Runs in the background (job_status).",
              {"properties": {
                  "qm_engine": {"type": "string", "enum": ["mopac", "psi4"]},
                  "method": {"type": "string", "description": "Psi4 method (default wb97x-d)"},
                  "basis": {"type": "string", "description": "Psi4 basis (default def2-SVP)"},
                  "threads": {"type": "integer", "minimum": 1, "maximum": 64},
                  "memory_gb": {"type": "integer", "minimum": 1, "maximum": 256},
                  "swarm_size": {"type": "integer", "minimum": 4, "maximum": 64},
                  "iterations": {"type": "integer", "minimum": 1, "maximum": 100},
                  "fit_mode": {"type": "string", "enum": ["typed_hybrid", "global"]},
              }},
              permission="compute_heavy", title="Ligand vibrational FF refinement", main_thread=True)
    def refine_ligand_vibrational_ff(ctx, qm_engine="mopac", method="wb97x-d", basis="def2-SVP", threads=2,
                                     memory_gb=4, swarm_size=8, iterations=8, fit_mode="typed_hybrid"):
        inputs = _ligand_inputs(ctx.dlg)
        out_dir = os.path.join(ctx.workspace("vib_refine"), time.strftime("refine_%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        argv = vib_refine_command(_python_executable(), inputs, out_dir, qm_engine=qm_engine, method=method,
                                  basis=basis, threads=threads, memory_gb=memory_gb, swarm_size=swarm_size,
                                  n_iter=iterations, fit_mode=fit_mode)
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        env = dict(os.environ)
        env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        log_path = os.path.join(out_dir, "refine_log.txt")
        log = open(log_path, "w")
        proc = subprocess.Popen(argv, cwd=out_dir, stdout=log, stderr=subprocess.STDOUT, env=env, **POPEN_GROUP)

        def poll():
            if proc.poll() is None:
                return {"done": False, "progress": _last_line(log_path)}
            log.close()
            summary_path = os.path.join(out_dir, "refinement_summary.json")
            summary = {}
            if os.path.isfile(summary_path):
                with open(summary_path) as fh:
                    summary = json.load(fh)
            if proc.returncode != 0 or not summary.get("ok"):
                err = summary.get("error") or _tail(log_path)
                return {"done": True, "error": f"refinement failed: {err}"}
            return {"done": True, "result": _refine_result(summary, out_dir)}

        def cancel():
            stop_process(proc, log)

        return ctx.jobs.start("vib_refine", poll, cancel=cancel,
                              description=f"ligand FF vibrational refinement ({qm_engine})")


def _last_line(path):
    try:
        with open(path) as fh:
            rows = [ln.strip() for ln in fh if ln.strip()]
        return rows[-1][:200] if rows else "starting"
    except OSError:
        return "starting"


def _tail(path, n=800):
    try:
        with open(path) as fh:
            return fh.read()[-n:]
    except OSError:
        return "no log"


def _refine_result(summary, out_dir):
    keep = {}
    for k, v in summary.items():
        if isinstance(v, (int, float, str, bool)) and k not in ("traceback",):
            keep[k] = v
    xml = dict(summary.get("xml_export") or {})
    keep["refined_xml"] = xml.get("xml") or xml.get("path") or xml.get("out_xml")
    keep["out_dir"] = out_dir
    keep["note"] = ("The refined parameters are written but not applied: load the refined XML/frcmod for MD "
                    "only after checking the report (mode comparison and IR plot).")
    keep["_card"] = f"Vibrational refinement done: report `{summary.get('report_md', '')}`"
    return keep
