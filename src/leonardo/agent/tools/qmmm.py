"""QM/MM with MOPAC (PM7) in OpenMM point charges: single points, electrostatics, optimization,
thermochemistry, relaxed reaction scans and transition-state refinement."""

import os

from .. import presentation
from ..registry import ToolError
from ._dialogjob import start_finalize_job
from ._widgets import jsonable, set_combo, set_value

_OPS = ["probe", "electrostatic", "optimization", "thermochemistry", "reaction_scan", "transition_state"]


def _summary(op, probe):
    q = dict(probe.get("qmmm") or {})
    out = {"energy_qm_hartree": q.get("energy_qm_hartree"),
           "energy_qmmm_elec_kj_mol": q.get("energy_qmmm_elec_kj_mol"),
           "qm_atom_charges": q.get("q_qm"), "workdir": q.get("workdir"),
           "qm_region": (probe.get("region") or {}).get("selection_hint"),
           "diagnostics": jsonable(probe.get("diagnostics") or {})}
    scan = dict(probe.get("scan") or {})
    if scan:
        out["scan"] = {"rows": jsonable(scan.get("rows") or []),
                       "barrier_estimate_kj_mol": scan.get("barrier_estimate_kj_mol"),
                       "highest_point": jsonable(scan.get("highest_point")),
                       "interior_maximum": scan.get("interior_maximum"),
                       "profile_csv": scan.get("profile_csv"), "path_xyz": scan.get("path_xyz")}
    ts = dict(probe.get("ts") or (scan.get("ts") if scan else {}) or {})
    if ts:
        out["ts"] = jsonable({k: ts.get(k) for k in ("converged", "stopped_reason", "energy_kj_mol", "barrier_kj_mol",
                                                     "max_force_kj_mol_ang", "frequencies_cm1",
                                                     "frequency_validation", "ts_xyz")})
    out = jsonable(out)             # numpy arrays -> lists before testing for empty values
    return {k: v for k, v in out.items() if v is not None and v != "" and v != {} and v != []}


def register(reg):
    @reg.tool("run_qmmm", "QM/MM calculation of a protein-ligand complex or enzyme: a QM region (semiempirical PM7 "
              "with MOPAC) embedded in the OpenMM force field. Operations: probe (single point), electrostatic "
              "(QM-MM electrostatics report), optimization, thermochemistry, reaction_scan (relaxed scan of the "
              "distance between two atoms, e.g. a bond forming/breaking) and transition_state (dimer TS refinement "
              "from the current geometry). Single points take seconds to minutes; scans and TS searches take tens "
              "of minutes to hours.",
              {"properties": {
                  "object": {"type": "string"},
                  "operation": {"type": "string", "enum": _OPS},
                  "region_mode": {"type": "string",
                                  "enum": ["auto_ligand_pocket", "auto_ligand", "pymol_selection", "qm_resnames"]},
                  "qm_selection": {"type": "string", "description": "PyMOL selection inside the object, for "
                                                                    "region_mode=pymol_selection"},
                  "qm_resnames": {"type": "string", "description": "e.g. 'LIG HIS' for region_mode=qm_resnames"},
                  "qm_cutoff_A": {"type": "number", "minimum": 2, "maximum": 12},
                  "max_qm_atoms": {"type": "integer", "minimum": 10, "maximum": 400},
                  "qm_charge": {"type": "integer"}, "qm_multiplicity": {"type": "integer", "minimum": 1},
                  "solvent": {"type": "string", "enum": ["gas", "implicit", "explicit"]},
                  "scan_atom_a": {"type": "string", "description": "selection of the first atom (reaction_scan)"},
                  "scan_atom_b": {"type": "string", "description": "selection of the second atom"},
                  "scan_start_A": {"type": "number"}, "scan_end_A": {"type": "number"},
                  "scan_points": {"type": "integer", "minimum": 3, "maximum": 60},
                  "timeout_s": {"type": "integer", "minimum": 30, "description": "per MOPAC call"},
              }, "required": ["object", "operation"]},
              permission="compute_heavy", title="QM/MM (MOPAC PM7)")
    def run_qmmm(ctx, object, operation, region_mode="auto_ligand_pocket", qm_selection="", qm_resnames="",
                 qm_cutoff_A=None, max_qm_atoms=None, qm_charge=0, qm_multiplicity=1, solvent="implicit",
                 scan_atom_a="", scan_atom_b="", scan_start_A=1.2, scan_end_A=2.8, scan_points=9, timeout_s=None):
        from pymol import cmd

        dlg = ctx.dlg
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        set_combo(dlg, "cmb_qmmm_object", object)
        set_combo(dlg, "cmb_qmmm_operation", operation)
        set_combo(dlg, "cmb_qmmm_mode", solvent)
        set_combo(dlg, "cmb_qmmm_region_mode", region_mode)
        if region_mode == "pymol_selection":
            if not qm_selection:
                raise ToolError("region_mode=pymol_selection needs qm_selection")
            set_value(dlg, "edit_qmmm_qm_selection", qm_selection)
            set_value(dlg, "edit_qmmm_mm_selection", "")
        if region_mode == "qm_resnames":
            if not qm_resnames:
                raise ToolError("region_mode=qm_resnames needs qm_resnames")
            set_value(dlg, "edit_qmmm_qm_resnames", qm_resnames)
        if qm_cutoff_A is not None:
            set_value(dlg, "spin_qmmm_qm_cutoff", float(qm_cutoff_A))
        if max_qm_atoms is not None:
            set_value(dlg, "spin_qmmm_max_qm_atoms", int(max_qm_atoms))
        set_value(dlg, "spin_qmmm_charge", int(qm_charge))
        set_value(dlg, "spin_qmmm_mult", int(qm_multiplicity))
        if timeout_s:
            set_value(dlg, "spin_qmmm_timeout", int(timeout_s))
        if operation == "reaction_scan":
            for sel in (scan_atom_a, scan_atom_b):
                if not sel or cmd.count_atoms(f"({object}) and ({sel})") != 1:
                    raise ToolError(f"scan atoms must each select exactly one atom of {object}: {sel!r}")
            dlg._qmmm_set_operation_params("reaction_scan", {
                "sel_a": scan_atom_a, "sel_b": scan_atom_b, "start_ang": float(scan_start_A),
                "end_ang": float(scan_end_A), "n_points": int(scan_points)})
        try:
            dlg._on_qmmm_operation_changed()
        except Exception:
            pass

        def collect(call):
            result = call.arg(1, "result") if call else None
            result = dict(result) if isinstance(result, dict) else {}
            if not result:
                raise ToolError("QM/MM did not finish (see Leonardo's message)")
            if not result.get("ok", True):
                raise ToolError(str(result.get("error") or "QM/MM failed"))
            probe = dict(result.get("qmmm_probe") or {})
            if probe and not probe.get("ok", True):
                raise ToolError(str(probe.get("error") or "QM/MM probe failed"))
            scan = dict(probe.get("scan") or {})
            if scan.get("rows"):
                ctx.__dict__.setdefault("plots", {})["qmmm_scan"] = {"object": object, "rows": jsonable(scan["rows"]),
                                                                    "barrier": scan.get("barrier_estimate_kj_mol")}
            out = dict(_summary(operation, probe or result), operation=operation, object=object,
                       warnings=list(result.get("warnings") or probe.get("warnings") or [])[:6],
                       job_dir=str(getattr(dlg, "_openmm_last_job_dir", "") or ""))
            xyz = ((out.get("ts") or {}).get("ts_xyz") or (out.get("scan") or {}).get("path_xyz") or "")
            if xyz and os.path.isfile(str(xyz)):
                name = f"qmmm_{operation}_{object}"[:60]
                cmd.load(str(xyz), name)
                presentation.style_ligand(name)
                out["qm_object"] = name
            presentation.focus(out.get("qm_object") or object, buffer=6.0)
            out["scene"] = ctx.scene(f"QM/MM {operation} on {object}")
            barrier = (out.get("scan") or {}).get("barrier_estimate_kj_mol") or (out.get("ts") or {}).get("barrier_kj_mol")
            e = out.get("energy_qm_hartree")
            out["_card"] = (f"QM/MM **{operation}** on {object}"
                            + (f"; QM region {out['qm_region']}" if out.get("qm_region") else "")
                            + (f"; E(QM) {float(e):.6f} Eh" if e is not None else "")
                            + (f"; QM-MM electrostatics **{float(out['energy_qmmm_elec_kj_mol']):.1f} kJ/mol**"
                               if out.get("energy_qmmm_elec_kj_mol") is not None else "")
                            + (f"; barrier ≈ **{float(barrier):.1f} kJ/mol**" if barrier is not None else "")
                            + (f"; TS converged: {out['ts'].get('converged')}" if out.get("ts") else "") + ".")
            return out

        return start_finalize_job(ctx, "qmmm", dlg._on_qmmm_run_probe, "_openmm_finalize_worker_result", collect,
                                  busy_button="btn_qmmm_run_probe", description=f"QM/MM {operation} on {object}")
