"""QM/MM with MOPAC (PM7) in OpenMM point charges: single points, electrostatics, optimization,
thermochemistry, relaxed reaction scans and transition-state refinement."""

import os
import re

from .. import presentation
from ..registry import ToolError
from ._dialogjob import start_finalize_job
from ._widgets import jsonable, set_combo, set_value

_OPS = ["probe", "electrostatic", "optimization", "thermochemistry", "reaction_scan", "transition_state",
        "mopac_ts_frequency", "mopac_irc", "mopac_qst2", "mopac_qst3", "mopac_reaction_path"]
_MOPAC_DIRECT = {"mopac_ts_frequency", "mopac_irc", "mopac_qst2", "mopac_qst3", "mopac_reaction_path"}


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


def _path_runs_backwards(reactant, xyz_path):
    """True when the IRC file starts at the product side (its last frame is closer to the reactant)."""
    try:
        import numpy as np
        from pymol import cmd

        from . import qmviz

        _s, frames = qmviz.read_xyz_frames(str(xyz_path))
        ref = np.asarray(cmd.get_coords(reactant, 1), dtype=float)
        if frames.shape[1] != ref.shape[0]:
            return False

        def rmsd(P):
            R, t = qmviz.kabsch(P, ref)
            return float(np.sqrt((((P @ R.T + t) - ref) ** 2).sum(axis=1).mean()))
        return rmsd(frames[-1]) < rmsd(frames[0])
    except Exception:
        return False


def _mopac_reaction_result(ctx, operation, object, res):
    """Direct MOPAC TS/IRC/QST workflows: TS validation, IRC profile, loaded path objects."""
    from pymol import cmd

    if not res.get("ok", False):
        raise ToolError(str(res.get("error") or "; ".join(map(str, res.get("warnings") or [])) or
                            "the MOPAC reaction workflow did not produce a validated result"))
    ts = dict(res.get("ts_validation") or {})
    irc = dict(res.get("irc_profile") or {})
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", object)
    objs = [o for o in (f"{safe}_mopac_ts", f"{safe}_mopac_irc") if o in cmd.get_names("objects")]
    out = {"operation": operation, "object": object, "job_dir": res.get("job_dir") or res.get("workdir"),
           "ts_ok": bool(ts.get("ok")), "imaginary_modes": ts.get("imaginary_mode_count"),
           "primary_imaginary_cm1": ts.get("primary_imag_cm1"), "objects": objs,
           "ts_xyz": res.get("ts_xyz"), "irc_xyz": res.get("irc_xyz"), "irc_csv": res.get("irc_csv"),
           "warnings": list(res.get("warnings") or [])[:6]}
    rel = [v for v in (irc.get("relative_energy_kcal_mol") or []) if v is not None]
    reverse = _path_runs_backwards(object, res.get("irc_xyz"))
    if rel:
        ts_i = int(irc.get("ts_frame_index", 0))
        e = list(irc["relative_energy_kcal_mol"])
        x = list(irc.get("coordinate_amu05_bohr") or range(len(e)))
        if reverse:                                     # present it reactant -> product
            e, ts_i = e[::-1], len(e) - 1 - ts_i
            x = [-v for v in x[::-1]]
        out["irc_points"] = irc.get("n_points")
        out["barrier_forward_kcal_mol"] = round(e[ts_i] - e[0], 2) if e[0] is not None else None
        out["barrier_reverse_kcal_mol"] = round(e[ts_i] - e[-1], 2) if e[-1] is not None else None
        out["reaction_energy_kcal_mol"] = round(e[-1] - e[0], 2) if None not in (e[0], e[-1]) else None
        ctx.__dict__.setdefault("plots", {})["irc_profile"] = {
            "object": object, "program": "MOPAC", "x": x, "y_kcal": e, "ts_index": ts_i}
    if res.get("irc_xyz"):
        ctx.__dict__.setdefault("qm", {})["path"] = {"source": "mopac", "xyz": res.get("irc_xyz"),
                                                    "reverse": reverse, "object": object}
    for o in objs:
        presentation.style_ligand(o)
    if objs:
        presentation.focus(objs[-1], buffer=4.0)
        if objs[-1].endswith("_irc"):
            cmd.mplay()
    out["scene"] = ctx.scene(f"MOPAC {operation} on {object}")
    out["_card"] = (f"MOPAC **{operation}** on {object}: TS {'validated' if out['ts_ok'] else 'needs checking'} "
                    f"({out['imaginary_modes']} imaginary mode(s), {float(out['primary_imaginary_cm1'] or 0):.0f}i "
                    "cm⁻¹)"
                    + (f"; barrier **{out['barrier_forward_kcal_mol']} kcal/mol** (reverse "
                       f"{out['barrier_reverse_kcal_mol']}), ΔE {out['reaction_energy_kcal_mol']} kcal/mol"
                       if rel else "")
                    + (f"; IRC animation in `{objs[-1]}`" if objs and objs[-1].endswith("_irc") else "") + ".")
    return out


def register(reg):
    @reg.tool("run_qmmm", "QM/MM calculation of a protein-ligand complex or enzyme: a QM region (semiempirical PM7 "
              "with MOPAC) embedded in the OpenMM force field. Operations: probe (single point), electrostatic "
              "(QM-MM electrostatics report), optimization, thermochemistry, reaction_scan (relaxed scan of the "
              "distance between two atoms, e.g. a bond forming/breaking) and transition_state (dimer TS refinement "
              "from the current geometry). Direct MOPAC reaction workflows on the QM atoms only (also for small "
              "molecules in the gas phase; region_mode=pymol_selection, qm_selection='all' by default): "
              "mopac_ts_frequency (TS refinement + frequency check), mopac_irc (TS + intrinsic reaction "
              "coordinate), mopac_qst2 (TS between reactant=object and product_object), mopac_qst3 (plus "
              "ts_guess_object), mopac_reaction_path (reactant + product -> TS -> IRC). Products and guesses need "
              "the same atoms in the same order. Single points take seconds to minutes; scans and TS searches take "
              "tens of minutes to hours.",
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
                  "product_object": {"type": "string", "description": "mopac_qst2/qst3/reaction_path"},
                  "ts_guess_object": {"type": "string", "description": "mopac_qst3 (required), ts_frequency, irc"},
                  "irc_direction": {"type": "string", "enum": ["both", "forward", "reverse"]},
                  "irc_max_points": {"type": "integer", "minimum": 5, "maximum": 200},
              }, "required": ["object", "operation"]},
              permission="compute_heavy", title="QM/MM (MOPAC PM7)")
    def run_qmmm(ctx, object, operation, region_mode="auto_ligand_pocket", qm_selection="", qm_resnames="",
                 qm_cutoff_A=None, max_qm_atoms=None, qm_charge=0, qm_multiplicity=1, solvent="implicit",
                 scan_atom_a="", scan_atom_b="", scan_start_A=1.2, scan_end_A=2.8, scan_points=9, timeout_s=None,
                 product_object="", ts_guess_object="", irc_direction="both", irc_max_points=25):
        from pymol import cmd

        dlg = ctx.dlg
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        direct = operation in _MOPAC_DIRECT
        if direct:
            for need, val in (("product_object", product_object), ("ts_guess_object", ts_guess_object)):
                if val and val not in cmd.get_names("objects"):
                    raise ToolError(f"no object named {val!r} ({need})")
            if operation in ("mopac_qst2", "mopac_qst3", "mopac_reaction_path") and not product_object:
                raise ToolError(f"{operation} needs product_object (same atoms, same order as {object})")
            if operation == "mopac_qst3" and not ts_guess_object:
                raise ToolError("mopac_qst3 needs ts_guess_object")
            if region_mode in ("auto_ligand_pocket", "auto_ligand"):
                region_mode, qm_selection = "pymol_selection", (qm_selection or "all")
            params = dict(dlg._qmmm_operation_defaults(operation))
            params.update({"product_object": product_object, "ts_guess_object": ts_guess_object})
            if "irc_direction" in params:
                params.update({"irc_direction": irc_direction, "irc_max_points": int(irc_max_points)})
            dlg._qmmm_set_operation_params(operation, params)
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
            if direct:
                return _mopac_reaction_result(ctx, operation, object, result)
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
