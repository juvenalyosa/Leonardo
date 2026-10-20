"""Short classical molecular dynamics with OpenMM."""

from .. import presentation
from ..registry import ToolError
from ._dialogjob import start_finalize_job


def _ca_rmsd(obj):
    from pymol import cmd

    try:
        n = cmd.count_states(obj)
        if n < 2:
            return None
        values = cmd.intra_fit(f"({obj}) and polymer and name CA", 1)
        vals = [float(v) for v in values if v is not None and float(v) >= 0]
        return {"frames": n, "final_A": round(vals[-1], 2), "max_A": round(max(vals), 2),
                "mean_A": round(sum(vals) / len(vals), 2)} if vals else None
    except Exception:
        return None


def _component_guard(ctx, object, family, allow_incomplete=False):
    """
    Leonardo drops residues it cannot parameterize before MD. Refuse when that would remove a
    metal, cofactor or modified residue (unless the model explicitly accepts it).
    """
    fam = str(family or "").strip().lower()
    if allow_incomplete or fam not in ("amber14", "charmm36"):
        return None
    try:
        from openmm import app, unit

        from ...component_library_runtime import resolve_components
        from .components import _save_object

        pdb = app.PDBFile(_save_object(ctx, object, sub="md_checks"))
        plan = resolve_components(pdb.topology, [tuple(p.value_in_unit(unit.nanometer)) for p in pdb.positions],
                                  family=fam)
    except ToolError:
        raise
    except Exception:
        return None
    bad = sorted({f"{r['resn']} ({r['source']})" for r in plan["components"]
                  if r["source"] in ("metal", "unmatched", "modified_residue", "covalent", "covalent_metal")})
    if bad:
        rec = plan.get("recommended_family")
        raise ToolError(f"{', '.join(bad)} would be removed before MD in {fam} (no parameters). "
                        + (f"Use force_field='{rec}', which covers them, " if rec and rec != fam else "Fix them first ")
                        + "or pass allow_incomplete=true only if the user accepts simulating without them.")
    return plan


def _health_guard(ctx, object, allow_unsafe=False):
    """
    Check the structure before simulating it.  Red findings (a chain bonded across a gap, residues with
    missing atoms that would be dropped, atoms on top of each other) make the run unphysical: refuse unless
    the model passes allow_unsafe (the user's OK).  Returns the findings for the result.
    """
    try:
        from ... import simulation_health_runtime as health_rt
        from .components import _save_object

        findings = health_rt.check_pdb(_save_object(ctx, object, sub="md_checks"))
    except ToolError:
        raise
    except Exception:
        return []
    reds = [f for f in findings if f["level"] == "red"]
    if reds and not allow_unsafe:
        raise ToolError("the structure would give an unphysical simulation: "
                        + " | ".join(health_rt.format_findings(reds))
                        + ". Fix it first (prepare_protein fixes these), or pass allow_unsafe=true only if the user "
                        "accepts simulating it as it is.")
    return findings


def _health_result(info, res):
    """The worker's health report, condensed for the model: confidence, findings, key statistics."""
    found = list(info.pop("_precheck", []) or [])
    health = dict(res.get("health") or {})
    if not health:
        return
    from ... import simulation_health_runtime as health_rt

    seen = {f["code"] for f in found}
    for key in ("precheck", "system", "findings"):
        for f in health.get(key) or []:
            if f["code"] not in seen:
                seen.add(f["code"])
                found.append(f)
    info["health"] = {"confidence": (health.get("summary") or health_rt.summarize(found))["confidence"],
                      "findings": [{k: f[k] for k in ("level", "code", "title", "detail", "where", "fix")}
                                   for f in found if f["level"] in ("red", "yellow")][:10],
                      "stats": health.get("stats") or {}, "checks": health.get("checks", 0)}
    if health.get("stopped"):
        info["health"]["stopped"] = health["stopped"]


def _current_family(dlg, requested=None):
    if requested:
        return requested
    try:
        return str(dlg.cmb_openmm_ff_family.currentData() or "")
    except Exception:
        return ""


def register(reg):
    @reg.tool("run_md", "Short classical MD of a protein or protein-ligand complex with OpenMM (minimization, "
              "heating, equilibration and production). Ligands/cofactors in the object are parameterized "
              "automatically. Runs in the background; the trajectory is loaded in the viewer. 0.1 ns in implicit "
              "water takes minutes on a laptop; explicit water is much slower.",
              {"properties": {
                  "object": {"type": "string"},
                  "production_ns": {"type": "number", "minimum": 0.001, "maximum": 50},
                  "solvent": {"type": "string", "enum": ["implicit", "explicit", "gas"]},
                  "temperature_K": {"type": "number", "minimum": 250, "maximum": 400},
                  "ph": {"type": "number", "minimum": 0, "maximum": 14},
                  "force_field": {"type": "string", "enum": ["amber14", "amber99sbildn", "charmm36"]},
                  "water_model": {"type": "string", "enum": ["tip3p", "spce", "tip4pew", "tip5p"],
                                  "description": "explicit solvent only"},
                  "ensemble": {"type": "string", "enum": ["npt", "nvt", "nve"],
                               "description": "NPT needs explicit solvent (a box); implicit/gas run NVT"},
                  "timestep_fs": {"type": "number", "minimum": 0.5, "maximum": 4.0},
                  "keep_crystal_waters": {"type": "boolean"}, "keep_ions": {"type": "boolean"},
                  "keep_cofactors": {"type": "boolean", "description": "crystal ligands/cofactors (parameterized). "
                                     "Setting any keep_* replaces a component choice made with "
                                     "configure_md_components; omit them to keep that choice."},
                  "equilibration_ns": {"type": "number", "minimum": 0.001, "maximum": 10,
                                       "description": "Default: 20% of production (min 4 ps, max 40 ps); heating "
                                                      "10% (min 2 ps, max 10 ps)."},
                  "allow_incomplete": {"type": "boolean", "description": "run even if metals/cofactors/modified "
                                       "residues without parameters would be removed (only with the user's OK)"},
                  "allow_unsafe": {"type": "boolean", "description": "run even if the health checks find the "
                                   "structure/system unphysical (gap bonded across, missing atoms, overlapping atoms, "
                                   "time step too long); only with the user's OK"},
              }, "required": ["object"]},
              permission="compute_heavy", title="Molecular dynamics (OpenMM)")
    def run_md(ctx, object, production_ns=0.1, solvent="implicit", temperature_K=300.0, ph=7.4,
               equilibration_ns=None, force_field=None, water_model=None, ensemble=None, timestep_fs=None,
               keep_crystal_waters=None, keep_ions=None, keep_cofactors=None, allow_incomplete=False,
               allow_unsafe=False):
        from pymol import cmd

        dlg = ctx.dlg
        precheck = _health_guard(ctx, object, allow_unsafe)
        if keep_cofactors is not False:
            _component_guard(ctx, object, _current_family(dlg, force_field), allow_incomplete)
        _select_openmm_object(dlg, object, solvent, ph)
        dlg.spin_openmm_temperature.setValue(float(temperature_K))
        from ._widgets import set_combo

        if force_field:
            set_combo(dlg, "cmb_openmm_ff_family", force_field, required=False)
        if water_model:
            set_combo(dlg, "cmb_openmm_water_model", water_model, required=False)
        set_combo(dlg, "cmb_openmm_ensemble", ensemble or ("npt" if solvent == "explicit" else "nvt"), required=False)
        if timestep_fs:
            dlg.spin_openmm_timestep_fs.setValue(float(timestep_fs))
        keeps = (("cb_openmm_keep_waters", keep_crystal_waters), ("cb_openmm_keep_ions", keep_ions),
                 ("cb_openmm_keep_cofactors", keep_cofactors))
        if any(val is not None for _attr, val in keeps):
            # an explicit component choice (configure_md_components) would override the switches
            cache = getattr(dlg, "_openmm_hetatm_selection_cache", None)
            if isinstance(cache, dict):
                cache.pop(object, None)
        for attr, val in keeps:
            if val is not None and hasattr(dlg, attr):
                getattr(dlg, attr).setChecked(bool(val))
        dt_fs = float(dlg.spin_openmm_timestep_fs.value())
        prod_steps = int(round(float(production_ns) * 1e6 / dt_fs))
        # heating/equilibration scale with the production run (Leonardo's own defaults, 10 + 40 ps,
        # would dominate a short run)
        heat_ps = min(10.0, max(2.0, 0.1 * float(production_ns) * 1000.0))
        equil_ps = (float(equilibration_ns) * 1000.0 if equilibration_ns
                    else min(40.0, max(4.0, 0.2 * float(production_ns) * 1000.0)))
        heat_steps = int(round(heat_ps * 1000.0 / dt_fs))
        equil_steps = int(round(equil_ps * 1000.0 / dt_fs))
        dlg.spin_openmm_prod_steps.setValue(prod_steps)
        dlg.spin_openmm_heat_steps.setValue(heat_steps)
        dlg.spin_openmm_equil_steps.setValue(equil_steps)
        for cb in ("cb_openmm_run_minimize", "cb_openmm_run_heating", "cb_openmm_run_equil", "cb_openmm_run_prod"):
            getattr(dlg, cb).setChecked(True)
        if hasattr(dlg, "cb_openmm_min_only"):
            dlg.cb_openmm_min_only.setChecked(False)

        def collect(res):
            out_obj = str(getattr(dlg, "_openmm_last_output_object", "") or "")
            info = {"_precheck": precheck, "object": out_obj, "production_ns": production_ns, "solvent": solvent,
                    "stages_ps": {"heating": round(heat_ps, 1), "equilibration": round(equil_ps, 1),
                                  "production": round(float(production_ns) * 1000.0, 1)},
                    "force_field": res.get("ff_used_label"), "platform": res.get("platform_used"),
                    "trajectory": res.get("dcd_path"), "energies_csv": res.get("csv_path"),
                    "job_dir": res.get("job_dir"), "notes": list(res.get("notes", []) or [])[:8],
                    "removed_residues": list(res.get("removed_residues", []) or []),
                    "component_plan": res.get("component_plan")}
            if out_obj and out_obj in cmd.get_names("objects"):
                presentation.style_protein(out_obj)
                het = f"({out_obj}) and not polymer and not resn HOH+WAT+NA+CL+K"
                if cmd.count_atoms(het):
                    presentation.style_ligand(het)
                info["ca_rmsd_vs_first_frame"] = _ca_rmsd(out_obj)
                presentation.declutter(keep=(out_obj,))
                cmd.disable(object)            # the trajectory sits on top of the input structure
                cmd.hide("labels", "all")
                presentation.focus(out_obj, buffer=2.0)
                info["scene"] = ctx.scene(f"MD trajectory {out_obj} ({production_ns} ns, {solvent})")
            r = info.get("ca_rmsd_vs_first_frame") or {}
            info["_card"] = (f"MD **{production_ns} ns** ({solvent}, {info.get('force_field')}, {info.get('platform')}); "
                             f"heating {info['stages_ps']['heating']} ps + equilibration "
                             f"{info['stages_ps']['equilibration']} ps. "
                             + (f"final Cα RMSD {r['final_A']} Å (max {r['max_A']} Å)." if r else ""))
            _health_result(info, res)
            if info.get("health"):
                info["_card"] += f" Health: {info['health']['confidence']} confidence."
            return info

        return _openmm_job(ctx, "md", False, collect, description=f"OpenMM MD of {object}",
                           allow_unsafe=allow_unsafe)

    @reg.tool("check_simulation_system", "Health check of a structure before simulating it (MD, minimization, "
              "QM/MM, free energies), without changing it: chains bonded across missing segments, residues with "
              "missing atoms (the force field would drop them), atoms on top of each other, steric clashes, "
              "unbonded cysteine pairs at disulfide distance, D-amino acids and non-proline cis peptides. Each "
              "finding is red (the simulation would be wrong), yellow (suspicious) or info, with how to fix it. "
              "run_md and minimize_structure run it themselves and refuse on red; during MD a monitor stops runs "
              "that blow up and flags drift, ligand loss, density and temperature problems.",
              {"properties": {"object": {"type": "string"},
                              "timestep_fs": {"type": "number", "description": "MD time step to check (optional)"}},
               "required": ["object"]},
              permission="read", title="Check system health")
    def check_simulation_system(ctx, object, timestep_fs=None):
        from ... import simulation_health_runtime as health_rt
        from .components import _save_object

        findings = health_rt.check_pdb(_save_object(ctx, object, sub="md_checks"))
        if timestep_fs and float(timestep_fs) > health_rt.THRESHOLDS["max_timestep_hmr_fs"]:
            findings.append(health_rt.finding(
                "red", "timestep", "Time step too long",
                f"{float(timestep_fs):g} fs is beyond what H-bond constraints with heavier hydrogens allow", (),
                f"use timestep_fs <= {health_rt.THRESHOLDS['max_timestep_hmr_fs']:g}"))
        elif timestep_fs and float(timestep_fs) > health_rt.THRESHOLDS["max_timestep_hbonds_fs"]:
            findings.append(health_rt.finding(
                "info", "hmr", "Hydrogen mass repartitioning",
                f"{float(timestep_fs):g} fs: Leonardo will make hydrogens 3 amu so the time step is stable"))
        summary = health_rt.summarize(findings)
        lines = health_rt.format_findings(findings, limit=6)
        card = (f"**{object}** health: **{summary['confidence']}** confidence ({summary['red']} red, "
                f"{summary['yellow']} yellow)." + ("" if not lines else " " + " | ".join(lines)))
        return {"object": object, "confidence": summary["confidence"], "findings": findings, "_card": card}

    @reg.tool("minimize_structure", "Energy-minimize a protein or complex with OpenMM: hydrogens added at the given pH, "
              "ligands parameterized automatically, and a staged restrained minimization (heavy atoms "
              "harmonically restrained, restraints released in stages) so the fold is kept. The minimized "
              "structure is loaded as a new object. About 1-4 min on CPU for a 300-residue protein.",
              {"properties": {
                  "object": {"type": "string"},
                  "solvent": {"type": "string", "enum": ["implicit", "explicit", "gas"]},
                  "max_iterations": {"type": "integer", "minimum": 1, "maximum": 200000},
                  "ph": {"type": "number", "minimum": 0, "maximum": 14},
                  "allow_incomplete": {"type": "boolean"},
                  "allow_unsafe": {"type": "boolean", "description": "minimize even if the health checks find the "
                                   "structure unphysical (only with the user's OK)"},
              }, "required": ["object"]},
              permission="compute_light", title="Energy minimization (OpenMM)")
    def minimize_structure(ctx, object, solvent="implicit", max_iterations=2000, ph=7.4, allow_incomplete=False,
                           allow_unsafe=False):
        dlg = ctx.dlg
        precheck = _health_guard(ctx, object, allow_unsafe)
        _component_guard(ctx, object, _current_family(dlg), allow_incomplete)
        _select_openmm_object(dlg, object, solvent, ph)
        for spin in ("spin_openmm_minonly_iters", "spin_openmm_min_steps"):
            if hasattr(dlg, spin):
                getattr(dlg, spin).setValue(int(max_iterations))

        def collect(res):
            from pymol import cmd

            out_obj = str(getattr(dlg, "_openmm_last_output_object", "") or "")
            info = {"_precheck": precheck, "object": out_obj, "input": object, "solvent": solvent,
                    "force_field": res.get("ff_used_label"),
                    "platform": res.get("platform_used"), "minimized_pdb": res.get("final_pdb") or res.get("pdb_path"),
                    "job_dir": res.get("job_dir"), "notes": list(res.get("notes", []) or [])[:8],
                    "removed_residues": list(res.get("removed_residues", []) or []),
                    "component_plan": res.get("component_plan")}
            profile = [p for p in (res.get("min_energy_profile") or []) if isinstance(p, (list, tuple)) and len(p) >= 2]
            if profile:
                info["energy_kJ_mol"] = {"start": round(float(profile[0][1]), 1), "end": round(float(profile[-1][1]), 1)}
            if out_obj and out_obj in cmd.get_names("objects"):
                n = cmd.count_states(out_obj)
                if n > 1:          # keep only the final minimized frame
                    cmd.create(f"{out_obj}_tmp", out_obj, n, 1)
                    cmd.delete(out_obj)
                    cmd.set_name(f"{out_obj}_tmp", out_obj)
                try:
                    info["heavy_atom_rmsd_vs_input_A"] = round(float(cmd.align(
                        f"({out_obj}) and polymer and not elem H", f"({object}) and polymer and not elem H",
                        cycles=0, transform=0)[0]), 2)
                except Exception:
                    pass
                presentation.style_protein(out_obj)
                cmd.disable(object)
                presentation.focus(out_obj, buffer=2.0)
                info["scene"] = ctx.scene(f"{object} minimized ({out_obj})")
            e = info.get("energy_kJ_mol")
            info["_card"] = (f"**{object}** minimized → **{out_obj}** ({solvent}, {info.get('force_field')})"
                             + (f"; E {e['start']} → {e['end']} kJ/mol" if e else "")
                             + (f"; heavy-atom RMSD {info['heavy_atom_rmsd_vs_input_A']} Å"
                                if "heavy_atom_rmsd_vs_input_A" in info else "") + ".")
            _health_result(info, res)
            return info

        return _openmm_job(ctx, "minimize", True, collect, description=f"OpenMM minimization of {object}",
                           allow_unsafe=allow_unsafe)


def _select_openmm_object(dlg, object, solvent, ph):
    from pymol import cmd

    if object not in cmd.get_names("objects"):
        raise ToolError(f"no object named {object!r}")
    dlg._refresh_openmm_object_combo()
    if dlg.cmb_openmm_object.findText(object) < 0:
        raise ToolError(f"{object!r} is not available to the OpenMM module")
    dlg.cmb_openmm_object.setCurrentText(object)
    idx = dlg.cmb_openmm_solvent_mode.findData(solvent)
    if idx >= 0:
        dlg.cmb_openmm_solvent_mode.setCurrentIndex(idx)
    dlg.spin_openmm_ph.setValue(float(ph))


def _openmm_job(ctx, kind, minimize_only, collect_result, description="", allow_unsafe=False):
    """Run Leonardo's OpenMM workflow as a job; ``collect_result(worker_result)`` builds the tool result.
    start_finalize_job restores the dialog's finalize method on every path (setup error, cancel)."""
    dlg = ctx.dlg

    def collect(call):
        res = call.arg(1, "result") if call is not None else None
        if res is None:
            raise ToolError("OpenMM did not finish (see Leonardo's message)")
        res = dict(res or {})
        if not res.get("ok", True) and res.get("error"):
            stopped = (res.get("health") or {}).get("stopped")
            extra = ""
            if res.get("health_stop_pdb"):
                extra += f" The frame where it stopped: {res['health_stop_pdb']}."
            if res.get("dcd_path") and stopped:
                extra += f" Trajectory up to there: {res['dcd_path']}."
            raise ToolError(str(res.get("error")) + extra)
        return collect_result(res)

    def launch():
        dlg._openmm_allow_unsafe = bool(allow_unsafe)
        try:
            return dlg._run_openmm_workflow(minimize_only=bool(minimize_only))
        finally:
            dlg._openmm_allow_unsafe = False

    return start_finalize_job(ctx, kind, launch, "_openmm_finalize_worker_result", collect,
                              busy_button="btn_openmm_run", description=description)
