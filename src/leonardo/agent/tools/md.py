"""Short classical molecular dynamics with OpenMM."""

from .. import presentation
from ..registry import ToolError
from ._dialogjob import start_dialog_job


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
                  "equilibration_ns": {"type": "number", "minimum": 0.001, "maximum": 10,
                                       "description": "Default: 20% of production (min 4 ps, max 40 ps); heating "
                                                      "10% (min 2 ps, max 10 ps)."},
              }, "required": ["object"]},
              permission="compute_heavy", title="Molecular dynamics (OpenMM)")
    def run_md(ctx, object, production_ns=0.1, solvent="implicit", temperature_K=300.0, ph=7.4,
               equilibration_ns=None):
        from pymol import cmd

        dlg = ctx.dlg
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        dlg._refresh_openmm_object_combo()
        if dlg.cmb_openmm_object.findText(object) < 0:
            raise ToolError(f"{object!r} is not available to the OpenMM module")
        dlg.cmb_openmm_object.setCurrentText(object)
        idx = dlg.cmb_openmm_solvent_mode.findData(solvent)
        if idx >= 0:
            dlg.cmb_openmm_solvent_mode.setCurrentIndex(idx)
        dlg.spin_openmm_temperature.setValue(float(temperature_K))
        dlg.spin_openmm_ph.setValue(float(ph))
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

        captured = {}
        original = dlg._openmm_finalize_worker_result

        def finalize(launch, result):
            try:
                return original(launch, result)
            finally:
                captured["result"] = dict(result or {})
                dlg._openmm_finalize_worker_result = original

        dlg._openmm_finalize_worker_result = finalize

        def start():
            before = set(getattr(dlg, "_task_monitor_state", {}) or {})
            dlg._run_openmm_workflow(minimize_only=False)
            after = set(getattr(dlg, "_task_monitor_state", {}) or {})
            captured["submitted"] = bool(after - before)

        def is_done():
            if "result" in captured:
                return True
            # nothing was submitted (setup error): finished immediately
            return not captured.get("submitted", True)

        def collect():
            dlg._openmm_finalize_worker_result = original
            res = captured.get("result")
            if res is None:
                raise ToolError("OpenMM did not start (see Leonardo's message)")
            if not res.get("ok", True) and res.get("error"):
                raise ToolError(str(res.get("error")))
            out_obj = str(getattr(dlg, "_openmm_last_output_object", "") or "")
            info = {"object": out_obj, "production_ns": production_ns, "solvent": solvent,
                    "stages_ps": {"heating": round(heat_ps, 1), "equilibration": round(equil_ps, 1),
                                  "production": round(float(production_ns) * 1000.0, 1)},
                    "force_field": res.get("ff_used_label"), "platform": res.get("platform_used"),
                    "trajectory": res.get("dcd_path"), "energies_csv": res.get("csv_path"),
                    "job_dir": res.get("job_dir"), "notes": list(res.get("notes", []) or [])[:6]}
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
            return info

        return start_dialog_job(ctx, "md", start, is_done, collect, description=f"OpenMM MD of {object}")
