"""Redocking of a crystal ligand (pose recovery check)."""

from .. import presentation
from ..registry import ToolError
from . import _headless


def register(reg):
    register_mc(reg)

    @reg.tool("run_redocking", "Redock the crystal ligand of a complex into its own receptor and measure pose "
              "recovery: symmetry-corrected heavy-atom RMSD of the top-scored pose to the crystal pose (success "
              "= RMSD <= 2 A) over repeats. The ligand is auto-detected unless given. Leonardo runs this on the "
              "main thread: the window does not refresh until it finishes (about 1-5 min).",
              {"properties": {
                  "object": {"type": "string"},
                  "ligand_chain": {"type": "string"}, "ligand_resi": {"type": "string"},
                  "ligand_resn": {"type": "string"},
                  "exhaustiveness": {"type": "integer", "minimum": 1, "maximum": 64},
                  "repeats": {"type": "integer", "minimum": 1, "maximum": 10},
                  "seed": {"type": "integer", "minimum": 1},
                  "ph": {"type": "number", "minimum": 0, "maximum": 14},
                  "mutant_object": {"type": "string", "description": "also redock into this mutant receptor "
                                                                     "(e.g. from model_mutant) and compare"},
                  "ensemble_structures": {"type": "integer", "minimum": 0, "maximum": 200,
                                          "description": "receptor ensemble: N (2-200) binding-site conformations "
                                                         "sampled by Monte Carlo from the best model (0 = rigid "
                                                         "single receptor)"},
                  "ensemble_temperature_K": {"type": "number", "minimum": 100, "maximum": 2000},
                  "ensemble_site_radius_A": {"type": "number", "minimum": 3, "maximum": 15},
              }, "required": ["object"]},
              permission="compute_heavy", title="Redocking")
    def run_redocking(ctx, object, ligand_chain="", ligand_resi="", ligand_resn="", exhaustiveness=16, repeats=3,
                      seed=1001, ph=7.4, mutant_object="", ensemble_structures=0, ensemble_temperature_K=300.0,
                      ensemble_site_radius_A=6.0):
        from pymol import cmd

        dlg = ctx.dlg
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        _headless.auto_hetatm_selection(dlg)
        with _headless.quiet() as q:
            dlg._refresh_redocking_object_combos()
            dlg.cmb_redock_reference_obj.setCurrentText(object)
            dlg.cmb_redock_wt_obj.setCurrentText(object)
            if (str(dlg.cmb_redock_reference_obj.currentText() or "") != object
                    or str(dlg.cmb_redock_wt_obj.currentText() or "") != object):
                raise ToolError(f"{object!r} is not offered for redocking (Leonardo lists protein objects "
                                "only, not generated complexes, poses or redock_* objects)")
            if ligand_resn:
                dlg.edit_redock_ref_chain.setText(str(ligand_chain))
                dlg.edit_redock_ref_resi.setText(str(ligand_resi))
                dlg.edit_redock_ref_resn.setText(str(ligand_resn))
            else:
                dlg._on_redock_autodetect_reference_ligand()
            if mutant_object:
                if mutant_object not in cmd.get_names("objects"):
                    raise ToolError(f"no object named {mutant_object!r}")
                dlg.cb_redock_run_mutant.setChecked(True)
                dlg.cmb_redock_mut_obj.setCurrentText(mutant_object)
                if str(dlg.cmb_redock_mut_obj.currentText() or "") != mutant_object:
                    raise ToolError(f"{mutant_object!r} is not offered as the mutant receptor")
            else:
                dlg.cb_redock_run_mutant.setChecked(False)
            dlg.spin_redock_exhaust.setValue(int(exhaustiveness))
            dlg.spin_redock_repeats.setValue(int(repeats))
            dlg.spin_redock_seed0.setValue(int(seed))
            dlg.spin_redock_ph.setValue(float(ph))
            if hasattr(dlg, "cb_redock_use_mc_ensemble"):
                dlg.cb_redock_use_mc_ensemble.setChecked(int(ensemble_structures) > 0)
                if int(ensemble_structures) == 1:
                    raise ToolError("an ensemble needs at least 2 structures (0 = rigid single receptor)")
                if int(ensemble_structures) > 0:
                    dlg.spin_redock_mc_n.setValue(int(ensemble_structures))
                    dlg.spin_redock_mc_temp.setValue(float(ensemble_temperature_K))
                    dlg.spin_redock_mc_radius.setValue(float(ensemble_site_radius_A))
            dlg._redock_last_mc_ensemble = {}
            dlg._redock_last_runs = {}
            dlg._on_run_redocking_workflow()
        run = dict((getattr(dlg, "_redock_last_runs", {}) or {}).get("WT", {}) or {})
        if not run.get("ok"):
            errors = [m["text"] for m in q.messages if m["kind"] in ("warning", "critical")]
            raise ToolError("; ".join(errors) or "redocking produced no result")
        best = dict(run.get("best") or {})
        reps = [{"repeat": r.get("repeat"), "seed": r.get("seed"), "top1_score_kcal_mol": r.get("best_affinity"),
                 "top1_rmsd_A": r.get("best_rmsd"), "lowest_rmsd_all_modes_A": r.get("min_rmsd_all_poses")}
                for r in run.get("repeat_rows", [])]
        ligand = (dlg.edit_redock_ref_chain.text(), dlg.edit_redock_ref_resi.text(), dlg.edit_redock_ref_resn.text())
        pose_obj = str(best.get("pose_object", "") or "")
        if pose_obj and pose_obj in cmd.get_names("objects"):
            presentation.style_pose(pose_obj, 0)
            cmd.frame(int(best.get("state", 1) or 1))
            crystal = f"({object}) and resn {ligand[2]} and resi {ligand[1]}"
            presentation.style_ligand(crystal)
            presentation.focus(f"({pose_obj}) or ({crystal})")
        success = dict(run.get("top1_success") or {})
        rms = [r["top1_rmsd_A"] for r in reps if r.get("top1_rmsd_A") is not None]
        card = (f"Redocking of **{ligand[2]}**: top-1 success **{success.get('n_success', 0)}/{success.get('n', 0)}** "
                f"(RMSD ≤ 2 Å); RMSD top-1 " + ", ".join(f"{float(v):.2f}" for v in rms) + " Å; score "
                f"{float(best.get('affinity', 0) or 0):.2f} kcal/mol.")
        comparison = []
        if mutant_object:
            import sys as _sys

            module = _sys.modules[type(dlg).__module__]
            try:
                from ._widgets import jsonable

                comparison = jsonable(list(module.redock_build_comparison_rows(dlg._redock_last_runs) or []))[:10]
            except Exception:
                comparison = []
            mut = next((v for k, v in (dlg._redock_last_runs or {}).items() if k != "WT"), {}) or {}
            mb = dict(mut.get("best") or {})
            if mb:
                card += (f" Mutant `{mutant_object}`: score {float(mb.get('affinity', 0) or 0):.2f} kcal/mol "
                         f"(WT {float(best.get('affinity', 0) or 0):.2f}).")
        ens = dict(getattr(dlg, "_redock_last_mc_ensemble", {}) or {})
        ensemble = {}
        if ens:
            ensemble = {"structures": len(ens.get("members") or []), "acceptance": ens.get("acceptance"),
                        "temperature_K": ens.get("temperature_k"), "site_residues": ens.get("site_residues"),
                        "members": [{k: m.get(k) for k in ("relative_energy_kj_mol", "site_rmsd_to_start_A")}
                                    for m in ens.get("members") or []]}
            card += (f" Receptor ensemble: best model + {ensemble['structures'] - 1} Monte Carlo conformations "
                     f"({ens.get('temperature_k'):.0f} K).")
        return {
            "_card": card,
            "receptor_ensemble": ensemble,
            "mutant_comparison": comparison,
            "object": object, "crystal_ligand": {"chain": ligand[0], "resi": ligand[1], "resn": ligand[2]},
            "top1_success": f"{success.get('n_success', 0)}/{success.get('n', 0)} repeats with RMSD <= 2 A",
            "repeats": reps,
            "shown_pose": {"mode": best.get("mode"), "score_kcal_mol": best.get("affinity"),
                           "rmsd_A": best.get("rmsd_ref"), "object": pose_obj},
            "summary_notes": [m["text"][:600] for m in q.messages if m["kind"] == "information"][-1:],
            "scene": ctx.scene(f"redocking {ligand[2]}: top pose vs crystal"),
        }


def register_mc(reg):
    @reg.tool("generate_mc_ensemble", "Receptor conformational ensemble by Monte Carlo: starting from one "
              "structure (the best model), binding-site side chains around a ligand/selection are sampled by "
              "Monte Carlo with minimization (chi-angle moves, Amber14/GB-OBC2, Metropolis at the temperature; "
              "backbone and the rest of the protein fixed). Loads the ensemble as a multi-state object; used by "
              "ensemble redocking (run_redocking ensemble_structures).",
              {"properties": {"object": {"type": "string"},
                              "site_selection": {"type": "string", "description": "ligand or residues that define "
                                                                                  "the binding site"},
                              "n_structures": {"type": "integer", "minimum": 1, "maximum": 200},
                              "temperature_K": {"type": "number", "minimum": 100, "maximum": 2000},
                              "radius_A": {"type": "number", "minimum": 3, "maximum": 15},
                              "seed": {"type": "integer", "minimum": 1}},
               "required": ["object", "site_selection"]},
              permission="compute_heavy", title="Monte Carlo ensemble")
    def generate_mc_ensemble(ctx, object, site_selection, n_structures=10, temperature_K=300.0, radius_A=6.0, seed=1):
        import json
        import os
        import subprocess
        import sys
        import time

        from pymol import cmd

        from ... import mc_ensemble_runtime as mc
        from .estimates import _profile

        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        if cmd.count_atoms(site_selection) == 0:
            raise ToolError(f"no atoms in {site_selection!r}")
        space = {"site": set()}
        cmd.iterate(f"byres (({object}) and polymer and not hydro within {float(radius_A)} of ({site_selection}))",
                    "site.add((chain, resi))", space=space)
        site = sorted(space["site"])
        if not site:
            raise ToolError("no protein residues around that selection")
        work = os.path.join(ctx.workspace("mc_ensembles"), time.strftime(f"{object}_%Y%m%d_%H%M%S"))
        os.makedirs(work, exist_ok=True)
        pdb_in = os.path.join(work, "input.pdb")
        cmd.save(pdb_in, f"({object}) and polymer", state=1)
        platform = str((_profile(ctx).get("md_platform") or {}).get("name") or "CPU")
        args = {"pdb_in": pdb_in, "out_dir": work, "n_structures": int(n_structures),
                "site_residues": [list(r) for r in site], "temperature_k": float(temperature_K),
                "steps_per_structure": 15, "seed": int(seed), "platform": platform}
        with open(os.path.join(work, "mc_args.json"), "w") as fh:
            json.dump(args, fh)
        py = os.path.join(sys.prefix, "bin", "python")
        log = open(os.path.join(work, "mc_log.txt"), "w")
        proc = subprocess.Popen([py if os.path.isfile(py) else sys.executable, mc.__file__, "mc_args.json"],
                                cwd=work, stdout=log, stderr=subprocess.STDOUT)

        def poll():
            if proc.poll() is None:
                try:
                    lines = [ln for ln in open(log.name) if ln.startswith("PROGRESS")]
                    return {"done": False, "progress": lines[-1].strip()[9:] if lines else "sampling"}
                except Exception:
                    return {"done": False}
            log.close()
            res_path = os.path.join(work, "mc_ensemble.json")
            if proc.returncode != 0 or not os.path.isfile(res_path):
                return {"done": True, "error": "Monte Carlo sampling failed: " + open(log.name).read()[-800:]}
            res = json.load(open(res_path))
            if not res.get("ok"):
                return {"done": True, "error": res.get("error", "no structure sampled")}
            name = f"mc_ensemble_{object}"[:60]
            if name in cmd.get_names("objects"):
                cmd.delete(name)
            for k, m in enumerate(res["structures"], start=1):
                cmd.load(m["path"], name, state=k)
            cmd.align(name, f"({object}) and polymer")
            presentation.style_protein(name)
            cmd.show("sticks", f"({name}) and byres (polymer within 0.1 of ({name}) and ("
                               + " or ".join(f"(chain {c} and resi {r})" if c else f"resi {r}" for c, r in site[:40])
                               + ")) and not name N+C+O and not hydro")
            presentation.focus(site_selection, buffer=8.0)
            cmd.mplay()
            ctx.__dict__.setdefault("mc_ensembles", {})[object] = res
            rms = [s["site_rmsd_to_start_A"] for s in res["structures"]]
            return {"done": True, "result": {
                "object": name, "structures": len(res["structures"]), "acceptance": res["acceptance"],
                "site_residues": res["site_residues"], "platform": res["platform"], "seconds": res["seconds"],
                "members": [{k: s[k] for k in ("relative_energy_kj_mol", "site_rmsd_to_start_A", "path")}
                            for s in res["structures"]],
                "method": res["method"], "scene": ctx.scene(f"Monte Carlo ensemble of {object}"),
                "_card": (f"**{len(res['structures'])} conformations** of the {object} site ({len(site)} residues) "
                          f"at {temperature_K:.0f} K on {res['platform']} in {res['seconds']} s; acceptance "
                          f"{res['acceptance']:.0%}; site RMSD {min(rms):.2f}-{max(rms):.2f} Å (play the states).")}}

        def cancel():
            if proc.poll() is None:
                proc.kill()

        return ctx.jobs.start("mc_ensemble", poll, cancel=cancel, description=f"Monte Carlo ensemble of {object}")
