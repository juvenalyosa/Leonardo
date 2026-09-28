"""Redocking of a crystal ligand (pose recovery check)."""

from .. import presentation
from ..registry import ToolError
from . import _headless


def register(reg):
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
              }, "required": ["object"]},
              permission="compute_heavy", title="Redocking")
    def run_redocking(ctx, object, ligand_chain="", ligand_resi="", ligand_resn="", exhaustiveness=16, repeats=3,
                      seed=1001, ph=7.4):
        from pymol import cmd

        dlg = ctx.dlg
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        _headless.auto_hetatm_selection(dlg)
        with _headless.quiet() as q:
            dlg._refresh_redocking_object_combos()
            dlg.cmb_redock_reference_obj.setCurrentText(object)
            dlg.cmb_redock_wt_obj.setCurrentText(object)
            if ligand_resn:
                dlg.edit_redock_ref_chain.setText(str(ligand_chain))
                dlg.edit_redock_ref_resi.setText(str(ligand_resi))
                dlg.edit_redock_ref_resn.setText(str(ligand_resn))
            else:
                dlg._on_redock_autodetect_reference_ligand()
            dlg.cb_redock_run_mutant.setChecked(False)
            dlg.spin_redock_exhaust.setValue(int(exhaustiveness))
            dlg.spin_redock_repeats.setValue(int(repeats))
            dlg.spin_redock_seed0.setValue(int(seed))
            dlg.spin_redock_ph.setValue(float(ph))
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
        return {
            "_card": card,
            "object": object, "crystal_ligand": {"chain": ligand[0], "resi": ligand[1], "resn": ligand[2]},
            "top1_success": f"{success.get('n_success', 0)}/{success.get('n', 0)} repeats with RMSD <= 2 A",
            "repeats": reps,
            "shown_pose": {"mode": best.get("mode"), "score_kcal_mol": best.get("affinity"),
                           "rmsd_A": best.get("rmsd_ref"), "object": pose_obj},
            "summary_notes": [m["text"][:600] for m in q.messages if m["kind"] == "information"][-1:],
            "scene": ctx.scene(f"redocking {ligand[2]}: top pose vs crystal"),
        }
