"""Docking with smina and protein-ligand interaction analysis."""

from collections import Counter

from .. import presentation
from ..registry import ToolError
from . import _headless
from ._dialogjob import start_dialog_job


def _cmd():
    from pymol import cmd

    return cmd


def register(reg):
    @reg.tool("run_docking", "Dock the prepared ligand (prepare_docking_ligand) into the selected pocket "
              "(select_pocket) with smina. Runs in the background (about 1-10 min per repeat). Returns the "
              "ranked poses (Vina-like score in kcal/mol, which is not a binding free energy) and loads them "
              "in the viewer.",
              {"properties": {
                  "exhaustiveness": {"type": "integer", "minimum": 1, "maximum": 64},
                  "num_modes": {"type": "integer", "minimum": 1, "maximum": 40},
                  "repeats": {"type": "integer", "minimum": 1, "maximum": 10},
                  "seed": {"type": "integer", "minimum": 0, "description": "0 = no fixed seed"},
                  "ph": {"type": "number", "minimum": 0, "maximum": 14},
                  "keep_hetero": {"type": "array", "items": {"type": "string"},
                                  "description": "Residue names of hetero groups to keep in the receptor besides "
                                                 "ions and known cofactors (the crystal ligand is removed)."},
              }},
              permission="compute_heavy", title="Docking (smina)")
    def run_docking(ctx, exhaustiveness=16, num_modes=9, repeats=1, seed=1001, ph=None, keep_hetero=None):
        dlg = ctx.dlg
        if not list(getattr(dlg, "_fpocket_pockets", []) or []):
            raise ToolError("no pocket selected: run find_pockets and select_pocket first")
        if not str(getattr(dlg, "_docking_ligand_file", "") or ""):
            raise ToolError("no docking ligand: run prepare_docking_ligand first")
        _headless.auto_hetatm_selection(dlg)
        _headless.set_keep_hetero(keep_hetero or [])
        try:
            dlg._dock_hetatm_selection_cache.pop(getattr(dlg, "_docking_receptor_object", None), None)
        except Exception:
            pass
        dlg.spin_exhaustiveness.setValue(int(exhaustiveness))
        dlg.spin_num_modes.setValue(int(num_modes))
        dlg.spin_dock_repeats.setValue(int(repeats))
        dlg.spin_dock_seed0.setValue(int(seed))
        if ph is not None:
            dlg.spin_docking_ph.setValue(float(ph))
        previous = getattr(dlg, "_docking_pose_object", None)

        def collect():
            pose_obj = getattr(dlg, "_docking_pose_object", None)
            scores = list(getattr(dlg, "_docking_scores", []) or [])
            if not pose_obj or pose_obj == previous or not scores:
                raise ToolError("docking did not produce poses")
            stats = dict(getattr(dlg, "_docking_last_stats", {}) or {})
            cmd = _cmd()
            presentation.declutter(keep=(pose_obj,))
            presentation.style_ligand(pose_obj)
            presentation.focus(pose_obj)
            cmd.frame(1)
            poses = [{"mode": int(p.get("mode", i + 1)), "score_kcal_mol": float(p.get("affinity")),
                      "rmsd_lb": p.get("rmsd_lb"), "rmsd_ub": p.get("rmsd_ub")}
                     for i, p in enumerate(scores[: int(num_modes)])]
            rep = dict(stats.get("repeat_stats", {}) or {})
            return {
                "pose_object": pose_obj, "receptor": getattr(dlg, "_docking_receptor_object", ""),
                "poses": poses, "best_score_kcal_mol": poses[0]["score_kcal_mol"] if poses else None,
                "repeats": {k: rep.get(k) for k in ("n", "mean", "std", "ci_low", "ci_high", "confidence")},
                "job_dir": stats.get("job_dir", ""),
                "scene": ctx.scene(f"docking poses ({pose_obj}), best {poses[0]['score_kcal_mol']:.1f} kcal/mol"),
                "_card": (f"**{len(poses)} poses**; best score **{poses[0]['score_kcal_mol']:.2f} kcal/mol** "
                          "(smina score, not a binding ΔG). "
                          + ", ".join(f"#{p['mode']}: {p['score_kcal_mol']:.1f}" for p in poses[1:5])),
            }

        return start_dialog_job(ctx, "docking", dlg._on_run_smina_docking,
                                lambda: not bool(getattr(dlg, "_smina_docking_busy", False)), collect,
                                description="smina docking")

    @reg.tool("analyze_interactions", "Protein-ligand interactions (hydrogen bonds, salt bridges, pi stacking, "
              "cation-pi, hydrophobic, halogen and metal contacts) of a docking pose (pose_rank from run_docking) "
              "or of a complex already in the session (object + ligand residue). Draws them in the viewer.",
              {"properties": {
                  "pose_rank": {"type": "integer", "minimum": 1},
                  "object": {"type": "string"},
                  "ligand_chain": {"type": "string"}, "ligand_resi": {"type": "string"},
                  "ligand_resn": {"type": "string"},
              }},
              permission="compute_light", title="Protein-ligand interactions")
    def analyze_interactions(ctx, pose_rank=None, object="", ligand_chain="", ligand_resi="", ligand_resn=""):
        dlg = ctx.dlg
        with _headless.quiet() as q:
            if pose_rank is not None:
                if not getattr(dlg, "_docking_pose_object", None):
                    raise ToolError("no docking poses; run run_docking first")
                if int(pose_rank) > dlg.tbl_dock_poses.rowCount():
                    raise ToolError(f"only {dlg.tbl_dock_poses.rowCount()} poses")
                dlg.tbl_dock_poses.selectRow(int(pose_rank) - 1)
                complex_obj, mode = dlg._prepare_selected_pose_complex()
                if not complex_obj:
                    raise ToolError("could not build the pose complex")
                label = f"pose {pose_rank} ({complex_obj})"
            elif object:
                cmd = _cmd()
                if object not in cmd.get_names("objects"):
                    raise ToolError(f"no object named {object!r}")
                dlg.loaded_object = object
                dlg._traj_topology_object = object
                if ligand_resn:
                    dlg.ligand_chain.setText(str(ligand_chain))
                    dlg.ligand_resi.setText(str(ligand_resi))
                    dlg.ligand_resn.setText(str(ligand_resn))
                label = f"{object} {ligand_resn}{ligand_resi}"
            else:
                raise ToolError("give pose_rank or object (+ ligand residue)")
            dlg._on_calc_interactions()
            interactions = list(getattr(dlg, "interactions", []) or [])
            try:
                dlg._on_redraw_visible_types()
            except Exception:
                pass
        if not interactions:
            errors = [m["text"] for m in q.messages if m["kind"] in ("warning", "critical")]
            raise ToolError("; ".join(errors) or "no interactions found")
        counts = Counter(str(i.get("type", "")) for i in interactions)
        top = sorted(interactions, key=lambda i: float(i.get("distance", 99) or 99))[:15]
        try:
            cmd = _cmd()
            obj = str(dlg.loaded_object or "")
            li = getattr(dlg, "ligand_info", None)     # Leonardo keeps (chain, resi, resn)
            if isinstance(li, dict):
                li = (li.get("chain", ""), li.get("resi", ""), li.get("resn", ""))
            lig = (f"({obj}) and resn {li[2]} and resi {li[1]}" if isinstance(li, (list, tuple)) and len(li) >= 3
                   and li[2] else f"({obj}) and not polymer and not resn HOH+WAT and not elem Ca+Mg+Zn+Na+Cl+K")
            if cmd.count_atoms(lig):
                presentation.declutter(keep=[obj] + [n for n in cmd.get_names("objects") if n.startswith(("int", "lbl_"))])
                if obj.startswith("dock_complex"):
                    receptor = str(getattr(dlg, "_docking_receptor_object", "") or "")
                    if receptor:
                        cmd.disable(receptor)      # the complex already holds the receptor
                presentation.style_ligand(lig)
                presentation.show_pocket_residues(obj, f"byres (({obj}) and polymer within 4.0 of ({lig}))",
                                                  name="ai_contacts")
                cmd.label("ai_contacts and name CA", '"%s%s" % (resn, resi)')
                presentation.focus(lig, buffer=8.0)
        except Exception as exc:
            q.messages.append({"kind": "warning", "title": "view", "text": f"view setup: {exc}"})
        card = (f"**{len(interactions)} interactions** in {label}: "
                + ", ".join(f"{n} {t}" for t, n in counts.most_common()) + ". Closest: "
                + "; ".join(f"{i.get('type')} {i.get('prot_res')} {float(i.get('distance', 0)):.1f} Å" for i in top[:4]))
        return {
            "_card": card,
            "complex": label, "total": len(interactions), "by_type": dict(counts),
            "closest": [{"type": i.get("type"), "residue": i.get("prot_res"), "protein_atom": i.get("prot_atom"),
                         "ligand_atom": i.get("lig_atom"), "distance_A": round(float(i.get("distance", 0)), 2)}
                        for i in top],
            "scene": ctx.scene(f"interactions: {label}"),
            "leonardo_messages": [m for m in q.messages if m["kind"] != "information"][:5],
        }
