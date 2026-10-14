"""Docking with smina and protein-ligand interaction analysis."""

from collections import Counter

from .. import presentation
from ..registry import ToolError
from . import _headless
from ._dialogjob import start_dialog_job


def _cmd():
    from pymol import cmd

    return cmd


def _residue_selection(obj, chain, resi, resn):
    """PyMOL selection of exactly one residue: chain-qualified ('chain ""' for a blank chain id) and
    with a negative residue number escaped ("resi -5" alone is the range up to 5). chain=None: any chain."""
    resi = str(resi).strip()
    parts = [f"({obj})", f"resn {str(resn).strip()}", "resi " + ("\\" + resi if resi.startswith("-") else resi)]
    if chain is not None:
        chain = str(chain).strip()
        parts.append(f"chain {chain}" if chain else 'chain ""')
    return " and ".join(parts)


def _ligand_residue(cmd, obj, chain="", resi="", resn=""):
    """(chain, resi, resn) of the ligand residue in ``obj`` matching the given fields; all empty when none
    is given (Leonardo then auto-detects the ligand of ``obj``). Leonardo's ligand resolver needs both
    resi and resn, else it auto-detects and the given field is ignored."""
    chain, resi, resn = str(chain or "").strip(), str(resi or "").strip(), str(resn or "").strip()
    if not resi and not resn:
        if chain:          # it would be dropped for an auto-detected ligand, possibly in another chain
            raise ToolError("ligand_chain alone does not name a residue: give ligand_resi or ligand_resn too")
        return "", "", ""
    parts = [f"({obj})"]
    if resn:
        parts.append(f"resn {resn}")
    if resi:
        parts.append("resi " + ("\\" + resi if resi.startswith("-") else resi))     # "-5" alone is a range
    if chain:
        parts.append(f"chain {chain}")
    sel = " and ".join(parts)
    found = []
    for extra in (" and hetatm", ""):
        space = {"found": found}
        cmd.iterate(sel + extra, "found.append((chain, resi, resn))", space=space)
        if found:
            break
    if not found:
        raise ToolError(f"no residue {resn or '*'} {chain}{resi} in {obj!r}")
    return tuple(str(v) for v in found[0])


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
                # always (re)set the fields: the previous analysis' ligand stays in them otherwise
                lig_c, lig_i, lig_n = _ligand_residue(cmd, object, ligand_chain, ligand_resi, ligand_resn)
                dlg.loaded_object = object
                dlg._traj_topology_object = object
                dlg.ligand_chain.setText(lig_c)
                dlg.ligand_resi.setText(lig_i)
                dlg.ligand_resn.setText(lig_n)
                label = f"{object} {lig_n} {lig_c}{lig_i}".rstrip()
            else:
                raise ToolError("give pose_rank or object (+ ligand residue)")
            dlg.interactions = []          # a handler that returns early must not leave the last complex's
            dlg._on_calc_interactions()
            interactions = list(getattr(dlg, "interactions", []) or [])
            if object and pose_rank is None and not lig_n:
                li = getattr(dlg, "ligand_info", None)       # the ligand Leonardo auto-detected
                if isinstance(li, (list, tuple)) and len(li) >= 3 and li[2]:
                    label = f"{object} {li[2]} {li[0]}{li[1]}"
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
            lig = f"({obj}) and not polymer and not resn HOH+WAT and not elem Ca+Mg+Zn+Na+Cl+K"
            if isinstance(li, (list, tuple)) and len(li) >= 3 and li[2]:
                # the analyzed residue only (another chain's copy or "resi -5" ranges must not join it)
                one = _residue_selection(obj, li[0], li[1], li[2])
                lig = one if cmd.count_atoms(one) else _residue_selection(obj, None, li[1], li[2])
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
