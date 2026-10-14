"""Redocking of a crystal ligand (pose recovery check)."""

import contextlib
import csv
import math
import os
import time

from .. import presentation
from ..jobs import POPEN_GROUP, stop_process
from ..registry import ToolError
from . import _headless
from .docking import _ligand_residue, _residue_selection

_KEEP_HETERO = {"type": "array", "items": {"type": "string"},
                "description": "Residue names of receptor hetero groups to keep besides ions and known cofactors "
                               "(answers Leonardo's HETATM chooser; the crystal ligand is always removed)."}
_XYZ = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}


def _num(x, nd=3):
    try:
        v = float(x)
        return round(v, nd) if math.isfinite(v) else None
    except Exception:
        return None


def _apply_grid_box(dlg, receptor, padding=None, size=None):
    """The GUI's grid controls: padding spin box and the "Adjust grid box" manual size delta
    (stored per crystal-ligand reference, relative to the auto box)."""
    if padding is not None:
        from ._widgets import set_value

        set_value(dlg, "spin_redock_padding", float(padding))
    fields = dlg._redock_get_reference_fields()
    if size is not None:
        payload = dict(dlg._redock_reference_grid_payload(receptor, fields) or {})
        base = [float(v) for v in list(payload.get("base_size") or [])[:3]]
        if not payload.get("ok") or len(base) < 3:
            raise ToolError("cannot build the redocking grid box from the crystal ligand "
                            f"({payload.get('reason', 'no reference ligand')})")
        dlg._redock_store_grid_size_delta(receptor, fields, [float(size[i]) - base[i] for i in range(3)])
    try:
        dlg._update_redock_grid_box_label()
    except Exception:
        pass


def _grid_box_info(dlg, receptor, center_override=None):
    payload = dict(dlg._redock_reference_grid_payload(receptor, dlg._redock_get_reference_fields()) or {})
    if not payload.get("ok"):
        return {}
    center = center_override if center_override is not None else payload.get("center")
    return {"center_A": [_num(v) for v in list(center)[:3]],
            "size_A": [_num(v) for v in payload.get("size") or []],
            "auto_size_A": [_num(v) for v in payload.get("base_size") or []],
            "manual_size_delta_A": [_num(v) for v in payload.get("size_delta") or []],
            "center_source": "manual" if center_override is not None else "crystal ligand"}


@contextlib.contextmanager
def _center_override(dlg, center):
    """Leonardo centers the redocking box on the crystal ligand (no GUI control for the center):
    a manual center replaces it for this run only."""
    if center is None:
        yield
        return
    import numpy as np

    original = dlg._redock_grid_from_reference
    own = dlg.__dict__.get("_redock_grid_from_reference")

    def grid(receptor_obj, ref_fields):
        c, size, nref = original(receptor_obj, ref_fields)
        if c is None:
            return c, size, nref
        return np.asarray([float(v) for v in center], dtype=float), size, nref
    dlg._redock_grid_from_reference = grid
    try:
        yield
    finally:
        if own is None:
            dlg.__dict__.pop("_redock_grid_from_reference", None)
        else:
            dlg._redock_grid_from_reference = own


def _pose_rows(dlg):
    """Leonardo's results table: the top-scored pose of each system (WT / mutant) and ensemble member."""
    return [dict(r or {}) for r in list(getattr(dlg, "_redock_results", []) or []) if isinstance(r, dict)]


def _member(r):
    return str(r.get("system_member", "") or "").strip() or f"{r.get('system', '')}_model_001"


def _score_key(r):
    a, m = _num(r.get("affinity")), _num(r.get("mode"))
    return (a if a is not None else float("inf"), m if m is not None else float("inf"))


def _ranked_poses(dlg):
    """Every redocked pose that has a pose object (all modes of every repeat), ranked by score within
    its (system, member); rank 1 is the table row Leonardo shows. Each item keeps the raw row and
    the table row index of its (system, member)."""
    table = _pose_rows(dlg)
    table_idx = {(str(r.get("system", "")), _member(r)): i for i, r in enumerate(table)}
    raw = [dict(r) for r in list(getattr(dlg, "_redock_raw_results", []) or [])
           if isinstance(r, dict) and str(r.get("pose_object", "") or "")]
    groups = {}
    for r in raw:
        groups.setdefault((str(r.get("system", "")), _member(r)), []).append(r)
    out = []
    for i, trow in enumerate(table):
        key = (str(trow.get("system", "")), _member(trow))
        rest = sorted(groups.get(key, []), key=_score_key)
        top = dict(trow)
        # the table row is rank 1; the other rows of the group follow (the same pose is not listed twice)
        same = [r for r in rest if (r.get("pose_object"), r.get("state"), r.get("mode")) ==
                (top.get("pose_object"), top.get("state"), top.get("mode"))]
        if same:
            rest.remove(same[0])
        for rank, r in enumerate([top] + rest, start=1):
            out.append({"system": key[0], "member": key[1], "rank": rank, "row": r, "table_index": table_idx[key]})
    return out


def _pose_summary(item):
    r = item["row"]
    return {"system": item["system"], "member": item["member"], "rank": item["rank"], "mode": r.get("mode"),
            "repeat": r.get("repeat"), "score_kcal_mol": _num(r.get("affinity")), "rmsd_A": _num(r.get("rmsd_ref")),
            "pose_object": str(r.get("pose_object", "") or ""), "state": r.get("state")}


def _pick_pose(items, rank, system="", member=""):
    """(item, system label) for a per-system/member rank; system "" = WT (else the first), "mutant" =
    the first non-WT system; member "" = the best-scoring member of the system."""
    systems = []
    for it in items:
        if it["system"] not in systems:
            systems.append(it["system"])
    if not systems:
        raise ToolError("no redocking poses: run run_redocking first")
    want = str(system or "").strip()
    if not want:
        label = "WT" if "WT" in systems else systems[0]
    elif want.lower() in ("mut", "mutant") and not any(s.lower() == want.lower() for s in systems):
        others = [s for s in systems if s != "WT"]
        if not others:
            raise ToolError(f"no mutant redocking results; systems: {systems}")
        label = others[0]
    else:
        match = [s for s in systems if s.lower() == want.lower()] or [s for s in systems if want.lower() in s.lower()]
        if not match:
            raise ToolError(f"no redocking system {system!r}; choose one of {systems}")
        label = match[0]
    cands = [it for it in items if it["system"] == label]
    members = []
    for it in cands:
        if it["member"] not in members:
            members.append(it["member"])
    if member:
        pick = [m for m in members if m.lower() == str(member).strip().lower()]
        if not pick:
            raise ToolError(f"no ensemble member {member!r} in {label}; members: {members}")
    else:
        pick = [min((it for it in cands if it["rank"] == 1), key=lambda it: _score_key(it["row"]))["member"]]
    cands = [it for it in cands if it["member"] == pick[0]]
    hit = [it for it in cands if it["rank"] == int(rank)]
    if not hit:
        raise ToolError(f"{label} {pick[0]} has ranks 1..{max(it['rank'] for it in cands)}")
    return hit[0], label


@contextlib.contextmanager
def _as_selected_row(dlg, item):
    """Leonardo's pose handlers act on the selected results-table row. Rank 1 is that row; another
    mode of the same system/member is shown to the handlers as the selected row for this call only
    (same receptor frame, that pose's object and state)."""
    table = dlg.tbl_redock_results
    table.selectRow(int(item["table_index"]))          # the table's selection handler previews it
    if item["rank"] == 1:
        yield
        return
    rows = _pose_rows(dlg)
    trow = rows[int(item["table_index"])]
    row = dict(item["row"], rank=item["rank"], system_member=item["member"],
               system_display=trow.get("system_display", item["system"]), traj_state=trow.get("traj_state", 1))

    class _OneRow:
        def currentRow(self):
            return 0

    saved = dlg._redock_results
    dlg._redock_results = [row]
    dlg.tbl_redock_results = _OneRow()
    try:
        yield
    finally:
        dlg.tbl_redock_results = table
        dlg._redock_results = saved


@contextlib.contextmanager
def _save_dialog_answer(path):
    """Answer QFileDialog.getSaveFileName with ``path`` (Leonardo's export handlers ask for a file)."""
    from pymol.Qt import QtWidgets

    QFD = QtWidgets.QFileDialog
    original = QFD.__dict__.get("getSaveFileName", QFD.getSaveFileName)
    QFD.getSaveFileName = staticmethod(lambda *a, **k: (str(path), "CSV (*.csv)"))
    try:
        yield
    finally:
        QFD.getSaveFileName = original


def register(reg):
    register_mc(reg)
    register_poses(reg)

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
                  "ensemble_keep_ligand_space": {"type": "boolean", "description": (
                      "holo-like ensemble: keep the crystal ligand's volume (soft) while sampling, so side chains "
                      "do not close the pocket; better for redocking that ligand, biased toward it")},
                  "ensemble_seed": {"type": "integer", "minimum": 1,
                                    "description": "Monte Carlo seed for the receptor ensemble (default 1)"},
                  "ensemble_reproducible": {"type": "boolean", "description": (
                      "sample the ensemble on one CPU thread so the same seed gives the identical ensemble "
                      "(slower; for audits and replay packs)")},
                  "keep_hetero": _KEEP_HETERO,
                  "grid_size_A": dict(_XYZ, description="manual box edges x,y,z in A (2-120), as the GUI's "
                                                       "'Adjust grid box'; default: auto from the crystal ligand "
                                                       "plus padding (a size set earlier for the same ligand stays)"),
                  "grid_padding_A": {"type": "number", "minimum": 2, "maximum": 20,
                                     "description": "padding added on each side of the crystal ligand (auto box)"},
                  "grid_center_A": dict(_XYZ, description="manual box center x,y,z in A; default: centre of the "
                                                         "crystal ligand (Leonardo's GUI has no center control)"),
              }, "required": ["object"]},
              permission="compute_heavy", title="Redocking")
    def run_redocking(ctx, object, ligand_chain="", ligand_resi="", ligand_resn="", exhaustiveness=16, repeats=3,
                      seed=1001, ph=7.4, mutant_object="", ensemble_structures=0, ensemble_temperature_K=300.0,
                      ensemble_site_radius_A=6.0, keep_hetero=None, grid_size_A=None, grid_padding_A=None,
                      ensemble_keep_ligand_space=False, ensemble_seed=1, ensemble_reproducible=False,
                      grid_center_A=None):
        from pymol import cmd

        dlg = ctx.dlg
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        if grid_size_A is not None and not all(2.0 <= float(v) <= 120.0 for v in grid_size_A):
            raise ToolError("grid_size_A edges must be 2-120 A")
        _headless.auto_hetatm_selection(dlg)
        for rec in (object, mutant_object):
            try:
                dlg._dock_hetatm_selection_cache.pop(rec, None)
            except Exception:
                pass
        _headless.set_keep_hetero(keep_hetero or [])
        with _headless.quiet() as q, _center_override(dlg, grid_center_A):
            dlg._refresh_redocking_object_combos()
            dlg.cmb_redock_reference_obj.setCurrentText(object)
            dlg.cmb_redock_wt_obj.setCurrentText(object)
            if (str(dlg.cmb_redock_reference_obj.currentText() or "") != object
                    or str(dlg.cmb_redock_wt_obj.currentText() or "") != object):
                raise ToolError(f"{object!r} is not offered for redocking (Leonardo lists protein objects "
                                "only, not generated complexes, poses or redock_* objects)")
            if str(ligand_resn or "").strip() or str(ligand_resi or "").strip():
                # resolve a partial residue (resi alone, resn alone): Leonardo needs both resi and resn,
                # and with resn blank the given residue was ignored for an auto-detected one
                lig_c, lig_i, lig_n = _ligand_residue(cmd, object, ligand_chain, ligand_resi, ligand_resn)
                dlg.edit_redock_ref_chain.setText(lig_c)
                dlg.edit_redock_ref_resi.setText(lig_i)
                dlg.edit_redock_ref_resn.setText(lig_n)
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
                    if hasattr(dlg, "cb_redock_mc_keep_ligand_space"):
                        dlg.cb_redock_mc_keep_ligand_space.setChecked(bool(ensemble_keep_ligand_space))
            dlg._redock_mc_seed = int(ensemble_seed)
            dlg._redock_mc_reproducible = bool(ensemble_reproducible)
            _apply_grid_box(dlg, object, grid_padding_A, grid_size_A)
            grid_box = _grid_box_info(dlg, object, grid_center_A)
            dlg._redock_last_mc_ensemble = {}
            dlg._redock_last_mc_ensemble_note = ""
            dlg._redock_last_runs = {}
            dlg._redock_results = []
            dlg._redock_raw_results = []
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
            crystal = _residue_selection(object, ligand[0], ligand[1], ligand[2])
            if not cmd.count_atoms(crystal):
                crystal = _residue_selection(object, None, ligand[1], ligand[2])
            presentation.style_ligand(crystal)
            presentation.focus(f"({pose_obj}) or ({crystal})")
        success = dict(run.get("top1_success") or {})
        rms = [_num(r.get("top1_rmsd_A"), 2) for r in reps if _num(r.get("top1_rmsd_A")) is not None]
        rms_text = (("RMSD top-1 " + ", ".join(f"{v:.2f}" for v in rms) + " Å") if rms else
                    "RMSD unavailable (the poses could not be matched atom by atom to the crystal ligand)")
        card = (f"Redocking of **{ligand[2]}**: top-1 success **{success.get('n_success', 0)}/{success.get('n', 0)}** "
                f"(RMSD ≤ 2 Å); {rms_text}; score {float(best.get('affinity', 0) or 0):.2f} kcal/mol.")
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
                        "ligand_space_kept": bool(ens.get("ligand_space_kept")),
                        "temperature_K": ens.get("temperature_k"), "site_residues": ens.get("site_residues"),
                        "members": [{k: m.get(k) for k in ("relative_energy_kj_mol", "site_rmsd_to_start_A")}
                                    for m in ens.get("members") or []]}
            card += (f" Receptor ensemble: best model + {ensemble['structures'] - 1} Monte Carlo conformations "
                     f"({ens.get('temperature_k'):.0f} K).")
            if ens.get("warning"):
                ensemble["warning"] = ens["warning"]
                card += f" Warning: {ens['warning']}."
        elif int(ensemble_structures) > 0:
            why = str(getattr(dlg, "_redock_last_mc_ensemble_note", "") or "the sampler did not run")
            ensemble = {"used": False, "requested": int(ensemble_structures), "reason": why}
            card += f" Warning: the receptor ensemble was NOT used ({why}); this is a single-receptor result."
        ranked = _ranked_poses(dlg)
        poses = [dict(_pose_summary(it), poses_available=sum(1 for o in ranked if (o["system"], o["member"])
                                                             == (it["system"], it["member"])))
                 for it in ranked if it["rank"] == 1]
        summary_csv = str(getattr(dlg, "_redock_last_summary_csv", "") or "")
        return {
            "_card": card,
            "grid_box": grid_box,
            "kept_hetero": sorted({str(r).strip().upper() for r in keep_hetero or [] if str(r).strip()}),
            "poses": poses[:40],
            "poses_note": "top pose of each system/member; show_redock_pose(system, member, rank) loads rank 1.."
                          "poses_available; export_redocking_csv writes all of them",
            "comparison_csv": summary_csv if summary_csv and os.path.isfile(summary_csv) else "",
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
                              "seed": {"type": "integer", "minimum": 1},
                              "keep_ligand_space": {"type": "boolean", "description": (
                                  "keep the hetero groups of site_selection (the ligand) as a soft volume while "
                                  "sampling: holo-like pockets that stay open for that ligand")}},
               "required": ["object", "site_selection"]},
              permission="compute_heavy", title="Monte Carlo ensemble")
    def generate_mc_ensemble(ctx, object, site_selection, n_structures=10, temperature_K=300.0, radius_A=6.0, seed=1,
                             keep_ligand_space=False):
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
        # the whole object: cofactors and ions stay fixed during sampling and are kept in every member
        cmd.save(pdb_in, f"({object})", state=1)
        # hetero groups that make up the site selection (typically the ligand) are the pocket, not obstacles
        lig = {"het": set()}
        cmd.iterate(f"({object}) and ({site_selection}) and hetatm and not resn HOH+WAT",
                    "het.add((chain, resi, resn))", space=lig)
        platform = str((_profile(ctx).get("md_platform") or {}).get("name") or "CPU")
        args = {"pdb_in": pdb_in, "out_dir": work, "n_structures": int(n_structures),
                "site_residues": [list(r) for r in site], "temperature_k": float(temperature_K),
                "steps_per_structure": 15, "seed": int(seed), "platform": platform,
                "exclude_hetero": [list(x) for x in sorted(lig["het"])],
                "ligand_obstacle": bool(keep_ligand_space and lig["het"])}
        with open(os.path.join(work, "mc_args.json"), "w") as fh:
            json.dump(args, fh)
        py = os.path.join(sys.prefix, "bin", "python")
        log = open(os.path.join(work, "mc_log.txt"), "w")
        proc = subprocess.Popen([py if os.path.isfile(py) else sys.executable, mc.__file__, "mc_args.json"],
                                cwd=work, stdout=log, stderr=subprocess.STDOUT, **POPEN_GROUP)

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
                "requested": res.get("requested", n_structures), "warning": res.get("warning", ""),
                "_card": (f"**{len(res['structures'])} conformations** of the {object} site ({len(site)} residues) "
                          f"at {temperature_K:.0f} K on {res['platform']} in {res['seconds']} s; acceptance "
                          f"{res['acceptance']:.0%}; site RMSD {min(rms):.2f}-{max(rms):.2f} Å (play the states)."
                          + (f" Warning: {res['warning']}." if res.get("warning") else ""))}}

        def cancel():
            stop_process(proc, log)

        return ctx.jobs.start("mc_ensemble", poll, cancel=cancel, description=f"Monte Carlo ensemble of {object}")


def register_poses(reg):
    @reg.tool("show_redock_pose", "Load one redocking pose (any rank, not only the top one) of the WT or mutant "
              "receptor (and ensemble member, if an ensemble was used) in the viewer next to the crystal ligand, "
              "as Leonardo's 'Load selected pose'; optionally build the pose complex and compute its "
              "protein-ligand interactions. Ranks come from run_redocking's 'poses'.",
              {"properties": {
                  "rank": {"type": "integer", "minimum": 1, "description": "rank within the system (1 = best)"},
                  "system": {"type": "string", "description": "'WT' (default), 'mutant' or a system label from "
                                                              "run_redocking's poses"},
                  "member": {"type": "string", "description": "ensemble member tag (e.g. WT_model_002)"},
                  "interactions": {"type": "boolean", "description": "also compute the pose's interactions"},
              }, "required": ["rank"]},
              permission="compute_light", title="Redocking pose")
    def show_redock_pose(ctx, rank, system="", member="", interactions=False):
        from collections import Counter

        from pymol import cmd

        dlg = ctx.dlg
        item, label = _pick_pose(_ranked_poses(dlg), rank, system, member)
        table = getattr(dlg, "tbl_redock_results", None)
        if table is None or table.rowCount() <= item["table_index"]:
            raise ToolError("the redocking results table is not filled; run run_redocking first")
        with _headless.quiet() as q, _as_selected_row(dlg, item):
            if interactions:
                dlg.interactions = []
                dlg._on_redock_compute_interactions_for_selected()
            else:
                dlg._on_redock_load_selected_pose(silent=False)
        problems = [m["text"] for m in q.messages if m["kind"] in ("warning", "critical")]
        known = set(cmd.get_names("objects"))
        shown = "redock_selected_complex" if interactions else "redock_selected_pose_preview"
        if shown not in known:
            shown = next((o for o in ("redock_selected_pose_preview", str(item["row"].get("pose_object", "")))
                          if o and o in known), "")
        if problems and not shown:
            raise ToolError("; ".join(problems))
        out = dict(_pose_summary(item), shown_object=shown)
        if interactions:
            inter = list(getattr(dlg, "interactions", []) or [])
            if not inter:
                raise ToolError("; ".join(problems) or "no interactions found for this pose")
            counts = Counter(str(i.get("type", "")) for i in inter)
            out["interactions_total"] = len(inter)
            out["interactions_by_type"] = dict(counts)
            out["closest"] = [{"type": i.get("type"), "residue": i.get("prot_res"),
                               "distance_A": _num(i.get("distance"), 2)}
                              for i in sorted(inter, key=lambda i: float(i.get("distance", 99) or 99))[:10]]
        if problems:
            out["leonardo_warnings"] = [p[:300] for p in problems[:3]]
        out["scene"] = ctx.scene(f"redocking {label} pose #{rank}")
        out["_card"] = (f"{label} redocking pose **#{rank}** [{out['member']}, mode {out['mode']}]"
                        + f": score {out['score_kcal_mol']} kcal/mol, RMSD to crystal {out['rmsd_A']} Å"
                        + (f"; {out['interactions_total']} interactions ("
                           + ", ".join(f"{n} {t}" for t, n in counts.most_common()) + ")" if interactions else "")
                        + ".")
        return out

    @reg.tool("export_redocking_csv", "Write the last redocking results to CSV: every ranked pose (system, "
              "member, mode, score, RMSD to the crystal pose) and, when a mutant was redocked, Leonardo's "
              "WT-vs-mutant comparison table (its 'Export comparison CSV'). Files go to the Leonardo workspace "
              "unless a path is given.",
              {"properties": {"path": {"type": "string", "description": "comparison CSV path (the pose table is "
                                                                        "written next to it, *_poses.csv)"}}},
              permission="compute_light", title="Export redocking CSV")
    def export_redocking_csv(ctx, path=""):
        dlg = ctx.dlg
        rows = _ranked_poses(dlg)
        runs = dict(getattr(dlg, "_redock_last_runs", {}) or {})
        if not rows and not runs:
            raise ToolError("no redocking results: run run_redocking first")
        if path:
            base = os.path.abspath(os.path.expanduser(str(path)))
            if not base.lower().endswith(".csv"):
                base += ".csv"
        else:
            base = os.path.join(ctx.workspace("redocking"), time.strftime("redocking_comparison_%Y%m%d_%H%M%S.csv"))
        os.makedirs(os.path.dirname(base), exist_ok=True)
        out = {}
        if rows:
            poses_csv = base[:-4] + "_poses.csv"
            fields = ["system", "member", "rank", "mode", "repeat", "score_kcal_mol", "rmsd_A", "pose_object",
                      "state"]
            with open(poses_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow(_pose_summary(r))
            out["poses_csv"] = poses_csv
            out["poses"] = len(rows)
        comparison = [r for r in list(getattr(dlg, "_redock_last_comparison_rows", []) or []) if r]
        if comparison or len(runs) > 1:
            with _headless.quiet() as q, _save_dialog_answer(base):
                dlg._on_redock_export_comparison_csv()
            saved = str(getattr(dlg, "_redock_last_summary_csv", "") or "")
            if saved and os.path.isfile(saved) and os.path.abspath(saved) == os.path.abspath(base):
                out["comparison_csv"] = saved
                out["comparison_rows"] = len(list(getattr(dlg, "_redock_last_comparison_rows", []) or []))
            else:
                out["comparison_note"] = "; ".join(m["text"][:300] for m in q.messages) or "comparison not written"
        else:
            out["comparison_note"] = "WT-only redocking: no WT-vs-mutant comparison table"
        if not out.get("poses_csv") and not out.get("comparison_csv"):
            raise ToolError(out.get("comparison_note") or "nothing to export")
        out["_card"] = "Redocking CSV: " + ", ".join(f"`{out[k]}`" for k in ("comparison_csv", "poses_csv") if k in out)
        return out
