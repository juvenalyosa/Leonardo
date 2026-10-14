"""The rest of Leonardo's panels: docking box/receptor/flexible residues, external trajectories,
single-interaction distributions, 2D interaction maps, CSV export, projects, charge/multiplicity
detection, QM/MM region preview, ligand force-field rebuild, AlphaFold import, lead-optimization hits."""

import os
import re
import sys
import time
import zipfile

from .. import presentation
from ..registry import ToolError
from . import _headless
from ._widgets import jsonable


def _cmd():
    from pymol import cmd

    return cmd


_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "CL": 17, "BR": 35, "I": 53}


def _atomic_number(element):
    sym = str(element or "").strip()
    if sym.upper() in _Z:
        return _Z[sym.upper()]
    try:
        from openmm.app import element

        return int(element.Element.getBySymbol(sym[:1].upper() + sym[1:].lower()).atomic_number)
    except Exception:
        return None


def electron_count(atoms, charge):
    """Electrons of a molecule from its atoms' elements (any element: Fe, Zn, Mg, Se...) and net charge;
    None when an element is unknown (never counted as carbon)."""
    z = [_atomic_number(a.get("element", "")) for a in atoms]
    return None if None in z else sum(z) - int(charge)


def _need_object(name):
    if name not in _cmd().get_names("objects"):
        raise ToolError(f"no object named {name!r}")


def _residue_keys(it):
    """The ways a user names the protein residue of an interaction ('ASP189', 'ASP A189', 'A:189', 'A:ASP189',
    '189'); Leonardo's prot_res is 'ASP A189', so a substring test never matched 'ASP189' (and 'A76'
    matched 'LEU A768')."""
    resn = str(it.get("prot_resn") or "").upper()
    chain = str(it.get("prot_chain") or "").upper()
    resi = str(it.get("prot_resi") or "").upper()
    if not resi:
        m = re.match(r"^\s*(\S+)\s+(\S+)\s*$", str(it.get("prot_res", "")))
        resn, resi = (m.group(1).upper(), m.group(2).upper()) if m else (resn, resi)
    keys = {resn + resi, resn + chain + resi, f"{chain}:{resi}", f"{chain}:{resn}{resi}", resi,
            str(it.get("prot_res", "")).upper().replace(" ", "")}
    return {k for k in keys if k and k != ":"}


def register(reg):
    # -- docking setup ---------------------------------------------------------------------------- #
    @reg.tool("set_docking_box", "Define the docking box by hand: centered on a selection (e.g. a crystal ligand "
              "or residues) with a padding, or from explicit center/size in Å. Use when fpocket's pocket is not the "
              "site you want.",
              {"properties": {"selection": {"type": "string"}, "padding_A": {"type": "number", "minimum": 0, "maximum": 20},
                              "center": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                              "size": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}}},
              permission="view", title="Docking box")
    def set_docking_box(ctx, selection="", padding_A=6.0, center=None, size=None):
        cmd = _cmd()
        dlg = ctx.dlg
        if selection:
            if cmd.count_atoms(selection) == 0:
                raise ToolError(f"no atoms in {selection!r}")
            lo, hi = cmd.get_extent(selection)
            center = [(a + b) / 2.0 for a, b in zip(lo, hi)]
            size = size or [max(12.0, (b - a) + 2 * float(padding_A)) for a, b in zip(lo, hi)]
        if not center or not size:
            raise ToolError("give a selection, or center and size")
        for attr, v in zip(("spin_cx", "spin_cy", "spin_cz", "spin_sx", "spin_sy", "spin_sz"), list(center) + list(size)):
            getattr(dlg, attr).setValue(float(v))
        dlg._dock_grid_user_defined = True
        with _headless.quiet():
            dlg._show_current_docking_grid(zoom=False)
        presentation.show_box(center, size)
        presentation.focus("ai_box", buffer=4.0)
        return {"center": [round(v, 2) for v in center], "size": [round(v, 2) for v in size],
                "scene": ctx.scene("docking box"),
                "_card": f"Docking box {' × '.join(f'{v:.1f}' for v in size)} Å at "
                         f"({', '.join(f'{v:.1f}' for v in center)})."}

    @reg.tool("set_docking_receptor", "Make an object the docking/screening receptor (for example a MODELLER or "
              "AlphaFold model, or a minimized structure).",
              {"properties": {"object": {"type": "string"}}, "required": ["object"]},
              permission="view", title="Docking receptor")
    def set_docking_receptor(ctx, object):
        _need_object(object)
        dlg = ctx.dlg
        with _headless.quiet():
            for refresh in ("_refresh_docking_object_combo", "_refresh_virtual_screening_receptor_combo"):
                try:
                    getattr(dlg, refresh)()
                except Exception:
                    pass
            dlg.dock_structure_combo.setCurrentText(object)
            if hasattr(dlg, "vs_structure_combo"):
                dlg.vs_structure_combo.setCurrentText(object)
            dlg._docking_receptor_object = object
        return {"receptor": object, "_card": f"Docking receptor: `{object}`."}

    @reg.tool("set_flexible_residues", "Flexible side chains for the next docking (induced fit): residues as "
              "'CHAIN:RESI' or 'RESI', or all residues within a distance of the box center. Empty list clears them.",
              {"properties": {"residues": {"type": "array", "items": {"type": "string"}},
                              "within_A": {"type": "number", "minimum": 2, "maximum": 10}}},
              permission="view", title="Flexible residues")
    def set_flexible_residues(ctx, residues=(), within_A=None):
        dlg = ctx.dlg
        rec = str(dlg._get_selected_docking_receptor() or "")
        center = dlg._dock_center_from_controls()
        if not rec or center is None:
            raise ToolError("set the receptor and the docking box first")
        # explicit residues are looked for in a wider shell than the automatic pick
        cutoff = float(within_A or (14.0 if residues else 8.0))
        cands = dlg._collect_flexible_residue_candidates(rec, center, cutoff_a=cutoff, max_keep=320)
        unmatched = []
        if residues:
            import re

            wanted = []
            for r in residues:
                m = re.match(r"^\s*(?:([A-Za-z0-9]):)?\s*([A-Za-z]{3})?\s*(-?\d+[A-Za-z]?)\s*$", str(r))
                if not m:
                    raise ToolError(f"cannot read residue {r!r}: use 'A:189', '189' or 'ASP189'")
                wanted.append(((m.group(1) or "").upper(), (m.group(2) or "").upper(), m.group(3), str(r)))

            def hit(c, ch, rn, rs):
                return ((not ch or str(c.get("chain", "")).upper() == ch) and str(c.get("resi")) == rs
                        and (not rn or str(c.get("resn", "")).upper() == rn))
            pick = [c for c in cands if any(hit(c, ch, rn, rs) for ch, rn, rs, _t in wanted)]
            unmatched = [t for ch, rn, rs, t in wanted if not any(hit(c, ch, rn, rs) for c in cands)]
        elif within_A:
            pick = [c for c in cands if float(c.get("min_dist", 99)) <= float(within_A)]
        else:
            pick = []
        dlg._dock_flex_residues = pick
        dlg._dock_flex_receptor_obj = rec
        if hasattr(dlg, "cb_dock_flexible"):
            dlg.cb_dock_flexible.setChecked(bool(pick))       # flexible docking on only with residues
        try:
            dlg._update_dock_flex_label()
        except Exception:
            pass
        labels = [dlg._flex_residue_label(c) for c in pick]
        if residues and not pick:
            raise ToolError("none of those residues is near the box (flexible residues must line the site; "
                            "ALA/GLY/PRO are excluded)")
        out = {"flexible_residues": labels, "_card": (f"{len(pick)} flexible residue(s): " + ", ".join(labels[:8])
                                                      if pick else "Flexible residues cleared (rigid receptor).")}
        if unmatched:
            out["not_used"] = unmatched
            out["note"] = (f"not used (not within {cutoff:.0f} Å of the box center, wrong residue name, or "
                           "ALA/GLY/PRO): " + ", ".join(unmatched))
            out["_card"] += f" Not used: {', '.join(unmatched)}."
        return out

    # -- trajectories ----------------------------------------------------------------------------- #
    @reg.tool("load_trajectory", "Load an external MD trajectory (DCD, XTC, TRR, NetCDF, multi-model PDB) on its "
              "topology (PDB/GRO/PSF...) for the trajectory analyses.",
              {"properties": {"topology": {"type": "string"}, "trajectory": {"type": "string"},
                              "name": {"type": "string"}, "stride": {"type": "integer", "minimum": 1}},
               "required": ["topology"]},
              permission="compute_light", title="Load trajectory")
    def load_trajectory(ctx, topology, trajectory="", name="traj", stride=1):
        cmd = _cmd()
        for p in (topology, trajectory):
            if p and not os.path.isfile(os.path.expanduser(p)):
                raise ToolError(f"no such file: {p}")
        name = re.sub(r"[^A-Za-z0-9_]", "_", name or "traj")[:60] or "traj"
        existing = set(cmd.get_names("objects"))
        if name in existing:          # never replace an object the user has (e.g. name="1abc")
            k = 2
            while f"{name}_{k}" in existing:
                k += 1
            name = f"{name}_{k}"
        cmd.load(os.path.expanduser(topology), name)
        if trajectory:
            cmd.load_traj(os.path.expanduser(trajectory), name, interval=int(stride))
        n = cmd.count_states(name)
        dlg = ctx.dlg
        dlg.loaded_object = name
        dlg._traj_topology_object = name
        try:
            dlg._populate_limit_combo()
        except Exception:
            pass
        presentation.style_protein(name)
        presentation.focus(name)
        return {"object": name, "frames": n, "atoms": cmd.count_atoms(name),
                "_card": f"Trajectory `{name}`: {n} frames, {cmd.count_atoms(name)} atoms."}

    @reg.tool("interaction_distribution", "Distance of one protein-ligand interaction along a trajectory (presence "
              "per frame, mean/min/max distance, occupancy): pick it by residue (e.g. 'ASP189') and optionally type.",
              {"properties": {"object": {"type": "string"}, "residue": {"type": "string"},
                              "interaction_type": {"type": "string"}},
               "required": ["object", "residue"]},
              permission="compute_light", title="Interaction distribution")
    def interaction_distribution(ctx, object, residue, interaction_type=""):
        _need_object(object)
        dlg = ctx.dlg
        dlg.loaded_object = object
        dlg._traj_topology_object = object
        with _headless.quiet():
            dlg._on_compute_reference_interactions()
        inter = list(getattr(dlg, "interactions", []) or [])
        key = str(residue).upper().replace(" ", "")
        idx = [i for i, it in enumerate(inter) if key in _residue_keys(it)
               and (not interaction_type or interaction_type.lower() in str(it.get("type", "")).lower())]
        if not idx:
            raise ToolError(f"no interaction with {residue} in the reference frame; interactions: "
                            + ", ".join(sorted({str(i.get('prot_res')) for i in inter}))[:400])
        dlg._last_clicked_rid = idx[0]
        module = sys.modules[type(dlg).__module__]
        original = getattr(module, "trajectory_build_selected_trace_payload")
        sink = []

        def spy(*a, **k):
            out = original(*a, **k)
            sink.append(out)
            return out
        setattr(module, "trajectory_build_selected_trace_payload", spy)
        try:
            with _headless.quiet():
                dlg._run_selected_interaction_distribution()
        finally:
            setattr(module, "trajectory_build_selected_trace_payload", original)
        if not sink:
            raise ToolError("the distribution could not be computed")
        p = dict(sink[-1])
        it = inter[idx[0]]
        keys = ("occupancy_pct", "distance_mean", "distance_median", "distance_min", "distance_max", "n_present")
        out = {k: jsonable(p.get(k)) for k in keys}
        out.update({"interaction": f"{it.get('type')} {it.get('prot_res')} {it.get('prot_atom')}-{it.get('lig_atom')}",
                    "frames": int(p.get("n_frames") or len(p.get("frames", [])))})
        out["_card"] = (f"{out['interaction']}: present {out['occupancy_pct']}% of {out['frames']} frames, "
                        f"distance {out['distance_mean']} Å (min {out['distance_min']}, max {out['distance_max']}).")
        return out

    # -- interaction outputs ------------------------------------------------------------------------ #
    @reg.tool("show_2d_interaction_map", "2D protein-ligand interaction diagram of the current complex (after "
              "analyze_interactions), as a window and an image in the chat.",
              permission="view", title="2D interaction map")
    def show_2d_interaction_map(ctx):
        dlg = ctx.dlg
        if not getattr(dlg, "ligand_info", None) or not getattr(dlg, "interactions", None):
            raise ToolError("analyze the interactions of a complex first (analyze_interactions)")
        path = os.path.join(ctx.workspace("plots"), time.strftime("interaction_map_%H%M%S.png"))
        try:
            with _headless.quiet(), _headless.plots_allowed():
                dlg._generate_2d_map_figure(path=path, show=True)
        except Exception as exc:
            raise ToolError(f"the 2D map could not be drawn: {exc}")
        if not os.path.isfile(path):
            raise ToolError("no 2D map was drawn")
        return {"image": path, "_image": path, "_card": "2D interaction map of the current complex."}

    @reg.tool("export_interaction_figure", "Publication-quality figure of the current protein-ligand interactions "
              "(after analyze_interactions): the ray-traced 3D view with a legend, and the 2D diagram.",
              {"properties": {"folder": {"type": "string"}, "width": {"type": "integer"},
                              "height": {"type": "integer"}}},
              permission="compute_light", title="Interaction figure")
    def export_interaction_figure(ctx, folder="", width=2400, height=1600):
        from leonardo import interaction_view

        dlg = ctx.dlg
        inter = list(getattr(dlg, "interactions", []) or [])
        li = getattr(dlg, "ligand_info", None)
        if not inter or not li:
            raise ToolError("analyze the interactions of a complex first (analyze_interactions)")
        folder = os.path.abspath(os.path.expanduser(folder)) if folder else ctx.workspace("figures")
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        chain, resi, resn = li[0], li[1], li[2]
        title = f"{resn} {chain}{resi}".strip()
        sub = f"{dlg.loaded_object} \u00b7 {len(inter)} interactions"
        fig3d = os.path.join(folder, f"interactions_3d_{stamp}.png")
        map2d = os.path.join(folder, f"interactions_2d_{stamp}.png")
        width, height = max(600, min(int(width), 6000)), max(400, min(int(height), 4000))
        with _headless.quiet():
            interaction_view.export_figure(fig3d, inter, width, height, title=title, subtitle=sub)
            dlg._generate_2d_map_figure(path=map2d, show=False)
        return {"figure_3d": fig3d, "map_2d": map2d, "_image": fig3d,
                "_card": f"Interaction figure saved: `{fig3d}` and the 2D diagram `{map2d}`."}

    @reg.tool("export_interactions_csv", "Write the current protein-ligand interaction table to a CSV file.",
              {"properties": {"path": {"type": "string"}}},
              permission="compute_light", title="Export interactions")
    def export_interactions_csv(ctx, path=""):
        import csv

        inter = list(getattr(ctx.dlg, "interactions", []) or [])
        if not inter:
            raise ToolError("no interactions: run analyze_interactions first")
        path = os.path.abspath(os.path.expanduser(path)) if path else os.path.join(
            ctx.workspace("exports"), time.strftime("interactions_%Y%m%d_%H%M%S.csv"))
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Type", "Prot Res", "Prot Atom", "Lig Atom", "Distance (A)", "Details"])
            for i in inter:
                w.writerow([i.get("type"), i.get("prot_res"), i.get("prot_atom"), i.get("lig_atom"),
                            f"{float(i.get('distance', 0)):.2f}", i.get("details", "")])
        return {"path": path, "rows": len(inter), "_card": f"{len(inter)} interactions written to `{path}`."}

    # -- projects ---------------------------------------------------------------------------------- #
    @reg.tool("save_project", "Save Leonardo's project (session objects, settings and results). Without an active "
              "project, a new one is created in the given folder (the scene is kept).",
              {"properties": {"folder": {"type": "string"}}}, permission="compute_light", title="Save project")
    def save_project(ctx, folder=""):
        dlg = ctx.dlg
        with _headless.quiet():
            if not str(getattr(dlg, "_project_root", "") or "").strip():
                root = os.path.abspath(os.path.expanduser(folder or os.path.join(
                    os.path.expanduser("~"), "Leonardo_projects", time.strftime("project_%Y%m%d_%H%M%S"))))
                os.makedirs(root, exist_ok=True)
                dlg._project_activate(root, load_state=False, clear_scene=False)
            saved = dlg._project_save_state(silent=True)
        root = str(dlg._project_root)
        if saved is False:
            raise ToolError(f"the project in {root} could not be saved (see Leonardo's log)")
        return {"project": root, "_card": f"Project saved in `{root}`."}

    @reg.tool("open_project", "Open a saved Leonardo project folder (replaces the current scene).",
              {"properties": {"folder": {"type": "string"}}, "required": ["folder"]},
              permission="destructive", title="Open project")
    def open_project(ctx, folder):
        from leonardo.config import PROJECT_MASTER_FILE_NAME, PROJECT_STATE_FILE_NAME

        root = os.path.abspath(os.path.expanduser(folder))
        if os.path.isfile(root) and os.path.basename(root) in (PROJECT_MASTER_FILE_NAME, PROJECT_STATE_FILE_NAME):
            root = os.path.dirname(root)
        if not os.path.isdir(root):
            raise ToolError(f"no such folder: {root}")
        # activating any folder clears the scene and creates project subfolders: check first
        if not any(os.path.isfile(os.path.join(root, f)) for f in (PROJECT_MASTER_FILE_NAME, PROJECT_STATE_FILE_NAME)):
            raise ToolError(f"{root} is not a Leonardo project (no {PROJECT_MASTER_FILE_NAME} or "
                            f"{PROJECT_STATE_FILE_NAME}); nothing was changed")
        with _headless.quiet() as q:
            ok = ctx.dlg._project_activate(root, load_state=True, clear_scene=True)
        if not ok:
            raise ToolError("; ".join(m["text"] for m in q.messages) or "not a Leonardo project")
        return {"project": root, "objects": _cmd().get_names("objects"), "_card": f"Project `{root}` opened."}

    # -- quantum chemistry helpers ------------------------------------------------------------------ #
    @reg.tool("detect_charge_multiplicity", "Net charge and spin multiplicity of a molecule/selection as Leonardo "
              "infers them (formal charges, protonation, electron count), before a QM calculation.",
              {"properties": {"selection": {"type": "string"}}, "required": ["selection"]},
              permission="read", title="Charge/multiplicity")
    def detect_charge_multiplicity(ctx, selection):
        cmd = _cmd()
        dlg = ctx.dlg
        if cmd.count_atoms(selection) == 0:
            raise ToolError(f"no atoms in {selection!r}")
        atoms = dlg._psi4_collect_atoms_from_selection(selection, state=1)
        with _headless.quiet() as q:
            dlg._psi4_auto_charge_multiplicity(atoms=atoms, source_file="", source_meta={}, quiet=True)
        charge, mult = int(dlg.spin_psi4_charge.value()), int(dlg.spin_psi4_mult.value())
        n_el = electron_count(atoms, charge)
        hyd = sum(1 for a in atoms if str(a.get("element", "")).upper() == "H")
        notes = [m["text"][:300] for m in q.messages]
        return {"charge": charge, "multiplicity": mult, "atoms": len(atoms), "electrons": n_el,
                "hydrogens_present": hyd, "notes": notes[:3],
                "warning": "no hydrogens in the selection: add them (or use the protonated ligand) before QM"
                if hyd == 0 else "",
                "_card": f"Charge **{charge:+d}**, multiplicity **{mult}** ({len(atoms)} atoms, {n_el if n_el is not None else '?'} electrons)"
                         + ("; ⚠ no hydrogens in the selection, so the charge ignores protonation: use the "
                            "protonated ligand (prepare_docking_ligand / compute_pka) for QM" if hyd == 0 else "")
                         + "."}

    @reg.tool("preview_qm_region", "Show which atoms would be QM, boundary and MM in a QM/MM calculation (and "
              "Leonardo's recommended QM region/charge), before running it.",
              {"properties": {"object": {"type": "string"},
                              "region_mode": {"type": "string", "enum": ["auto_ligand_pocket", "auto_ligand",
                                                                         "pymol_selection", "qm_resnames"]},
                              "qm_selection": {"type": "string"}, "qm_resnames": {"type": "string"},
                              "recommend": {"type": "boolean"}},
               "required": ["object"]},
              permission="view", title="QM region")
    def preview_qm_region(ctx, object, region_mode="auto_ligand_pocket", qm_selection="", qm_resnames="",
                          recommend=False):
        _need_object(object)
        dlg = ctx.dlg
        from ._widgets import set_combo, set_value

        set_combo(dlg, "cmb_qmmm_object", object)
        set_combo(dlg, "cmb_qmmm_region_mode", region_mode)
        if qm_selection:
            set_value(dlg, "edit_qmmm_qm_selection", qm_selection)
        if qm_resnames:
            set_value(dlg, "edit_qmmm_qm_resnames", qm_resnames)
        with _headless.quiet() as q:
            if recommend:
                dlg._on_qmmm_recommend_region()
            dlg._on_qmmm_preview_region()
        cmd = _cmd()
        objs = [n for n in cmd.get_names("all") if "qm" in n.lower() and n not in (object,)]
        info = [m["text"][:500] for m in q.messages]
        status = str(getattr(dlg, "lbl_qmmm_status", None).text()) if hasattr(dlg, "lbl_qmmm_status") else ""
        return {"status": status, "objects": objs, "messages": info[:3],
                "qm_charge": int(dlg.spin_qmmm_charge.value()), "scene": ctx.scene(f"QM/MM region of {object}"),
                "_card": (status or "QM region shown in the viewer")[:300]}

    @reg.tool("rebuild_ligand_forcefield", "Rebuild the ligand's Amber/OpenMM parameters keeping its current "
              "charges (for example after Psi4 ESP-fit charges), then write the OpenMM XML.",
              permission="compute_light", title="Rebuild ligand force field")
    def rebuild_ligand_forcefield(ctx):
        dlg = ctx.dlg
        if not str(getattr(dlg, "_parameterized_ligand_mol2", "") or ""):
            raise ToolError("no charged ligand MOL2: parametrize_ligand or run_psi4 esp_fit first")
        with _headless.quiet() as q:
            res = dict(dlg._rebuild_ligand_amber_from_current_charged_mol2(quiet=True) or {})
            if res.get("ok"):
                dlg._on_export_ligand_openmm_xml()
        if not res.get("ok"):
            raise ToolError(str(res.get("error") or "; ".join(m["text"] for m in q.messages) or "rebuild failed"))
        xml = str(getattr(dlg, "_parameterized_ligand_xml", "") or "")
        return {"prmtop": res.get("prmtop"), "openmm_xml": xml,
                "_card": "Ligand force field rebuilt with its current charges" + (" (OpenMM XML ready)." if xml else ".")}

    # -- AlphaFold / lead optimization -------------------------------------------------------------- #
    @reg.tool("import_alphafold_result", "Load a finished AlphaFold/ColabFold result: a model file (.pdb/.cif) or "
              "the ZIP downloaded from Colab (the best-ranked model is picked); colored by pLDDT.",
              {"properties": {"path": {"type": "string"}}, "required": ["path"]},
              permission="compute_light", title="Import AlphaFold result")
    def import_alphafold_result(ctx, path):
        dlg = ctx.dlg
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise ToolError(f"no such file: {path}")
        best = path
        if path.lower().endswith(".zip"):
            dst = os.path.join(ctx.workspace("alphafold"), time.strftime("import_%Y%m%d_%H%M%S"))
            os.makedirs(dst, exist_ok=True)
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dst)
            best = dlg._af_find_best_model_recursive(dst, token="", prefer_relaxed=dlg._af_relax_requested())
        if not best or not os.path.isfile(str(best)):
            raise ToolError("no model file in that result")
        dlg._af_last_best_pdb = str(best)
        with _headless.quiet():
            dlg._on_af_load_best_model(str(best))
        from .alphafold import _loaded_result

        return _loaded_result(ctx)

    @reg.tool("show_lead_optimization_hit", "Load a lead-optimization analog (by rank of the last run) as a "
              "complex and optionally analyze its interactions.",
              {"properties": {"rank": {"type": "integer", "minimum": 1}, "interactions": {"type": "boolean"}},
               "required": ["rank"]},
              permission="compute_light", title="Lead optimization hit")
    def show_lead_optimization_hit(ctx, rank, interactions=True):
        dlg = ctx.dlg
        rows = list(getattr(dlg, "_leadopt_results", []) or [])
        if not 1 <= int(rank) <= len(rows):
            raise ToolError(f"rank must be 1..{len(rows)} (run run_lead_optimization first)")
        if hasattr(dlg, "tbl_lo_hits"):
            dlg.tbl_lo_hits.selectRow(int(rank) - 1)
        with _headless.quiet():
            if interactions:
                dlg.interactions = []
                dlg._on_leadopt_compute_interactions_selected()
            else:
                dlg._prepare_leadopt_selected_complex()
        obj = str(getattr(dlg, "loaded_object", "") or "")
        r = rows[int(rank) - 1]
        if obj:
            presentation.focus(obj, buffer=6.0)
        out = {"rank": rank, "name": r.get("name"), "smiles": r.get("smiles"), "complex": obj,
               "scene": ctx.scene(f"lead optimization analog #{rank}"),
               "_card": f"Analog **#{rank} {r.get('name')}** loaded as `{obj}`."}
        if interactions:
            # computed by Leonardo but not returned before: the caller asked for them
            from collections import Counter

            inter = list(getattr(dlg, "interactions", []) or [])
            counts = Counter(str(i.get("type", "")) for i in inter)
            out["interactions_total"] = len(inter)
            out["interactions_by_type"] = dict(counts)
            out["closest"] = [{"type": i.get("type"), "residue": i.get("prot_res"),
                               "distance_A": round(float(i.get("distance", 0) or 0), 2)}
                              for i in sorted(inter, key=lambda i: float(i.get("distance", 99) or 99))[:10]]
            out["_card"] += (f" {len(inter)} interactions (" + ", ".join(f"{n} {t}" for t, n in counts.most_common())
                             + ")." if inter else " No interactions found.")
        return out
