"""Virtual screening of a ligand library with smina, and lead optimization (analogs of a seed, docked
and ranked by a multi-parameter score)."""

import math
import os
import time

from .. import presentation
from ..registry import ToolError
from . import _headless
from ._dialogjob import start_deferred_job
from ._widgets import set_value


def _finite(x):
    try:
        v = float(x)
        return round(v, 3) if math.isfinite(v) else None
    except Exception:
        return None


def _write_smiles_library(ctx, compounds):
    from rdkit import Chem

    path = os.path.join(ctx.workspace("screening"), time.strftime("library_%Y%m%d_%H%M%S.sdf"))
    n = 0
    with Chem.SDWriter(path) as w:
        for i, c in enumerate(compounds, start=1):
            smi = c.get("smiles") if isinstance(c, dict) else str(c)
            name = (c.get("name") if isinstance(c, dict) else "") or f"cpd_{i}"
            mol = Chem.MolFromSmiles(str(smi or ""))
            if mol is None:
                continue
            mol.SetProp("_Name", str(name))
            w.write(mol)
            n += 1
    if n == 0:
        raise ToolError("none of the SMILES could be parsed")
    return path, n


def _select_pocket(dlg, pocket_rank):
    pockets = list(getattr(dlg, "_fpocket_pockets", []) or [])
    if not pockets:
        raise ToolError("no pockets: run find_pockets (and select_pocket) first")
    dlg._refresh_virtual_screening_pocket_combo()
    if pocket_rank is None:
        # the pocket chosen for docking, else the best one
        current = dlg.cmb_dock_pocket.currentIndex() if hasattr(dlg, "cmb_dock_pocket") else -1
        pocket_rank = int(current) + 1 if current >= 0 else 1
    if not 1 <= int(pocket_rank) <= dlg.cmb_vs_pocket.count():
        raise ToolError(f"pocket_rank must be 1..{dlg.cmb_vs_pocket.count()}")
    dlg.cmb_vs_pocket.setCurrentIndex(int(pocket_rank) - 1)
    return int(pocket_rank)


def _select_receptor(dlg, receptor):
    dlg._refresh_virtual_screening_receptor_combo()
    if receptor:
        from pymol import cmd

        if receptor not in cmd.get_names("objects"):
            raise ToolError(f"no object named {receptor!r}")
        dlg.vs_structure_combo.setCurrentText(receptor)
        if str(dlg.vs_structure_combo.currentText() or "") != receptor:
            raise ToolError(f"{receptor!r} is not offered as a screening receptor (Leonardo lists protein "
                            "objects only, not generated complexes/poses)")
    rec = str(dlg.vs_structure_combo.currentText() or "")
    if not rec:
        raise ToolError("no receptor: give receptor=<object>")
    pocket_rec = str(getattr(dlg, "_fpocket_receptor_object", "") or "")
    if pocket_rec and pocket_rec != rec:
        raise ToolError(f"the pockets were found on {pocket_rec!r}, not on {rec!r}: run find_pockets on {rec!r} "
                        "first (or screen against the same receptor)")
    return rec


def register(reg):
    register_druglike(reg)

    @reg.tool("run_virtual_screening", "Dock a library of compounds into a pocket with smina and rank them. The "
              "library is an SDF file or a list of SMILES (with optional names). Uses the pocket from "
              "find_pockets/select_pocket. About 10-60 s per compound.",
              {"properties": {
                  "library_sdf": {"type": "string", "description": "path of an SDF library"},
                  "compounds": {"type": "array", "items": {
                      "type": "object", "properties": {"smiles": {"type": "string"}, "name": {"type": "string"}},
                      "required": ["smiles"]}},
                  "receptor": {"type": "string"},
                  "pocket_rank": {"type": "integer", "minimum": 1},
                  "exhaustiveness": {"type": "integer", "minimum": 1, "maximum": 64},
                  "max_ligands": {"type": "integer", "minimum": 1, "maximum": 5000},
                  "ph": {"type": "number", "minimum": 0, "maximum": 14},
                  "druglike_filter": {"type": "string", "enum": ["none", "lipinski", "lipinski_strict", "veber",
                                                                 "lipinski_veber", "ghose", "egan", "lead_like"],
                                      "description": "compounds failing the rule are not docked"},
                  "exclude_pains": {"type": "boolean"},
              }},
              permission="compute_heavy", title="Virtual screening (smina)")
    def run_virtual_screening(ctx, library_sdf="", compounds=None, receptor="", pocket_rank=None,
                              exhaustiveness=8, max_ligands=250, ph=7.4, druglike_filter="none",
                              exclude_pains=False):
        dlg = ctx.dlg
        if compounds:
            library_sdf, n = _write_smiles_library(ctx, compounds)
        elif library_sdf:
            library_sdf = os.path.abspath(os.path.expanduser(str(library_sdf)))
            if not os.path.isfile(library_sdf):
                raise ToolError(f"no such file: {library_sdf}")
        else:
            raise ToolError("give library_sdf or compounds")
        rec = _select_receptor(dlg, receptor)
        rank = _select_pocket(dlg, pocket_rank)
        dlg._vs_library_sdf = library_sdf
        set_value(dlg, "spin_vs_exhaustiveness", int(exhaustiveness))
        set_value(dlg, "spin_vs_max_ligands", int(max_ligands))
        set_value(dlg, "spin_vs_ph", float(ph))
        from ._widgets import set_combo

        set_combo(dlg, "cmb_vs_druglike_filter", druglike_filter, required=False)
        if hasattr(dlg, "cb_vs_exclude_pains"):
            dlg.cb_vs_exclude_pains.setChecked(bool(exclude_pains))
        dlg._vs_filter_rejected = []
        _headless.auto_hetatm_selection(dlg)
        dlg._vs_results = []

        def collect(outcome):
            if outcome["error"]:
                raise ToolError(outcome["error"])
            rows = list(getattr(dlg, "_vs_results", []) or [])
            if not rows:
                raise ToolError("the screen produced no results (see Leonardo's message)")
            ok = [r for r in rows if str(r.get("status")) == "ok"]
            from ... import druglike_runtime as dl

            props = dict(getattr(dlg, "_vs_library_props", {}) or {})
            table = [dict({"rank": r.get("rank"), "name": r.get("name"),
                           "affinity_kcal_mol": _finite(r.get("affinity")), "library_index": r.get("index")},
                          **dl.summary_row(props.get(int(r.get("index") or 0), {})))
                     for r in ok[:30]]
            removed = list(getattr(dlg, "_vs_filter_rejected", []) or [])
            failed = [{"name": r.get("name"), "notes": str(r.get("notes", ""))[:120]}
                      for r in rows if str(r.get("status")) != "ok"][:10]
            if ok:
                dlg.tbl_vs_hits.selectRow(0)
                try:
                    dlg._show_vs_selected_hit_preview(zoom=True)
                except Exception:
                    pass
            return {"receptor": rec, "pocket_rank": rank, "screened": len(rows), "docked": len(ok),
                    "druglike_filter": druglike_filter + (" + PAINS" if exclude_pains else ""),
                    "removed_by_filter": removed[:30], "removed_count": len(removed),
                    "failed": failed, "ranking": table, "job_dir": str(getattr(dlg, "_vs_job_dir", "") or ""),
                    "scene": ctx.scene(f"virtual screening on {rec}: best hit {table[0]['name'] if table else ''}"),
                    "_card": ((f"{len(removed)} compound(s) removed by the {druglike_filter} filter; " if removed else "")
                              + f"**{len(ok)}/{len(rows)}** compounds docked; best "
                              + ", ".join(f"{t['name']} {t['affinity_kcal_mol']}" for t in table[:3])
                              + " kcal/mol (smina scores, not ΔG).")}

        return start_deferred_job(ctx, "screening", dlg._on_run_virtual_screening, collect,
                                  description=f"virtual screening of {os.path.basename(library_sdf)}")

    @reg.tool("show_screening_hit", "Load a virtual-screening hit (by rank) as a protein-ligand complex and "
              "optionally analyze its interactions.",
              {"properties": {"rank": {"type": "integer", "minimum": 1}, "interactions": {"type": "boolean"}},
               "required": ["rank"]},
              permission="compute_light", title="Screening hit")
    def show_screening_hit(ctx, rank, interactions=True):
        dlg = ctx.dlg
        rows = list(getattr(dlg, "_vs_results", []) or [])
        idx = [i for i, r in enumerate(rows) if r.get("rank") == int(rank)]
        if not idx:
            raise ToolError(f"no hit with rank {rank}; run run_virtual_screening first")
        dlg.tbl_vs_hits.selectRow(idx[0])
        with _headless.quiet() as q:
            if interactions:
                dlg.interactions = []
                dlg._on_vs_compute_interactions_selected_hit()
            else:
                dlg._prepare_vs_selected_complex()
        obj = str(getattr(dlg, "loaded_object", "") or "")
        out = {"rank": rank, "name": rows[idx[0]].get("name"), "complex": obj,
               "affinity_kcal_mol": _finite(rows[idx[0]].get("affinity"))}
        if interactions:
            from collections import Counter

            inter = list(getattr(dlg, "interactions", []) or [])
            out["interactions_by_type"] = dict(Counter(str(i.get("type", "")) for i in inter))
            out["closest"] = [{"type": i.get("type"), "residue": i.get("prot_res"),
                               "distance_A": round(float(i.get("distance", 0)), 2)}
                              for i in sorted(inter, key=lambda i: float(i.get("distance", 99) or 99))[:10]]
        if obj:
            presentation.focus(obj, buffer=6.0)
        out["scene"] = ctx.scene(f"screening hit #{rank} {out['name']}")
        out["messages"] = [m["text"][:200] for m in q.messages if m["kind"] in ("warning", "critical")][:3]
        out["_card"] = f"Hit **#{rank} {out['name']}** ({out['affinity_kcal_mol']} kcal/mol) loaded as `{obj}`."
        return out

    @reg.tool("run_lead_optimization", "Generate analogs of a seed compound (a screening hit by rank, or a SMILES), "
              "filter them (similarity window, drug-likeness, PAINS), dock them in the screening pocket and rank "
              "them by a multi-parameter score (docking, ADMET, QED, similarity). Minutes to an hour.",
              {"properties": {
                  "seed_rank": {"type": "integer", "minimum": 1},
                  "seed_smiles": {"type": "string"}, "seed_name": {"type": "string"},
                  "receptor": {"type": "string"}, "pocket_rank": {"type": "integer", "minimum": 1},
                  "max_analogs": {"type": "integer", "minimum": 1, "maximum": 500},
                  "min_similarity": {"type": "number", "minimum": 0, "maximum": 1},
                  "max_similarity": {"type": "number", "minimum": 0, "maximum": 1},
                  "exhaustiveness": {"type": "integer", "minimum": 1, "maximum": 64},
                  "exclude_pains": {"type": "boolean"},
              }},
              permission="compute_heavy", title="Lead optimization")
    def run_lead_optimization(ctx, seed_rank=None, seed_smiles="", seed_name="seed", receptor="", pocket_rank=None,
                              max_analogs=None, min_similarity=None, max_similarity=None, exhaustiveness=None,
                              exclude_pains=True):
        dlg = ctx.dlg
        if not hasattr(dlg, "_leadopt_dialog"):
            dlg._setup_lead_optimization_ui()          # builds the controls without showing the window
        if seed_smiles:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(str(seed_smiles))
            if mol is None:
                raise ToolError(f"RDKit cannot parse {seed_smiles!r}")
            molh = Chem.AddHs(mol)
            AllChem.EmbedMolecule(molh, randomSeed=0xF00D)
            dlg._leadopt_seed_hit = {"name": str(seed_name), "index": 0, "affinity": float("nan"), "mol": molh,
                                     "identifier": str(seed_name)}
        elif seed_rank is not None:
            rows = list(getattr(dlg, "_vs_results", []) or [])
            idx = [i for i, r in enumerate(rows) if r.get("rank") == int(seed_rank)]
            if not idx:
                raise ToolError(f"no screening hit with rank {seed_rank}")
            dlg.tbl_vs_hits.selectRow(idx[0])
            dlg._leadopt_seed_hit = None
            dlg._on_leadopt_use_selected_vs_hit(silent=True)
        else:
            raise ToolError("give seed_rank (a screening hit) or seed_smiles")
        _select_receptor(dlg, receptor)
        _select_pocket(dlg, pocket_rank)
        for attr, val in (("spin_lo_max_analogs", max_analogs), ("spin_lo_min_sim", min_similarity),
                          ("spin_lo_max_sim", max_similarity), ("spin_lo_exhaustiveness", exhaustiveness)):
            if val is not None:
                set_value(dlg, attr, val)
        set_value(dlg, "cb_lo_block_pains", bool(exclude_pains))
        _headless.auto_hetatm_selection(dlg)
        dlg._leadopt_results = []

        def collect(outcome):
            if outcome["error"]:
                raise ToolError(outcome["error"])
            rows = list(getattr(dlg, "_leadopt_results", []) or [])
            if not rows:
                raise ToolError("lead optimization produced no analogs (see Leonardo's message)")
            # Leonardo's own order (refined ΔG first when refinement ran): rank N here is the rank that
            # show_lead_optimization_hit(rank=N) opens
            ok = [(i, r) for i, r in enumerate(rows, start=1) if str(r.get("status", "ok")).lower() == "ok"]
            table = [{"rank": int(r.get("rank") or i), "name": r.get("name"), "smiles": r.get("smiles"),
                      "mpo": _finite(r.get("mpo")), "affinity_kcal_mol": _finite(r.get("affinity")),
                      "qed": _finite(r.get("qed")),
                      "similarity": _finite(r.get("sim_seed", r.get("similarity", r.get("sim"))))}
                     for i, r in ok[:20]]
            return {"seed": str((dlg._leadopt_seed_hit or {}).get("name", "")), "analogs_scored": len(ok),
                    "ranking": table, "job_dir": str(getattr(dlg, "_leadopt_job_dir", "") or ""),
                    "scene": ctx.scene("lead optimization: best analog"),
                    "_card": f"**{len(ok)} analogs** scored; top "
                             + ", ".join(f"#{t['rank']} {t['name']} (MPO {t['mpo']})" for t in table[:3]) + "."}

        return start_deferred_job(ctx, "leadopt", dlg._on_run_lead_optimization, collect,
                                  description="lead optimization")


def register_druglike(reg):
    @reg.tool("druglikeness", "Drug-likeness and estimated ADME of molecules: MW, cLogP, HBD/HBA, TPSA, rotatable "
              "bonds, QED; Lipinski Ro5, Veber, Ghose, Egan and lead-like rules; PAINS and Brenk alerts; ESOL "
              "solubility, GI absorption and BBB permeation (BOILED-Egg approximation). Give SMILES, a PyMOL "
              "selection or an SDF library. Estimates for ranking, not measured ADME.",
              {"properties": {
                  "smiles": {"type": "array", "items": {"type": "string"}},
                  "selection": {"type": "string"}, "library_sdf": {"type": "string"},
                  "rule": {"type": "string", "enum": ["none", "lipinski", "lipinski_strict", "veber",
                                                     "lipinski_veber", "ghose", "egan", "lead_like"]},
                  "exclude_pains": {"type": "boolean"},
              }},
              permission="read", title="Drug-likeness / ADME")
    def druglikeness(ctx, smiles=(), selection="", library_sdf="", rule="lipinski", exclude_pains=True):
        from rdkit import Chem

        from ... import druglike_runtime as dl

        mols = []
        for i, smi in enumerate(smiles or []):
            m = Chem.MolFromSmiles(str(smi))
            if m is None:
                raise ToolError(f"RDKit cannot parse {smi!r}")
            mols.append((str(smi), m))
        if selection:
            from pymol import cmd

            if cmd.count_atoms(selection) == 0:
                raise ToolError(f"no atoms in {selection!r}")
            m = Chem.MolFromMolBlock(cmd.get_str("mol", selection), removeHs=False)
            if m is None:
                raise ToolError("could not read the selection as a molecule")
            mols.append((selection, m))
        if library_sdf:
            path = os.path.abspath(os.path.expanduser(library_sdf))
            if not os.path.isfile(path):
                raise ToolError(f"no such file: {path}")
            for i, m in enumerate(Chem.SDMolSupplier(path)):
                if m is not None:
                    mols.append((m.GetProp("_Name") if m.HasProp("_Name") and m.GetProp("_Name") else f"cpd_{i + 1}", m))
                if len(mols) >= 2000:
                    break
        if not mols:
            raise ToolError("give smiles, a selection or a library_sdf")
        rows = []
        for name, m in mols:
            p = dl.properties(m)
            ok, reason = dl.passes(p, rule, exclude_pains)
            rows.append(dict({"name": name, "passes": ok, "reason": reason}, **dl.summary_row(p),
                             lipinski_violations=p["lipinski_violations"], pains_alerts=p["pains_alerts"],
                             brenk_alerts=p["brenk_alerts"][:3], solubility_class=p["solubility_class"]))
        n_ok = sum(r["passes"] for r in rows)
        return {"rule": dl.FILTERS.get(rule, rule) + (" + no PAINS" if exclude_pains else ""),
                "passing": n_ok, "total": len(rows), "compounds": rows[:60],
                "_card": f"**{n_ok}/{len(rows)}** pass {dl.FILTERS.get(rule, rule)}"
                         + (" without PAINS" if exclude_pains else "") + ". "
                         + "; ".join(f"{r['name'][:24]}: MW {r['mw']}, cLogP {r['clogp']}, QED {r['qed']}"
                                     + ("" if r["passes"] else f" ✗ {r['reason']}") for r in rows[:3])}
