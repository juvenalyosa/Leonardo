"""Protein design: disulfide bonds (candidate pairs, refined Cys/Cys models, geometry filters, rigidity model)."""

import os
import time

from ..registry import ToolError
from .components import _save_object
from .membrane import _launch


def register(reg):
    @reg.tool("design_disulfides", "Design disulfide bonds in a protein. (1) Candidate residue pairs with "
              "disulfide-compatible backbone geometry (Calpha-Calpha 4.0-7.5 A, Cbeta-Cbeta 3.2-5.2 A, not Gly/Pro, "
              ">= min_separation apart), ranked by closeness to natural disulfides. (2) For the best max_candidates: "
              "both residues -> Cys, S-S bond, restrained Amber14 refinement (backbone held, reproducible). "
              "(3) Filters of natural disulfides: S-S 1.95-2.15 A, C-S-S angles 90-120 deg, |chi3| 65-115 deg, "
              "Calpha RMS shift < 0.7 A; passing designs ranked by ||chi3|-90|. (4) Rigidity: Gaussian network model "
              "with and without the new cross-link (change in the pair's relative motion and in mean fluctuation). "
              "A structural design and rigidity model, not a measured stability. About 30 s per refined design; "
              "runs in the background. Loads the best design.",
              {"properties": {
                  "object": {"type": "string"},
                  "chain": {"type": "string", "description": "restrict to one chain (default: all)"},
                  "max_candidates": {"type": "integer", "minimum": 1, "maximum": 20,
                                     "description": "how many top candidates to build and refine (default 5)"},
                  "min_separation": {"type": "integer", "minimum": 2, "maximum": 50,
                                     "description": "minimum sequence separation within a chain (default 3)"},
                  "seed": {"type": "integer", "minimum": 1},
              }, "required": ["object"]},
              permission="compute_heavy", title="Disulfide design", main_thread=True)
    def design_disulfides(ctx, object, chain="", max_candidates=5, min_separation=3, seed=20260915):
        from pymol import cmd

        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}; call list_objects")
        if cmd.count_atoms(f"({object}) and polymer.protein and name CA") < 10:
            raise ToolError(f"{object} has no protein chain to design")
        path = _save_object(ctx, object, sub="design")
        workdir = os.path.join(ctx.workspace("design"), time.strftime(f"{object}_disulfides_%Y%m%d_%H%M%S"))
        args = {"pdb": path, "out_dir": workdir, "chain": chain, "max_candidates": int(max_candidates),
                "min_separation": int(min_separation), "seed": int(seed)}

        def done(res):
            designs = res.get("designs") or []
            best = res.get("best")
            rows = []
            for d in designs:
                rows.append({k: d.get(k) for k in ("label", "ca_ca_A", "cb_cb_A", "ss_A", "css_deg", "chi3_deg",
                                                   "ca_rmsd_A", "passes", "checks", "error")}
                            | {"rigidity": d.get("rigidity"), "model_pdb": d.get("model_pdb")})
            out = {"object": object, "n_candidates": len(res.get("candidates") or []), "designs": rows,
                   "n_passing": res.get("n_passing", 0), "filters": res.get("filters"), "note": res.get("note")}
            if best:
                name = f"{object}_{best['label'].replace('/', '_')}"[:60]
                if name in cmd.get_names("objects"):
                    cmd.delete(name)
                cmd.load(best["model_pdb"], name)
                cmd.super(f"({name}) and name CA", f"({object}) and polymer and name CA")
                blank = '""'
                sel = " or ".join(f"(chain {c or blank} and resi {i})" for c, i, _n in best["residues"])
                cmd.show("sticks", f"({name}) and ({sel}) and not name N+C+O")
                cmd.color("yellow", f"({name}) and ({sel}) and elem S")
                out["best"] = {"label": best["label"], "object": name, "ss_A": round(best["ss_A"], 3),
                               "chi3_deg": round(best["chi3_deg"], 1), "ca_rmsd_A": round(best["ca_rmsd_A"], 2),
                               "rigidity": best.get("rigidity")}
                rig = best.get("rigidity") or {}
                out["_card"] = (f"**{best['label']}** is the best of {res.get('n_passing')} passing design(s) "
                                f"({len(designs)} refined): S–S {best['ss_A']:.2f} Å, χ3 {best['chi3_deg']:.0f}°, Cα "
                                f"shift {best['ca_rmsd_A']:.2f} Å; relative motion of the pair "
                                f"{rig.get('pair_relative_motion_change_pct', 0):+.0f}% (network model). "
                                "Structural design, not a measured stability.")
            else:
                out["_card"] = (f"No design passed the disulfide geometry filters ({len(designs)} refined from "
                                f"{out['n_candidates']} candidates).")
            return out

        return _launch(ctx, "design", args, workdir, done, f"Disulfide design for {object}",
                       module="leonardo.disulfide_runtime")
