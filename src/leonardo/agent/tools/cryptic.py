"""Cryptic pockets: sites that open with the protein's motions (Crypsis 0.2: normal-mode ensembles, a C++ pocket grid,
49 physical descriptors and gradient-boosted trees, optionally with ESM-2 embeddings)."""

import os
import threading
import time

from ..registry import ToolError


def _cmd():
    from pymol import cmd

    return cmd


def _save(ctx, object, chains):
    cmd = _cmd()
    if object not in cmd.get_names("objects"):
        raise ToolError(f"no object named {object!r}; call list_objects")
    sel = f"({object}) and polymer.protein and not hydro and not alt B+C+D"
    if chains:
        sel += " and chain " + "+".join(chains)
    if cmd.count_atoms(sel + " and name CA") < 10:
        raise ToolError("need a protein with at least 10 residues")
    path = os.path.join(ctx.workspace("cryptic_pockets"), time.strftime(f"{object}_%Y%m%d_%H%M%S.pdb"))
    cmd.save(path, sel, state=1)
    return path


def _resi_sel(object, keys):
    parts = []
    for key in keys:
        chain, _sep, resi = str(key).partition("_")
        resi = "\\" + resi if resi.startswith("-") else resi
        parts.append(f"(chain {chain} and resi {resi})" if chain.strip() else f"(resi {resi})")
    return f"({object}) and (" + " or ".join(parts) + ")" if parts else "none"


def show_result(object, result, cmd=None, top=5):
    """Colour the object by cryptic-site probability and make one selection per site."""
    cmd = cmd or _cmd()
    probs = dict(zip(result["residues"], (float(p) for p in result["probability"])))
    space = {"probs": probs}
    cmd.alter(f"({object}) and polymer.protein",
              "b = probs.get(chain + '_' + resi, 0.0)", space=space)
    cmd.show("cartoon", object)
    cmd.hide("nonbonded", f"({object}) and solvent")
    cmd.spectrum("b", "blue_white_red", f"({object}) and polymer.protein", minimum=0.0,
                 maximum=max(0.3, max(probs.values() or [0.3])))
    names = []
    for site in result["sites"][:top]:
        name = f"{object}_cryptic_{site['rank']}"
        cmd.select(name, _resi_sel(object, site["residues"]), enable=0)
        cmd.show("sticks", f"{name} and not name N+C+O")
        names.append(name)
    return names


def register(reg):
    @reg.tool("find_cryptic_pockets", "Where pockets can OPEN that the structure does not show (cryptic sites), "
              "with Crypsis 0.2 (developed from Leonardo's method). Normal modes of elastic networks generate "
              "ensembles of conformations (with full and cut-back side chains); in each, a C++ grid finds buried, "
              "probe-accessible pockets; 49 physical descriptors per residue (pockets, exposure, hinges) go into "
              "gradient-boosted trees fitted on CryptoBench. Sites are ranked; each says whether it is already open, "
              "grows or forms with motion, with pocket volumes in the structure vs the ensemble. CryptoBench test "
              "(222 apo structures), residue AUC: physics model 0.86, hybrid with ESM-2 650M 0.89 (P2Rank 0.79, "
              "PocketMiner 0.76). model='auto' (default) takes the most accurate model this computer can run: the "
              "hybrid with ESM-2 650M from 16 GB of memory, 150M from 8 GB, the physics model otherwise (10-60 s). "
              "The object is coloured by probability "
              "(blue->red) and the sites are selections <object>_cryptic_<rank>. Each site has a docking_box (center, "
              "size in A) to pass to set_docking_box before run_docking. Use apo structures; follow up with docking "
              "or MD.",
              {"properties": {
                  "object": {"type": "string"},
                  "chains": {"type": "string", "description": "e.g. 'A' or 'A,B' (default all protein chains)"},
                  "top": {"type": "integer", "minimum": 1, "maximum": 20, "description": "sites reported (default 5)"},
                  "model": {"type": "string", "enum": ["auto", "physics", "hybrid", "v1"],
                            "description": "auto (default), physics (fastest), hybrid (+ ESM-2, most accurate), "
                                           "v1 (Leonardo's original)"},
                  "esm": {"type": "string", "enum": ["650M", "150M"],
                          "description": "with model='hybrid': its language model (650M default; 150M needs less "
                                         "memory)"},
              }, "required": ["object"]},
              permission="compute_light", title="Cryptic pockets")
    def find_cryptic_pockets(ctx, object, chains="", top=5, model="auto", esm="650M"):
        from ... import cryptic_pocket_runtime as cp

        if model not in cp.MODELS:
            raise ToolError(f"model must be one of {', '.join(cp.MODELS)}")
        if esm not in cp.ESM_SIZES:
            raise ToolError(f"esm must be one of {', '.join(cp.ESM_SIZES)}")
        if model == "hybrid" and not cp.have_esm():
            raise ToolError("the hybrid model needs PyTorch and fair-esm in Leonardo's environment (the installer "
                            "adds them; or pip install fair-esm); use model='physics'")
        if model == "auto":
            model, esm = cp.default_model()
        chain_list = [c.strip() for c in str(chains or "").split(",") if c.strip()]
        path = _save(ctx, object, chain_list)
        state = {"progress": "reading structure", "result": None, "error": None}

        def work():
            try:
                chain = cp.read_chain(path)
                state["n_res"] = chain.n_res
                state["result"] = cp.analyze(
                    chain, progress=lambda k, n: state.__setitem__("progress", f"conformer {k}/{n}"),
                    model=model, esm=esm)
            except Exception as exc:
                state["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        def poll():
            if thread.is_alive():
                return {"done": False, "progress": state["progress"]}
            if state["error"]:
                return {"done": True, "error": state["error"]}
            res = state["result"]
            names = show_result(object, res, top=int(top))
            sites = []
            for s in res["sites"][: int(top)]:
                sites.append({k: s[k] for k in ("rank", "kind", "n_residues", "mean_probability",
                                                 "pocket_volume_structure_A3", "pocket_volume_ensemble_A3",
                                                 "opens_in_ensemble", "residues", "centre", "docking_box")}
                             | {"selection": f"{object}_cryptic_{s['rank']}"})
            lines = [f"#{s['rank']} {s['kind']} ({s['n_residues']} residues, "
                     f"{s['pocket_volume_structure_A3']:.0f}->{s['pocket_volume_ensemble_A3']:.0f} A^3)" for s in sites]
            name = f"hybrid, ESM-2 {res['esm']}" if res.get("esm") else res["model"]
            card = (f"**{object}**: {len(res['sites'])} candidate site(s) from {state.get('n_res')} residues "
                    f"(Crypsis {name} model, " + ("C++ grid" if res["extension"] else "numpy grid") + "). "
                    + "; ".join(lines[:5])
                    + ". Coloured by probability (blue->red). Predictions, not proof: confirm with MD or docking.")
            return {"done": True, "result": {"object": object, "sites": sites, "selections": names,
                                             "model": res["model"], "esm": res.get("esm"),
                                             "threshold": res["threshold"], "extension": res["extension"],
                                             "residues_called": int((res["probability"] >= res["threshold"]).sum()),
                                             "benchmark": res["benchmark"], "_card": card}}

        return ctx.jobs.start("cryptic_pockets", poll, description=f"Cryptic pockets of {object}")
