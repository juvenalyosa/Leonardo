"""Component library: which parameters every cofactor, ion, lipid, sugar and ligand of a structure gets,
and a validated OpenMM system built from them."""

import os
import time

from ..registry import ToolError

FAMILIES = ["amber14", "charmm36"]


def _cmd():
    from pymol import cmd

    return cmd


def _save_object(ctx, object, sub="components"):
    cmd = _cmd()
    if object not in cmd.get_names("objects"):
        raise ToolError(f"no object named {object!r}; call list_objects")
    path = os.path.join(ctx.workspace(sub), time.strftime(f"{object}_%Y%m%d_%H%M%S.pdb"))
    # CONECT records carry the component bonds; alternate locations would duplicate atoms
    previous = cmd.get("pdb_conect_all")
    cmd.set("pdb_conect_all", 1)
    try:
        cmd.save(path, f"({object}) and not alt B+C+D", state=1)
    finally:
        cmd.set("pdb_conect_all", previous)
    return path


def plan_rows(plan):
    """Compact per-residue-type view of a resolver plan (copies grouped)."""
    groups = {}
    for r in plan.get("components", []):
        key = (r["resn"], r["source"], r.get("template", ""))
        g = groups.setdefault(key, {"resn": r["resn"], "source": r["source"], "template": r.get("template"),
                                    "family": r.get("family"), "charge": r.get("charge"), "copies": 0,
                                    "category": r.get("category"), "matched_by": r.get("matched_by"),
                                    "confidence": r.get("confidence"), "where": []})
        g["copies"] += 1
        if len(g["where"]) < 6:
            g["where"].append(f"{r['chain']}{r['resi']}")
        for k in ("reason", "notes", "available_in", "description"):
            if r.get(k) and k not in g:
                g[k] = r[k]
    return list(groups.values())


def register(reg):
    @reg.tool("resolve_components", "Before MD, QM/MM or any system build: for every non-protein residue of a "
              "structure (cofactors like heme/NAD/ATP/FAD, metal ions, lipids, sugars, ligands, crystallization "
              "additives, modified residues) say where its force-field parameters come from — a curated library "
              "template (CHARMM36 heme/nucleotides/lipids, Amber ions/lipid21/GLYCAM) matched by name or by "
              "chemical graph, GAFF2/OpenFF for other organics, or 'metal'/'modified_residue'/'unmatched' when a "
              "special model is needed — the total component charges, metal-coordination contacts and which "
              "force-field family covers the system best. Phosphoserine/-threonine/-tyrosine (SEP/TPO/PTR) are "
              "covered (CHARMM36 patches for SEP/PTR; Amber phosaa14SB for all three). Metal sites (heme Fe-His/Cys, "
              "Zn-Cys/His, Mg...) are kept with distance restraints and the right Cys/His protonation. "
              "Components covalently bound to the protein are detected ('covalent', with the partner atom): "
              "Cys-linked ligands (acrylamides, chloroacetamides...) are buildable in amber14; ligands on Lys/Ser/"
              "His... (retinal Schiff base, acetyl-Lys) are built as one merged residue (CHARMM36 template, e.g. "
              "LYR) or a GAFF2 non-canonical residue (amber14, protonated Schiff base); c-type heme (HEC) goes to "
              "MOPAC QM/MM; nothing covalent is modelled as a free molecule. "
              "Leonardo removes residues without parameters before MD, "
              "so run this first and follow recommended_family.",
              {"properties": {"object": {"type": "string"},
                              "family": {"type": "string", "enum": FAMILIES}},
               "required": ["object"]},
              permission="read", title="Component parameters")
    def resolve_components(ctx, object, family="amber14"):
        from openmm import app, unit

        from ...component_library_runtime import resolve_components as resolve

        path = _save_object(ctx, object)
        pdb = app.PDBFile(path)
        pos = [tuple(p.value_in_unit(unit.nanometer)) for p in pdb.positions]
        plan = resolve(pdb.topology, pos, family=family)
        rows = plan_rows(plan)
        out = {"object": object, "family": family, "recommended_family": plan["recommended_family"],
               "ready_for_md": plan["ready"], "components": rows, "counts": plan["counts"],
               "metal_coordination": plan["metal_coordination"][:12], "warnings": plan["warnings"][:10]}
        lib = ", ".join(f"{r['resn']}→{r['template']}" for r in rows if r["source"] == "library") or "none"
        other = ", ".join(f"{r['resn']} ({r['source']})" for r in rows if r["source"] != "library")
        out["_card"] = (f"Components of **{object}** in {family}: library {lib}"
                        + (f"; needs attention: {other}" if other else "")
                        + (f". Recommended: **{plan['recommended_family']}**"
                           if plan["recommended_family"] != family else ""))
        return out

    @reg.tool("build_validated_system", "Build and check an OpenMM system of a structure with all its components "
              "parameterized from the component library: templates for every residue, template-derived "
              "hydrogens, integer total charge, and a short stable minimization (vacuum, or explicit water with "
              "solvate=true). Crystallization additives are removed unless keep_additives. Writes the minimized "
              "PDB, the serialized System XML and the component plan, and loads the minimized structure. Ligands "
              "without a curated template get GAFF2 + AM1-BCC parameters (bond orders from the PDB chemical "
              "component dictionary, protonation at the given pH with Leonardo's pKa model; cached). Refuses (and "
              "says why) when a component still needs a metal or modified-residue model. Covalent ligands on Cys "
              "(amber14) get a capped-fragment GAFF2 template bonded to the Cys SG (Cys becomes CYX).",
              {"properties": {"object": {"type": "string"},
                              "family": {"type": "string", "enum": FAMILIES},
                              "ph": {"type": "number", "minimum": 0, "maximum": 14},
                              "keep_waters": {"type": "boolean"},
                              "keep_additives": {"type": "boolean"},
                              "remove_residues": {"type": "array", "items": {"type": "string"}},
                              "solvate": {"type": "boolean"},
                              "padding_A": {"type": "number", "minimum": 5, "maximum": 30},
                              "ionic_strength_M": {"type": "number", "minimum": 0, "maximum": 2},
                              "minimize_steps": {"type": "integer", "minimum": 0, "maximum": 5000},
                              "metal_restraints": {"type": "boolean", "description": "flat-bottom restraints that "
                                                   "keep metal-ligand distances (default true)"}},
               "required": ["object"]},
              permission="compute_light", title="Build validated system")
    def build_validated_system(ctx, object, family="charmm36", ph=7.0, keep_waters=False, keep_additives=False,
                               remove_residues=None, solvate=False, padding_A=10.0, ionic_strength_M=0.15,
                               minimize_steps=300, metal_restraints=True):
        from ...component_library_runtime import build_and_validate

        path = _save_object(ctx, object)
        out_dir = os.path.join(ctx.workspace("systems"), time.strftime(f"{object}_{family}_%Y%m%d_%H%M%S"))
        rep = build_and_validate(path, family=family, ph=ph, keep_waters=keep_waters,
                                 remove_resn=list(remove_residues or []), remove_additives=not keep_additives,
                                 solvate=solvate, padding_nm=float(padding_A) / 10.0,
                                 ionic_strength_molar=ionic_strength_M, minimize_steps=minimize_steps, out_dir=out_dir,
                                 metal_restraints=metal_restraints)
        plan = rep.pop("plan", {}) or {}
        rep["components"] = plan_rows(plan)
        rep["recommended_family"] = plan.get("recommended_family")
        rep["warnings"] = plan.get("warnings", [])[:10]
        if not rep.get("ok"):
            raise ToolError(str(rep.get("error") or "the system did not validate")
                            + (f" (recommended family: {plan.get('recommended_family')})"
                               if plan.get("recommended_family") not in (None, family) else ""))
        name = f"{object}_{family}_sys"[:60]
        cmd = _cmd()
        if name in cmd.get_names("objects"):
            cmd.delete(name)
        cmd.load(rep["pdb"], name)
        rep["object"] = name
        rep["_card"] = (f"System **{name}** ({family}): {rep['n_atoms']} atoms, total charge {rep['total_charge']:+.0f}, "
                        f"minimized {rep['energy_start_kj']} → {rep['energy_minimized_kj']} kJ/mol; components "
                        + ", ".join(f"{k} {v:+g}" for k, v in list(rep.get("component_charges", {}).items())[:6]))
        return rep


def register_qm(reg):
    @reg.tool("plan_qm_region", "Propose MOPAC QM/MM regions for sites a force field cannot describe or that "
              "deserve QM: metal cofactors (heme b/c, Fe-S...), covalently bound metal cofactors (c-type heme), "
              "metal clusters and protein-bound metal ions (Zn, Mg, Mn...). One proposal per site: the component "
              "plus the side chains coordinating its metal and its covalent partners (cut at CA-CB, capped with "
              "link atoms), a PyMOL selection for run_qmmm (region_mode='pymol_selection'), the charge breakdown "
              "(organic part from the PDB chemical component, Cys thiolates, Asp/Glu, Lys/Arg) and the metal "
              "oxidation/spin options. Components without MM parameters are put in the QM region automatically "
              "during run_qmmm. ALWAYS confirm the oxidation state and spin (qm_charge, qm_multiplicity) with the "
              "user; keep crystal waters that coordinate the metal.",
              {"properties": {"object": {"type": "string"},
                              "family": {"type": "string", "enum": FAMILIES},
                              "ph": {"type": "number", "minimum": 0, "maximum": 14}},
               "required": ["object"]},
              permission="read", title="QM/MM region plan")
    def plan_qm_region(ctx, object, family="amber14", ph=7.4):
        from openmm import app, unit

        from ...component_library_runtime import plan_qm_region as plan_region
        from ...component_library_runtime import resolve_components as resolve

        path = _save_object(ctx, object, sub="qm_plans")
        pdb = app.PDBFile(path)
        pos = [tuple(p.value_in_unit(unit.nanometer)) for p in pdb.positions]
        plan = resolve(pdb.topology, pos, family=family)
        out = plan_region(pdb.topology, pos, plan, ph=ph)
        out["object"] = object
        if out["sites"]:
            first = out["sites"][0]
            opts = "; ".join(f"{o['metal']}({o['oxidation']}) -> charge {o['qm_charge']}, mult {o['multiplicities']}"
                             for o in first["charge_options"])
            out["_card"] = (f"QM/MM site **{first['component']}** + {len(first['side_chains'])} side chain(s): "
                            + (opts or "no metal") + (f" · {len(out['sites'])} sites" if len(out["sites"]) > 1 else ""))
        return out
