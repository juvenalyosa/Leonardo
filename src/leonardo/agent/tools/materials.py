"""Materials and hybrid systems: metal surfaces/nanoparticles, graphene, proteins on surfaces (biosensors),
QM clusters for MOPAC/Psi4."""

import os
import re
import time

from ..registry import ToolError
from .components import _save_object
from .membrane import _launch

MATERIALS = ["Au", "Ag", "Pt", "Pd", "Cu", "Ni", "Al", "Pb", "graphene", "SiO2", "TiO2"]
_OXIDE_PARAMS = {"type": "object", "description": "TiO2 only (no bundled MM set): per element [charge, sigma_nm, "
                 "epsilon_kJ] from a published force field, e.g. {\"Ti\": [q, s, e], \"O\": [...]}; must be neutral",
                 "additionalProperties": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}}


def _cmd():
    from pymol import cmd

    return cmd


def register(reg):
    @reg.tool("build_material", "Build an inorganic model in the viewer: an fcc metal surface slab (Au, Ag, Pt, Pd, "
              "Cu, Ni, Al, Pb; facet 111/100/110, top layer at z=0, orthogonal periodic cell) or an icosahedral "
              "metal nanoparticle (shells 2-5: 13/55/147/309 atoms), a graphene sheet, hydroxylated silica (SiO2: "
              "alpha-quartz (001), geminal silanols 9.6 OH/nm², ClayFF) or rutile TiO2 (110). For QM clusters or to "
              "inspect a substrate; to put a protein on a surface use build_protein_surface_system.",
              {"properties": {
                  "material": {"type": "string", "enum": MATERIALS},
                  "kind": {"type": "string", "enum": ["slab", "nanoparticle"]},
                  "facet": {"type": "string", "enum": ["111", "100", "110", "001"],
                            "description": "metals; oxides use their own (SiO2 001, TiO2 110)"},
                  "size_nm": {"type": "number", "minimum": 1, "maximum": 30},
                  "layers": {"type": "integer", "minimum": 1, "maximum": 12},
                  "shells": {"type": "integer", "minimum": 2, "maximum": 5},
                  "name": {"type": "string"},
              }, "required": ["material"]},
              permission="view", title="Build material")
    def build_material(ctx, material, kind="slab", facet="111", size_nm=4.0, layers=4, shells=3, name=""):
        from ...materials_runtime import build_material as build
        from ...materials_runtime import write_material_pdb

        model = build(material, kind, facet, size_nm=(size_nm, size_nm), layers=layers, shells=shells)
        facet_tag = model.get("facet", facet)
        obj = name or (f"{material}{facet_tag}_slab" if kind == "slab" else f"{material}{model['n_atoms']}_np")
        obj = re.sub(r"[^A-Za-z0-9_]", "_", obj)[:60] or "material"
        cmd = _cmd()
        existing = set(cmd.get_names("objects"))
        if obj in existing:           # never replace one of the user's objects
            k = 2
            while f"{obj}_{k}" in existing:
                k += 1
            obj = f"{obj}_{k}"
        path = write_material_pdb(model, os.path.join(ctx.workspace("materials"), f"{obj}.pdb"))
        cmd.load(path, obj)
        if material in ("SiO2", "TiO2"):
            cmd.show_as("sticks", obj)
            cmd.show("spheres", f"{obj} and not elem O+H")
            cmd.set("sphere_scale", 0.3, obj)
        else:
            cmd.show_as("spheres", obj)
            cmd.set("sphere_scale", 0.45, obj)
        info = {k: v for k, v in model.items() if k not in ("positions_nm", "symbols", "types", "bonds", "charges")}
        info["object"], info["pdb"] = obj, path
        info["_card"] = f"**{obj}**: {model['n_atoms']} atoms" + (
            f", {model['cell_nm'][0]:.2f} x {model['cell_nm'][1]:.2f} nm" if model.get("cell_nm") else
            f", {model.get('diameter_nm')} nm across")
        return info

    @reg.tool("build_protein_surface_system", "Hybrid bio-inorganic system for MD (e.g. an enzyme/antibody on gold "
              "for a biosensor): protein with all components parameterized (component library), oriented so a chosen "
              "site faces the surface (toward_surface) or the solution (away_from_surface, e.g. the active site of a "
              "detector enzyme), placed gap_A above a frozen metal slab (INTERFACE-FF Lennard-Jones, Au/Ag/Pt/Pd/Cu/Ni/"
              "Al/Pb), graphene, hydroxylated silica (SiO2, ClayFF, neutral: SiO- at pH 7 not modelled) or rutile "
              "TiO2 (only with oxide_params from the user/literature), water and ions in a periodic box, minimized. Optional anchor_cys adds Cys SG-metal "
              "restraints (chemisorbed thiolate). Reports residues touching the surface, protein-surface LJ "
              "adsorption energy and the site height. No image charges: say so when discussing electrostatics. "
              "Background job; loads the system.",
              {"properties": {
                  "object": {"type": "string"},
                  "material": {"type": "string", "enum": MATERIALS},
                  "facet": {"type": "string", "enum": ["111", "100", "110", "001"]},
                  "layers": {"type": "integer", "minimum": 1, "maximum": 8},
                  "oxide_params": _OXIDE_PARAMS,
                  "toward_surface": {"type": "string", "description": "residues to face the surface, e.g. 'A:32 A:16'"},
                  "away_from_surface": {"type": "string", "description": "residues to face the solution, e.g. an active site"},
                  "gap_A": {"type": "number", "minimum": 1.5, "maximum": 20},
                  "water_above_A": {"type": "number", "minimum": 8, "maximum": 40},
                  "anchor_cys": {"type": "string", "description": "Cys residues bound to the metal, e.g. 'A:112'"},
                  "family": {"type": "string", "enum": ["amber14"],
                             "description": "substrate models use Amber-style nonbonded terms"},
                  "ionic_strength_M": {"type": "number", "minimum": 0, "maximum": 1},
                  "ph": {"type": "number", "minimum": 0, "maximum": 14},
              }, "required": ["object"]},
              permission="compute_heavy", title="Protein on surface", main_thread=True)
    def build_protein_surface_system(ctx, object, material="Au", facet="111", layers=4, toward_surface="",
                                     away_from_surface="", gap_A=3.0, water_above_A=15.0, anchor_cys="",
                                     family="amber14", ionic_strength_M=0.15, ph=7.0, oxide_params=None):
        if toward_surface and away_from_surface:
            raise ToolError("give toward_surface or away_from_surface, not both")
        if material not in ("SiO2", "TiO2") and str(facet) not in ("111", "100", "110"):
            raise ToolError(f"{material} slabs are built as the 111, 100 or 110 facet")
        if material in ("SiO2", "TiO2"):
            from ...materials_runtime import OXIDES, _oxide_forcefield_xml

            facet = OXIDES[material]
            if anchor_cys:
                raise ToolError("anchor_cys is a thiolate-metal model: not for oxides")
            if material == "TiO2":
                try:
                    _oxide_forcefield_xml("TiO2", {"symbols": ["Ti", "O", "O"]}, oxide_params)
                except ValueError as exc:
                    raise ToolError(str(exc)) from exc
        path = _save_object(ctx, object, sub="materials")
        workdir = os.path.join(ctx.workspace("materials"), time.strftime(f"{object}_{material}_%Y%m%d_%H%M%S"))
        args = {"pdb_path": path, "material": material, "facet": facet, "layers": int(layers), "family": family,
                "ph": float(ph), "gap_nm": float(gap_A) / 10, "toward_surface": toward_surface,
                "away_from_surface": away_from_surface, "water_above_nm": float(water_above_A) / 10,
                "ionic_strength_molar": float(ionic_strength_M), "anchor_cys": anchor_cys, "out_dir": workdir,
                "oxide_params": oxide_params}

        def done(res):
            cmd = _cmd()
            name = f"{object}_on_{material}"[:60]
            if name in cmd.get_names("objects"):
                cmd.delete(name)
            cmd.load(res["pdb"], name)
            cmd.hide("everything", name)
            cmd.show("cartoon", f"{name} and polymer.protein")
            cmd.show("spheres", f"{name} and chain Z")
            cmd.set("sphere_scale", 0.5, f"{name} and chain Z")
            sa = res.get("surface_analysis", {})
            res["object"] = name
            res["_card"] = (f"**{name}**: {res.get('n_atoms')} atoms; {sa.get('n_contact_residues')} residues touch "
                            f"{material}({facet}) (closest {sa.get('min_distance_A')} Å), protein-surface LJ "
                            f"{sa.get('protein_surface_LJ_kj_mol')} kJ/mol"
                            + (f", unscreened Coulomb {sa['protein_surface_coulomb_unscreened_kj_mol']} kJ/mol"
                               if "protein_surface_coulomb_unscreened_kj_mol" in sa else "")
                            + (f"; site {sa.get('site_height_above_surface_nm')} nm above the surface"
                               if sa.get("site_height_above_surface_nm") is not None else ""))
            return res

        return _launch(ctx, "build", args, workdir, done, f"{object} on {material}({facet})",
                       module="leonardo.materials_runtime")

    @reg.tool("cut_qm_cluster", "Cut a QM cluster around an adsorption/binding site of a structure (e.g. a protein on "
              "gold): whole residues within radius_A of the centre residues plus the substrate atoms within the same "
              "radius, loaded as a new object for run_mopac (PM7 covers Au/Ag/Pt/Cu...) or run_psi4 (use def2 basis "
              "sets with ECPs for heavy metals). Reports composition; confirm charge and multiplicity with the user.",
              {"properties": {"object": {"type": "string"},
                              "center": {"type": "string", "description": "residues, e.g. 'A:32' or 'A:112'"},
                              "radius_A": {"type": "number", "minimum": 2, "maximum": 10}},
               "required": ["object", "center"]},
              permission="compute_light", title="Cut QM cluster")
    def cut_qm_cluster(ctx, object, center, radius_A=4.5):
        from ...materials_runtime import cut_qm_cluster as cut

        path = _save_object(ctx, object, sub="qm_clusters")
        res = cut(path, center, radius_nm=float(radius_A) / 10)
        name = f"{object}_qm_{center.replace(':', '').replace(' ', '_')}"[:60]
        cmd = _cmd()
        if name in cmd.get_names("objects"):
            cmd.delete(name)
        cmd.load(res["xyz"], name)
        res["object"] = name
        res["_card"] = (f"QM cluster **{name}**: {res['n_atoms']} atoms "
                        + ", ".join(f"{k}{v}" for k, v in sorted(res["composition"].items())))
        return res
