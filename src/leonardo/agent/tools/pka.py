"""pKa / protonation states of small molecules and the docking ligand."""

import os

from ..registry import ToolError

LIGAND_SOURCE = {
    "smiles": {"type": "string", "description": "SMILES of the ligand."},
    "selection": {"type": "string", "description": "PyMOL selection of a ligand in a structure, e.g. "
                                                   "'3PTB and resn BEN' (bond orders from the PDB CCD template)."},
    "file": {"type": "string", "description": "Path to an SDF/MOL2/PDB ligand file."},
}
METHOD = {"type": "string", "enum": ["empirical", "pm7", "psi4"],
          "description": "empirical: instant, class-typical pKa with site coupling; pm7: PM7/COSMO, seconds to "
                         "minutes; psi4: experimental, slow."}


def _molecule(ctx, smiles="", selection="", file=""):
    from rdkit import Chem

    given = [bool(smiles), bool(selection), bool(file)]
    if sum(given) != 1:
        raise ToolError("give exactly one of smiles, selection or file")
    if smiles:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ToolError("RDKit could not parse the SMILES")
        return mol, f"SMILES {smiles}"
    if file:
        path = os.path.expanduser(str(file))
        if not os.path.isfile(path):
            raise ToolError(f"file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        if ext == ".sdf":
            mol = next((m for m in Chem.SDMolSupplier(path) if m is not None), None)
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(path)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBFile(path)
        else:
            raise ToolError("ligand files: .sdf, .mol2 or .pdb")
        if mol is None:
            raise ToolError("RDKit could not read the file")
        return mol, os.path.basename(path)

    def from_selection():
        from pymol import cmd

        if cmd.count_atoms(f"({selection}) and not hydro") <= 0:
            raise ToolError(f"selection {selection!r} is empty")
        try:
            # the tool's own selection: the user's 'sele' is left untouched
            mol, _label = ctx.dlg._pka_selection_molecule(str(selection))
        except RuntimeError as exc:
            raise ToolError(str(exc)) from exc
        return mol
    return ctx.run_on_main(from_selection), f"selection {selection}"


def _site_rows(result):
    return [{"atom": s["atom"], "class": s["class"], "kind": s["kind"], "pka": round(float(s.get("pka", 0)), 2),
             "uncertainty": round(float(s.get("sd", 0)), 2), "source": s.get("pka_source", ""),
             "flags": s.get("flags", [])} for s in result.get("sites", [])]


def register(reg):
    @reg.tool("compute_pka", "pKa of each ionizable site of a small molecule, macroscopic pKa values and the "
              "dominant protonation state/charge at a pH. Does not change the scene or the docking ligand.",
              {"properties": dict(LIGAND_SOURCE, method=METHOD, ph={"type": "number", "minimum": 0, "maximum": 14}),
               "required": []},
              permission="compute_light", title="Compute pKa", main_thread=False)
    def compute_pka(ctx, smiles="", selection="", file="", method="empirical", ph=7.4):
        from ... import pka_runtime as P

        mol, label = _molecule(ctx, smiles, selection, file)
        options = ctx.run_on_main(lambda: ctx.dlg._pka_backend_options(method)) if ctx.dlg is not None else None
        res = P.compute_pka(mol, method=method, focus_ph=float(ph), backend_options=options)
        states = P.states_at_ph(res, float(ph), min_population=0.05)
        return {
            "molecule": label, "smiles": res.get("smiles"), "method": res.get("method"), "ph": float(ph),
            "sites": _site_rows(res), "macroscopic_pkas": res.get("macro_pkas", []),
            "dominant_charge": states[0]["charge"] if states else None,
            "dominant_charge_population": round(float(states[0].get("charge_population", 1.0)), 3) if states else None,
            "mean_charge": round(float(P.mean_charge(res, float(ph)) or 0.0), 2),
            "warnings": res.get("warnings", [])[:8],
            "_card": ("pKa: " + ", ".join(f"{s['class']} {float(s.get('pka', 0)):.1f}" for s in res.get("sites", []))
                      + f"; dominant charge at pH {ph:g}: "
                      + (f"{states[0]['charge']:+d}" if states and states[0]['charge'] is not None else "n/a")),
        }

    @reg.tool("prepare_docking_ligand", "Protonate a ligand at a pH with Leonardo's pKa module and set it as the "
              "docking ligand (its protonation is locked so docking keeps it). Crystal ligands keep their "
              "coordinates; SMILES are embedded in 3D.",
              {"properties": dict(LIGAND_SOURCE, method=METHOD, ph={"type": "number", "minimum": 0, "maximum": 14}),
               "required": []},
              permission="compute_light", title="Prepare ligand", main_thread=False)
    def prepare_docking_ligand(ctx, smiles="", selection="", file="", method="empirical", ph=7.4):
        from ... import pka_runtime as P

        mol, label = _molecule(ctx, smiles, selection, file)
        options = ctx.run_on_main(lambda: ctx.dlg._pka_backend_options(method))
        res = P.compute_pka(mol, method=method, focus_ph=float(ph), backend_options=options)
        best = P.states_at_ph(res, float(ph), min_population=0.0)[0]
        state_mol = P.molecule_for_state(mol, res, best["state"], embed_if_2d=True)
        from rdkit import Chem

        info = {"method": res.get("method"), "ph": float(ph), "state": best["state"],
                "population": float(best["population"]), "charge": int(Chem.GetFormalCharge(state_mol)),
                "sites": res.get("sites", []), "warnings": res.get("warnings", []),
                "calibration_version": res.get("provenance", {}).get("calibration_version", "none")}

        def hand_off():
            from pymol import cmd

            from .. import presentation

            message = ctx.dlg._pka_use_state(state_mol, info, target="docking")
            if "dock_ligand_preview" in cmd.get_names("objects"):
                presentation.style_ligand("dock_ligand_preview")
                presentation.focus("dock_ligand_preview")
            return {"message": message, "ligand_file": str(getattr(ctx.dlg, "_docking_ligand_file", "") or ""),
                    "scene": ctx.scene(f"docking ligand ({label}), charge {info['charge']:+d} at pH {ph:g}")}
        out = ctx.run_on_main(hand_off)
        out["_card"] = (f"Ligand ready for docking: `{Chem.MolToSmiles(Chem.RemoveHs(state_mol))}`, charge "
                        f"**{info['charge']:+d}** at pH {ph:g} ({info['method']}; population "
                        f"{100 * float(best['charge_population']):.0f}%).")
        out.update({"molecule": label, "smiles": Chem.MolToSmiles(Chem.RemoveHs(state_mol)),
                    "net_charge": info["charge"], "charge_population": round(float(best["charge_population"]), 3),
                    "method": info["method"], "sites": _site_rows(res), "warnings": info["warnings"][:6]})
        return out
