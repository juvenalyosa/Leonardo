"""Ligand Tools: ligand from SMILES / file / PubChem, Amber (GAFF) parameterization, PDBQT and OpenMM XML."""

import os
import re
import time

from .. import presentation
from ..registry import ToolError
from . import _headless
from ._dialogjob import start_deferred_job, start_dialog_job
from ._widgets import set_combo, set_value


def _ligand_state(dlg):
    keys = {"ligand_file": "_docking_ligand_file", "charged_mol2": "_parameterized_ligand_mol2",
            "frcmod": "_parameterized_ligand_frcmod", "prmtop": "_parameterized_ligand_prmtop",
            "openmm_xml": "_parameterized_ligand_xml", "pdbqt": "_docking_ligand_pdbqt"}
    out = {k: str(getattr(dlg, a, "") or "") for k, a in keys.items()}
    out["charge_source"] = str(getattr(dlg, "_ligand_charge_source", "") or "")
    return {k: v for k, v in out.items() if v}


def _show_preview():
    from pymol import cmd

    if "dock_ligand_preview" in cmd.get_names("objects"):
        presentation.style_ligand("dock_ligand_preview")
        presentation.focus("dock_ligand_preview", buffer=4.0)


def register(reg):
    @reg.tool("set_ligand_from_smiles", "Make a SMILES string the active ligand for docking, screening seeds and "
              "parameterization: 3D structure with RDKit (ETKDG + MMFF), shown in the viewer. For a ligand "
              "already in a structure use prepare_docking_ligand instead (it also sets the protonation state).",
              {"properties": {"smiles": {"type": "string"}, "name": {"type": "string"}}, "required": ["smiles"]},
              permission="compute_light", title="Ligand from SMILES")
    def set_ligand_from_smiles(ctx, smiles, name="ligand"):
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ToolError(f"RDKit cannot parse the SMILES {smiles!r}")
        molh = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(molh, randomSeed=0xF00D) != 0:
            if AllChem.EmbedMolecule(molh, randomSeed=0xF00D, useRandomCoords=True) != 0:
                raise ToolError("could not build 3D coordinates")
        try:
            AllChem.MMFFOptimizeMolecule(molh, maxIters=2000)
        except Exception:
            AllChem.UFFOptimizeMolecule(molh, maxIters=2000)
        safe = re.sub(r"[^A-Za-z0-9_]", "_", str(name)) or "ligand"
        molh.SetProp("_Name", safe)
        path = os.path.join(ctx.workspace("ligands"), time.strftime(f"{safe}_%Y%m%d_%H%M%S.sdf"))
        with Chem.SDWriter(path) as w:
            w.write(molh)
        with _headless.quiet():
            ctx.dlg._set_docking_ligand_file(path, source_label="smiles")
        _show_preview()
        charge = Chem.GetFormalCharge(mol)
        return {"ligand_file": path, "smiles": Chem.MolToSmiles(mol), "heavy_atoms": mol.GetNumHeavyAtoms(),
                "formal_charge": charge, "scene": ctx.scene(f"ligand {safe} from SMILES"),
                "note": "Protonation is as written in the SMILES; use compute_pka to check the state at a pH.",
                "_card": f"Ligand **{safe}** ({mol.GetNumHeavyAtoms()} heavy atoms, charge {charge:+d}) is the "
                         "active ligand."}

    @reg.tool("load_ligand_file", "Make a ligand file on disk (.sdf, .mol2, .mol, .pdb, .pdbqt) the active ligand.",
              {"properties": {"path": {"type": "string"}}, "required": ["path"]},
              permission="compute_light", title="Ligand from file")
    def load_ligand_file(ctx, path):
        path = os.path.abspath(os.path.expanduser(str(path)))
        if not os.path.isfile(path):
            raise ToolError(f"no such file: {path}")
        if os.path.splitext(path)[1].lower() not in (".sdf", ".mol2", ".mol", ".pdb", ".pdbqt"):
            raise ToolError("ligand files must be .sdf, .mol2, .mol, .pdb or .pdbqt")
        with _headless.quiet():
            ctx.dlg._set_docking_ligand_file(path, source_label="file")
        _show_preview()
        return {"ligand_file": path, "scene": ctx.scene(f"ligand {os.path.basename(path)}"),
                "_card": f"Active ligand: `{os.path.basename(path)}`."}

    @reg.tool("search_pubchem", "Search PubChem by name, identifier or SMILES. Returns CID, title, formula, "
              "molecular weight and SMILES of the hits.",
              {"properties": {"query": {"type": "string"}}, "required": ["query"]},
              permission="read", title="PubChem search")
    def search_pubchem(ctx, query):
        dlg = ctx.dlg
        set_value(dlg, "edit_pubchem_query", str(query))

        def collect():
            hits = list(getattr(dlg, "_pubchem_hits", []) or [])
            rows = [{k: h.get(k) for k in ("cid", "title", "formula", "mw", "isomeric_smiles", "smiles")}
                    for h in hits[:25]]
            return {"query": query, "hits": rows,
                    "_card": f"**{len(hits)} PubChem hits** for “{query}”"
                             + (": " + "; ".join(f"CID {r['cid']} {r['title']}" for r in rows[:3]) if rows else ".")}

        def start():
            dlg._pubchem_hits = []          # a failed search must not return the previous query's hits
            dlg._on_search_pubchem_compounds()

        return start_dialog_job(ctx, "pubchem", start,
                                lambda: not bool(getattr(dlg, "_pubchem_search_busy", False)), collect,
                                description=f"PubChem search {query}")

    @reg.tool("use_pubchem_compound", "Download a PubChem hit (by CID, from search_pubchem) as a 3D structure and "
              "make it the active ligand.",
              {"properties": {"cid": {"type": "integer"}}, "required": ["cid"]},
              permission="compute_light", title="Use PubChem compound")
    def use_pubchem_compound(ctx, cid):
        dlg = ctx.dlg
        hits = list(getattr(dlg, "_pubchem_hits", []) or [])
        rows = [i for i, h in enumerate(hits) if str(h.get("cid")) == str(cid)]
        if not rows:
            raise ToolError(f"CID {cid} is not among the last PubChem hits; run search_pubchem first")
        dlg.tbl_pubchem.selectRow(rows[0])
        before = getattr(dlg, "_docking_ligand_file", None)
        with _headless.quiet() as q:
            dlg._on_use_selected_pubchem_hit()
        path = str(getattr(dlg, "_docking_ligand_file", "") or "")
        # the file is named after the CID: re-using the active compound gives the same path
        if not path or (path == before and f"cid{int(cid)}" not in os.path.basename(path).lower()):
            raise ToolError("; ".join(m["text"] for m in q.messages if m["kind"] in ("warning", "critical"))
                            or "the PubChem download did not produce a ligand file")
        _show_preview()
        h = hits[rows[0]]
        return {"cid": cid, "title": h.get("title"), "ligand_file": path,
                "scene": ctx.scene(f"PubChem CID {cid} ({h.get('title')})"),
                "_card": f"PubChem **CID {cid}** ({h.get('title')}) is the active ligand."}

    @reg.tool("parametrize_ligand", "Build the Amber force field of the active ligand with antechamber: partial "
              "charges (AM1-BCC by default), GAFF2 atom types, frcmod, prmtop and the OpenMM XML. Needed before "
              "MD/ABFE of a complex with this ligand when automatic parameterization is not enough. Takes about "
              "0.5-5 min (AM1-BCC).",
              {"properties": {
                  "charge_method": {"type": "string", "enum": ["bcc", "gas", "rc"],
                                    "description": "bcc = AM1-BCC; gas = Gasteiger; rc = keep charges in the MOL2"},
                  "atom_types": {"type": "string", "enum": ["gaff2", "gaff"]},
                  "net_charge": {"type": "integer", "minimum": -8, "maximum": 8,
                                 "description": "only used when RDKit cannot tell the formal charge"},
              }},
              permission="compute_heavy", title="Ligand parameterization (antechamber)")
    def parametrize_ligand(ctx, charge_method="bcc", atom_types="gaff2", net_charge=None):
        dlg = ctx.dlg
        if not str(getattr(dlg, "_docking_ligand_file", "") or ""):
            raise ToolError("no active ligand: set one first (set_ligand_from_smiles, prepare_docking_ligand, ...)")
        set_combo(dlg, "cmb_charge_method", charge_method)
        set_combo(dlg, "cmb_atom_type", atom_types)
        if net_charge is not None:
            set_value(dlg, "spin_net_charge", int(net_charge))

        def _stamp():
            path = str(getattr(dlg, "_parameterized_ligand_mol2", "") or "")
            try:
                return path, os.path.getmtime(path)
            except OSError:
                return path, None
        before = _stamp()

        def collect(outcome):
            if outcome["error"]:
                raise ToolError(outcome["error"])
            state = _ligand_state(dlg)
            if not state.get("charged_mol2"):
                raise ToolError("antechamber did not produce a charged MOL2 (see Leonardo's message)")
            if before[0] and _stamp() == before:
                raise ToolError("antechamber did not produce new files (the previous parameterization is "
                                "unchanged; see Leonardo's message)")
            return dict(state, _card=f"Ligand parameterized: {charge_method.upper()} charges, {atom_types.upper()} "
                                     f"types; OpenMM XML {'ready' if state.get('openmm_xml') else 'not built'}.")

        return start_deferred_job(ctx, "parametrize", dlg._on_parametrize_ligand_with_antechamber, collect,
                                  description="antechamber parameterization")

    @reg.tool("export_ligand_files", "Write docking/simulation files of the active ligand: 'pdbqt' (for smina/Vina, "
              "keeping the current charges or recomputing Gasteiger) and/or 'openmm_xml' (needs parametrize_ligand).",
              {"properties": {
                  "formats": {"type": "array", "items": {"type": "string", "enum": ["pdbqt", "openmm_xml"]}},
                  "pdbqt_charges": {"type": "string", "enum": ["preserve", "gasteiger"]},
              }, "required": ["formats"]},
              permission="compute_light", title="Export ligand files")
    def export_ligand_files(ctx, formats, pdbqt_charges="preserve"):
        dlg = ctx.dlg
        with _headless.quiet() as q:
            if "pdbqt" in formats:
                set_combo(dlg, "cmb_ligand_pdbqt_charge", pdbqt_charges)
                dlg._on_export_ligand_pdbqt_from_source()
            if "openmm_xml" in formats:
                dlg._on_export_ligand_openmm_xml()
        state = _ligand_state(dlg)
        errors = [m["text"] for m in q.messages if m["kind"] in ("warning", "critical")]
        missing = [f for f in formats if not state.get("pdbqt" if f == "pdbqt" else "openmm_xml")]
        if missing:
            raise ToolError(f"could not write {', '.join(missing)}: " + ("; ".join(errors) or "see Leonardo"))
        return dict(state, _card="Ligand files written: " + ", ".join(formats) + ".")

    @reg.tool("ligand_status", "What the active ligand is and which prepared files exist (charges, force field, "
              "PDBQT, OpenMM XML).", permission="read", title="Ligand status")
    def ligand_status(ctx):
        return _ligand_state(ctx.dlg) or {"ligand_file": None, "note": "no active ligand"}
