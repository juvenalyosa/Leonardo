"""Structures: fetch (RCSB, AlphaFold DB), load files, list and describe what is in the session."""

import os

from ..registry import ToolError

_WATER = "HOH+WAT+DOD+H2O"
_IONS = "NA+K+CL+MG+CA+ZN+MN+FE+CU+CO+NI+CD+HG+SO4+PO4+IOD+BR"


def _cmd():
    from pymol import cmd

    return cmd


def _object_summary(obj):
    cmd = _cmd()
    chains = cmd.get_chains(obj)
    n_res = {}
    for ch in chains:
        space = {"res": set()}
        cmd.iterate(f"({obj}) and polymer and chain {ch} and name CA", "res.add(resi)", space=space)
        n_res[ch or "-"] = len(space["res"])
    ligands = []
    space = {"seen": {}}
    cmd.iterate(f"({obj}) and not polymer and not resn {_WATER}",
                "seen.setdefault((resn, chain, resi), 0); seen[(resn, chain, resi)] += 1", space=space)
    for (resn, chain, resi), n in sorted(space["seen"].items()):
        ligands.append({"resn": resn, "chain": chain, "resi": resi, "atoms": n,
                        "kind": "ion" if resn in _IONS.split("+") else "ligand/cofactor"})
    waters = cmd.count_atoms(f"({obj}) and resn {_WATER} and not elem H")
    return {"object": obj, "atoms": cmd.count_atoms(obj), "chains": n_res, "hetero_groups": ligands[:40],
            "hetero_groups_total": len(ligands), "waters": waters, "states": cmd.count_states(obj)}


def register_in_leonardo(ctx, name, path):
    """Make Leonardo's modules see the new structure without _load_structure_file_into_session
    (which reinitializes PyMOL and would wipe the user's session)."""
    dlg = ctx.dlg
    if dlg is None:
        return
    try:
        dlg.pdb_path = path
        dlg.loaded_object = name
        dlg._traj_topology_object = name
        dlg._load_structure(path)
    except Exception:
        pass
    try:
        lig = dlg._auto_detect_ligand_in_object(name)
        if lig:
            dlg.ligand_info = lig
    except Exception:
        pass
    for refresh in ("_refresh_docking_object_combo", "_refresh_redocking_object_combos",
                    "_refresh_openmm_object_combo"):
        try:
            getattr(dlg, refresh)()
        except Exception:
            pass


def _object_name(name):
    """A PyMOL-safe object name (what cmd.load would turn the name into)."""
    import re

    safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", str(name).strip()) or "obj"
    return ("_" + safe) if safe[0] in "+-." else safe


def _free_name(name):
    """``name``, or ``name_2``, ``name_3``... when an object of that name exists: cmd.load into an existing
    object appends the file as a new state (PDB) or fails (mmCIF), and must never replace the user's object."""
    existing = set(_cmd().get_names("all"))
    if name not in existing:
        return name
    k = 2
    while f"{name}_{k}" in existing:
        k += 1
    return f"{name}_{k}"


def register(reg):
    @reg.tool("list_objects", "List the molecular objects loaded in PyMOL with atom counts and chains.",
              permission="read", title="Session objects")
    def list_objects(ctx):
        def run():
            cmd = _cmd()
            out = []
            for name in cmd.get_names("objects", enabled_only=0):
                if cmd.get_type(name) != "object:molecule":
                    continue
                out.append({"object": name, "atoms": cmd.count_atoms(name), "chains": cmd.get_chains(name)})
            return {"objects": out}
        return run()

    @reg.tool("describe_object", "Chains, residue counts, ligands/cofactors/ions (resn, chain, resi) and waters "
              "of a loaded object. Use it to find the crystal ligand before docking or redocking.",
              {"properties": {"object": {"type": "string"}}, "required": ["object"]},
              permission="read", title="Describe structure")
    def describe_object(ctx, object):
        cmd = _cmd()
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}; call list_objects")
        return _object_summary(object)

    @reg.tool("fetch_structure", "Download a structure and load it: an experimental structure from the RCSB "
              "PDB (4-character PDB ID) or a predicted one from the AlphaFold DB (UniProt accession). Waters are "
              "kept unless remove_waters is true.",
              {"properties": {
                  "source": {"type": "string", "enum": ["rcsb", "alphafold"]},
                  "identifier": {"type": "string"},
                  "object_name": {"type": "string", "description": "Optional PyMOL object name."},
                  "remove_waters": {"type": "boolean"},
              }, "required": ["source", "identifier"]},
              permission="compute_light", title="Fetch structure", main_thread=False)
    def fetch_structure(ctx, source, identifier, object_name="", remove_waters=False):
        from ... import structure_fetch_runtime as sf

        request = sf.build_structure_fetch_request(source, identifier)
        if not request.get("ok", True) and request.get("error"):
            raise ToolError(str(request["error"]))
        try:
            payload = sf.download_structure_file(request, ctx.workspace("structures"))
        except Exception as exc:
            raise ToolError(f"download failed: {exc}") from exc
        path = str(payload.get("path") or payload.get("output_path") or "")
        if not path or not os.path.isfile(path):
            raise ToolError(str(payload.get("error") or "download failed"))
        # PyMOL renames objects with spaces or symbols: use the name it will really have
        base = _object_name(object_name or os.path.splitext(os.path.basename(path))[0])

        def load():
            from .. import presentation

            cmd = _cmd()
            name = _free_name(base)          # on the main thread, where the scene is read
            cmd.load(path, name)
            if remove_waters:
                cmd.remove(f"({name}) and resn {_WATER}")
            presentation.apply_house_style()
            presentation.declutter()
            presentation.style_protein(name)
            het = f"({name}) and not polymer and not resn {_WATER}"
            if cmd.count_atoms(het):
                presentation.style_ligand(het)
            presentation.focus(name)
            register_in_leonardo(ctx, name, path)
            summary = _object_summary(name)
            summary["scene"] = ctx.scene(f"{name}: {request.get('label', source)} {identifier}")
            summary["file"] = path
            hets = ", ".join(f"{h['resn']} {h['chain']}{h['resi']}" for h in summary["hetero_groups"][:6]) or "none"
            summary["_card"] = (f"**{name}** loaded from {request.get('label', source)}: "
                                f"{len(summary['chains'])} chain(s), {sum(summary['chains'].values())} residues. "
                                f"HET groups: {hets}.")
            if payload.get("metadata"):
                summary["metadata"] = payload.get("metadata")
            return summary
        return ctx.run_on_main(load)

    @reg.tool("load_structure_file", "Load a local structure or ligand file (PDB, mmCIF, SDF, MOL2) into PyMOL.",
              {"properties": {"path": {"type": "string"}, "object_name": {"type": "string"}},
               "required": ["path"]},
              permission="compute_light", title="Load file")
    def load_structure_file(ctx, path, object_name=""):
        from .. import presentation

        path = os.path.expanduser(str(path))
        if not os.path.isfile(path):
            raise ToolError(f"file not found: {path}")
        if os.path.splitext(path)[1].lower() not in (".pdb", ".cif", ".mmcif", ".ent", ".sdf", ".mol2", ".mol",
                                                     ".pdbqt", ".xyz"):
            raise ToolError("unsupported file type")
        cmd = _cmd()
        name = _free_name(_object_name(object_name or os.path.splitext(os.path.basename(path))[0]))
        cmd.load(path, name)
        presentation.apply_house_style()
        if cmd.count_atoms(f"({name}) and polymer"):
            presentation.style_protein(name)
        else:
            presentation.style_ligand(name)
        presentation.focus(name)
        if cmd.count_atoms(f"({name}) and polymer"):
            register_in_leonardo(ctx, name, path)
        summary = _object_summary(name)
        summary["scene"] = ctx.scene(f"{name}: loaded from file")
        return summary

    @reg.tool("delete_objects", "Delete PyMOL objects (irreversible).",
              {"properties": {"objects": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
               "required": ["objects"]},
              permission="destructive", title="Delete objects")
    def delete_objects(ctx, objects):
        cmd = _cmd()
        names = set(cmd.get_names("all"))
        missing = [o for o in objects if o not in names]
        if missing:
            raise ToolError(f"not found: {', '.join(missing)}")
        for o in objects:
            cmd.delete(o)
        return {"deleted": list(objects)}
