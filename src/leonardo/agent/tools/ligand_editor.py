"""Ligand Editor: read and edit the 2D sketch (atoms numbered as in the editor) and build the 3D ligand."""

import os

from ...ligand_sketcher_runtime import (
    SketchError,
    add_atom,
    attach_fragment,
    attach_ring,
    change_charge,
    clean_2d,
    delete_atoms,
    delete_bond,
    ensure_sketch_coords,
    fuse_ring,
    molecule_summary,
    resolve_group,
    set_bond_order,
    set_center_cip,
    set_element,
    stereo_labels,
    strip_hydrogens,
)
from ..registry import ToolError
from . import _headless

RINGS = ["benzene", "ring3", "ring4", "ring5", "ring6", "ring7"]
OPS = ["attach_group", "add_atom", "set_element", "set_charge", "set_bond", "delete_atoms", "delete_bond",
       "add_ring", "fuse_ring", "set_stereo", "clean_layout"]


def _sketcher(ctx):
    sk = getattr(ctx.dlg, "liged_sketcher", None)
    if sk is None:
        raise ToolError("the Ligand Editor is not available (RDKit missing or the editor failed to build)")
    return sk


def _idx(mol, number, what="atom"):
    n = int(mol.GetNumAtoms()) if mol is not None else 0
    try:
        i = int(number) - 1
    except (TypeError, ValueError):
        raise ToolError(f"{what} must be an atom number (1..{n})")
    if not 0 <= i < n:
        raise ToolError(f"{what} {number} does not exist; the sketch has atoms 1..{n}")
    return i


def sketch_state(mol, max_atoms=150):
    """JSON description of a sketch: SMILES, properties, numbered atoms and bonds."""
    info = molecule_summary(mol)
    out = {k: info[k] for k in ("smiles", "formula", "atoms", "charge", "fragments", "stereo_undefined",
                                "lipinski_violations")}
    out.update({"mw": round(info["mw"], 2), "clogp": round(info["clogp"], 2), "tpsa": round(info["tpsa"], 1),
                "hbd": info["hbd"], "hba": info["hba"], "rotatable_bonds": info["rotb"], "rings": info["rings"]})
    if mol is None or int(mol.GetNumAtoms()) == 0:
        out["note"] = "the sketch is empty"
        return out
    stereo = stereo_labels(mol)
    atoms = []
    for a in list(mol.GetAtoms())[:max_atoms]:
        i = int(a.GetIdx())
        row = {"n": i + 1, "el": a.GetSymbol(), "h": int(a.GetTotalNumHs()),
               "nbrs": [int(nb.GetIdx()) + 1 for nb in a.GetNeighbors()]}
        if a.GetFormalCharge():
            row["charge"] = int(a.GetFormalCharge())
        if a.GetIsAromatic():
            row["aromatic"] = True
        if i in stereo:
            row["stereo"] = stereo[i]
        atoms.append(row)
    bonds = []
    for b in mol.GetBonds():
        order = "aromatic" if b.GetIsAromatic() else str(b.GetBondType()).lower()
        if order != "single":
            bonds.append([int(b.GetBeginAtomIdx()) + 1, int(b.GetEndAtomIdx()) + 1, order])
    out["atom_list"] = atoms
    out["non_single_bonds"] = bonds
    if int(mol.GetNumAtoms()) > max_atoms:
        out["truncated"] = f"only the first {max_atoms} atoms are listed"
    return out


def apply_operations(mol, operations):
    """Apply edit operations in order; atom numbers refer to the sketch *after* the previous operations."""
    changes = []
    for k, op in enumerate(operations or []):
        kind = str(op.get("op", "")).strip()
        where = f"operation {k + 1} ({kind})"
        try:
            if kind == "attach_group":
                smi = resolve_group(op.get("group"))
                mol, _anchor = attach_fragment(mol, _idx(mol, op.get("atom")), smi)
                changes.append(f"attached {op.get('group')} to atom {op.get('atom')}")
            elif kind == "add_atom":
                element = str(op.get("element") or "C")
                if op.get("atom") is None:
                    mol, _new = add_atom(mol, None, element, xy=(0.0, 0.0))
                else:
                    mol, _new = add_atom(mol, _idx(mol, op.get("atom")), element, str(op.get("order") or "single"))
                changes.append(f"added {element}" + (f" on atom {op.get('atom')}" if op.get("atom") else ""))
            elif kind == "set_element":
                mol = set_element(mol, _idx(mol, op.get("atom")), str(op.get("element") or ""))
                changes.append(f"atom {op.get('atom')} -> {op.get('element')}")
            elif kind == "set_charge":
                i = _idx(mol, op.get("atom"))
                target = int(op.get("charge", 0))
                mol = change_charge(mol, i, target - int(mol.GetAtomWithIdx(i).GetFormalCharge()))
                changes.append(f"atom {op.get('atom')} charge {target:+d}")
            elif kind == "set_bond":
                a, b = (op.get("atoms") or [None, None])[:2]
                mol = set_bond_order(mol, _idx(mol, a), _idx(mol, b), str(op.get("order") or "single"))
                changes.append(f"bond {a}-{b} {op.get('order')}")
            elif kind == "delete_atoms":
                nums = list(op.get("atoms") or ([op["atom"]] if op.get("atom") else []))
                if not nums:
                    raise ToolError("give 'atoms' to delete")
                mol = delete_atoms(mol, [_idx(mol, x) for x in nums])
                changes.append(f"deleted atoms {', '.join(str(x) for x in nums)} (later atoms renumbered)")
            elif kind == "delete_bond":
                a, b = (op.get("atoms") or [None, None])[:2]
                mol = delete_bond(mol, _idx(mol, a), _idx(mol, b))
                changes.append(f"deleted bond {a}-{b}")
            elif kind == "add_ring":
                ring = str(op.get("ring") or "benzene")
                mol = attach_ring(mol, _idx(mol, op.get("atom")), ring, spiro=bool(op.get("spiro")))
                changes.append(f"{ring} on atom {op.get('atom')}" + (" (spiro)" if op.get("spiro") else ""))
            elif kind == "fuse_ring":
                a, b = (op.get("atoms") or [None, None])[:2]
                ring = str(op.get("ring") or "benzene")
                mol = fuse_ring(mol, _idx(mol, a), _idx(mol, b), ring)
                changes.append(f"{ring} fused on bond {a}-{b}")
            elif kind == "set_stereo":
                mol = set_center_cip(mol, _idx(mol, op.get("atom")), str(op.get("config") or ""))
                changes.append(f"atom {op.get('atom')} -> {op.get('config')}")
            elif kind == "clean_layout":
                mol = clean_2d(mol)
                changes.append("2D layout cleaned")
            else:
                raise ToolError(f"unknown op {kind!r}; use one of {', '.join(OPS)}")
        except ToolError as exc:
            raise ToolError(f"{where}: {exc}") from exc
        except SketchError as exc:
            raise ToolError(f"{where}: {exc}") from exc
    return mol, changes


def _load_smiles(smiles):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ToolError(f"RDKit cannot parse the SMILES {smiles!r}")
    return ensure_sketch_coords(strip_hydrogens(mol))


def _show(ctx):
    dlg = getattr(ctx.dlg, "_ligand_editor_dialog", None)
    if dlg is not None:
        dlg.show()
        dlg.raise_()


def register(reg):
    @reg.tool("open_ligand_editor", "Open Leonardo's 2D Ligand Editor so the user sees the molecule. source="
              "'active' loads the active ligand (default), 'smiles' loads the given SMILES, 'keep' leaves the "
              "current sketch. Returns the sketch with numbered atoms (the numbers edit_ligand_sketch uses).",
              {"properties": {"source": {"type": "string", "enum": ["active", "smiles", "keep"]},
                              "smiles": {"type": "string"}}},
              permission="view", title="Open Ligand Editor")
    def open_ligand_editor(ctx, source="active", smiles=""):
        sk = _sketcher(ctx)
        dlg = ctx.dlg
        if source == "smiles":
            if not smiles:
                raise ToolError("source='smiles' needs a smiles value")
            sk.set_molecule(_load_smiles(smiles), record=True, fit=True)
            dlg._on_liged_sketcher_changed(sk.current_mol())
        elif source == "active":
            path = str(getattr(dlg, "_docking_ligand_file", "") or "")
            if not path or not os.path.isfile(path):
                raise ToolError("there is no active ligand; use source='smiles' or set a ligand first")
            with _headless.quiet():
                if not dlg._ligand_editor_load_from_file(path, announce=False):
                    raise ToolError(f"the Ligand Editor could not read {os.path.basename(path)}")
        _show(ctx)
        state = sketch_state(sk.mol)
        state["_card"] = f"Ligand Editor open: **{state['formula'] or 'empty'}** `{state['smiles']}`"
        return state

    @reg.tool("ligand_sketch_state", "The molecule currently drawn in the Ligand Editor: SMILES, properties "
              "(MW, cLogP, TPSA, HBD/HBA, charge, Lipinski), numbered atoms with element, H count, neighbours "
              "and R/S/? stereo, and the non-single bonds.",
              permission="read", title="Ligand sketch")
    def ligand_sketch_state(ctx):
        return sketch_state(_sketcher(ctx).mol)

    @reg.tool("edit_ligand_sketch", "Edit the Ligand Editor sketch with a list of operations applied in order, "
              "all-or-nothing, as one undoable step the user sees. Atoms are numbered as in ligand_sketch_state "
              "(1-based); numbers refer to the sketch after the previous operations (deletions renumber later "
              "atoms, so delete last). Ops: attach_group {atom, group: Me Et iPr tBu cPr OH OMe SH NH2 NHMe NMe2 "
              "NHAc F Cl Br CF3 CN NO2 CHO COOH COOMe CONH2 SO2Me SO2NH2 Ph Bn Py Morph Pip Tz or a [*]-SMILES}; "
              "add_atom {atom?, element, order}; set_element {atom, element}; set_charge {atom, charge}; "
              "set_bond {atoms:[a,b], order}; delete_atoms {atoms}; delete_bond {atoms:[a,b]}; add_ring {atom, "
              "ring, spiro?}; fuse_ring {atoms:[a,b], ring}; set_stereo {atom, config: R|S|none}; clean_layout. "
              "Optional start_smiles replaces the sketch first. Hydrogens are implicit.",
              {"properties": {
                  "start_smiles": {"type": "string"},
                  "operations": {"type": "array", "maxItems": 40, "items": {
                      "type": "object",
                      "properties": {
                          "op": {"type": "string", "enum": OPS},
                          "atom": {"type": "integer", "minimum": 1},
                          "atoms": {"type": "array", "items": {"type": "integer", "minimum": 1}},
                          "element": {"type": "string"},
                          "group": {"type": "string"},
                          "order": {"type": "string", "enum": ["single", "double", "triple"]},
                          "ring": {"type": "string", "enum": RINGS},
                          "spiro": {"type": "boolean"},
                          "charge": {"type": "integer", "minimum": -3, "maximum": 3},
                          "config": {"type": "string", "enum": ["R", "S", "none"]},
                      },
                      "required": ["op"],
                      "additionalProperties": False,
                  }},
              }, "required": ["operations"]},
              permission="compute_light", title="Edit ligand sketch")
    def edit_ligand_sketch(ctx, operations, start_smiles=""):
        sk = _sketcher(ctx)
        mol = _load_smiles(start_smiles) if start_smiles else sk.current_mol()
        if mol is None and operations and operations[0].get("op") not in ("add_atom",):
            raise ToolError("the sketch is empty: pass start_smiles or open_ligand_editor first")
        before = molecule_summary(mol)["smiles"] if mol is not None else ""
        new, changes = apply_operations(mol, operations)
        sk._apply(new, message="Edited by the assistant (Ctrl+Z to undo).")
        if start_smiles:
            sk.canvas.fit_view()
        _show(ctx)
        state = sketch_state(new)
        state["changes"] = changes
        state["before_smiles"] = before
        state["_card"] = (f"Sketch edited ({len(changes)} change(s)): `{before or '∅'}` → `{state['smiles']}`"
                          + (f" · ⚠ {state['fragments']} fragments" if state["fragments"] > 1 else ""))
        return state

    @reg.tool("build_ligand_from_sketch", "Turn the Ligand Editor sketch into the active 3D ligand: hydrogens "
              "at the workflow pH (optional), RDKit ETKDG embedding, MMFF/UFF relaxation, written to SDF and set "
              "as the ligand for docking/parameterization. Refuses sketches with disconnected fragments.",
              {"properties": {"add_hydrogens": {"type": "boolean"}, "relax": {"type": "boolean"}}},
              permission="compute_light", title="Build 3D ligand from sketch")
    def build_ligand_from_sketch(ctx, add_hydrogens=True, relax=True):
        sk = _sketcher(ctx)
        dlg = ctx.dlg
        state = sketch_state(sk.mol)
        if not state.get("atoms"):
            raise ToolError("the sketch is empty")
        if state["fragments"] > 1:
            raise ToolError(f"the sketch has {state['fragments']} disconnected fragments; delete the extra ones")
        sk.cb_include_h.setChecked(bool(add_hydrogens))
        sk.cb_relax.setChecked(bool(relax))
        before = str(getattr(dlg, "_docking_ligand_file", "") or "")
        with _headless.quiet() as q:
            dlg._on_ligand_editor_generate_3d_apply()
        path = str(getattr(dlg, "_docking_ligand_file", "") or "")
        if not path or path == before or not os.path.isfile(path):
            errors = [m["text"] for m in q.messages if m["kind"] in ("warning", "critical")]
            raise ToolError("; ".join(errors) or "the 3D build did not produce a ligand file")
        from rdkit import Chem

        built = next(iter(Chem.SDMolSupplier(path, removeHs=False)), None)
        charge = Chem.GetFormalCharge(built) if built is not None else None
        return {"ligand_file": path, "smiles": state["smiles"], "atoms_with_h": built.GetNumAtoms() if built else None,
                "formal_charge": charge,
                "note": "Protonation follows Leonardo's workflow pH; use compute_pka to check ionizable groups.",
                "_card": f"3D ligand built from the sketch and set active: `{os.path.basename(path)}`"
                         + (f" (charge {charge:+d})" if charge is not None else "")}
