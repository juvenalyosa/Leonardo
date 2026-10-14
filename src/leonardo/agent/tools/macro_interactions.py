"""Interface interactions of macromolecular complexes (protein-protein, protein-DNA, protein-RNA).

The chemistry lives in :mod:`leonardo.macro_interaction_runtime`; this module reads the atoms from
PyMOL, resolves the two selections and draws the result.  The drawing is kept in small separate
functions (selections, dash objects, style) so the visuals can be restyled without touching the
analysis.
"""

import re

from ..registry import ToolError

_DASH_SUFFIX = {"Hydrogen Bond": "hb", "Salt Bridge": "sb", "Pi-Pi Stacking": "pi", "T-Shaped Pi-Pi": "pi",
                "Cation-Pi": "cpi", "Hydrophobic": "hyd", "Water Bridge": "wat", "Metal Bridge": "metal",
                "Disulfide": "ss"}
_DASH_COLOR = {"hb": "yellow", "sb": "magenta", "pi": "green", "cpi": "orange", "hyd": "grey60",
               "wat": "lightblue", "metal": "violet", "ss": "yelloworange"}
_KEYWORDS = {"protein", "nucleic", "dna", "rna", "polymer"}


def _cmd():
    from pymol import cmd

    return cmd


def _model_atoms(cmd, obj):
    """Atom dicts of ``obj`` (state 1, alternate location blank/A) for the runtime, keyed by PyMOL index."""
    model = cmd.get_model(f"({obj}) and not alt B+C+D+E+F", state=1)
    atoms = []
    for a in model.atom:
        atoms.append({"name": a.name, "resn": a.resn, "resi": str(a.resi), "chain": a.chain,
                      "element": a.symbol, "coord": list(a.coord), "hetatm": bool(a.hetatm),
                      "pymol_index": int(a.index)})
    return atoms


def resolve_side(cmd, obj, text, atoms):
    """A runtime selection spec: chain ids / keywords / ``A:10-50`` pass through; anything else is a
    PyMOL selection inside ``obj`` and becomes a list of atom positions."""
    text = str(text or "").strip()
    if not text:
        return None
    chains = set(cmd.get_chains(f"({obj})"))
    if text.lower() in _KEYWORDS:
        return text.lower()
    groups = [g.strip() for g in re.split(r"[;+]", text) if g.strip()]
    if groups and all((":" in g and g.split(":", 1)[0].strip() in chains) or
                      all(c.strip() in chains for c in g.split(",")) for g in groups):
        return text
    sel = f"({obj}) and ({text})"
    try:
        n = cmd.count_atoms(sel)
    except Exception as exc:
        raise ToolError(f"cannot read selection {text!r}: {exc}")
    if not n:
        raise ToolError(f"selection {text!r} matches no atom of {obj} (chains: {', '.join(sorted(chains))})")
    wanted = {idx for _o, idx in cmd.index(sel)}
    return [k for k, a in enumerate(atoms) if a["pymol_index"] in wanted]


# --------------------------------------------------------------------------- #
# visuals (restyle here; the analysis never depends on them)
# --------------------------------------------------------------------------- #
def _residue_sel(obj, chain, resi):
    chain_part = f"chain {chain}" if chain else 'chain ""'
    resi = str(resi)
    resi = "\\" + resi if resi.startswith("-") else resi      # "resi -5" alone is a range
    return f"(({obj}) and {chain_part} and resi {resi})"


def select_interface(obj, result, prefix):
    """Named selections of the interface residues: <prefix>_a, <prefix>_b and <prefix>_res (both)."""
    cmd = _cmd()
    names = {}
    for side in ("A", "B"):
        rows = [h for h in result.get("hotspots", []) if h["side"] == side]
        name = f"{prefix}_{side.lower()}"
        sel = " or ".join(_residue_sel(obj, h["chain"], h["resi"]) for h in rows) or "none"
        cmd.select(name, sel, enable=0)
        names[side] = name
    cmd.select(f"{prefix}_res", f"{names['A']} or {names['B']}", enable=0)
    return {"side_a": names["A"], "side_b": names["B"], "both": f"{prefix}_res"}


def _atom_sel(obj, ref):
    return f"({obj}) and index {int(ref['pymol_index'])}"


def draw_interaction_dashes(obj, result, prefix, types=None):
    """One distance object per interaction family: <prefix>_hb, _sb, _pi, _cpi, _hyd, _wat, _metal, _ss.
    Ring contacts are drawn between the ring atoms closest to the partner; bridges through their water/ion."""
    cmd = _cmd()
    made = []
    for rec in result.get("interactions", []):
        if types and rec["type"] not in types:
            continue
        suffix = _DASH_SUFFIX.get(rec["type"])
        a, b = rec["a"], rec["b"]
        if not suffix or a.get("pymol_index") is None or b.get("pymol_index") is None:
            continue
        name = f"{prefix}_{suffix}"
        via = rec.get("via") or {}
        try:
            if via.get("pymol_index") is not None:
                mid = _atom_sel(obj, via)
                cmd.distance(name, _atom_sel(obj, a), mid, mode=0, cutoff=20.0, label=0)
                cmd.distance(name, mid, _atom_sel(obj, b), mode=0, cutoff=20.0, label=0)
            else:
                cmd.distance(name, _atom_sel(obj, a), _atom_sel(obj, b), mode=0, cutoff=20.0, label=0)
        except Exception:
            continue
        if name not in made:
            made.append(name)
    return made


def style_interface_minimal(obj, selections, dash_objects):
    """Minimal style: interface residues as sticks (no backbone), one plain colour per dash family."""
    cmd = _cmd()
    both = selections["both"]
    cmd.show("sticks", f"({both}) and not name N+C+O and not elem H")
    for name in dash_objects:
        suffix = name.rsplit("_", 1)[-1]
        cmd.color(_DASH_COLOR.get(suffix, "white"), name)
        cmd.hide("labels", name)


# --------------------------------------------------------------------------- #
# SASA with PyMOL (optional)
# --------------------------------------------------------------------------- #
def _pymol_buried_sasa(cmd, obj, atoms, idx_a, idx_b, prefix):
    """Buried SASA from PyMOL get_area (dot_solvent on, heavy atoms only)."""
    def sel_of(idx):
        ids = "+".join(str(atoms[k]["pymol_index"]) for k in idx)
        return f"({obj}) and index {ids} and not elem H"

    tmp = {"a": f"_{prefix}_sasa_a", "b": f"_{prefix}_sasa_b", "ab": f"_{prefix}_sasa_ab"}
    areas = {}
    try:
        cmd.create(tmp["a"], sel_of(idx_a))
        cmd.create(tmp["b"], sel_of(idx_b))
        cmd.create(tmp["ab"], f"{tmp['a']} or {tmp['b']}")
        for key, name in tmp.items():
            cmd.set("dot_solvent", 1, name)
            cmd.set("dot_density", 3, name)
            cmd.set("solvent_radius", 1.4, name)
            areas[key] = float(cmd.get_area(name))
    finally:
        for name in tmp.values():
            cmd.delete(name)
    return {"method": "pymol_get_area", "probe_A": 1.4, "dot_density": 3,
            "buried_total": round(areas["a"] + areas["b"] - areas["ab"], 1),
            "sasa_A_alone": round(areas["a"], 1), "sasa_B_alone": round(areas["b"], 1),
            "sasa_complex": round(areas["ab"], 1)}


# --------------------------------------------------------------------------- #
# tool
# --------------------------------------------------------------------------- #
def _compact(rec):
    a, b = rec["a"], rec["b"]
    row = {"type": rec["type"], "a": f"{a['residue']} {a['atom']}", "b": f"{b['residue']} {b['atom']}",
           "distance_A": rec["distance"], "distance_def": rec["distance_def"]}
    if rec.get("angle") is not None:
        row["angle_deg"] = rec["angle"]
        row["angle_def"] = rec["angle_def"]
    if rec.get("donor_side"):
        row["donor_side"] = rec["donor_side"]
    if rec.get("readout"):
        row["readout"] = rec["readout"]
    if rec.get("via"):
        row["via"] = rec["via"].get("residue")
    if rec.get("conditional"):
        row["note"] = rec["conditional"]
    return row


def register(reg):
    @reg.tool("analyze_interface", "Interface interactions between two parts of a macromolecular complex: "
              "protein-protein (two chains or any two selections), protein-DNA and protein-RNA. Finds hydrogen "
              "bonds (backbone/side chain; explicit H when present, otherwise heavy-atom distance + angle), salt "
              "bridges (Arg/Lys/His vs Asp/Glu/phosphate), hydrophobic contacts, pi stacking (parallel/T, including "
              "aromatic side chains on nucleobases), cation-pi (Arg/Lys over aromatic rings and bases), water "
              "bridges (crystal waters), disulfides and metal bridges. Nucleic-acid contacts are labelled phosphate "
              "/ sugar / base and, for DNA bases, major or minor groove (base readout vs backbone contact). Also "
              "returns interface residues, buried solvent-accessible area, residue-residue contacts and per-residue "
              "counts. Defaults: protein vs nucleic acid when both exist, otherwise the first chain vs the chain it "
              "touches most. Selections: chain ids ('A', 'A+B'), 'protein', 'nucleic', 'dna', 'rna', chain:residues "
              "('A:10-50'), or any PyMOL selection inside the object. Draws the contacts as dashes.",
              {"properties": {
                  "object": {"type": "string"},
                  "selection_a": {"type": "string", "description": "side A (default: automatic)"},
                  "selection_b": {"type": "string", "description": "side B (default: automatic / the rest)"},
                  "include_waters": {"type": "boolean", "description": "water-mediated bridges (default true)"},
                  "sasa_method": {"type": "string", "enum": ["numpy", "pymol"],
                                  "description": "Shrake-Rupley in numpy (default, also per residue) or PyMOL "
                                                 "get_area (total only)"},
                  "show": {"type": "boolean", "description": "create selections and dash objects (default true)"},
                  "max_rows": {"type": "integer", "minimum": 1, "maximum": 100,
                               "description": "rows of top contacts / hotspots to return (default 15)"},
              }, "required": ["object"]},
              permission="compute_light", title="Macromolecular interface")
    def analyze_interface(ctx, object, selection_a="", selection_b="", include_waters=True, sasa_method="numpy",
                          show=True, max_rows=15):
        from ... import macro_interaction_runtime as rt

        cmd = _cmd()
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}; call list_objects")
        atoms = _model_atoms(cmd, object)
        if not atoms:
            raise ToolError(f"{object} has no atoms")
        side_a = resolve_side(cmd, object, selection_a, atoms)
        side_b = resolve_side(cmd, object, selection_b, atoms)
        use_pymol_sasa = str(sasa_method or "numpy").lower() == "pymol"
        try:
            result = rt.analyze_interface(atoms, side_a, side_b, include_waters=bool(include_waters),
                                          compute_sasa=not use_pymol_sasa)
        except ValueError as exc:
            chains = ", ".join(cmd.get_chains(f"({object})"))
            raise ToolError(f"{exc} (chains of {object}: {chains})")
        positions = result.pop("_positions")
        if use_pymol_sasa:
            result["sasa"] = _pymol_buried_sasa(cmd, object, atoms, positions["A"], positions["B"], "iface")

        prefix = re.sub(r"[^A-Za-z0-9_]+", "_", f"iface_{object}")[:40]
        objects = {}
        if show:
            for name in cmd.get_names("all"):
                if name.startswith(prefix + "_"):
                    cmd.delete(name)
            sels = select_interface(object, result, prefix)
            dashes = draw_interaction_dashes(object, result, prefix)
            style_interface_minimal(object, sels, dashes)
            objects = {"selections": sels, "dash_objects": dashes}
            try:
                cmd.zoom(sels["both"], buffer=4.0)
            except Exception:
                pass

        max_rows = int(max_rows or 15)
        top = [r for r in result["interactions"] if r["type"] != "Hydrophobic"][:max_rows]
        if len(top) < max_rows:
            top += [r for r in result["interactions"] if r["type"] == "Hydrophobic"][:max_rows - len(top)]
        hotspots = [{"residue": h["residue"], "side": h["side"], "n_specific": h["n_specific"],
                     "counts": h["counts"], "buried_sasa_A2": h.get("buried_sasa"),
                     "n_atom_contacts": h["n_atom_contacts"]} for h in result["hotspots"][:max_rows]]
        sasa = result.get("sasa") or {}
        out = {
            "object": object,
            "interface_type": result["interface_type"],
            "sides": result["sides"],
            "counts": result["counts"],
            "interface_residues": result["interface_residues"],
            "buried_sasa": {k: v for k, v in sasa.items() if k != "radii"},
            "top_contacts": [_compact(r) for r in top],
            "hotspots": hotspots,
            "residue_contacts": result["residue_contacts"][:max_rows],
            "nucleic_acid": result.get("nucleic_acid"),
            "hydrogens_present": result["hydrogens_present"],
            "waters_present": result["waters_present"],
            "notes": result["notes"],
            "objects": objects,
        }
        counts = ", ".join(f"{n} {t}" for t, n in result["counts"].items()) or "no interactions"
        res = result["interface_residues"]
        card = (f"**{result['interface_type']} interface** of {object} "
                f"({result['sides']['A']['selection']} vs {result['sides']['B']['selection']}): "
                f"{len(res['A'])} + {len(res['B'])} residues; {counts}.")
        if sasa.get("buried_total") is not None:
            card += f" Buried SASA {sasa['buried_total']:.0f} Å² ({sasa.get('method')})."
        na = result.get("nucleic_acid")
        if na:
            card += (f" Nucleic acid: {na['base_readout_specific']} base-specific vs {na['backbone_specific']} "
                     "backbone contacts.")
        if hotspots:
            card += " Top residues: " + ", ".join(f"{h['residue']} ({h['n_specific']})" for h in hotspots[:5]) + "."
        out["_card"] = card
        try:
            out["scene"] = ctx.scene(f"interface: {object}")
        except Exception:
            out["scene"] = ""
        return out
