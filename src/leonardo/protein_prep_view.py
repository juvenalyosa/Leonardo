"""
PyMOL view of a prepared protein: what Prepare Protein changed, as named selections and colours.

    <name>_built        residues built in gaps (yellow cartoon, side chains as sticks)
    <name>_protonated   residues in an unusual charge state at the chosen pH (ASH/GLH/HIP/LYN/CYM/CYX), sticks
                        with their polar hydrogens
    <name>_flipped      Asn/Gln/His side chains flipped by reduce (sticks, cyan carbons)
    <name>_completed    side chains whose missing atoms were added (sticks, light-orange carbons)
"""

LEGEND = ("yellow: built residues, magenta: unusual charge state, cyan: flipped Asn/Gln/His, "
          "orange: completed side chains")
GROUPS = (("built", "built", "yellow"), ("protonation", "protonated", "magenta"),
          ("flipped", "flipped", "cyan"), ("completed", "completed", "lightorange"))


def _sel(name, keys):
    parts = []
    for chain, resi in keys:
        resi = str(resi)
        resi = "\\" + resi if resi.startswith("-") else resi
        chain_part = f'chain "{chain}"' if str(chain).strip() else 'chain ""'
        parts.append(f"({chain_part} and resi {resi})")
    return f"({name}) and (" + " or ".join(parts) + ")" if parts else "none"


def show_prepared(name, report, cmd=None):
    """Style object ``name`` (loaded from the prepared PDB) and make the selections.  Returns {group: count}."""
    if cmd is None:
        from pymol import cmd

    changed = report.get("changed_residues") or {}
    cmd.hide("everything", name)
    cmd.show("cartoon", f"({name}) and polymer")
    cmd.color("grey80", f"({name}) and polymer and elem C")
    cmd.show("sticks", f"({name}) and not polymer and not solvent and not hydro")
    cmd.show("spheres", f"({name}) and not polymer and not solvent and (metal or inorganic)")
    cmd.set("sphere_scale", 0.35, f"({name}) and not polymer")
    counts = {}
    for key, label, color in GROUPS:
        keys = [tuple(k) for k in changed.get(key) or []]
        sel_name = f"{name}_{label}"
        counts[label] = len(keys)
        if not keys:
            cmd.delete(sel_name)
            continue
        cmd.select(sel_name, _sel(name, keys), enable=0)
        # side chains only: the cartoon shows the backbone (for a built loop, in its colour)
        cmd.show("sticks", f"{sel_name} and not hydro and not name N+C+O")
        cmd.color(color, f"{sel_name} and elem C")
        if key == "built":
            cmd.set("cartoon_color", color, sel_name)
        if key == "protonation":
            # the polar hydrogens show which atom carries the extra proton
            cmd.show("sticks", f"{sel_name} and hydro and (neighbor (elem N+O+S))")
    cmd.util.cnc(f"({name}) and rep sticks")
    return counts
