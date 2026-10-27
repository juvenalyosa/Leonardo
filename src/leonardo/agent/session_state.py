"""
Short description of what the session already has, sent with each request (no Qt here).

The model otherwise spends look-ups (list_objects, describe_object, ligand_status) to learn what is loaded,
or plans steps that are already done. The text lists the loaded objects (atoms, chains, hetero groups), the
docking state (pocket list, box, ligand, ligand force field) and the background jobs that still run.
"""

from . import plan_check

MAX_OBJECTS = 12


def _objects():
    from pymol import cmd

    lines = []
    names = list(cmd.get_names("objects") or [])
    for name in names[:MAX_OBJECTS]:
        try:
            n = cmd.count_atoms(f"%{name}")
        except Exception:
            continue
        if not n:
            continue
        chains = sorted(set(cmd.get_chains(f"%{name}") or []))
        het = {"resn": set()}
        cmd.iterate(f"%{name} and hetatm and not resn HOH+WAT+DOD", "het['resn'].add(resn)", space={"het": het})
        waters = cmd.count_atoms(f"%{name} and resn HOH+WAT+DOD and not hydro")
        states = cmd.count_states(f"%{name}")
        parts = [f"{n} atoms"]
        if chains:
            parts.append("chains " + ",".join(c or "-" for c in chains))
        if het["resn"]:
            parts.append("hetero " + ",".join(sorted(het["resn"])[:8]))
        if waters:
            parts.append(f"{waters} waters")
        if states > 1:
            parts.append(f"{states} states")
        lines.append(f"{name} ({'; '.join(parts)})")
    if len(names) > MAX_OBJECTS:
        lines.append(f"... and {len(names) - MAX_OBJECTS} more")
    return lines


def describe(dlg=None, jobs=None):
    """The text (empty scene included); never raises."""
    try:
        objects = _objects()
    except Exception:
        objects = []
    facts = plan_check.scene_facts(dlg)
    out = ["Session state (automatic, at this request):",
           "- objects: " + ("; ".join(objects) if objects else "none loaded")]
    docking = [label for fact, label in (("pockets_found", "fpocket pocket list"), ("docking_box", "docking box set"),
                                         ("ligand", "active ligand"), ("ligand_ff", "ligand force field"))
               if fact in facts]
    if dlg is not None:
        lig = str(getattr(dlg, "_docking_ligand_file", "") or "")
        if lig:
            docking = [d + (f" ({lig.rsplit('/', 1)[-1]})" if d == "active ligand" else "") for d in docking]
    out.append("- docking/ligand: " + (", ".join(docking) if docking else "nothing set"))
    try:
        running = [j for j in (jobs.list() if jobs is not None else []) if j.get("status") == "running"]
    except Exception:
        running = []
    if running:
        out.append("- running jobs: " + "; ".join(f"{j['job_id']} {j.get('description') or j.get('kind')}"
                                                  for j in running))
    return "\n".join(out)
