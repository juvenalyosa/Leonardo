"""
How the assistant shows its work in the PyMOL viewer.

One house style for every tool, instead of styling scattered across them: a clean
studio look (soft gradient background, specular highlights, smooth cartoon), a fixed
palette per role (protein, ligand, pocket, poses, contacts), animated camera moves to
whatever a step produced, and one PyMOL scene per step (with a caption) so the user can
replay the session from the scene list.

All functions run on the Qt main thread (they call pymol.cmd).
"""

import re

PALETTE = {
    "protein": [0.78, 0.82, 0.88],        # soft blue-grey cartoon
    "protein_surface": [0.86, 0.89, 0.93],
    "ligand_carbon": [1.00, 0.62, 0.10],  # amber
    "pose_carbons": [[0.20, 0.65, 0.95], [0.55, 0.40, 0.90], [0.25, 0.80, 0.55], [0.95, 0.40, 0.55],
                     [0.95, 0.80, 0.20]],
    "pocket": [0.30, 0.75, 0.95],
    "pocket_residues": [0.45, 0.62, 0.85],
    "hbond": [0.25, 0.70, 1.00],
    "salt_bridge": [1.00, 0.35, 0.35],
    "hydrophobic": [0.70, 0.70, 0.70],
    "pi": [0.60, 0.40, 0.90],
    "box": [0.30, 0.85, 0.60],
}
_SCENE_PREFIX = "AI"


def _cmd():
    from pymol import cmd

    return cmd


def _color(name, rgb):
    cmd = _cmd()
    cname = f"leo_ai_{name}"
    cmd.set_color(cname, [float(v) for v in rgb])
    return cname


def safe_name(text, prefix="ai"):
    base = re.sub(r"[^A-Za-z0-9_]+", "_", str(text or "")).strip("_") or "obj"
    return f"{prefix}_{base}"[:60]


def apply_house_style():
    """Global render settings (idempotent)."""
    cmd = _cmd()
    for key, value in (
        ("bg_gradient", 1), ("bg_rgb_top", [0.10, 0.12, 0.16]), ("bg_rgb_bottom", [0.20, 0.23, 0.29]),
        ("antialias", 2), ("specular", 0.35), ("shininess", 40), ("ambient", 0.35), ("direct", 0.65),
        ("cartoon_fancy_helices", 1), ("cartoon_smooth_loops", 1), ("cartoon_transparency", 0.0),
        ("stick_radius", 0.18), ("sphere_scale", 0.22), ("dash_gap", 0.35), ("dash_radius", 0.06),
        ("label_size", 16), ("label_color", "white"), ("label_outline_color", "black"),
        ("float_labels", 1), ("depth_cue", 1), ("fog_start", 0.55), ("ray_shadows", 1),
        ("ambient_occlusion_mode", 1), ("transparency_mode", 1), ("orthoscopic", 0),
        ("scene_buttons", 1),
    ):
        try:
            cmd.set(key, value)
        except Exception:
            pass
    return {"ok": True}


def style_protein(obj, surface=False):
    cmd = _cmd()
    sel = f"({obj}) and polymer.protein"
    cmd.hide("everything", f"({obj}) and polymer")
    cmd.show("cartoon", sel)
    cmd.color(_color("protein", PALETTE["protein"]), sel)
    if surface:
        cmd.show("surface", sel)
        cmd.set("transparency", 0.72, obj)
        cmd.set("surface_color", _color("protein_surface", PALETTE["protein_surface"]), obj)
    return {"object": obj, "surface": bool(surface)}


def style_ligand(sel, carbon_rgb=None):
    cmd = _cmd()
    cmd.show("sticks", sel)
    cmd.hide("lines", sel)
    cmd.util.cbag(sel)
    cmd.color(_color("ligand_carbon" if carbon_rgb is None else safe_name(str(carbon_rgb), "c"),
                     carbon_rgb or PALETTE["ligand_carbon"]), f"({sel}) and elem C")
    return {"selection": sel}


def style_pose(sel, index=0):
    rgb = PALETTE["pose_carbons"][int(index) % len(PALETTE["pose_carbons"])]
    return style_ligand(sel, carbon_rgb=rgb)


def show_pocket_residues(obj, residue_sel, name="ai_pocket"):
    cmd = _cmd()
    cmd.select(name, residue_sel)
    cmd.show("sticks", f"{name} and not name N+C+O")
    cmd.color(_color("pocket_residues", PALETTE["pocket_residues"]), f"{name} and elem C")
    cmd.deselect()
    return {"selection": name}


def show_box(center, size, name="ai_box"):
    """Search box as a CGO wire cube."""
    from pymol import cgo

    cmd = _cmd()
    cx, cy, cz = [float(v) for v in center]
    sx, sy, sz = [float(v) / 2.0 for v in size]
    r, g, b = PALETTE["box"]
    corners = [(cx + dx * sx, cy + dy * sy, cz + dz * sz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)]
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    obj = [cgo.LINEWIDTH, 2.0, cgo.BEGIN, cgo.LINES, cgo.COLOR, r, g, b]
    for a, bb in edges:
        obj += [cgo.VERTEX, *corners[a], cgo.VERTEX, *corners[bb]]
    obj.append(cgo.END)
    cmd.delete(name)
    cmd.load_cgo(obj, name)
    return {"object": name}


# helpers and the products of earlier steps (each step keeps what it produced visible)
HELPER_PATTERNS = ("*grid*", "*Grid*", "ai_box", "ai_dist", "ai_highlight", "dock_*", "redock_*", "openmm_*",
                   "int_dash_*", "intRes_*", "intCloud*", "intRingDisc*", "lbl_*")


def declutter(keep=()):
    """Hide helper objects left by earlier steps (boxes, grids, pocket surfaces, previews)."""
    import fnmatch

    cmd = _cmd()
    keep = set(keep or ())
    hidden = []
    for name in cmd.get_names("objects", enabled_only=1):
        if name in keep:
            continue
        if any(fnmatch.fnmatch(name, pat) for pat in HELPER_PATTERNS):
            cmd.disable(name)
            hidden.append(name)
    return hidden


def focus(sel, animate=0.0, buffer=5.0):
    """
    Camera onto ``sel`` (atoms, or a CGO object such as the search box). Instant by default so the
    step scene stored right after holds the final view; recalling scenes animates the transitions.
    Only what is shown counts: the displayed state (over a whole trajectory the drifting protein would be a
    speck) and the visible atoms (a hidden water box would leave the protein small in the middle).
    """
    cmd = _cmd()
    if sel in cmd.get_names("objects") and cmd.get_type(sel) == "object:cgo":
        cmd.zoom(sel, buffer=float(buffer), animate=float(animate))
        return {"ok": True}
    if cmd.count_atoms(sel) <= 0:
        return {"ok": False, "reason": "empty selection"}
    if cmd.count_atoms(f"({sel}) and visible") > 0:
        sel = f"({sel}) and visible"
    cmd.orient(sel, state=-1, animate=0)
    cmd.zoom(sel, buffer=float(buffer), state=-1, complete=1, animate=float(animate))
    return {"ok": True}


def save_step_scene(step_index, caption):
    """One scene per step, with the caption shown when it is recalled."""
    cmd = _cmd()
    key = f"{_SCENE_PREFIX}{int(step_index):02d}"
    try:
        cmd.scene(key, "store", message=str(caption)[:120])
    except Exception:
        return {"scene": ""}
    return {"scene": key}


def recall_scene(key, animate=1.5):
    cmd = _cmd()
    cmd.scene(str(key), "recall", animate=float(animate))
    return {"scene": key}
