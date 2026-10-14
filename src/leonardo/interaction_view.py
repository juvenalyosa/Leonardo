"""
Publication-style views of protein–ligand interactions.

Takes the interaction records produced by Leonardo's detector (type, protein residue/atom, ligand atom,
distance, the two coordinates used for the measurement) and draws them in two ways:

* ``build_scene``: the PyMOL scene. White background, a light cartoon for context, the ligand as balls and
  sticks (one copy), only the residues that interact as thin sticks with one label per residue ("Asp189"),
  dashed cylinders coloured by interaction type, ring centroids, and bridging waters only.
* ``draw_2d_map``: a 2D diagram. The ligand is drawn by RDKit with correct bond orders, aromaticity and
  charges (PDB chemical component template, else the geometry), and the residues are placed around it
  without overlaps, joined to the atoms they touch by dashed lines coloured by type and labelled with the
  distance.

``export_figure`` ray-traces the scene and adds a legend panel.
"""
from __future__ import annotations

import math
import os
import re
from collections import OrderedDict

import numpy as np

# One palette for the 3D scene, the 2D map and the legends (readable on white, distinct in print).
TYPE_STYLE = OrderedDict([
    ("Hydrogen Bond", {"hex": "#1F6FEB", "short": "H-bond", "radius": 0.055, "dash": 0.22, "gap": 0.16}),
    ("Salt Bridge", {"hex": "#D7263D", "short": "Salt bridge", "radius": 0.075, "dash": 0.30, "gap": 0.18}),
    ("Metal Coordination", {"hex": "#7B3FBF", "short": "Metal", "radius": 0.070, "dash": 0.30, "gap": 0.16}),
    ("Halogen Bond", {"hex": "#0E9AA7", "short": "Halogen bond", "radius": 0.060, "dash": 0.24, "gap": 0.16}),
    ("Pi-Pi Stacking", {"hex": "#2E9E44", "short": "π–π stacking", "radius": 0.065, "dash": 0.26, "gap": 0.18}),
    ("T-Shaped Pi-Pi", {"hex": "#2E9E44", "short": "π–π T-shaped", "radius": 0.065, "dash": 0.26, "gap": 0.18}),
    ("Cation-Pi", {"hex": "#E8710A", "short": "Cation–π", "radius": 0.065, "dash": 0.26, "gap": 0.18}),
    ("Anion-Pi", {"hex": "#B5367A", "short": "Anion–π", "radius": 0.065, "dash": 0.26, "gap": 0.18}),
    ("Hydrophobic", {"hex": "#8A8F98", "short": "Hydrophobic", "radius": 0.040, "dash": 0.12, "gap": 0.14}),
    ("Van der Waals", {"hex": "#B9BEC6", "short": "van der Waals", "radius": 0.030, "dash": 0.10, "gap": 0.16}),
])
_DEFAULT_STYLE = {"hex": "#6B7280", "short": "", "radius": 0.05, "dash": 0.2, "gap": 0.16}
# drawn as dashes in 3D by default; contact-only types are shown by the residue itself
POLAR_TYPES = ("Hydrogen Bond", "Salt Bridge", "Metal Coordination", "Halogen Bond")
RING_TYPES = ("Pi-Pi Stacking", "T-Shaped Pi-Pi", "Cation-Pi", "Anion-Pi")
CONTACT_TYPES = ("Hydrophobic", "Van der Waals")

INK = (31, 41, 55)
MUTED = (107, 114, 128)
LIGAND_CARBON = (0.93, 0.62, 0.20)
RESIDUE_CARBON = (0.47, 0.54, 0.62)
CARTOON = (0.85, 0.88, 0.92)
# residue classes for the 2D pills: (fill, outline)
RES_CLASS = {
    "negative": ((253, 226, 228), (215, 38, 61)),
    "positive": ((222, 235, 255), (31, 111, 235)),
    "polar": ((220, 245, 247), (14, 154, 167)),
    "hydrophobic": ((228, 244, 230), (46, 158, 68)),
    "glycine": ((240, 241, 243), (138, 143, 152)),
    "water": ((230, 242, 255), (96, 145, 210)),
    "metal": ((238, 228, 250), (123, 63, 191)),
    "other": ((243, 244, 246), (107, 114, 128)),
}
_CLASS_OF = {}
for _cls, _names in (("negative", "ASP GLU"), ("positive", "LYS ARG HIS HIP HID HIE"),
                     ("polar", "SER THR ASN GLN CYS TYR CYX"), ("hydrophobic", "ALA VAL LEU ILE MET PHE TRP PRO"),
                     ("glycine", "GLY"), ("water", "HOH WAT DOD"),
                     ("metal", "ZN MG CA MN FE FE2 CU CU1 CO NI NA K CD HG")):
    for _n in _names.split():
        _CLASS_OF[_n] = _cls


def style_of(itype):
    return TYPE_STYLE.get(str(itype), _DEFAULT_STYLE)


def rgb01(hexstr):
    s = hexstr.lstrip("#")
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb255(hexstr):
    return tuple(int(round(255 * v)) for v in rgb01(hexstr))


def residue_label(resn, resi, chain="", with_chain=False):
    """'ASP', '189' -> 'Asp189' (waters 'HOH 301' -> 'HOH301', ions keep their symbol)."""
    resn = str(resn or "?").strip().upper()
    if len(resn) == 3 and resn not in ("HOH", "WAT", "DOD") and resn.isalpha():
        name = resn.capitalize()
    else:
        name = resn
    text = f"{name}{str(resi).strip()}"
    if with_chain and str(chain or "").strip():
        text += f":{str(chain).strip()}"
    return text


def residue_class(resn):
    return _CLASS_OF.get(str(resn or "").strip().upper(), "other")


def _residue_key(inter):
    """(chain, resi, resn) of the protein side; parsed from 'prot_res' ('ASP A189') when the fields are absent."""
    chain, resi, resn = (str(inter.get(k) or "").strip() for k in ("prot_chain", "prot_resi", "prot_resn"))
    if not resi:
        m = re.match(r"^\s*(\S+)\s+([A-Za-z]?)(-?\d+[A-Za-z]?)\s*$", str(inter.get("prot_res") or ""))
        if m:
            resn, chain, resi = resn or m.group(1), chain or m.group(2), m.group(3)
    return (chain, resi, resn)


def group_by_residue(interactions):
    """OrderedDict residue key -> list of interactions, strongest kinds first."""
    rank = {t: i for i, t in enumerate(TYPE_STYLE)}
    out = OrderedDict()
    for inter in sorted(interactions, key=lambda r: (rank.get(str(r.get("type")), 99), float(r.get("distance", 9)))):
        out.setdefault(_residue_key(inter), []).append(inter)
    return out


def summarize(interactions):
    counts = OrderedDict()
    for t in TYPE_STYLE:
        n = sum(1 for r in interactions if str(r.get("type")) == t)
        if n:
            counts[t] = n
    for r in interactions:
        t = str(r.get("type"))
        if t not in counts:
            counts[t] = sum(1 for x in interactions if str(x.get("type")) == t)
    return counts


# ----------------------------------------------------------------------------------------- 3D scene

def _dashed_cylinder(p, q, rgb, radius, dash, gap):
    from pymol import cgo

    p, q = np.asarray(p, float), np.asarray(q, float)
    v = q - p
    length = float(np.linalg.norm(v))
    if length < 1e-3:
        return []
    u = v / length
    out = []
    period = dash + gap
    n = max(1, int(length // period))
    # centre the pattern so both ends look the same
    start = (length - (n * period - gap)) / 2.0
    for k in range(n):
        a = p + u * (start + k * period)
        b = a + u * dash
        out += [cgo.CYLINDER, *a, *b, radius, *rgb, *rgb]
        out += [cgo.SPHERE, *a, radius, cgo.SPHERE, *b, radius]
    return [cgo.COLOR, *rgb] + out


def _ring_marker(center, rgb, radius=0.16):
    from pymol import cgo

    return [cgo.COLOR, *rgb, cgo.SPHERE, *[float(x) for x in center], radius]


def _coords(inter, key):
    c = inter.get(key)
    if c is None:
        return None
    try:
        arr = np.asarray(c, dtype=float).reshape(3)
    except Exception:
        return None
    return arr if np.all(np.isfinite(arr)) else None


def _atom_sel(obj, key):
    chain, resi, resn = key
    sel = f"({obj}) and resi \\{resi}" if resi.startswith("-") else f"({obj}) and resi {resi}"
    if chain:
        sel += f" and chain {chain}"
    if resn:
        sel += f" and resn {resn}"
    return sel


RESIDUE_LABEL_A = 1.05        # label heights in Angstrom (resolution independent)
DISTANCE_LABEL_A = 0.78


def label_box(text, height):
    """Approximate (width, height) in Angstrom of a label of ``height`` Angstrom."""
    return (0.62 * height * len(str(text)) + 0.3, 1.25 * height)


def place_label(anchor, away_from, box, groups, radii=(1.0, 1.5, 2.1, 2.8, 3.6, 4.5)):
    """
    Point near ``anchor`` (2D, camera plane) on the side away from ``away_from`` for a label ``box``.
    ``groups`` is [(weight, [rects])]: overlaps are penalised by weight (other labels > ligand > sticks)
    and distance from the anchor a little, so a label stays next to its residue when it can.
    """
    anchor = np.asarray(anchor, float)
    v = anchor - np.asarray(away_from, float)
    base = math.atan2(v[1], v[0]) if np.linalg.norm(v) > 1e-6 else math.pi / 2
    best, best_cost = anchor, None
    for r in radii:
        for dang in (0, 25, -25, 50, -50, 75, -75, 100, -100, 130, -130, 160, -160, 180):
            ang = base + math.radians(dang)
            xy = anchor + r * np.array([math.cos(ang), math.sin(ang)])
            rect = (xy[0] - box[0] / 2, xy[1] - box[1] / 2, xy[0] + box[0] / 2, xy[1] + box[1] / 2)
            cost = 0.35 * r + 0.004 * abs(dang)
            for weight, rects in groups:
                cost += weight * sum(1 for t in rects if _rect_overlap(rect, t))
            if best_cost is None or cost < best_cost:
                best, best_cost = xy, cost
            if r == 0.0:
                break
    return best


def _current_rotation():
    from pymol import cmd

    v = cmd.get_view()
    return np.array(v[0:9], float).reshape(3, 3, order="F")


def _best_view_direction(lig, center, out, env, focus):
    """
    Viewing direction (towards the viewer) that shows the ligand face-on and the interacting residues spread
    out, with few protein atoms between the camera and the ligand, kept on the solvent side (within 80 degrees
    of ``out``).
    """
    rng = np.random.default_rng(7)
    cands = rng.normal(size=(600, 3))
    cands /= np.linalg.norm(cands, axis=1)[:, None]
    cands = np.vstack([out, cands[cands @ out > math.cos(math.radians(80))]])
    lig_rel = lig - center
    foc_rel = (focus - center) if focus is not None and len(focus) else lig_rel
    env_rel = (env - center) if env is not None and len(env) else np.zeros((0, 3))
    radius = float(np.max(np.linalg.norm(lig_rel, axis=1))) + 2.5 if len(lig_rel) else 4.0
    best, best_score = out, None
    for d in cands:
        def spread(points):
            pr = points - np.outer(points @ d, d)
            sv = np.linalg.svd(pr - pr.mean(axis=0), compute_uv=False) if len(pr) > 2 else np.array([1.0, 1.0])
            return float(sv[1] / (sv[0] + 1e-6)) if len(sv) > 1 else 0.0
        along = env_rel @ d
        perp = np.linalg.norm(env_rel - np.outer(along, d), axis=1)
        occl = int(np.sum((along > 2.0) & (along < 16.0) & (perp < radius)))
        score = 1.0 * spread(lig_rel) + 0.6 * spread(foc_rel) - 0.012 * occl + 0.15 * float(d @ out)
        if best_score is None or score > best_score:
            best, best_score = d, score
    return best


def view_into_pocket(obj, lig_sel, focus_sel, aspect=1.45, margin=2.5):
    """
    Camera looking from the solvent into the binding site (the viewer sits on the side of the ligand away
    from the local protein mass), the ligand's long axis horizontal, zoomed on ``focus_sel``.
    Returns the model-to-camera rotation.
    """
    from pymol import cmd

    lig = cmd.get_coords(f"({lig_sel}) and not elem H")
    if lig is None or not len(lig):
        lig = cmd.get_coords(lig_sel)
    center = lig.mean(axis=0)
    env = cmd.get_coords(f"({obj}) and polymer and not ({lig_sel}) and (({lig_sel}) around 14)")
    out = center - env.mean(axis=0) if env is not None and len(env) else np.array([0.0, 0.0, 1.0])
    out = out / (np.linalg.norm(out) or 1.0)
    focus = cmd.get_coords(f"({focus_sel}) and not elem H")
    z = _best_view_direction(lig, center, out, env, focus)
    pts = lig - center
    pts = pts - np.outer(pts @ z, z)
    try:
        x = np.linalg.svd(pts, full_matrices=False)[2][0]
    except Exception:
        x = np.array([1.0, 0.0, 0.0])
    x = x - (x @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 0.0, 1.0] if abs(z[2]) < 0.9 else [1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.vstack([x, y, z])                      # rows: camera axes in model space
    pts = cmd.get_coords(f"({focus_sel}) and not elem H")
    if pts is None or not len(pts):
        pts = lig
    cam = (pts - center) @ rot.T
    # centre the frame on the focus atoms and fit them (the same frame for any output size of this aspect)
    mid = (cam.min(axis=0) + cam.max(axis=0)) / 2.0
    mid[2] = 0.0
    origin = center + rot.T @ mid
    half_w, half_h = (cam[:, 0].max() - cam[:, 0].min()) / 2.0, (cam[:, 1].max() - cam[:, 1].min()) / 2.0
    half = max(half_h, half_w / float(aspect)) + float(margin)
    fov = float(cmd.get("field_of_view") or 20.0)
    dist = half / math.tan(math.radians(fov / 2.0))
    view = list(cmd.get_view())
    view[0:9] = list(rot.flatten(order="F"))         # PyMOL stores the matrix column-major
    view[9:12] = [0.0, 0.0, -dist]
    view[12:15] = [float(c) for c in origin]
    # clip what lies in front of the binding site (loops crossing the line of sight), keep the back
    view[15] = max(1.0, dist - (float(cam[:, 2].max()) + 4.0))
    view[16] = dist - float(cam[:, 2].min()) + 14.0
    view[17] = abs(float(view[17]) or fov)           # positive: orthoscopic (labels stay where they are placed)
    cmd.set_view(view)
    cmd.set("orthoscopic", 1)
    return rot


def build_scene(obj, lig_sel, interactions, *, prefix="iv", show_contacts=False, distance_labels=True,
                surface=False, keep_objects=(), orient=True, context_radius=0.0, max_labels=12):
    """
    Compose the publication-style PyMOL scene for ``interactions`` of the ligand ``lig_sel`` in ``obj``.

    Contact-only types (hydrophobic, van der Waals) show their residues but no dashes unless
    ``show_contacts``. Returns a summary of what was drawn.
    """
    from pymol import cmd

    for name in cmd.get_names("objects"):
        if name.startswith(f"{prefix}_"):
            cmd.delete(name)
    keep = {obj, *keep_objects}
    for name in cmd.get_names("objects", enabled_only=1):
        if name not in keep:
            cmd.disable(name)
    cmd.enable(obj)

    for key, value in (("bg_rgb", [1.0, 1.0, 1.0]), ("bg_gradient", 0), ("ray_opaque_background", 1),
                       ("depth_cue", 0), ("ray_trace_fog", 0), ("fog", 0), ("antialias", 2),
                       ("ray_trace_mode", 0), ("ray_interior_color", "grey80"),
                       ("ray_shadows", 0), ("ambient", 0.45), ("direct", 0.6), ("reflect", 0.2),
                       ("specular", 0.25), ("shininess", 30), ("spec_count", 1), ("light_count", 2),
                       ("ambient_occlusion_mode", 0), ("orthoscopic", 1),
                       ("cartoon_transparency", 0.25), ("cartoon_tube_radius", 0.17), ("cartoon_fancy_helices", 1), ("cartoon_smooth_loops", 1),
                       ("cartoon_loop_radius", 0.18), ("cartoon_oval_width", 0.18),
                       ("stick_radius", 0.13), ("sphere_scale", 0.22), ("stick_ball", 0), ("valence", 0),
                       ("label_font_id", 7), ("label_size", 22), ("label_color", "black"),
                       ("label_outline_color", "white"), ("float_labels", 0), ("label_position", [0, 0, 1.5]),
                       ("transparency_mode", 2)):
        try:
            cmd.set(key, value)
        except Exception:
            pass

    cmd.set_color("iv_cartoon", list(CARTOON))
    cmd.set_color("iv_lig_c", list(LIGAND_CARBON))
    cmd.set_color("iv_res_c", list(RESIDUE_CARBON))
    cmd.hide("everything", obj)
    cmd.label(obj, "")
    poly = f"({obj}) and polymer and not ({lig_sel})"
    cmd.show("cartoon", f"byres (({poly}) within {float(context_radius)} of ({lig_sel}))"
             if context_radius else poly)
    # a thin translucent backbone tube for context: no sheets or helices crossing the line of sight
    cmd.cartoon("tube", poly)
    cmd.color("iv_cartoon", poly)
    # atom-level values left by other views (e.g. an opaque white cartoon) override object settings
    cmd.set("cartoon_color", "iv_cartoon", poly)
    cmd.set("cartoon_transparency", 0.25, poly)

    # ligand: balls and sticks, CPK with amber carbons, polar H only
    lig_heavy = f"({lig_sel}) and not (elem H and neighbor elem C)"
    cmd.show("sticks", lig_heavy)
    cmd.show("spheres", lig_heavy)
    cmd.set("stick_radius", 0.16, obj)
    cmd.set("sphere_scale", 0.24, lig_sel)
    cmd.set("sphere_scale", 0.14, f"({lig_sel}) and elem H")
    cmd.util.cbag(lig_sel)
    cmd.color("iv_lig_c", f"({lig_sel}) and elem C")

    groups = group_by_residue(interactions)
    chains = {k[0] for k in groups if k[0]}
    multi_chain = len(chains) > 1
    lig_center = np.asarray(cmd.get_coords(lig_sel).mean(axis=0) if cmd.count_atoms(lig_sel) else [0, 0, 0])

    residue_sels, key_sels, waters, labels = [], [], [], []
    many = sum(1 for rows in groups.values() if any(str(r.get("type")) not in CONTACT_TYPES for r in rows))
    label_contacts = len(groups) <= max_labels
    cmd.set_color("iv_contact_c", [0.66, 0.71, 0.77])
    for key, rows in groups.items():
        chain, resi, resn = key
        if not resi:
            continue
        rsel = _atom_sel(obj, key)
        if not cmd.count_atoms(rsel):
            continue
        cls = residue_class(resn)
        if cls == "water":
            if any(str(r.get("type")) in POLAR_TYPES for r in rows):
                waters.append(rsel)
            else:
                continue
        elif cls == "metal":
            cmd.show("spheres", rsel)
            cmd.set("sphere_scale", 0.45, rsel)
        else:
            backbone = any(str(r.get("prot_atom") or "") in ("N", "O", "C", "OXT", "H", "HN") for r in rows)
            shown = f"({rsel}) and not elem H" + ("" if backbone or resn == "GLY" else " and not name N+C+O")
            cmd.show("sticks", shown)
            polar_h = [str(r.get("prot_atom")) for r in rows if str(r.get("prot_atom") or "").startswith("H")]
            if polar_h:
                cmd.show("sticks", f"({rsel}) and name " + "+".join(polar_h))
            cmd.util.cbag(rsel)
            contact_only = all(str(r.get("type")) in CONTACT_TYPES for r in rows)
            if contact_only:
                # contacts: thinner, lighter sticks so the specific interactions stand out
                cmd.color("iv_contact_c", f"({rsel}) and elem C")
                cmd.set("stick_radius", 0.09, rsel)
                cmd.set("stick_transparency", 0.15, rsel)
            else:
                cmd.color("iv_res_c", f"({rsel}) and elem C")
                key_sels.append(rsel)
            residue_sels.append(rsel)
            if contact_only and not label_contacts:
                continue
        # label anchor: the protein atom of the strongest interaction (else the residue atom nearest the ligand)
        anchor = None
        for r in rows:
            if str(r.get("type")) not in CONTACT_TYPES:
                anchor = _coords(r, "prot_coord")
                if anchor is not None:
                    break
        if anchor is None:
            xyz = cmd.get_coords(f"({rsel}) and not elem H")
            if xyz is not None and len(xyz):
                anchor = xyz[int(np.argmin(np.linalg.norm(xyz - lig_center, axis=1)))]
        if anchor is not None:
            labels.append((residue_label(resn, resi, chain, multi_chain), np.asarray(anchor, float), cls))

    for wsel in waters:
        cmd.show("spheres", f"({wsel}) and elem O")
        cmd.set("sphere_scale", 0.26, wsel)
        cmd.color("red", f"({wsel}) and elem O")

    # dashes per type (one CGO object per type, so each can be toggled)
    per_type = OrderedDict()
    rings = []
    dist_pts = []
    for inter in interactions:
        itype = str(inter.get("type"))
        if itype in CONTACT_TYPES and not show_contacts:
            continue
        p, q = _coords(inter, "prot_coord"), _coords(inter, "lig_coord")
        if p is None or q is None:
            continue
        st = style_of(itype)
        rgb = rgb01(st["hex"])
        seg = _dashed_cylinder(p, q, rgb, st["radius"], st["dash"], st["gap"])
        if itype in RING_TYPES:
            rings += _ring_marker(q, rgb)
            if "Pi-Pi" in itype:
                rings += _ring_marker(p, rgb)
        if seg:
            name = f"{prefix}_{st['short'] or itype}".replace("–", "-").replace("π", "pi")
            name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
            per_type.setdefault(name, []).extend(seg)
            if distance_labels and itype not in CONTACT_TYPES:
                dist_pts.append(((p + q) / 2.0, f"{float(inter.get('distance', 0)):.1f}", st["hex"]))
    for name, seg in per_type.items():
        cmd.load_cgo(seg, name)
    if surface:
        pocket = f"byres (({obj}) and polymer within 7 of ({lig_sel}))"
        cmd.create(f"{prefix}_pocket", pocket)
        cmd.hide("everything", f"{prefix}_pocket")
        cmd.show("surface", f"{prefix}_pocket")
        cmd.set("surface_color", "grey90", f"{prefix}_pocket")
        cmd.set("transparency", 0.78, f"{prefix}_pocket")

    focus = f"({lig_sel})" + "".join(f" or ({s})" for s in (key_sels if many else residue_sels) + waters)
    rot = view_into_pocket(obj, lig_sel, focus) if orient else _current_rotation()
    final_view = cmd.get_view()
    origin = np.asarray(final_view[12:15], float)
    lig_xyz = cmd.get_coords(f"({lig_sel}) and not elem H")
    lig_2d = [] if lig_xyz is None else [(rot @ (x - origin))[:2] for x in lig_xyz]
    to_world = lambda xy, z: origin + rot.T @ np.array([xy[0], xy[1], z])      # noqa: E731

    from pymol import cgo

    leaders = []
    placed = []
    lig_boxes = [(x - 0.75, y - 0.75, x + 0.75, y + 0.75) for x, y in lig_2d]
    stick_boxes = []
    for rsel in residue_sels:
        xyz = cmd.get_coords(f"({rsel}) and rep sticks and not elem H")
        for x in ([] if xyz is None else xyz):
            c2 = (rot @ (x - origin))[:2]
            stick_boxes.append((c2[0] - 0.3, c2[1] - 0.3, c2[0] + 0.3, c2[1] + 0.3))
    focus_xyz = cmd.get_coords(f"({focus}) and not elem H")
    z_label = float(((focus_xyz - origin) @ rot[2]).max()) + 1.2 if focus_xyz is not None else 3.0
    span = float(np.ptp((focus_xyz - origin) @ rot[:2].T, axis=0).max()) if focus_xyz is not None else 16.0
    k = max(1.0, span / 17.0)
    res_h, dist_h = RESIDUE_LABEL_A * k, DISTANCE_LABEL_A * k
    lig_c2 = np.mean(lig_2d, axis=0) if lig_2d else np.zeros(2)
    for text, anchor, cls in labels:
        cam = rot @ (anchor - origin)
        box = label_box(text, res_h)
        xy = place_label(cam[:2], lig_c2, box, [(12.0, placed), (4.0, lig_boxes), (0.6, stick_boxes)])
        placed.append((xy[0] - box[0] / 2, xy[1] - box[1] / 2, xy[0] + box[0] / 2, xy[1] + box[1] / 2))
        cmd.pseudoatom(f"{prefix}_labels", pos=[float(v) for v in to_world(xy, z_label)], label=text)
        gap = float(np.linalg.norm(xy - cam[:2]))
        if gap > 2.4:
            # leader line from the residue to its displaced label (stops short of the text)
            end = cam[:2] + (xy - cam[:2]) * (1.0 - 0.6 * box[1] / gap)
            leaders += [cgo.CYLINDER, *anchor, *to_world(end, z_label - 0.3), 0.02, 0.70, 0.73, 0.77,
                        0.70, 0.73, 0.77]
    for mid, text, hexcol in dist_pts:
        cam = rot @ (mid - origin)
        box = label_box(text, dist_h)
        xy = place_label(cam[:2], cam[:2] + np.array([0.0, -1.0]), box,
                         [(12.0, placed), (4.0, lig_boxes), (0.3, stick_boxes)], radii=(0.0, 0.5, 0.9, 1.3))
        placed.append((xy[0] - box[0] / 2, xy[1] - box[1] / 2, xy[0] + box[0] / 2, xy[1] + box[1] / 2))
        name = f"{prefix}_dist"
        cmd.pseudoatom(name, pos=[float(v) for v in to_world(xy, z_label)], label=text)
        cmd.set_color(f"{prefix}_c_{hexcol.lstrip('#')}", list(rgb01(hexcol)))
        cmd.set("label_color", f"{prefix}_c_{hexcol.lstrip('#')}", f"{name} and index {cmd.count_atoms(name)}")
    if leaders:
        cmd.load_cgo(leaders, f"{prefix}_leaders")
    for name, size in ((f"{prefix}_labels", -res_h), (f"{prefix}_dist", -dist_h)):
        if name in cmd.get_names("objects"):
            cmd.hide("everything", name)
            cmd.show("labels", name)
            cmd.set("label_size", size, name)
            cmd.set("label_font_id", 7, name)
            cmd.set("label_position", [0.0, 0.0, 0.0], name)
            cmd.set("label_outline_color", "white", name)
    if f"{prefix}_labels" in cmd.get_names("objects"):
        cmd.set("label_color", "grey15", f"{prefix}_labels")
    cmd.set_view(final_view)          # new objects trigger PyMOL's auto-zoom
    cmd.set("orthoscopic", 1)
    return {"residues": len(groups), "labels": len(labels), "dash_objects": list(per_type), "waters": len(waters)}


# ---------------------------------------------------------------------------------- fonts / drawing

def _font(size, bold=False):
    from PIL import ImageFont

    try:
        import matplotlib

        base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        return ImageFont.truetype(os.path.join(base, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


def _dashed_line(draw, p, q, fill, width, dash, gap):
    p, q = np.asarray(p, float), np.asarray(q, float)
    v = q - p
    length = float(np.linalg.norm(v))
    if length < 1:
        return
    u = v / length
    t = 0.0
    while t < length:
        a = p + u * t
        b = p + u * min(length, t + dash)
        draw.line([tuple(a), tuple(b)], fill=fill, width=width)
        r = width / 2.0
        for c in (a, b):
            draw.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fill)
        t += dash + gap


def legend_items(interactions):
    return [(style_of(t)["short"] or t, style_of(t)["hex"], n) for t, n in summarize(interactions).items()]


def _draw_legend(draw, x, y, max_w, items, scale=1.0):
    """Wrapped row of [dash sample + 'H-bond (2)'] entries. Returns the height used."""
    f = _font(int(22 * scale))
    cx, cy, row_h = x, y, int(34 * scale)
    for text, hexcol, n in items:
        label = f"{text}  {n}"
        w = int(46 * scale) + draw.textbbox((0, 0), label, font=f)[2] + int(30 * scale)
        if cx + w > x + max_w and cx > x:
            cx, cy = x, cy + row_h
        _dashed_line(draw, (cx, cy + 13 * scale), (cx + 36 * scale, cy + 13 * scale), rgb255(hexcol),
                     max(2, int(4 * scale)), 8 * scale, 5 * scale)
        draw.text((cx + 46 * scale, cy), label, font=f, fill=INK)
        cx += w
    return cy + row_h - y


def export_figure(path, interactions, width=1800, height=1200, title="", subtitle=""):
    """Ray-trace the current scene and add a title and a legend panel below it."""
    import tempfile

    from PIL import Image, ImageDraw
    from pymol import cmd

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    try:
        cmd.png(tmp, width=int(width), height=int(height), dpi=300, ray=1)
        scene = Image.open(tmp).convert("RGB")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    pad = 40
    header = 90 if title else 0
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    legend_h = _draw_legend(probe, 0, 0, scene.width - 2 * pad, legend_items(interactions)) + 40
    out = Image.new("RGB", (scene.width, scene.height + header + legend_h), "white")
    out.paste(scene, (0, header))
    d = ImageDraw.Draw(out)
    if title:
        d.text((pad, 22), title, font=_font(34, True), fill=INK)
        if subtitle:
            d.text((pad + d.textbbox((0, 0), title, font=_font(34, True))[2] + 24, 34), subtitle,
                   font=_font(22), fill=MUTED)
    y = header + scene.height + 14
    d.line((pad, y - 6, scene.width - pad, y - 6), fill=(229, 231, 235), width=2)
    _draw_legend(d, pad, y + 6, scene.width - 2 * pad, legend_items(interactions))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out.save(path)
    return path


# ----------------------------------------------------------------------------------------- ligand 2D

def ligand_molecule(atoms, resn="", fetch_ccd=True, ph=7.4):
    """
    RDKit molecule (heavy atoms, correct bond orders and formal charges) from the ligand's atoms
    ``[(name, element, xyz), ...]``. Bond orders come from the PDB chemical component template when it
    matches, else from the geometry; charges from Leonardo's pKa model at ``ph``.
    Returns (mol, {atom name: index}, info).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    heavy = [(str(n).strip(), str(e).strip().capitalize(), np.asarray(x, float)) for n, e, x in atoms
             if str(e).strip().upper() not in ("H", "D")]
    if not heavy:
        raise ValueError("the ligand has no heavy atoms")
    rw = Chem.RWMol()
    for _name, elem, _x in heavy:
        rw.AddAtom(Chem.Atom(elem if elem else "C"))
    xyz = np.array([x for _n, _e, x in heavy])
    pt = Chem.GetPeriodicTable()
    for i in range(len(heavy)):
        for j in range(i + 1, len(heavy)):
            ri = pt.GetRcovalent(rw.GetAtomWithIdx(i).GetAtomicNum())
            rj = pt.GetRcovalent(rw.GetAtomWithIdx(j).GetAtomicNum())
            if float(np.linalg.norm(xyz[i] - xyz[j])) <= ri + rj + 0.45:
                rw.AddBond(i, j, Chem.BondType.SINGLE)
    conf = Chem.Conformer(len(heavy))
    for i, x in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in x]))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    mol.UpdatePropertyCache(strict=False)
    names = {n: i for i, (n, _e, _x) in enumerate(heavy)}
    info = {"bond_orders": "single bonds only", "charges": "none"}

    assigned = None
    try:
        from .component_library_runtime import ccd_reference

        ref = ccd_reference(resn, fetch=fetch_ccd) if resn else None
        if ref is not None and ref.GetNumAtoms() == mol.GetNumAtoms():
            assigned = AllChem.AssignBondOrdersFromTemplate(ref, mol)
            info["bond_orders"] = f"PDB chemical component {str(resn).upper()}"
    except Exception:
        assigned = None
    if assigned is None:
        try:
            from .component_library_runtime import _bond_orders_from_geometry

            assigned = _bond_orders_from_geometry(mol)
            info["bond_orders"] = "perceived from the 3D geometry"
        except Exception:
            assigned = mol
    try:
        Chem.SanitizeMol(assigned)
    except Exception:
        assigned = mol
    try:
        from .pka_runtime import protonate_molecule

        prot, pinfo = protonate_molecule(assigned, ph=float(ph), add_hs=False)
        if prot.GetNumAtoms() == assigned.GetNumAtoms():
            for i in range(assigned.GetNumAtoms()):
                assigned.GetAtomWithIdx(i).SetFormalCharge(prot.GetAtomWithIdx(i).GetFormalCharge())
                assigned.GetAtomWithIdx(i).SetNumExplicitHs(prot.GetAtomWithIdx(i).GetTotalNumHs())
                assigned.GetAtomWithIdx(i).SetNoImplicit(True)
            Chem.SanitizeMol(assigned)
            info["charges"] = f"pH {float(ph):.1f} (Leonardo pKa model, net {int(Chem.GetFormalCharge(assigned)):+d})"
    except Exception as exc:
        info["charges"] = f"neutral ({exc.__class__.__name__})"
    return assigned, names, info


def _rect_overlap(a, b, margin=0):
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def _seg_hits_rect(p, q, r):
    """Does the segment p-q pass through rectangle r (sampled)?"""
    for t in np.linspace(0.1, 0.9, 9):
        x, y = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
        if r[0] <= x <= r[2] and r[1] <= y <= r[3]:
            return True
    return False


def draw_2d_map(path, interactions, ligand_atoms, *, resn="", title="", subtitle="", scale=2, fetch_ccd=True,
                ph=7.4, show_contacts=True):
    """
    Draw the 2D interaction diagram to ``path`` (PNG). ``ligand_atoms`` is ``[(name, element, xyz), ...]``.
    Returns a dict with the image path and what was drawn.
    """
    from PIL import Image, ImageDraw
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    mol, names, info = ligand_molecule(ligand_atoms, resn=resn, fetch_ccd=fetch_ccd, ph=ph)
    rdDepictor.SetPreferCoordGen(True)
    AllChem.Compute2DCoords(mol)
    # orient the depiction like the 3D view is not meaningful; keep RDKit's canonical layout

    S = int(scale)
    W, H = 1400 * S, 1050 * S
    mol_w, mol_h = 700 * S, 560 * S
    drawer = rdMolDraw2D.MolDraw2DCairo(mol_w, mol_h)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 3 * S
    opts.minFontSize = 15 * S
    opts.maxFontSize = 22 * S
    opts.padding = 0.06
    opts.additionalAtomLabelPadding = 0.12
    opts.fixedBondLength = 66 * S
    try:
        opts.setAtomPalette({0: (0.1, 0.1, 0.1), 6: (0.12, 0.12, 0.12), 7: (0.12, 0.33, 0.85),
                             8: (0.85, 0.15, 0.15), 9: (0.1, 0.6, 0.2), 15: (0.9, 0.5, 0.0),
                             16: (0.8, 0.6, 0.0), 17: (0.1, 0.6, 0.2), 35: (0.6, 0.2, 0.1), 53: (0.5, 0.1, 0.6)})
    except Exception:
        pass
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    import io

    mol_img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")
    ox, oy = (W - mol_w) // 2, (H - mol_h) // 2 + 20 * S
    atom_xy = {}
    for i in range(mol.GetNumAtoms()):
        pt = drawer.GetDrawCoords(i)
        atom_xy[i] = np.array([pt.x + ox, pt.y + oy])
    all_xy = np.array(list(atom_xy.values()))
    center = all_xy.mean(axis=0)

    def anchor_of(inter):
        """2D point a dash ends on: the ligand atom, or the centroid of the ligand ring for ring types."""
        idx = names.get(str(inter.get("lig_atom") or "").strip())
        if idx is not None:
            return atom_xy[idx], [idx]
        lc = _coords(inter, "lig_coord")
        if lc is None:
            return center, []
        # ring centroid: average the 2D positions of the ligand atoms nearest to the 3D point
        d = {n: float(np.linalg.norm(np.asarray(x) - lc)) for n, _e, x in ligand_atoms if n.strip() in names}
        near = [names[n.strip()] for n, dist in sorted(d.items(), key=lambda kv: kv[1])[:6] if dist < 2.0]
        if not near:
            near = [names[min(d, key=d.get).strip()]] if d else []
        return (np.mean([atom_xy[i] for i in near], axis=0) if near else center), near

    groups = group_by_residue([r for r in interactions if show_contacts or str(r.get("type")) not in CONTACT_TYPES])
    multi_chain = len({k[0] for k in groups if k[0]}) > 1
    f_res = _font(19 * S, True)
    f_small = _font(14 * S)
    placed, lines = [], []
    occupied = [(x - 26 * S, y - 22 * S, x + 26 * S, y + 22 * S) for x, y in all_xy]
    order = sorted(groups.items(), key=lambda kv: min(list(TYPE_STYLE).index(str(r.get("type")))
                                                    if str(r.get("type")) in TYPE_STYLE else 99 for r in kv[1]))
    for key, rows in order:
        chain, resi, resn_r = key
        anchors = [anchor_of(r) for r in rows]
        a_pts = np.array([a for a, _ in anchors])
        a_mean = a_pts.mean(axis=0)
        text = residue_label(resn_r, resi, chain, multi_chain)
        tw, th = (f_res.getbbox(text)[2] - f_res.getbbox(text)[0]) + 30 * S, 44 * S
        direction = a_mean - center
        if np.linalg.norm(direction) < 1:
            direction = np.array([0.0, -1.0])
        base_ang = math.atan2(direction[1], direction[0])
        best, best_cost = None, None
        for dist in (150, 180, 215, 255, 300, 350, 410):
            for dang in (0, 12, -12, 24, -24, 36, -36, 50, -50, 65, -65, 80, -80, 100, -100, 125, -125, 150, -150,
                         180):
                ang = base_ang + math.radians(dang)
                cx = a_mean[0] + dist * S * math.cos(ang)
                cy = a_mean[1] + dist * S * math.sin(ang)
                rect = (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2)
                if rect[0] < 20 * S or rect[2] > W - 20 * S or rect[1] < 110 * S or rect[3] > H - 120 * S:
                    continue
                if any(_rect_overlap(rect, o, 8 * S) for o in occupied):
                    continue
                cost = dist + 0.9 * abs(dang)
                cost += sum(60 for p, _ in anchors for o in occupied[len(all_xy):] if _seg_hits_rect(p, (cx, cy), o))
                cost += sum(40 for (p, q) in lines for a, _ in anchors if _segments_cross(p, q, a, (cx, cy)))
                if best_cost is None or cost < best_cost:
                    best, best_cost = rect, cost
            if best is not None and best_cost < dist + 200:
                break
        if best is None:
            continue
        occupied.append(best)
        c = ((best[0] + best[2]) / 2, (best[1] + best[3]) / 2)
        placed.append((key, rows, best, text, anchors))
        for a, _ in anchors:
            lines.append((a, c))

    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    d = ImageDraw.Draw(canvas)
    # soft halo behind the ligand
    pad = 34 * S
    x0, y0 = all_xy.min(axis=0) - pad
    x1, y1 = all_xy.max(axis=0) + pad
    d.rounded_rectangle((x0, y0, x1, y1), radius=40 * S, fill=(250, 250, 251), outline=(236, 237, 240), width=2 * S)

    dist_labels = []
    for key, rows, rect, text, anchors in placed:
        c = np.array(((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2))
        for r, (a, _idx) in zip(rows, anchors):
            st = style_of(str(r.get("type")))
            col = rgb255(st["hex"])
            v = c - a
            n = float(np.linalg.norm(v)) or 1.0
            start = a + v / n * 16 * S
            # stop at the pill border
            end = _to_rect_border(a, c, rect)
            contact = str(r.get("type")) in CONTACT_TYPES
            width = (2 if contact else 4) * S
            _dashed_line(d, start, end, col, width, (7 if contact else 11) * S, (7 if contact else 7) * S)
            if not contact:
                # on the dash, as far from the ligand atoms as the dash allows
                cands = [start + (end - start) * t for t in (0.5, 0.62, 0.38, 0.75)]
                pos = max(cands, key=lambda c: float(np.min(np.linalg.norm(all_xy - c, axis=1))) - 0.3 * float(
                    np.linalg.norm(c - (start + end) / 2)))
                dist_labels.append((pos, f"{float(r.get('distance', 0)):.1f}", col))
            if str(r.get("type")) in RING_TYPES:
                rr = 9 * S
                d.ellipse((a[0] - rr, a[1] - rr, a[0] + rr, a[1] + rr), outline=col, width=3 * S)

    mol_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mol_layer.paste(mol_img, (ox, oy), mol_img)
    canvas = Image.alpha_composite(canvas, mol_layer)
    d = ImageDraw.Draw(canvas)

    for key, rows, rect, text, anchors in placed:
        fill, outline = RES_CLASS[residue_class(key[2])]
        d.rounded_rectangle(rect, radius=(rect[3] - rect[1]) / 2, fill=fill, outline=outline, width=3 * S)
        bb = d.textbbox((0, 0), text, font=f_res)
        d.text(((rect[0] + rect[2]) / 2 - (bb[2] - bb[0]) / 2 - bb[0], (rect[1] + rect[3]) / 2 - (bb[3] - bb[1]) / 2
                - bb[1]), text, font=f_res, fill=INK)
    for (x, y), label, col in dist_labels:
        bb = d.textbbox((0, 0), label, font=f_small)
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        d.rounded_rectangle((x - w / 2 - 6 * S, y - h / 2 - 5 * S, x + w / 2 + 6 * S, y + h / 2 + 5 * S),
                            radius=6 * S, fill=(255, 255, 255), outline=col, width=1 * S)
        d.text((x - w / 2 - bb[0], y - h / 2 - bb[1]), label, font=f_small, fill=col)

    # crop the diagram to its content, then add the header and the legend
    boxes = [r for _k, _rows, r, *_ in placed] + [(x0, y0, x1, y1)]
    m = 40 * S
    cx0 = max(0, int(min(b[0] for b in boxes) - m))
    cy0 = max(0, int(min(b[1] for b in boxes) - m))
    cx1 = min(W, int(max(b[2] for b in boxes) + m))
    cy1 = min(H, int(max(b[3] for b in boxes) + m))
    body = canvas.crop((cx0, cy0, cx1, cy1))
    used = [r for _k, rows, *_ in placed for r in rows]
    out_w = max(body.width, 1100 * S)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    leg_h = _draw_legend(probe, 0, 0, out_w - 80 * S, legend_items(used), scale=S)
    head_h = (110 if subtitle else 80) * S if title else 30 * S
    out = Image.new("RGB", (out_w, head_h + body.height + leg_h + 70 * S), "white")
    out.paste(body.convert("RGB"), ((out_w - body.width) // 2, head_h))
    d = ImageDraw.Draw(out)
    if title:
        d.text((40 * S, 30 * S), title, font=_font(30 * S, True), fill=INK)
    if subtitle:
        d.text((40 * S, 72 * S), subtitle, font=_font(18 * S), fill=MUTED)
    y_leg = head_h + body.height + 10 * S
    d.line((40 * S, y_leg - 10 * S, out_w - 40 * S, y_leg - 10 * S), fill=(229, 231, 235), width=2 * S)
    _draw_legend(d, 40 * S, y_leg, out_w - 80 * S, legend_items(used), scale=S)
    for k, note in enumerate(("Distances in Å between the interacting atoms (ring centroids for π interactions).",
                              f"Ligand: bond orders from {info['bond_orders']}; charges at {info['charges']}.")):
        d.text((40 * S, y_leg + leg_h + (14 + 20 * k) * S), note, font=_font(13 * S), fill=MUTED)
    dx, dy = (out_w - body.width) // 2 - cx0, head_h - cy0
    index_of = {id(r): i for i, r in enumerate(interactions)}
    hotspots = [{"residue": text, "rect": [(rect[0] + dx) / S, (rect[1] + dy) / S, (rect[2] + dx) / S,
                                           (rect[3] + dy) / S],
                 "interactions": [index_of[id(r)] for r in rows if id(r) in index_of]}
                for _key, rows, rect, text, _anchors in placed]
    img = out.resize((out.width // S, out.height // S), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    img.save(path)
    missing = [k for k in groups if k not in {p[0] for p in placed}]
    return {"image": path, "residues": len(placed), "not_placed": [residue_label(k[2], k[1], k[0]) for k in missing],
            "ligand": info, "smiles": Chem.MolToSmiles(mol), "hotspots": hotspots, "size": list(img.size)}


def _segments_cross(p1, p2, q1, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = ccw(q1, q2, p1), ccw(q1, q2, p2)
    d3, d4 = ccw(p1, p2, q1), ccw(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _to_rect_border(a, c, rect):
    """Point where the segment from ``a`` to the rectangle centre ``c`` enters the rectangle."""
    a, c = np.asarray(a, float), np.asarray(c, float)
    v = a - c
    hw, hh = (rect[2] - rect[0]) / 2.0, (rect[3] - rect[1]) / 2.0
    if abs(v[0]) < 1e-9 and abs(v[1]) < 1e-9:
        return c
    t = min(hw / abs(v[0]) if abs(v[0]) > 1e-9 else 1e9, hh / abs(v[1]) if abs(v[1]) > 1e-9 else 1e9)
    return c + v * min(1.0, t)


def ligand_atoms_from_pymol(sel):
    """[(name, element, xyz)] of a PyMOL selection."""
    from pymol import cmd

    model = cmd.get_model(sel)
    return [(a.name, a.symbol, tuple(a.coord)) for a in model.atom]
