"""
PyMOL stage of the 60 s Leonardo trailer: ray-traces the molecular layer of every scene from real
calculation outputs (artifacts/trailer_2026/data, written by Leonardo's tools and replay packs).

Run inside PyMOL (no GUI):
    pymol -cq scripts/trailer_molecules.py -- <scene> [first_frame last_frame] [--test]

Frames go to artifacts/trailer_2026/frames/<scene>/mol_NNNN.png; existing frames are kept, so an
interrupted run resumes. Per-frame screen positions of labelled atoms go to <scene>/anchors.json.
"""
import json
import math
import os
import sys

import numpy as np
from pymol import cmd

ROOT = os.environ.get("LEONARDO_REPO", os.getcwd())   # run from the repository root (PyMOL hides __file__)
DATA = os.path.join(ROOT, "artifacts", "trailer_2026", "data")
FRAMES = os.path.join(ROOT, "artifacts", "trailer_2026", "frames")
W, H, FPS = 1920, 1080, 24
BG = (0.012, 0.024, 0.05)
# (start s, end s) of each scene in the 60 s cut; the intro shares the docking shot
SCENES = {"docking": (0.0, 15.0), "md": (15.0, 25.0), "qmmm": (25.0, 36.0), "design": (36.0, 44.0),
          "pka": (44.0, 52.0), "outro": (52.0, 60.0)}


def nframes(scene):
    a, b = SCENES[scene]
    return int(round((b - a) * FPS))


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def smootherstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * x * (x * (x * 6 - 15) + 10)


# ------------------------------------------------------------------------------------------ look
def base_look():
    # trajectories are written in file order: PyMOL must not sort the atoms it loads
    cmd.set("retain_order", 1)
    cmd.set("bg_rgb", list(BG))
    cmd.set("ray_opaque_background", 1)
    cmd.set("ray_shadows", 1)
    cmd.set("ray_shadow_decay_factor", 0.1)
    cmd.set("depth_cue", 1)
    cmd.set("fog_start", 0.42)
    cmd.set("ambient", 0.18)
    cmd.set("direct", 0.62)
    cmd.set("reflect", 0.42)
    cmd.set("specular", 0.22)
    cmd.set("spec_power", 90)
    cmd.set("spec_reflect", 0.3)
    cmd.set("light_count", 3)
    cmd.set("antialias", 2)
    cmd.set("hash_max", 300)
    cmd.set("max_threads", 6)
    cmd.set("surface_quality", 1)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_highlight_color", -1)
    cmd.set("cartoon_sampling", 14)
    cmd.set("stick_quality", 24)
    cmd.set("sphere_quality", 3)
    cmd.set("field_of_view", 22)
    cmd.set("ray_interior_color", "0x0a1830")      # faces cut open by the front clipping plane
    cmd.set("orthoscopic", 0)


def centroid(sel, state=1):
    xyz = cmd.get_coords(sel, state=state)
    return xyz.mean(axis=0)


def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


def set_camera(target, direction, distance, up=(0.0, 1.0, 0.0), roll_deg=0.0, depth=80.0, front=None, left=0.0,
               raise_=0.0):
    """Camera on target + direction*distance, looking at target; PyMOL view (column-major rotation).
    left: put the target this fraction of the half-width left of the screen centre (room for a data panel);
    raise_: the same upward, as a fraction of the half-height (vertical formats keep the panels below)."""
    z = unit(direction)
    up = np.asarray(up, float)
    if abs(float(np.dot(unit(up), z))) > 0.98:
        up = np.array([1.0, 0.0, 0.0])
    x = unit(np.cross(up, z))
    y = np.cross(z, x)
    if roll_deg:
        r = math.radians(roll_deg)
        x, y = x * math.cos(r) + y * math.sin(r), -x * math.sin(r) + y * math.cos(r)
    half_w = distance * math.tan(math.radians(float(cmd.get("field_of_view")) / 2)) * W / H
    half_h = distance * math.tan(math.radians(float(cmd.get("field_of_view")) / 2))
    target = np.asarray(target, float) + x * left * half_w - y * raise_ * half_h
    view = [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2],
            0.0, 0.0, -float(distance), float(target[0]), float(target[1]), float(target[2]),
            max(1.0, distance - depth) if front is None else max(1.0, front), distance + depth, 0.0]
    cmd.set_view(view)


def rotate(v, axis, deg):
    """Rodrigues rotation of v about axis."""
    k = unit(axis)
    v = np.asarray(v, float)
    t = math.radians(deg)
    return v * math.cos(t) + np.cross(k, v) * math.sin(t) + k * np.dot(k, v) * (1 - math.cos(t))


def screen_xy(points):
    """Pixel coordinates (W x H frame) of model points under the current view."""
    v = cmd.get_view()
    R = np.array([[v[0], v[3], v[6]], [v[1], v[4], v[7]], [v[2], v[5], v[8]]])
    origin = np.array(v[12:15])
    cam = np.array(v[9:12])
    fov = math.radians(float(cmd.get("field_of_view")))
    out = []
    for p in points:
        c = R @ (np.asarray(p, float) - origin) + cam
        depth = -c[2]
        f = (H / 2) / math.tan(fov / 2)
        out.append([W / 2 + f * c[0] / depth, H / 2 - f * c[1] / depth])
    return out


def wanted(scene, i):
    """Frame i must be rendered: missing, or the single frame asked for with --t=SECONDS (scene time)."""
    for a in sys.argv:
        if a.startswith("--t="):
            return i == int(round(float(a[4:]) * FPS))
    return not os.path.exists(frame_path(scene, i))


def frame_path(scene, i):
    return os.path.join(FRAMES, scene, f"mol_{i:04d}.png")


def ray_frame(scene, i):
    os.makedirs(os.path.join(FRAMES, scene), exist_ok=True)
    tmp = frame_path(scene, i) + ".tmp.png"
    cmd.ray(W, H)
    cmd.png(tmp, dpi=150)
    os.replace(tmp, frame_path(scene, i))


# ------------------------------------------------------------------------------ docking (+ intro)
def docking():
    """Trypsin + benzamidine redocked by Leonardo (pack 01: RMSD 0.39 A, -6.11 kcal/mol). The 15 s shot
    opens far away under the title (intro) and flies into the S1 pocket."""
    cmd.load(os.path.join(DATA, "trypsin_receptor.pdb"), "rec")
    # the pose Leonardo showed (rank 1 of the smina run): the first record of the docked SDF
    with open(os.path.join(DATA, "benzamidine_docked.sdf")) as fh:
        first = fh.read().split("$$$$")[0] + "$$$$\n"
    cmd.read_molstr(first.split("\nM  END")[0] + "\nM  END\n", "lig")
    cmd.load(os.path.join(DATA, "benzamidine_crystal.pdb"), "xtal")
    cmd.remove("hydro and (rec or xtal)")
    cmd.remove("rec and not polymer")
    cmd.hide("everything")
    lig_c = centroid("lig")
    prot_c = centroid("rec and name CA")
    # colour the surface by distance to the pocket: a lighter rim leads the eye in
    cmd.alter_state(1, "rec", "b=((x-cx)**2+(y-cy)**2+(z-cz)**2)**0.5",
                    space={"cx": float(lig_c[0]), "cy": float(lig_c[1]), "cz": float(lig_c[2])})
    cmd.show("surface", "rec")
    cmd.set("surface_color", -1)
    cmd.spectrum("b", "0x7fc8ff 0x2e64b4 0x16305e 0x0b1a36", "rec", minimum=5, maximum=34)
    cmd.set("transparency", 0.0)
    cmd.show("sticks", "lig")
    cmd.show("spheres", "lig")
    cmd.set("sphere_scale", 0.24, "lig")
    cmd.set("stick_radius", 0.17, "lig")
    cmd.color("0xffb347", "lig and elem C")
    cmd.color("0x6fa8ff", "lig and elem N")
    cmd.color("0xf4f7fb", "lig and elem H")
    cmd.show("sticks", "xtal")
    cmd.color("0xffffff", "xtal")
    cmd.set("stick_radius", 0.11, "xtal")
    # toward the pocket mouth: away from the protein atoms that surround the ligand
    around = cmd.get_coords("rec within 12 of lig")
    outward = -unit(np.mean([unit(p - lig_c) for p in around], axis=0))
    outward = unit(outward * 0.75 + unit(lig_c - prot_c) * 0.25)
    pocket = "(byres (rec within 4.5 of lig))"
    revealed = False
    n = nframes("docking")
    anchors = {}
    for i in range(n):
        t = i / FPS
        if not wanted("docking", i):
            continue
        # 0-6.5 s: slow orbit far away (title over it); 6.5-12 s: fly into the pocket; 12-15 s: hold
        fly = smootherstep((t - 5.5) / 6.5)
        orbit = 70.0 * (1 - fly) + 8.0 * math.sin(t * 0.35)
        direction = rotate(outward, (0.0, 1.0, 0.0), orbit)
        direction = rotate(direction, (1.0, 0.0, 0.0), -18.0 * (1 - fly))
        dist = 150.0 * (1 - fly) + 26.0 * fly
        target = prot_c * (1 - fly) + lig_c * fly
        # the reveal (masked by a light flash in compositing): the pocket residues leave the surface and
        # stay as thin sticks, opening a crater onto the docked ligand
        if t >= 10.2 and not revealed:
            cmd.flag("ignore", pocket, "set")
            cmd.rebuild()
            cmd.show("sticks", pocket + " and not name N+C+O+CA and not resn GLY")
            cmd.set("stick_radius", 0.14, "rec")
            cmd.set("stick_transparency", 0.15, "rec")
            cmd.color("0x9fc6ff", pocket + " and elem C")
            cmd.color("0xff6b6b", pocket + " and elem O")
            cmd.color("0x8ab4ff", pocket + " and elem N")
            revealed = True
        set_camera(target, direction, dist, roll_deg=6.0 * (1 - fly))
        lig_a = ease((t - 10.2) / 1.2)
        cmd.set("stick_transparency", 1 - lig_a, "lig")
        cmd.set("sphere_transparency", 1 - lig_a, "lig")
        cmd.set("stick_transparency", 1 - 0.55 * ease((t - 12.4) / 1.0), "xtal")
        if lig_a <= 0.0:
            cmd.hide("sticks", "lig")
            cmd.hide("spheres", "lig")
        else:
            cmd.show("sticks", "lig")
            cmd.show("spheres", "lig")
        if t < 12.4:
            cmd.hide("sticks", "xtal")
        else:
            cmd.show("sticks", "xtal")
        # the pocket is cut open as the camera arrives (front slab)
        anchors[i] = {"ligand": screen_xy([lig_c])[0]}
        ray_frame("docking", i)
    return anchors


# ------------------------------------------------------------------------------------------- md
def md():
    """100 ps of trypsin, Amber14/GB-OBC2 in OpenMM through Leonardo (pack 03): 500 real frames."""
    z = np.load(os.path.join(DATA, "md_aligned.npz"))
    xyz = z["xyz"]
    cmd.load(os.path.join(DATA, "md_trypsin_amber.pdb"), "md")
    cmd.hide("everything")
    cmd.show("cartoon", "md")
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("cartoon_oval_length", 1.25)
    cmd.set("cartoon_loop_radius", 0.28)
    cmd.spectrum("resi", "0x1b3766 0x2e64b4 0x4fa3ff 0x2be3c8 0x4fa3ff 0x2e64b4 0x1b3766", "md and name CA")
    cmd.set("cartoon_loop_radius", 0.32)
    cmd.set("cartoon_rect_length", 1.5)
    triad = "md and ((resi 57 and resn HI*) or (resi 102 and resn ASP) or (resi 195 and resn SER)) and not name N+C+O+H*"
    cmd.show("sticks", triad)
    cmd.set("stick_radius", 0.24)
    cmd.color("0x2be3c8", triad + " and elem C")
    cmd.color("0xff6b6b", triad + " and elem O")
    cmd.color("0x8ab4ff", triad + " and elem N")
    cmd.show("spheres", "md and resi 195 and name OG")
    cmd.show("spheres", "md and resi 57 and name NE2")
    cmd.set("sphere_scale", 0.42)
    cmd.color("0xffb347", "md and resi 195 and name OG")
    cmd.color("0xffb347", "md and resi 57 and name NE2")
    site = centroid(triad)
    center = centroid("md and name CA")
    base = unit(site - center)
    n = nframes("md")
    anchors = {}
    for i in range(n):
        t = i / FPS
        k = min(len(xyz) - 1, int(round(i * (len(xyz) - 1) / (n - 1))))
        cmd.load_coords(xyz[k], "md")
        if not wanted("md", i):
            continue
        direction = rotate(rotate(base, (0, 1, 0), -35 + 70 * smootherstep(t / 10.0)), (1, 0, 0), 12)
        dist = 66 - 14 * smootherstep(t / 10.0)
        target = center * 0.55 + site * 0.45
        set_camera(target, direction, dist, roll_deg=-4)
        og = cmd.get_coords("md and resi 195 and name OG")[0]
        ne2 = cmd.get_coords("md and resi 57 and name NE2")[0]
        anchors[i] = {"frame": k, "ser195": screen_xy([og])[0], "his57": screen_xy([ne2])[0]}
        ray_frame("md", i)
    return anchors


# ---------------------------------------------------------------------------------------- qmmm
def qmmm():
    """DhlA SN2 dehalogenation, QM/MM reaction path through Leonardo's run_qmmm (pack 02): 259 real
    geometries (NEB + TS + IRC); the Asp124 carboxylate O-C bond forms while C-Cl breaks."""
    from pymol.cgo import COLOR, CYLINDER, SPHERE

    z = np.load(os.path.join(DATA, "qmmm_path.npz"))
    coords = z["coords"]
    ts = int(z["ts_index"])
    cmd.load(os.path.join(DATA, "qmmm_dhla_protein.pdb"), "prot")
    cmd.load(os.path.join(DATA, "qmmm_qm_atoms.pdb"), "qm")
    cmd.remove("prot and resi 124 and not name N+CA+C+O")      # the QM copy of Asp124 is drawn instead
    cmd.remove("prot and resn DCE")
    cmd.remove("prot and hydro")
    cmd.hide("everything")
    names = [a.name for a in cmd.get_model("qm").atom]
    i_od2, i_c1, i_cl1 = names.index("OD2"), names.index("C1"), names.index("CL1")
    # chemical bonds only (PyMOL guessed them from distances in one frame); the two reacting bonds are drawn
    # per frame from their real length
    cmd.unbond("qm", "qm")
    bonds = [("N", "H"), ("N", "CA"), ("CA", "HA"), ("CA", "C"), ("C", "O"), ("CA", "CB"), ("CB", "HB2"),
             ("CB", "HB3"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2"), ("C1", "C2"), ("C2", "CL2"),
             ("C1", "H01"), ("C1", "H02"), ("C2", "H03"), ("C2", "H04")]
    for a, b in bonds:
        cmd.bond(f"qm and name {a}", f"qm and name {b}")
    cmd.show("cartoon", "prot")
    cmd.set("cartoon_transparency", 0.72, "prot")
    cmd.color("0x23477e", "prot")
    lining = "(byres (prot within 4.2 of qm)) and not name N+C+O+CA"
    cmd.show("sticks", lining)
    cmd.set("stick_transparency", 0.55, "prot")
    cmd.set("stick_radius", 0.11, "prot")
    cmd.color("0x4f74ab", lining)
    cmd.show("sticks", "qm")
    cmd.show("spheres", "qm")
    cmd.set("sphere_scale", 0.24, "qm")
    cmd.set("sphere_scale", 0.15, "qm and elem H")
    cmd.set("sphere_scale", 0.40, "qm and elem Cl")
    cmd.set("stick_radius", 0.15, "qm")
    cmd.color("0xd9e2ee", "qm and elem C")
    cmd.color("0xff5a5f", "qm and elem O")
    cmd.color("0x7dff9b", "qm and elem Cl")
    cmd.color("0x8ab4ff", "qm and elem N")
    cmd.color("0xffffff", "qm and elem H")
    n = nframes("qmmm")
    # scene time -> path index: reach the transition state at 45% of the shot, slowing down around it
    u = np.linspace(0, 1, n)
    idx = np.where(u < 0.45, ts * np.sin(np.clip(u / 0.45, 0, 1) * np.pi / 2) ** 0.8,
                   ts + (len(coords) - 1 - ts) * (1 - np.cos(np.clip((u - 0.45) / 0.55, 0, 1) * np.pi / 2)) ** 0.9)
    idx = np.clip(np.round(idx).astype(int), 0, len(coords) - 1)
    center = coords[ts][[i_od2, i_c1, i_cl1]].mean(axis=0)
    prot_c = centroid("prot and name CA")
    axis = unit(coords[ts][i_cl1] - coords[ts][i_od2])            # the SN2 line: O -> C -> Cl
    away = unit(center - prot_c)
    view_dir = unit(away - np.dot(away, axis) * axis)               # perpendicular to the reaction axis
    up = np.cross(view_dir, axis)
    anchors = {}
    for i in range(n):
        t = i / FPS
        k = int(idx[i])
        xyz = coords[k]
        cmd.load_coords(xyz, "qm")
        if not wanted("qmmm", i):
            continue
        cmd.delete("bonds_cgo")
        obj = []
        for a, b, rgb, full in ((i_od2, i_c1, (1.0, 0.70, 0.28), 1.43), (i_c1, i_cl1, (0.49, 1.0, 0.61), 1.80)):
            d = float(np.linalg.norm(xyz[a] - xyz[b]))
            strength = max(0.0, min(1.0, (3.0 - d) / (3.0 - full)))
            if strength <= 0.02:
                continue
            r = 0.04 + 0.12 * strength
            obj += [CYLINDER, *map(float, xyz[a]), *map(float, xyz[b]), r, *rgb, *rgb]
            mid = (xyz[a] + xyz[b]) / 2
            obj += [COLOR, *rgb, SPHERE, *map(float, mid), 0.04 + 0.08 * strength]
        if obj:
            cmd.load_cgo(obj, "bonds_cgo")
        swing = -22 + 44 * smootherstep(t / 11.0)
        direction = rotate(view_dir, up, swing)
        dist = 31 - 5 * smootherstep(t / 11.0)
        set_camera(center, direction, dist, up=up, depth=24, left=0.26)
        anchors[i] = {"path_index": k, "o": screen_xy([xyz[i_od2]])[0], "c": screen_xy([xyz[i_c1]])[0],
                      "cl": screen_xy([xyz[i_cl1]])[0]}
        ray_frame("qmmm", i)
    return anchors


# -------------------------------------------------------------------------------------- design
def design():
    """GB1 K10C/E56C disulfide designed by Leonardo's design_disulfides (pack 06); the cartoon is coloured
    by the computed drop in residue mobility (elastic network, original -> design)."""
    from pymol.cgo import CYLINDER

    net = np.load(os.path.join(DATA, "gb1_network.npz"))
    drop = net["original_mobility"] - net["design_mobility"]
    cmd.load(os.path.join(DATA, "gb1_design_K10C_E56C.pdb"), "gb1")
    cmd.remove("hydro")
    cmd.hide("everything")
    cmd.show("cartoon", "gb1")
    cmd.set("cartoon_oval_length", 1.3)
    cmd.alter("gb1", "b=0")
    for r, v in zip(net["resids"], drop):
        cmd.alter(f"gb1 and resi {int(r)}", f"b={float(v):.5f}")
    # colour along the chain (the computed stiffening is local; its number goes in the data panel)
    cmd.spectrum("resi", "0x1b3766 0x2e64b4 0x4fa3ff 0x2be3c8 0x4fa3ff 0x2e64b4", "gb1 and name CA")
    cmd.set("cartoon_loop_radius", 0.34)
    cys = "gb1 and resi 10+56 and resn CYS and not name N+C+O"
    cmd.show("sticks", cys)
    cmd.set("stick_radius", 0.26)
    cmd.color("0xd9e2ee", cys + " and elem C")
    cmd.color("0xffd23f", cys + " and elem S")
    cmd.show("spheres", cys + " and elem S")
    cmd.set("sphere_scale", 0.45)
    cmd.unbond("gb1 and resi 10 and name SG", "gb1 and resi 56 and name SG")
    s1 = cmd.get_coords("gb1 and resi 10 and name SG")[0]
    s2 = cmd.get_coords("gb1 and resi 56 and name SG")[0]
    center = centroid("gb1 and name CA")
    mid = (s1 + s2) / 2
    # the domain's long axis runs across the screen; the camera faces the side with the new disulfide
    ca = cmd.get_coords("gb1 and name CA") - center
    _w, vecs = np.linalg.eigh(np.cov(ca.T))
    long_axis = vecs[:, 2]
    base = unit(mid - center)
    base = unit(base - np.dot(base, long_axis) * long_axis)
    up = np.cross(base, long_axis)
    n = nframes("design")
    anchors = {}
    for i in range(n):
        t = i / FPS
        if not wanted("design", i):
            continue
        grow = smootherstep((t - 1.4) / 1.6)
        cmd.delete("ss_cgo")
        if grow > 0:
            end = s1 + (s2 - s1) * grow
            cmd.load_cgo([CYLINDER, *map(float, s1), *map(float, end), 0.24, 1.0, 0.84, 0.25, 1.0, 0.84, 0.25],
                         "ss_cgo")
        direction = rotate(base, long_axis, 18 - 36 * smootherstep(t / 8.0))
        direction = rotate(direction, up, -14 + 28 * smootherstep(t / 8.0))
        dist = 80 - 12 * smootherstep(t / 8.0)
        set_camera(center, direction, dist, up=up, roll_deg=-4, left=0.28)
        anchors[i] = {"ss": screen_xy([mid])[0]}
        ray_frame("design", i)
    return anchors


# ----------------------------------------------------------------------------------------- pka
def pka():
    """Levodopa, the dominant microstate at pH 7.4 from Leonardo's calibrated PM7/COSMO pKa run."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    res = json.load(open(os.path.join(DATA, "pka_levodopa.json")))
    mol = Chem.AddHs(Chem.MolFromSmiles("[NH3+][C@@H](Cc1ccc(O)c(O)c1)C(=O)[O-]"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    cmd.read_molstr(Chem.MolToMolBlock(mol), "ldopa")
    cmd.hide("everything")
    cmd.show("sticks", "ldopa")
    cmd.show("spheres", "ldopa")
    cmd.set("sphere_scale", 0.23)
    cmd.set("sphere_scale", 0.14, "ldopa and elem H")
    cmd.set("stick_radius", 0.15)
    cmd.color("0xd9e2ee", "ldopa and elem C")
    cmd.color("0xff5a5f", "ldopa and elem O")
    cmd.color("0x6fa8ff", "ldopa and elem N")
    cmd.color("0xffffff", "ldopa and elem H")
    center = centroid("ldopa")
    sites = {}
    for s in res["result"]["sites"]:
        sites[s["class"] + str(s["atom"])] = (int(s["atom"]), float(s["pka"]))
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    n = nframes("pka")
    anchors = {}
    for i in range(n):
        t = i / FPS
        if not wanted("pka", i):
            continue
        direction = rotate(np.array([0.0, 0.25, 1.0]), (0, 1, 0), 18 * t)
        set_camera(center, direction, 27.0, depth=16, left=0.30)
        xyz = cmd.get_coords("ldopa")
        anchors[i] = {"atoms": {str(a.GetIdx()): screen_xy([xyz[a.GetIdx()]])[0] for a in heavy}}
        ray_frame("pka", i)
    anchors["sites"] = sites
    return anchors


# --------------------------------------------------------------------------------------- outro
def outro():
    """Slow, distant orbit of trypsin under the closing card (heavily dimmed in compositing)."""
    cmd.load(os.path.join(DATA, "md_trypsin_amber.pdb"), "md")
    cmd.remove("hydro")
    cmd.hide("everything")
    cmd.show("surface", "md")
    cmd.color("0x1d3d70", "md")
    center = centroid("md")
    n = nframes("outro")
    for i in range(n):
        t = i / FPS
        if not wanted("outro", i):
            continue
        set_camera(center, rotate(np.array([0.3, 0.2, 1.0]), (0, 1, 0), 12 * t), 120, depth=60)
        ray_frame("outro", i)
    return {}


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    scene = args[0]
    base_look()
    anchors = {"docking": docking, "md": md, "qmmm": qmmm, "design": design, "pka": pka, "outro": outro}[scene]()
    path = os.path.join(FRAMES, scene, "anchors.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({str(k): v for k, v in anchors.items()})
    json.dump(old, open(path, "w"))
    print(f"{scene}: done")


if __name__ == "pymol" or __name__ == "__main__":
    main(sys.argv[1:])
