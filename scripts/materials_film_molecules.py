"""
PyMOL stage of the 60 s Leonardo materials film (YouTube 16:9 and Instagram 9:16): ray-traced shots of
the structures and results of a real Leonardo run (artifacts/materials_2026/pack, replayed with
``python -m leonardo.replay``): built materials, lysozyme on three surfaces, the QM cluster and its MOPAC
PM7 frontier orbitals.

    pymol -cq scripts/materials_film_molecules.py -- <yt|ig> <scene> [--t=SECONDS]

Frames: artifacts/materials_2026/frames/<format>/<scene>/mol_NNNN.png (resumable).
"""
import json
import math
import os
import sys

import numpy as np
from pymol import cmd

ROOT = os.environ.get("LEONARDO_REPO", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import trailer_molecules as T  # noqa: E402  (camera, look and projection helpers)

DATA = os.path.join(ROOT, "artifacts", "materials_2026", "data")
FPS = 24
SCENES = {"np": (0.0, 8.6), "pt": (8.6, 10.8), "graphene": (10.8, 13.0), "silica": (13.0, 15.2),
          "rutile": (15.2, 17.2), "au": (17.2, 24.0), "gr": (24.0, 30.0), "si": (30.0, 36.4), "qm": (36.4, 51.0)}
FORMATS = {"yt": (1920, 1080, 0.27, 0.0), "ig": (1080, 1920, 0.0, 0.30)}      # W, H, left, raise


def setup(fmt):
    w, h, left, up = FORMATS[fmt]
    T.W, T.H = w, h
    T.FRAMES = os.path.join(ROOT, "artifacts", "materials_2026", "frames", fmt)
    return left, up


def nframes(scene):
    a, b = SCENES[scene]
    return int(round((b - a) * FPS))


def wanted(scene, i):
    for a in sys.argv:
        if a.startswith("--t="):
            return i == int(round(float(a[4:]) * FPS))
    return not os.path.exists(T.frame_path(scene, i))


def zoom_for(radius, fmt):
    """Camera distance that fits a sphere of `radius` A in the narrow side of the frame."""
    fov = math.radians(float(cmd.get("field_of_view")))
    narrow = min(1.0, T.W / T.H)              # vertical frames are narrow
    return radius / (math.tan(fov / 2) * narrow) * 1.15


def metal_look(sel, color, scale=0.98):
    cmd.show("spheres", sel)
    cmd.set("sphere_scale", scale, sel)
    cmd.color(color, sel)


def frames_loop(scene, camera, per_frame=None):
    anchors = {}
    for i in range(nframes(scene)):
        t = i / FPS
        if per_frame:
            per_frame(t, i)
        if not wanted(scene, i):
            continue
        anchors[i] = camera(t) or {}
        T.ray_frame(scene, i)
    return anchors


# ------------------------------------------------------------------------------------------ build
def material_shot(scene, fmt, left, up):
    files = {"np": "Au309.pdb", "pt": "Pt111.pdb", "graphene": "graphene.pdb", "silica": "silica.pdb",
             "rutile": "rutile.pdb"}
    cmd.load(os.path.join(DATA, files[scene]), "m")
    cmd.hide("everything")
    if scene == "np":
        metal_look("m", "0xd4a537")
        cmd.set("specular", 0.9)
        cmd.set("spec_power", 40)
        cmd.set("reflect", 0.55)
    elif scene == "pt":
        metal_look("m", "0xc9ced6")
        cmd.set("specular", 0.8)
        cmd.set("spec_power", 50)
    elif scene == "graphene":
        cmd.show("sticks", "m")
        cmd.show("spheres", "m")
        cmd.set("sphere_scale", 0.24)
        cmd.set("stick_radius", 0.16)
        cmd.color("0x9aa6ba", "m")
        cmd.set("specular", 0.7)
    elif scene == "silica":
        cmd.show("sticks", "m")
        cmd.show("spheres", "m")
        cmd.set("sphere_scale", 0.26)
        cmd.set("sphere_scale", 0.17, "m and elem H")
        cmd.set("stick_radius", 0.14)
        cmd.color("0xf2d16b", "m and elem Si")
        cmd.color("0xff5a5f", "m and elem O")
        cmd.color("0xffffff", "m and elem H")
    else:
        cmd.show("spheres", "m")
        cmd.set("sphere_scale", 0.55, "m and elem Ti")
        cmd.set("sphere_scale", 0.42, "m and elem O")
        cmd.color("0x9fb4c8", "m and elem Ti")
        cmd.color("0xff5a5f", "m and elem O")
        cmd.set("specular", 0.5)
    xyz = cmd.get_coords("m")
    c = xyz.mean(axis=0)
    radius = float(np.max(np.linalg.norm(xyz - c, axis=1)))
    if scene == "np":
        dist = zoom_for(radius, fmt)

        def camera(t):
            d = T.rotate(T.rotate(np.array([0.2, 0.3, 1.0]), (0, 1, 0), 22 * t), (1, 0, 0), 6 * math.sin(t * 0.4))
            k = T.smootherstep((t - 6.0) / 2.6)          # under the title it is far, then it comes forward
            T.set_camera(c, d, dist * (1.65 - 0.45 * k), left=left * k, raise_=up * k)
    else:
        # slabs: a low, raking angle over the surface, the camera gliding across it
        span = float(np.max(xyz[:, 0]) - np.min(xyz[:, 0]))
        top = float(np.max(xyz[:, 2]))
        target = np.array([c[0], c[1], top - 1.0])
        dist = zoom_for(span * 0.62, fmt)

        def camera(t):
            d = T.rotate(np.array([0.0, -1.0, 0.55]), (0, 0, 1), 18 + 14 * t)
            T.set_camera(target, d, dist * (1.08 - 0.06 * t), up=(0, 0, 1), left=left, raise_=up)
    return frames_loop(scene, camera)


# ------------------------------------------------------------------------------------- surfaces
SURFACE_FILES = {"au": ("lyso_on_Au.pdb", "MAU"), "gr": ("lyso_on_graphene.pdb", "GRC"),
                 "si": ("lyso_on_SiO2.pdb", "QSI+QOB+QOH")}


def surface_shot(scene, fmt, left, up, contacts):
    path, resn = SURFACE_FILES[scene]
    cmd.load(os.path.join(DATA, path), "sys")
    cmd.remove("sys and (resn HOH+NA+CL or hydro) and not resn QOH")
    cmd.hide("everything")
    slab = f"sys and resn {resn}"
    prot = "sys and polymer.protein"
    if scene == "au":
        metal_look(slab, "0xd4a537")
        cmd.set("specular", 0.85)
    elif scene == "gr":
        cmd.show("sticks", slab)
        cmd.show("spheres", slab)
        cmd.set("sphere_scale", 0.22, slab)
        cmd.set("stick_radius", 0.13, slab)
        cmd.color("0x9aa6ba", slab)
    else:
        cmd.show("sticks", slab)
        cmd.show("spheres", slab)
        cmd.set("sphere_scale", 0.26, slab)
        cmd.set("stick_radius", 0.13, slab)
        cmd.color("0xe8c75a", slab + " and elem Si")
        cmd.color("0xff6b6b", slab + " and elem O")
        cmd.color("0xffffff", slab + " and elem H")
    cmd.show("cartoon", prot)
    cmd.set("cartoon_transparency", 0.08)
    cmd.spectrum("resi", "0x1b3766 0x2e64b4 0x4fa3ff 0x2be3c8 0x4fa3ff 0x2e64b4", prot + " and name CA")
    sel = []
    for r in contacts:          # e.g. "ARG68A"
        num = "".join(ch for ch in r if ch.isdigit())
        sel.append(num)
    if sel:
        touch = f"{prot} and resi {'+'.join(sel)} and not name N+C+O"
        cmd.show("sticks", touch)
        cmd.set("stick_radius", 0.28, touch)
        cmd.color("0xffb347", touch + " and elem C")
        cmd.color("0x8ab4ff", touch + " and elem N")
        cmd.color("0xff6b6b", touch + " and elem O")
    pc = T.centroid(prot)
    top = float(np.max(cmd.get_coords(slab)[:, 2]))
    target = np.array([pc[0], pc[1], top + 0.78 * (pc[2] - top)])      # protein low in the frame: titles on top
    xyz = cmd.get_coords(prot)
    radius = float(np.max(np.linalg.norm(xyz - pc, axis=1)))
    dist = zoom_for(radius * 1.3, fmt)

    def camera(t):
        d = T.rotate(np.array([0.0, -1.0, 0.22]), (0, 0, 1), 25 + 9 * t)
        T.set_camera(target, d, dist * (1.06 - 0.025 * t), up=(0, 0, 1), left=left, raise_=up)
        return {"touch": T.screen_xy([T.centroid(touch)])[0] if sel else None}

    return frames_loop(scene, camera)


# ------------------------------------------------------------------------------------------- qm
def qm_shot(fmt, left, up, homo_cube, lumo_cube, switch_s):
    cmd.load(os.path.join(DATA, "qm_cluster.xyz"), "qm")
    cmd.hide("everything")
    cmd.show("sticks", "qm and not elem Au")
    cmd.show("spheres", "qm")
    cmd.set("sphere_scale", 0.22, "qm and not elem Au")
    cmd.set("sphere_scale", 0.15, "qm and elem H")
    cmd.set("sphere_scale", 0.42, "qm and elem Au")
    cmd.set("stick_radius", 0.13)
    cmd.color("0xd4a537", "qm and elem Au")
    cmd.color("0xd9e2ee", "qm and elem C")
    cmd.color("0x6fa8ff", "qm and elem N")
    cmd.color("0xff5a5f", "qm and elem O")
    cmd.color("0xffffff", "qm and elem H")
    for name, cube in (("homo", homo_cube), ("lumo", lumo_cube)):
        cmd.load(os.path.join(DATA, cube), f"{name}_map")
        cmd.isosurface(f"{name}_p", f"{name}_map", 0.03)
        cmd.isosurface(f"{name}_n", f"{name}_map", -0.03)
        cmd.color("0x2be3c8", f"{name}_p")
        cmd.color("0xff8c42", f"{name}_n")
        cmd.set("transparency", 0.28, f"{name}_p")
        cmd.set("transparency", 0.28, f"{name}_n")
        cmd.disable(f"{name}_map")
    xyz = cmd.get_coords("qm")
    c = xyz.mean(axis=0)
    au = cmd.get_coords("qm and elem Au").mean(axis=0)
    radius = float(np.max(np.linalg.norm(xyz - c, axis=1)))
    dist = zoom_for(radius * 1.05, fmt)
    side = T.unit(c - au)
    c = c * 0.75 + au * 0.25 + side * 3.0          # cluster low in the frame: titles on top                     # look across the interface: gold below, residues above

    def per_frame(t, _i):
        lumo = t >= switch_s
        for nm, on in (("homo", not lumo), ("lumo", lumo)):
            for s in ("_p", "_n"):
                (cmd.enable if on else cmd.disable)(nm + s)

    def camera(t):
        axis = T.unit(np.cross(side, [0.0, 0.0, 1.0]) if abs(side[2]) < 0.9 else np.cross(side, [1.0, 0.0, 0.0]))
        d = T.rotate(T.unit(np.cross(axis, side)), side, 16 * t)
        T.set_camera(c, d, dist * (1.08 - 0.012 * t), up=side, left=left, raise_=up, depth=40)
        return {}

    return frames_loop("qm", camera, per_frame)


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    fmt, scene = args[0], args[1]
    left, up = setup(fmt)
    T.base_look()
    res = json.load(open(os.path.join(DATA, "results.json")))
    if scene in ("np", "pt", "graphene", "silica", "rutile"):
        anchors = material_shot(scene, fmt, left, up)
    elif scene in SURFACE_FILES:
        step = {"au": "7", "gr": "8", "si": "9"}[scene]
        contacts = list(res[step]["surface_analysis"]["contact_residues"])
        anchors = surface_shot(scene, fmt, left, up, contacts)
    else:
        homo = f"mo_{res['11']['homo_index']}.cube"
        lumo = f"mo_{res['11']['homo_index'] + 1}.cube"
        anchors = qm_shot(fmt, left, up, homo, lumo, switch_s=(SCENES["qm"][1] - SCENES["qm"][0]) / 2)
    path = os.path.join(T.FRAMES, scene, "anchors.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({str(k): v for k, v in anchors.items()})
    json.dump(old, open(path, "w"))
    print(f"{fmt}/{scene}: done")


if __name__ == "pymol" or __name__ == "__main__":
    main(sys.argv[1:])
