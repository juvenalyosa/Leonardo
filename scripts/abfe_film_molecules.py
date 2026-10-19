"""
PyMOL stage of the 60 s Leonardo ABFE-lite film (YouTube 16:9 and Instagram 9:16): ray-traced shots of the
trajectories of a real Leonardo run (examples/films/11_abfe_lite, gathered by scripts/abfe_film_data.py):
benzamidine in trypsin's S1 pocket, one replica's guided exit (pulled out, held, returned), and the three
replicas together.  Every ligand position shown is a state Leonardo sampled.

    pymol -cq scripts/abfe_film_molecules.py -- <yt|ig> <site|exit|replicas> [--t=SECONDS]

Frames: artifacts/abfe_2026/frames/<format>/<scene>/mol_NNNN.png (resumable).
"""
import json
import math
import os
import sys

import numpy as np
from pymol import cmd

ROOT = os.environ.get("LEONARDO_REPO", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import abfe_film_data as AD  # noqa: E402  (timeline, colours)
import trailer_molecules as T  # noqa: E402  (camera, look and projection helpers)

DATA = os.path.join(ROOT, "artifacts", "abfe_2026", "data")
FPS = AD.FPS
SCENES = AD.SCENES
FORMATS = {"yt": (1920, 1080, 0.27, 0.0), "ig": (1080, 1920, 0.0, 0.2)}      # W, H, left, raise
POCKET = "resi 189+190+195+213+214+215+216+219+226"


def hexcol(rgb):
    return "0x%02x%02x%02x" % tuple(rgb)


def setup(fmt):
    w, h, left, up = FORMATS[fmt]
    T.W, T.H = w, h
    T.FRAMES = os.path.join(ROOT, "artifacts", "abfe_2026", "frames", fmt)
    return left, up


def nframes(scene):
    a, b = SCENES[scene]
    return int(round((b - a) * FPS))


def wanted(scene, i):
    for a in sys.argv:
        if a.startswith("--t="):
            return i == int(round(float(a[4:]) * FPS))
    return not os.path.exists(T.frame_path(scene, i))


def protein_look(obj):
    cmd.hide("everything", obj)
    cmd.show("cartoon", obj)
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("cartoon_oval_length", 1.25)
    cmd.set("cartoon_loop_radius", 0.3)
    cmd.set("cartoon_rect_length", 1.5)
    cmd.spectrum("resi", "0x1b3766 0x2e64b4 0x4fa3ff 0x3c7fd6 0x2e64b4 0x1b3766", f"{obj} and name CA")
    pocket = f"{obj} and {POCKET} and not name N+C+O and not elem H"
    cmd.show("sticks", pocket)
    cmd.set("stick_radius", 0.17, obj)
    cmd.color("0x9fb4d6", pocket + " and elem C")
    cmd.color("0xff6b6b", pocket + " and elem O")
    cmd.color("0x8ab4ff", pocket + " and elem N")
    cmd.set("stick_transparency", 0.25, obj)


def ligand_look(obj, rgb):
    cmd.hide("everything", obj)
    cmd.show("sticks", obj + " and not elem H")
    cmd.show("spheres", obj + " and not elem H")
    cmd.set("sphere_scale", 0.33, obj)
    cmd.set("stick_radius", 0.2, obj)
    cmd.color(hexcol(rgb), obj + " and elem C")
    cmd.color("0x8ab4ff", obj + " and elem N")


def load_data():
    z = np.load(os.path.join(DATA, "trajectories.npz"))
    lig = [z[f"lig_{k}"] for k in range(3) if f"lig_{k}" in z.files]
    return z["xyz_0"], z["lig_idx"], lig


def scene_geometry(xyz0, lig_idx, lig):
    """Camera centre, the exit direction and the view direction (side-on to the exit)."""
    prot = np.delete(xyz0[0], lig_idx, axis=0)
    site = lig[0][0].mean(0)
    far = [l[int(np.argmax(np.linalg.norm(l.mean(1) - site, axis=1)))].mean(0) for l in lig]
    exit_dir = T.unit(np.mean([f - site for f in far], axis=0))
    side = T.unit(np.cross(exit_dir, (0.0, 0.0, 1.0)))
    if np.linalg.norm(np.cross(exit_dir, (0.0, 0.0, 1.0))) < 0.2:
        side = T.unit(np.cross(exit_dir, (0.0, 1.0, 0.0)))
    return prot.mean(0), site, exit_dir, side


def clearest_view(site, exit_dir, side, prot):
    """Direction (about the exit axis, tilted a little against the exit) with the fewest protein atoms within
    4 Å of the sight line from the pocket to 30 Å out; the swept angles stay clear for the camera moves."""
    def blockers(d):
        rel = prot - site
        along = rel @ d
        near = (along > 2.0) & (along < 30.0)
        perp = np.linalg.norm(rel[near] - np.outer(along[near], d), axis=1)
        return int(np.sum(perp < 4.0))
    best = None
    for deg in range(0, 360, 10):
        d = T.unit(T.rotate(side, exit_dir, deg) * 0.95 - exit_dir * 0.3)
        score = sum(blockers(T.unit(T.rotate(d, exit_dir, off))) for off in (-20, 0, 20))
        if best is None or score < best[0]:
            best = (score, d)
    return best[1]


def zoom(radius):
    fov = math.radians(float(cmd.get("field_of_view")))
    narrow = min(1.0, T.W / T.H)
    return radius / (math.tan(fov / 2) * narrow)


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    fmt, scene = args[0], args[1]
    left, up = setup(fmt)
    T.base_look()
    cmd.set("fog_start", 0.5)
    xyz0, lig_idx, lig = load_data()
    cmd.load(os.path.join(DATA, "complex.pdb"), "cx")
    cmd.remove("cx and resn HOH")
    # the trajectory object carries no secondary structure and its first MD state is too loose for DSS:
    # the cartoon takes the assignment of the crystal structure (trypsin as loaded by the pack)
    cmd.load(os.path.join(DATA, "trypsin_full.pdb"), "xtal")
    cmd.dss("xtal")
    ss = {}
    cmd.iterate("xtal and name CA", "ss_[(chain, resi)] = ss", space={"ss_": ss})
    cmd.alter("cx", "ss = ss_.get((chain, resi), 'L')", space={"ss_": ss})
    cmd.delete("xtal")
    protein_look("cx")
    cmd.create("lig0", "cx and resn BEN")
    cmd.hide("everything", "cx and resn BEN")
    ligand_look("lig0", AD.REPLICA_COLORS[0])
    centre, site, exit_dir, side = scene_geometry(xyz0, lig_idx, lig)
    n_states = len(lig[0])
    lig_order = [i for i in range(xyz0.shape[1]) if i in set(int(v) for v in lig_idx)]
    radius = float(np.max(np.linalg.norm(np.delete(xyz0[0], lig_idx, axis=0) - centre, axis=1)))
    # side-on to the exit path, turned about it to the angle with the clearest line of sight into the pocket
    prot0 = np.delete(xyz0[0], lig_idx, axis=0)
    view_dir = clearest_view(site, exit_dir, side, prot0)
    anchors = {}
    lift = up if T.H > T.W else 0.0
    if scene == "site":
        for i in range(nframes(scene)):
            t = i / FPS
            if not wanted(scene, i):
                continue
            k = T.smootherstep(t / 9.5)
            dirn = T.rotate(view_dir, exit_dir, -70 + 60 * k)
            target = centre * (1 - 0.55 * k) + site * 0.55 * k
            T.set_camera(target, dirn, zoom(radius * (1.25 - 0.55 * k)), left=left * T.smootherstep((t - 5) / 3),
                         raise_=lift * T.smootherstep((t - 5) / 3))
            anchors[i] = {"ligand": T.screen_xy([site])[0]}
            T.ray_frame(scene, i)
    elif scene == "exit":
        cmd.create("start", "lig0")
        ligand_look("start", (200, 214, 235))
        cmd.color("0xc8d6eb", "start")
        cmd.set("stick_transparency", 0.45, "start")
        cmd.set("sphere_transparency", 0.45, "start")
        n = nframes(scene)
        mid = site + exit_dir * 0.45 * float(np.max(np.linalg.norm(lig[0].mean(1) - site, axis=1)))
        last = int(round(AD.EXIT_FRACTION * (n_states - 1)))
        for i in range(n):
            t = i / FPS
            s = min(last, int(round(i * last / (n - 1))))
            if not wanted(scene, i):
                continue
            prot = xyz0[s]
            cmd.load_coords(prot.tolist(), "cx")
            cmd.load_coords(prot[lig_order].tolist(), "lig0")
            dirn = T.rotate(view_dir, exit_dir, -10 + 20 * t / (n / FPS))
            target = site * 0.45 + mid * 0.55
            # 16:9: the path runs upward, so the scene sits low, clear of the title block at the top left
            T.set_camera(target, dirn, zoom(radius * 1.5), left=left, raise_=lift if T.H > T.W else -0.07)
            com = prot[lig_order].mean(0)
            anchors[i] = {"state": s, "ligand": T.screen_xy([com])[0], "site": T.screen_xy([site])[0]}
            T.ray_frame(scene, i)
    elif scene == "replicas":
        for k in range(1, len(lig)):
            cmd.create(f"lig{k}", "lig0")
            ligand_look(f"lig{k}", AD.REPLICA_COLORS[k])
        n = nframes(scene)
        for i in range(n):
            t = i / FPS
            s = min(n_states - 1, int(round(i * (n_states - 1) / (n - 1))))
            if not wanted(scene, i):
                continue
            cmd.load_coords(xyz0[s].tolist(), "cx")
            for k in range(len(lig)):
                cmd.load_coords(lig[k][min(s, len(lig[k]) - 1)].tolist(), f"lig{k}")
            dirn = T.rotate(view_dir, exit_dir, 10 + 25 * t / (n / FPS))
            T.set_camera(site * 0.6 + centre * 0.4, dirn, zoom(radius * 0.95), left=left, raise_=lift)
            anchors[i] = {"state": s}
            T.ray_frame(scene, i)
    path = os.path.join(T.FRAMES, scene, "anchors.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({str(k): v for k, v in anchors.items()})
    json.dump(old, open(path, "w"))
    print(f"{fmt}/{scene}: done")


if __name__ == "pymol" or __name__ == "__main__":
    main(sys.argv[1:])
