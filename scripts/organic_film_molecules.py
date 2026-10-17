"""
PyMOL stage of the 60 s Leonardo organic chemistry & catalysis film (YouTube 16:9 and Instagram 9:16):
ray-traced shots of the structures of a real Leonardo run (examples/films/10_organic_catalysis, gathered
by scripts/organic_film_data.py): aspirin vibrating along two of its computed normal modes, and the
acetaldehyde -> vinyl alcohol tautomerization through its uncatalysed and water-relayed transition states.
The reaction shots move the atoms between the computed stationary points (keto minimum, TS, enol minimum).

    pymol -cq scripts/organic_film_molecules.py -- <yt|ig> <aspirin|uncat|water> [--t=SECONDS]

Frames: artifacts/organic_2026/frames/<format>/<scene>/mol_NNNN.png (resumable).
"""
import json
import math
import os
import sys

import numpy as np
from pymol import cmd

ROOT = os.environ.get("LEONARDO_REPO", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import organic_film_data as OD  # noqa: E402  (timeline, shown modes, reaction path)
import trailer_molecules as T  # noqa: E402  (camera, look and projection helpers)

DATA = os.path.join(ROOT, "artifacts", "organic_2026", "data")
FPS = OD.FPS
SCENES = OD.SCENES
FORMATS = {"yt": (1920, 1080, 0.27, 0.0), "ig": (1080, 1920, 0.0, 0.17)}      # W, H, left, raise
C_COL, O_COL, H_COL, HOT = "0xd9e2ee", "0xff5a5f", "0xf4f7fb", "0x2be3c8"


def setup(fmt):
    w, h, left, up = FORMATS[fmt]
    T.W, T.H = w, h
    T.FRAMES = os.path.join(ROOT, "artifacts", "organic_2026", "frames", fmt)
    return left, up


def nframes(scene):
    a, b = SCENES[scene]
    return int(round((b - a) * FPS))


def wanted(scene, i):
    for a in sys.argv:
        if a.startswith("--t="):
            return i == int(round(float(a[4:]) * FPS))
    return not os.path.exists(T.frame_path(scene, i))


def zoom_for(radius):
    fov = math.radians(float(cmd.get("field_of_view")))
    narrow = min(1.0, T.W / T.H)
    phone = 0.72 if T.H > T.W else 1.0          # the phone frame is narrow: the molecule fills more of it
    return radius / (math.tan(fov / 2) * narrow) * 1.15 * phone


def read_xyz(name):
    rows = open(os.path.join(DATA, name + ".xyz")).read().splitlines()
    n = int(rows[0].split()[0])
    sym = [r.split()[0] for r in rows[2:2 + n]]
    xyz = np.array([[float(v) for v in r.split()[1:4]] for r in rows[2:2 + n]])
    return sym, xyz


def kabsch(mobile, ref):
    """mobile rigidly superposed on ref (same atom order)."""
    a, b = mobile - mobile.mean(0), ref - ref.mean(0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return a @ r + ref.mean(0)


def molecule_look(obj):
    cmd.hide("everything")
    cmd.show("sticks", obj)
    cmd.show("spheres", obj)
    cmd.set("valence", 0)
    cmd.set("stick_radius", 0.15)
    cmd.set("sphere_scale", 0.27, obj)
    cmd.set("sphere_scale", 0.19, obj + " and elem H")
    cmd.color(C_COL, obj + " and elem C")
    cmd.color(O_COL, obj + " and elem O")
    cmd.color(H_COL, obj + " and elem H")
    cmd.set("specular", 0.55)
    cmd.set("spec_power", 70)


def frames_loop(scene, camera, per_frame=None):
    anchors = {}
    for i in range(nframes(scene)):
        t = i / FPS
        if not wanted(scene, i):
            continue
        anchors[i] = camera(t) or {}
        if per_frame:                  # after the camera: anchors are projected with this frame's view
            anchors[i].update(per_frame(t, i) or {})
        T.ray_frame(scene, i)
    return anchors


# --------------------------------------------------------------------------------------- aspirin
def aspirin_shot(left, up):
    sym, eq = read_xyz("aspirin")
    cmd.load(os.path.join(DATA, "aspirin.xyz"), "m")
    molecule_look("m")
    d = np.load(os.path.join(DATA, "aspirin_modes.npz"), allow_pickle=True)
    modes = OD.shown_modes(os.path.join(DATA, "aspirin_modes.npz"))
    vecs = []
    for idx, _f, _lab in modes:
        v = np.asarray(d["mode_cart_ang"][idx], float)
        vecs.append(v / float(np.max(np.linalg.norm(v, axis=1))))
    c = eq.mean(0)
    radius = float(np.max(np.linalg.norm(eq - c, axis=1)))
    dist = zoom_for(radius * 1.3)
    # the plane of the ring faces the camera, tilted
    heavy = eq[[i for i, s in enumerate(sym) if s != "H"]]
    normal = np.linalg.svd(heavy - heavy.mean(0))[2][2]
    a0 = SCENES["aspirin"][0]

    def per_frame(t, _i):
        k = 0
        if t >= 7.2:
            k = 1 if t < OD.IR_SWITCH - a0 else 2
        xyz = eq.copy()
        hot = []
        if k and k <= len(vecs):
            ts = t - (7.2 if k == 1 else OD.IR_SWITCH - a0)
            amp = 0.30 * T.smootherstep(ts / 1.0) * (1 - T.smootherstep((ts - 5.6) / 0.6))
            xyz = eq + vecs[k - 1] * amp * math.sin(2 * math.pi * 1.6 * ts)
            hot = [j for j in range(len(sym)) if np.linalg.norm(vecs[k - 1][j]) > 0.35]
        cmd.load_coords(xyz.tolist(), "m")
        cmd.color(C_COL, "m and elem C")
        cmd.color(O_COL, "m and elem O")
        cmd.color(H_COL, "m and elem H")
        if hot:
            cmd.color(HOT, "m and index " + "+".join(str(j + 1) for j in hot))
        return {"mode": k, "hot": T.screen_xy([eq[hot].mean(0)])[0] if hot else None}

    def camera(t):
        dirn = T.rotate(T.rotate(normal, (0, 1, 0), -22 + 2.4 * t), (1, 0, 0), 16 * math.sin(t * 0.3))
        k = T.smootherstep((t - 4.6) / 2.4)        # far under the title, then forward to make room for the panel
        lift = up if T.H > T.W else -0.14             # 16:9: a little low, clear of the title block
        T.set_camera(c, dirn, dist * (1.75 - 0.6 * k), left=left * k, raise_=lift * k)
        return {}

    return frames_loop("aspirin", camera, per_frame)


# ------------------------------------------------------------------------------------- reactions
def reaction_shot(scene, left, up):
    names = ("keto", "ts", "enol") if scene == "uncat" else ("keto_w", "ts_w", "enol_w")
    sym, ts = read_xyz(names[1])
    keto = kabsch(read_xyz(names[0])[1], ts)
    enol = kabsch(read_xyz(names[2])[1], ts)
    path = OD.MOVES[scene]
    cmd.load(os.path.join(DATA, names[1] + ".xyz"), "m")
    molecule_look("m")
    for h, _a, _b in path:
        cmd.color(HOT, f"m and index {h + 1}")
        cmd.set("sphere_scale", 0.23, f"m and index {h + 1}")
    cmd.set("dash_color", HOT)
    cmd.set("dash_radius", 0.06)
    cmd.set("dash_gap", 0.22)
    cmd.set("dash_length", 0.18)
    if scene == "water":
        cmd.set("sphere_scale", 0.3, "m and index 5")        # the catalyst oxygen
    # imaginary mode of the TS: a small shiver while the system sits on the saddle
    md = np.load(os.path.join(DATA, names[1] + "_modes.npz"), allow_pickle=True)
    im = int(np.argmax(md["freq_cm_imag"]))
    mode = np.asarray(md["mode_cart_ang"][im], float)
    mode = mode / float(np.max(np.linalg.norm(mode, axis=1)))
    c = ts.mean(0)
    radius = float(np.max(np.linalg.norm(ts - c, axis=1)))
    dist = zoom_for(radius * (1.7 if scene == "uncat" else 1.45))
    # look straight at the ring the protons travel round (4 atoms uncatalysed, 6 with the water relay)
    ring = sorted({0, 1, 2} | {j for move in path for j in move})
    pts = ts[ring] - ts[ring].mean(0)
    normal = np.linalg.svd(pts)[2][2]
    c = ts[ring].mean(0) * 0.6 + c * 0.4

    def geometry(u):
        s = OD.path_s(u)
        if s <= 1.0:
            xyz = keto + (ts - keto) * s
        else:
            xyz = ts + (enol - ts) * (s - 1.0)
        if 7.4 <= u < 9.0:
            w = math.sin(math.pi * (u - 7.4) / 1.6)
            xyz = xyz + mode * 0.12 * w * math.sin(2 * math.pi * 2.0 * (u - 7.4))
        return s, xyz

    def per_frame(t):
        s, xyz = geometry(t)
        cmd.load_coords(xyz.tolist(), "m")
        for h, a, b in path:
            for q in (a, b):
                name = f"pb_{h}_{q}"
                cmd.delete(name)
                cmd.unbond(f"m and index {h + 1}", f"m and index {q + 1}")
                dd = float(np.linalg.norm(xyz[h] - xyz[q]))
                if dd < 1.22:              # a bond
                    cmd.bond(f"m and index {h + 1}", f"m and index {q + 1}")
                elif dd < 2.05:            # a bond being made or broken: dashed
                    cmd.distance(name, f"m and index {h + 1}", f"m and index {q + 1}")
                    cmd.hide("labels", name)
        return {"s": s, "h": T.screen_xy([xyz[path[0][0]]])[0]}

    def camera(t):
        dirn = T.rotate(T.rotate(normal, (0, 1, 0), -12 + 2.0 * t), (1, 0, 0), 10)
        lift = 0.7 * up if T.H > T.W else -0.12       # clear of the title block (long subtitles on the phone)
        T.set_camera(c, dirn, dist * (1.05 - 0.008 * t), left=left, raise_=lift)

    anchors = {}
    for i in range(nframes(scene)):
        t = i / FPS
        if not wanted(scene, i):
            continue
        camera(t)                      # projection of the moving proton uses this frame's camera
        anchors[i] = per_frame(t)
        T.ray_frame(scene, i)
    return anchors


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    fmt, scene = args[0], args[1]
    left, up = setup(fmt)
    T.base_look()
    cmd.set("fog_start", 0.55)
    if scene == "aspirin":
        anchors = aspirin_shot(left, up)
    else:
        anchors = reaction_shot(scene, left, up)
    path = os.path.join(T.FRAMES, scene, "anchors.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update({str(k): v for k, v in anchors.items()})
    json.dump(old, open(path, "w"))
    print(f"{fmt}/{scene}: done")


if __name__ == "pymol" or __name__ == "__main__":
    main(sys.argv[1:])
