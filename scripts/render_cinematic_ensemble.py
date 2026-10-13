#!/usr/bin/env python3
"""Molecular film 08: ensemble redocking (P450cam + camphor), white-stage cinematic_v2 design, ~66 s.

Every number on screen comes from artifacts/mc_redock_panel (Leonardo's redocking with a Monte Carlo receptor
ensemble); the poses shown are re-checked against the reported RMSDs before rendering. Only the side chains that
line the camphor are drawn, and the crystal pose is never drawn on top of a docked pose: the two are compared by a
cross-dissolve of renders taken from the same camera.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/leonardo-ensemble-mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymol2  # noqa: E402
import render_cinematic_v2 as style  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "artifacts/mc_redock_panel"
CASE = PANEL / "p450cam_2CPP"
XTAL = Path(os.environ.get("LEONARDO_2CPP", PANEL / "2CPP.pdb"))
OUT = ROOT / "artifacts/cinematic_v2"
FPS = 24
W, H = 1920, 1080
MW, MH = 1600, 900
TEAL, GOLD, INK, MUTED, LINE, RED = style.TEAL, style.GOLD, style.INK, style.MUTED, style.LINE, style.RED
txt = style.txt
MEMBER_FILES = ["best_model.pdb", "mc_001.pdb", "mc_002.pdb", "mc_003.pdb"]
MEMBER_NAMES = ["STARTING STRUCTURE", "MONTE CARLO CONFORMATION 1", "MONTE CARLO CONFORMATION 2",
                "MONTE CARLO CONFORMATION 3"]
# side chains drawn: the camphor contacts that tell the story (Tyr96 H-bonds the carbonyl; Thr101 and Asp297
# are the residues the Monte Carlo moves most); the other contacts stay as ribbon
SHOWN = "87+96+101+244+247+295+297"
MOVERS = "101+297"

# timeline (seconds)
T_TITLE, T_DIVE, T_CONF, T_DOCK, T_CHECK, T_PANEL, T_END = 0.0, 8.0, 16.0, 31.0, 47.0, 55.0, 62.0
DURATION = 66.0
FRAMES = int(DURATION * FPS)
CONF_SPAN = (T_DOCK - T_CONF) / 4
DOCK_SPAN = (T_CHECK - T_DOCK) / 4

PANEL_CASES = [  # (folder, label, note)
    ("trypsin_3PTB", "Trypsin · benzamidine", "Ca²⁺ kept"),
    ("p450cam_2CPP", "P450cam · camphor", "heme kept"),
    ("hiv_protease_1HVR", "HIV protease · XK263", "in-chain CSO repaired"),
    ("abl_imatinib_1IEP", "Abl kinase · imatinib", "37 heavy atoms"),
    ("mse_3F4I", "3F4I · β-octylglucoside", "free MSE kept"),
    ("carbonic_anhydrase_3HS4", "Carbonic anhydrase · AZM", "Zn²⁺: Vina has no metal term"),
]


def heavy_xyz_from_sdf(path, index=0):
    blocks = open(path).read().split("$$$$")
    lines = blocks[index].strip("\n").splitlines()
    n = int(lines[3][:3])
    return np.array([[float(x) for x in ln.split()[:3]] for ln in lines[4:4 + n] if ln.split()[3] != "H"])


def load():
    summary = json.loads((CASE / "summary.json").read_text())
    xtal = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in open(XTAL)
                     if ln.startswith("HETATM") and ln[17:20] == "CAM"])
    poses = []
    for k in range(1, 5):
        folder = next(p for p in sorted((CASE / "docking").iterdir()) if f"mc_{k:03d}" in p.name)
        xyz = heavy_xyz_from_sdf(folder / "docked_poses_view.sdf", 0)
        d = np.linalg.norm(xyz[:, None, :] - xtal[None, :, :], axis=2)
        nearest = float(np.sqrt((d.min(axis=1) ** 2).mean()))
        reported = float(summary["poses"][k - 1]["rmsd_A"])
        assert nearest <= reported + 0.05, (k, nearest, reported)
        poses.append({"file": folder / "docked_poses_view.sdf", "rmsd": reported, "xyz": xyz,
                      "score": float(summary["poses"][k - 1]["score_kcal_mol"])})
    panel = []
    for folder, label, note in PANEL_CASES:
        if not (PANEL / folder / "summary.json").exists() and os.environ.get("LEONARDO_PREVIEW"):
            continue
        s = json.loads((PANEL / folder / "summary.json").read_text())
        r = [float(p["rmsd_A"]) for p in s["poses"]]
        panel.append({"label": label, "note": note, "rmsd": r, "ok": sum(x <= 2.0 for x in r),
                      "checks": sum(c["ok"] for c in s["checks"]), "n_checks": len(s["checks"])})
    return {"summary": summary, "poses": poses, "members": summary["members"], "xtal": xtal, "panel": panel,
            "acceptance": summary["ensemble"]["acceptance"], "site": summary["ensemble"]["site_residues"]}


def setup(c, data):
    c.reinitialize()
    c.set("retain_order", 1)
    c.set("pdb_retain_ids", 1)
    c.set("max_threads", 6)
    c.bg_color("white")
    for key, value in {"orthoscopic": 1, "ray_opaque_background": 0, "ray_shadows": 1, "antialias": 1,
                       "depth_cue": 0, "specular": 0.42, "shininess": 45, "ambient": 0.40, "direct": 0.67,
                       "reflect": 0.12, "cartoon_fancy_helices": 1, "cartoon_smooth_loops": 1,
                       "cartoon_sampling": 14, "cartoon_flat_sheets": 1, "cartoon_highlight_color": -1,
                       "surface_quality": 1, "sphere_quality": 2, "stick_quality": 16, "transparency_mode": 2,
                       "ray_transparency_oblique": 0.2, "ray_trace_mode": 0, "auto_zoom": 0}.items():
        c.set(key, value)
    for name, rgb in {"teal": [0.20, 0.51, 0.55], "slate": [0.55, 0.65, 0.70], "gold": [0.84, 0.60, 0.24],
                      "nitrogen": [0.24, 0.45, 0.73], "oxygen": [0.78, 0.28, 0.25], "iron": [0.62, 0.33, 0.22],
                      "hemec": [0.46, 0.36, 0.30], "mover": [0.16, 0.56, 0.60]}.items():
        c.set_color(name, rgb)
    c.load(str(CASE / "ensemble" / MEMBER_FILES[0]), "enzyme")
    c.remove("enzyme and (resn HOH or hydro)")
    c.hide("everything")
    c.create("ribbon", "enzyme and polymer")
    c.show("cartoon", "ribbon")
    c.color("slate", "ribbon")
    c.color("teal", "ribbon and resi 87-101+244-252+295-297+395-396")
    c.create("envelope", "enzyme and polymer and not hydro")
    c.show("surface", "envelope")
    c.color("slate", "envelope")
    c.create("heme", "enzyme and resn HEM")
    c.show("sticks", "heme")
    c.show("spheres", "heme and elem Fe")
    c.color("hemec", "heme and elem C")
    c.color("nitrogen", "heme and elem N")
    c.color("oxygen", "heme and elem O")
    c.color("iron", "heme and elem Fe")
    c.set("stick_radius", 0.14, "heme")
    c.set("sphere_scale", 0.45, "heme and elem Fe")
    c.delete("enzyme")
    # binding-site side chains: one object, coordinates swapped/interpolated between members
    site_sel = f"polymer and resi {SHOWN} and not name N+C+O and not hydro"
    coords, names = [], None
    for k, f in enumerate(MEMBER_FILES):
        c.load(str(CASE / "ensemble" / f), "tmp")
        c.remove("tmp and hydro")
        atoms = [(a.resi, a.name) for a in c.get_model(f"tmp and {site_sel}").atom]
        assert names is None or atoms == names, "site atoms differ between members"
        names = atoms
        coords.append(c.get_coords(f"tmp and {site_sel}"))
        if k == 0:
            c.create("site", f"tmp and {site_sel}")
        c.delete("tmp")
    c.hide("everything", "site")
    c.show("sticks", "site")
    c.color("slate", "site and elem C")
    c.color("mover", f"site and elem C and resi {MOVERS}")
    c.color("oxygen", "site and elem O")
    c.color("nitrogen", "site and elem N")
    c.set("stick_radius", 0.15, "site")
    c.set("stick_radius", 0.19, f"site and resi {MOVERS}")
    # docked camphor: one object, coordinates per member (same heavy-atom order in every pose file)
    c.load(str(data["poses"][0]["file"]), "lig", multiplex=0, state=1)
    c.remove("lig and hydro")
    pose_xyz = [np.asarray(p["xyz"], float) for p in data["poses"]]
    assert all(x.shape == pose_xyz[0].shape for x in pose_xyz)
    c.hide("everything", "lig")
    c.show("sticks", "lig")
    c.show("spheres", "lig")
    c.color("gold", "lig")
    c.color("oxygen", "lig and elem O")
    c.set("stick_radius", 0.17, "lig")
    c.set("sphere_scale", 0.25, "lig")
    c.load(str(XTAL), "xtal_all")
    c.create("crystal", "xtal_all and resn CAM")
    c.delete("xtal_all")
    c.hide("everything", "crystal")
    c.show("sticks", "crystal")
    c.show("spheres", "crystal")
    c.color("teal", "crystal")
    c.color("oxygen", "crystal and elem O")
    c.set("stick_radius", 0.17, "crystal")
    c.set("sphere_scale", 0.25, "crystal")
    c.orient("heme or crystal")
    c.turn("x", 90)           # heme seen edge-on, camphor above its plane
    c.turn("y", 20)
    c.turn("z", 180)
    c.zoom("ribbon", 1)
    wide = np.array(c.get_view())
    c.zoom("crystal or heme or site", 1.0)
    close = np.array(c.get_view())
    for v in (wide, close):
        v[15] = max(0.1, -v[11] - 90)
        v[16] = -v[11] + 90
    fe = c.get_coords("heme and elem Fe")[0]
    yaw0, pitch0 = clear_angle(close, coords, pose_xyz, c.get_coords("heme and not hydro"))
    print(f"camera: yaw {yaw0:+.0f}, pitch {pitch0:+.0f} keeps the side chains off the camphor", flush=True)
    return {"wide": wide, "close": close, "yaw0": yaw0, "pitch0": pitch0, "site": coords, "poses": pose_xyz, "fe": fe,
            "crystal_center": np.asarray(data["xtal"]).mean(axis=0)}


def screen_xy(view, xyz, yaw, pitch):
    """Camera-plane coordinates of model points after the same turns as render_molecule (PyMOL turn = rotate
    the camera about its own axes)."""
    m = np.array(view[:9]).reshape(3, 3)           # column-major model->camera rotation, row-vector form
    cam = (np.asarray(xyz) - np.array(view[12:15])) @ m
    cam = rotate(cam, np.array([0.0, 1.0, 0.0]), yaw)
    cam = rotate(cam, np.array([1.0, 0.0, 0.0]), pitch)
    return cam[:, :2]


def clear_angle(view, site_coords, poses, heme):
    """Base camera turn that keeps every drawn side chain and the heme away (on screen) from the docked camphor
    during the whole rocking orbit, for every member; small pitches keep the heme edge-on."""
    best = None
    for yaw in range(-180, 180, 15):
        for pitch in (-12, -6, 0, 6, 12):
            overlaps, worst = 0, 9e9
            for rock in (-22, -11, 0, 11, 22):
                for site, lig in zip(site_coords, poses):
                    a = screen_xy(view, np.vstack([site, heme]), yaw + rock, pitch)
                    b = screen_xy(view, lig, yaw + rock, pitch)
                    d = np.linalg.norm(a[:, None] - b[None], axis=2)
                    overlaps += int((d < 1.5).sum())
                    worst = min(worst, float(d.min()))
            score = (overlaps, -worst)
            if best is None or score < best[0]:
                best = (score, yaw, pitch)
    return float(best[1]), float(best[2])


def enable_only(c, names):
    c.disable("all")
    for n in names:
        c.enable(n)


def rotate(xyz, axis, deg):
    axis = axis / np.linalg.norm(axis)
    a = math.radians(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k
    return xyz @ rot.T


def shot(seconds, scene):
    """Molecular content and camera for one instant: (view, objects, site coords, ligand coords, info)."""
    e = style.ease
    info = {"member": 0, "phase": "title", "lig": False, "blend": None}
    wide, close = scene["wide"], scene["close"]
    site = scene["site"][0]
    lig = None
    if seconds < T_DIVE:
        view, yaw, pitch = wide, -40 + 40 * seconds / T_DIVE, -6 * (1 - seconds / T_DIVE)
        objs = ["heme", "site"]
    elif seconds < T_CONF:
        p = e((seconds - T_DIVE) / (T_CONF - T_DIVE))
        view = wide * (1 - p) + close * p
        turn = e((p - 0.35) / 0.65)          # turn once the camera is centred on the pocket
        yaw = (scene["yaw0"] + 22) * turn
        pitch = scene["pitch0"] * turn
        objs = ["heme", "site"]
        info["phase"] = "dive"
        info["push"] = p
    else:
        view = close
        if seconds < T_DOCK:
            u = (seconds - T_CONF) / CONF_SPAN
            k = min(3, int(u))
            frac = u - k
            # hold 65 %, then move smoothly into the next conformation (a visual transition, not a trajectory)
            t = e((frac - 0.65) / 0.35) if k < 3 else 0.0
            nxt = min(3, k + 1)
            site = scene["site"][k] * (1 - t) + scene["site"][nxt] * t
            info.update(phase="conformations", member=k if t < 0.5 else nxt)
            objs = ["heme", "site"]
        elif seconds < T_CHECK:
            u = (seconds - T_DOCK) / DOCK_SPAN
            k = min(3, int(u))
            frac = u - k
            site = scene["site"][k]
            land = e(frac / 0.45)
            # camphor arrives through the substrate channel (away from the heme iron), spinning down
            target = scene["poses"][k]
            center = target.mean(axis=0)
            out = center - scene["fe"]
            out /= np.linalg.norm(out)
            spin = rotate(target - center, np.array([0.3, 1.0, 0.2]), 260 * (1 - land)) + center
            lig = spin + out * 14.0 * (1 - land) ** 1.5
            info.update(phase="docking", member=k, lig=True, landed=land >= 0.999, land=land)
            objs = ["heme", "site", "lig"]
        elif seconds < T_PANEL:
            k = int(np.argmin([p["rmsd"] for p in DATA["poses"]]))
            site = scene["site"][k]
            lig = scene["poses"][k]
            # crystal vs docked: dissolve between the two renders, never both drawn at once
            u = (seconds - T_CHECK) / (T_PANEL - T_CHECK)
            wave = 0.5 - 0.5 * math.cos(u * 2 * math.pi * 2)     # docked -> crystal -> docked -> crystal -> docked
            blend = style.ease((wave - 0.38) / 0.24)             # hold each pose; short dissolves
            info.update(phase="check", member=k, lig=True, blend=blend, landed=True)
            objs = ["heme", "site", "lig"]
        else:
            k = int(np.argmin([p["rmsd"] for p in DATA["poses"]]))
            site, lig = scene["site"][k], scene["poses"][k]
            info.update(phase="panel" if seconds < T_END else "end", member=k, lig=True, landed=True)
            objs = ["heme", "site", "lig"]
        t = seconds - T_CONF
        yaw = scene["yaw0"] + 22 * math.cos(t * 2 * math.pi / 30.0)      # slow rocking orbit around the pocket
        pitch = scene["pitch0"] + 6 * math.sin(t * 2 * math.pi / 21.0)
    return view, yaw, pitch, objs, site, lig, info


def plot_image(data, member, phase, landed):
    plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 12})
    fig = plt.figure(figsize=(7.5, 5.35), dpi=140, facecolor="white")
    a = fig.add_axes([0.13, 0.58, 0.82, 0.33])
    b = fig.add_axes([0.13, 0.105, 0.82, 0.33])
    for ax in (a, b):
        style.axes_style(ax)
    xs = np.arange(1, 5)
    site = [m["site_rmsd_to_start_A"] for m in data["members"]]
    a.bar(xs, site, color=[TEAL if k - 1 == member else "#C9D6DA" for k in xs], width=0.55)
    for k, v in zip(xs, site):
        a.text(k, v + 0.03, f"{v:.2f}", ha="center", color=INK if k - 1 == member else MUTED, fontsize=11)
    a.set_ylabel("Site RMSD (Å)")
    a.set_xticks(xs, ["start", "MC 1", "MC 2", "MC 3"])
    a.set_ylim(0, max(site) * 1.35 + 0.05)
    a.set_title("Pocket conformations · side-chain heavy atoms vs start", loc="left", fontsize=11, color=MUTED)
    r = [p["rmsd"] for p in data["poses"]]
    done = {"conformations": 0, "title": 0, "dive": 0, "docking": member + (1 if landed else 0)}.get(phase, 4)
    heights = [v if k <= done else 0.0 for k, v in zip(xs, r)]
    b.bar(xs, heights, color=[GOLD if v <= 2.0 else "#9FB3B9" for v in r], width=0.55)
    b.axhline(2.0, color=TEAL, lw=1.1, ls="--")
    b.text(4.45, 2.05, "2 Å success threshold", color=TEAL, fontsize=10, ha="right", va="bottom")
    for k, v in zip(xs, r):
        if k <= done:
            b.text(k, v + 0.06, f"{v:.2f}", ha="center", fontsize=11, color=GOLD, weight="bold")
    b.set_ylabel("Top-1 RMSD (Å)")
    b.set_xticks(xs, ["start", "MC 1", "MC 2", "MC 3"])
    b.set_ylim(0, 2.6)
    b.set_title("Redocked camphor vs crystal pose", loc="left", fontsize=11, color=MUTED)
    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return im


def panel_image(data, reveal):
    """Full-width chart: top-1 RMSD of every member for every complex of the panel."""
    plt.rcParams.update({"font.family": "Arial", "font.size": 13})
    fig = plt.figure(figsize=(12.6, 5.4), dpi=140, facecolor="white")
    ax = fig.add_axes([0.30, 0.10, 0.66, 0.84])
    style.axes_style(ax)
    ax.grid(axis="x", color="#E7EDEF", lw=0.6)
    ax.grid(axis="y", visible=False)
    cases = data["panel"]
    n = len(cases)
    for i, case in enumerate(cases):
        y = n - 1 - i
        show = float(np.clip(reveal * n - i, 0, 1))
        if show <= 0:
            continue
        for j, v in enumerate(case["rmsd"]):
            x = min(v, 8.0)
            ax.scatter([x * show], [y + (j - 1.5) * 0.12], s=70, color=GOLD if v <= 2.0 else "#9FB3B9",
                       zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(8.25, y, f"{case['ok']}/{len(case['rmsd'])}", va="center", fontsize=14, weight="bold",
                color=GOLD if case["ok"] else MUTED, alpha=show)
    ax.axvline(2.0, color=TEAL, lw=1.2, ls="--")
    ax.text(2.08, n - 0.45, "2 Å", color=TEAL, fontsize=11)
    ax.set_yticks(range(n), [c["label"] for c in reversed(cases)], fontsize=13, color=INK)
    for lab, case in zip(ax.get_yticklabels(), reversed(cases)):
        lab.set_color(INK)
    ax.set_xlim(0, 8.9)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("Top-1 RMSD to the crystal pose, per ensemble member (Å)", color=INK)
    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return im


DATA = None


def compose(molecule, plot, info, seconds, panel=None):
    data = DATA
    im = Image.new("RGBA", (W, H), "white")
    e = style.ease
    title = 1 - e((seconds - 6.4) / 1.2)
    analysis = e((seconds - (T_CONF - 0.6)) / 1.2) * (1 - e((seconds - (T_PANEL - 0.2)) / 0.8))
    panel_on = e((seconds - T_PANEL) / 0.8) * (1 - e((seconds - T_END) / 0.8))
    end_on = e((seconds - T_END) / 0.8)
    # molecule placement: right during the title, centred during the dive, left with the analysis
    scale = 1 - 0.31 * analysis
    px = int(160 + 380 * title - 160 * analysis)
    py = int(125 + 115 * analysis)
    mol = molecule.resize((int(MW * scale), int(MH * scale)), Image.Resampling.LANCZOS)
    mol_layer = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    edge = np.ones(mol.size[0])
    ramp = max(1, int(140 * scale))
    edge[:ramp] = np.linspace(0, 1, ramp)
    edge[-ramp:] = np.linspace(1, 0, ramp)
    mask = Image.fromarray((np.tile(edge, (mol.size[1], 1)) * 255).astype(np.uint8))
    mol_layer.paste(mol, (px, py), mask)
    style.faded(im, mol_layer, 1 - max(panel_on, end_on))
    d = ImageDraw.Draw(im)
    txt(d, (78, 42), "L E O N A R D O", 25, bold=True)
    txt(d, (1510, 46), "MOLECULAR FILMS  /  08", 18, MUTED)
    d.line((78, 93, 1842, 93), fill=LINE, width=1)
    if title > 0:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        for x in range(0, 900):
            ld.line((x, 100, x, 999), fill=(255, 255, 255, int(235 * (1 - e((x - 560) / 340)))))
        rise = 1 - e(seconds / 1.4)
        txt(ld, (85, 252 + 30 * rise), "08 / ENSEMBLE REDOCKING", 20, TEAL, True)
        for j, line in enumerate(["A pocket that", "breathes."]):
            txt(ld, (78, 307 + j * 102 + 40 * rise), line, 83, bold=True)
        txt(ld, (86, 563), "CYTOCHROME P450cam · camphor above the heme", 23, MUTED)
        txt(ld, (86, 604), "PDB 2CPP · Monte Carlo receptor ensemble · smina / Vina", 20, MUTED)
        ld.line((86, 680, 86 + int(92 * e((seconds - 1.0) / 1.0)), 680), fill=GOLD, width=4)
        txt(ld, (86, 713), "Four conformations. Four docking runs. One question:", 21)
        txt(ld, (86, 746), "does the pocket's flexibility change the answer?", 21)
        style.faded(im, layer, title * e(seconds / 0.8))
    phase = info["phase"]
    if phase == "dive":
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        txt(ld, (80, 128), "INSIDE THE ENZYME", 18, TEAL, True)
        txt(ld, (76, 166), "The camphor pocket, sealed above the heme.", 48, bold=True)
        txt(ld, (80, 930), "Side chains lining the pocket · Thr101 and Asp297 highlighted in teal", 20, MUTED)
        style.faded(im, layer, e((seconds - T_DIVE - 0.3) / 1.0) * (1 - e((seconds - T_CONF + 0.8) / 0.6)))
    if analysis > 0:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        caption = {"conformations": ("MONTE CARLO · 300 K · AMBER14 / GB-OBC2", "Sampling the pocket's side chains."),
                   "docking": ("SMINA / VINA · ONE RUN PER CONFORMATION", "Redocking into every conformation."),
                   "check": ("CRYSTAL POSE vs DOCKED POSE · SAME CAMERA", "Every conformation finds the crystal pose."),
                   }.get(phase, ("", ""))
        txt(ld, (80, 128), caption[0], 18, TEAL, True)
        txt(ld, (76, 166), caption[1], 48, bold=True)
        for y in range(760, 816):
            ld.line((0, y, 1058, y), fill=(255, 255, 255, int(255 * e((y - 760) / 55))))
        ld.rectangle((0, 816, 1058, 999), fill="white")
        ld.rectangle((1060, 243, 1870, 997), fill="white")
        txt(ld, (1090, 254), "The ensemble, quantified.", 29, bold=True)
        txt(ld, (1092, 300), f"Monte Carlo at 300 K · acceptance {data['acceptance']:.0%} · "
                             f"{len(data['site'])} site residues", 19, MUTED)
        layer.paste(plot.resize((805, 575), Image.Resampling.LANCZOS), (1050, 338))
        k = info["member"]
        m, p = data["members"][k], data["poses"][k]
        txt(ld, (86, 273), MEMBER_NAMES[k], 20, TEAL, True)
        if phase == "check":
            showing = "CRYSTAL POSE (teal)" if info["blend"] > 0.5 else "DOCKED POSE (gold)"
            label, metric, unit = "NOW SHOWING · " + showing, f"{p['rmsd']:.2f}", "Å apart"
            detail = f"Vina score {p['score']:.2f} kcal/mol · camphor carbonyl toward Tyr96"
        elif phase == "docking":
            label = "TOP-1 POSE · RMSD TO CRYSTAL"
            metric, unit = (f"{p['rmsd']:.2f}", "Å") if info.get("landed") else ("···", "")
            detail = f"Vina score {p['score']:.2f} kcal/mol" if info.get("landed") else "searching the pocket"
        else:
            label = "SIDE-CHAIN SHIFT FROM THE START"
            metric, unit = f"{m['site_rmsd_to_start_A']:.2f}", "Å"
            detail = f"Relative energy {m['relative_energy_kj_mol']:+.1f} kJ/mol (Amber14 / GB-OBC2)"
        txt(ld, (84, 826), label, 18, TEAL, True)
        txt(ld, (79, 858), metric, 68, bold=True)
        width = ld.textbbox((0, 0), metric, font=style.font(68, True))[2]
        txt(ld, (94 + width, 897), unit, 24, MUTED)
        txt(ld, (86, 945), detail, 22)
        note = {"conformations": "Backbone fixed · only binding-site side chains move",
                "docking": "Rigid-receptor docking into each member",
                "check": "Same camera, same frame of reference: teal = crystal, gold = docked"}.get(phase, "")
        txt(ld, (1092, 934), note, 17, MUTED)
        style.faded(im, layer, analysis)
    if panel_on > 0 and panel is not None:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        ld.rectangle((0, 100, W, 999), fill="white")
        txt(ld, (80, 128), "THE SAME PIPELINE ON SIX DIFFERENT COMPLEXES", 18, TEAL, True)
        txt(ld, (76, 166), "One protocol, any protein–ligand pair.", 48, bold=True)
        layer.paste(panel.resize((1640, 703), Image.Resampling.LANCZOS), (140, 236))
        txt(ld, (86, 958), "Gold: within 2 Å of the crystal pose · every case: ensemble built, cofactors kept, "
                           "all automatic checks passed", 18, MUTED)
        style.faded(im, layer, panel_on)
    if end_on > 0:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        ld.rectangle((0, 100, W, 999), fill="white")
        txt(ld, (85, 330), "08 / ENSEMBLE REDOCKING", 20, TEAL, True)
        txt(ld, (78, 385), "Flexible pockets,", 83, bold=True)
        txt(ld, (78, 487), "verified answers.", 83, bold=True)
        ld.line((86, 640, 178, 640), fill=GOLD, width=4)
        txt(ld, (86, 672), "Leonardo · open-source PyMOL plugin · Monte Carlo receptor ensembles + smina / Vina",
            22, MUTED)
        txt(ld, (86, 710), "github.com/juvenalyosa/Leonardo", 22, INK, True)
        style.faded(im, layer, end_on)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 1000, W, H), fill="white")
    d.line((78, 1007, 1842, 1007), fill=LINE, width=1)
    d.line((78, 1007, 78 + int(1764 * seconds / ((FRAMES - 1) / FPS)), 1007), fill=TEAL, width=3)
    txt(d, (80, 1030), "Monte Carlo side-chain ensemble · rigid-receptor docking per member · "
                       "Vina score is not binding free energy", 18, MUTED)
    txt(d, (1660, 1030), "PDB 2CPP", 18, MUTED)
    return im.convert("RGB")


def render_molecule(c, scratch, view, yaw, pitch, objs, push):
    c.set_view(view.tolist())
    c.turn("y", yaw)
    c.turn("x", pitch)
    # context layers at half resolution: they are blurred and faded anyway
    enable_only(c, ["ribbon"])
    c.png(str(scratch / "ribbon.png"), MW // 2, MH // 2, ray=1, quiet=1)
    enable_only(c, ["envelope"])
    c.set("transparency", 0.0, "envelope")
    c.png(str(scratch / "surface.png"), MW // 2, MH // 2, ray=1, quiet=1)
    enable_only(c, objs)
    c.png(str(scratch / "active.png"), MW, MH, ray=1, quiet=1)
    ribbon = Image.open(scratch / "ribbon.png").convert("RGBA").resize((MW, MH), Image.Resampling.BICUBIC)
    surface = Image.open(scratch / "surface.png").convert("RGBA").resize((MW, MH), Image.Resampling.BICUBIC)
    active = Image.open(scratch / "active.png").convert("RGBA")
    ribbon = ribbon.filter(ImageFilter.GaussianBlur(1.4 * push))
    surface = surface.filter(ImageFilter.GaussianBlur(2.5 * push))
    ribbon.putalpha(ribbon.getchannel("A").point(lambda a: int(a * (0.94 - 0.71 * push))))
    surface.putalpha(surface.getchannel("A").point(lambda a: int(a * (0.55 - 0.41 * push))))
    im = Image.new("RGBA", (MW, MH), "white")
    im.alpha_composite(surface)
    im.alpha_composite(ribbon)
    im.alpha_composite(active)
    return im.convert("RGB")


def render(preview=False, no_encode=False, start=0, stop=None):
    global DATA
    DATA = data = load()
    folder = OUT / "renders" / "ensemble_redocking_v2"
    folder.mkdir(parents=True, exist_ok=True)
    scratch = folder / f"worker_{os.getpid()}"
    scratch.mkdir(exist_ok=True)
    marks = [0.5, 10.0, 12.0, 14.0, 19.0, 33.2, 41.0, 46.0, 49.0, 58.0, 64.0]
    targets = [int(s * FPS) for s in marks] if preview else range(start, stop or FRAMES)
    plots, panels = {}, {}
    started = time.perf_counter()
    p = pymol2.PyMOL()
    p.start()
    try:
        c = p.cmd
        scene = setup(c, data)
        for f in targets:
            path = folder / (f"preview_{f:05d}.png" if preview else f"frame_{f:05d}.png")
            if path.exists() and not preview:
                continue
            seconds = f / FPS
            view, yaw, pitch, objs, site, lig, info = shot(seconds, scene)
            push = info.get("push", 0.0 if seconds < T_DIVE else 1.0)
            c.load_coords(np.asarray(site, float).tolist(), "site")
            if lig is not None:
                c.load_coords(np.asarray(lig, float).tolist(), "lig")
            if seconds >= T_END + 0.8:
                molecule = Image.new("RGB", (MW, MH), "white")
            elif info["blend"] is not None and 0.0 < info["blend"] < 1.0:
                docked = render_molecule(c, scratch, view, yaw, pitch, objs, push)
                crystal = render_molecule(c, scratch, view, yaw, pitch, ["heme", "site", "crystal"], push)
                molecule = Image.blend(docked, crystal, float(info["blend"]))
            elif info["blend"] == 1.0:
                molecule = render_molecule(c, scratch, view, yaw, pitch, ["heme", "site", "crystal"], push)
            else:
                molecule = render_molecule(c, scratch, view, yaw, pitch, objs, push)
            key = (info["member"], info["phase"], bool(info.get("landed")))
            if key not in plots:
                plots[key] = plot_image(data, info["member"], info["phase"], bool(info.get("landed")))
            panel = None
            if seconds >= T_PANEL - 0.1:
                reveal = round(float(np.clip((seconds - T_PANEL - 0.3) / 3.5, 0, 1)), 2)
                if reveal not in panels:
                    panels[reveal] = panel_image(data, reveal)
                panel = panels[reveal]
            compose(molecule, plots[key], info, seconds, panel).save(path)
            if f % 48 == 0 or preview:
                print("frame", f, "/", FRAMES, round(time.perf_counter() - started, 1), "s", flush=True)
    finally:
        p.stop()
    if not preview and not no_encode and stop is None:
        encode()


def encode():
    folder = OUT / "renders" / "ensemble_redocking_v2"
    missing = [i for i in range(FRAMES) if not (folder / f"frame_{i:05d}.png").exists()]
    assert not missing, f"{len(missing)} frames missing, first {missing[:5]}"
    dest = OUT / "videos" / "08_ensemble_redocking.mp4"
    proc = subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-framerate", str(FPS), "-i",
                           str(folder / "frame_%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)],
                          capture_output=True, text=True)
    (folder / "ffmpeg.log").write_text(proc.stderr)
    assert proc.returncode == 0, proc.stderr[-2000:]
    print("Encoded", dest, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--no-encode", action="store_true")
    ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    a = ap.parse_args()
    if a.encode_only:
        encode()
    else:
        render(a.preview, a.no_encode, a.start, a.stop)
