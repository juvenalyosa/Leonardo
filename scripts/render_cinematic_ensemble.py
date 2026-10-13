#!/usr/bin/env python3
"""Molecular film 08: ensemble redocking (P450cam + camphor), white-stage cinematic_v2 design.

Every number on screen comes from artifacts/mc_redock_panel/p450cam_2CPP (Leonardo's redocking with a Monte
Carlo receptor ensemble); the poses shown are re-checked against those RMSDs before rendering.
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
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "artifacts/mc_redock_panel/p450cam_2CPP"
XTAL = Path(os.environ.get("LEONARDO_2CPP", ROOT / "artifacts/mc_redock_panel/2CPP.pdb"))
OUT = ROOT / "artifacts/cinematic_v2"
FRAMES, FPS = 576, 24
W, H = 1920, 1080
TEAL, GOLD, INK, MUTED, LINE = style.TEAL, style.GOLD, style.INK, style.MUTED, style.LINE
txt = style.txt
MEMBER_FILES = ["best_model.pdb", "mc_001.pdb", "mc_002.pdb", "mc_003.pdb"]


def heavy_xyz_from_sdf(path, index=0):
    blocks = open(path).read().split("$$$$")
    lines = blocks[index].strip("\n").splitlines()
    n = int(lines[3][:3])
    out = []
    for ln in lines[4:4 + n]:
        p = ln.split()
        if p[3] != "H":
            out.append([float(p[0]), float(p[1]), float(p[2])])
    return np.array(out)


def load():
    summary = json.loads((CASE / "summary.json").read_text())
    xtal = np.array([[float(ln[30:38]), float(ln[38:46]), float(ln[46:54])] for ln in open(XTAL)
                     if ln.startswith("HETATM") and ln[17:20] == "CAM"])
    poses = []
    for k in range(1, 5):
        folder = next(p for p in sorted((CASE / "docking").iterdir()) if f"mc_{k:03d}" in p.name)
        xyz = heavy_xyz_from_sdf(folder / "docked_poses_view.sdf", 0)
        # symmetry-free check against the crystal (camphor is near-symmetric: allow the reported value or less)
        d = np.linalg.norm(xyz[:, None, :] - xtal[None, :, :], axis=2)
        nearest = float(np.sqrt((d.min(axis=1) ** 2).mean()))
        reported = float(summary["poses"][k - 1]["rmsd_A"])
        assert nearest <= reported + 0.05, (k, nearest, reported)
        poses.append({"file": folder / "docked_poses_view.sdf", "rmsd": reported,
                      "score": float(summary["poses"][k - 1]["score_kcal_mol"])})
    members = summary["members"]
    return {"summary": summary, "poses": poses, "members": members, "xtal": xtal,
            "acceptance": summary["ensemble"]["acceptance"],
            "site": summary["ensemble"]["site_residues"]}


def setup(c, data):
    c.reinitialize()
    c.set("retain_order", 1)
    c.set("max_threads", 4)
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
                      "hemec": [0.46, 0.36, 0.30]}.items():
        c.set_color(name, rgb)
    base = CASE / "ensemble" / "best_model.pdb"
    c.load(str(base), "enzyme")
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
    c.set("stick_radius", 0.13, "heme")
    c.set("sphere_scale", 0.45, "heme and elem Fe")
    resi = "+".join(s[3:-1] for s in data["site"])
    for k, f in enumerate(MEMBER_FILES, start=1):
        c.load(str(CASE / "ensemble" / f), f"m{k}")
        c.remove(f"m{k} and hydro")
        c.create(f"site{k}", f"m{k} and polymer and resi {resi} and not name N+C+O")
        c.delete(f"m{k}")
        c.hide("everything", f"site{k}")
        c.show("sticks", f"site{k}")
        c.color("slate", f"site{k} and elem C")
        c.color("oxygen", f"site{k} and elem O")
        c.color("nitrogen", f"site{k} and elem N")
        c.set("stick_radius", 0.11, f"site{k}")
        c.load(str(data["poses"][k - 1]["file"]), f"pose{k}", multiplex=0, state=1)
        c.remove(f"pose{k} and hydro")
        c.hide("everything", f"pose{k}")
        c.show("sticks", f"pose{k}")
        c.show("spheres", f"pose{k}")
        c.color("gold", f"pose{k}")
        c.color("oxygen", f"pose{k} and elem O")
        c.set("stick_radius", 0.16, f"pose{k}")
        c.set("sphere_scale", 0.24, f"pose{k}")
    c.load(str(XTAL), "xtal_all")
    c.create("crystal", "xtal_all and resn CAM")
    c.delete("xtal_all")
    c.hide("everything", "crystal")
    c.show("sticks", "crystal")
    c.color("teal", "crystal")
    c.set("stick_radius", 0.10, "crystal")
    c.set("stick_transparency", 0.35, "crystal")
    c.orient("heme or crystal")
    c.turn("x", 90)           # heme seen edge-on, camphor clearly above its plane
    c.turn("y", 20)
    c.turn("z", 180)
    c.zoom("ribbon", 2)
    wide = np.array(c.get_view())
    c.zoom("crystal or heme or site1", 2.0)
    close = np.array(c.get_view())
    for v in (wide, close):
        v[15] = max(0.1, -v[11] - 90)
        v[16] = -v[11] + 90
    c.set("transparency", 0, "envelope")
    return wide, close


def show_member(c, k, with_pose):
    for j in range(1, 5):
        c.disable(f"site{j}")
        c.disable(f"pose{j}")
    c.enable(f"site{k}")
    c.enable("heme")
    c.enable("crystal" if with_pose else "crystal")
    if not with_pose:
        c.disable("crystal")
    if with_pose:
        c.enable(f"pose{k}")


def state(seconds):
    """(member, show_pose, phase)."""
    if seconds < 6.6:
        return 1, False, "intro"
    if seconds < 13.8:
        return min(4, 1 + int((seconds - 6.6) / 1.8)), False, "conformations"
    if seconds < 20.6:
        return min(4, 1 + int((seconds - 13.8) / 1.7)), True, "docking"
    return 3, True, "result"


def plot_image(data, member, phase):
    plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 12})
    fig = plt.figure(figsize=(7.5, 5.35), dpi=140, facecolor="white")
    a = fig.add_axes([0.13, 0.58, 0.82, 0.33])
    b = fig.add_axes([0.13, 0.105, 0.82, 0.33])
    for ax in (a, b):
        style.axes_style(ax)
    xs = np.arange(1, 5)
    site = [m["site_rmsd_to_start_A"] for m in data["members"]]
    colors = [TEAL if k == member else "#C9D6DA" for k in xs]
    a.bar(xs, site, color=colors, width=0.55)
    for k, v in zip(xs, site):
        a.text(k, v + 0.03, f"{v:.2f}", ha="center", color=INK if k == member else MUTED, fontsize=11)
    a.set_ylabel("Site RMSD (Å)")
    a.set_xticks(xs, ["start", "MC 1", "MC 2", "MC 3"])
    a.set_ylim(0, max(site) * 1.35 + 0.05)
    a.set_title("Pocket conformations · side-chain heavy atoms vs start", loc="left", fontsize=11, color=MUTED)
    r = [p["rmsd"] for p in data["poses"]]
    shown = phase in ("docking", "result")
    bcol = []
    for k, v in zip(xs, r):
        if not shown or (phase == "docking" and k > member):
            bcol.append("#EEF2F3")
        elif v <= 2.0:
            bcol.append(GOLD)
        else:
            bcol.append("#9FB3B9" if k == member else "#C9D6DA")
    heights = [v if (phase == "result" or (shown and k <= member)) else 0.0 for k, v in zip(xs, r)]
    b.bar(xs, heights, color=bcol, width=0.55)
    b.axhline(2.0, color=TEAL, lw=1.1, ls="--")
    b.text(4.45, 2.05, "2 Å", color=TEAL, fontsize=10, ha="right", va="bottom")
    if shown:
        for k, v in zip(xs, r):
            if phase == "result" or k <= member:
                b.text(k, v + 0.08, f"{v:.2f}", ha="center", fontsize=11,
                       color=GOLD if v <= 2.0 else (INK if k == member else MUTED), weight="bold" if v <= 2 else None)
    b.set_ylabel("Top-1 RMSD (Å)")
    b.set_xticks(xs, ["start", "MC 1", "MC 2", "MC 3"])
    b.set_ylim(0, max(r) * 1.3)
    b.set_title("Redocked camphor vs crystal pose", loc="left", fontsize=11, color=MUTED)
    fig.canvas.draw()
    im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return im


def compose(data, molecule, plot, member, show_pose, phase, seconds):
    im = Image.new("RGBA", (W, H), "white")
    intro = 1 - style.ease((seconds - 2.6) / 1.0)
    analysis = style.ease((seconds - 6.7) / 1.1)
    scale = 1 - 0.31 * analysis
    px, py = int(160 + 440 * intro - 160 * analysis), int(125 + 115 * analysis)
    im.paste(molecule.resize((int(1600 * scale), int(900 * scale)), Image.Resampling.LANCZOS), (px, py))
    d = ImageDraw.Draw(im)
    txt(d, (78, 42), "L E O N A R D O", 25, bold=True)
    txt(d, (1510, 46), "MOLECULAR FILMS  /  08", 18, MUTED)
    d.line((78, 93, 1842, 93), fill=LINE, width=1)
    if intro > 0:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        txt(ld, (85, 252), "08 / ENSEMBLE REDOCKING", 20, TEAL, True)
        for j, line in enumerate(["A pocket that", "breathes."]):
            txt(ld, (78, 307 + j * 102), line, 83, bold=True)
        txt(ld, (86, 563), "CYTOCHROME P450cam · camphor above the heme", 23, MUTED)
        txt(ld, (86, 604), "PDB 2CPP · Monte Carlo receptor ensemble · smina / Vina", 20, MUTED)
        ld.line((86, 680, 178, 680), fill=GOLD, width=4)
        txt(ld, (86, 713), "Four conformations. Four docking runs.", 21)
        style.faded(im, layer, intro)
    if intro < 1:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        txt(ld, (80, 128), "CYTOCHROME P450cam · CAMPHOR · HEME KEPT IN EVERY MEMBER", 18, TEAL, True)
        caption = {"intro": "The camphor pocket.", "conformations": "Sampling the pocket's side chains.",
                   "docking": "Redocking into every conformation.",
                   "result": "One conformation recovers the crystal pose."}[phase]
        txt(ld, (76, 166), caption, 48, bold=True)
        style.faded(im, layer, 1 - intro)
    if analysis > 0:
        layer = Image.new("RGBA", (W, H))
        ld = ImageDraw.Draw(layer)
        for y in range(760, 816):
            ld.line((0, y, 1058, y), fill=(255, 255, 255, int(255 * style.ease((y - 760) / 55))))
        ld.rectangle((0, 816, 1058, 999), fill="white")
        ld.rectangle((1060, 243, 1870, 997), fill="white")
        txt(ld, (1090, 254), "The ensemble, quantified.", 29, bold=True)
        txt(ld, (1092, 300), f"Monte Carlo at 300 K · acceptance {data['acceptance']:.0%} · "
                             f"{len(data['site'])} site residues", 19, MUTED)
        layer.paste(plot.resize((805, 575), Image.Resampling.LANCZOS), (1050, 338))
        names = ["STARTING STRUCTURE", "MONTE CARLO CONFORMATION 1", "MONTE CARLO CONFORMATION 2",
                 "MONTE CARLO CONFORMATION 3"]
        txt(ld, (86, 273), names[member - 1], 20, TEAL, True)
        m = data["members"][member - 1]
        p = data["poses"][member - 1]
        if phase in ("docking", "result"):
            label = "BEST POSE IN THE ENSEMBLE" if phase == "result" else "TOP-1 POSE · RMSD TO CRYSTAL"
            metric, unit = f"{p['rmsd']:.2f}", "Å"
            detail = f"Vina score {p['score']:.2f} kcal/mol" + ("  ·  crystal pose recovered" if p["rmsd"] <= 2 else "")
        else:
            label = "SIDE-CHAIN SHIFT FROM THE START"
            metric, unit = f"{m['site_rmsd_to_start_A']:.2f}", "Å"
            detail = f"Relative energy {m['relative_energy_kj_mol']:+.1f} kJ/mol (Amber14 / GB-OBC2)"
        txt(ld, (86, 777), "Gold: redocked camphor · teal outline: crystal pose", 19, MUTED)
        txt(ld, (84, 826), label, 18, TEAL, True)
        txt(ld, (79, 858), metric, 68, bold=True)
        width = ld.textbbox((0, 0), metric, font=style.font(68, True))[2]
        txt(ld, (94 + width, 897), unit, 24, MUTED)
        txt(ld, (86, 945), detail, 22)
        note = ("The best Vina score picks a 2.90 Å pose: score alone does not rank the members"
                if phase == "result" else "Backbone fixed · only binding-site side chains move")
        txt(ld, (1092, 934), note, 17, MUTED)
        style.faded(im, layer, analysis)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 1000, W, H), fill="white")
    d.line((78, 1007, 1842, 1007), fill=LINE, width=1)
    d.line((78, 1007, 78 + int(1764 * seconds / ((FRAMES - 1) / FPS)), 1007), fill=TEAL, width=3)
    txt(d, (80, 1030), "Monte Carlo side-chain ensemble · rigid-receptor docking per member · "
                       "Vina score is not binding free energy", 18, MUTED)
    txt(d, (1660, 1030), "PDB 2CPP", 18, MUTED)
    return im.convert("RGB")


def render(preview=False, no_encode=False):
    data = load()
    folder = OUT / "renders" / "ensemble_redocking"
    folder.mkdir(parents=True, exist_ok=True)
    scratch = folder / f"worker_{os.getpid()}"
    scratch.mkdir(exist_ok=True)
    targets = [0, 100, 170, 260, 340, 420, 520, 575] if preview else range(FRAMES)
    cached, plots = {}, {}
    started = time.perf_counter()
    p = pymol2.PyMOL()
    p.start()
    try:
        c = p.cmd
        wide, close = setup(c, data)
        context = None
        for f in targets:
            path = folder / f"frame_{f:05d}.png"
            if path.exists() and not preview:
                continue
            seconds = f / FPS
            member, show_pose, phase = state(seconds)
            push = style.ease((seconds - 2.6) / 4)
            key = (member, show_pose, f if seconds < 6.6 else "close")
            if key not in cached:
                c.set_view((wide * (1 - push) + close * push).tolist())
                c.turn("y", -13 * (1 - push) + 8 * math.sin(min(seconds / 6.6, 1) * math.pi) * (1 - push))
                c.turn("x", -3 * (1 - push))
                show_member(c, member, show_pose)
                if seconds >= 6.6 and context is None:
                    enabled = c.get_names("objects", enabled_only=1)
                    c.disable("all")
                    c.enable("ribbon")
                    c.png(str(scratch / "static_ribbon.png"), 1600, 900, ray=1, quiet=1)
                    c.disable("all")
                    c.enable("envelope")
                    c.png(str(scratch / "static_surface.png"), 1600, 900, ray=1, quiet=1)
                    context = (Image.open(scratch / "static_ribbon.png").convert("RGBA"),
                               Image.open(scratch / "static_surface.png").convert("RGBA"))
                    c.disable("all")
                    for name in enabled:
                        c.enable(name)
                cached[key] = style.ray(c, scratch, push, context if seconds >= 6.6 else None)
            if (member, phase) not in plots:
                plots[member, phase] = plot_image(data, member, phase)
            compose(data, cached[key], plots[member, phase], member, show_pose, phase, seconds).save(path)
            if seconds < 6.6:
                del cached[key]
            if f % 48 == 0 or preview:
                print("frame", f, "/", FRAMES, round(time.perf_counter() - started, 1), "s", flush=True)
    finally:
        p.stop()
    if not preview and not no_encode:
        encode()


def encode():
    folder = OUT / "renders" / "ensemble_redocking"
    assert all((folder / f"frame_{i:05d}.png").exists() for i in range(FRAMES))
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
    a = ap.parse_args()
    if a.encode_only:
        encode()
    else:
        render(a.preview, a.no_encode)
