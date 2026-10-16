#!/usr/bin/env python3
"""
Compositing stage of the 60 s Leonardo materials film, for YouTube (16:9) and Instagram (9:16): the
PyMOL frames of scripts/materials_film_molecules.py, graded, with titles, live data panels from the
real Leonardo run (artifacts/materials_2026/data/results.json) and music.

    python scripts/materials_film_compose.py yt|ig [--only SECONDS ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trailer_compose as tc  # noqa: E402  (fonts, grade, panels, plots)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "materials_2026"
DATA = ART / "data"
FPS, TOTAL = 24, 60.0
SCENES = {"np": (0.0, 8.6), "pt": (8.6, 10.8), "graphene": (10.8, 13.0), "silica": (13.0, 15.2),
          "rutile": (15.2, 17.2), "au": (17.2, 24.0), "gr": (24.0, 30.0), "si": (30.0, 36.4), "qm": (36.4, 51.0),
          "outro": (51.0, 60.0)}
MUSIC = ART / "electronic.wav"
MUSIC_START = 145.0          # "Ouroboros"/"Exhilarate": a build starts here
INK, MUTED, TEAL, AMBER, GOLD = tc.INK, tc.MUTED, tc.TEAL, tc.AMBER, (232, 190, 80)
card, metric, ease, clamp, window = tc.card, tc.metric, tc.ease, tc.clamp, tc.window


def text(layer, xy, s, size, *a, **k):
    """tc.text; on the phone format small print is 22 % larger."""
    tc.text(layer, xy, s, int(round(size * 1.22)) if FMT == "ig" and size < 40 else size, *a, **k)

FMT = "yt"
W, H = 1920, 1080


def configure(fmt):
    global FMT, W, H
    FMT = fmt
    W, H = (1920, 1080) if fmt == "yt" else (1080, 1920)
    tc.W, tc.H = W, H
    tc._vignette = tc._grain = tc._scrim = None


def mol_frame(scene, i):
    d = ART / "frames" / FMT / scene
    n = len(list(d.glob("mol_*.png")))
    p = d / f"mol_{min(i, max(0, n - 1)):04d}.png"
    return Image.open(p).convert("RGBA") if p.exists() else Image.new("RGBA", (W, H), (3, 6, 13, 255))


def scene_at(t):
    for name, (a, b) in SCENES.items():
        if a <= t < b:
            return name, a
    return "outro", SCENES["outro"][0]


# ------------------------------------------------------------------------------------------- data
def load():
    r = json.load(open(DATA / "results.json"))
    mats = {
        "np": ("Gold nanoparticle", f"Au{r['1']['n_atoms']}", "icosahedron · INTERFACE-FF",
               [("atoms", f"{r['1']['n_atoms']}"), ("diameter", f"{r['1'].get('diameter_nm', 0):.1f} nm")]),
        "pt": ("Platinum (111) surface", "Pt(111)", "fcc slab · INTERFACE-FF",
               [("atoms", f"{r['2']['n_atoms']}"), ("cell", " × ".join(f"{x:.1f}" for x in r['2']['cell_nm'][:2]) + " nm")]),
        "graphene": ("Graphene", "C", "Steele carbon",
                     [("atoms", f"{r['3']['n_atoms']}"), ("cell", " × ".join(f"{x:.1f}" for x in r['3']['cell_nm'][:2]) + " nm")]),
        "silica": ("Hydroxylated silica", "SiO₂ (001)", "α-quartz · ClayFF · silanol surface",
                   [("atoms", f"{r['4']['n_atoms']}"), ("silanols", f"{r['4'].get('silanols_per_nm2', 9.6)} OH/nm²")]),
        "rutile": ("Rutile titania", "TiO₂ (110)", "stoichiometric slab, bridging-O rows",
                   [("atoms", f"{r['5']['n_atoms']}"), ("cell", " × ".join(f"{x:.1f}" for x in r['5']['cell_nm'][:2]) + " nm")]),
    }
    surf = {}
    for key, step, label in (("au", "7", "Au(111)"), ("gr", "8", "Graphene"), ("si", "9", "Silica")):
        sa = r[step]["surface_analysis"]
        surf[key] = {"label": label, "lj": float(sa["protein_surface_LJ_kj_mol"]),
                     "contacts": [c[:3].capitalize() + c[3:-1] for c in sa["contact_residues"]],
                     "closest": float(sa["min_distance_A"]), "atoms": int(r[step]["n_atoms"])}
    qm = r["11"]
    cl = r["10"]
    loc = json.load(open(DATA / "orbital_localization.json"))
    return {"mats": mats, "surf": surf,
            "qm": {"homo": float(qm["homo_ev"]), "lumo": float(qm["lumo_ev"]), "gap": float(qm["gap_ev"]),
                   "atoms": int(cl["n_atoms"]), "links": int(cl["n_link_atoms"]), "loc": loc}}


# ------------------------------------------------------------------------------------ layout bits
def tag(lay, t, a, num, label, title, sub):
    if FMT == "yt":
        tc.scene_tag(lay, t, a, num, label, title, sub)
        return
    # vertical: centred, larger, at the top
    k = window(t, a, a + 99, 0.6)
    if k <= 0:
        return
    shade = Image.new("RGBA", lay.size)
    text(shade, (W / 2, 190), title, 64, (0, 0, 0), "demi", 0.8 * k, "mm")
    lay.alpha_composite(shade.filter(ImageFilter.GaussianBlur(8)))
    text(lay, (W / 2, 120), f"{num}  ·  {label}", 26, TEAL, "demi", k, "mm", spacing=4)
    text(lay, (W / 2, 190), title, 64, INK, "demi", ease((t - a - 0.15) / 0.6), "mm")
    for j, line in enumerate(wrap(sub, 30 if FMT == "ig" else 999)):
        text(lay, (W / 2, 252 + 36 * j), line, 28, (196, 210, 232), "regular", ease((t - a - 0.35) / 0.6), "mm")


def wrap(s, n):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n and line:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    return out + ([line] if line else [])


def panel_box():
    return (1350, 300, 1840, 860) if FMT == "yt" else (60, 1180, 1020, 1760)


def provenance(lay, t, a, s):
    if FMT == "yt":
        tc.provenance(lay, t, a, s)
        return
    k = ease((t - a - 0.8) / 0.8)
    text(lay, (W / 2, H - 112), "REAL CALCULATION", 20, TEAL, "demi", 0.9 * k, "mm", spacing=4)
    for j, line in enumerate(wrap(s, 52)):
        text(lay, (W / 2, H - 78 + 28 * j), line, 20, (196, 210, 232), "regular", 0.9 * k, "mm")


def title_card(base, t, a, end, y, subtitle):
    k_logo = window(t, a, end, 0.7)
    if k_logo > 0:
        s = int((150 if FMT == "yt" else 190) * (0.86 + 0.14 * tc.ease_out((t - a) / 1.2)))
        lg = tc.logo(s)
        lg.putalpha(lg.getchannel("A").point(lambda v: int(v * k_logo)))
        halo = Image.new("RGBA", base.size)
        halo.alpha_composite(lg, (W // 2 - s // 2, y - 130 - s // 2 - (40 if FMT == "ig" else 0)))
        base.alpha_composite(halo.filter(ImageFilter.GaussianBlur(28)))
        base.alpha_composite(halo)
    k = window(t, a + 0.5, end, 0.7)
    spacing = int((26 if FMT == "yt" else 16) - 12 * tc.ease_out((t - a - 0.5) / 1.6))
    tc.glow_text(base, (W / 2, y + 30), "LEONARDO", 112 if FMT == "yt" else 104, INK, "demi", k, "mm",
                 spacing=spacing, radius=18)
    sub = Image.new("RGBA", base.size)
    text(sub, (W / 2, y + 112), subtitle, 30, TEAL, "demi", window(t, a + 1.1, end, 0.7), "mm", spacing=10)
    base.alpha_composite(sub)


# --------------------------------------------------------------------------------------- overlay
def overlay(base, t, i, D):
    lay = Image.new("RGBA", (W, H))
    cx = W / 2
    # ------------------------------------------------------------------------------------- intro
    if t < 6.6:
        y0 = 430 if FMT == "yt" else 820
        title_card(base, t, 0.4, 6.2, y0, "M A T E R I A L S")
        k1 = window(t, 3.2, 6.2, 0.6)
        lines = (["Metals, carbon, oxides —", "and the proteins that touch them."])
        for j, line in enumerate(lines):
            text(lay, (cx, y0 + 230 + 52 * j), line, 40 if FMT == "yt" else 44, INK, "medium", k1, "mm")
    # -------------------------------------------------------------------------------------- build
    if 6.5 <= t < 17.3:
        a = 7.0
        tag(lay, t, a, "01", "BUILD", "Materials, atom by atom",
            "Leonardo builds nanoparticles, metal slabs, graphene and oxide surfaces")
        scene, sa = scene_at(t)
        if scene in D["mats"]:
            name, formula, model, rows = D["mats"][scene]
            k = ease((t - max(sa, a) - 0.1) / 0.45) * (1 - ease((t - (SCENES[scene][1] - 0.25)) / 0.25))
            x0, y0, x1, y1 = (1350, 360, 1840, 760) if FMT == "yt" else (60, 1250, 1020, 1660)
            card(lay, (x0, y0, x1, y1), k)
            text(lay, (x0 + 40, y0 + 52), name.upper(), 18, MUTED, "demi", k, "lm", spacing=3)
            text(lay, (x0 + 40, y0 + 120), formula, 64, GOLD if scene in ("np",) else INK, "demi", k, "lm")
            text(lay, (x0 + 40, y0 + 182), model, 22, (196, 210, 232), "regular", k, "lm")
            for j, (lab, val) in enumerate(rows):
                xx = x0 + 40 + j * ((x1 - x0 - 80) / 2)
                text(lay, (xx, y0 + 250), lab.upper(), 16, MUTED, "demi", k, "la", spacing=2)
                text(lay, (xx, y0 + 278), val, 36, TEAL, "demi", k, "la")
        provenance(lay, t, a, "build_material in Leonardo · Leonardo materials pack (python -m leonardo.replay)")
    # ----------------------------------------------------------------------------------- surfaces
    if 17.2 <= t < 36.6:
        a = 17.2
        tag(lay, t, a + 0.2, "02", "BIO-INTERFACES", "One protein, three surfaces",
            "Lysozyme (PDB 1AKI) in water on gold, graphene and silica · built and minimized by Leonardo")
        scene, sa = scene_at(t)
        x0, y0, x1, y1 = panel_box()
        k = ease((t - a - 0.8) / 0.7)
        card(lay, (x0, y0, x1, y1), k)
        text(lay, (x0 + 40, y0 + 46), "PROTEIN–SURFACE CONTACT ENERGY", 17, MUTED, "demi", k, "lm", spacing=2)
        text(lay, (x0 + 40, y0 + 74), "Lennard-Jones, kJ/mol (after minimization)", 17, MUTED, "regular", k, "lm")
        order = [("au", 17.2), ("gr", 24.0), ("si", 30.0)]
        vmax = max(abs(D["surf"][s]["lj"]) for s, _ in order)
        bar_x0, bar_x1 = x0 + (190 if FMT == "yt" else 250), x1 - 120
        for j, (s, start) in enumerate(order):
            S = D["surf"][s]
            kb = ease((t - start - 0.6) / 0.8) * k
            yy = y0 + 140 + j * 74
            active = scene == s
            col = {"au": GOLD, "gr": (180, 190, 210), "si": (255, 130, 120)}[s]
            text(lay, (x0 + 40, yy), S["label"], 26, INK if active else MUTED, "demi", max(kb, 0.35 * k), "lm")
            dr = ImageDraw.Draw(lay)
            dr.rounded_rectangle((bar_x0, yy - 13, bar_x1, yy + 13), 8, fill=(40, 60, 95, int(160 * k)))
            if kb > 0:
                L = (bar_x1 - bar_x0) * abs(S["lj"]) / vmax * kb
                dr.rounded_rectangle((bar_x0, yy - 13, bar_x0 + max(16, L), yy + 13), 8,
                                     fill=(*col, int((255 if active else 150) * kb)))
                text(lay, (bar_x1 + 16, yy), f"{S['lj'] * kb:.0f}".replace("-", "−"), 26, INK, "demi", kb, "lm")
        if scene in D["surf"]:
            S = D["surf"][scene]
            kc = ease((t - sa - 1.0) / 0.6) * k
            yy = y0 + 380
            text(lay, (x0 + 40, yy), "RESIDUES TOUCHING THE SURFACE", 16, MUTED, "demi", kc, "lm", spacing=2)
            text(lay, (x0 + 40, yy + 40), "  ·  ".join(S["contacts"]), 30, AMBER, "demi", kc, "lm")
            text(lay, (x0 + 40, yy + 86), f"closest approach {S['closest']:.2f} Å  ·  {S['atoms']:,} atoms with water",
                 19, (196, 210, 232), "regular", kc, "lm")
        provenance(lay, t, a, "build_protein_surface_system in Leonardo · INTERFACE-FF / Steele / ClayFF · "
                              "contact energy, not a binding free energy")
    # ----------------------------------------------------------------------------------------- qm
    if 36.4 <= t < 51.2:
        a = 36.4
        Q = D["qm"]
        tag(lay, t, a + 0.2, "03", "QUANTUM", "Electrons at the interface",
            f"{Q['atoms']}-atom cluster cut by Leonardo at Arg68 on gold · MOPAC PM7 in water")
        x0, y0, x1, y1 = panel_box()
        k = ease((t - a - 0.8) / 0.7)
        card(lay, (x0, y0, x1, y1), k)
        lumo = t >= a + (SCENES["qm"][1] - SCENES["qm"][0]) / 2
        # energy-level diagram: HOMO and LUMO lines, the shown one highlighted
        ly0, ly1 = y0 + 90, y0 + 330
        e_lo, e_hi = Q["homo"] - 1.0, Q["lumo"] + 1.0

        def ey(e):
            return ly1 - (e - e_lo) / (e_hi - e_lo) * (ly1 - ly0)

        dr = ImageDraw.Draw(lay)
        text(lay, (x0 + 40, y0 + 46), "FRONTIER ORBITALS  (PM7)", 17, MUTED, "demi", k, "lm", spacing=2)
        for nm, e, on in (("HOMO", Q["homo"], not lumo), ("LUMO", Q["lumo"], lumo)):
            yy = ey(e)
            col = TEAL if on else (90, 110, 150)
            dr.line((x0 + 60, yy, x0 + 230, yy), fill=(*col, int(255 * k)), width=6 if on else 3)
            text(lay, (x0 + 250, yy), f"{nm}  {e:.2f} eV".replace("-", "−"), 28 if on else 22,
                 INK if on else MUTED, "demi", k, "lm")
        ya, yb = ey(Q["homo"]), ey(Q["lumo"])
        dr.line((x0 + 145, yb + 8, x0 + 145, ya - 8), fill=(*MUTED, int(160 * k)), width=2)
        text(lay, (x0 + 160, (ya + yb) / 2), f"gap {Q['gap']:.2f} eV", 20, MUTED, "regular", k, "lm")
        # where the shown orbital lives
        L = Q["loc"]["lumo" if lumo else "homo"]
        yy = y0 + 400
        text(lay, (x0 + 40, yy), ("LUMO" if lumo else "HOMO") + " LIVES ON", 16, MUTED, "demi", k, "lm", spacing=2)
        parts = [("gold", L["Au"], GOLD), ("Arg68 + residues", 1 - L["Au"], TEAL)]
        bx0, bx1 = x0 + 40, x1 - 40
        x = bx0
        for lab, frac, col in parts:
            w_ = (bx1 - bx0) * frac
            if w_ > 2:
                dr.rectangle((x, yy + 26, x + w_, yy + 50), fill=(*col, int(230 * k)))
            x += w_
        text(lay, (bx0, yy + 76), f"gold {100 * L['Au']:.0f} %", 24, GOLD, "demi", k, "lm")
        text(lay, (bx1, yy + 76), f"protein {100 * (1 - L['Au']):.0f} %", 24, TEAL, "demi", k, "rm")
        provenance(lay, t, a, f"cut_qm_cluster ({Q['links']} hydrogen caps) + run_mopac + show_orbitals in Leonardo · "
                              "teal/orange: orbital phases")
    # -------------------------------------------------------------------------------------- outro
    if t >= 51.0:
        y0 = 400 if FMT == "yt" else 760
        title_card(base, t, 51.4, 60.5, y0, "M A T E R I A L S")
        k = window(t, 53.0, 60.5, 0.7)
        lines = ["Every result you saw came from a real", "calculation run with Leonardo."] if FMT == "ig" else \
            ["Every result you saw came from a real calculation run with Leonardo."]
        for j, line in enumerate(lines):
            text(lay, (cx, y0 + 200 + 48 * j), line, 34, INK, "medium", k, "mm")
        k2 = window(t, 54.0, 60.5, 0.7)
        yy = y0 + 200 + 48 * len(lines) + 50
        text(lay, (cx, yy), "github.com/juvenalyosa/Leonardo", 28, TEAL, "demi", k2, "mm")
        text(lay, (cx, yy + 44), "juvenal.yosa@gmail.com", 22, MUTED, "regular", k2, "mm")
        k3 = window(t, 54.6, 60.5, 0.7)
        foot = ["Built using PyMOL™* · Leonardo is an independent project, not a Schrodinger product",
                "* PyMOL is a trademark of Schrodinger, LLC.",
                "Music: “Ouroboros” / “Exhilarate” by Kevin MacLeod (incompetech.com), CC BY 4.0"]
        if FMT == "ig":
            foot = [ln for f in foot for ln in wrap(f, 58)]
        for j, line in enumerate(foot):
            text(lay, (cx, H - 40 - 28 * (len(foot) - j)), line, 16 if FMT == "yt" else 19, MUTED, "regular", k3, "mm")
    base.alpha_composite(lay)


def compose(i, D):
    t = i / FPS
    scene, a = scene_at(t)
    j = i - int(round(a * FPS))
    if scene == "outro":
        img = mol_frame("np", max(0, 150 - int(j * 0.7)))
    else:
        img = mol_frame(scene, j)
    if t < 7.0:
        k = ease((t - 4.8) / 2.2)
        blur = 6 * (1 - k)
        if blur > 0.2:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        img = Image.blend(Image.new("RGBA", (W, H), (3, 6, 13, 255)), img, 0.42 + 0.58 * k)
    if scene == "outro":
        img = Image.blend(Image.new("RGBA", (W, H), (3, 6, 13, 255)), img.filter(ImageFilter.GaussianBlur(5)), 0.36)
    for _n, (sa, _sb) in SCENES.items():
        if sa > 0 and abs(t - sa) < 0.3:
            img = Image.blend(img, Image.new("RGBA", (W, H), (3, 6, 13, 255)), 0.8 * ease(1 - abs(t - sa) / 0.3))
    rgb = tc.grade(img)
    # orbital switch (HOMO -> LUMO) under a light pulse
    mid = SCENES["qm"][0] + (SCENES["qm"][1] - SCENES["qm"][0]) / 2
    if abs(t - mid) < 0.5:
        rgb = Image.blend(rgb, Image.new("RGB", (W, H), (200, 228, 255)), 0.6 * ease(1 - abs(t - mid) / 0.5))
    base = rgb.convert("RGBA")
    if 6.8 <= t < 51.0:
        if FMT == "yt":
            sc = tc.scrim()
        else:
            sc = vertical_scrim()
        k = ease((t - 6.8) / 0.8) * (1 - ease((t - 50.4) / 0.6))
        if k < 1:
            sc = sc.copy()
            sc.putalpha(sc.getchannel("A").point(lambda v: int(v * k)))
        base.alpha_composite(sc)
    overlay(base, t, i, D)
    out = tc.add_grain(base.convert("RGB"), i)
    if t < 0.6:
        out = Image.blend(Image.new("RGB", (W, H)), out, ease(t / 0.6))
    if t > TOTAL - 0.9:
        out = Image.blend(out, Image.new("RGB", (W, H)), ease((t - (TOTAL - 0.9)) / 0.9))
    return out


_vscrim = None


def vertical_scrim():
    global _vscrim
    if _vscrim is None:
        yy = np.mgrid[0:H, 0:W][0]
        top = np.clip(1 - yy / 380.0, 0, 1) ** 1.5
        bottom = np.clip((yy - 1050) / 300.0, 0, 1) ** 1.2
        a = np.clip(0.85 * top + 0.8 * bottom, 0, 0.88)
        arr = np.zeros((H, W, 4), np.uint8)
        arr[..., :3] = (3, 8, 18)
        arr[..., 3] = (255 * a).astype(np.uint8)
        _vscrim = Image.fromarray(arr, "RGBA")
    return _vscrim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fmt", choices=("yt", "ig"))
    ap.add_argument("--only", type=float, nargs="*")
    a = ap.parse_args()
    configure(a.fmt)
    D = load()
    if a.only:
        for t in a.only:
            p = ART / "stills" / f"{a.fmt}_{t:05.2f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            compose(int(round(t * FPS)), D).save(p)
            print(p)
        return
    out = ART / f"leonardo_materials_{'youtube_16x9' if a.fmt == 'yt' else 'instagram_9x16'}.mp4"
    n = int(TOTAL * FPS)
    proc = subprocess.Popen([tc.FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(FPS), "-i", "-", "-ss", str(MUSIC_START), "-t", str(TOTAL), "-i", str(MUSIC),
                             "-filter_complex", f"[1:a]afade=t=in:st=0:d=1.5,afade=t=out:st={TOTAL - 3.5}:d=3.5,"
                             "volume=0.95[a]", "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "slow",
                             "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags",
                             "+faststart", "-shortest", str(out)], stdin=subprocess.PIPE)
    for i in range(n):
        proc.stdin.write(compose(i, D).tobytes())
        if i % 96 == 0:
            print(f"{i / FPS:5.1f} s", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
