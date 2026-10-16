#!/usr/bin/env python3
"""
Compositing stage of the 60 s Leonardo trailer: the PyMOL frames (scripts/trailer_molecules.py) get a
cinematic grade (bloom, vignette, grain), animated titles, live data panels drawn from the real
calculation outputs in artifacts/trailer_2026/data, and the music; ffmpeg writes the MP4.

    python scripts/trailer_compose.py [--only SECONDS ...] [--out PATH]

Every number on screen is read from those outputs (docking, MD, QM/MM path, disulfide design, pKa);
nothing is typed in by hand except labels.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "trailer_2026"
DATA, FRAMES = ART / "data", ART / "frames"
W, H, FPS, TOTAL = 1920, 1080, 24, 60.0
SCENES = {"docking": (0.0, 15.0), "md": (15.0, 25.0), "qmmm": (25.0, 36.0), "design": (36.0, 44.0),
          "pka": (44.0, 52.0), "outro": (52.0, 60.0)}
FFMPEG = "/opt/homebrew/bin/ffmpeg"
MUSIC = ART / "audio" / "killers.wav"
MUSIC_START = 150.0          # s into "Killers": quiet build, then the climax lands on the last scenes

INK = (238, 244, 252)
MUTED = (150, 168, 196)
TEAL = (43, 227, 200)
AMBER = (255, 179, 71)
GREEN = (125, 255, 155)
RED = (255, 107, 107)
BLUE = (111, 168, 255)
CARD = (10, 22, 44)

FONT = "/System/Library/Fonts/Avenir Next.ttc"
FACES = {"bold": 0, "demi": 2, "medium": 5, "regular": 7, "light": 10}
MONO = "/System/Library/Fonts/SFNSMono.ttf"
_fonts: dict = {}


def font(size, face="regular"):
    key = (size, face)
    if key not in _fonts:
        _fonts[key] = (ImageFont.truetype(MONO, size) if face == "mono"
                       else ImageFont.truetype(FONT, size, index=FACES[face]))
    return _fonts[key]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp(x)
    return 1 - (1 - x) ** 3


def window(t, a, b, fade=0.45):
    """Opacity of an element shown from a to b with fades."""
    return ease((t - a) / fade) * (1 - ease((t - (b - fade)) / fade))


# ------------------------------------------------------------------------------------------- data
def load_data():
    d = {}
    dock = json.load(open(DATA / "docking_results.json"))["2"]
    d["dock"] = {"rmsd": float(dock["shown_pose"]["rmsd_A"]), "score": float(dock["shown_pose"]["score_kcal_mol"]),
                 "repeats": len(dock["repeats"]),
                 "ok": sum(1 for r in dock["repeats"] if float(r["top1_rmsd_A"]) <= 2.0)}
    md = np.load(DATA / "md_aligned.npz")
    d["md"] = {"analysis": md["analysis"], "prov": json.load(open(DATA / "md_provenance.json"))}
    q = np.load(DATA / "qmmm_path.npz")
    d["qmmm"] = {"s": q["s"], "e_kj": q["relative"] * 4.184, "ts": int(q["ts_index"]), "d_oc": q["d_oc"],
                 "d_ccl": q["d_ccl"], "imag": abs(float(q["imaginary_frequency"])),
                 "prov": json.load(open(DATA / "qmmm_provenance.json"))}
    d["design"] = json.load(open(DATA / "gb1_design_summary.json"))
    pk = json.load(open(DATA / "pka_levodopa.json"))
    charges = sorted({s["charge"] for c in pk["curves"] for s in c["states"]}, reverse=True)
    ph = np.array([c["ph"] for c in pk["curves"]])
    pops = {q_: np.array([sum(s["pop"] for s in c["states"] if s["charge"] == q_) for c in pk["curves"]])
            for q_ in charges}
    d["pka"] = {"ph": ph, "pops": pops, "macro": pk["result"]["macro_pkas"], "sites": pk["result"]["sites"],
                "level": pk["level"]}
    return d


def anchors(scene):
    p = FRAMES / scene / "anchors.json"
    return json.load(open(p)) if p.exists() else {}


# ------------------------------------------------------------------------------------- drawing kit
SUPERSCRIPTS = set("⁻⁺¹²³⁰₀₁₂₃₄")
SF = "/System/Library/Fonts/SFNS.ttf"


def text(layer, xy, s, size, fill=INK, face="regular", alpha=1.0, anchor="la", spacing=0):
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(layer)
    col = (*fill, int(255 * alpha))
    if SUPERSCRIPTS & set(s):          # Avenir has no super/subscript glyphs: the system font has them
        key = (size, "sf")
        if key not in _fonts:
            _fonts[key] = ImageFont.truetype(SF, size)
        d.text(xy, s, font=_fonts[key], fill=col, anchor=anchor)
        return
    if spacing:
        x, y = xy
        f = font(size, face)
        if anchor[0] == "m":
            total = sum(d.textlength(ch, font=f) for ch in s) + spacing * (len(s) - 1)
            x -= total / 2
        for ch in s:
            d.text((x, y), ch, font=f, fill=col, anchor="l" + anchor[1])
            x += d.textlength(ch, font=f) + spacing
        return
    d.text(xy, s, font=font(size, face), fill=col, anchor=anchor)


def glow_text(base, xy, s, size, fill, face="demi", alpha=1.0, anchor="la", spacing=0, radius=14):
    layer = Image.new("RGBA", base.size)
    text(layer, xy, s, size, fill, face, alpha, anchor, spacing)
    halo = layer.filter(ImageFilter.GaussianBlur(radius))
    base.alpha_composite(halo)
    base.alpha_composite(halo)
    base.alpha_composite(layer)


def card(layer, box, alpha=1.0):
    """Frosted dark glass panel with a hairline border."""
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=18, fill=(*CARD, int(178 * alpha)), outline=(120, 160, 220, int(70 * alpha)),
                        width=1)


def scene_tag(layer, t, a, num, label, title, subtitle):
    """Top-left: number + label, big title, subtitle, all sliding in."""
    k = window(t, a, a + 99, 0.6)
    if k <= 0:
        return
    dx = 40 * (1 - ease_out((t - a) / 0.8))
    d = ImageDraw.Draw(layer)
    d.line((96 - dx, 112, 96 - dx + 56 * ease_out((t - a) / 0.6), 112), fill=(*TEAL, int(255 * k)), width=3)
    text(layer, (168 - dx, 112), f"{num}  ·  {label}", 22, TEAL, "demi", k, "lm", spacing=3)
    shade = Image.new("RGBA", layer.size)
    text(shade, (94 - dx, 150), title, 58, (0, 0, 0), "demi", 0.8 * ease((t - a - 0.15) / 0.6), "la")
    text(shade, (96 - dx, 228), subtitle, 25, (0, 0, 0), "demi", 0.9 * ease((t - a - 0.35) / 0.6), "la")
    layer.alpha_composite(shade.filter(ImageFilter.GaussianBlur(7)))
    text(layer, (94 - dx, 150), title, 58, INK, "demi", ease((t - a - 0.15) / 0.6), "la")
    text(layer, (96 - dx, 228), subtitle, 25, (196, 210, 232), "regular", ease((t - a - 0.35) / 0.6), "la")


def provenance(layer, t, a, s):
    k = ease((t - a - 0.8) / 0.8)
    text(layer, (96, H - 70), "REAL CALCULATION", 16, TEAL, "demi", 0.9 * k, "lm", spacing=3)
    text(layer, (318, H - 70), s, 17, (196, 210, 232), "regular", 0.9 * k, "lm")


_scrim = None


def scrim():
    """Dark gradients at the top-left (titles) and bottom (provenance) keep text readable on bright scenes."""
    global _scrim
    if _scrim is None:
        yy, xx = np.mgrid[0:H, 0:W]
        top = np.clip(1 - yy / 330.0, 0, 1) ** 1.6 * np.clip(1.25 - xx / 1500.0, 0, 1)
        bottom = np.clip((yy - (H - 190)) / 190.0, 0, 1) ** 1.4
        a = np.clip(0.78 * top + 0.7 * bottom, 0, 0.85)
        arr = np.zeros((H, W, 4), np.uint8)
        arr[..., :3] = (3, 8, 18)
        arr[..., 3] = (255 * a).astype(np.uint8)
        _scrim = Image.fromarray(arr, "RGBA")
    return _scrim


def leader(layer, pt, label, k, color=INK, dx=None, dy=None, sub=None):
    """Label with a thin leader line to a screen point; without dx/dy it goes where there is room (away from
    the title block at the top left and from the data panel on the right)."""
    if k <= 0.01 or pt is None:
        return
    if dx is None:
        dx = -150 if pt[0] > 760 else 150
    if dy is None:
        dy = 110 if pt[1] < 430 else -100
    d = ImageDraw.Draw(layer)
    x, y = pt
    ex, ey = x + dx, y + dy
    d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(*color, int(230 * k)), width=2)
    d.line((x + 4 * np.sign(dx), y + 4 * np.sign(dy), ex, ey), fill=(*color, int(200 * k)), width=2)
    d.line((ex, ey, ex + 26 * np.sign(dx), ey), fill=(*color, int(200 * k)), width=2)
    ax = ex + 34 * np.sign(dx)
    shade = Image.new("RGBA", layer.size)
    text(shade, (ax, ey), label, 24, (0, 0, 0), "demi", k, "lm" if dx > 0 else "rm")
    if sub:
        text(shade, (ax, ey + 28), sub, 19, (0, 0, 0), "demi", k, "lm" if dx > 0 else "rm")
    layer.alpha_composite(shade.filter(ImageFilter.GaussianBlur(5)))
    layer.alpha_composite(shade.filter(ImageFilter.GaussianBlur(5)))
    text(layer, (ax, ey), label, 24, color, "demi", k, "lm" if dx > 0 else "rm")
    if sub:
        text(layer, (ax, ey + 28), sub, 19, (196, 210, 232), "regular", k, "lm" if dx > 0 else "rm")


def metric(layer, x, y, label, value, unit, k, color=INK):
    text(layer, (x, y), label.upper(), 15, MUTED, "demi", k, "la", spacing=2)
    text(layer, (x, y + 26), value, 46, color, "demi", k, "la")
    if unit:
        w = ImageDraw.Draw(layer).textlength(value, font=font(46, "demi"))
        text(layer, (x + w + 10, y + 52), unit, 22, MUTED, "medium", k, "la")


# ------------------------------------------------------------------------------------- plots (PIL)
def plot_axes(layer, box, xr, yr, k, xlabel, ylabel, yticks=(), xticks=()):
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = box
    a = int(255 * k)
    for v in yticks:
        yy = y1 - (v - yr[0]) / (yr[1] - yr[0]) * (y1 - y0)
        d.line((x0, yy, x1, yy), fill=(120, 160, 220, int(40 * k)), width=1)
        text(layer, (x0 - 12, yy), f"{v:g}", 16, MUTED, "regular", k, "rm")
    for v in xticks:
        xx = x0 + (v - xr[0]) / (xr[1] - xr[0]) * (x1 - x0)
        text(layer, (xx, y1 + 16), f"{v:g}", 16, MUTED, "regular", k, "mm")
    d.line((x0, y1, x1, y1), fill=(170, 195, 230, a), width=1)
    text(layer, ((x0 + x1) / 2, y1 + 42), xlabel, 17, MUTED, "medium", k, "mm")
    text(layer, (x0, y0 - 34), ylabel, 17, MUTED, "medium", k, "lm")


def polyline(layer, pts, color, k, width=3, glow=True):
    if len(pts) < 2 or k <= 0.01:
        return
    big = Image.new("RGBA", (layer.size[0] * 2, layer.size[1] * 2))
    d = ImageDraw.Draw(big)
    d.line([(x * 2, y * 2) for x, y in pts], fill=(*color, int(255 * k)), width=width * 2, joint="curve")
    small = big.resize(layer.size, Image.LANCZOS)
    if glow:
        layer.alpha_composite(small.filter(ImageFilter.GaussianBlur(6)))
    layer.alpha_composite(small)


def to_px(box, xr, yr, x, y):
    x0, y0, x1, y1 = box
    return (x0 + (x - xr[0]) / (xr[1] - xr[0]) * (x1 - x0), y1 - (y - yr[0]) / (yr[1] - yr[0]) * (y1 - y0))


# ----------------------------------------------------------------------------------------- grade
_vignette = None
_grain = None


def grade(img, bloom=0.16):
    global _vignette, _grain
    if _vignette is None:
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        _vignette = Image.fromarray((255 * np.clip(1.08 - 0.42 * r ** 1.6, 0.35, 1)).astype(np.uint8), "L")
        rng = np.random.default_rng(3)
        _grain = [Image.fromarray(rng.normal(128, 4.5, (H // 2, W // 2)).clip(0, 255).astype(np.uint8), "L")
                  .resize((W, H)) for _ in range(6)]
    rgb = img.convert("RGB")
    small = rgb.resize((W // 4, H // 4), Image.BILINEAR)
    bright = small.point(lambda v: max(0, v - 160) * 3)
    halo = bright.filter(ImageFilter.GaussianBlur(10)).resize((W, H), Image.BILINEAR)
    rgb = ImageChops.add(rgb, Image.blend(Image.new("RGB", (W, H)), halo, bloom))
    rgb = ImageChops.multiply(rgb, Image.merge("RGB", [_vignette] * 3))
    return rgb


def add_grain(rgb, i):
    g = _grain[i % len(_grain)]
    return ImageChops.add(rgb, Image.merge("RGB", [g] * 3), scale=1.0, offset=-128)


# ---------------------------------------------------------------------------------------- frames
def mol_frame(scene, i):
    n = len(list((FRAMES / scene).glob("mol_*.png")))
    p = FRAMES / scene / f"mol_{min(i, max(0, n - 1)):04d}.png"
    if p.exists():
        return Image.open(p).convert("RGBA")
    return Image.new("RGBA", (W, H), (3, 6, 13, 255))


def scene_at(t):
    for name, (a, b) in SCENES.items():
        if a <= t < b:
            return name, a
    return "outro", SCENES["outro"][0]


LOGO = None


def logo(size):
    global LOGO
    if LOGO is None:
        LOGO = Image.open(ROOT / "assets" / "logo.png").convert("RGBA")
    return LOGO.resize((size, size), Image.LANCZOS)


def title_block(base, t, a, end, y=430, small=False):
    """Logo + LEONARDO + subtitle (intro and outro)."""
    k_logo = window(t, a, end, 0.7)
    if k_logo > 0:
        s = int((150 if not small else 120) * (0.86 + 0.14 * ease_out((t - a) / 1.2)))
        lg = logo(s)
        alpha = lg.getchannel("A").point(lambda v: int(v * k_logo))
        lg.putalpha(alpha)
        halo = Image.new("RGBA", base.size)
        halo.alpha_composite(lg, (W // 2 - s // 2, y - 190 - s // 2 + 60))
        base.alpha_composite(halo.filter(ImageFilter.GaussianBlur(28)))
        base.alpha_composite(halo)
    k = window(t, a + 0.5, end, 0.7)
    spacing = int(26 - 12 * ease_out((t - a - 0.5) / 1.6))
    glow_text(base, (W / 2, y), "LEONARDO", 112, INK, "demi", k, "mm", spacing=spacing, radius=18)
    sub = Image.new("RGBA", base.size)
    text(sub, (W / 2, y + 82), "MOLECULAR  MODELING  SUITE", 26, TEAL, "demi", window(t, a + 1.1, end, 0.7),
         "mm", spacing=6)
    base.alpha_composite(sub)


def overlay(base, t, i, D, A):
    """Everything drawn over the graded molecular frame at global time t."""
    lay = Image.new("RGBA", (W, H))
    # ---------------------------------------------------------------- intro (over the far docking shot)
    if t < 6.6:
        title_block(base, t, 0.4, 6.2)
        k1 = window(t, 3.2, 6.2, 0.6)
        text(lay, (W / 2, 680), "Real calculations. One window.", 40, INK, "medium", k1, "mm")
        text(lay, (W / 2, 740), "DOCKING   ·   DYNAMICS   ·   QM/MM   ·   DESIGN   ·   pKa", 22, MUTED, "demi",
             window(t, 3.8, 6.2, 0.6), "mm", spacing=2)
    # ------------------------------------------------------------------------------------- docking
    if 6.5 <= t < 15.2:
        a = 7.0
        scene_tag(lay, t, a, "01", "DOCKING", "Finding where a drug binds",
                  "smina inside Leonardo  ·  trypsin + benzamidine (PDB 3PTB)")
        k = ease((t - 10.6) / 0.7)
        if k > 0:
            card(lay, (1390, 300, 1830, 780), k)
            dd = D["dock"]
            metric(lay, 1430, 340, "pose vs crystal (RMSD)", f"{dd['rmsd'] * clamp((t - 10.6) / 1.2) ** 0.4:.2f}"
                   if t < 11.8 else f"{dd['rmsd']:.2f}", "Å", k, AMBER)
            metric(lay, 1430, 480, "docking score", f"{dd['score']:.2f}".replace("-", "−"), "kcal/mol",
                   ease((t - 11.0) / 0.7))
            metric(lay, 1430, 620, "repeats on target", f"{dd['ok']} / {dd['repeats']}", "", ease((t - 11.4) / 0.7),
                   TEAL)
        kl = ease((t - 12.4) / 0.6)
        if kl > 0:
            d = ImageDraw.Draw(lay)
            d.ellipse((1430, 724, 1446, 740), fill=(*AMBER, int(255 * kl)))
            text(lay, (1456, 732), "docked by Leonardo", 19, INK, "medium", kl, "lm")
            d.ellipse((1680, 724, 1696, 740), outline=(*INK, int(255 * kl)), width=2)
            text(lay, (1706, 732), "crystal", 19, INK, "medium", kl, "lm")
        provenance(lay, t, a, f"Leonardo replay pack 01 · smina {D['dock']['repeats']} seeded repeats · pH 7.4")
    # ------------------------------------------------------------------------------------------- md
    if 15.0 <= t < 25.2:
        a = 15.0
        an = D["md"]["analysis"]
        scene_tag(lay, t, a + 0.3, "02", "MOLECULAR DYNAMICS", "A protein in motion",
                  "OpenMM in Leonardo  ·  Amber14 / GB-OBC2  ·  100 ps  ·  500 frames")
        anc = A["md"].get(str(i - int(round(a * FPS))))
        frame = anc["frame"] if anc else 0
        k = ease((t - a - 0.8) / 0.7)
        if k > 0:
            card(lay, (1300, 300, 1840, 860), k)
            box = (1370, 380, 1800, 560)
            plot_axes(lay, box, (0, 100), (0, 1.4), k, "time (ps)", "Cα RMSD (Å)", yticks=(0, 0.5, 1.0),
                      xticks=(0, 50, 100))
            pts = [to_px(box, (0, 100), (0, 1.4), an[j, 1], an[j, 2]) for j in range(0, frame + 1)]
            polyline(lay, pts, TEAL, k)
            if pts:
                x, y = pts[-1]
                ImageDraw.Draw(lay).ellipse((x - 6, y - 6, x + 6, y + 6), fill=(*TEAL, int(255 * k)))
            box2 = (1370, 660, 1800, 790)
            plot_axes(lay, box2, (0, 100), (2.5, 6.0), k, "time (ps)", "Ser195 OG – His57 NE2 (Å)",
                      yticks=(3, 4, 5), xticks=())
            pts2 = [to_px(box2, (0, 100), (2.5, 6.0), an[j, 1], min(6.0, an[j, 3])) for j in range(0, frame + 1)]
            polyline(lay, pts2, AMBER, k)
            text(lay, (1800, 330), f"t = {an[frame, 1]:5.1f} ps", 28, INK, "mono", k, "ra")
        if anc:
            kl = ease((t - a - 1.6) / 0.6) * (1 - ease((t - 24.5) / 0.5))
            p1, p2 = anc["ser195"], anc["his57"]
            up1 = p1[1] < p2[1]
            leader(lay, p1, "Ser195", kl, AMBER, dx=-150, dy=-90 if up1 and p1[1] > 380 else 100)
            leader(lay, p2, "His57", kl, AMBER, dx=-150, dy=100 if up1 or p2[1] < 380 else -90)
        provenance(lay, t, a, f"Leonardo replay pack 03 · final Cα RMSD {D['md']['prov']['ca_rmsd_final_A']:.2f} Å "
                              f"· catalytic Ser–His {D['md']['prov']['ser195_his57_mean_A']:.2f} Å (mean)")
    # ----------------------------------------------------------------------------------------- qmmm
    if 25.0 <= t < 36.2:
        a = 25.0
        Q = D["qmmm"]
        scene_tag(lay, t, a + 0.3, "03", "QM/MM REACTION", "Bonds forming, bonds breaking",
                  "Reaction path in the enzyme  ·  PM7 (MOPAC) QM region  ·  haloalkane dehalogenase DhlA")
        anc = A["qmmm"].get(str(i - int(round(a * FPS))))
        k = ease((t - a - 0.8) / 0.7)
        if k > 0 and anc:
            j = int(anc["path_index"])
            card(lay, (1300, 300, 1840, 880), k)
            box = (1380, 380, 1800, 640)
            e = Q["e_kj"]
            yr = (-115, 50)
            plot_axes(lay, box, (0, 1), yr, k, "reaction coordinate", "energy (kJ/mol)", yticks=(-100, -50, 0),
                      xticks=())
            pts = [to_px(box, (0, 1), yr, Q["s"][m], e[m]) for m in range(len(e))]
            polyline(lay, pts, (70, 100, 150), k, width=2, glow=False)
            polyline(lay, pts[: j + 1], TEAL, k)
            x, y = pts[j]
            ImageDraw.Draw(lay).ellipse((x - 8, y - 8, x + 8, y + 8), fill=(*INK, int(255 * k)))
            tsx, tsy = pts[Q["ts"]]
            kt = ease((j - Q["ts"] + 6) / 8)
            text(lay, (tsx + 14, tsy - 18), f"TS  +{Q['prov']['barrier_kj_mol']:.1f} kJ/mol", 22, AMBER, "demi", kt,
                 "lm")
            text(lay, (tsx + 14, tsy + 10), f"imaginary mode {Q['imag']:.0f}i cm⁻¹", 17, MUTED, "regular", kt, "lm")
            kp = ease((j - len(e) + 30) / 20)
            px, py = pts[-1]
            text(lay, (px, py - 26), f"{Q['prov']['reaction_energy_kj_mol']:.1f} kJ/mol".replace("-", "−"), 22, TEAL,
                 "demi", kp, "rm")
            metric(lay, 1380, 700, "O–C  forming", f"{Q['d_oc'][j]:.2f}", "Å", k, AMBER)
            metric(lay, 1620, 700, "C–Cl  breaking", f"{Q['d_ccl'][j]:.2f}", "Å", k, GREEN)
        if anc:
            kl = ease((t - a - 1.8) / 0.6) * (1 - ease((t - 35.5) / 0.5))
            leader(lay, anc["o"], "Asp124", kl, RED, dx=-140, dy=140, sub="nucleophile")
            leader(lay, anc["cl"], "Cl⁻", kl, GREEN, dx=-60, dy=170, sub="leaving group")
        provenance(lay, t, a, "Leonardo replay pack 02 · run_qmmm reaction_path: NEB + transition state + IRC")
    # --------------------------------------------------------------------------------------- design
    if 36.0 <= t < 44.2:
        a = 36.0
        G = D["design"]
        scene_tag(lay, t, a + 0.3, "04", "PROTEIN DESIGN", "Locking a protein together",
                  "Disulfide design in Leonardo  ·  protein G B1 domain (PDB 1PGA)")
        k = ease((t - a - 1.2) / 0.7)
        if k > 0:
            card(lay, (1360, 300, 1840, 820), k)
            metric(lay, 1400, 340, "designed mutation", G["design"].replace(" ", ""), "", k, INK)
            metric(lay, 1400, 470, "S–S bond length", f"{G['geometry']['SS_A']:.2f}", "Å", ease((t - a - 1.5) / 0.7),
                   (255, 210, 63))
            red = G["pair_relative_rms_reduction_percent"] * ease((t - a - 2.0) / 1.6)
            metric(lay, 1400, 600, "pair fluctuation", f"−{red:.0f}", "%", ease((t - a - 1.8) / 0.7), TEAL)
            d = ImageDraw.Draw(lay)
            d.rounded_rectangle((1400, 730, 1800, 744), 7, fill=(40, 60, 95, int(255 * k)))
            d.rounded_rectangle((1400, 730, 1400 + 4 * red, 744), 7, fill=(*TEAL, int(255 * k)))
            text(lay, (1400, 772), "elastic network model, original vs design", 17, MUTED, "regular", k, "lm")
        anc = A["design"].get(str(i - int(round(a * FPS))))
        if anc:
            kl = ease((t - a - 3.0) / 0.6) * (1 - ease((t - 43.5) / 0.5))
            leader(lay, anc["ss"], "New disulfide bond", kl, (255, 210, 63), dx=-150, dy=-130,
                   sub="Cys10 – Cys56")
        provenance(lay, t, a, "Leonardo replay pack 06 · design_disulfides picks K10C/E56C")
    # ------------------------------------------------------------------------------------------ pka
    if 44.0 <= t < 52.2:
        a = 44.0
        P = D["pka"]
        scene_tag(lay, t, a + 0.3, "05", "pKa & PROTONATION", "Charge, predicted",
                  "PM7 / COSMO, calibrated  ·  levodopa  ·  4 sites, 16 microstates")
        k = ease((t - a - 0.8) / 0.7)
        if k > 0:
            card(lay, (1240, 300, 1840, 860), k)
            box = (1310, 380, 1800, 700)
            plot_axes(lay, box, (0, 14), (0, 1), k, "pH", "population by net charge", yticks=(0, 0.5, 1),
                      xticks=(0, 2, 4, 6, 8, 10, 12, 14))
            reveal = clamp((t - a - 1.0) / 3.2)
            colors = [BLUE, TEAL, AMBER, RED, (200, 150, 255), (150, 150, 150)]
            ph = P["ph"]
            m = int(len(ph) * reveal)
            for (q, pop), col in zip(P["pops"].items(), colors):
                pts = [to_px(box, (0, 14), (0, 1), ph[j], pop[j]) for j in range(m)]
                polyline(lay, pts, col, k)
                if m > 5:
                    jmax = int(np.argmax(pop))
                    if jmax < m:
                        x, y = to_px(box, (0, 14), (0, 1), ph[jmax], pop[jmax])
                        x += 24 if jmax < 4 else (-24 if jmax > len(ph) - 5 else 0)
                        text(lay, (x, y - 18), "0" if q == 0 else f"{q:+d}", 19, col, "demi", k, "mm")
            k74 = ease((t - a - 4.4) / 0.6)
            if k74 > 0:
                x, _ = to_px(box, (0, 14), (0, 1), 7.4, 0)
                ImageDraw.Draw(lay).line((x, 380, x, 700), fill=(*INK, int(150 * k74)), width=2)
                j = int(np.argmin(abs(ph - 7.4)))
                p0 = P["pops"].get(0, np.zeros_like(ph))[j]
                text(lay, (x - 10, 690), f"pH 7.4 · {100 * p0:.0f} % zwitterion", 18, INK, "demi", k74, "rb")
            macro = "   ".join(f"{v:.2f}" for v in P["macro"])
            text(lay, (1310, 770), "MACROSCOPIC pKa", 15, MUTED, "demi", ease((t - a - 1.6) / 0.6), "la", spacing=2)
            text(lay, (1310, 798), macro, 34, INK, "demi", ease((t - a - 1.6) / 0.6), "la")
        anc = A["pka"].get(str(i - int(round(a * FPS))))
        sites = {int(s["atom"]): s for s in P["sites"]}
        if anc:
            kl = ease((t - a - 2.0) / 0.6) * (1 - ease((t - 51.5) / 0.5))
            pts = np.array(list(anc["atoms"].values()), float)
            mid = pts.mean(axis=0)
            labels = []
            for atom, s in sites.items():
                pt = anc["atoms"].get(str(atom))
                if pt is None:
                    continue
                name = {"amine1": "NH₃⁺", "carboxylic_aliphatic": "COO⁻", "phenol": "OH"}.get(s["class"], s["class"])
                v = np.array(pt) - mid
                v = v / (np.linalg.norm(v) or 1.0)
                # radial: out from the molecule's centre, so the rotating sites do not cross each other
                side = 1 if v[0] >= 0 else -1
                labels.append([pt, f"{name}  pKa {s['pka']:.1f}", side, pt[1] + 120 * v[1]])
            # keep clear of the title block, then space labels on the same side at least 50 px apart
            for side in (-1, 1):
                group = sorted((lb for lb in labels if lb[2] == side), key=lambda lb: lb[3])
                y_prev = 330.0 - 50
                for lb in group:
                    lb[3] = max(lb[3], y_prev + 50, 330.0)
                    y_prev = lb[3]
            for pt, label, side, ey in labels:
                leader(lay, pt, label, kl, INK, dx=side * 170, dy=ey - pt[1])
        provenance(lay, t, a, f"computed for this film in Leonardo · {P['level'].get('level', 'PM7/COSMO')} · "
                              f"calibration {P['level'].get('calibration_version', '')}")
    # ---------------------------------------------------------------------------------------- outro
    if t >= 52.0:
        title_block(base, t, 52.4, 60.5, y=400)
        k = window(t, 54.0, 60.5, 0.7)
        text(lay, (W / 2, 560), "Every result you saw came from a real calculation run with Leonardo.", 34, INK, "medium", k, "mm")
        text(lay, (W / 2, 612), "Reproducible with one command  ·  open source (MIT)", 24, MUTED, "regular",
             window(t, 54.6, 60.5, 0.7), "mm")
        k2 = window(t, 55.4, 60.5, 0.7)
        text(lay, (W / 2, 700), "github.com/juvenalyosa/Leonardo", 26, TEAL, "demi", k2, "mm")
        text(lay, (W / 2, 742), "juvenal.yosa@gmail.com", 20, MUTED, "regular", k2, "mm")
        k3 = window(t, 55.8, 60.5, 0.7)
        text(lay, (W / 2, H - 120), "Built using PyMOL™*  ·  Leonardo is an independent project, not a Schrodinger "
             "product", 17, MUTED, "regular", k3, "mm")
        text(lay, (W / 2, H - 92), "* PyMOL is a trademark of Schrodinger, LLC.   ·   Music: “Killers” by Kevin "
             "MacLeod (incompetech.com), CC BY 4.0", 15, MUTED, "regular", k3, "mm")
    base.alpha_composite(lay)


def compose(i, D, A):
    t = i / FPS
    scene, a = scene_at(t)
    j = i - int(round(a * FPS))
    if scene == "outro":
        # bookend: the far orbit of the opening shot, played backwards under the closing card
        img = mol_frame("docking", max(0, 150 - int(j * 0.75)))
    else:
        img = mol_frame(scene, j)
    # intro: the far shot sits dimmed and soft under the title; outro background is dimmed too
    if t < 7.0:
        k = ease((t - 4.8) / 2.2)
        dim = 0.42 + 0.58 * k
        blur = 6 * (1 - k)
        if blur > 0.2:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        img = Image.blend(Image.new("RGBA", (W, H), (3, 6, 13, 255)), img, dim)
    if scene == "outro":
        img = Image.blend(Image.new("RGBA", (W, H), (3, 6, 13, 255)), img.filter(ImageFilter.GaussianBlur(5)), 0.38)
    # scene changes: a short dip and a soft light pulse
    for _name, (sa, _sb) in SCENES.items():
        if sa > 0 and abs(t - sa) < 0.35:
            k = 1 - abs(t - sa) / 0.35
            img = Image.blend(img, Image.new("RGBA", (W, H), (3, 6, 13, 255)), 0.85 * ease(k))
    rgb = grade(img)
    # the docking reveal (10.2 s) hides the surface change in a flash of light
    if 9.9 <= t <= 10.9:
        k = 1 - abs(t - 10.2) / (0.3 if t < 10.2 else 0.7)
        rgb = Image.blend(rgb, Image.new("RGB", (W, H), (200, 228, 255)), 0.75 * ease(clamp(k)))
    base = rgb.convert("RGBA")
    if 6.8 <= t < 52.0:
        k = ease((t - 6.8) / 0.8) * (1 - ease((t - 51.4) / 0.6))
        sc = scrim()
        if k < 1:
            sc = sc.copy()
            sc.putalpha(sc.getchannel("A").point(lambda v: int(v * k)))
        base.alpha_composite(sc)
    overlay(base, t, i, D, A)
    out = base.convert("RGB")
    out = add_grain(out, i)
    # global fade in/out
    if t < 0.6:
        out = Image.blend(Image.new("RGB", (W, H)), out, ease(t / 0.6))
    if t > TOTAL - 0.9:
        out = Image.blend(out, Image.new("RGB", (W, H)), ease((t - (TOTAL - 0.9)) / 0.9))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=float, nargs="*", help="render stills at these times (s) instead of the film")
    ap.add_argument("--out", default=str(ART / "leonardo_trailer_60s.mp4"))
    a = ap.parse_args()
    D = load_data()
    A = {s: anchors(s) for s in SCENES}
    if a.only:
        for t in a.only:
            p = ART / "stills" / f"still_{t:05.2f}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            compose(int(round(t * FPS)), D, A).save(p)
            print(p)
        return
    n = int(TOTAL * FPS)
    video = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", str(FPS), "-i", "-", "-ss", str(MUSIC_START), "-t", str(TOTAL), "-i", str(MUSIC),
                              "-filter_complex", f"[1:a]afade=t=in:st=0:d=1.5,afade=t=out:st={TOTAL - 3.5}:d=3.5,"
                              "volume=0.95[a]", "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "slow",
                              "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-movflags",
                              "+faststart", "-shortest", a.out], stdin=subprocess.PIPE)
    for i in range(n):
        video.stdin.write(compose(i, D, A).tobytes())
        if i % 48 == 0:
            print(f"{i / FPS:5.1f} s", flush=True)
    video.stdin.close()
    video.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
