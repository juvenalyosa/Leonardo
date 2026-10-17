#!/usr/bin/env python3
"""
Compositing stage of the 60 s Leonardo organic chemistry & catalysis film, for YouTube (16:9) and
Instagram (9:16): the PyMOL frames of scripts/organic_film_molecules.py, graded, with titles, live data
panels from the real Leonardo run (artifacts/organic_2026/data/results.json, gathered by
scripts/organic_film_data.py) and music.

    python scripts/organic_film_compose.py yt|ig [--only SECONDS ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import organic_film_data as OD  # noqa: E402  (timeline, shown modes, reaction path)
import trailer_compose as tc  # noqa: E402  (fonts, grade, panels, plots)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "organic_2026"
DATA = ART / "data"
FPS, TOTAL = OD.FPS, OD.TOTAL
SCENES = OD.SCENES
MUSIC = ART / "upbeat.wav"
MUSIC_START = 180.0          # out of a quiet dip, building to the track's peak near the end of the film
INK, MUTED, TEAL, AMBER = tc.INK, tc.MUTED, tc.TEAL, tc.AMBER
CORAL = (255, 120, 110)
card, ease, window = tc.card, tc.ease, tc.window

FMT = "yt"
W, H = 1920, 1080


def text(layer, xy, s, size, *a, **k):
    """tc.text; on the phone format small print is 22 % larger."""
    tc.text(layer, xy, s, int(round(size * 1.22)) if FMT == "ig" and size < 40 else size, *a, **k)


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


_anchors = {}


def anchor(scene, j):
    if scene not in _anchors:
        p = ART / "frames" / FMT / scene / "anchors.json"
        _anchors[scene] = json.load(open(p)) if p.exists() else {}
    return _anchors[scene].get(str(j), {})


def scene_at(t):
    for name, (a, b) in SCENES.items():
        if a <= t < b:
            return name, a
    return "outro", SCENES["outro"][0]


# ------------------------------------------------------------------------------------------- data
def load():
    r = json.load(open(DATA / "results.json"))
    modes = OD.shown_modes(DATA / "aspirin_modes.npz")
    freq, inten = [], []
    with open(DATA / "aspirin_ir.csv") as fh:
        for row in csv.DictReader(fh):
            if row["trv"] == "V" and float(row["frequency_cm-1"]) > 0:
                freq.append(float(row["frequency_cm-1"]))
                inten.append(float(row["ir_km_mol"]))
    x = np.linspace(400, 4000, 900)
    y = np.zeros_like(x)
    for f, a in zip(freq, inten):           # Lorentzian bands, 20 cm-1 wide
        y += a * (10.0 ** 2) / ((x - f) ** 2 + 10.0 ** 2)
    y /= y.max()
    return {"r": r, "modes": modes, "ir_x": x, "ir_y": y, "n_modes": len(freq)}


# ------------------------------------------------------------------------------------ layout bits
def wrap(s, n):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n and line:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    return out + ([line] if line else [])


def tag(lay, t, a, num, label, title, sub):
    if FMT == "yt":
        tc.scene_tag(lay, t, a, num, label, title, sub)
        return
    k = window(t, a, a + 99, 0.6)
    if k <= 0:
        return
    shade = Image.new("RGBA", lay.size)
    text(shade, (W / 2, 190), title, 64, (0, 0, 0), "demi", 0.8 * k, "mm")
    lay.alpha_composite(shade.filter(ImageFilter.GaussianBlur(8)))
    text(lay, (W / 2, 120), f"{num}  ·  {label}", 26, TEAL, "demi", k, "mm", spacing=4)
    text(lay, (W / 2, 190), title, 64, INK, "demi", ease((t - a - 0.15) / 0.6), "mm")
    for j, line in enumerate(wrap(sub, 30)):
        text(lay, (W / 2, 252 + 36 * j), line, 28, (196, 210, 232), "regular", ease((t - a - 0.35) / 0.6), "mm")


def panel_box():
    return (1330, 270, 1850, 880) if FMT == "yt" else (60, 1160, 1020, 1775)


def metric(layer, x, y, label, value, unit, k, color=INK):
    """tc.metric, scaled up on the phone format."""
    if FMT == "yt":
        tc.metric(layer, x, y, label, value, unit, k, color)
        return
    text(layer, (x, y), label.upper(), 15, MUTED, "demi", k, "la", spacing=2)
    text(layer, (x, y + 30), value, 64, color, "demi", k, "la")
    w = ImageDraw.Draw(layer).textlength(value, font=tc.font(64, "demi"))
    text(layer, (x + w + 12, y + 70), unit, 22, MUTED, "medium", k, "la")


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
    text(sub, (W / 2, y + 112), subtitle, 30 if FMT == "yt" else 24, TEAL, "demi", window(t, a + 1.1, end, 0.7), "mm",
         spacing=10 if FMT == "yt" else 6)
    base.alpha_composite(sub)


def profile(s, barrier, reaction):
    """Energy along the drawn path, through the computed stationary points (zero slope at each)."""
    if s <= 1.0:
        return barrier * (1 - math.cos(math.pi * s)) / 2
    return barrier + (reaction - barrier) * (1 - math.cos(math.pi * (s - 1.0))) / 2


def energy_panel(lay, t, a, D, which, k, s):
    """Energy profile(s) keto -> TS -> enol with the system's position s on the shown one."""
    x0, y0, x1, y1 = panel_box()
    card(lay, (x0, y0, x1, y1), k)
    r = D["r"]
    U, Wt = r["uncatalysed"], r["water"]
    text(lay, (x0 + 40, y0 + 44), "ENERGY ALONG THE REACTION", 17, MUTED, "demi", k, "lm", spacing=2)
    text(lay, (x0 + 40, y0 + 72), "B3LYP-D3(BJ)/def2-SVP · kcal/mol", 17, MUTED, "regular", k, "lm")
    box = (x0 + 80, y0 + 150, x1 - 40, y0 + 400) if FMT == "yt" else (x0 + 90, y0 + 140, x1 - 60, y0 + 390)
    xr, yr = (0.0, 2.0), (-5.0, 75.0)
    tc.plot_axes(lay, box, xr, yr, k, "", "", yticks=(0, 35, 70))
    for xs, lab in ((0.0, "keto"), (1.0, "TS"), (2.0, "enol")):
        px, _ = tc.to_px(box, xr, yr, xs, 0)
        text(lay, (px, box[3] + 22), lab, 18, MUTED, "demi", k, "mm")
    curves = [("uncat", U, CORAL)] + ([("water", Wt, TEAL)] if which == "water" else [])
    for name, P, col in curves:
        active = name == which
        kc = k * (1.0 if active else 0.45)
        if which == "water" and name == "water":
            kc *= ease((t - a - 0.6) / 0.8)
        pts = [tc.to_px(box, xr, yr, x, profile(x, P["barrier_kcal_mol"], P["reaction_kcal_mol"]))
               for x in np.linspace(0, 2, 120)]
        tc.polyline(lay, pts, col, kc, width=3 if active else 2, glow=active)
        if active and s is not None:
            px, py = tc.to_px(box, xr, yr, s, profile(s, P["barrier_kcal_mol"], P["reaction_kcal_mol"]))
            dr = ImageDraw.Draw(lay)
            glow = Image.new("RGBA", lay.size)
            ImageDraw.Draw(glow).ellipse((px - 16, py - 16, px + 16, py + 16), fill=(*col, int(150 * k)))
            lay.alpha_composite(glow.filter(ImageFilter.GaussianBlur(8)))
            dr.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(255, 255, 255, int(255 * k)))
    # numbers under the plot
    yy = y0 + 452 if FMT == "yt" else y0 + 450
    x2 = x0 + (290 if FMT == "yt" else 520)
    if which == "uncat":
        kb = k * ease((t - a - 7.4) / 0.6)
        metric(lay, x0 + 40, yy, "Barrier ΔE‡", f"{U['barrier_kcal_mol']:.1f}", "kcal/mol", k, CORAL)
        metric(lay, x2, yy, "TS imaginary mode", f"{U['ts_imaginary_cm1']:.0f}i", "cm⁻¹", kb, INK)
    else:
        drop = U["barrier_kcal_mol"] - Wt["barrier_kcal_mol"]
        metric(lay, x0 + 40, yy, "No catalyst", f"{U['barrier_kcal_mol']:.1f}", "kcal/mol", 0.55 * k, CORAL)
        metric(lay, x2, yy, "With one water", f"{Wt['barrier_kcal_mol']:.1f}", "kcal/mol",
                  k * ease((t - a - 1.0) / 0.8), TEAL)
        kd = k * ease((t - a - 8.6) / 0.8)
        text(lay, ((x0 + x1) / 2, yy + (104 if FMT == "yt" else 128)), f"barrier lowered by {drop:.1f} kcal/mol", 26, TEAL, "demi", kd, "mm")


# --------------------------------------------------------------------------------------- overlay
def overlay(base, t, i, D):
    lay = Image.new("RGBA", (W, H))
    cx = W / 2
    # ------------------------------------------------------------------------------------- intro
    if t < 6.6:
        y0 = 430 if FMT == "yt" else 820
        title_card(base, t, 0.4, 6.2, y0, "ORGANIC CHEMISTRY  ·  CATALYSIS")
        k1 = window(t, 3.2, 6.2, 0.6)
        lines = (["Bonds that vibrate, protons that hop —", "and one water molecule that makes it easier."]
                 if FMT == "yt" else ["Bonds that vibrate,", "protons that hop — and one", "water molecule that",
                                      "makes it easier."])
        for j, line in enumerate(lines):
            text(lay, (cx, y0 + 230 + 52 * j), line, 40 if FMT == "yt" else 44, INK, "medium", k1, "mm")
    # -------------------------------------------------------------------------------- spectroscopy
    if 6.5 <= t < SCENES["aspirin"][1] + 0.2:
        a = 7.0
        tag(lay, t, a, "01", "SPECTROSCOPY", "Aspirin, in infrared",
            f"Leonardo optimizes the molecule and computes all {D['n_modes']} of its vibrations")
        x0, y0, x1, y1 = panel_box()
        k = ease((t - a - 0.6) / 0.7) * (1 - ease((t - SCENES["aspirin"][1] + 0.3) / 0.3))
        card(lay, (x0, y0, x1, y1), k)
        text(lay, (x0 + 40, y0 + 44), "COMPUTED IR SPECTRUM", 17, MUTED, "demi", k, "lm", spacing=2)
        text(lay, (x0 + 40, y0 + 72), "harmonic, unscaled · B3LYP-D3(BJ)/def2-SVP", 17, MUTED, "regular", k, "lm")
        box = (x0 + 40, y0 + 130, x1 - 40, y0 + 360) if FMT == "yt" else (x0 + 50, y0 + 130, x1 - 50, y0 + 360)
        xr, yr = (4000.0, 400.0), (0.0, 1.08)          # wavenumber decreasing to the right, as on a spectrometer
        tc.plot_axes(lay, box, xr, yr, k, "wavenumber, cm⁻¹", "", xticks=(4000, 3000, 2000, 1000))
        reveal = ease((t - a - 0.9) / 2.2)
        xs, ys = D["ir_x"][::-1], D["ir_y"][::-1]
        n = max(2, int(len(xs) * reveal))
        pts = [tc.to_px(box, xr, yr, x, y) for x, y in zip(xs[:n], ys[:n])]
        tc.polyline(lay, pts, (196, 214, 240), k, width=2, glow=False)
        mode = 1 if t < OD.IR_SWITCH else 2
        if t >= 7.2 and mode <= len(D["modes"]):
            ms = 7.2 if mode == 1 else OD.IR_SWITCH
            _idx, f, lab = D["modes"][mode - 1]
            km = k * ease((t - ms - 0.2) / 0.5) * (1 - ease((t - ms - 6.0) / 0.4))
            px, _ = tc.to_px(box, xr, yr, f, 0)
            dr = ImageDraw.Draw(lay)
            dr.line((px, box[1] - 6, px, box[3]), fill=(*TEAL, int(220 * km)), width=3)
            yy = y0 + 440 if FMT == "yt" else y0 + 450
            text(lay, (x0 + 40, yy), "NOW VIBRATING", 15, MUTED, "demi", km, "lm", spacing=2)
            text(lay, (x0 + 40, yy + 44), lab, 40, TEAL, "demi", km, "lm")
            text(lay, (x0 + 40, yy + 96), f"{f:.0f} cm⁻¹  ·  shown ~10¹³ times slower", 20, (196, 210, 232), "regular",
                 km, "lm")
            j = int(round((t - SCENES["aspirin"][0]) * FPS))
            hot = anchor("aspirin", j).get("hot")
            if hot is not None and FMT == "yt":
                tc.leader(lay, hot, lab, km, TEAL, dx=-170, dy=-120)
        provenance(lay, t, a, "run_psi4 (optimization + frequencies) in Leonardo · Leonardo organic pack "
                              "(python -m leonardo.replay)")
    # ---------------------------------------------------------------------------------- reactions
    for scene, num, label, title, sub in (
            ("uncat", "02", "REACTION", "A proton on the move",
             "Acetaldehyde becomes vinyl alcohol: keto–enol tautomerism through a strained four-membered ring"),
            ("water", "03", "CATALYSIS", "One water molecule",
             "A water relay turns the four-ring into a six-ring: two protons move in concert")):
        a, b = SCENES[scene]
        if a - 0.1 <= t < b + 0.2:
            tag(lay, t, a + 0.2, num, label, title, sub)
            j = int(round((t - a) * FPS))
            s = anchor(scene, j).get("s", OD.path_s(t - a))
            k = ease((t - a - 0.6) / 0.7) * (1 - ease((t - b + 0.3) / 0.3))
            energy_panel(lay, t, a, D, scene, k, s)
            hp = anchor(scene, j).get("h")
            kl = ease((t - a - 1.2) / 0.6) * (1 - ease((t - a - 6.8) / 0.5))
            if hp is not None:
                if FMT == "yt":
                    tc.leader(lay, hp, "this proton moves" if scene == "uncat" else "protons relayed by water",
                              kl, TEAL, dx=-180, dy=130)
                kt = ease((t - a - 7.5) / 0.4) * (1 - ease((t - a - 9.0) / 0.4))
                lab = "transition state"
                if FMT == "yt":
                    tc.leader(lay, hp, lab, kt, INK, dx=-180, dy=-130, sub="bonds half made, half broken")
                else:
                    text(lay, (W / 2, 1090), lab.upper(), 26, INK, "demi", kt, "mm", spacing=4)
            prov = ("run_psi4 in Leonardo: keto and enol minima, TS search + frequencies · motion drawn between "
                    "the computed stationary points")
            provenance(lay, t, a, prov)
    # -------------------------------------------------------------------------------------- outro
    if t >= SCENES["outro"][0]:
        a = SCENES["outro"][0]
        y0 = 400 if FMT == "yt" else 760
        title_card(base, t, a + 0.4, 60.5, y0, "ORGANIC CHEMISTRY  ·  CATALYSIS")
        k = window(t, a + 2.0, 60.5, 0.7)
        lines = ["Every result you saw came from a real", "calculation run with Leonardo."] if FMT == "ig" else \
            ["Every result you saw came from a real calculation run with Leonardo."]
        for j, line in enumerate(lines):
            text(lay, (cx, y0 + 200 + 48 * j), line, 34, INK, "medium", k, "mm")
        k2 = window(t, a + 3.0, 60.5, 0.7)
        yy = y0 + 200 + 48 * len(lines) + 50
        text(lay, (cx, yy), "github.com/juvenalyosa/Leonardo", 28, TEAL, "demi", k2, "mm")
        text(lay, (cx, yy + 44), "juvenal.yosa@gmail.com", 22, MUTED, "regular", k2, "mm")
        k3 = window(t, a + 3.6, 60.5, 0.7)
        foot = ["Built using PyMOL™* · Leonardo is an independent project, not a Schrodinger product",
                "* PyMOL is a trademark of Schrodinger, LLC.",
                "Music: “Getting it Done” / “Rocket” by Kevin MacLeod (incompetech.com), CC BY 4.0"]
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
        img = mol_frame("aspirin", max(0, 150 - int(j * 0.6)))
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
    # a soft flash as each system crosses its transition state
    for sc in ("uncat", "water"):
        ts_t = SCENES[sc][0] + 7.4
        if abs(t - ts_t) < 0.3:
            rgb = Image.blend(rgb, Image.new("RGB", (W, H), (200, 240, 235)), 0.16 * ease(1 - abs(t - ts_t) / 0.3))
    base = rgb.convert("RGBA")
    end = SCENES["outro"][0]
    if 6.8 <= t < end:
        sc = tc.scrim() if FMT == "yt" else vertical_scrim()
        k = ease((t - 6.8) / 0.8) * (1 - ease((t - (end - 0.6)) / 0.6))
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
    out = ART / f"leonardo_organic_{'youtube_16x9' if a.fmt == 'yt' else 'instagram_9x16'}.mp4"
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
