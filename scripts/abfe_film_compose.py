#!/usr/bin/env python3
"""
Compositing stage of the 60 s Leonardo ABFE-lite film, for YouTube (16:9) and Instagram (9:16): the PyMOL frames
of scripts/abfe_film_molecules.py, graded, with titles, live data panels from the real Leonardo run
(artifacts/abfe_2026/data, gathered by scripts/abfe_film_data.py) and music.  ABFE-lite is experimental in
Leonardo: the film says so, and shows its estimators next to the measured value.

    python scripts/abfe_film_compose.py yt|ig [--only SECONDS ...]
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
import abfe_film_data as AD  # noqa: E402  (timeline, colours)
import organic_film_compose as oc  # noqa: E402  (format-aware tags, metrics, title card)
import trailer_compose as tc  # noqa: E402  (fonts, grade, panels, plots)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "abfe_2026"
DATA = ART / "data"
FPS, TOTAL = AD.FPS, AD.TOTAL
SCENES = AD.SCENES
MUSIC = ROOT / "artifacts" / "materials_2026" / "electronic.wav"
MUSIC_START = 60.0
INK, MUTED, TEAL, AMBER = tc.INK, tc.MUTED, tc.TEAL, tc.AMBER
CORAL = (255, 120, 110)
COLS = AD.REPLICA_COLORS
card, ease, window = tc.card, tc.ease, tc.window

FMT = "yt"
W, H = 1920, 1080


def configure(fmt):
    global FMT, W, H
    FMT = fmt
    W, H = (1920, 1080) if fmt == "yt" else (1080, 1920)
    oc.configure(fmt)


def text(*a, **k):
    oc.text(*a, **k)


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


def load():
    r = json.load(open(DATA / "results.json"))
    z = np.load(DATA / "trajectories.npz")
    disp = [z[f"disp_{k}"] for k in range(r["replicas"])]
    return {"r": r, "disp": disp}


def fmt_kj(v, signed=True):
    if v is None:
        return "n/a"
    return (f"{v:+.1f}" if signed else f"{v:.1f}").replace("-", "−")


def panel_box():
    return (1330, 270, 1850, 880) if FMT == "yt" else (60, 1160, 1020, 1775)


def badge(lay, xy, k, size=22, anchor="mm"):
    """The amber EXPERIMENTAL pill."""
    if k <= 0.01:
        return
    label = "EXPERIMENTAL"
    if FMT == "ig" and size < 40:          # oc.text draws small print 22 % larger on the phone format
        size = int(round(size * 1.22))
    f = tc.font(size, "demi")
    d = ImageDraw.Draw(lay)
    sp = int(0.18 * size)
    w = d.textlength(label, font=f) + sp * (len(label) - 1) + 1.6 * size
    h = size * 1.9
    x, y = xy
    if anchor == "mm":
        x -= w / 2
    y -= h / 2
    d.rounded_rectangle((x, y, x + w, y + h), radius=h / 2, fill=(255, 179, 71, int(40 * k)),
                        outline=(*AMBER, int(230 * k)), width=2)
    tc.text(lay, (x + w / 2, y + h / 2), label, size, AMBER, "demi", k, "mm", spacing=sp)


def trace_panel(lay, t, k, D, replicas, frac, title, sub, span=1.0):
    """Ligand displacement from its crystal position against simulated time, revealed up to frac."""
    x0, y0, x1, y1 = panel_box()
    card(lay, (x0, y0, x1, y1), k)
    r = D["r"]
    text(lay, (x0 + 40, y0 + 44), title, 17, MUTED, "demi", k, "lm", spacing=2)
    text(lay, (x0 + 40, y0 + 72), sub, 17, MUTED, "regular", k, "lm")
    box = (x0 + 80, y0 + 150, x1 - 40, y0 + 390) if FMT == "yt" else (x0 + 90, y0 + 140, x1 - 60, y0 + 380)
    ymax = max(5.0, float(np.ceil(max(float(np.max(D["disp"][j])) for j in replicas) / 5.0) * 5.0))
    tmax = r["sim_ns"] * span
    xr, yr = (0.0, tmax), (0.0, ymax)
    tc.plot_axes(lay, box, xr, yr, k, "simulated time, ns", "Å", yticks=(0, ymax / 2, ymax),
                 xticks=(0, round(tmax / 2, 2), round(tmax, 2)))
    for j in replicas:
        d = D["disp"][j]
        n = max(2, int(round(frac * span * (len(d) - 1))) + 1)
        xs = np.linspace(0.0, r["sim_ns"], len(d))[:n]
        pts = [tc.to_px(box, xr, yr, x, y) for x, y in zip(xs, d[:n])]
        tc.polyline(lay, pts, COLS[j], k, width=2, glow=len(replicas) == 1)
        if pts:
            px, py = pts[-1]
            ImageDraw.Draw(lay).ellipse((px - 6, py - 6, px + 6, py + 6), fill=(255, 255, 255, int(255 * k)))
    return box


def overlay(base, t, i, D):
    lay = Image.new("RGBA", (W, H))
    cx = W / 2
    r = D["r"]
    per = r.get("per_replica") or []
    # ----------------------------------------------------------------------------------------- intro
    if t < 6.2:
        y0 = 430 if FMT == "yt" else 820
        oc.title_card(base, t, 0.4, 5.8, y0, "BINDING FREE ENERGY  ·  ABFE-LITE")
        k1 = window(t, 2.6, 5.8, 0.6)
        lines = (["How tightly does a drug hold on to its protein?"] if FMT == "yt"
                 else ["How tightly does a drug", "hold on to its protein?"])
        for j, line in enumerate(lines):
            text(lay, (cx, y0 + 220 + 52 * j), line, 40 if FMT == "yt" else 44, INK, "medium", k1, "mm")
        badge(lay, (cx, y0 + 220 + 52 * len(lines) + 40), window(t, 3.2, 5.8, 0.6), 22)
    # ---------------------------------------------------------------------------------- the complex
    a, b = SCENES["site"]
    if 6.0 <= t < b + 0.2:
        oc.tag(lay, t, 6.2, "01", "THE COMPLEX", "Benzamidine in trypsin",
               "Crystal structure 3PTB · Leonardo protonates the ligand at pH 7.4 and builds GAFF2 / AM1-BCC parameters")
        j = int(round((t - a) * FPS))
        p = anchor("site", j).get("ligand")
        if p is not None and FMT == "yt":
            tc.leader(lay, p, "benzamidine", window(t, 7.0, b - 0.2, 0.5), TEAL, dx=170, dy=-110,
                      sub="in the S1 pocket")
        oc.provenance(lay, t, 6.2, "load_structure_file · prepare_docking_ligand · parametrize_ligand  "
                                   "(Leonardo film pack 11)")
    # ------------------------------------------------------------------------------------ the exit
    a, b = SCENES["exit"]
    if a - 0.1 <= t < b + 0.2:
        oc.tag(lay, t, a + 0.2, "02", "GUIDED EXIT", "Pulled out, held, let back in",
               "Steered MD on the GPU: Leonardo guides the ligand out through a tunnel, holds it, and lets it return")
        frac = min(1.0, max(0.0, (t - a) / (b - a)))
        k = ease((t - a - 0.6) / 0.7) * (1 - ease((t - b + 0.3) / 0.3))
        span = AD.EXIT_FRACTION
        box = trace_panel(lay, t, k, D, [0], frac, "LIGAND DISPLACEMENT · REPLICA 1",
                          f"from its crystal position, Å · first {span * r['sim_ns']:.1f} of {r['sim_ns']:g} ns",
                          span=span)
        x0, y0, x1, _y1 = panel_box()
        d0 = D["disp"][0]
        now = float(d0[min(len(d0) - 1, int(round(frac * span * (len(d0) - 1))))])
        yy = y0 + 452 if FMT == "yt" else y0 + 450
        oc.metric(lay, x0 + 40, yy, "Distance now", f"{now:.1f}", "Å", k, COLS[0])
        oc.metric(lay, x0 + (290 if FMT == "yt" else 520), yy, "Time", f"{frac * span * r['sim_ns']:.2f}", "ns", k, INK)
        j = int(round((t - a) * FPS))
        p = anchor("exit", j).get("site")
        if p is not None and FMT == "yt":
            tc.leader(lay, p, "crystal pose", window(t, a + 1.0, a + 6.0, 0.5), (200, 214, 235), dx=-170, dy=120)
        oc.provenance(lay, t, a, f"run_abfe in Leonardo · {r['replicas']} replicas × {r['sim_ns']:g} ns · OpenMM on "
                                 f"{r.get('platform') or 'GPU'} · every frame is a sampled state")
    # ---------------------------------------------------------------------------------- replicas
    a, b = SCENES["replicas"]
    if a - 0.1 <= t < b + 0.2:
        oc.tag(lay, t, a + 0.2, "03", "STATISTICS", f"{r['replicas']} independent replicas",
               "Leonardo counts how often the ligand is bound or unbound in each one")
        frac = min(1.0, max(0.0, (t - a) / (b - a - 2.0)))
        k = ease((t - a - 0.6) / 0.7) * (1 - ease((t - b + 0.3) / 0.3))
        trace_panel(lay, t, k, D, list(range(r["replicas"])), frac, "LIGAND DISPLACEMENT · ALL REPLICAS",
                    "from its crystal position, Å")
        x0, y0, x1, _y1 = panel_box()
        yy = y0 + 452 if FMT == "yt" else y0 + 450
        step = (x1 - x0 - 80) / max(1, len(per))
        for q, rep in enumerate(per):
            kq = k * ease((t - a - 3.0 - 0.5 * q) / 0.6)
            pb = rep.get("p_bound")
            oc.metric(lay, x0 + 40 + step * q, yy, f"Replica {q + 1} bound",
                      f"{100 * pb:.0f}" if pb is not None else "n/a", "%", kq, COLS[q % 3])
        oc.provenance(lay, t, a, "run_abfe in Leonardo · populations and ΔG estimators from Leonardo's own report")
    # ------------------------------------------------------------------------------------ result
    a, b = SCENES["result"]
    if a - 0.1 <= t < b + 0.2:
        k = window(t, a + 0.2, b, 0.6)
        if FMT == "yt":
            box = (cx - 520, 150, cx + 520, 930)
        else:
            box = (60, 330, W - 60, 1420)
        card(lay, box, k)
        x0, y0, x1, y1 = box
        badge(lay, (cx, y0 + 70), k, 24)
        text(lay, (cx, y0 + 150), "What ABFE-lite estimated, ΔG in kJ/mol", 40, INK, "demi", k, "mm")
        prev = r.get("previous_run_kj") or {}
        rows = [("Selected estimator (solvent-rescored hold windows)", r.get("dg_selected_kj"), prev.get("selected"), CORAL),
                ("Bias-reweighted, standard state", r.get("dg_reweighted_standard_kj"),
                 prev.get("reweighted_standard"), AMBER),
                ("Bound / unbound populations", r.get("dg_pop_kj"), prev.get("population"), (196, 210, 232)),
                ("Exponential average", r.get("dg_exp_kj"), prev.get("exp_avg"), (196, 210, 232))]
        yy = y0 + 240
        if FMT == "yt" and prev:
            text(lay, (x1 - 190, yy - 46), "THIS RUN", 15, MUTED, "demi", k, "rm", spacing=2)
            text(lay, (x1 - 50, yy - 46), "PREVIOUS", 15, MUTED, "demi", k, "rm", spacing=2)
        for q, (lab, v, pv, col) in enumerate(rows):
            kr = k * ease((t - a - 0.8 - 0.35 * q) / 0.5)
            if FMT == "yt":
                text(lay, (x0 + 70, yy), lab, 26, (196, 210, 232), "regular", kr, "lm")
                text(lay, (x1 - 190, yy), fmt_kj(v), 30, col, "demi", kr, "rm")
                text(lay, (x1 - 50, yy), fmt_kj(pv), 26, MUTED, "medium", kr, "rm")
                yy += 66
            else:
                for jj, line in enumerate(oc.wrap(lab, 34)):
                    text(lay, (x0 + 50, yy + 34 * jj), line, 28, (196, 210, 232), "regular", kr, "lm")
                text(lay, (x1 - 50, yy), f"{fmt_kj(v)}", 38, col, "demi", kr, "rm")
                if pv is not None:
                    text(lay, (x1 - 50, yy + 40), f"previous run {fmt_kj(pv)}", 22, MUTED, "regular", kr, "rm")
                yy += 34 * len(oc.wrap(lab, 34)) + 84
        km = k * ease((t - a - 2.6) / 0.6)
        ImageDraw.Draw(lay).line((x0 + 60, yy - 10, x1 - 60, yy - 10), fill=(120, 160, 220, int(90 * km)), width=1)
        yy += 30
        if FMT == "yt":
            text(lay, (x0 + 70, yy), "Measured (experiment), kJ/mol", 26, TEAL, "demi", km, "lm")
            text(lay, (x1 - 190, yy), f"≈ {fmt_kj(r['dg_measured_kj'])}", 30, TEAL, "demi", km, "rm")
        else:
            text(lay, (x0 + 50, yy), "Measured (experiment)", 28, TEAL, "demi", km, "lm")
            text(lay, (x1 - 50, yy), f"≈ {fmt_kj(r['dg_measured_kj'])}", 38, TEAL, "demi", km, "rm")
        kn = k * ease((t - a - 3.6) / 0.6)
        note = ("The estimators disagree with each other and change from run to run; one may land near "
                "experiment by chance. Use ABFE-lite to compare ligands qualitatively, not as a binding free energy.")
        for jj, line in enumerate(oc.wrap(note, 70 if FMT == "yt" else 40)):
            text(lay, (cx, yy + 90 + 36 * jj), line, 24 if FMT == "yt" else 27, INK, "medium", kn, "mm")
    # ------------------------------------------------------------------------------------- outro
    if t >= SCENES["outro"][0]:
        a = SCENES["outro"][0]
        y0 = 400 if FMT == "yt" else 760
        oc.title_card(base, t, a + 0.4, 60.5, y0, "BINDING FREE ENERGY  ·  ABFE-LITE")
        k = window(t, a + 1.6, 60.5, 0.7)
        lines = (["A real calculation, run with Leonardo on a laptop GPU.", "The method is still experimental."]
                 if FMT == "yt" else ["A real calculation, run with", "Leonardo on a laptop GPU.",
                                      "The method is still experimental."])
        for j, line in enumerate(lines):
            text(lay, (cx, y0 + 200 + 48 * j), line, 34, INK if j < len(lines) - 1 else AMBER, "medium", k, "mm")
        k2 = window(t, a + 2.6, 60.5, 0.7)
        yy = y0 + 200 + 48 * len(lines) + 50
        text(lay, (cx, yy), "github.com/juvenalyosa/Leonardo", 28, TEAL, "demi", k2, "mm")
        text(lay, (cx, yy + 44), "juvenal.yosa@gmail.com", 22, MUTED, "regular", k2, "mm")
        k3 = window(t, a + 3.2, 60.5, 0.7)
        foot = ["Built using PyMOL™* · Leonardo is an independent project, not a Schrodinger product",
                "* PyMOL is a trademark of Schrodinger, LLC.",
                "Music: “Ouroboros” / “Exhilarate” by Kevin MacLeod (incompetech.com), CC BY 4.0"]
        if FMT == "ig":
            foot = [ln for f in foot for ln in oc.wrap(f, 58)]
        for j, line in enumerate(foot):
            text(lay, (cx, H - 40 - 28 * (len(foot) - j)), line, 16 if FMT == "yt" else 19, MUTED, "regular", k3, "mm")
    base.alpha_composite(lay)


def compose(i, D):
    t = i / FPS
    scene, a = scene_at(t)
    j = i - int(round(a * FPS))
    if scene in ("result", "outro"):
        nrep = int(round((SCENES["replicas"][1] - SCENES["replicas"][0]) * FPS))
        img = mol_frame("replicas", nrep - 1)
    else:
        img = mol_frame(scene, j)
    if t < 6.6:
        k = ease((t - 4.4) / 2.2)
        blur = 6 * (1 - k)
        if blur > 0.2:
            img = img.filter(ImageFilter.GaussianBlur(blur))
        img = Image.blend(Image.new("RGBA", (W, H), (3, 6, 13, 255)), img, 0.42 + 0.58 * k)
    if scene in ("result", "outro"):
        img = Image.blend(Image.new("RGBA", (W, H), (3, 6, 13, 255)), img.filter(ImageFilter.GaussianBlur(5)), 0.36)
    for _n, (sa, _sb) in SCENES.items():
        if sa > 0 and abs(t - sa) < 0.3:
            img = Image.blend(img, Image.new("RGBA", (W, H), (3, 6, 13, 255)), 0.8 * ease(1 - abs(t - sa) / 0.3))
    base = tc.grade(img).convert("RGBA")
    end = SCENES["result"][0]
    if 6.0 <= t < end:
        sc = tc.scrim() if FMT == "yt" else oc.vertical_scrim()
        k = ease((t - 6.0) / 0.8) * (1 - ease((t - (end - 0.6)) / 0.6))
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
    out = ART / f"leonardo_abfe_{'youtube_16x9' if a.fmt == 'yt' else 'instagram_9x16'}.mp4"
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
