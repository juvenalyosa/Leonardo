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
    figs = {name: Image.open(DATA / f"{name}.png").convert("RGB") for name in ("analysis", "exit_profiles")}
    PANELS.update(find_panels(figs["analysis"]))
    return {"r": r, "disp": disp, "figs": figs}


def find_panels(img):
    """Panel rectangles of Leonardo's 2 x 2 analysis figure, from its axes frames (#D1D5DB spines): the header
    above them grows with the run's summary, so fixed fractions would cut titles."""
    a = np.asarray(img, dtype=int)
    h, w = a.shape[:2]
    spine = np.all(np.abs(a - np.array([209, 213, 219])) <= 14, axis=2)
    rows = np.where(spine[:, : w // 2].sum(axis=1) > 0.3 * (w // 2))[0]
    if rows.size < 4:
        return {}
    groups = np.split(rows, np.where(np.diff(rows) > 3)[0] + 1)
    ys = [int(g.mean()) for g in groups]
    if len(ys) < 4:
        return {}
    top_u, bot_u, top_l = ys[0], ys[1], ys[2]
    title = 0.038 * h                           # panel title above its frame
    y0 = (top_u - title - 0.008 * h) / h
    split = (top_l - title) / h                 # upper x-axis labels end above the lower titles
    return {"conv": (0.0, y0, 0.505, split), "epmf": (0.505, y0, 1.0, split),
            "rpmf": (0.0, split + 0.006, 0.505, 1.0), "surf": (0.505, split + 0.006, 1.0, 1.0)}


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


# Leonardo's 4-panel analysis figure: where each panel sits (fractions of the image, title included)
PANELS = {"full": (0.0, 0.0, 1.0, 1.0), "conv": (0.0, 0.14, 0.5, 0.585), "epmf": (0.5, 0.14, 1.0, 0.585),
          "rpmf": (0.0, 0.565, 0.5, 1.0), "surf": (0.5, 0.565, 1.0, 1.0)}
# (start s into the scene, figure, view, label, caption) — the camera moves to each view in turn
ANALYSIS_SHOTS = [
    (0.0, "analysis", "full", "LEONARDO'S ANALYSIS",
     "The report Leonardo draws after every ABFE-lite run, from all three replicas"),
    (3.4, "analysis", "conv", "FREE-ENERGY CONVERGENCE",
     "ΔG as frames accumulate. The cumulant estimate swings by thousands of kJ/mol: it has not converged"),
    (6.6, "analysis", "epmf", "PMF FROM THE ENERGY DISTRIBUTION",
     "Free-energy profile of the protein–ligand interaction energy over the sampled states"),
    (9.8, "analysis", "rpmf", "PMF ALONG THE EXIT COORDINATE",
     "The bound basin in the pocket and the cost of leaving it; dashed line: Leonardo's bound cutoff"),
    (13.0, "analysis", "surf", "2D FREE-ENERGY SURFACE",
     "Exit distance against interaction energy: the deep bound basin, the shallow unbound one, and the flat "
     "plateau where the ligand never went"),
    (16.2, "surface3d", "full", "THE SAME SURFACE, IN 3D",
     "Leonardo's saved surface data as a landscape: the bound state is the deep well, the plateau where the "
     "ligand never went"),
    (20.6, "exit_profiles", "full", "EXIT PROFILES",
     "Every replica leaves the pocket and comes back; dashed line: Leonardo's bound cutoff"),
]


def window_box():
    return (96, 292, 1110, 968) if FMT == "yt" else (40, 430, 1040, 1180)


def shot_at(u):
    k = 0
    for q, sh in enumerate(ANALYSIS_SHOTS):
        if u >= sh[0]:
            k = q
    return k


def view_rect(view, img):
    x0, y0, x1, y1 = PANELS[view]
    w, h = img.size
    return np.array([x0 * w, y0 * h, x1 * w, y1 * h])


def surface3d_frame(u_shot, size):
    """Frame of the rotating 3D surface (scripts/abfe_film_surface3d.py), u_shot seconds into its shot."""
    d = ART / "frames" / FMT / "surface3d"
    n = len(list(d.glob("mol_*.png")))
    p = d / f"mol_{min(max(0, int(u_shot * FPS)), max(0, n - 1)):04d}.png"
    img = Image.open(p).convert("RGB") if p.exists() else Image.new("RGB", (int(size[0]), int(size[1])), (18, 26, 40))
    return img.resize((int(size[0]), int(size[1])), Image.LANCZOS)


def shot_content(q, u, size):
    """Window content of analysis shot q at u seconds into the scene (no glide or fade)."""
    start, fig_name, view = ANALYSIS_SHOTS[q][:3]
    if fig_name == "surface3d":
        return surface3d_frame(u - start, size)
    img = _FIGS[fig_name]
    return render_view(img, view_rect(view, img), size)


_FIGS = {}


def render_view(img, rect, size):
    """The figure region rect fitted whole into size, padded with the figure's own background colour."""
    cw, ch = size
    part = img.crop(tuple(int(round(v)) for v in rect))
    f = min(cw / part.width, ch / part.height)
    part = part.resize((max(1, int(part.width * f)), max(1, int(part.height * f))), Image.LANCZOS)
    out = Image.new("RGB", (int(cw), int(ch)), img.getpixel((4, 4)))
    out.paste(part, ((int(cw) - part.width) // 2, (int(ch) - part.height) // 2))
    return out


def figure_window(base, t, a, b, D):
    """Leonardo's figures in an app-style window; the view glides from panel to panel."""
    u = t - a
    k = window(t, a + 0.1, b, 0.6)
    if k <= 0:
        return
    x0, y0, x1, y1 = window_box()
    bar = 46
    cw, ch = x1 - x0, y1 - y0 - bar
    q = shot_at(u)
    start, fig_name, view = ANALYSIS_SHOTS[q][:3]
    _FIGS.update(D["figs"])
    if fig_name != "surface3d" and q > 0 and ANALYSIS_SHOTS[q - 1][1] == fig_name:   # glide from the previous view
        img = D["figs"][fig_name]
        prev = view_rect(ANALYSIS_SHOTS[q - 1][2], img)
        rect = prev + (view_rect(view, img) - prev) * ease((u - start) / 0.9)
        content = render_view(img, rect, (cw, ch))
    else:
        content = shot_content(q, u, (cw, ch))
        if q > 0 and ANALYSIS_SHOTS[q - 1][1] != fig_name:      # cross-fade from another figure
            prev_start = ANALYSIS_SHOTS[q - 1][0]
            before = shot_content(q - 1, min(u, start - 1.0 / FPS) if ANALYSIS_SHOTS[q - 1][1] == "surface3d"
                                  else prev_start + 1.0, (cw, ch))
            content = Image.blend(before, content.convert("RGB"), ease((u - start) / 0.7))
    # window: soft shadow, dark title bar with traffic lights, the figure
    scale = 0.94 + 0.06 * tc.ease_out((t - a) / 0.9)
    win = Image.new("RGBA", (int(cw), int(ch + bar)), (0, 0, 0, 0))
    d = ImageDraw.Draw(win)
    d.rounded_rectangle((0, 0, cw - 1, ch + bar - 1), radius=14, fill=(18, 26, 40, 255))
    for j, col in enumerate(((255, 95, 87), (254, 188, 46), (40, 200, 64))):
        d.ellipse((18 + 22 * j, bar / 2 - 6, 30 + 22 * j, bar / 2 + 6), fill=(*col, 255))
    titles = {"analysis": "Leonardo · ABFE-lite analysis", "exit_profiles": "Leonardo · Plot: ABFE-lite exit profiles",
              "surface3d": "Leonardo's free-energy surface data · 3D view"}
    tc.text(win, (cw / 2, bar / 2), titles[fig_name], 18, (196, 210, 232), "medium", 1.0, "mm")
    mask = Image.new("L", (int(cw), int(ch)), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, -20, cw - 1, ch - 1), radius=14, fill=255)
    win.paste(content, (0, bar), mask)
    if scale < 0.999:
        win = win.resize((int(win.width * scale), int(win.height * scale)), Image.LANCZOS)
    px = int(x0 + (cw - win.width) / 2)
    py = int(y0 + (ch + bar - win.height) / 2)
    shadow = Image.new("RGBA", base.size)
    ImageDraw.Draw(shadow).rounded_rectangle((px + 6, py + 18, px + win.width + 6, py + win.height + 18), radius=18,
                                             fill=(0, 0, 0, int(170 * k)))
    base.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(22)))
    glow = Image.new("RGBA", base.size)
    ImageDraw.Draw(glow).rounded_rectangle((px - 2, py - 2, px + win.width + 2, py + win.height + 2), radius=16,
                                           outline=(*TEAL, int(90 * k)), width=2)
    base.alpha_composite(glow.filter(ImageFilter.GaussianBlur(6)))
    if k < 1:
        win.putalpha(win.getchannel("A").point(lambda v: int(v * k)))
    base.alpha_composite(win, (px, py))


def analysis_caption(lay, t, a, b):
    u = t - a
    q = shot_at(u)
    start, _fig, _view, label, cap = ANALYSIS_SHOTS[q]
    end = ANALYSIS_SHOTS[q + 1][0] if q + 1 < len(ANALYSIS_SHOTS) else (b - a)
    k = window(u, start + 0.35, end, 0.4) * window(t, a, b, 0.4)
    if FMT == "yt":
        x, y = 1180, 470
        text(lay, (x, y - 40), f"{q + 1} / {len(ANALYSIS_SHOTS)}", 18, MUTED, "demi", k, "lm", spacing=3)
        text(lay, (x, y), label, 26, TEAL, "demi", k, "lm", spacing=3)
        for j, line in enumerate(oc.wrap(cap, 34)):
            text(lay, (x, y + 52 + 40 * j), line, 28, INK, "regular", k, "lm")
    else:
        y = 1270
        text(lay, (W / 2, y - 44), f"{q + 1} / {len(ANALYSIS_SHOTS)}", 18, MUTED, "demi", k, "mm", spacing=3)
        text(lay, (W / 2, y), label, 26, TEAL, "demi", k, "mm", spacing=3)
        for j, line in enumerate(oc.wrap(cap, 36)):
            text(lay, (W / 2, y + 56 + 42 * j), line, 30, INK, "regular", k, "mm")


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
            tc.leader(lay, p, "benzamidine", window(t, 7.0, b - 0.2, 0.5), TEAL, dx=190, dy=90,
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
            oc.metric(lay, x0 + 40 + step * q, yy, f"{'Rep.' if FMT == 'yt' else 'Replica'} {q + 1} bound",
                      f"{100 * pb:.0f}" if pb is not None else "n/a", "%", kq, COLS[q % 3])
        oc.provenance(lay, t, a, "run_abfe in Leonardo · populations and ΔG estimators from Leonardo's own report")
    # ---------------------------------------------------------------------------------- analysis
    a, b = SCENES["analysis"]
    if a - 0.1 <= t < b + 0.2:
        oc.tag(lay, t, a + 0.2, "04", "ANALYSIS", "What Leonardo reports",
               "The free-energy analysis Leonardo draws from the three replicas, unedited")
        figure_window(base, t, a, b, D)
        analysis_caption(lay, t, a, b)
        oc.provenance(lay, t, a, "Leonardo's ABFE-lite report figure and show_plot('abfe'); the 3D view is drawn from the "
                                 "surface data Leonardo saves with the figure")
    # ------------------------------------------------------------------------------------ result
    a, b = SCENES["result"]
    if a - 0.1 <= t < b + 0.2:
        k = window(t, a + 0.2, b, 0.6)
        if FMT == "yt":
            box = (cx - 580, 150, cx + 580, 930)
        else:
            box = (60, 330, W - 60, 1420)
        card(lay, box, k)
        x0, y0, x1, y1 = box
        badge(lay, (cx, y0 + 70), k, 24)
        text(lay, (cx, y0 + 150), "What ABFE-lite estimated, ΔG in kJ/mol", 40, INK, "demi", k, "mm")
        prevs = list(r.get("previous_runs_kj") or [])
        keys = ("selected", "reweighted_standard", "population", "exp_avg")
        rows = [("Selected (solvent-rescored hold windows)", r.get("dg_selected_kj"), CORAL),
                ("Bias-reweighted, standard state", r.get("dg_reweighted_standard_kj"), AMBER),
                ("Bound / unbound populations", r.get("dg_pop_kj"), (196, 210, 232)),
                ("Exponential average", r.get("dg_exp_kj"), (196, 210, 232))]
        yy = y0 + 240
        col_now, col_prev = x1 - 50, x1 - 230
        if FMT == "yt":
            text(lay, (col_prev, yy - 46), f"{len(prevs)} EARLIER RUNS", 16, MUTED, "demi", k, "rm")
            text(lay, (col_now, yy - 46), "THIS FILM", 16, TEAL, "demi", k, "rm")

        def span(vals):
            vals = [v for v in vals if v is not None]
            if not vals:
                return ""
            lo, hi = min(vals), max(vals)
            return fmt_kj(lo) if abs(hi - lo) < 0.05 else f"{fmt_kj(lo)} to {fmt_kj(hi)}"

        for q, (lab, v, col) in enumerate(rows):
            kr = k * ease((t - a - 0.8 - 0.35 * q) / 0.5)
            pvs = [p_.get(keys[q]) for p_ in prevs]
            if FMT == "yt":
                text(lay, (x0 + 70, yy), lab, 24, (196, 210, 232), "regular", kr, "lm")
                text(lay, (col_prev, yy), span(pvs), 24, MUTED, "medium", kr, "rm")
                text(lay, (col_now, yy), fmt_kj(v), 30, col, "demi", kr, "rm")
                yy += 66
            else:
                for jj, line in enumerate(oc.wrap(lab, 34)):
                    text(lay, (x0 + 50, yy + 34 * jj), line, 28, (196, 210, 232), "regular", kr, "lm")
                text(lay, (x1 - 50, yy), f"{fmt_kj(v)}", 38, col, "demi", kr, "rm")
                if pvs:
                    text(lay, (x1 - 50, yy + 40), f"{len(pvs)} earlier runs: {span(pvs)}", 22, MUTED, "regular", kr,
                         "rm")
                yy += 34 * len(oc.wrap(lab, 34)) + 84
        km = k * ease((t - a - 2.6) / 0.6)
        ImageDraw.Draw(lay).line((x0 + 60, yy - 10, x1 - 60, yy - 10), fill=(120, 160, 220, int(90 * km)), width=1)
        yy += 30
        if FMT == "yt":
            text(lay, (x0 + 70, yy), "Measured (experiment)", 26, TEAL, "demi", km, "lm")
            text(lay, (x1 - 50, yy), f"≈ {fmt_kj(r['dg_measured_kj'])}", 30, TEAL, "demi", km, "rm")
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
        oc.title_card(base, t, a + 0.4, TOTAL + 0.5, y0, "BINDING FREE ENERGY  ·  ABFE-LITE")
        k = window(t, a + 1.6, TOTAL + 0.5, 0.7)
        lines = (["A real calculation, run with Leonardo on a laptop GPU.", "The method is still experimental."]
                 if FMT == "yt" else ["A real calculation, run with", "Leonardo on a laptop GPU.",
                                      "The method is still experimental."])
        for j, line in enumerate(lines):
            text(lay, (cx, y0 + 200 + 48 * j), line, 34, INK if j < len(lines) - 1 else AMBER, "medium", k, "mm")
        k2 = window(t, a + 2.6, TOTAL + 0.5, 0.7)
        yy = y0 + 200 + 48 * len(lines) + 50
        text(lay, (cx, yy), "github.com/juvenalyosa/Leonardo", 28, TEAL, "demi", k2, "mm")
        text(lay, (cx, yy + 44), "juvenal.yosa@gmail.com", 22, MUTED, "regular", k2, "mm")
        k3 = window(t, a + 3.2, TOTAL + 0.5, 0.7)
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
    if scene in ("analysis", "result", "outro"):
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
    if scene in ("analysis", "result", "outro"):
        img = Image.blend(Image.new("RGBA", (W, H), (3, 6, 13, 255)), img.filter(ImageFilter.GaussianBlur(5)), 0.36)
    for _n, (sa, _sb) in SCENES.items():
        if sa > 0 and abs(t - sa) < 0.3:
            img = Image.blend(img, Image.new("RGBA", (W, H), (3, 6, 13, 255)), 0.8 * ease(1 - abs(t - sa) / 0.3))
    base = tc.grade(img).convert("RGBA")
    end = SCENES["analysis"][0]
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
