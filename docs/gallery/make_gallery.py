#!/usr/bin/env python3
"""
README gallery: cinematic 1920x1080 posters ray traced with PyMOL from structures Leonardo built.

    python docs/gallery/make_gallery.py --work /path/to/work [--only membrane gold ...]

Inputs (``--work``): membrane/membrane_system_minimized.pdb (build_membrane_system),
gold/surface_system_minimized.pdb (build_protein_surface_system), orb/*.cube (Psi4 cubeprop),
1A6M.pdb; the docking / MD / QM/MM backplates come from artifacts/promo_cinematic_20260916/renders.
Every number printed on a poster is passed in from the calculation that produced the scene.
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/leonardo-gallery-mpl")
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PROMO = ROOT / "artifacts" / "promo_cinematic_20260916" / "renders"

S = 0.5                                   # layout designed at 3840x2160, rendered at half size
W, H = 3840, 2160
MW, MH = 2800, 2100                       # molecular area (layout units)
MX, MY = 1040, 20
FONTS = Path("/System/Library/Fonts/Supplemental")
WHITE = (238, 244, 247)
MUTED = (148, 174, 189)
GOLD = (236, 190, 103)
BLUE = (114, 205, 229)


def font(size, style="regular"):
    name = {"regular": "Arial.ttf", "bold": "Arial Bold.ttf", "serif": "Baskerville.ttc"}[style]
    return ImageFont.truetype(str(FONTS / name), max(8, round(size * S)))


def text(d, xy, value, size, color=WHITE, style="regular"):
    d.text((round(xy[0] * S), round(xy[1] * S)), value, font=font(size, style), fill=color)


def tracked(d, xy, value, size, color, spacing=7):
    x, y = xy[0] * S, xy[1] * S
    f = font(size, "bold")
    for ch in value:
        d.text((round(x), round(y)), ch, font=f, fill=color)
        x += d.textlength(ch, font=f) + spacing * S


def hline(d, coords, fill, width=1):
    d.line(tuple(round(v * S) for v in coords), fill=fill, width=max(1, round(width * S)))


def background(warm=True):
    w, h = round(W * S), round(H * S)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= w
    yy /= h
    a = np.exp(-(((xx - .75) / .30) ** 2 + ((yy - .43) / .52) ** 2) * 2)
    b = np.exp(-(((xx - .90) / .22) ** 2 + ((yy - .72) / .40) ** 2) * 2)
    hot = np.exp(-(((xx - .62) / .16) ** 2 + ((yy - .14) / .19) ** 2) * 2)
    c = np.zeros((h, w, 3), np.float32) + [5, 12, 21]
    c += a[:, :, None] * np.array([7, 28, 42])
    c += b[:, :, None] * np.array([3, 13, 22])
    if warm:
        c += hot[:, :, None] * np.array([15, 10, 2])
    c += np.random.default_rng(491).normal(0, .48, (h, w, 1))
    return Image.fromarray(np.uint8(np.clip(c, 0, 255)), "RGB").convert("RGBA")


# ------------------------------------------------------------------------------------ PyMOL


def pymol_style(c):
    c.set_color("g_bg", [.02, .04, .07])
    c.bg_color("g_bg")
    for key, value in {
        "ray_opaque_background": 0, "orthoscopic": 1, "antialias": 2, "max_threads": 6,
        "ambient": .26, "direct": .85, "reflect": .55, "specular": .85, "shininess": 65,
        "ray_shadows": 1, "ray_trace_mode": 0, "depth_cue": 0, "surface_quality": 1,
        "cartoon_sampling": 14, "cartoon_loop_radius": .18, "cartoon_oval_length": 1.05,
        "cartoon_oval_width": .23, "cartoon_rect_length": 1.0, "cartoon_rect_width": .23,
        "cartoon_fancy_helices": 0, "cartoon_flat_sheets": 1, "cartoon_smooth_loops": 1,
        "valence": 1, "stick_quality": 16, "sphere_quality": 2, "light_count": 4,
        "light": [-.45, -.65, 1.0], "light2": [.8, .1, .7], "light3": [-.7, .6, .4],
    }.items():
        c.set(key, value)
    for name, rgb in {
        "g_teal": [.16, .61, .73], "g_blue": [.29, .56, .72], "g_gold": [.93, .64, .23],
        "g_surface": [.20, .41, .51], "g_silver": [.61, .75, .79], "g_oxygen": [.96, .23, .17],
        "g_nitrogen": [.22, .46, 1.], "g_iron": [.85, .35, .20], "g_metal": [.95, .73, .32],
        "g_lipid": [.45, .58, .66], "g_pos": [.25, .55, 1.0], "g_neg": [.95, .30, .25],
    }.items():
        c.set_color(name, rgb)


def ray(c, work, name, objects):
    c.disable("all")
    for obj in objects:
        c.enable(obj)
    path = work / f"{name}.png"
    print("rendering", path.name, flush=True)
    c.png(str(path), round(MW * S), round(MH * S), ray=1, quiet=1)
    return Image.open(path).convert("RGBA")


def fit(c, selection, rect):
    """Place ``selection`` in the layout rectangle (left, top, right, bottom), keeping orientation."""
    view = np.array(c.get_view())
    xyz = c.get_coords(selection)
    proj = (xyz - view[12:15]) @ view[:9].reshape(3, 3)
    lo, hi = proj[:, :2].min(axis=0) - 1.5, proj[:, :2].max(axis=0) + 1.5
    left, top, right, bottom = rect
    factor = min((right - left) / (hi[0] - lo[0]), (bottom - top) / (hi[1] - lo[1]))
    mid = (lo + hi) / 2
    view[9] = ((left + right) / 2 - MX - MW / 2) / factor - mid[0]
    view[10] = -((top + bottom) / 2 - MY - MH / 2) / factor - mid[1]
    view[11] = -MH / (2 * factor * math.tan(math.radians(float(c.get("field_of_view"))) / 2))
    view[15] = max(.1, abs(view[11]) - 200)
    view[16] = abs(view[11]) + 200
    c.set_view(view.tolist())


def project(c, selection):
    view = np.array(c.get_view())
    xyz = c.get_coords(selection).mean(axis=0)
    q = view[:9].reshape(3, 3).T @ (xyz - view[12:15]) + view[9:12]
    factor = MH / (2 * abs(view[11]) * math.tan(math.radians(float(c.get("field_of_view"))) / 2))
    return [MX + MW / 2 + q[0] * factor, MY + MH / 2 - q[1] * factor]


# ------------------------------------------------------------------------------------ scenes


def scene_membrane(work, src):
    import pymol2

    pdb = src / "membrane" / "membrane_system_minimized.pdb"
    with pymol2.PyMOL() as pm:
        c = pm.cmd
        pymol_style(c)
        c.load(str(pdb), "sys")
        c.remove("resn HOH+NA+CL+SOD+CLA")
        c.create("prot", "sys and polymer.protein")
        c.create("chol", "sys and resn CHL+CHL1")
        c.create("lip", "sys and not polymer.protein and not resn CHL+CHL1")
        c.delete("sys")
        c.hide("everything")
        c.show("cartoon", "prot")
        c.color("g_teal", "prot")
        c.set("cartoon_transparency", 0, "prot")
        c.show("sticks", "chol and not hydro")
        c.color("g_gold", "chol")
        c.set("stick_radius", .22, "chol")
        c.show("lines", "lip and not hydro")
        c.color("g_lipid", "lip")
        c.show("spheres", "lip and elem P")
        c.set("sphere_scale", .9, "lip and elem P")
        c.color("g_silver", "lip and elem P")
        c.orient("prot")
        c.turn("x", -90)                        # bilayer normal (z) vertical in the picture
        c.turn("x", 12)
        c.turn("y", 25)
        fit(c, "prot or chol or (lip and elem P)", (1320, 140, 3780, 1900))
        layers = [("context", ray(c, work, "lipids", ["lip"]), .55, 0),
                  ("focus", ray(c, work, "focus", ["prot", "chol"]), 1, 0)]
        # label one cholesterol: the hydroxyl O3 lowest-left in the picture
        o3 = [project(c, f"chol and name O3 and resi {a.resi} and chain {a.chain}") for a in c.get_model("chol and name O3").atom]
        chol = min(o3, key=lambda p: p[0] - .6 * p[1]) if o3 else project(c, "chol")
        anchors = {"prot": project(c, "prot and name CA"), "chol": chol}
    return layers, anchors


def scene_gold(work, src):
    import pymol2

    with pymol2.PyMOL() as pm:
        c = pm.cmd
        pymol_style(c)
        c.load(str(src / "gold" / "surface_system_minimized.pdb"), "sys")
        c.create("metal", "sys and chain Z")
        c.create("prot", "sys and polymer.protein")
        c.delete("sys")
        c.hide("everything")
        c.show("spheres", "metal")
        c.set("sphere_scale", .95, "metal")
        c.color("g_metal", "metal")
        c.set("specular", 1.0)
        c.show("cartoon", "prot")
        c.color("g_teal", "prot")
        c.show("sticks", "prot and resi 44 and not hydro and not name N+C+O")
        c.color("g_gold", "prot and resi 44 and elem C")
        c.set("stick_radius", .3, "prot")
        c.orient("metal")
        c.turn("x", -72)
        c.turn("z", 18)
        fit(c, "prot or metal", (1300, 220, 3760, 1880))
        layers = [("focus", ray(c, work, "focus", ["metal", "prot"]), 1, 0)]
        anchors = {"ile44": project(c, "prot and resi 44 and name CD1")}
    return layers, anchors


def scene_cofactor(work, src):
    import pymol2

    with pymol2.PyMOL() as pm:
        c = pm.cmd
        pymol_style(c)
        c.load(str(src / "1A6M.pdb"), "sys")
        c.remove("resn HOH or hydro")
        c.create("prot", "sys and polymer.protein")
        c.create("heme", "sys and resn HEM")
        c.create("his", "sys and polymer.protein and resi 64+93 and not name N+C+O")
        c.delete("sys")
        c.hide("everything")
        c.show("cartoon", "prot")
        c.color("g_blue", "prot")
        c.set("cartoon_transparency", .35, "prot")
        c.show("sticks", "heme or his")
        c.show("spheres", "heme and elem Fe")
        c.set("sphere_scale", .55, "heme and elem Fe")
        c.set("stick_radius", .2)
        c.color("g_gold", "heme and elem C")
        c.color("g_oxygen", "(heme or his) and elem O")
        c.color("g_nitrogen", "(heme or his) and elem N")
        c.color("g_iron", "heme and elem Fe")
        c.color("g_silver", "his and elem C")
        c.orient("heme")
        c.turn("x", -55)
        c.turn("y", 20)
        fit(c, "heme or his", (1700, 330, 3500, 1750))              # close-up on the heme pocket
        layers = [("focus", ray(c, work, "focus", ["prot", "heme", "his"]), 1, 0)]
        anchors = {"fe": project(c, "heme and elem Fe"), "his": project(c, "his and resi 93 and name NE2")}
    return layers, anchors


def scene_orbitals(work, src):
    import pymol2

    orb = src / "orb"
    panels = []
    for name, cube in (("homo", "Psi_a_15_15-A.cube"), ("lumo", "Psi_a_16_16-A.cube")):
        with pymol2.PyMOL() as pm:
            c = pm.cmd
            pymol_style(c)
            c.load(str(orb / "butadiene.xyz"), "mol")
            c.load(str(orb / cube), "cube")
            c.isosurface("pos", "cube", .035)
            c.isosurface("neg", "cube", -.035)
            c.color("g_pos", "pos")
            c.color("g_neg", "neg")
            c.set("transparency", .12)
            c.show_as("sticks", "mol")
            c.show("spheres", "mol")
            c.set("sphere_scale", .22, "mol")
            c.set("stick_radius", .14)
            c.color("g_silver", "mol and elem C")
            c.color("white", "mol and elem H")
            c.orient("mol")
            c.turn("x", -58)
            c.turn("z", 8)
            c.zoom("mol", 3.2)
            panels.append(ray(c, work, name, ["mol", "pos", "neg"]))
    w, h = panels[0].size
    sheet = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for i, p in enumerate(panels):
        p = p.resize((round(w * .62), round(h * .62)), Image.Resampling.LANCZOS)
        sheet.alpha_composite(p, (round(w * (.02 + .38 * i)), round(h * (.10 + .30 * i))))
    sheet.save(work / "focus.png")
    # label anchors on the upper lobes of each panel (panel placement above, in layout units)
    return [("focus", sheet, 1, 0)], {"homo": [MX + MW * .43, MY + MH * .33],
                                     "lumo": [MX + MW * .70, MY + MH * .62]}


def scene_sketcher(work, src):
    """The Ligand Editor itself (Qt widget rendered offscreen) with imatinib loaded."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import sys

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "legacy")]
    from pymol.Qt import QtWidgets
    from rdkit import Chem

    from leonardo import ligand_sketcher_runtime as ls

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = ls.LigandSketcherWidget()
    widget.resize(1300, 860)
    mol = Chem.MolFromSmiles("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1")
    for setter in ("set_molecule", "load_molecule", "setMolecule"):
        if hasattr(widget, setter):
            getattr(widget, setter)(mol)
            break
    widget.show()
    app.processEvents()
    shot = work / "sketcher_raw.png"
    widget.grab().save(str(shot))
    img = Image.open(shot).convert("RGBA")
    scale = min(MW * S * .86 / img.width, MH * S * .78 / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=round(22 * S * 2), fill=255)
    img.putalpha(mask)
    canvas = Image.new("RGBA", (round(MW * S), round(MH * S)), (0, 0, 0, 0))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 200))
    shadow.putalpha(mask.point(lambda v: v * 0.6))
    ox, oy = round((canvas.width - img.width) / 2 + 40 * S), round((canvas.height - img.height) / 2)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(24 * S)), (ox + 14, oy + 22))
    canvas.alpha_composite(img, (ox, oy))
    return [("focus", canvas, 1, 0)], {}


def scene_promo(kind):
    """Backplates of the earlier 4K series (docking, MD, QM/MM), scaled to this size."""
    folder = PROMO / kind
    settings = {"docking": [("surface", .55, 2), ("ribbon", .16, 1.6), ("focus", 1, 0)],
                "md": [("surface", .10, 2.4), ("traces", .13, .45), ("focus", 1, 0)],
                "qmmm": [("surface", .13, 4), ("ribbon", .27, 2.5), ("focus", 1, 0)]}[kind]
    layers = []
    for name, opacity, blur in settings:
        im = Image.open(folder / f"{name}.png").convert("RGBA")
        im = im.resize((round(MW * S), round(MH * S)), Image.Resampling.LANCZOS)
        layers.append((name, im, opacity, blur * S))
    meta = json.loads((folder / "geometry.json").read_text())
    return layers, meta.get("anchors", {})


# ------------------------------------------------------------------------------------ poster


def leader(d, start, end, label, color=BLUE, align="left"):
    x, y = start
    ax, ay = end
    tw = d.textlength(label, font=font(27)) / S
    text(d, (x, y), label, 27, color)
    if align == "left":
        ex, ey = x + tw + 20, y + 20
        pts = [ex, ey, ex + 35, ey, ax, ay]
    else:
        ex, ey = x - 18, y + 20
        pts = [ex, ey, ex - 35, ey, ax, ay]
    d.line([(round(pts[i] * S), round(pts[i + 1] * S)) for i in range(0, 6, 2)], fill=(*color, 150), width=1)
    r = 4 * S * 2
    d.ellipse((ax * S - r, ay * S - r, ax * S + r, ay * S + r), fill=color)


def compose(spec, layers, anchors, dest):
    im = background(spec.get("warm", True))
    for name, layer, opacity, blur in layers:
        if blur:
            layer = layer.filter(ImageFilter.GaussianBlur(blur))
        if name == "focus":
            glow = layer.copy().filter(ImageFilter.GaussianBlur(16 * S))
            glow.putalpha(glow.getchannel("A").point(lambda v: round(v * .13)))
            im.alpha_composite(glow, (round(MX * S), round(MY * S)))
        else:
            lh, lw = layer.height, layer.width
            yy, xx = np.mgrid[0:lh, 0:lw]
            fade = np.minimum(np.clip(yy / (130 * S), 0, 1), np.clip((lh - 1 - yy) / (240 * S), 0, 1))
            fade *= np.minimum(np.clip(xx / (100 * S), 0, 1), np.clip((lw - 1 - xx) / (100 * S), 0, 1))
            layer.putalpha(Image.fromarray(np.uint8(np.asarray(layer.getchannel("A")) * fade), "L"))
        layer.putalpha(layer.getchannel("A").point(lambda v, o=opacity: round(v * o)))
        im.alpha_composite(layer, (round(MX * S), round(MY * S)))
    ww, hh = im.size
    x = np.arange(ww, dtype=float) / S
    mask = np.clip((1480 - x) / 460, 0, 1) * .98
    shade = Image.new("RGBA", im.size, (5, 12, 21, 0))
    shade.putalpha(Image.fromarray(np.uint8(np.tile(mask * 255, (hh, 1))), "L"))
    im.alpha_composite(shade)
    d = ImageDraw.Draw(im)
    accent = spec["accent"]
    tracked(d, (160, 126), "LEONARDO", 53, WHITE, 12)
    text(d, (164, 209), "MOLECULAR MODELING SUITE", 22, MUTED)
    tracked(d, (2895, 145), "MOLECULAR SERIES", 22, MUTED, 4)
    text(d, (3560, 112), spec["number"], 78, accent, "serif")
    hline(d, (160, 295, 3680, 295), (60, 81, 94))
    tracked(d, (164, 500), spec["module"], 26, accent, 4)
    for i, t in enumerate(spec["title"]):
        text(d, (151, 591 + i * 154), t, 133, WHITE, "bold")
    for i, t in enumerate(spec["sub"]):
        text(d, (164, 968 + i * 55), t, 39, MUTED)
    hline(d, (164, 1162, 270, 1162), accent, 4)
    text(d, (164, 1214), spec["method"], 29, WHITE)
    text(d, (164, 1271), spec["specimen"], 22, MUTED)
    text(d, (158, 1520), spec["metric"], 100, accent, "serif")
    text(d, (164, 1657), spec["metric_label"], 28, MUTED)
    if spec.get("metric_note"):
        text(d, (164, 1720), spec["metric_note"], 27, WHITE)
    for lab in spec.get("labels", []):
        if lab["anchor"] in anchors:
            leader(d, lab["at"], anchors[lab["anchor"]], lab["text"], lab.get("color", BLUE), lab.get("align", "left"))
    tracked(d, (1800, 1838), spec["tag"], 21, MUTED, 3)
    hline(d, (160, 1975, 3680, 1975), (60, 81, 94))
    text(d, (164, 2020), spec["note"], 25, MUTED)
    text(d, (3225, 2020), "leonardo · " + spec["number"], 25, WHITE)
    im.convert("RGB").save(dest, quality=90, optimize=True, progressive=True)
    print("saved", dest, flush=True)


# ------------------------------------------------------------------------------------ specs


def specs(src):
    mem = {}
    report = src / "membrane" / "membrane_report.json"
    if report.is_file():
        mem = json.loads(report.read_text())
    thick = (mem.get("membrane") or {}).get("thickness_PP_nm")
    counts = (mem.get("packing") or {}).get("counts") or {}
    gold = json.loads((src / "gold" / "surface_report.json").read_text()) if (src / "gold" / "surface_report.json").is_file() else {}
    height = (gold.get("surface_analysis") or {}).get("site_height_above_surface_nm")
    return {
        "docking": dict(number="01", module="MOLECULAR DOCKING", title=["The chemistry", "of a perfect fit."],
                        sub=["See how a drug recognizes", "its protein target."], method="smina / Vina",
                        specimen="TRYPSIN + BENZAMIDINE", metric="0.40 Å", metric_label="RMSD to the crystal pose",
                        metric_note="Vina score: −5.96 kcal/mol",
                        note="Rigid-receptor redocking · best pose by score · PDB 3PTB",
                        tag="THE BINDING SITE, IN DETAIL", accent=GOLD,
                        labels=[dict(anchor="ligand", at=(2930, 760), text="Benzamidinium · charge +1", color=GOLD, align="right"),
                                dict(anchor="asp", at=(1680, 570), text="Asp189 · carboxylate")]),
        "md": dict(number="02", module="MOLECULAR DYNAMICS", title=["Structures", "come alive."],
                   sub=["Watch proteins move,", "atom by atom."], method="OpenMM · Amber14 / OBC-II",
                   specimen="TRYPSIN", metric="300 K", metric_label="120 saved states · 12 ps of production",
                   note="Short relaxation · implicit solvent · 2 fs integration · PDB 3PTB",
                   tag="ONE STRUCTURE. MANY CONFORMATIONS.", accent=BLUE, warm=False),
        "qmmm": dict(number="03", module="QUANTUM CHEMISTRY · QM/MM", title=["At the edge", "of a reaction."],
                     sub=["Follow bonds breaking and forming", "inside an enzyme."], method="MOPAC PM7 / Amber14",
                     specimen="HALOALKANE DEHALOGENASE · DhlA", metric="8.41 kcal/mol",
                     metric_label="Potential-energy barrier of the model", metric_note="One imaginary mode: 425.46i cm\u2212\u00b9",
                     note="Transition state with 33 mobile QM coordinates · PDB 2DHC",
                     tag="TRANSITION STATE · SN2 STEP", accent=GOLD,
                     labels=[dict(anchor="CB", at=(1690, 640), text="Asp124"),
                             dict(anchor="CL1", at=(3060, 1090), text="Cl · leaving group", color=GOLD, align="right")]),
        "membrane": dict(number="04", module="MEMBRANE PROTEINS", title=["A protein in", "its membrane."],
                         sub=["Embed a membrane protein in a realistic", "lipid mixture in one step."],
                         method="OPM orientation · PACKMOL-Memgen · Amber Lipid17",
                         specimen="BACTERIORHODOPSIN TRIMER · POPC : CHOLESTEROL 3 : 1",
                         metric=f"{thick:.2f} nm" if thick else "POPC:CHL1",
                         metric_label="Phosphate-to-phosphate thickness after minimization",
                         metric_note=(f"{counts.get('POPC', '?')} POPC · {counts.get('CHL1', '?')} cholesterol"
                                      if counts else ""),
                         note="PDB 1C3W oriented from OPM · water and ions hidden",
                         tag="CHOLESTEROL IN GOLD", accent=BLUE, warm=False,
                         labels=[dict(anchor="chol", at=(1600, 1780), text="Cholesterol", color=GOLD)]),
        "gold": dict(number="05", module="BIO-INORGANIC INTERFACES", title=["Proteins", "meet materials."],
                     sub=["Place a protein on gold, graphene or silica", "with its active site toward the solution."],
                     method="INTERFACE-FF · OpenMM", specimen="UBIQUITIN ON Au(111)",
                     metric=f"{height:.1f} nm" if height else "Au(111)",
                     metric_label="Height of Ile44 above the gold surface",
                     note="Frozen Au(111) slab · water and ions hidden · PDB 1UBQ",
                     tag="", accent=GOLD,
                     labels=[dict(anchor="ile44", at=(1620, 520), text="Ile44 · faces the solution", color=GOLD)]),
        "cofactor": dict(number="06", module="COFACTORS & METALS", title=["Every piece,", "parameterized."],
                         sub=["Hemes, metals, nucleotides and modified residues", "get force-field parameters automatically."],
                         method="Component library · CHARMM36 / GAFF2", specimen="MYOGLOBIN · HEME b",
                         metric="−2 e", metric_label="Heme charge assigned from its chemistry",
                         metric_note="Fe–His93 kept with a coordination restraint",
                         note="PDB 1A6M · heme b, proximal His93 and distal His64",
                         tag="THE HEME POCKET", accent=GOLD,
                         labels=[dict(anchor="fe", at=(3420, 380), text="Fe · heme b", color=GOLD, align="right"),
                                 dict(anchor="his", at=(1700, 1640), text="His93 · proximal")]),
        "orbitals": dict(number="07", module="QUANTUM CHEMISTRY", title=["See the", "electrons."],
                         sub=["Molecular orbitals, spectra and reaction", "paths with Psi4 and MOPAC."],
                         method="Psi4 · B3LYP / def2-SVP", specimen="1,3-BUTADIENE · HOMO AND LUMO",
                         metric="5.55 eV", metric_label="HOMO–LUMO gap (−6.48 → −0.94 eV)",
                         note="Optimized geometry · isosurfaces at ±0.035 a.u.",
                         tag="π2 (HOMO)  ·  π3* (LUMO)", accent=BLUE, warm=False,
                         labels=[dict(anchor="homo", at=(2420, 470), text="HOMO · π2"),
                                 dict(anchor="lumo", at=(2380, 1110), text="LUMO · π3*")]),
        "sketcher": dict(number="08", module="LIGAND EDITOR", title=["Draw it.", "Dock it."],
                         sub=["Sketch a molecule and get it in 3D", "with the right protonation."],
                         method="RDKit · ETKDG · MMFF", specimen="IMATINIB",
                         metric="One click", metric_label="From the drawing to a 3D ligand with hydrogens",
                         metric_note="Ready for docking and parameterization",
                         note="Stereo, charges and valence checked as you draw",
                         tag="", accent=GOLD),
    }


def hero(names=("04_membrane", "05_gold", "07_orbitals", "03_qmmm")):
    """2x2 mosaic of posters for the top of the README."""
    tiles = [Image.open(OUT / f"{n}.jpg").convert("RGB") for n in names if (OUT / f"{n}.jpg").is_file()]
    if len(tiles) < 4:
        return
    tw, th, gap = 960, 540, 6
    sheet = Image.new("RGB", (2 * tw + gap, 2 * th + gap), (5, 12, 21))
    for i, t in enumerate(tiles[:4]):
        sheet.paste(t.resize((tw, th), Image.Resampling.LANCZOS), ((i % 2) * (tw + gap), (i // 2) * (th + gap)))
    sheet.save(OUT / "hero.jpg", quality=88, optimize=True, progressive=True)
    print("saved", OUT / "hero.jpg", flush=True)


SCENES = {"membrane": scene_membrane, "gold": scene_gold, "cofactor": scene_cofactor,
          "orbitals": scene_orbitals, "sketcher": scene_sketcher}
ORDER = ["docking", "md", "qmmm", "membrane", "gold", "cofactor", "orbitals", "sketcher"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=ORDER)
    args = ap.parse_args()
    sp = specs(args.work)
    for kind in args.only:
        work = args.work / "renders" / kind
        work.mkdir(parents=True, exist_ok=True)
        if kind in SCENES:
            layers, anchors = SCENES[kind](work, args.work)
        else:
            layers, anchors = scene_promo(kind)
        compose(sp[kind], layers, anchors, OUT / f"{sp[kind]['number']}_{kind}.jpg")
    hero()


if __name__ == "__main__":
    main()
