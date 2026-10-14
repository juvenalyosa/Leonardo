#!/usr/bin/env python3
"""48-second, six-scene film: one fold, five double-mutant designs."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from functools import lru_cache

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/leonardo-mutation-mpl')
import numpy as np
import pymol2
import render_cinematic_stability as single
import render_cinematic_v2 as style
from cinematic_stability_multimutant import OUT, geometry, load, matrix_plot, torsion_plot
from PIL import Image, ImageDraw, ImageOps

NAME = '07_protein_stability_multimutant'
FRAMES = 1152
FPS = 24
W, H = 1920, 1080
FOLDER = OUT / 'renders/multimutant'
ASSETS = FOLDER / 'assets'
TEAL, GOLD, INK, MUTED, LINE = style.TEAL, style.GOLD, style.INK, style.MUTED, style.LINE
txt = style.txt
ease = style.ease
PREVIEW = [0, 96, 168, 252, 300, 348, 396, 444, 492, 540, 588, 636, 684, 732, 780, 840, 936, 1032, 1128, 1151]


def setup(c, row, original=False):
    variant = 'original' if original else row['folder']
    views = single.setup(c, {'residue_pair': row['pair']}, variant)
    c.delete('candidate')
    if not original:
        i, j = row['pair']
        c.bond(f'site and resi {i} and name SG', f'site and resi {j} and name SG')
    return views


def update(c, row, idx):
    data = row['states']
    xyz = data['xyz'][idx]
    c.load_coords(xyz, 'protein')
    c.load_coords(xyz, 'ribbon')
    c.load_coords(xyz[data['heavy']], 'envelope')
    ids = [
        a['index']
        for a in row['atoms']
        if a['resid'] in row['pair'] and a['element'] != 'H' and a['name'] not in ['N', 'C', 'O', 'OXT']
    ]
    c.load_coords(xyz[ids], 'site')


def make_assets(group, preview=False):
    data = load()
    ASSETS.mkdir(parents=True, exist_ok=True)
    scratch = ASSETS / f'worker_{os.getpid()}'
    scratch.mkdir(exist_ok=True)
    started = time.perf_counter()
    chosen = data['mutants'][-1]
    with pymol2.PyMOL() as p:
        wide, close = setup(p.cmd, chosen)
        common_wide = wide.copy()
        if group in ['all', 'hero']:
            setup(p.cmd, chosen, True)
            frames = [0, 96, 119] if preview else range(120)
            for f in frames:
                dest = ASSETS / f'hero_{f:03d}.png'
                if dest.exists():
                    continue
                c = p.cmd
                c.set_view(common_wide.tolist())
                c.turn('y', -18 + f * 0.30)
                c.turn('x', -5)
                style.ray(c, scratch, 0).save(dest)
                if f % 24 == 0:
                    print('hero assets', f, '/120', round(time.perf_counter() - started, 1), 's', flush=True)
        if group in ['all', 'candidates']:
            for q, row in enumerate(data['mutants']):
                wide, close = setup(p.cmd, row)
                update(p.cmd, row, 30)
                p.cmd.set_view(common_wide.tolist())
                dest = ASSETS / f'wide_{q}.png'
                if not dest.exists():
                    style.ray(p.cmd, scratch, 0).save(dest)
                p.cmd.set_view(close.tolist())
                # Native sessions retain their actual final structures and shared style.
                p.cmd.set('transparency', 0.85, 'envelope')
                p.cmd.save(str(OUT / 'sessions' / f'multimutant_{row["folder"]}.pse'))
                p.cmd.set('transparency', 0, 'envelope')
                indices = [0, 18, 30] if preview else range(31)
                for idx in indices:
                    dest = ASSETS / f'close_{q}_{idx:02d}.png'
                    if dest.exists():
                        continue
                    update(p.cmd, row, idx)
                    style.ray(p.cmd, scratch, 0.70).save(dest)
                print('candidate assets', row['label'], round(time.perf_counter() - started, 1), 's', flush=True)
        if group in ['all', 'final']:
            wide, close = setup(p.cmd, chosen)
            update(p.cmd, chosen, 30)
            single.network(
                p.cmd,
                {
                    'structures': {'design': chosen['states']},
                    'network': {'original_kirchhoff': data['reference']['kirchhoff']},
                },
            )
            frames = [0, 48, 71] if preview else range(72)
            for f in frames:
                dest = ASSETS / f'network_{f:03d}.png'
                if dest.exists():
                    continue
                blend = 0.90 - 0.76 * ease(f / 71)
                view = wide * (1 - blend) + close * blend
                p.cmd.set_view(view.tolist())
                p.cmd.turn('y', 12 * ease(f / 71))
                style.ray(p.cmd, scratch, 0.4 * blend).save(dest)
                if f % 24 == 0:
                    print('network assets', f, '/72', round(time.perf_counter() - started, 1), 's', flush=True)
    print('Asset group complete:', group, flush=True)


@lru_cache(maxsize=24)
def asset(name, trim=False):
    im = Image.open(ASSETS / name).convert('RGB')
    if trim:
        a = np.asarray(im)
        mask = np.min(a, axis=2) < 235
        y, x = np.where(mask)
        im = im.crop(
            (max(0, x.min() - 35), max(0, y.min() - 35), min(im.width, x.max() + 36), min(im.height, y.max() + 36))
        )
    return im


def fit(im, box):
    return ImageOps.contain(im, box, Image.Resampling.LANCZOS)


def soften(im, top=70, bottom=45):
    """Feather the context image into the white stage, without changing atoms."""
    y = np.arange(im.height)
    alpha = np.minimum(np.clip(y / top, 0, 1), np.clip((im.height - 1 - y) / bottom, 0, 1))
    mask = Image.fromarray(np.tile((255 * alpha).astype(np.uint8)[:, None], (1, im.width)))
    return Image.composite(im, Image.new('RGB', im.size, 'white'), mask)


def base(seconds, chapter, scope):
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    txt(d, (78, 42), 'L E O N A R D O', 25, bold=True)
    txt(d, (1510, 46), 'MOLECULAR FILMS  /  07', 18, MUTED)
    d.line((78, 93, 1842, 93), fill=LINE, width=1)
    txt(d, (80, 128), chapter, 18, TEAL, True)
    return im


def footer(im, seconds, scope):
    d = ImageDraw.Draw(im)
    d.rectangle((0, 1000, W, H), fill='white')
    d.line((78, 1007, 1842, 1007), fill=LINE, width=1)
    d.line((78, 1007, 78 + int(1764 * seconds / ((FRAMES - 1) / FPS)), 1007), fill=TEAL, width=3)
    txt(d, (80, 1030), scope, 18, MUTED)
    txt(d, (1660, 1030), 'PDB 1PGA', 18, MUTED)


def draw_sequence(d, box, pair, active=1):
    x, y, width = box
    step = width / 55
    d.line((x, y, x + width, y), fill=LINE, width=2)
    for n in range(1, 57):
        px = x + (n - 1) * step
        color = GOLD if n in pair else '#BBCDD2'
        radius = 5 if n in pair else 2
        d.ellipse((px - radius, y - radius, px + radius, y + radius), fill=color)
    for n in pair:
        txt(d, (x + (n - 1) * step - 8, y + 14), str(n), 17, GOLD, True)
    txt(d, (x, y - 35), 'SEQUENCE POSITIONS', 15, MUTED)


def headline(d, kicker, title):
    txt(d, (76, 166), title, 48, bold=True)


def hero(data, t, preview=False):
    im = base(t, '07 / PROTEIN DESIGN', '')
    d = ImageDraw.Draw(im)
    f = min(119, int(t * 24))
    f = min([0, 96, 119], key=lambda n: abs(n - f)) if preview else f
    molecule = asset(f'hero_{f:03d}.png').resize((1536, 864), Image.Resampling.LANCZOS)
    im.paste(molecule, (540, 132))
    d = ImageDraw.Draw(im)
    txt(d, (80, 279), 'One fold.', 85, bold=True)
    txt(d, (80, 386), 'Five futures.', 85, bold=True)
    txt(d, (86, 533), 'EXPLORING PROTEIN STABILITY BY DESIGN', 21, MUTED)
    txt(d, (86, 576), 'GB1 · five double mutants · one selection', 23)
    d.line((86, 650, 181, 650), fill=GOLD, width=4)
    txt(d, (86, 690), 'Propose. Refine. Compare.', 27, bold=True)
    txt(d, (86, 738), 'Let the molecular geometry decide.', 23, MUTED)
    txt(d, (1450, 919), '56 RESIDUES', 20, TEAL, True)
    footer(im, t, 'Five computational designs · structural screening with lightweight calculations')
    return im


def portfolio(data, t, preview=False):
    im = base(t, '01 / PROPOSE · FIVE DISTINCT CONNECTIONS', '')
    d = ImageDraw.Draw(im)
    headline(d, '', 'A small library. Five possible locks.')
    txt(d, (82, 234), 'Two substitutions per design. The same 56-residue fold.', 23, MUTED)
    for q, r in enumerate(data['mutants']):
        x = 80 + q * 354
        delay = q * 0.16
        progress = ease((t - 5 - delay) / 0.7)
        dy = int(45 * (1 - progress))
        card = Image.new('RGB', (342, 624), 'white')
        cd = ImageDraw.Draw(card)
        cd.line((10, 0, 330, 0), fill=LINE, width=1)
        txt(cd, (18, 27), f'DESIGN 0{q+1}', 16, MUTED)
        txt(cd, (18, 61), r['label'], 27, bold=True)
        scan = float(np.clip(1 - abs((t - 6) / 0.82 - q), 0, 1))
        model = fit(asset(f'wide_{q}.png', True), (int(312 + 12 * scan), int(370 + 15 * scan)))
        card.paste(model, ((342 - model.width) // 2, 118 + (385 - model.height) // 2))
        if scan > 0:
            cd.line((18, 104, 18 + int(302 * scan), 104), fill=GOLD, width=2)
        draw_sequence(cd, (22, 536, 296), r['pair'])
        txt(cd, (20, 595), 'PROPOSED S–S LINK', 15, GOLD, True)
        if progress < 1:
            card = Image.blend(Image.new('RGB', card.size, 'white'), card, progress)
        im.paste(card, (x, 322 + dy))
    d = ImageDraw.Draw(im)
    footer(im, t, 'Oxidized double-mutant models · gold marks the proposed disulfide')
    return im


def showcase(data, t, plots, preview=False):
    q = min(4, int((t - 11) / 4))
    local = t - (11 + q * 4)
    row = data['mutants'][q]
    idx = min(30, int(max(0, (local - 0.25)) / 2.3 * 30))
    idx = min([0, 18, 30], key=lambda n: abs(n - idx)) if preview else idx
    im = base(t, f'02 / REFINE · DESIGN 0{q+1} OF 05', '')
    mol = asset(f'close_{q}_{idx:02d}.png').resize((1168, 657), Image.Resampling.LANCZOS).crop((0, 0, 1168, 596))
    im.paste(soften(mol, bottom=75), (-18, 230))
    d = ImageDraw.Draw(im)
    d.rectangle((1030, 235, 1920, 999), fill='white')
    headline(d, '', f'{row["label"]} · inspect the connection.')
    txt(d, (1091, 273), 'Five geometries. One shared filter.', 27, bold=True)
    txt(d, (1092, 319), 'Final geometry after local refinement', 20, MUTED)
    plot_im = plots[q]
    im.paste(plot_im.resize((828, 537), Image.Resampling.LANCZOS), (1030, 344))
    d = ImageDraw.Draw(im)
    # Place readings below the molecule, outside the ray image.
    d.rectangle((0, 826, 1046, 999), fill='white')
    g = geometry(row['states']['xyz'][idx], row['atoms'], row['pair'], data['reference']['coords'])
    for x, label, value, unit in [
        (84, 'S–S DISTANCE', f'{g["SS_A"]:.2f}', 'Å'),
        (420, '|χ3| TORSION', f'{abs(g["chi3_deg"]):.1f}', '°'),
        (765, 'Cα SHIFT', f'{g["ca_displacement_rms_A"]:.2f}', 'Å'),
    ]:
        txt(d, (x, 849), label, 16, TEAL, True)
        txt(d, (x - 4, 880), value, 54, bold=True)
        width = d.textbbox((0, 0), value, font=style.font(54, True))[2]
        txt(d, (x + width + 6, 903), unit, 24, MUTED)
    txt(d, (87, 965), f'SAVED MINIMIZATION BATCH  {idx:02d} / 30', 17, MUTED)
    status = (
        'FINAL SCREEN: PASSES · ALL FOUR CHECKS'
        if row['passed_geometry_filter']
        else 'FINAL SCREEN: FAILS · '
        + ' AND '.join(
            n
            for n, ok in zip(['S–S DISTANCE', 'C–S–S ANGLES', 'TORSION', 'Cα SHIFT'], row['geometry_checks'])
            if not ok
        )
        + ' OUTSIDE WINDOW'
    )
    txt(d, (1091, 917), status, 17, GOLD if row['passed_geometry_filter'] else '#A07565', True)
    txt(d, (1091, 953), 'Window for |χ3|: 65–115°', 19, MUTED)
    footer(im, t, 'Amber14 / OBC-II · restrained minimization · saved structures, not a dynamics trajectory')
    # Brief clean dissolve between candidate views, preserving readable plots.
    if local < 0.20:
        white = Image.new('RGB', im.size, 'white')
        im = Image.blend(white, im, 0.65 + 0.35 * ease(local / 0.2))
    return im


def selection(data, t, matrix, preview=False):
    im = base(t, '03 / COMPARE · THE COMPLETE STRUCTURAL SCREEN', '')
    d = ImageDraw.Draw(im)
    headline(d, '', 'A good bond length is only the beginning.')
    n_pass = sum(r['passed_geometry_filter'] for r in data['mutants'])
    count = {1: 'One passes', 2: 'Two pass', 3: 'Three pass', 4: 'Four pass', 5: 'All five pass'}.get(
        n_pass, 'None passes'
    )
    txt(d, (81, 235), f'All five retain similar backbone geometry. {count} every check.', 24, MUTED)
    im.paste(matrix.resize((1315, 664), Image.Resampling.LANCZOS), (505, 287))
    model = fit(asset('wide_4.png', True), (436, 468))
    im.paste(model, (85 + (436 - model.width) // 2, 331 + (468 - model.height) // 2))
    chosen = data['mutants'][-1]
    tag = f'SELECTED · {n_pass} OF 05 PASS' if n_pass > 1 else '01 / 05  RETAINED'
    d = ImageDraw.Draw(im)
    txt(d, (87, 824), chosen['label'], 36, bold=True)
    txt(d, (89, 879), tag, 20, GOLD, True)
    footer(im, t, 'Selection applies to structural compatibility · experimental stabilization has not been measured')
    return im


def finalist(data, t, mobility, preview=False):
    im = base(t, '04 / SELECT · A STRUCTURALLY COMPATIBLE PROPOSAL', '')
    idx = min(71, int((t - 37) * 24))
    idx = min([0, 48, 71], key=lambda n: abs(n - idx)) if preview else idx
    mol = asset(f'network_{idx:03d}.png').resize((1008, 567), Image.Resampling.LANCZOS)
    im.paste(soften(mol, top=25, bottom=30), (32, 264))
    d = ImageDraw.Draw(im)
    d.rectangle((1050, 238, 1920, 999), fill='white')
    headline(d, '', 'One connection. A change in local motion.')
    txt(d, (85, 259), f"{data['mutants'][-1]['label']} · CONTACT-NETWORK VIEW", 18, GOLD, True)
    txt(d, (1090, 277), 'The selected design, quantified.', 28, bold=True)
    txt(d, (1090, 322), 'Gaussian network · conditional prediction', 20, MUTED)
    im.paste(mobility.resize((805, 574), Image.Resampling.LANCZOS), (1047, 348))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 835, 1048, 999), fill='white')
    txt(d, (86, 852), 'LOCAL RELATIVE RMS MOTION', 17, TEAL, True)
    chosen = data['mutants'][-1]
    txt(d, (78, 878), f'−{chosen["local_rms_reduction_percent"]:.0f}%', 71, bold=True)
    txt(d, (320, 923), f'between residues {chosen["pair"][0]} and {chosen["pair"][1]}', 24, MUTED)
    txt(
        d,
        (86, 969),
        f'Whole-domain RMS reduction: {chosen["global_rms_reduction_percent"]:.2f}% in the same model',
        20,
        MUTED,
    )
    txt(d, (1090, 950), '7.5 Å contacts · additional spring stiffness 10×', 18, MUTED)
    footer(im, t, 'Harmonic rigidity prediction · neither folding ΔΔG nor a measured increase in thermal stability')
    return im


def outro(data, t, preview=False):
    im = base(t, '05 / DESIGN WITH EVIDENCE', '')
    d = ImageDraw.Draw(im)
    mol = asset('wide_4.png', True)
    size = int(640 + 20 * ease((t - 44) / 4))
    mol = fit(mol, (size, 650))
    im.paste(mol, (1170 + (650 - mol.width) // 2, 270 + (650 - mol.height) // 2))
    d = ImageDraw.Draw(im)
    txt(d, (78, 287), 'Explore more.', 76, bold=True)
    txt(d, (78, 383), 'Choose with evidence.', 76, bold=True)
    n_pass = sum(r['passed_geometry_filter'] for r in data['mutants'])
    txt(
        d,
        (85, 515),
        'FIVE DESIGNS. ONE STRUCTURAL CANDIDATE.'
        if n_pass <= 1
        else f'FIVE DESIGNS. {n_pass} PASS. THE BEST |χ3| IS SELECTED.',
        21,
        TEAL,
        True,
    )
    txt(d, (83, 563), data['mutants'][-1]['label'], 45, bold=True)
    txt(d, (86, 633), 'A proposed molecular lock for the GB1 fold.', 26, MUTED)
    for q, row in enumerate(data['mutants']):
        x = 87 + q * 204
        color = GOLD if q == 4 else '#A7BCC3'
        d.line((x, 752, x + 169, 752), fill=color, width=3 if q == 4 else 1)
        txt(d, (x, 774), row['label'], 20, color, True)
        txt(
            d,
            (x, 812),
            ('SELECTED' if n_pass > 1 else 'RETAINED')
            if q == 4
            else ('PASSES' if row['passed_geometry_filter'] and n_pass > 1 else 'SCREENED'),
            14,
            color,
        )
    txt(d, (87, 906), 'LEONARDO  /  MOLECULAR DESIGN, MADE VISIBLE', 22, bold=True)
    footer(
        im,
        t,
        'A computational proposal for experimental testing · five structures, comparative plots and reproducible data',
    )
    return im


def render(preview=False, start=0, stop=FRAMES):
    FOLDER.mkdir(exist_ok=True)
    data = load()
    plots = [torsion_plot(data, q) for q in range(5)]
    matrix = matrix_plot(data)
    # Same normalization and model as the previously verified single-candidate film.
    chosen = data['mutants'][-1]
    if 'pair_variance_original' in chosen:
        # the selected design's own network model (same normalization as the single-design film)
        net = {
            'resids': np.arange(1, len(data['reference']['mobility']) + 1),
            'original_mobility': data['reference']['mobility'],
            'design_mobility': chosen['network']['mobility'],
        }
        mobility = single.plot(
            {
                'network': net,
                'residue_pair': chosen['pair'],
                'pair_variance_original': chosen['pair_variance_original'],
                'pair_variance_design': chosen['pair_variance_design'],
            }
        )
    else:
        mobility = single.plot(single.load())
    frames = PREVIEW if preview else range(start, min(stop, FRAMES))
    started = time.perf_counter()
    for f in frames:
        dest = FOLDER / f'frame_{f:05d}.png'
        if dest.exists() and not preview:
            continue
        t = f / FPS
        if t < 5:
            im = hero(data, t, preview)
        elif t < 11:
            im = portfolio(data, t, preview)
        elif t < 31:
            im = showcase(data, t, plots, preview)
        elif t < 37:
            im = selection(data, t, matrix, preview)
        elif t < 44:
            im = finalist(data, t, mobility, preview)
        else:
            im = outro(data, t, preview)
        # Preview files never enter the production frame sequence.
        if preview:
            dest = FOLDER / f'preview_{f:05d}.png'
        im.save(dest)
        if f % 96 == 0 or preview:
            print('compose', f, '/1152', round(time.perf_counter() - started, 1), 's', flush=True)
    if preview:
        sheet = Image.new('RGB', (1920, 864), 'white')
        for q, f in enumerate(PREVIEW):
            sheet.paste(
                Image.open(FOLDER / f'preview_{f:05d}.png').resize((384, 216), Image.Resampling.LANCZOS),
                ((q % 5) * 384, (q // 5) * 216),
            )
        sheet.save(FOLDER / 'preview_sheet.jpg', quality=95)


def encode():
    assert all((FOLDER / f'frame_{i:05d}.png').exists() for i in range(FRAMES))
    p = subprocess.run(
        [
            '/opt/homebrew/bin/ffmpeg',
            '-y',
            '-framerate',
            '24',
            '-i',
            str(FOLDER / 'frame_%05d.png'),
            '-c:v',
            'libx264',
            '-preset',
            'slow',
            '-crf',
            '16',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
            '-an',
            str(OUT / 'videos' / f'{NAME}.mp4'),
        ],
        capture_output=True,
        text=True,
    )
    (FOLDER / 'ffmpeg.log').write_text(p.stderr)
    assert p.returncode == 0, p.stderr[-1000:]
    print('Encoded', NAME, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['assets', 'compose', 'encode'])
    p.add_argument('--group', default='all', choices=['all', 'hero', 'candidates', 'final'])
    p.add_argument('--preview', action='store_true')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--stop-frame', type=int, default=FRAMES)
    a = p.parse_args()
    if a.stage == 'assets':
        make_assets(a.group, a.preview)
    elif a.stage == 'compose':
        render(a.preview, a.start_frame, a.stop_frame)
    else:
        encode()
