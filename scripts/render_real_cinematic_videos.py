#!/usr/bin/env python3
"""Ray-traced PyMOL films using only saved calculated coordinates and results.

Camera motion is cosmetic; molecular states are never interpolated. All plots
are direct CSV/NPZ values. Native PyMOL sessions and vector plots are exported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymol2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cinematic_real'
W, H = 1920, 1080
MW, MH = 1180, 780
INK = '#16303F'
MUTED = '#657C88'
TEAL = '#25858B'
GOLD = '#C38C37'
BLUE = '#397DB1'
RED = '#CD5C55'
LINE = '#DCE5E8'
FONT = Path('/System/Library/Fonts/Supplemental')


def font(size, bold=False):
    return ImageFont.truetype(str(FONT / ('Arial Bold.ttf' if bold else 'Arial.ttf')), size)


def ease(x):
    x = np.clip(x, 0, 1)
    return float(x * x * (3 - 2 * x))


def text(d, xy, s, size=24, fill=INK, bold=False):
    d.text(xy, str(s), font=font(size, bold), fill=fill)


def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    for sp in ['left', 'bottom']:
        ax.spines[sp].set_color('#A8BBC3')
        ax.spines[sp].set_linewidth(0.7)
    ax.tick_params(labelsize=11, colors=MUTED, length=3)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.grid(axis='y', color='#E9EEF0', lw=0.6)
    ax.set_axisbelow(True)


def plot_image(kind, data, idx, export=False):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'axes.labelsize': 12, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(5.8, 5.6), dpi=110, facecolor='white')
    a = fig.add_axes([0.16, 0.57, 0.78, 0.36])
    b = fig.add_axes([0.16, 0.10, 0.78, 0.31])
    style_axes(a)
    style_axes(b)
    if kind == 'docking':
        ranks = np.arange(1, len(data) + 1)
        scores = np.array([r['score_kcal_mol'] for r in data])
        rmsd = np.array([r['crystal_rmsd_A'] for r in data])
        a.hlines(ranks, scores, 0, color='#E4EDF0', lw=8)
        a.scatter(scores, ranks, c=[GOLD if i == idx else '#9AB3BE' for i in range(len(data))], s=58, zorder=3)
        a.set_yticks([1, 3, 5, 7, 9])
        a.invert_yaxis()
        a.set_ylabel('Pose rank')
        a.set_xlabel('Vina score (kcal mol$^{-1}$)')
        a.set_xlim(-6.5, 0)
        b.scatter(rmsd, scores, c='#9AB3BE', s=35, zorder=3)
        b.scatter([rmsd[idx]], [scores[idx]], c=GOLD, s=90, edgecolors='white', zorder=5)
        b.axvline(2, ls='--', color='#B0BEC5', lw=1)
        b.set_xlabel('Crystal RMSD (Å)')
        b.set_ylabel('Score (kcal mol$^{-1}$)')
    elif kind == 'qmmm':
        x = 2 * data['distance'] - float(data['distance'][0] + data['distance'][-1] + 0.02)
        y = data['relative']
        # Actual coordinate uses donor-acceptor distance from the source summary.
        summary = json.loads((OUT / 'qmmm/summary.json').read_text())
        x = 2 * data['distance'] - summary['donor_acceptor_A']
        a.plot(x, y, color=TEAL, lw=1.5, marker='o', ms=3, mfc='white', mew=0.8)
        a.scatter(x[idx], y[idx], s=78, c=GOLD, edgecolor='white', zorder=5)
        a.set_xlabel('$d_{O-H}-d_{H-N}$ (Å)')
        a.set_ylabel('Relative energy (kcal mol$^{-1}$)')
        b.plot(np.arange(len(x)) + 1, data['distance'], color=TEAL, lw=1.7, label='O–H')
        b.plot(
            np.arange(len(x)) + 1,
            summary['donor_acceptor_A'] - data['distance'],
            color=GOLD,
            lw=1.7,
            ls='--',
            label='H···N',
        )
        b.axvline(idx + 1, color='#BAC9CF', lw=1)
        b.set_xlabel('Calculated image')
        b.set_ylabel('Distance (Å)')
        b.legend(frameon=False, fontsize=11, loc='upper right')
    else:
        rows = data['analysis']
        t = rows[:, 1]
        r = rows[:, 2]
        dist = rows[:, 3]
        a.plot(t, r, color=TEAL, lw=1.5)
        a.scatter(t[idx], r[idx], s=70, c=GOLD, edgecolor='white', zorder=5)
        a.set_ylabel('Cα RMSD (Å)')
        a.set_xlabel('Production time (ps)')
        b.plot(t, dist, color=BLUE, lw=1.4)
        b.scatter(t[idx], dist[idx], s=70, c=GOLD, edgecolor='white', zorder=5)
        b.set_ylabel('Ser Oγ ··· His Nε2 (Å)')
        b.set_xlabel('Production time (ps)')
    if export:
        p = OUT / 'plots'
        p.mkdir(exist_ok=True)
        fig.savefig(p / f'{kind}.svg')
        fig.savefig(p / f'{kind}.pdf')
        fig.savefig(p / f'{kind}.png', dpi=300)
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
    plt.close(fig)
    return im


def setup(c, kind):
    c.reinitialize()
    c.set('retain_order', 1)
    c.set('max_threads', 2)
    c.bg_color('white')
    for key, value in {
        'orthoscopic': 1,
        'ray_opaque_background': 1,
        'ray_shadows': 0,
        'antialias': 2,
        'depth_cue': 0,
        'specular': 0.35,
        'shininess': 35,
        'ambient': 0.45,
        'direct': 0.65,
        'reflect': 0.12,
        'cartoon_fancy_helices': 1,
        'cartoon_smooth_loops': 1,
        'cartoon_sampling': 14,
        'cartoon_flat_sheets': 1,
        'cartoon_highlight_color': -1,
        'surface_quality': 1,
        'sphere_quality': 2,
        'stick_quality': 16,
        'transparency_mode': 2,
        'ray_transparency_oblique': 0.2,
        'ray_trace_mode': 0,
    }.items():
        c.set(key, value)
    for name, rgb in {
        'ink': [0.15, 0.32, 0.39],
        'teal': [0.20, 0.51, 0.55],
        'slate': [0.55, 0.65, 0.70],
        'gold': [0.84, 0.60, 0.24],
        'nitrogen': [0.24, 0.45, 0.73],
        'oxygen': [0.78, 0.28, 0.25],
        'hydrogen': [0.89, 0.91, 0.92],
        'purple': [0.54, 0.44, 0.66],
    }.items():
        c.set_color(name, rgb)
    c.load(str(OUT / 'data/trypsin_amber.pdb'), 'enzyme')
    c.hide('everything')
    c.show('cartoon', 'enzyme')
    c.color('teal', 'enzyme')
    c.color('slate', 'enzyme and resi 16-122')
    c.set('cartoon_color', -1, 'enzyme')
    # Separate surface object permits cutaway treatment without clipping ribbons.
    c.create('envelope', 'enzyme and not hydro')
    c.hide('everything', 'envelope')
    c.show('surface', 'envelope')
    c.color('slate', 'envelope')
    c.set('transparency', 0.86, 'envelope')
    if kind == 'docking':
        c.load(str(OUT / 'docking/pose_01.pdb'), 'ligand')
        c.remove('ligand and hydro')
        c.hide('everything', 'ligand')
        c.show('sticks', 'ligand')
        c.show('spheres', 'ligand')
        c.color('gold', 'ligand')
        c.color('nitrogen', 'ligand and elem N')
        c.set('stick_radius', 0.16, 'ligand')
        c.set('sphere_scale', 0.25, 'ligand')
        c.set('stick_ball', 0)
        c.select('focus', 'ligand')
        c.select('contacts', 'enzyme and resi 189+190+191+192+214+215+216+226 and not hydro and not name N+C+O')
        c.show('sticks', 'contacts')
        c.set('stick_radius', 0.10, 'contacts')
        c.color('slate', 'contacts and elem C')
        c.color('oxygen', 'contacts and elem O')
        c.color('nitrogen', 'contacts and elem N')
        c.distance('contact', 'ligand and elem N', 'enzyme and resi 189 and name OD1+OD2', 3.5)
        c.hide('labels', 'contact')
        c.set('dash_color', 'slate', 'contact')
        c.set('dash_radius', 0.04)
    elif kind == 'qmmm':
        c.load(str(OUT / 'qmmm/image_000.pdb'), 'quantum')
        c.hide('everything', 'quantum')
        c.select('focus', 'quantum')
        c.show('sticks', 'quantum')
        c.show('spheres', 'quantum')
        c.color('gold', 'quantum')
        c.color('purple', 'quantum and resi 57 and elem C')
        c.color('slate', 'quantum and resi 102 and elem C')
        c.color('oxygen', 'quantum and elem O')
        c.color('nitrogen', 'quantum and elem N')
        c.color('hydrogen', 'quantum and elem H')
        c.color('gold', 'quantum and resi 195 and name HG')
        c.set('sphere_scale', 0.23, 'quantum')
        c.set('stick_radius', 0.13, 'quantum')
        c.hide('spheres', 'quantum and name HL')
        c.hide('sticks', 'quantum and name HL')
        # The transferring proton has its own sphere; bond display follows each computed image.
        c.select('moving_h', 'quantum and resi 195 and name HG')
        c.set('sphere_scale', 0.42, 'moving_h')
        c.set('cartoon_transparency', 0.22, 'enzyme')
    else:
        c.select('focus', 'enzyme and resi 57+102+195 and not hydro and not name N+CA+C+O')
        c.show('sticks', 'focus')
        c.show('spheres', 'focus')
        c.color('gold', 'focus and elem C')
        c.color('oxygen', 'focus and elem O')
        c.color('nitrogen', 'focus and elem N')
        c.set('stick_radius', 0.13, 'focus')
        c.set('sphere_scale', 0.23, 'focus')
    c.create('ribbon', 'enzyme')
    c.hide('everything', 'ribbon')
    c.show('cartoon', 'ribbon')
    c.hide('cartoon', 'enzyme')
    # Orient by active-site geometry and hold that basis across the camera push.
    c.orient('focus')
    c.turn('y', -18)
    c.turn('x', 12)
    c.zoom('enzyme', 2)
    wide = np.array(c.get_view())
    wide[15] = max(0.1, -wide[11] - 70)
    wide[16] = -wide[11] + 70
    c.zoom('focus', 3)
    close = np.array(c.get_view())
    close[15] = max(0.1, -close[11] - 70)
    close[16] = -close[11] + 70
    return wide, close


def frame_state(kind, f, n, data):
    t = f / (n - 1)
    if kind == 'docking':
        # Inspect the ranked poses individually, then return to the best pose.
        idx = 0 if t < 0.28 or t > 0.68 else min(2, int((t - 0.28) / 0.40 * 3))
    else:
        idx = min(
            len(data['relative'] if kind == 'qmmm' else data['analysis']) - 1,
            int(np.clip((t - 0.20) / 0.65, 0, 1) * (len(data['relative'] if kind == 'qmmm' else data['analysis']) - 1)),
        )
    return t, idx


def molecular_image(c, work, push=0):
    """Two registered ray passes keep the complete ligand visible through skin.

    The surface is a transparent context layer; neither layer changes geometry.
    This avoids many nested transparent surfaces obscuring the binding pocket.
    """
    enabled = c.get_names('objects', enabled_only=1)
    c.set('ray_opaque_background', 0)
    c.disable('envelope')
    c.disable('ribbon')
    c.set('cartoon_transparency', 0, 'enzyme')
    c.png(str(work / 'molecule_pass.png'), MW, MH, ray=1, quiet=1)
    main = Image.open(work / 'molecule_pass.png').convert('RGBA')
    c.disable('all')
    c.enable('ribbon')
    c.set('cartoon_transparency', 0, 'ribbon')
    c.png(str(work / 'ribbon_pass.png'), MW, MH, ray=1, quiet=1)
    ribbon = Image.open(work / 'ribbon_pass.png').convert('RGBA')
    ribbon.putalpha(ribbon.getchannel('A').point(lambda a: int(a * (1 - 0.72 * push))))
    c.disable('all')
    c.enable('envelope')
    c.set('transparency', 0, 'envelope')
    c.png(str(work / 'surface_pass.png'), MW, MH, ray=1, quiet=1)
    skin = Image.open(work / 'surface_pass.png').convert('RGBA')
    skin.putalpha(skin.getchannel('A').point(lambda a: int(a * (0.22 - 0.07 * push))))
    result = Image.new('RGBA', (MW, MH), 'white')
    result.alpha_composite(skin)
    result.alpha_composite(ribbon)
    result.alpha_composite(main)
    for name in enabled:
        c.enable(name)
    c.set('transparency', 0.86, 'envelope')
    c.set('ray_opaque_background', 1)
    return result.convert('RGB')


def composite(kind, molecule, inset, plot, idx, data, t):
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    text(d, (68, 38), 'L E O N A R D O', 26, bold=True)
    text(d, (1510, 43), 'MOLECULAR SCIENCE', 20, fill=MUTED)
    d.line((68, 88, 1852, 88), fill=LINE, width=2)
    titles = {
        'docking': 'A pose. A protein. A precise fit.',
        'qmmm': 'Inside the catalytic triad.',
        'md': 'A protein in motion.',
    }
    subtitles = {
        'docking': 'Trypsin · benzamidinium  /  PDB 3PTB',
        'qmmm': 'Trypsin · Ser195 / His57 / Asp102  /  PDB 3PTB',
        'md': 'Apo trypsin · Amber14 / OBC-II  /  PDB 3PTB',
    }
    text(d, (68, 118), titles[kind], 52, bold=True)
    text(d, (71, 185), subtitles[kind], 25, fill=MUTED)
    im.paste(molecule, (32, 224))
    # A persistent structural locator retains protein context during the close-up.
    if t > 0.28:
        thumb = inset.resize((270, 179), Image.Resampling.LANCZOS)
        im.paste(thumb, (72, 250))
        text(d, (82, 431), 'Protein context', 17, fill=MUTED)
    d.line((1240, 248, 1240, 957), fill=LINE, width=2)
    if kind == 'docking':
        row = data[idx]
        kicker = '01 / MOLECULAR DOCKING'
        metric = f"{row['score_kcal_mol']:.2f}"
        unit = 'kcal/mol'
        detail = f"Pose {idx+1}   ·   Crystal RMSD {row['crystal_rmsd_A']:.2f} Å"
        note = 'smina / Vina scoring · 9 calculated poses'
        foot = 'Rigid-receptor pocket redocking. Poses are ranked solutions; they are not a dynamical trajectory.'
    elif kind == 'qmmm':
        kicker = '02 / ENZYME QM/MM'
        metric = f"{data['relative'][idx]:.1f}"
        unit = 'kcal/mol'
        detail = f"Image {idx+1:02d} / 33   ·   25 QM atoms + 3,192 MM charges"
        note = 'MOPAC PM7 / Amber14 electrostatic embedding'
        foot = 'Constrained proton-transfer scan; protein fixed. This calculation has not validated an enzyme transition state.'
    else:
        row = data['analysis'][idx]
        kicker = '03 / MOLECULAR DYNAMICS'
        metric = f"{row[1]:.1f}"
        unit = 'ps'
        detail = f"Saved frame {idx+1:03d} / 120   ·   300 K"
        note = 'OpenMM · 2 fs integration · 12 ps production'
        foot = 'Short apo-protein relaxation with implicit solvent. Each displayed structure is an actual saved simulation frame.'
    text(d, (1290, 249), kicker, 20, fill=TEAL, bold=True)
    text(d, (1284, 290), metric, 66, bold=True)
    bbox = d.textbbox((0, 0), metric, font=font(66, True))
    text(d, (1300 + bbox[2], 329), unit, 25, fill=MUTED)
    text(d, (1290, 372), detail, 20, fill=MUTED)
    im.paste(plot.resize((605, 584), Image.Resampling.LANCZOS), (1256, 409))
    d.line((68, 1010, 1852, 1010), fill=LINE, width=2)
    text(d, (70, 1024), note, 20, bold=True)
    text(d, (1810, 1025), {'docking': '01', 'qmmm': '02', 'md': '03'}[kind], 20, fill=MUTED)
    # Small but legible method qualification lives above the footer, on white.
    d.rectangle((55, 967, 1234, 1008), fill='white')
    text(d, (70, 978), foot, 16, fill=MUTED)
    d.rectangle((68, 1008, 68 + int(1784 * t), 1011), fill=TEAL)
    return im


def render(kind, frames, fps, preview=False, start_frame=0, stop_frame=None, no_encode=False, force=False):
    work = OUT / 'renders' / kind
    work.mkdir(parents=True, exist_ok=True)
    scratch = work / f'worker_{os.getpid()}'
    scratch.mkdir(exist_ok=True)
    sessions = OUT / 'sessions'
    sessions.mkdir(exist_ok=True)
    data = (
        json.loads((OUT / 'docking/summary.json').read_text())['poses']
        if kind == 'docking'
        else np.load(OUT / ('qmmm/path.npz' if kind == 'qmmm' else 'md/aligned.npz'))
    )
    plot_image(kind, data, 0, True)
    targets = (
        [0, int(frames * 0.52), frames - 1] if preview else list(range(start_frame, min(frames, stop_frame or frames)))
    )
    with pymol2.PyMOL() as pm:
        c = pm.cmd
        wide, close = setup(c, kind)
        c.set_view(wide.tolist())
        inset = molecular_image(c, scratch)
        inset.save(work / 'context.png')
        c.save(str(sessions / f'{kind}.pse'))
        start = time.perf_counter()
        last_idx = -1
        plot = None
        for f in targets:
            target = work / f'frame_{f:05d}.png'
            if target.exists() and not preview and not force:
                continue
            t, idx = frame_state(kind, f, frames, data)
            if kind == 'qmmm':
                xyz = data['coords'][idx]
                c.load_coords(xyz, 'quantum')
                c.unbond('moving_h', 'quantum and resi 195 and name OG')
                c.unbond('moving_h', 'quantum and resi 57 and name NE2')
                dOH = float(data['distance'][idx])
                D = json.loads((OUT / 'qmmm/summary.json').read_text())['donor_acceptor_A']
                if dOH < 1.25:
                    c.bond('moving_h', 'quantum and resi 195 and name OG')
                if D - dOH < 1.25:
                    c.bond('moving_h', 'quantum and resi 57 and name NE2')
            elif kind == 'md' and idx != last_idx:
                c.load_coords(data['xyz'][idx], 'enzyme')
                c.load_coords(data['xyz'][idx], 'ribbon')
                c.load_coords(data['xyz'][idx][data['heavy']], 'envelope')
            elif idx != last_idx:
                # All pose PDB files share the engine's atom order.
                from rdkit import Chem

                m = Chem.MolFromPDBFile(str(OUT / f'docking/pose_{idx+1:02d}.pdb'), removeHs=True)
                c.load_coords(m.GetConformer().GetPositions(), 'ligand')
                c.delete('contact')
                c.distance('contact', 'ligand and elem N', 'enzyme and resi 189 and name OD1+OD2', 3.5)
                c.hide('labels', 'contact')
                c.set('dash_color', 'slate', 'contact')
                c.set('dash_radius', 0.04)
            push = ease((t - 0.12) / 0.28)
            # Gently pull back at the end, while preserving the final scientific state.
            push *= 1 - 0.18 * ease((t - 0.86) / 0.14)
            view = wide * (1 - push) + close * push
            c.set_view(view.tolist())
            c.turn('y', -8 + 16 * t)
            c.turn('x', 2 * math.sin(t * math.pi))
            molecule = molecular_image(c, scratch, push)
            if idx != last_idx or plot is None:
                plot = plot_image(kind, data, idx)
            image = composite(kind, molecule, inset, plot, idx, data, t)
            temporary = scratch / target.name
            image.save(temporary)
            temporary.replace(target)
            last_idx = idx
            if f % 24 == 0 or preview:
                print(kind, f, '/', frames, 'elapsed', round(time.perf_counter() - start, 1), flush=True)
        if preview or no_encode:
            return
    if not all((work / f'frame_{f:05d}.png').exists() for f in range(frames)):
        raise RuntimeError('Rendering incomplete; refusing to encode a partial film')
    videos = OUT / 'videos'
    videos.mkdir(exist_ok=True)
    name = {
        'docking': '01_leonardo_docking_real',
        'qmmm': '02_leonardo_enzyme_qmmm_real',
        'md': '03_leonardo_dynamics_real',
    }[kind]
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate',
        str(fps),
        '-i',
        str(work / 'frame_%05d.png'),
        '-c:v',
        'libx264',
        '-preset',
        'slow',
        '-crf',
        '17',
        '-pix_fmt',
        'yuv420p',
        '-movflags',
        '+faststart',
        '-an',
        str(videos / (name + '.mp4')),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    (work / 'ffmpeg.log').write_text(cp.stderr)
    if cp.returncode:
        raise RuntimeError(cp.stderr[-1000:])
    print('Finished', name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['docking', 'qmmm', 'md'])
    p.add_argument('--frames', type=int, default=480)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--stop-frame', type=int)
    p.add_argument('--no-encode', action='store_true')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    render(a.kind, a.frames, a.fps, a.preview, a.start_frame, a.stop_frame, a.no_encode, a.force)
