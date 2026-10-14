#!/usr/bin/env python3
"""White-stage, multi-shot molecular films from verified calculation files."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pymol2
import render_real_cinematic_videos as base
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OLD = Path(os.environ.get('LEONARDO_FILM_REAL', ROOT / 'artifacts/cinematic_real'))
OUT = Path(os.environ.get('LEONARDO_FILM_V2', ROOT / 'artifacts/cinematic_v2'))
W, H = 1920, 1080
MW, MH = 1600, 900
INK = '#142E3B'
MUTED = '#69818B'
TEAL = '#277F87'
GOLD = '#BB883A'
BLUE = '#467AAF'
RED = '#C45549'
LINE = '#DCE6E8'
FONT = Path('/System/Library/Fonts/Supplemental')


def font(n, bold=False):
    return ImageFont.truetype(str(FONT / ('Arial Bold.ttf' if bold else 'Arial.ttf')), n)


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


def txt(d, pos, s, n=24, fill=INK, bold=False):
    d.text(pos, str(s), font=font(n, bold), fill=fill)


def faded(im, layer, alpha):
    layer.putalpha(layer.getchannel('A').point(lambda x: int(x * np.clip(alpha, 0, 1))))
    im.alpha_composite(layer)


def axes_style(a):
    a.spines[['top', 'right']].set_visible(False)
    for s in ['left', 'bottom']:
        a.spines[s].set_color('#A8BBC0')
        a.spines[s].set_linewidth(0.65)
    a.tick_params(colors=MUTED, labelsize=10.5, length=3, width=0.6, pad=5)
    a.grid(axis='y', color='#E7EDEF', lw=0.6)
    a.set_axisbelow(True)
    a.xaxis.label.set_color(INK)
    a.yaxis.label.set_color(INK)


def plot_image(kind, data, idx, export=False):
    if export and kind == 'qmmm':
        idx = int(data['ts_index'])
    elif export and kind == 'md':
        idx = len(data['analysis']) - 1
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'svg.fonttype': 'none', 'axes.labelsize': 12})
    fig = plt.figure(figsize=(7.5, 5.35), dpi=140, facecolor='white')
    a = fig.add_axes([0.115, 0.45, 0.84, 0.49])
    b = fig.add_axes([0.115, 0.095, 0.84, 0.23])
    axes_style(a)
    axes_style(b)
    if kind == 'qmmm':
        x = data['s']
        y = data['relative']
        peak = int(data['ts_index'])
        a.plot(x, y, color='#CBD9DE', lw=1.4, zorder=1)
        a.plot(x if export else x[: idx + 1], y if export else y[: idx + 1], color=TEAL, lw=2.1, zorder=3)
        a.scatter(x, y, s=8, facecolors='white', edgecolors='#9CB3BC', linewidths=0.5, zorder=2)
        a.scatter(x[idx], y[idx], s=86, c=GOLD, edgecolors='white', linewidths=1.5, zorder=6)
        a.axvline(x[peak], color='#A8BDC5', lw=0.8, ls=(0, (3, 4)), zorder=0)
        a.set_ylabel('Relative energy (kcal mol$^{-1}$)')
        a.set_xticks([x[0], x[peak], x[-1]], ['Reactant', 'TS', 'Alkyl–enzyme + Cl$^-$'])
        a.get_xticklabels()[-1].set_horizontalalignment('right')
        a.set_xlim(-0.035, 1.035)
        a.set_ylim(min(y) - 2, max(y) + max(2, 0.16 * np.ptp(y)))
        bx = x[peak] + 0.115
        a.annotate('', (bx, y[peak]), (bx, 0), arrowprops=dict(arrowstyle='<->', lw=0.8, color=INK))
        a.text(
            bx + 0.022,
            y[peak] * 0.50,
            f'$\Delta E^{{\ddagger}}$ = {y[peak]:.2f}',
            fontsize=10.5,
            color=INK,
            va='center',
        )
        a.text(
            x[peak],
            y[peak] + 0.07 * np.ptp(y),
            f"{abs(float(data['imaginary_frequency'])):.1f}i cm$^{{-1}}$",
            ha='center',
            fontsize=10.5,
            color=TEAL,
        )
        b.plot(x, data['d_oc'], color=TEAL, lw=1.7, label='O–C forming')
        b.plot(x, data['d_ccl'], color=GOLD, lw=1.7, label='C–Cl breaking')
        b.axvline(x[idx], color='#A8BDC5', lw=0.7)
        b.set_xlim(-0.035, 1.035)
        b.set_ylabel('Distance (Å)')
        b.set_xlabel('Normalized mass-weighted path')
        b.legend(frameon=False, fontsize=10, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.21))
    elif kind == 'docking':
        scores = np.array([r['score_kcal_mol'] for r in data])
        rmsd = np.array([r['crystal_rmsd_A'] for r in data])
        ranks = np.arange(1, len(data) + 1)
        a.bar(ranks, scores, color=[GOLD if i == idx else '#B4C8CE' for i in range(len(data))], width=0.58, zorder=3)
        a.set_xlabel('Ranked docking pose')
        a.set_ylabel('Vina score (kcal mol$^{-1}$)')
        a.set_xticks(ranks)
        a.set_ylim(-6.8, 0)
        a.text(1, scores[0] - 0.35, f'{scores[0]:.2f}', ha='center', fontsize=11, color=INK)
        b.scatter(rmsd, scores, c='#A6BEC7', s=35, zorder=3)
        b.scatter(rmsd[idx], scores[idx], s=86, c=GOLD, edgecolors='white', linewidths=1.2, zorder=5)
        b.axvline(2, color='#ADBFC6', lw=0.8, ls=(0, (3, 4)))
        b.set_xlabel('Symmetry-aware crystal RMSD (Å)')
        b.set_ylabel('Score')
    else:
        rows = data['analysis']
        t = rows[:, 1]
        a.plot(t, rows[:, 2], color='#C0D2D8', lw=1)
        a.plot(t[: idx + 1], rows[: idx + 1, 2], color=TEAL, lw=1.8)
        a.scatter(t[idx], rows[idx, 2], s=75, c=GOLD, edgecolor='white', zorder=5)
        a.set_xlabel('Production time (ps)')
        a.set_ylabel('Cα RMSD (Å)')
        b.plot(t, rows[:, 3], color=BLUE, lw=1.5)
        b.scatter(t[idx], rows[idx, 3], s=55, c=GOLD, edgecolor='white', zorder=5)
        b.set_xlabel('Production time (ps)')
        b.set_ylabel('Oγ···Nε2 (Å)')
        b.text(0.99, 0.93, 'Ser195 · His57', ha='right', va='top', transform=b.transAxes, fontsize=10, color=MUTED)
    if export:
        fig.text(
            0.115,
            0.98,
            {
                'qmmm': 'DhlA · PM7/Amber14 QM/MM · fixed protein',
                'docking': 'Trypsin · benzamidinium · smina/Vina',
                'md': 'Apo trypsin · Amber14/OBC-II · 300 K',
            }[kind],
            fontsize=12,
            color=INK,
            weight='bold',
            va='top',
        )
        (OUT / 'plots').mkdir(exist_ok=True)
        for ext in ['svg', 'pdf']:
            fig.savefig(OUT / 'plots' / f'{kind}.{ext}')
        fig.savefig(OUT / 'plots' / f'{kind}.png', dpi=300)
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
    plt.close(fig)
    return im


def setup(c, kind, data):
    wide, close = base.setup(c, kind if kind != 'qmmm' else 'docking')
    if kind == 'qmmm':
        c.delete('all')
        c.load(str(OUT / 'data/dhla_amber.pdb'), 'enzyme')
        c.hide('everything')
        c.color('slate', 'enzyme')
        c.color('teal', 'enzyme and resi 156-229')
        c.create('ribbon', 'enzyme')
        c.hide('everything', 'ribbon')
        c.show('cartoon', 'ribbon')
        c.create('envelope', 'enzyme and not hydro')
        c.hide('everything', 'envelope')
        c.show('surface', 'envelope')
        c.color('slate', 'envelope')
        initial = OUT / 'qmmm/path_000.pdb'
        c.load(str(initial if initial.exists() else OUT / 'qmmm/ts.pdb'), 'quantum')
        c.hide('everything', 'quantum')
        c.show('sticks', 'quantum and not name HL')
        c.show('spheres', 'quantum and not name HL')
        c.color('gold', 'quantum')
        c.color('teal', 'quantum and resi 124 and elem C')
        c.color('oxygen', 'quantum and elem O')
        c.color('nitrogen', 'quantum and elem N')
        c.color('hydrogen', 'quantum and elem H')
        c.set_color('chlorine', [0.32, 0.63, 0.40])
        c.color('chlorine', 'quantum and elem Cl')
        c.set('stick_radius', 0.14, 'quantum')
        c.set('sphere_scale', 0.25, 'quantum')
        c.set('sphere_scale', 0.22, 'quantum and hydro')
        c.select('contacts', 'enzyme and resi 125+175 and not hydro and not name N+CA+C+O')
        c.show('sticks', 'contacts')
        c.set('stick_radius', 0.095, 'contacts')
        c.color('slate', 'contacts and elem C')
        c.color('nitrogen', 'contacts and elem N')
        c.create('pocket', 'contacts')
        c.hide('everything', 'pocket')
        c.show('sticks', 'pocket')
        c.hide('sticks', 'enzyme')
        c.select('focus', 'quantum and not hydro')
        c.orient('focus')
        c.turn('y', -12)
        c.turn('x', 20)
        c.turn('z', -18)
        c.zoom('enzyme', 2)
        wide = np.array(c.get_view())
        c.zoom('focus', 2.5)
        close = np.array(c.get_view())
    for v in [wide, close]:
        v[15] = max(0.1, -v[11] - 90)
        v[16] = -v[11] + 90
    c.set('ray_shadows', 1)
    c.set('ambient', 0.40)
    c.set('direct', 0.67)
    c.set('specular', 0.42)
    c.set('shininess', 45)
    c.set('antialias', 1)
    c.set('max_threads', 2)
    c.set('ray_opaque_background', 0)
    c.set('cartoon_transparency', 0, 'ribbon')
    c.set('transparency', 0, 'envelope')
    return wide, close


def update_state(c, kind, data, idx):
    if kind == 'qmmm':
        xyz = data['coords'][idx]
        c.load_coords(xyz, 'quantum')
        c.unbond('quantum and name OD2', 'quantum and name C1')
        c.unbond('quantum and name C1', 'quantum and name CL1')
        c.delete('forming')
        c.delete('breaking')
        if data['d_oc'][idx] < 1.8:
            c.bond('quantum and name OD2', 'quantum and name C1')
        else:
            c.distance('forming', 'quantum and name OD2', 'quantum and name C1')
        if data['d_ccl'][idx] < 2.1:
            c.bond('quantum and name C1', 'quantum and name CL1')
        else:
            c.distance('breaking', 'quantum and name C1', 'quantum and name CL1')
        for name, color in [('forming', 'teal'), ('breaking', 'gold')]:
            if name in c.get_names():
                c.hide('labels', name)
                c.set('dash_color', color, name)
                c.set('dash_radius', 0.028, name)
                c.set('dash_gap', 0.15, name)
    elif kind == 'md':
        c.load_coords(data['xyz'][idx], 'enzyme')
        c.load_coords(data['xyz'][idx], 'ribbon')
        c.load_coords(data['xyz'][idx][data['heavy']], 'envelope')
    else:
        from rdkit import Chem

        mol = Chem.MolFromPDBFile(str(OLD / f'docking/pose_{idx+1:02d}.pdb'), removeHs=True)
        c.load_coords(mol.GetConformer().GetPositions(), 'ligand')
        c.delete('contact')
        c.distance('contact', 'ligand and elem N', 'enzyme and resi 189 and name OD1+OD2', 3.5)
        c.hide('labels', 'contact')
        c.set('dash_color', 'slate', 'contact')
        c.set('dash_radius', 0.035)


def ray(c, work, push, context_cache=None):
    enabled = c.get_names('objects', enabled_only=1)
    c.disable('envelope')
    c.disable('ribbon')
    if 'pocket' in c.get_names():
        c.disable('pocket')
    c.png(str(work / 'active.png'), MW, MH, ray=1, quiet=1)
    active = Image.open(work / 'active.png').convert('RGBA')
    if context_cache is None:
        c.disable('all')
        c.enable('ribbon')
        if 'pocket' in c.get_names():
            c.enable('pocket')
        c.png(str(work / 'ribbon.png'), MW, MH, ray=1, quiet=1)
        ribbon = Image.open(work / 'ribbon.png').convert('RGBA')
        c.disable('all')
        c.enable('envelope')
        c.png(str(work / 'surface.png'), MW, MH, ray=1, quiet=1)
        surface = Image.open(work / 'surface.png').convert('RGBA')
    else:
        ribbon, surface = context_cache
    ribbon = ribbon.filter(ImageFilter.GaussianBlur(1.4 * push))
    surface = surface.filter(ImageFilter.GaussianBlur(2.5 * push))
    ribbon.putalpha(ribbon.getchannel('A').point(lambda a: int(a * (0.94 - 0.71 * push))))
    surface.putalpha(surface.getchannel('A').point(lambda a: int(a * (0.27 - 0.13 * push))))
    im = Image.new('RGBA', (MW, MH), 'white')
    im.alpha_composite(surface)
    im.alpha_composite(ribbon)
    im.alpha_composite(active)
    c.disable('all')
    for name in enabled:
        c.enable(name)
    return im.convert('RGB')


def state_at(kind, seconds, data):
    p = float(np.clip((seconds - 8.0) / 10.0, 0, 1))
    if kind == 'docking':
        return 0 if seconds < 8 or seconds > 18 else min(2, int(p * 3))
    if kind == 'qmmm':
        n = len(data['relative'])
        ts = int(data['ts_index'])
        # Hold the actual stationary geometry briefly so its bond exchange is readable.
        if p < 0.45:
            return int(p / 0.45 * ts)
        if p < 0.57:
            return ts
        return min(n - 1, ts + int((p - 0.57) / 0.43 * (n - 1 - ts)))
    return min(len(data['analysis']) - 1, int(p * (len(data['analysis']) - 1)))


def composite(kind, molecule, locator, plot, idx, data, seconds, total, labels=None):
    im = Image.new('RGBA', (W, H), 'white')
    intro = 1 - ease((seconds - 2.6) / 1.0)
    analysis = ease((seconds - 6.7) / 1.1)
    scale = 1 - 0.31 * analysis
    px = int(160 + 440 * intro - 160 * analysis)
    py = int(125 + 115 * analysis)
    mol = molecule.resize((int(MW * scale), int(MH * scale)), Image.Resampling.LANCZOS)
    im.paste(mol, (px, py))
    d = ImageDraw.Draw(im)
    txt(d, (78, 42), 'L E O N A R D O', 25, bold=True)
    txt(d, (1510, 46), 'MOLECULAR FILMS  /  ' + {'docking': '01', 'qmmm': '02', 'md': '03'}[kind], 18, fill=MUTED)
    d.line((78, 93, 1842, 93), fill=LINE, width=1)
    titles = {
        'docking': ['The molecular', 'fit.'],
        'qmmm': ['Across the', 'reaction barrier.'],
        'md': ['Motion,', 'atom by atom.'],
    }
    subtitles = {
        'docking': 'TRYPSIN · BENZAMIDINIUM',
        'qmmm': 'HALOALKANE DEHALOGENASE · DhlA',
        'md': 'TRYPSIN · MOLECULAR DYNAMICS',
    }
    chapter = {
        'docking': 'A binding pocket, resolved.',
        'qmmm': 'Where two bonds exchange.',
        'md': 'Inside a moving catalytic site.',
    }
    if intro > 0:
        layer = Image.new('RGBA', (W, H))
        ld = ImageDraw.Draw(layer)
        txt(
            ld,
            (85, 252),
            {'docking': '01 / MOLECULAR DOCKING', 'qmmm': '02 / QST · ENZYME QM/MM', 'md': '03 / MOLECULAR DYNAMICS'}[
                kind
            ],
            20,
            TEAL,
            True,
        )
        for j, line in enumerate(titles[kind]):
            txt(ld, (78, 307 + 102 * j), line, 86, bold=True)
        txt(ld, (86, 563), subtitles[kind], 21, MUTED)
        txt(
            ld,
            (86, 604),
            {
                'docking': 'PDB 3PTB  ·  smina / Vina',
                'qmmm': 'PDB 2DHC  ·  MOPAC PM7 / Amber14',
                'md': 'PDB 3PTB  ·  OpenMM / Amber14',
            }[kind],
            20,
            MUTED,
        )
        ld.line((86, 680, 178, 680), fill=GOLD, width=4)
        txt(ld, (86, 713), 'Real structures. Calculated results.', 21)
        faded(im, layer, intro)
    if intro < 1:
        layer = Image.new('RGBA', (W, H))
        ld = ImageDraw.Draw(layer)
        txt(ld, (80, 128), subtitles[kind], 18, TEAL, True)
        txt(ld, (76, 166), chapter[kind], 48, bold=True)
        faded(im, layer, 1 - intro)
    if analysis > 0:
        layer = Image.new('RGBA', (W, H))
        ld = ImageDraw.Draw(layer)
        # A white falloff opens breathing room below the molecular close-up.
        for y in range(760, 861):
            ld.line((0, y, 1058, y), fill=(255, 255, 255, int(255 * ease((y - 760) / 82))))
        if kind == 'qmmm' and labels:
            # Atom-registered leaders use the actual PyMOL camera projection.
            for name, caption, position in [
                ('OD2', 'Asp124 · nucleophile', (86, 453)),
                ('CL1', 'Cl1 · leaving group', (774, 453)),
            ]:
                anchor = np.array(labels[name]) * scale + np.array([px, py])
                tx, ty = position
                edge = (tx + 135, ty + 28) if name == 'OD2' else (tx, ty + 28)
                ld.line((edge[0], edge[1], float(anchor[0]), float(anchor[1])), fill='#9BAFB7', width=1)
                txt(ld, (tx, ty), caption, 17, MUTED)
            caption = (
                'Asp124 + 1,2-dichloroethane'
                if idx < int(data['ts_index'])
                else 'SN2 transition state'
                if idx == int(data['ts_index'])
                else 'Alkyl–enzyme intermediate + chloride'
            )
            txt(ld, (86, 777), caption, 19, MUTED)
        # The main evidence panel enters after the uninterrupted molecular close-up.
        ld.rectangle((1060, 243, 1870, 997), fill='white')
        txt(
            ld,
            (1090, 254),
            {
                'docking': 'The search, quantified.',
                'qmmm': 'The calculated reaction path.',
                'md': 'Every frame, measured.',
            }[kind],
            29,
            bold=True,
        )
        note = {
            'docking': '9 poses · crystal-reference comparison',
            'qmmm': 'Reactant → transition state → alkyl–enzyme',
            'md': (f"{float(data['production_ps']):g} ps production · {len(data['analysis'])} saved structures"
                   if 'production_ps' in data else '12 ps production · 120 saved structures'),
        }[kind]
        txt(ld, (1092, 300), note, 19, MUTED)
        layer.paste(plot.resize((805, 575), Image.Resampling.LANCZOS), (1050, 338))
        if kind == 'qmmm':
            state = (
                'TRANSITION STATE'
                if idx == int(data['ts_index'])
                else 'REACTANT BASIN'
                if idx < int(data['ts_index'])
                else 'PRODUCT BASIN'
            )
            metric = f"{data['relative'][idx]:+.2f}"
            unit = 'kcal/mol'
            detail = f"O–C  {data['d_oc'][idx]:.2f} Å     C–Cl  {data['d_ccl'][idx]:.2f} Å"
            label = f"{len(data['relative'])} calculated path geometries"
        elif kind == 'docking':
            state = f'CALCULATED POSE  {idx+1:02d} / 09'
            metric = f"{data[idx]['score_kcal_mol']:.2f}"
            unit = 'kcal/mol'
            detail = f"Crystal RMSD  {data[idx]['crystal_rmsd_A']:.2f} Å"
            label = 'Benzamidinium · Asp189 recognition pocket'
        else:
            state = f"SAVED STRUCTURE  {idx+1:03d} / {len(data['analysis'])}"
            metric = f"{data['analysis'][idx,1]:.1f}"
            unit = 'ps'
            detail = f"Cα RMSD  {data['analysis'][idx,2]:.2f} Å     {float(data.get('temperature_K', 300)):g} K"
            label = 'Ser195 · His57 · Asp102'
        txt(ld, (84, 826), state, 18, TEAL, True)
        txt(ld, (79, 858), metric, 68, bold=True)
        width = ld.textbbox((0, 0), metric, font=font(68, True))[2]
        txt(ld, (94 + width, 897), unit, 24, MUTED)
        txt(ld, (86, 945), detail, 22)
        txt(ld, (1092, 934), label, 19, MUTED)
        # Small full-protein locator maintains scale while the site occupies the frame.
        thumb = locator.resize((210, 118), Image.Resampling.LANCZOS)
        layer.paste(thumb, (825, 265))
        txt(ld, (867, 386), 'Protein context', 14, MUTED)
        faded(im, layer, analysis)
    if 3.6 < seconds < 6.7:
        layer = Image.new('RGBA', (W, H))
        ld = ImageDraw.Draw(layer)
        txt(
            ld,
            (84, 932),
            {
                'docking': 'A crystallographic pocket. An independently docked ligand.',
                'qmmm': 'Asp124 Oδ2  →  substrate C1  →  leaving chloride',
                'md': 'Cartoon + molecular surface. Exact saved coordinates.',
            }[kind],
            23,
        )
        faded(im, layer, ease((seconds - 3.6) / 0.6) * (1 - ease((seconds - 6.1) / 0.6)))
    d = ImageDraw.Draw(im)
    foot = {
        'docking': 'Rigid-receptor redocking · Vina score is a scoring-function result',
        'qmmm': 'First alkylation step · fixed-protein QM/MM · potential energy, not free energy',
        'md': 'Short apo-protein relaxation · OBC-II implicit solvent · 2 fs integration',
    }[kind]
    d.rectangle((0, 1000, W, H), fill='white')
    d.line((78, 1007, 1842, 1007), fill=LINE, width=1)
    d.line((78, 1007, 78 + int(1764 * seconds / total), 1007), fill=TEAL, width=3)
    txt(d, (80, 1030), foot, 18, MUTED)
    txt(d, (1660, 1030), 'PDB ' + ('2DHC' if kind == 'qmmm' else '3PTB'), 18, MUTED)
    return im.convert('RGB')


def render(kind, frames=576, fps=24, preview=False, start_frame=0, stop_frame=None, no_encode=False, intro_only=False):
    work = OUT / 'renders' / kind
    work.mkdir(parents=True, exist_ok=True)
    scratch = work / f'worker_{os.getpid()}'
    scratch.mkdir(exist_ok=True)
    (OUT / 'sessions').mkdir(exist_ok=True)
    if intro_only:
        assert kind == 'qmmm' and not preview
        # The opening contains only the optimized reactant and camera motion.
        # It can render while the independent product validation is running;
        # no energy plot is generated or shown in these frames.
        minimum = np.load(OUT / 'qmmm/downhill_minus.npz')['xyz']
        atoms = json.loads((OUT / 'qmmm/qm_atoms.json').read_text())
        names = {a['name']: i for i, a in enumerate(atoms)}
        data = dict(
            coords=minimum[None, :, :],
            relative=np.array([0.0]),
            ts_index=0,
            d_oc=np.array([np.linalg.norm(minimum[names['OD2']] - minimum[names['C1']])]),
            d_ccl=np.array([np.linalg.norm(minimum[names['C1']] - minimum[names['CL1']])]),
        )
        (OUT / 'qmmm/path_000.pdb').write_bytes((OUT / 'qmmm/downhill_minus.pdb').read_bytes())
        stop_frame = min(stop_frame or frames, int(6.65 * fps))
        no_encode = True
    else:
        data = (
            json.loads((OLD / 'docking/summary.json').read_text())['poses']
            if kind == 'docking'
            else dict(np.load((OUT / 'qmmm/path.npz') if kind == 'qmmm' else (OLD / 'md/aligned.npz')))
        )
    if not intro_only:
        plot_image(kind, data, 0, True)
    targets = [0, 120, 300, frames - 1] if preview else list(range(start_frame, min(stop_frame or frames, frames)))
    started = time.perf_counter()
    with pymol2.PyMOL() as p:
        c = p.cmd
        wide, close = setup(c, kind, data)
        update_state(c, kind, data, 0)
        c.set_view(wide.tolist())
        locator = ray(c, scratch, 0)
        locator.save(work / 'context.png')
        c.set('transparency', 0.87, 'envelope')
        c.save(str(OUT / 'sessions' / f'{kind}.pse'))
        c.set('transparency', 0, 'envelope')
        last_idx = -1
        last_key = None
        cached_molecule = None
        plot = None
        context = None
        for f in targets:
            target = work / f'frame_{f:05d}.png'
            if target.exists() and not preview:
                continue
            seconds = f / fps
            idx = state_at(kind, seconds, data)
            push = ease((seconds - 2.6) / 4.0)
            view = wide * (1 - push) + close * push
            c.set_view(view.tolist())
            c.turn('y', -13 * (1 - push) + 8 * math.sin(min(seconds / 6.6, 1) * math.pi) * (1 - push))
            c.turn('x', -3 * (1 - push))
            if idx != last_idx:
                update_state(c, kind, data, idx)
                plot = Image.new('RGB', (805, 575), 'white') if intro_only else plot_image(kind, data, idx)
            key = (f if seconds < 6.6 else 'close', idx)
            if key != last_key:
                # Freeze the scientific camera while the path evolves. Registered
                # protein passes can be reused because the QM/MM protein is fixed.
                if seconds >= 6.6 and kind != 'md' and context is None:
                    enabled = c.get_names('objects', enabled_only=1)
                    c.disable('all')
                    c.enable('ribbon')
                    if 'pocket' in c.get_names():
                        c.enable('pocket')
                    c.png(str(scratch / 'static_ribbon.png'), MW, MH, ray=1, quiet=1)
                    c.disable('all')
                    c.enable('envelope')
                    c.png(str(scratch / 'static_surface.png'), MW, MH, ray=1, quiet=1)
                    context = (
                        Image.open(scratch / 'static_ribbon.png').convert('RGBA'),
                        Image.open(scratch / 'static_surface.png').convert('RGBA'),
                    )
                    c.disable('all')
                    for name in enabled:
                        c.enable(name)
                cached_molecule = ray(c, scratch, push, context if seconds >= 6.6 and kind != 'md' else None)
                last_key = key
            labels = None
            if kind == 'qmmm':
                actual = np.array(c.get_view())
                rot = actual[:9].reshape(3, 3).T
                factor = MH / (2 * abs(actual[11]) * math.tan(math.radians(float(c.get('field_of_view'))) / 2))
                labels = {}
                for atom in ['OD2', 'CL1']:
                    point = c.get_coords('quantum and name ' + atom)[0]
                    projected = rot @ (point - actual[12:15]) + actual[9:12]
                    labels[atom] = (MW / 2 + factor * projected[0], MH / 2 - factor * projected[1])
            im = composite(kind, cached_molecule, locator, plot, idx, data, seconds, (frames - 1) / fps, labels)
            temp = scratch / target.name
            im.save(temp)
            temp.replace(target)
            last_idx = idx
            if f % 24 == 0 or preview:
                print(kind, f, '/', frames, 'elapsed', round(time.perf_counter() - started, 1), flush=True)
    if not preview and not no_encode:
        encode(kind, frames, fps)


def encode(kind, frames=576, fps=24):
    work = OUT / 'renders' / kind
    if not all((work / f'frame_{f:05d}.png').exists() for f in range(frames)):
        raise RuntimeError('Incomplete frame sequence ' + kind)
    (OUT / 'videos').mkdir(exist_ok=True)
    name = {'docking': '01_molecular_docking', 'qmmm': '02_dhla_reaction_qmmm', 'md': '03_molecular_dynamics'}[kind]
    cp = subprocess.run(
        [
            '/opt/homebrew/bin/ffmpeg',
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
            '16',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
            '-an',
            str(OUT / 'videos' / f'{name}.mp4'),
        ],
        capture_output=True,
        text=True,
    )
    (work / 'ffmpeg.log').write_text(cp.stderr)
    if cp.returncode:
        raise RuntimeError(cp.stderr[-2000:])
    print('Encoded', name, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=['docking', 'qmmm', 'md'])
    p.add_argument('--preview', action='store_true')
    p.add_argument('--frames', type=int, default=576)
    p.add_argument('--fps', type=int, default=24)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--stop-frame', type=int)
    p.add_argument('--no-encode', action='store_true')
    p.add_argument('--encode-only', action='store_true')
    p.add_argument('--intro-only', action='store_true')
    a = p.parse_args()
    if a.encode_only:
        encode(a.kind, a.frames, a.fps)
    else:
        render(a.kind, a.frames, a.fps, a.preview, a.start_frame, a.stop_frame, a.no_encode, a.intro_only)
