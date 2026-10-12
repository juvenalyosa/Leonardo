#!/usr/bin/env python3
"""Molecular films 04/05, matching the white-stage cinematic_v2 design."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/leonardo-mutation-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pymol2
import render_cinematic_v2 as style

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cinematic_v2'
WORK = OUT / 'mutations'
OLD = ROOT / 'artifacts/cinematic_real'
W, H = 1920, 1080
TEAL, GOLD, INK, MUTED, LINE = style.TEAL, style.GOLD, style.INK, style.MUTED, style.LINE
NAMES = {'binding': '04_mutation_protein_ligand', 'enzyme': '05_mutation_enzyme_reaction'}
MUTANTS = {'binding': 'D189A', 'enzyme': 'W125A'}
txt = style.txt


def load(kind):
    result = json.loads((WORK / kind / 'summary.json').read_text())
    if kind == 'enzyme':
        result['paths'] = {'WT': dict(np.load(OUT / 'qmmm/path.npz')),
                           'W125A': dict(np.load(WORK / 'enzyme/W125A/path.npz'))}
    return result


def plot_image(kind, data, variant='WT', idx=0, export=False):
    if export and kind == 'enzyme':
        variant = 'W125A'
        idx = int(data['paths'][variant]['ts_index'])
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'svg.fonttype': 'none', 'axes.labelsize': 12})
    fig = plt.figure(figsize=(7.5, 5.35), dpi=140, facecolor='white')
    a = fig.add_axes([0.13, 0.51, 0.82, 0.40])
    b = fig.add_axes([0.13, 0.105, 0.82, 0.235])
    for ax in [a, b]:
        style.axes_style(ax)
    mutant = MUTANTS[kind]
    if kind == 'binding':
        for j, v in enumerate(['WT', mutant]):
            values = np.array([r['score_kcal_mol'] for r in data['best_poses'][v]])
            color = TEAL if j == 0 else GOLD
            a.scatter(j + np.linspace(-0.10, 0.10, len(values)), values, marker='o' if j == 0 else 's',
                      color=color, s=50, zorder=5, edgecolor='white', linewidth=0.7)
            a.hlines(np.median(values), j-0.24, j+0.24, color=color, lw=2.2, zorder=4)
            a.text(j, max(values)+0.08, f'{np.median(values):.2f}', ha='center', color=color, fontsize=12, weight='bold')
        for i in range(3):
            a.plot([0, 1], [data['best_poses']['WT'][i]['score_kcal_mol'], data['best_poses'][mutant][i]['score_kcal_mol']],
                   color='#D5DFE2', lw=0.8, zorder=2)
        a.set_xticks([0, 1], ['Wild type', 'D189A'])
        a.set_xlim(-0.5, 1.5)
        scores = [r['score_kcal_mol'] for rr in data['best_poses'].values() for r in rr]
        a.set_ylim(min(scores)-0.13, max(scores)+0.25)
        a.set_ylabel('Best Vina score (kcal mol$^{-1}$)')
        a.text(0.98, 0.03, '3 matched search seeds · line = median', ha='right', va='bottom', transform=a.transAxes, color=MUTED, fontsize=10)
        for j, v in enumerate(['WT', mutant]):
            rr = [r for r in data['rows'] if r['variant'] == v and r['seed'] == data['seeds'][0]]
            b.plot([r['rank'] for r in rr], [r['crystal_rmsd_A'] for r in rr], color=TEAL if j == 0 else GOLD,
                   marker='o' if j == 0 else 's', ms=3.2, lw=1.4, label='Wild type' if j == 0 else v)
            if v == variant:
                b.scatter(rr[idx]['rank'], rr[idx]['crystal_rmsd_A'], color=TEAL if j == 0 else GOLD, s=75, edgecolor='white', zorder=5)
        b.axhline(2, color='#ADBFC6', ls=(0, (3, 4)), lw=0.7)
        b.set_xlabel('Pose rank · first matched seed')
        b.set_ylabel('Crystal RMSD (Å)')
        b.set_xticks(range(1, max(len([r for r in data['rows'] if r['variant'] == v and r['seed'] == data['seeds'][0]]) for v in ['WT', mutant])+1))
        b.legend(frameon=False, fontsize=10, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.39))
    else:
        for v, color, ls in [('WT', TEAL, '-'), (mutant, GOLD, '--')]:
            p = data['paths'][v]
            # Physical bond-difference coordinate avoids equating the two arc lengths.
            x = p['d_oc'] - p['d_ccl']
            a.plot(x, p['relative'], color=color, lw=1.9, ls=ls, label='Wild type' if v == 'WT' else v)
            k = int(p['ts_index'])
            a.scatter(x[k], p['relative'][k], marker='o' if v == 'WT' else 's', s=45, c=color, edgecolors='white', zorder=4)
            if v == variant:
                a.scatter(x[idx], p['relative'][idx], s=100, c=color, edgecolors='white', linewidths=1.5, zorder=6)
        a.invert_xaxis()
        a.set_ylabel('Relative energy (kcal mol$^{-1}$)')
        a.set_xlabel('$d$(O–C) − $d$(C–Cl) (Å)')
        a.legend(frameon=False, fontsize=10, loc='upper left', ncol=2, bbox_to_anchor=(-0.02, 1.26))
        a.text(0.01, 0.02, 'Reactant', transform=a.transAxes, color=MUTED, fontsize=10)
        a.text(0.99, 0.02, 'Alkyl–enzyme', transform=a.transAxes, ha='right', color=MUTED, fontsize=10)
        vals = [data['WT_barrier_kcal_mol'], data['mutant_barrier_kcal_mol']]
        b.barh([1, 0], vals, height=0.46, color=[TEAL, GOLD], zorder=3)
        b.set_yticks([1, 0], ['Wild type', mutant])
        b.set_xlim(0, max(vals)*1.24)
        for y, value in zip([1, 0], vals):
            b.text(value+0.2, y, f'{value:.2f}', va='center', color=INK, fontsize=11)
        b.set_xlabel('Potential-energy barrier (kcal mol$^{-1}$)')
    if export:
        fig.text(0.13, 0.985, 'Trypsin · D189A · smina/Vina' if kind == 'binding' else 'DhlA · W125A · PM7/Amber14 QM/MM',
                 fontsize=12, color=INK, weight='bold', va='top')
        for ext in ['svg', 'pdf', 'png']:
            fig.savefig(OUT / 'plots' / f'mutation_{kind}.{ext}', dpi=300)
    fig.canvas.draw()
    result = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
    plt.close(fig)
    return result


def setup(c, kind, variant):
    wide, close = style.setup(c, 'docking' if kind == 'binding' else 'qmmm', {})
    residue = '189' if kind == 'binding' else '125'
    if variant != 'WT':
        for name in ['enzyme', 'ribbon', 'envelope', 'pocket']:
            c.delete(name)
        pdb = WORK / kind / 'data' / ('trypsin_amber.pdb' if kind == 'binding' else 'dhla_amber.pdb')
        c.load(str(pdb), 'enzyme')
        c.hide('everything', 'enzyme')
        c.color('slate', 'enzyme')
        c.color('teal', 'enzyme and resi ' + ('123-245' if kind == 'binding' else '156-229'))
        c.create('ribbon', 'enzyme'); c.hide('everything', 'ribbon'); c.show('cartoon', 'ribbon')
        c.create('envelope', 'enzyme and not hydro'); c.hide('everything', 'envelope'); c.show('surface', 'envelope')
        c.color('slate', 'envelope'); c.set('transparency', 0, 'envelope')
    c.hide('everything', 'enzyme')
    c.delete('pocket')
    neighbors = '190+191+192+214+215+216+226' if kind == 'binding' else '175'
    c.create('pocket', 'enzyme and resi ' + neighbors + ' and not hydro and not name N+CA+C+O')
    c.hide('everything', 'pocket'); c.show('sticks', 'pocket'); c.set('stick_radius', 0.085, 'pocket')
    c.color('slate', 'pocket and elem C'); c.color('oxygen', 'pocket and elem O'); c.color('nitrogen', 'pocket and elem N')
    c.create('mutation', 'enzyme and resi ' + residue + ' and not hydro and not name N+C+O')
    c.hide('everything', 'mutation'); c.show('sticks', 'mutation'); c.show('spheres', 'mutation')
    c.color('teal' if variant == 'WT' else 'gold', 'mutation')
    c.color('oxygen', 'mutation and elem O'); c.color('nitrogen', 'mutation and elem N')
    c.set('stick_radius', 0.15, 'mutation'); c.set('sphere_scale', 0.24, 'mutation')
    if kind == 'binding':
        # Show the same three ranks used in the movie from the first matched seed.
        c.delete('ligand')
        c.load(str(WORK / 'binding' / variant / 'pose_20260915_01.pdb'), 'ligand')
        c.remove('ligand and hydro'); c.hide('everything', 'ligand')
        c.show('sticks', 'ligand'); c.show('spheres', 'ligand')
        c.color('gold', 'ligand'); c.color('nitrogen', 'ligand and elem N')
        c.set('stick_radius', 0.16, 'ligand'); c.set('sphere_scale', 0.25, 'ligand')
    # A common reference geometry fixes the camera for both variants, even though
    # the mutant has fewer atoms. Include WT side chain in the close-up framing.
    c.load(str(OLD / 'data/trypsin_amber.pdb' if kind == 'binding' else OUT / 'data/dhla_amber.pdb'), 'reference')
    c.hide('everything', 'reference')
    c.set_view(close.tolist())
    c.zoom(('ligand' if kind == 'binding' else 'quantum') + ' or (reference and resi ' + residue + ' and not hydro and not name N+C+O)', 3.2)
    close = np.array(c.get_view())
    close[15] = max(0.1, -close[11]-90); close[16] = -close[11]+90
    if kind == 'enzyme':
        # Separate Trp125 from Asp124 and the exchanging bonds in projection.
        # Rotate both endpoints equally to preserve a rigid camera basis.
        for view in [wide, close]:
            c.set_view(view.tolist()); c.turn('x', -20); c.turn('y', 65)
            view[:] = c.get_view()
        c.set_view(close.tolist())
    c.delete('reference')
    c.set('cartoon_transparency', 0, 'ribbon')
    c.set('auto_zoom', 0)
    return wide, close


def update(c, kind, data, variant, idx):
    if kind == 'binding':
        c.delete('pose_source')
        c.load(str(WORK / 'binding' / variant / f'pose_20260915_{idx+1:02d}.pdb'), 'pose_source')
        c.remove('pose_source and hydro')
        c.load_coords(c.get_coords('pose_source'), 'ligand'); c.delete('pose_source')
        c.delete('contact')
        if variant == 'WT':
            c.distance('contact', 'ligand and elem N', 'mutation and name OD1+OD2', 3.5)
            c.hide('labels', 'contact'); c.set('dash_color', 'teal', 'contact'); c.set('dash_radius', 0.035, 'contact')
    else:
        style.update_state(c, 'qmmm', data['paths'][variant], idx)
        c.delete('halide_contact')
        if variant == 'WT':
            c.distance('halide_contact', 'mutation and name NE1', 'quantum and name CL1')
            c.hide('labels', 'halide_contact'); c.set('dash_color', 'teal', 'halide_contact')
            c.set('dash_radius', 0.027, 'halide_contact')


def state(kind, data, seconds):
    variant = 'WT' if seconds < 12 else MUTANTS[kind]
    if kind == 'binding':
        if seconds < 8 or seconds >= 19:
            return variant, 0
        return variant, min(2, int((seconds - (8 if variant == 'WT' else 12)) / (1.34 if variant == 'WT' else 2.3)))
    p = data['paths'][variant]
    if seconds < 7.6:
        return variant, 0
    if seconds >= 19:
        return variant, int(p['ts_index'])
    t = np.clip((seconds - (7.6 if variant == 'WT' else 12)) / (4.3 if variant == 'WT' else 6.9), 0, 1)
    ts = int(p['ts_index'])
    if t < 0.43:
        return variant, int(t/0.43*ts)
    if t < 0.63:
        return variant, ts
    return variant, min(len(p['relative'])-1, ts + int((t-0.63)/0.37*(len(p['relative'])-1-ts)))


def compose(kind, data, molecule, pair, plot, variant, idx, seconds):
    im = Image.new('RGBA', (W, H), 'white')
    intro = 1-style.ease((seconds-2.6)/1.0)
    analysis = style.ease((seconds-6.7)/1.1)
    comparison = style.ease((seconds-19)/0.6)
    scale = 1-0.31*analysis
    px, py = int(160+440*intro-160*analysis), int(125+115*analysis)
    if comparison < 1:
        mol = molecule.resize((int(1600*scale), int(900*scale)), Image.Resampling.LANCZOS)
        im.paste(mol, (px, py))
    if comparison > 0:
        layer = Image.new('RGBA', (W, H), 'white')
        for j, v in enumerate(['WT', MUTANTS[kind]]):
            pic = pair[v].crop((320, 25, 1280, 875)).resize((520, 460), Image.Resampling.LANCZOS)
            layer.paste(pic, (12+j*520, 329))
        ld = ImageDraw.Draw(layer)
        txt(ld, (102, 303), 'WILD TYPE', 20, TEAL, True)
        txt(ld, (622, 303), MUTANTS[kind] + ' MUTANT', 20, GOLD, True)
        ld.line((536, 353, 536, 772), fill=LINE, width=1)
        style.faded(im, layer, comparison)
    d = ImageDraw.Draw(im)
    txt(d, (78, 42), 'L E O N A R D O', 25, bold=True)
    txt(d, (1510, 46), 'MOLECULAR FILMS  /  ' + ('04' if kind == 'binding' else '05'), 18, MUTED)
    d.line((78, 93, 1842, 93), fill=LINE, width=1)
    if intro > 0:
        layer = Image.new('RGBA', (W, H)); ld = ImageDraw.Draw(layer)
        txt(ld, (85, 252), '04 / MUTATION · LIGAND BINDING' if kind == 'binding' else '05 / MUTATION · ENZYME REACTION', 20, TEAL, True)
        lines = ['One residue.', 'A different fit.'] if kind == 'binding' else ['One mutation.', 'A new barrier.']
        for j, line in enumerate(lines):
            txt(ld, (78, 307+j*102), line, 83, bold=True)
        txt(ld, (86, 563), 'TRYPSIN · Asp189 → Ala' if kind == 'binding' else 'DhlA · Trp125 → Ala', 23, MUTED)
        txt(ld, (86, 604), 'PDB 3PTB · smina / Vina' if kind == 'binding' else 'PDB 2DHC · MOPAC PM7 / Amber14', 20, MUTED)
        ld.line((86, 680, 178, 680), fill=GOLD, width=4)
        txt(ld, (86, 713), 'One change. Calculated consequences.', 21)
        style.faded(im, layer, intro)
    if intro < 1:
        layer = Image.new('RGBA', (W, H)); ld = ImageDraw.Draw(layer)
        txt(ld, (80, 128), 'TRYPSIN · BENZAMIDINIUM · D189A' if kind == 'binding' else 'HALOALKANE DEHALOGENASE · W125A', 18, TEAL, True)
        caption = ('One pocket. Two molecular fits.' if kind == 'binding' else 'One reaction. Two energy barriers.') if comparison > 0 else (
                  'Changing the recognition pocket.' if kind == 'binding' else 'Reshaping the catalytic environment.')
        txt(ld, (76, 166), caption, 48, bold=True)
        style.faded(im, layer, 1-intro)
    if analysis > 0:
        layer = Image.new('RGBA', (W, H)); ld = ImageDraw.Draw(layer)
        for y in range(760, 816):
            ld.line((0, y, 1058, y), fill=(255, 255, 255, int(255*style.ease((y-760)/55))))
        ld.rectangle((0, 816, 1058, 999), fill='white')
        ld.rectangle((1060, 243, 1870, 997), fill='white')
        txt(ld, (1090, 254), 'The mutation, quantified.' if kind == 'binding' else 'The barrier, recalculated.', 29, bold=True)
        txt(ld, (1092, 300), 'Matched docking · 3 independent seeds' if kind == 'binding' else 'Wild type and W125A · each reactant = 0', 19, MUTED)
        layer.paste(plot.resize((805, 575), Image.Resampling.LANCZOS), (1050, 338))
        if comparison < 0.5:
            txt(ld, (86, 273), 'WILD TYPE' if variant == 'WT' else MUTANTS[kind] + ' MUTANT', 20, TEAL if variant == 'WT' else GOLD, True)
        if kind == 'binding':
            r = next(r for r in data['rows'] if r['variant'] == variant and r['seed'] == data['seeds'][0] and r['rank'] == idx+1)
            label = 'MEDIAN DOCKING SCORE SHIFT' if comparison > 0 else f'{variant} · DOCKING POSE {idx+1:02d}'
            metric = f"{data['delta_median_score_kcal_mol']:+.2f}" if comparison > 0 else f"{r['score_kcal_mol']:.2f}"
            detail = 'Asp189 carboxylate contacts removed by D189A' if comparison > 0 else (
                f"Asp189 N···O contacts ≤ 3.5 Å: {r['asp189_N_O_contacts_3p5_A']}" if variant == 'WT' else 'Ala189 replaces the charged carboxylate')
            note = 'Scores and poses respond to the substitution.'
            subtitle = 'Asp189 → Ala · loss of a charged side chain'
        else:
            p = data['paths'][variant]
            at_ts = idx == int(p['ts_index'])
            label = 'CHANGE IN POTENTIAL-ENERGY BARRIER' if comparison > 0 else variant + ' · ' + ('TRANSITION STATE' if at_ts else 'CALCULATED REACTION IMAGE')
            metric = f"{data['delta_barrier_kcal_mol']:+.2f}" if comparison > 0 else f"{p['relative'][idx]:+.2f}"
            detail = 'Both transition states: one imaginary mode' if comparison > 0 else f"O–C  {p['d_oc'][idx]:.2f} Å     C–Cl  {p['d_ccl'][idx]:.2f} Å"
            note = 'Fixed-protein model · optimized stationary states'
            subtitle = 'Trp125 → Ala · the halide-binding environment'
        txt(ld, (86, 777), subtitle, 19, MUTED)
        txt(ld, (84, 826), label, 18, TEAL, True)
        txt(ld, (79, 858), metric, 68, bold=True)
        width = ld.textbbox((0, 0), metric, font=style.font(68, True))[2]
        txt(ld, (94+width, 897), 'kcal/mol', 24, MUTED)
        txt(ld, (86, 945), detail, 22)
        txt(ld, (1092, 934), note, 18, MUTED)
        style.faded(im, layer, analysis)
    if 3.6 < seconds < 6.7:
        layer = Image.new('RGBA', (W, H)); ld = ImageDraw.Draw(layer)
        txt(ld, (84, 932), 'Asp189 anchors a positively charged ligand.' if kind == 'binding' else 'Trp125 sits beside the departing chloride.', 23)
        style.faded(im, layer, style.ease((seconds-3.6)/0.6)*(1-style.ease((seconds-6.1)/0.6)))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 1000, W, H), fill='white'); d.line((78, 1007, 1842, 1007), fill=LINE, width=1)
    d.line((78, 1007, 78+int(1764*seconds/(575/24)), 1007), fill=TEAL, width=3)
    foot = 'Rigid-receptor docking · Vina score is not binding free energy' if kind == 'binding' else 'First alkylation step · fixed-protein QM/MM · potential energy, not free energy'
    txt(d, (80, 1030), foot, 18, MUTED)
    txt(d, (1660, 1030), 'PDB ' + ('3PTB' if kind == 'binding' else '2DHC'), 18, MUTED)
    return im.convert('RGB')


def render(kind, preview=False, start=0, stop=576, no_encode=False):
    data = load(kind)
    folder = OUT / 'renders' / f'mutation_{kind}'; folder.mkdir(exist_ok=True)
    scratch = folder / f'worker_{os.getpid()}'; scratch.mkdir(exist_ok=True)
    plot_image(kind, data, export=True)
    instances = {}; views = {}; pair = {}; cached = {}; contexts = {}; plots = {}
    started = time.perf_counter()
    try:
        for variant in ['WT', MUTANTS[kind]]:
            p = pymol2.PyMOL(); p.start(); instances[variant] = p
            views[variant] = setup(p.cmd, kind, variant)
            if variant != 'WT':
                views[variant] = views['WT']
            final_idx = 0 if kind == 'binding' else int(data['paths'][variant]['ts_index'])
            update(p.cmd, kind, data, variant, final_idx)
            p.cmd.set_view(views[variant][1].tolist())
            pair[variant] = style.ray(p.cmd, scratch, 1)
            p.cmd.set('transparency', 0.88, 'envelope')
            p.cmd.save(str(OUT / 'sessions' / f'mutation_{kind}_{variant}.pse'))
            p.cmd.set('transparency', 0, 'envelope')
        # Force identical framing across WT and mutant.
        views[MUTANTS[kind]] = views['WT']
        targets = [0, 120, 240, 312, 396, 528, 575] if preview else range(start, min(stop, 576))
        for f in targets:
            path = folder / f'frame_{f:05d}.png'
            if path.exists() and not preview:
                continue
            seconds = f/24
            variant, idx = state(kind, data, seconds)
            push = style.ease((seconds-2.6)/4)
            p = instances[variant]; c = p.cmd
            key = (variant, idx, f if seconds < 6.6 else 'close')
            if key not in cached:
                wide, close = views[variant]
                c.set_view((wide*(1-push)+close*push).tolist())
                c.turn('y', -13*(1-push)+8*math.sin(min(seconds/6.6, 1)*math.pi)*(1-push))
                c.turn('x', -3*(1-push))
                update(c, kind, data, variant, idx)
                if seconds >= 6.6 and variant not in contexts:
                    enabled = c.get_names('objects', enabled_only=1)
                    c.disable('all'); c.enable('ribbon'); c.enable('pocket')
                    c.png(str(scratch / 'static_ribbon.png'), 1600, 900, ray=1, quiet=1)
                    c.disable('all'); c.enable('envelope')
                    c.png(str(scratch / 'static_surface.png'), 1600, 900, ray=1, quiet=1)
                    contexts[variant] = (Image.open(scratch / 'static_ribbon.png').convert('RGBA'), Image.open(scratch / 'static_surface.png').convert('RGBA'))
                    c.disable('all')
                    for name in enabled:
                        c.enable(name)
                cached[key] = style.ray(c, scratch, push, contexts.get(variant) if seconds >= 6.6 else None)
            if (variant, idx) not in plots:
                plots[variant, idx] = plot_image(kind, data, variant, idx)
            im = compose(kind, data, cached[key], pair, plots[variant, idx], variant, idx, seconds)
            im.save(path)
            if seconds < 6.6:
                del cached[key]
            if f % 24 == 0 or preview:
                print(kind, f, '/576', round(time.perf_counter()-started, 1), 'seconds', flush=True)
    finally:
        for p in instances.values():
            p.stop()
    if not preview and not no_encode:
        encode(kind)


def encode(kind):
    folder = OUT / 'renders' / f'mutation_{kind}'
    assert all((folder / f'frame_{i:05d}.png').exists() for i in range(576))
    p = subprocess.run(['/opt/homebrew/bin/ffmpeg', '-y', '-framerate', '24', '-i', str(folder / 'frame_%05d.png'),
                        '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an',
                        str(OUT / 'videos' / f'{NAMES[kind]}.mp4')], capture_output=True, text=True)
    (folder / 'ffmpeg.log').write_text(p.stderr)
    assert p.returncode == 0, p.stderr[-2000:]
    print('Encoded', NAMES[kind], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('kind', choices=list(NAMES))
    p.add_argument('--preview', action='store_true'); p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--stop-frame', type=int, default=576); p.add_argument('--no-encode', action='store_true')
    p.add_argument('--encode-only', action='store_true')
    a = p.parse_args()
    if a.encode_only:
        encode(a.kind)
    else:
        render(a.kind, a.preview, a.start_frame, a.stop_frame, a.no_encode)
