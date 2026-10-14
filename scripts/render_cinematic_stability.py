#!/usr/bin/env python3
"""Film 06: a designed disulfide, with lightweight structural evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/leonardo-mutation-mpl')
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymol2
import render_cinematic_v2 as style
from cinematic_stability_design import OUT, WORK
from PIL import Image, ImageDraw
from pymol.cgo import ALPHA, BEGIN, COLOR, END, LINES, LINEWIDTH, VERTEX

NAME = '06_protein_stability_design'
W, H = 1920, 1080
TEAL, GOLD, INK, MUTED, LINE = style.TEAL, style.GOLD, style.INK, style.MUTED, style.LINE
txt = style.txt


def load():
    s = json.loads((WORK / 'summary.json').read_text())
    s['network'] = dict(np.load(WORK / 'network.npz'))
    s['structures'] = {v: dict(np.load(WORK / v / 'relaxation.npz')) for v in ['original', 'design']}
    s['atoms'] = {v: json.loads((WORK / v / 'atoms.json').read_text()) for v in ['original', 'design']}
    return s


def plot(data, export=False):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'svg.fonttype': 'none', 'axes.labelsize': 12})
    fig = plt.figure(figsize=(7.5, 5.35), dpi=140, facecolor='white')
    a = fig.add_axes([0.13, 0.51, 0.82, 0.39])
    b = fig.add_axes([0.13, 0.12, 0.82, 0.23])
    for ax in [a, b]:
        style.axes_style(ax)
    net = data['network']
    resids = net['resids']
    pair = data['residue_pair']
    a.plot(resids, net['original_mobility'], color=TEAL, lw=1.8, label='Original')
    a.plot(resids, net['design_mobility'], color=GOLD, lw=1.8, ls='--', label='Disulfide model')
    for resid in pair:
        k = list(resids).index(resid)
        a.axvspan(resid - 0.7, resid + 0.7, color=GOLD, alpha=0.12, lw=0)
        a.scatter([resid], [net['design_mobility'][k]], s=48, c=GOLD, edgecolor='white', zorder=5)
    a.set_xlabel('Residue')
    a.set_ylabel('Relative RMS mobility')
    a.set_xlim(1, 56)
    a.set_xticks([1, 10, 20, 30, 40, 50, 56])
    a.set_ylim(bottom=0)
    a.legend(frameon=False, loc='lower left', bbox_to_anchor=(-0.02, 1.0), fontsize=10, ncol=2)
    ratio = np.sqrt(data['pair_variance_design'] / data['pair_variance_original'])
    b.barh([1, 0], [1, ratio], color=[TEAL, GOLD], height=0.45, zorder=3)
    b.set_yticks([1, 0], ['Original', 'Design'])
    b.set_xlim(0, 1.25)
    b.text(1.025, 1, '1.00', va='center', color=INK, fontsize=11)
    b.text(ratio + 0.025, 0, f'{ratio:.2f}', va='center', color=INK, fontsize=11)
    b.set_xlabel(f'Relative RMS motion between residues {pair[0]} and {pair[1]}')
    if export:
        fig.text(
            0.13,
            0.985,
            'GB1 · disulfide design · Gaussian network prediction',
            va='top',
            fontsize=12,
            color=INK,
            weight='bold',
        )
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(OUT / 'plots' / f'protein_stability.{ext}', dpi=300)
        extra, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
        for ax in axes:
            style.axes_style(ax)
        for v, color in [('original', TEAL), ('design', GOLD)]:
            energy = data['structures'][v]['energy_kcal_mol']
            axes[0].plot(np.arange(len(energy)), energy - energy[0], color=color, label=v.title(), lw=1.6)
        axes[0].set(xlabel='Saved minimization batch', ylabel='Energy change within variant (kcal/mol)')
        axes[0].legend(frameon=False)
        axes[0].set_title('Separate restrained relaxations', color=INK, fontsize=12)
        for cutoff, color in [(7, '#A5BDC5'), (7.5, TEAL), (8, GOLD)]:
            rows = [r for r in data['sensitivity'] if r['cutoff_A'] == cutoff]
            axes[1].plot(
                [r['extra_stiffness'] for r in rows],
                [r['pair_relative_rms_reduction_percent'] for r in rows],
                marker='o',
                color=color,
                label=f'Cutoff {cutoff:g} Å',
            )
        axes[1].set(
            xlabel='Additional cross-link stiffness / contact stiffness', ylabel='Local RMS motion reduction (%)'
        )
        axes[1].legend(frameon=False)
        axes[1].set_title('Sensitivity to explicit model assumptions', color=INK, fontsize=12)
        for ext in ['png', 'pdf', 'svg']:
            extra.savefig(OUT / 'plots' / f'protein_stability_checks.{ext}', dpi=300)
        plt.close(extra)
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
    plt.close(fig)
    return im


def setup(c, data, variant):
    style.base.setup(c, 'docking')
    c.delete('all')
    c.load(str(WORK / variant / 'model.pdb'), 'protein')
    c.hide('everything')
    c.color('slate', 'protein')
    c.color('teal', 'protein and resi 23-36')
    pair = '+'.join(map(str, data['residue_pair']))
    c.color('teal', 'protein and resi ' + pair)
    c.create('ribbon', 'protein')
    c.hide('everything', 'ribbon')
    c.show('cartoon', 'ribbon')
    c.create('envelope', 'protein and not hydro')
    c.hide('everything', 'envelope')
    c.show('surface', 'envelope')
    c.color('slate', 'envelope')
    c.create('site', 'protein and resi ' + pair + ' and not hydro and not name N+C+O+OXT')
    c.hide('everything', 'site')
    c.show('sticks', 'site')
    c.show('spheres', 'site')
    c.color('teal', 'site')
    c.color('oxygen', 'site and elem O')
    c.color('nitrogen', 'site and elem N')
    c.color('gold', 'site and elem S')
    c.set('stick_radius', 0.16, 'site')
    c.set('sphere_scale', 0.27, 'site')
    c.set('sphere_scale', 0.33, 'site and elem S')
    i, j = data['residue_pair']
    if variant == 'design':
        c.bond(f'site and resi {i} and name SG', f'site and resi {j} and name SG')
    else:
        c.distance('candidate', f'site and resi {i} and name CB', f'site and resi {j} and name CB')
        c.hide('labels', 'candidate')
        c.set('dash_color', 'gold', 'candidate')
        c.set('dash_radius', 0.025, 'candidate')
    c.orient('site')
    c.turn('x', 20)
    c.turn('y', -25)
    c.turn('z', 15)
    c.zoom('protein', 3.0)
    wide = np.array(c.get_view())
    c.zoom('site', 3.1)
    close = np.array(c.get_view())
    for view in [wide, close]:
        view[15] = max(0.1, -view[11] - 90)
        view[16] = -view[11] + 90
        c.set_view(view.tolist())
        c.turn('y', 40)
        view[:] = c.get_view()
    for key, value in [
        ('ray_shadows', 1),
        ('ambient', 0.40),
        ('direct', 0.67),
        ('specular', 0.42),
        ('shininess', 45),
        ('antialias', 1),
        ('max_threads', 2),
        ('ray_opaque_background', 0),
        ('auto_zoom', 0),
    ]:
        c.set(key, value)
    c.set('transparency', 0, 'envelope')
    c.set('cartoon_transparency', 0, 'ribbon')
    return wide, close


def update(c, data, variant, idx):
    info = data['structures'][variant]
    xyz = info['xyz'][idx]
    c.load_coords(xyz, 'protein')
    c.load_coords(xyz, 'ribbon')
    c.load_coords(xyz[info['heavy']], 'envelope')
    inds = [
        a['index']
        for a in data['atoms'][variant]
        if a['resid'] in data['residue_pair'] and a['element'] != 'H' and a['name'] not in ['N', 'C', 'O', 'OXT']
    ]
    c.load_coords(xyz[inds], 'site')


def network(c, data):
    coords = data['structures']['design']['xyz'][-1, data['structures']['design']['ca']]
    k = data['network']['original_kirchhoff']
    drawing = [LINEWIDTH, 1, ALPHA, 0.22, COLOR, 0.33, 0.52, 0.55, BEGIN, LINES]
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if k[i, j] < 0:
                drawing.extend([VERTEX, *coords[i], VERTEX, *coords[j]])
    drawing.extend([END])
    c.load_cgo(drawing, 'elastic_network')
    c.create('nodes', 'protein and name CA')
    c.hide('everything', 'nodes')
    c.show('spheres', 'nodes')
    c.set('sphere_scale', 0.12, 'nodes')
    c.color('teal', 'nodes')
    c.set('sphere_transparency', 0.40, 'nodes')


def compose(data, mol, plot_im, seconds, variant, idx):
    im = Image.new('RGBA', (W, H), 'white')
    intro = 1 - style.ease((seconds - 2.6) / 1.0)
    analysis = style.ease((seconds - 7) / 1)
    scale = 1 - 0.31 * analysis
    px = int(160 + 440 * intro - 160 * analysis)
    py = int(125 + 115 * analysis)
    im.paste(mol.resize((int(1600 * scale), int(900 * scale)), Image.Resampling.LANCZOS), (px, py))
    d = ImageDraw.Draw(im)
    txt(d, (78, 42), 'L E O N A R D O', 25, bold=True)
    txt(d, (1510, 46), 'MOLECULAR FILMS  /  06', 18, MUTED)
    d.line((78, 93, 1842, 93), fill=LINE, width=1)
    if intro > 0:
        layer = Image.new('RGBA', (W, H))
        ld = ImageDraw.Draw(layer)
        txt(ld, (85, 252), '06 / PROTEIN DESIGN', 20, TEAL, True)
        txt(ld, (78, 307), 'Built to', 86, bold=True)
        txt(ld, (78, 409), 'hold.', 86, bold=True)
        txt(ld, (86, 563), 'DESIGNING A MORE RIGID FOLD', 21, MUTED)
        txt(ld, (86, 604), 'GB1 · 56 residues · one new disulfide', 20, MUTED)
        ld.line((86, 680, 178, 680), fill=GOLD, width=4)
        txt(ld, (86, 713), 'A small change. A molecular lock.', 21)
        style.faded(im, layer, intro)
    if intro < 1:
        layer = Image.new('RGBA', (W, H))
        ld = ImageDraw.Draw(layer)
        txt(ld, (80, 128), 'PROTEIN G B1 DOMAIN · ' + data['design'], 18, TEAL, True)
        title = (
            'Find the right connection.'
            if seconds < 10
            else 'A new covalent lock.'
            if seconds < 19
            else 'A fold, reinforced by design.'
        )
        txt(ld, (76, 166), title, 48, bold=True)
        style.faded(im, layer, 1 - intro)
    if analysis > 0:
        layer = Image.new('RGBA', (W, H))
        ld = ImageDraw.Draw(layer)
        for y in range(760, 816):
            ld.line((0, y, 1058, y), fill=(255, 255, 255, int(255 * style.ease((y - 760) / 55))))
        ld.rectangle((0, 816, 1058, 999), fill='white')
        ld.rectangle((1060, 243, 1870, 997), fill='white')
        txt(ld, (1090, 254), 'The design, quantified.', 29, bold=True)
        txt(ld, (1092, 300), 'Gaussian network · predicted relative mobility', 19, MUTED)
        layer.paste(plot_im.resize((805, 575), Image.Resampling.LANCZOS), (1050, 338))
        caption = 'ORIGINAL STRUCTURE' if variant == 'original' else 'PROPOSED DISULFIDE DESIGN'
        txt(ld, (86, 272), caption, 18, TEAL if variant == 'original' else GOLD, True)
        if seconds < 10:
            metric = str(data['sequence_span'])
            unit = 'residues apart'
            state = 'TWO SITES. ONE FOLD.'
            detail = 'Candidate sites meet in three-dimensional space.'
        elif seconds < 19:
            atom = data['atoms']['design']
            xyz = data['structures']['design']['xyz'][idx]
            sg = [a['index'] for a in atom if a['name'] == 'SG']
            metric = f'{np.linalg.norm(xyz[sg[0]]-xyz[sg[1]]):.2f}'
            unit = 'Å'
            state = 'S–S DISTANCE · LOCAL REFINEMENT'
            detail = 'Amber14 / OBC-II · softly restrained backbone'
        else:
            metric = f"−{data['pair_relative_rms_reduction_percent']:.0f}%"
            unit = 'local RMS motion'
            state = 'ELASTIC-NETWORK PREDICTION'
            detail = 'Model comparison at the two linked residues'
        txt(
            ld,
            (86, 777),
            f"Cys{data['residue_pair'][0]}–Cys{data['residue_pair'][1]} · a proposed molecular clamp",
            19,
            MUTED,
        )
        txt(ld, (84, 826), state, 18, TEAL, True)
        txt(ld, (79, 858), metric, 68, bold=True)
        width = ld.textbbox((0, 0), metric, font=style.font(68, True))[2]
        txt(ld, (94 + width, 897), unit, 24, MUTED)
        txt(ld, (86, 945), detail, 22)
        txt(ld, (1092, 934), '7.5 Å contact cutoff · added link stiffness 10×', 18, MUTED)
        style.faded(im, layer, analysis)
    if 3.6 < seconds < 6.9:
        layer = Image.new('RGBA', (W, H))
        ld = ImageDraw.Draw(layer)
        txt(ld, (84, 932), 'Search the fold. Connect two distant sequence positions.', 23)
        style.faded(im, layer, style.ease((seconds - 3.6) / 0.6) * (1 - style.ease((seconds - 6.3) / 0.6)))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 1000, W, H), fill='white')
    d.line((78, 1007, 1842, 1007), fill=LINE, width=1)
    d.line((78, 1007, 78 + int(1764 * seconds / (575 / 24)), 1007), fill=TEAL, width=3)
    txt(d, (80, 1030), 'Structural design + harmonic rigidity model · not a thermal-stability measurement', 18, MUTED)
    txt(d, (1660, 1030), 'PDB 1PGA', 18, MUTED)
    return im.convert('RGB')


def render(preview=False, start=0, stop=576, no_encode=False):
    data = load()
    folder = OUT / 'renders/stability'
    folder.mkdir(exist_ok=True)
    scratch = folder / f'worker_{os.getpid()}'
    scratch.mkdir(exist_ok=True)
    plot(data, True)
    plot_im = plot(data)
    instances = {}
    views = {}
    cache = {}
    started = time.perf_counter()
    try:
        for variant in ['original', 'design']:
            p = pymol2.PyMOL()
            p.start()
            instances[variant] = p
            views[variant] = setup(p.cmd, data, variant)
        # Both variants use the designed site's camera, guaranteeing registration.
        views['original'] = views['design']
        for variant, p in instances.items():
            p.cmd.set_view(views[variant][1].tolist())
            p.cmd.set('transparency', 0.85, 'envelope')
            p.cmd.save(str(OUT / 'sessions' / f'stability_{variant}.pse'))
            p.cmd.set('transparency', 0, 'envelope')
        targets = [0, 120, 216, 264, 336, 432, 528, 575] if preview else range(start, min(stop, 576))
        network_ready = False
        for f in targets:
            target = folder / f'frame_{f:05d}.png'
            if target.exists() and not preview:
                continue
            seconds = f / 24
            variant = 'original' if seconds < 10 else 'design'
            c = instances[variant].cmd
            idx = 30 if variant == 'original' else int(np.clip((seconds - 10) / 5, 0, 1) * 30)
            push = style.ease((seconds - 2.6) / 4)
            out = style.ease((seconds - 18.5) / 2)
            wide, close = views[variant]
            # Pull back for the network reveal; do not deform molecular positions.
            blend = push * (1 - 0.68 * out)
            view = wide * (1 - blend) + close * blend
            c.set_view(view.tolist())
            turn = (-16 + seconds * 3.1) * (1 - push)
            c.turn('y', turn)
            c.turn('x', -4 * (1 - push))
            update(c, data, variant, idx)
            if seconds >= 19 and not network_ready:
                network(c, data)
                network_ready = True
            if variant == 'design':
                if seconds >= 19:
                    c.enable('elastic_network')
                    c.enable('nodes')
                else:
                    c.disable('elastic_network')
                    c.disable('nodes')
            key = (variant, idx, round(blend, 8), round(turn, 8), seconds >= 19)
            if key not in cache:
                cache[key] = style.ray(c, scratch, float(0.65 * push))
            molecule = cache[key]
            im = compose(data, molecule, plot_im, seconds, variant, idx)
            temp = scratch / target.name
            im.save(temp)
            temp.replace(target)
            if seconds < 6.6 or 18.5 < seconds < 20.5:
                del cache[key]
            if f % 24 == 0 or preview:
                print('stability', f, '/576', round(time.perf_counter() - started, 1), 'seconds', flush=True)
    finally:
        for p in instances.values():
            p.stop()
    if not preview and not no_encode:
        encode()


def encode():
    folder = OUT / 'renders/stability'
    assert all((folder / f'frame_{i:05d}.png').exists() for i in range(576))
    p = subprocess.run(
        [
            '/opt/homebrew/bin/ffmpeg',
            '-y',
            '-framerate',
            '24',
            '-i',
            str(folder / 'frame_%05d.png'),
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
    (folder / 'ffmpeg.log').write_text(p.stderr)
    assert p.returncode == 0, p.stderr[-1000:]
    print('Encoded', NAME, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--preview', action='store_true')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--stop-frame', type=int, default=576)
    p.add_argument('--no-encode', action='store_true')
    p.add_argument('--encode-only', action='store_true')
    a = p.parse_args()
    if a.encode_only:
        encode()
    else:
        render(a.preview, a.start_frame, a.stop_frame, a.no_encode)
