#!/usr/bin/env python3
"""Film 07: compare five actual GB1 double-mutant relaxations, cheaply."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/leonardo-mutation-mpl')
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import render_cinematic_v2 as style
from cinematic_stability_design import OUT, angle, covariance, dump, source_residues, torsion
from cinematic_stability_design import WORK as SOURCE
from PIL import Image

WORK = OUT / 'stability_multimutant'
ORDER = [(8, 55), (4, 51), (42, 55), (4, 17), (10, 56)]
TEAL, GOLD, INK, MUTED, LINE = style.TEAL, style.GOLD, style.INK, style.MUTED, style.LINE
COLORS = ['#7B9CAC', '#68889B', '#97ABAE', '#566E84', GOLD]
ONE = dict(
    zip(
        [
            'ALA',
            'ARG',
            'ASN',
            'ASP',
            'CYS',
            'GLN',
            'GLU',
            'GLY',
            'HIS',
            'ILE',
            'LEU',
            'LYS',
            'MET',
            'PHE',
            'PRO',
            'SER',
            'THR',
            'TRP',
            'TYR',
            'VAL',
        ],
        'ARNDCQEGHILKMFPSTWYV',
    )
)


def table(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def geometry(xyz, atoms, pair, reference):
    index = {(a['resid'], a['name']): a['index'] for a in atoms}
    i, j = pair
    a, b, c, d = [xyz[index[k]].copy() for k in [(i, 'CB'), (i, 'SG'), (j, 'SG'), (j, 'CB')]]
    ca = np.array([xyz[a['index']] for a in atoms if a['name'] == 'CA'])
    return dict(
        SS_A=float(np.linalg.norm(b - c)),
        CSS_i_deg=angle(a, b, c),
        CSS_j_deg=angle(b, c, d),
        chi3_deg=torsion(a, b, c, d),
        ca_displacement_rms_A=float(np.sqrt(np.mean(np.sum((ca - reference) ** 2, axis=1)))),
    )


def filters(g):
    return [
        1.95 <= g['SS_A'] <= 2.15,
        90 <= g['CSS_i_deg'] <= 120 and 90 <= g['CSS_j_deg'] <= 120,
        65 <= abs(g['chi3_deg']) <= 115,
        g['ca_displacement_rms_A'] < 0.7,
    ]


def build():
    started = time.perf_counter()
    WORK.mkdir(exist_ok=True)
    residues = source_residues()
    coords = np.array([r['CA'] for r in residues.values()])
    n = len(coords)
    old = json.loads((SOURCE / 'screen.json').read_text())
    old_rows = {tuple(r['pair']): r for r in old['relaxed_candidates']}
    k0, c0, _, _ = covariance(coords, [0, 1], 0)
    norm = np.sqrt(np.mean(np.diag(c0)))
    np.savez_compressed(
        WORK / 'reference_network.npz', coords=coords, kirchhoff=k0, covariance=c0, mobility=np.sqrt(np.diag(c0)) / norm
    )
    rows = []
    steps = []
    sensitivities = []
    mobility = []
    sequences = {}
    for variant in ['original'] + [f'candidate_{i}_{j}' for i, j in ORDER]:
        atoms = json.loads((SOURCE / variant / 'atoms.json').read_text())
        sequences[variant] = ''.join(ONE[a['resname']] for a in atoms if a['name'] == 'CA')
    for q, (i, j) in enumerate(ORDER):
        variant = f'candidate_{i}_{j}'
        atoms = json.loads((SOURCE / variant / 'atoms.json').read_text())
        d = np.load(SOURCE / variant / 'relaxation.npz')
        label = f'{ONE[residues[i]["resname"]]}{i}C/{ONE[residues[j]["resname"]]}{j}C'
        final = geometry(d['xyz'][-1], atoms, [i, j], coords)
        passed = filters(final)
        assert all(passed) == old_rows[(i, j)]['passed_geometry_filter']
        assert abs(final['chi3_deg'] - old_rows[(i, j)]['chi3_deg']) < 1e-9
        e = np.zeros(n)
        e[i - 1] = 1
        e[j - 1] = -1
        _, c, _, _ = covariance(coords, [i - 1, j - 1], 10)
        # Verify the eigensolve against the independent rank-one inverse update.
        ce = c0 @ e
        independent = c0 - 10 * np.outer(ce, ce) / (1 + 10 * e @ ce)
        error = float(np.max(np.abs(c - independent)))
        assert error < 1e-10
        assert np.linalg.eigvalsh(c0 - c).min() > -1e-9
        local = 100 * (1 - np.sqrt((e @ c @ e) / (e @ c0 @ e)))
        global_rms = 100 * (1 - np.sqrt(np.trace(c) / np.trace(c0)))
        row = dict(
            id=q + 1,
            label=label,
            folder=variant,
            pair=[i, j],
            color=COLORS[q],
            **final,
            geometry_checks=passed,
            passed_geometry_filter=all(passed),
            local_rms_reduction_percent=float(local),
            global_rms_reduction_percent=float(global_rms),
            independent_covariance_max_error=error,
        )
        rows.append(row)
        np.savez_compressed(WORK / f'{variant}_network.npz', covariance=c, mobility=np.sqrt(np.diag(c)) / norm)
        for idx, xyz in enumerate(d['xyz']):
            steps.append(
                dict(
                    mutant=label,
                    batch=idx,
                    energy_change_kcal_mol=float(d['energy_kcal_mol'][idx] - d['energy_kcal_mol'][0]),
                    **geometry(xyz, atoms, [i, j], coords),
                )
            )
        for cutoff in [7, 7.5, 8]:
            _, ref, _, _ = covariance(coords, [i - 1, j - 1], 0, cutoff)
            for strength in [5, 10, 20]:
                _, m, _, _ = covariance(coords, [i - 1, j - 1], strength, cutoff)
                sensitivities.append(
                    dict(
                        mutant=label,
                        cutoff_A=cutoff,
                        extra_stiffness=strength,
                        local_rms_reduction_percent=float(100 * (1 - np.sqrt((e @ m @ e) / (e @ ref @ e)))),
                        global_rms_reduction_percent=float(100 * (1 - np.sqrt(np.trace(m) / np.trace(ref)))),
                    )
                )
        for r in range(n):
            mobility.append(dict(mutant=label, residue=r + 1, relative_RMS=float(np.sqrt(c[r, r]) / norm)))
    for r in range(n):
        mobility.append(dict(mutant='Original', residue=r + 1, relative_RMS=float(np.sqrt(c0[r, r]) / norm)))
    for variant, seq in sequences.items():
        (WORK / f'{variant}.fasta').write_text(f'>GB1_1PGA_{variant}\n{seq}\n')
    (WORK / 'all_variants.fasta').write_text(''.join(f'>GB1_1PGA_{v}\n{s}\n' for v, s in sequences.items()))
    table(
        WORK / 'comparison.csv',
        [{k: v for k, v in r.items() if k not in ['pair', 'color', 'geometry_checks']} for r in rows],
    )
    table(WORK / 'refinement.csv', steps)
    table(WORK / 'sensitivity.csv', sensitivities)
    table(WORK / 'mobility.csv', mobility)
    summary = dict(
        system='GB1 / protein G B1 domain / PDB 1PGA chain A / 56 residues',
        mutants=rows,
        accepted=[r['label'] for r in rows if r['passed_geometry_filter']],
        selection_protocol=old['selection_protocol'],
        relaxation_seconds=sum(
            json.loads((SOURCE / r['folder'] / 'preparation.json').read_text())['runtime_seconds'] for r in rows
        ),
        comparison_seconds=time.perf_counter() - started,
        reference_sequence=sequences['original'],
        source='https://files.rcsb.org/download/1PGA.pdb',
        source_doi='10.1021/bi00181a032',
        model='Same crystal CA coordinates/contact map for all; 7.5 A contacts; unit contact spring; proposed disulfide adds a 10x spring.',
        scope='Five proposed oxidized double mutants. Geometry filter and conditional harmonic rigidity only, not measured thermostability, folding free energy, or activity. Added springs reduce covariance by construction.',
    )
    dump(WORK / 'summary.json', summary)
    print(
        json.dumps(
            dict(
                mutants=[
                    dict(
                        label=r['label'],
                        chi3=r['chi3_deg'],
                        passed=r['passed_geometry_filter'],
                        local_rms=r['local_rms_reduction_percent'],
                    )
                    for r in rows
                ],
                seconds=summary['comparison_seconds'],
            ),
            indent=2,
        ),
        flush=True,
    )


def load():
    data = json.loads((WORK / 'summary.json').read_text())
    data['reference'] = dict(np.load(WORK / 'reference_network.npz'))
    for row in data['mutants']:
        row['states'] = dict(np.load(SOURCE / row['folder'] / 'relaxation.npz'))
        row['atoms'] = json.loads((SOURCE / row['folder'] / 'atoms.json').read_text())
        row['network'] = dict(np.load(WORK / f'{row["folder"]}_network.npz'))
    data['refinement'] = list(csv.DictReader((WORK / 'refinement.csv').open()))
    data['sensitivity'] = list(csv.DictReader((WORK / 'sensitivity.csv').open()))
    return data


def finish(fig, name, export):
    if export:
        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(OUT / 'plots' / f'{name}.{ext}', dpi=300)
    fig.canvas.draw()
    im = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
    plt.close(fig)
    return im


def torsion_plot(data, focus=None, export=False):
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 12, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(8.4, 5.45), dpi=140)
    fig.subplots_adjust(left=0.225, right=0.94, top=0.84, bottom=0.18)
    style.axes_style(ax)
    ax.grid(False)
    ax.axvspan(65, 115, color=TEAL, alpha=0.085, lw=0)
    ax.axvline(90, color=TEAL, lw=0.7, ls='--')
    ax.text(90, 4.67, 'GEOMETRY WINDOW', ha='center', fontsize=10, color=TEAL)
    for q, r in enumerate(data['mutants']):
        y = 4 - q
        active = focus is None or q == focus
        color = GOLD if r['passed_geometry_filter'] else r['color']
        ax.hlines(y, 50, 160, color=LINE, lw=0.6, zorder=0)
        if active:
            ax.axhspan(y - 0.32, y + 0.32, color=color, alpha=0.06, lw=0)
        ax.scatter(
            abs(r['chi3_deg']),
            y,
            s=125 if active else 58,
            color=color if active else '#CBD8DD',
            edgecolors='white',
            zorder=3,
        )
        ax.text(
            abs(r['chi3_deg']) + 3,
            y,
            f'{abs(r["chi3_deg"]):.1f}°',
            va='center',
            color=INK if active else MUTED,
            fontsize=13 if active else 11,
        )
    ax.set(
        yticks=list(range(4, -1, -1)),
        yticklabels=[r['label'] for r in data['mutants']],
        xlim=(50, 168),
        ylim=(-0.65, 5.05),
        xlabel='Absolute disulfide torsion |χ3| (degrees)',
        xticks=[60, 90, 115, 150],
    )
    ax.tick_params(axis='y', length=0, labelsize=13, pad=12)
    ax.spines['left'].set_visible(False)
    if export:
        fig.text(
            0.225,
            0.97,
            'Five designs · geometry after restrained relaxation',
            va='top',
            weight='bold',
            color=INK,
            fontsize=14,
        )
    fig.text(0.225, 0.035, 'Screen: 65–115° · other geometric criteria assessed separately', fontsize=10, color=MUTED)
    return finish(fig, 'multimutant_torsion', export)


def matrix_plot(data, export=False):
    fig, ax = plt.subplots(figsize=(10.2, 5.15), dpi=140)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.83, bottom=0.10)
    ax.set(xlim=(-0.5, 3.5), ylim=(-0.65, 4.7))
    ax.axis('off')
    heads = ['S–S distance', 'C–S–S angles', '|χ3| torsion', 'Cα shift']
    units = ['1.95–2.15 Å', 'both 90–120°', '65–115°', '< 0.70 Å']
    for k in range(4):
        ax.text(k, 4.62, heads[k], ha='center', color=INK, fontsize=12, weight='bold')
        ax.text(k, 4.25, units[k], ha='center', color=MUTED, fontsize=10)
    for q, r in enumerate(data['mutants']):
        y = 3.45 - q * 0.9
        ax.text(
            -0.57,
            y,
            r['label'],
            ha='right',
            va='center',
            fontsize=13,
            color=INK,
            weight='bold' if r['passed_geometry_filter'] else 'normal',
        )
        vals = [
            f'{r["SS_A"]:.3f}',
            f'{r["CSS_i_deg"]:.1f} / {r["CSS_j_deg"]:.1f}',
            f'{abs(r["chi3_deg"]):.1f}',
            f'{r["ca_displacement_rms_A"]:.3f}',
        ]
        for k, passed in enumerate(r['geometry_checks']):
            color = TEAL if passed else '#AD7864'
            ax.add_patch(
                plt.Rectangle((k - 0.46, y - 0.34), 0.92, 0.68, color=color, alpha=0.10 if passed else 0.17, lw=0)
            )
            ax.text(k, y + 0.055, vals[k], ha='center', va='center', color=color, fontsize=13, weight='bold')
            ax.text(k, y - 0.19, 'PASS' if passed else 'OUTSIDE', ha='center', va='center', color=color, fontsize=7.5)
        if r['passed_geometry_filter']:
            ax.add_patch(plt.Rectangle((-0.49, y - 0.38), 3.98, 0.76, fill=False, edgecolor=GOLD, lw=1.4))
    if export:
        fig.text(
            0.22,
            0.97,
            'Structural selection · all four criteria must pass',
            weight='bold',
            color=INK,
            fontsize=14,
            va='top',
        )
    fig.text(
        0.22,
        0.025,
        'Cα shift: RMS displacement from crystal coordinates, without superposition',
        fontsize=10,
        color=MUTED,
    )
    return finish(fig, 'multimutant_geometry', export)


def plots():
    data = load()
    torsion_plot(data, export=True)
    matrix_plot(data, True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for ax in axes:
        style.axes_style(ax)
    for r in data['mutants']:
        rows = [x for x in data['refinement'] if x['mutant'] == r['label']]
        for ax, col in zip(axes, ['energy_change_kcal_mol', 'chi3_deg', 'SS_A']):
            values = [float(x[col]) for x in rows]
            if col == 'chi3_deg':
                values = np.abs(values)
            ax.plot(range(31), values, label=r['label'], color=r['color'], lw=2 if r['passed_geometry_filter'] else 1.3)
    axes[0].set_ylabel('Energy change within variant (kcal/mol)')
    axes[1].set_ylabel('Absolute disulfide torsion |χ3| (°)')
    axes[2].set_ylabel('S–S distance (Å)')
    axes[1].axhspan(65, 115, color=TEAL, alpha=0.08, lw=0)
    axes[2].axhspan(1.95, 2.15, color=TEAL, alpha=0.08, lw=0)
    for ax in axes:
        ax.set_xlabel('Saved minimization batch')
        ax.set_xlim(0, 30)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Five independent restrained relaxations · no molecular dynamics', color=INK, fontsize=14)
    finish(fig, 'multimutant_refinement', True)
    fig = plt.figure(figsize=(12.5, 8), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1])
    for ax in [a, b, c]:
        style.axes_style(ax)
    a.plot(range(1, 57), data['reference']['mobility'], color=TEAL, lw=2, label='Original')
    for q, r in enumerate(data['mutants']):
        a.plot(
            range(1, 57),
            r['network']['mobility'],
            color=r['color'],
            lw=1.8 if r['passed_geometry_filter'] else 0.95,
            ls='--',
            label=r['label'],
            alpha=1 if q == 4 else 0.65,
        )
        y = 4 - q
        passed = r['passed_geometry_filter']
        s = [float(x['local_rms_reduction_percent']) for x in data['sensitivity'] if x['mutant'] == r['label']]
        b.plot([min(s), max(s)], [y, y], color=r['color'], lw=2)
        b.scatter(r['local_rms_reduction_percent'], y, s=65, color=r['color'], marker='o' if passed else 'x', zorder=3)
        c.barh(
            y,
            r['global_rms_reduction_percent'],
            height=0.48,
            color=r['color'],
            alpha=1 if passed else 0.5,
            hatch=None if passed else '//',
        )
    a.set(xlabel='Residue', ylabel='Relative RMS mobility', xlim=(1, 56))
    a.legend(frameon=False, ncol=3, fontsize=10)
    for ax in [b, c]:
        ax.set_yticks(range(4, -1, -1), [r['label'] for r in data['mutants']])
        ax.grid(axis='y', visible=False)
    b.set(xlabel='Conditional local RMS reduction (%)', xlim=(0, 80))
    c.set(xlabel='Conditional whole-domain RMS reduction (%)', xlim=(0, 2))
    b.set_title('Ranges: model sensitivity, not confidence intervals', fontsize=10, color=MUTED)
    c.set_title('Hatched / ×: fails structural geometry filter', fontsize=10, color=MUTED)
    fig.suptitle(
        'Harmonic predictions · adding a spring reduces covariance by construction\n7.5 Å / 10× central model; ranges span 7–8 Å / 5–20×. No folding ΔΔG or Tm.',
        fontsize=13,
        color=INK,
    )
    finish(fig, 'multimutant_rigidity', True)
    print('Four comparison figures exported in PNG, PDF and SVG.', flush=True)


def validate():
    import pymol2
    from openmm import Context, Platform, VerletIntegrator, XmlSerializer, unit
    from openmm.app import PDBFile

    data = load()
    checks = {}
    expected_sequence = data['reference_sequence']
    for variant in ['original'] + [r['folder'] for r in data['mutants']]:
        folder = SOURCE / variant
        atoms = json.loads((folder / 'atoms.json').read_text())
        d = np.load(folder / 'relaxation.npz')
        pdb = PDBFile(str(folder / 'model.pdb'))
        assert [(a.name, int(a.residue.id)) for a in pdb.topology.atoms()] == [(a['name'], a['resid']) for a in atoms]
        sequence = ''.join(ONE[a['resname']] for a in atoms if a['name'] == 'CA')
        changed = [i + 1 for i, (a, b) in enumerate(zip(expected_sequence, sequence)) if a != b]
        assert len(sequence) == 56 and d['xyz'].shape == (31, len(atoms), 3)
        bonds = [
            sorted([int(a.residue.id), int(b.residue.id)]) for a, b in pdb.topology.bonds() if a.name == b.name == 'SG'
        ]
        if variant == 'original':
            assert not changed and not bonds
        else:
            row = next(r for r in data['mutants'] if r['folder'] == variant)
            assert changed == row['pair'] and bonds == [row['pair']]
            assert all(sequence[i - 1] == 'C' for i in row['pair'])
            assert np.max(np.abs(row['states']['xyz'] - d['xyz'])) == 0
            g = geometry(d['xyz'][-1], atoms, row['pair'], data['reference']['coords'])
            for key, val in g.items():
                assert abs(row[key] - val) < 1e-9
            assert filters(g) == row['geometry_checks']
            # PDB is rounded; it must represent the same final structure.
        pdb_xyz = np.array(pdb.positions.value_in_unit(unit.angstrom))
        assert np.max(np.abs(pdb_xyz - d['xyz'][-1])) <= 0.00051
        ctx = Context(
            XmlSerializer.deserialize((folder / 'system.xml').read_text()),
            VerletIntegrator(0.001),
            Platform.getPlatformByName('CPU'),
            {'Threads': '2'},
        )
        errors = []
        for idx in [0, 15, 30]:
            ctx.setPositions(d['xyz'][idx] * unit.angstrom)
            energy = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            errors.append(abs(energy - d['energy_kcal_mol'][idx]))
        assert max(errors) < 0.002
        del ctx
        with pymol2.PyMOL() as p:
            p.cmd.set('retain_order', 1)
            p.cmd.load(str(folder / 'model.pdb'), 'molecule')
            assert [(a.name, int(a.resi)) for a in p.cmd.get_model('molecule').atom] == [
                (a['name'], a['resid']) for a in atoms
            ]
            p.cmd.load_coords(d['xyz'][-1], 'molecule')
            assert np.max(np.abs(p.cmd.get_coords('molecule') - d['xyz'][-1])) < 1e-5
        checks[variant] = dict(
            atoms=len(atoms),
            residues=56,
            changed_positions=changed,
            disulfides=bonds,
            energy_max_error_kcal_mol=max(errors),
            pdb_coordinates_verified=True,
            pymol_atom_order_verified=True,
        )
    assert data['accepted'] == ['K10C/E56C']
    dump(
        WORK / 'validation.json',
        dict(
            passed=True,
            variants=checks,
            accepted=data['accepted'],
            all_saved_geometries_recomputed=True,
            conditional_network_covariances_verified_by_independent_rank_one_identity=True,
            scope=data['scope'],
        ),
    )
    print(
        'All six structures, five mutation pairs, disulfides, coordinates and force-field energies verified.',
        flush=True,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['all', 'build', 'plots', 'validate'])
    a = p.parse_args()
    if a.stage in ['all', 'build']:
        build()
    if a.stage in ['all', 'plots']:
        plots()
    if a.stage in ['all', 'validate']:
        validate()
