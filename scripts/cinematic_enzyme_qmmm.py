#!/usr/bin/env python3
"""PDB-derived, fixed-environment catalytic proton transfer in trypsin.

MOPAC reads real Amber electrostatic potentials through its native mol.in
interface, in kcal/mol/e (see OpenMOPAC src/SCF/hcore.F90). No cartoon energies.
The fixed-environment approximation is explicit in every exported summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from leonardo.qmmm_mopac_runtime import compute_phi_on_qm_atoms, parse_mopac_aux_text  # noqa: E402

OUT = ROOT / 'artifacts/cinematic_real'
DATA = OUT / 'data'
WORK = OUT / os.environ.get('LEONARDO_QMMM_OUTPUT', 'qmmm')
MOPAC = '/opt/anaconda3/envs/protliginteract-full/bin/mopac'
HARTREE_KCAL = 627.5094740631
os.environ['OMP_NUM_THREADS'] = '1'


class Enzyme:
    def __init__(self):
        WORK.mkdir(exist_ok=True)
        full = np.load(DATA / 'trypsin_amber.npz')
        self.xyz = full['xyz']
        self.charges = full['charges']
        self.meta = json.loads((DATA / 'trypsin_atoms.json').read_text())
        backbone = {'N', 'H', 'CA', 'HA', 'C', 'O'}
        self.qidx = [a['index'] for a in self.meta if a['resid'] in {'57', '102', '195'} and a['name'] not in backbone]
        self.hglobal = next(a['index'] for a in self.meta if a['resid'] == '195' and a['name'] == 'HG')
        # Place transferring proton last, after link atoms, for native TS optimization.
        self.qidx.remove(self.hglobal)
        coords = [self.xyz[i] for i in self.qidx]
        self.qmeta = [self.meta[i] for i in self.qidx]
        self.links = []
        self.excluded = []
        interface_receivers = []
        for resid in ['57', '102', '195']:
            ca = next(a['index'] for a in self.meta if a['resid'] == resid and a['name'] == 'CA')
            cb = next(a['index'] for a in self.meta if a['resid'] == resid and a['name'] == 'CB')
            link = self.xyz[cb] + 1.09 * (self.xyz[ca] - self.xyz[cb]) / np.linalg.norm(self.xyz[ca] - self.xyz[cb])
            coords.append(link)
            self.qmeta.append(
                dict(
                    index=-1,
                    name='HL',
                    resname=self.meta[cb]['resname'],
                    resid=resid,
                    chain='A',
                    element='H',
                    atomic_number=1,
                )
            )
            self.links.append(len(coords) - 1)
            # Exclude the charge at the cut and its bonded backbone hydrogens;
            # redistribute their sum to the other backbone atoms of that residue.
            boundary = [a['index'] for a in self.meta if a['resid'] == resid and a['name'] in {'CA', 'HA'}]
            receivers = [a['index'] for a in self.meta if a['resid'] == resid and a['name'] in {'N', 'H', 'C', 'O'}]
            correction = self.charges[boundary].sum() / len(receivers)
            self.charges[receivers] += correction
            interface_receivers.extend(receivers)
            self.excluded.extend(boundary)
        coords.append(self.xyz[self.hglobal])
        self.qmeta.append(self.meta[self.hglobal])
        self.h = len(coords) - 1
        self.qcoords = np.array(coords)
        self.mmidx = [i for i in range(len(self.xyz)) if i not in self.qidx + [self.hglobal] + self.excluded]
        self.mmxyz = self.xyz[self.mmidx]
        self.mmcharges = self.charges[self.mmidx]
        # Replacing partial-charge side chains by an integer-charge capped QM
        # region must conserve the total charge of the enzyme (+6 here).
        target_mm_charge = float(full['charges'].sum()) - (-1)
        self.interface_charge_correction = target_mm_charge - float(self.mmcharges.sum())
        for global_index in interface_receivers:
            self.mmcharges[self.mmidx.index(global_index)] += self.interface_charge_correction / len(
                interface_receivers
            )
        self.o = next(i for i, a in enumerate(self.qmeta) if a['resid'] == '195' and a['name'] == 'OG')
        self.n = next(i for i, a in enumerate(self.qmeta) if a['resid'] == '57' and a['name'] == 'NE2')
        self.origin = self.qcoords[self.o].copy()
        self.direction = self.qcoords[self.n] - self.origin
        self.distance = np.linalg.norm(self.direction)
        self.direction /= self.distance
        self.counter = 0
        self.write_pdb(WORK / 'qm_region.pdb', self.qcoords)
        np.savez_compressed(
            WORK / 'embedding.npz',
            qm_xyz=self.qcoords,
            mm_xyz=self.mmxyz,
            mm_charges=self.mmcharges,
            qm_global_indices=self.qidx + [self.hglobal],
            excluded_boundary_indices=self.excluded,
        )
        (WORK / 'qm_atoms.json').write_text(json.dumps(self.qmeta, indent=2))

    def write_pdb(self, path, xyz):
        lines = []
        for i, (a, p) in enumerate(zip(self.qmeta, xyz), 1):
            lines.append(
                f"HETATM{i:5d} {a['name']:>4s} {a['resname']:>3s} A{int(a['resid']):4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00          {a['element']:>2s}"
            )
        path.write_text('\n'.join(lines) + '\nEND\n')

    def execute(self, xyz, tag, keywords='1SCF GRAD', product=None, embedding=True):
        work = WORK / tag
        work.mkdir(parents=True, exist_ok=True)
        phi = (
            compute_phi_on_qm_atoms(xyz, self.mmxyz, self.mmcharges) * HARTREE_KCAL if embedding else np.zeros(len(xyz))
        )
        # Native MOPAC format: title; atom and link counts; symbol,x,y,z,V.
        lines = ['Amber14 electrostatic potential / kcal mol-1 e-1', f'{len(xyz)-3} 3']
        lines += [
            f"{a['element']} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f} {v:.12f}" for a, p, v in zip(self.qmeta, xyz, phi)
        ]
        (work / 'mol.in').write_text('\n'.join(lines) + '\n')
        header = f'PM7 {keywords} QMMM AUX(9) XYZ CHARGE=-1 SINGLET PRECISE GEO-OK'
        lines = [header, 'Trypsin 3PTB catalytic triad; frozen environment', '']

        def geometry(arr):
            return [
                f"{a['element']} {p[0]:.10f} {int(i==self.h)} {p[1]:.10f} {int(i==self.h)} {p[2]:.10f} {int(i==self.h)}"
                for i, (a, p) in enumerate(zip(self.qmeta, arr))
            ]

        lines += geometry(xyz)
        if product is not None:
            lines += [''] + geometry(product)
        (work / 'step.mop').write_text('\n'.join(lines) + '\n')
        cp = subprocess.run([MOPAC, 'step.mop'], cwd=work, capture_output=True, text=True, timeout=120)
        (work / 'console.log').write_text(cp.stdout + '\n' + cp.stderr)
        out = (work / 'step.out').read_text(errors='replace') if (work / 'step.out').exists() else ''
        # Read explicitly labeled kcal/mol heat of formation, never treat it as eV.
        matches = re.findall(r'FINAL HEAT OF FORMATION\s*=\s*([-+0-9.DEd]+)\s*KCAL', out)
        if cp.returncode or not matches:
            raise RuntimeError(f'MOPAC failed in {tag}: ' + out[-1500:] + cp.stderr[-500:])
        energy = float(matches[-1].replace('D', 'E'))
        aux = (
            parse_mopac_aux_text((work / 'step.aux').read_text(errors='replace'))
            if (work / 'step.aux').exists()
            else {}
        )
        return energy, aux

    def energy(self, position, tag=None):
        self.counter += 1
        xyz = self.qcoords.copy()
        xyz[self.h] = position
        energy, _ = self.execute(xyz, tag or f'refinement/eval_{self.counter:04d}')
        return energy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['scan', 'saddle'])
    a = p.parse_args()
    m = Enzyme()
    start = time.perf_counter()
    if a.stage == 'saddle':
        r = m.qcoords.copy()
        r[m.h] = m.origin + 0.99 * m.direction
        prod = r.copy()
        prod[m.h] = m.origin + (m.distance - 1.02) * m.direction
        e, aux = m.execute(r, 'native_saddle', 'SADDLE CYCLES=200', product=prod)
        print('Native SADDLE', e, aux.get('optimized_coords_ang'), flush=True)
        return
    ds = np.linspace(0.92, m.distance - 0.94, 33)
    coords = []
    energies = []
    for i, d in enumerate(ds):
        xyz = m.qcoords.copy()
        xyz[m.h] = m.origin + d * m.direction
        e = m.energy(xyz[m.h], f'path/image_{i:03d}')
        energies.append(e)
        coords.append(xyz)
        print(i, round(float(d), 4), e, flush=True)
    energies = np.array(energies)
    peak = int(energies.argmax())
    validation = dict(
        full_enzyme_TS_validated=False,
        scope='One proton on a fixed donor-acceptor axis; all other nuclei fixed. Not a full enzymatic activation free energy or an IRC.',
    )
    if 0 < peak < len(ds) - 1:
        # Quadratic synchronous-transit maximum from R, highest image, P.
        # Refine the constrained stationary point with independently evaluated PM7 energies.
        result = minimize_scalar(
            lambda d: -m.energy(m.origin + d * m.direction),
            bounds=(ds[peak - 1], ds[peak + 1]),
            method='bounded',
            options={'xatol': 1e-5},
        )
        dts = float(result.x)
        ets = -float(result.fun)
        delta = 0.005
        em = m.energy(m.origin + (dts - delta) * m.direction, 'validation/minus')
        ep = m.energy(m.origin + (dts + delta) * m.direction, 'validation/plus')
        curvature = (ep - 2 * ets + em) / delta**2
        gradient = (ep - em) / (2 * delta)
        validation.update(
            constrained_stationary_point_A=dts,
            energy_kcal_mol=ets,
            curvature_kcal_mol_A2=curvature,
            gradient_kcal_mol_A=gradient,
            constrained_maximum=bool(curvature < 0 and abs(gradient) < 0.2),
        )
    # Embedding control proves that the protein field enters the Hamiltonian.
    e0, _ = m.execute(coords[0], 'validation/zero_field', embedding=False)
    validation['embedding_energy_shift_kcal_mol'] = float(energies[0] - e0)
    with (WORK / 'profile.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(
            [
                'image',
                'O_H_A',
                'H_N_A',
                'coordinate_OH_minus_HN_A',
                'heat_of_formation_kcal_mol',
                'relative_to_first_kcal_mol',
            ]
        )
        for i, d in enumerate(ds):
            w.writerow([i, d, m.distance - d, 2 * d - m.distance, energies[i], energies[i] - energies[0]])
    np.savez_compressed(
        WORK / 'path.npz', coords=np.array(coords), distance=ds, energies=energies, relative=energies - energies[0]
    )
    for i, c in enumerate(coords):
        m.write_pdb(WORK / f'image_{i:03d}.pdb', c)
    summary = dict(
        system='Trypsin catalytic triad, PDB 3PTB',
        reaction='Ser195 O-H to His57 NE2 proton transfer; Asp102 included in QM',
        engine='MOPAC PM7 / Amber14 electrostatic embedding',
        qm_atoms=len(m.qcoords),
        qm_charge=-1,
        link_hydrogens=3,
        mm_point_charges=len(m.mmidx),
        mm_total_charge=float(m.mmcharges.sum()),
        enzyme_total_charge=float(m.mmcharges.sum() - 1),
        interface_charge_correction_e=m.interface_charge_correction,
        boundary='CA and HA charges removed at each CB-CA cut and redistributed to remaining backbone atoms of the same residue; integer QM charge replacement compensated on these interface backbone atoms to conserve enzyme total charge',
        donor_acceptor_A=float(m.distance),
        path_images=len(ds),
        runtime_seconds=time.perf_counter() - start,
        validation=validation,
        scope='Constrained synchronous-transit scan, protein fixed, not a relaxed enzyme TS. Inhibitor, waters and calcium absent in the QM/MM model.',
        energy='MOPAC heat of formation including external MM electrostatic field; fixed-environment relative energy, kcal/mol; no MM energy is double-counted.',
    )
    (WORK / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
