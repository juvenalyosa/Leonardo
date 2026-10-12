#!/usr/bin/env python3
"""Auditable fixed-protein PM7/Amber QM/MM alkylation in DhlA (2DHC).

The energy is MOPAC's electrostatically embedded heat of formation plus
QM--MM Lennard-Jones interactions. Potentials and their coordinate derivatives
are refreshed on EVERY call. Fixed MM-only terms cancel in relative energies.
No energy profile is imposed and no movie coordinates are interpolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENMM_CPU_THREADS'] = '2'
import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cinematic_v2'
DATA = OUT / 'data'
WORK = OUT / 'qmmm'
MOPAC = '/opt/anaconda3/envs/protliginteract-full/bin/mopac'
COULOMB = 332.063713299


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def prepare():
    from openmm import NonbondedForce, XmlSerializer, unit
    from openmm.app import ForceField, Modeller, NoCutoff, PDBFile
    from rdkit import Chem

    DATA.mkdir(parents=True, exist_ok=True)
    source = Path('/private/tmp/leonardo_2DHC.pdb')
    shutil.copyfile(source, DATA / '2DHC.pdb')
    lines = source.read_text().splitlines()
    (DATA / 'dhla_heavy.pdb').write_text(
        '\n'.join(line for line in lines if line.startswith(('ATOM  ', 'TER', 'SSBOND'))) + '\nEND\n'
    )
    pdb = PDBFile(str(DATA / 'dhla_heavy.pdb'))
    ff = ForceField('amber14-all.xml')
    mod = Modeller(pdb.topology, pdb.positions)
    random.seed(20260915)
    variants = ['HIE' if r.name == 'HIS' and r.id == '289' else None for r in mod.topology.residues()]
    mod.addHydrogens(ff, pH=7.0, variants=variants)
    system = ff.createSystem(mod.topology, nonbondedMethod=NoCutoff, constraints=None)
    xyz = np.asarray(mod.positions.value_in_unit(unit.angstrom))
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    params = np.array(
        [
            [
                q.value_in_unit(unit.elementary_charge),
                s.value_in_unit(unit.angstrom),
                e.value_in_unit(unit.kilocalorie_per_mole),
            ]
            for q, s, e in (nb.getParticleParameters(i) for i in range(len(xyz)))
        ]
    )
    exc = []
    for i in range(nb.getNumExceptions()):
        a, b, q, s, e = nb.getExceptionParameters(i)
        exc.append([a, b, s.value_in_unit(unit.angstrom), e.value_in_unit(unit.kilocalorie_per_mole)])
    with (DATA / 'dhla_amber.pdb').open('w') as f:
        PDBFile.writeFile(mod.topology, mod.positions, f, keepIds=True)
    (DATA / 'dhla_system.xml').write_text(XmlSerializer.serialize(system))
    meta = [
        dict(
            index=a.index,
            name=a.name,
            resname=a.residue.name,
            resid=a.residue.id,
            chain=a.residue.chain.id,
            element=a.element.symbol,
            mass=a.element.mass.value_in_unit(unit.dalton),
        )
        for a in mod.topology.atoms()
    ]
    dump(DATA / 'dhla_atoms.json', meta)
    np.savez_compressed(DATA / 'dhla_amber.npz', xyz=xyz, params=params, exceptions=np.array(exc))
    # Explicit graph and deposited atom order: CL1-C1-C2-CL2.
    mol = Chem.MolFromSmiles('ClCCCl')
    conf = Chem.Conformer(4)
    ligand = [
        next(
            line for line in lines if line.startswith('HETATM') and line[17:20] == 'DCE' and line[12:16].strip() == name
        )
        for name in ['CL1', 'C1', 'C2', 'CL2']
    ]
    for i, line in enumerate(ligand):
        conf.SetAtomPosition(i, [float(line[30:38]), float(line[38:46]), float(line[46:54])])
    mol.AddConformer(conf)
    mol = Chem.AddHs(mol, addCoords=True)
    Chem.SanitizeMol(mol)
    with Chem.SDWriter(str(DATA / 'dce_crystal_h.sdf')) as writer:
        writer.write(mol)
    np.save(DATA / 'dce_xyz.npy', mol.GetConformer().GetPositions())
    dump(
        DATA / 'preparation.json',
        dict(
            pdb='2DHC',
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            citation_doi='10.1038/363693a0',
            resolution_A=2.3,
            atom_count=len(xyz),
            total_charge=float(params[:, 0].sum()),
            histidine_289='HIE',
            pH=7,
            ligand='1,2-dichloroethane',
            ligand_CL2_deposited_occupancy=0.3,
            treatment='Deposited substrate pose used as starting guess and then relaxed; no occupancy claim for optimized model. Crystal waters and separate chloride ions omitted. Protein heavy atoms fixed.',
        ),
    )
    print('Prepared DhlA', len(xyz), 'protein atoms', flush=True)


def aux_array(text, name):
    match = re.search(re.escape(name) + r'[^\n=]*\[(\d+)\]=\s*\n', text)
    if not match:
        raise ValueError('Missing ' + name)
    return np.array([float(v.replace('D', 'E')) for v in text[match.end() :].split()[: int(match.group(1))]])


class Model:
    def __init__(self):
        WORK.mkdir(parents=True, exist_ok=True)
        d = np.load(DATA / 'dhla_amber.npz')
        meta = json.loads((DATA / 'dhla_atoms.json').read_text())
        self.fullxyz = d['xyz']
        pars = d['params'].copy()
        self.qidx = [
            a['index'] for a in meta if a['resid'] == '124' and a['name'] not in {'N', 'H', 'CA', 'HA', 'C', 'O'}
        ]
        self.meta = [meta[i] for i in self.qidx]
        xyz = [self.fullxyz[i] for i in self.qidx]
        ca = next(a['index'] for a in meta if a['resid'] == '124' and a['name'] == 'CA')
        cb = next(a['index'] for a in meta if a['resid'] == '124' and a['name'] == 'CB')
        v = self.fullxyz[ca] - self.fullxyz[cb]
        xyz.append(self.fullxyz[cb] + 1.09 * v / np.linalg.norm(v))
        self.link = len(xyz) - 1
        self.meta.append(dict(index=-1, name='HL', resname='ASP', resid='124', chain='A', element='H', mass=1.008))
        ligandxyz = np.load(DATA / 'dce_xyz.npy')
        self.substrate_start = len(xyz)
        for name, element, mass, p in zip(
            ['CL1', 'C1', 'C2', 'CL2', 'H11', 'H12', 'H21', 'H22'],
            ['Cl', 'C', 'C', 'Cl', 'H', 'H', 'H', 'H'],
            [35.45, 12.011, 12.011, 35.45, 1.008, 1.008, 1.008, 1.008],
            ligandxyz,
        ):
            self.meta.append(
                dict(index=-1, name=name, resname='DCE', resid='600', chain='A', element=element, mass=mass)
            )
            xyz.append(p)
        self.xyz = np.array(xyz)
        self.mobile = np.array(
            [i for i, a in enumerate(self.meta) if a['name'] in {'CG', 'OD1', 'OD2'} or a['resname'] == 'DCE']
        )
        self.o = next(i for i, a in enumerate(self.meta) if a['name'] == 'OD2')
        self.c = next(i for i, a in enumerate(self.meta) if a['name'] == 'C1')
        self.cl = next(i for i, a in enumerate(self.meta) if a['name'] == 'CL1')
        # Charge-shift boundary at CB--CA; keep enzyme charge invariant.
        boundary = [a['index'] for a in meta if a['resid'] == '124' and a['name'] in {'CA', 'HA'}]
        receivers = [a['index'] for a in meta if a['resid'] == '124' and a['name'] in {'N', 'H', 'C', 'O'}]
        pars[receivers, 0] += pars[boundary, 0].sum() / len(receivers)
        mmidx = [i for i in range(len(meta)) if i not in self.qidx + boundary]
        correction = d['params'][:, 0].sum() + 1 - pars[mmidx, 0].sum()
        pars[receivers, 0] += correction / len(receivers)
        self.mmidx = np.array(mmidx)
        self.mmxyz = self.fullxyz[mmidx]
        self.mmq = pars[mmidx, 0]
        # Amber14 LJ on the Asp atoms, GAFF2.11 c3/h1/cl LJ on substrate.
        # GAFF files tabulate Rmin/2 in A and epsilon in kcal/mol.
        gaff = Path(
            '/opt/anaconda3/envs/protliginteract-full/lib/python3.11/site-packages/openmmforcefields/ffxml/amber/gaff/dat/gaff-2.11.dat'
        ).read_text()
        vdw = {}
        for typ in ['c3', 'h1', 'cl']:
            matches = re.findall(r'^\s*' + typ + r'\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$', gaff, re.M)
            r, e = map(float, matches[-1])
            vdw[typ] = (r * 2 / 2 ** (1 / 6), e)
        qpars = [tuple(pars[i, 1:]) for i in self.qidx] + [(0, 0)]
        qpars += [
            vdw['cl' if a['element'] == 'Cl' else 'c3' if a['element'] == 'C' else 'h1']
            for a in self.meta[self.substrate_start :]
        ]
        qpars = np.array(qpars)
        self.sigma = 0.5 * (qpars[:, None, 0] + pars[mmidx, 1][None, :])
        self.epsilon = np.sqrt(qpars[:, None, 1] * pars[mmidx, 2][None, :])
        # Apply protein bond exclusions and 1-4 LJ factors across the boundary.
        qmap = {g: i for i, g in enumerate(self.qidx)}
        mmap = {g: i for i, g in enumerate(mmidx)}
        for a, b, s, e in d['exceptions']:
            a, b = int(a), int(b)
            if b in qmap and a in mmap:
                a, b = b, a
            if a in qmap and b in mmap:
                self.sigma[qmap[a], mmap[b]] = s
                self.epsilon[qmap[a], mmap[b]] = e
        self.count = 0
        self.folder = WORK / 'evaluations'
        self.folder.mkdir(exist_ok=True)
        existing = list(self.folder.glob('e*'))
        if existing:
            self.count = max(int(p.name[1:]) for p in existing)
        self.masses = np.repeat([self.meta[i]['mass'] for i in self.mobile], 3)
        self.write_pdb(WORK / 'qm_initial.pdb', self.xyz)
        dump(WORK / 'qm_atoms.json', self.meta)
        np.savez_compressed(
            WORK / 'model.npz',
            qm_xyz=self.xyz,
            mobile=self.mobile,
            mm_xyz=self.mmxyz,
            mm_charges=self.mmq,
            mm_indices=self.mmidx,
            qm_indices=self.qidx,
            sigma=self.sigma,
            epsilon=self.epsilon,
        )
        dump(
            WORK / 'model.json',
            dict(
                qm_atoms=len(self.xyz),
                mobile_atoms=len(self.mobile),
                qm_charge=-1,
                mm_atoms=len(mmidx),
                mm_charge=float(self.mmq.sum()),
                boundary_charge_correction_e=float(correction),
                boundary_exclusions='CA124 and HA124 omitted from explicit QM-MM electrostatic and Lennard-Jones interactions at the cut; their charges are redistributed to the remaining backbone atoms of Asp124. Frozen coordinates remain in the structural depiction.',
                qm_definition='Asp124 sidechain with one H cap plus substrate; CB, HB2, HB3 and cap fixed; carboxylate and all substrate atoms mobile',
                reaction='Asp124 OD2 attacks DCE C1; CL1 departs. First alkylation step, not the complete hydrolytic catalytic cycle.',
                model='PM7 electrostatic embedding in Amber14 charges, Amber14/GAFF2.11 QM-MM LJ; fixed protein; no solvent sampling',
                lj_gaff=vdw,
                stationarity_space='33 mobile Cartesian coordinates; frozen protein and four anchor atoms',
                energy='MOPAC embedded heat of formation + QM/MM LJ; kcal/mol. Potential energy, not activation free energy.',
            ),
        )

    def write_pdb(self, path, xyz):
        lines = []
        for i, (a, p) in enumerate(zip(self.meta, xyz), 1):
            lines.append(
                f"HETATM{i:5d} {a['name']:>4s} {a['resname']:>3s} A{int(a['resid']):4d}    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00 20.00          {a['element']:>2s}"
            )
        Path(path).write_text('\n'.join(lines) + '\nEND\n')

    def unpack(self, x):
        xyz = self.xyz.copy()
        xyz[self.mobile] = np.asarray(x).reshape(-1, 3)
        return xyz

    def evaluate(self, x, field=True):
        xyz = self.unpack(x)
        self.count += 1
        folder = self.folder / f'e{self.count:06d}'
        folder.mkdir()
        delta = xyz[:, None, :] - self.mmxyz[None, :, :]
        dist = np.linalg.norm(delta, axis=-1)
        phi = COULOMB * np.sum(self.mmq[None, :] / dist, axis=1) if field else np.zeros(len(xyz))
        lines = ['Amber14 potential, kcal mol-1 e-1', f'{len(xyz)-1} 1'] + [
            f"{a['element']} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f} {v:.12f}" for a, p, v in zip(self.meta, xyz, phi)
        ]
        (folder / 'mol.in').write_text('\n'.join(lines) + '\n')
        lines = [
            'PM7 1SCF GRAD QMMM AUX(PRECISION=8) XYZ CHARGE=-1 SINGLET PRECISE SCFCRT=1.D-10 GEO-OK',
            'DhlA 2DHC: first alkylation step',
            '',
        ] + [f"{a['element']} {p[0]:.12f} 1 {p[1]:.12f} 1 {p[2]:.12f} 1" for a, p in zip(self.meta, xyz)]
        (folder / 'step.mop').write_text('\n'.join(lines) + '\n')
        cp = subprocess.run([MOPAC, 'step.mop'], cwd=folder, capture_output=True, text=True, timeout=60)
        out = (folder / 'step.out').read_text(errors='replace') if (folder / 'step.out').exists() else ''
        if cp.returncode or 'SCF FIELD WAS ACHIEVED' not in out:
            raise RuntimeError('MOPAC SCF failure ' + str(folder) + ' ' + out[-1000:] + cp.stderr)
        aux = (folder / 'step.aux').read_text()
        hof = float(re.search(r'HEAT_OF_FORMATION:KCAL/MOL=\s*([-+0-9.DE]+)', aux).group(1).replace('D', 'E'))
        g = aux_array(aux, 'GRADIENTS:KCAL/MOL/ANGSTROM').reshape(-1, 3)
        q = aux_array(aux, 'ATOM_CHARGES')
        # hcore reads a static potential; this is the missing coordinate derivative.
        if field:
            g += -COULOMB * q[:, None] * np.sum(self.mmq[None, :, None] * delta / dist[:, :, None] ** 3, axis=1)
        sr6 = (self.sigma / dist) ** 6
        lj = float(np.sum(4 * self.epsilon * (sr6**2 - sr6)))
        g += np.sum((24 * self.epsilon * (sr6 - 2 * sr6**2) / dist**2)[:, :, None] * delta, axis=1)
        energy = hof + lj
        dump(
            folder / 'result.json',
            dict(
                energy_kcal_mol=energy,
                embedded_hof_kcal_mol=hof,
                lj_kcal_mol=lj,
                gradient_mobile=g[self.mobile].tolist(),
                scf_converged=True,
                field=field,
            ),
        )
        return energy, g[self.mobile].ravel()

    def rc(self, x):
        p = self.unpack(x)
        return np.linalg.norm(p[self.o] - p[self.c]) - np.linalg.norm(p[self.c] - p[self.cl])

    def rc_grad(self, x):
        p = self.unpack(x)
        g = np.zeros_like(p)
        v = p[self.o] - p[self.c]
        v /= np.linalg.norm(v)
        g[self.o] += v
        g[self.c] -= v
        v = p[self.c] - p[self.cl]
        v /= np.linalg.norm(v)
        g[self.c] -= v
        g[self.cl] += v
        return g[self.mobile].ravel()

    def minimum(self, x, label, rc=None, maxiter=600):
        history = []

        def fun(x):
            e, g = self.evaluate(x)
            history.append([self.count, e, float(np.linalg.norm(g)), float(self.rc(x))])
            return e, g

        if rc is None:
            r = minimize(
                fun,
                x,
                jac=True,
                method='L-BFGS-B',
                options=dict(gtol=1e-4, ftol=1e-14, maxiter=maxiter, maxls=40, maxcor=30),
            )
        else:
            r = minimize(
                fun,
                x,
                jac=True,
                method='SLSQP',
                constraints=[dict(type='eq', fun=lambda x: self.rc(x) - rc, jac=self.rc_grad)],
                options=dict(ftol=1e-10, maxiter=maxiter),
            )
        e, g = self.evaluate(r.x)
        np.savez_compressed(WORK / f'{label}.npz', x=r.x, xyz=self.unpack(r.x), energy=e, gradient=g, history=history)
        self.write_pdb(WORK / f'{label}.pdb', self.unpack(r.x))
        dump(
            WORK / f'{label}.json',
            dict(
                success=bool(r.success),
                message=str(r.message),
                iterations=int(r.nit),
                energy_kcal_mol=e,
                gradient_norm=float(np.linalg.norm(g)),
                gradient_max=float(np.max(np.abs(g))),
                reaction_coordinate_A=self.rc(r.x),
                target_rc_A=rc,
            ),
        )
        print(label, r.success, str(r.message), 'E', e, 'g', np.linalg.norm(g), 'RC', self.rc(r.x), flush=True)
        return r.x

    def hessian(self, x, step=0.001):
        cols = []
        for j in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[j] += step
            xm[j] -= step
            gp = self.evaluate(xp)[1]
            gm = self.evaluate(xm)[1]
            cols.append((gp - gm) / (2 * step))
        h = np.array(cols).T
        return (h + h.T) / 2, float(np.max(np.abs(h - h.T)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        'stage', choices=['prepare', 'check', 'reactant', 'scan', 'ts', 'qst', 'validate', 'descent', 'path']
    )
    p.add_argument('--start', default='reactant')
    args = p.parse_args()
    if args.stage == 'prepare':
        prepare()
        return
    m = Model()
    x = m.xyz[m.mobile].ravel()
    if args.stage == 'check':
        e, g = m.evaluate(x)
        rows = []
        for j in [0, 3, 7, 10, 15, 20, 25, 32]:
            xp = x.copy()
            xm = x.copy()
            xp[j] += 0.0002
            xm[j] -= 0.0002
            fd = (m.evaluate(xp)[0] - m.evaluate(xm)[0]) / 0.0004
            rows.append(dict(component=j, analytical=float(g[j]), finite_difference=fd, error=float(g[j] - fd)))
        dump(WORK / 'gradient_check.json', rows)
        print(json.dumps(rows, indent=2), flush=True)
    elif args.stage == 'reactant':
        m.minimum(x, 'reactant')
    elif args.stage == 'scan':
        x = np.load(WORK / f'{args.start}.npz')['x']
        start = m.rc(x)
        targets = np.linspace(start, -1.9, 31)
        for i, t in enumerate(targets):
            path = WORK / f'scan_{i:03d}.npz'
            if path.exists():
                x = np.load(path)['x']
                continue
            x = m.minimum(x, f'scan_{i:03d}', rc=float(t), maxiter=350)
    elif args.stage in {'ts', 'qst'}:
        prefix = 'ts' if args.stage == 'ts' else 'qst_ts'
        if args.stage == 'qst':
            # Quadratic synchronous transit through two independently optimized
            # endpoints and a relaxed-scan intermediate. This is an INITIAL
            # GUESS; every energy is separately evaluated with MOPAC.
            endpoints = [np.load(WORK / f'downhill_{side}.npz')['x'] for side in ['minus', 'plus']]
            endpoints.sort(key=m.rc, reverse=True)
            reactant, product = endpoints
            guess = np.load(WORK / 'scan_013.npz')['x']

            def quadratic(t):
                return (1 - t) * reactant + t * product + 4 * t * (1 - t) * (guess - 0.5 * (reactant + product))

            samples = []

            def negative_energy(t):
                energy = m.evaluate(quadratic(t))[0]
                samples.append([float(t), energy, m.count])
                return -energy

            result = minimize_scalar(negative_energy, bounds=(0.25, 0.75), method='bounded', options={'xatol': 1e-6})
            x = quadratic(result.x)
            np.savez_compressed(
                WORK / 'qst_seed.npz', x=x, reactant=reactant, product=product, intermediate=guess, samples=samples
            )
            dump(
                WORK / 'qst_initialization.json',
                dict(
                    method='R / intermediate guess / P quadratic synchronous transit, followed by full active-space eigenvector following',
                    parameter=float(result.x),
                    seed_energy_kcal_mol=-float(result.fun),
                    samples=samples,
                    intermediate_source='scan_013.npz',
                    engine='External QST driver; MOPAC PM7 supplies the embedded energy and gradients at every evaluation',
                ),
            )
            print('QST seed', result.x, -result.fun, flush=True)
        else:
            files = sorted(WORK.glob('scan_*.npz'))
            arr = [np.load(f) for f in files]
            idx = int(np.argmax([float(d['energy']) for d in arr]))
            x = arr[idx]['x']
            if args.start != 'reactant':
                x = np.load(WORK / f'{args.start}.npz')['x']
                idx = next(i for i, f in enumerate(files) if f.stem == args.start)
            print('Starting TS from', files[idx].name, flush=True)
        # Eigenvector-following Newton refinement in the 33-coordinate active space.
        for k in range(30):
            e, g = m.evaluate(x)
            h, asym = m.hessian(x)
            w, v = np.linalg.eigh(h)
            print('TS iteration', k, 'E', e, 'g', np.linalg.norm(g), 'eig', w[:3], flush=True)
            np.savez_compressed(WORK / f'{prefix}_progress.npz', x=x, energy=e, gradient=g, hessian=h)
            if np.max(np.abs(g)) < 0.001:
                break
            # Enforce exactly one uphill direction while retaining eigenvectors.
            eig = np.maximum(np.abs(w), 0.5)
            eig[0] *= -1
            dx = -(v @ ((v.T @ g) / eig))
            maxstep = np.max(np.linalg.norm(dx.reshape(-1, 3), axis=1))
            dx *= min(1, 0.15 / maxstep)
            x += dx
        e, g = m.evaluate(x)
        np.savez_compressed(WORK / f'{prefix}.npz', x=x, xyz=m.unpack(x), energy=e, gradient=g, hessian=h)
        m.write_pdb(WORK / f'{prefix}.pdb', m.unpack(x))
        dump(
            WORK / f'{prefix}.json',
            dict(
                energy_kcal_mol=e,
                gradient_norm=float(np.linalg.norm(g)),
                max_gradient=float(np.max(np.abs(g))),
                rc_A=m.rc(x),
                hessian_eigenvalues=w.tolist(),
                hessian_asymmetry=asym,
                converged=bool(np.max(np.abs(g)) < 0.001 and np.sum(w < 0) == 1),
            ),
        )
    elif args.stage == 'descent':
        d = np.load(WORK / 'ts.npz')
        sqrtm = np.sqrt(m.masses)
        w, v = np.linalg.eigh(d['hessian'] / np.sqrt(m.masses[:, None] * m.masses[None, :]))
        xts = d['x']
        for sign, label in [(-1, 'downhill_minus'), (1, 'downhill_plus')]:
            x = xts + sign * 0.06 * v[:, 0] / sqrtm
            e, g = m.evaluate(x)
            saved = [(xts.copy(), float(d['energy'])), (x.copy(), e)]
            # Explicit mass-weighted steepest descent with midpoint direction
            # and an energy-decreasing adaptive step. This is a numerical
            # downhill connection, not a harmonic animation or interpolated path.
            for k in range(2000):
                gz = g / sqrtm
                norm = np.linalg.norm(gz)
                if norm < 0.025:
                    break
                ds = 0.055
                accepted = False
                for trial in range(15):
                    mid = x - 0.5 * ds * gz / (norm * sqrtm)
                    _, gm = m.evaluate(mid)
                    gzm = gm / sqrtm
                    xn = x - ds * gzm / (np.linalg.norm(gzm) * sqrtm)
                    en, gn = m.evaluate(xn)
                    if en < e - 1e-10:
                        accepted = True
                        break
                    ds *= 0.5
                if not accepted:
                    break
                x, e, g = xn, en, gn
                saved.append((x.copy(), e))
                if k % 50 == 0:
                    print(label, k, 'E', e, 'g_mass', np.linalg.norm(g / sqrtm), 'rc', m.rc(x), flush=True)
                if k % 100 == 0:
                    np.savez_compressed(
                        WORK / f'{label}_trajectory.npz',
                        x=np.array([r[0] for r in saved]),
                        energies=[r[1] for r in saved],
                    )
            np.savez_compressed(
                WORK / f'{label}_trajectory.npz', x=np.array([r[0] for r in saved]), energies=[r[1] for r in saved]
            )
            m.minimum(x, label)
    elif args.stage == 'path':
        vm = json.loads((WORK / 'stationary_validation.json').read_text())
        labels = ['downhill_minus', 'downhill_plus']
        # Assign chemistry by O-C minus C-Cl, not energy ordering.
        labels.sort(key=lambda label: vm[label]['rc_A'], reverse=True)
        r, pdt = labels
        dr = np.load(WORK / f'{r}_trajectory.npz')
        dp = np.load(WORK / f'{pdt}_trajectory.npz')
        xr = np.load(WORK / f'{r}.npz')
        xp = np.load(WORK / f'{pdt}.npz')

        # Equal arc-length nearest SAVED samples, never interpolated coordinates.
        def select(traj, minimum, label, n=100):
            positions = list(traj['x'])
            values = list(traj['energies'])
            relaxation = []
            # Retain real, successively lower-energy evaluations from the final
            # L-BFGS relaxation. This prevents a visual jump straight from the
            # descent endpoint to the minimized geometry; no coordinates are
            # interpolated and these are identified as minimization evaluations.
            for evaluation, energy, _, _ in minimum['history']:
                if energy >= values[-1] - 1e-10:
                    continue
                folder = m.folder / f'e{int(evaluation):06d}'
                if (folder / 'source.zip').exists():
                    with zipfile.ZipFile(folder / 'source.zip') as archive:
                        raw = archive.read('step.mop').decode()
                else:
                    raw = (folder / 'step.mop').read_text()
                allxyz = np.array(
                    [[float(line.split()[j]) for j in [1, 3, 5]] for line in raw.splitlines()[3:] if line.strip()]
                )
                positions.append(allxyz[m.mobile].ravel())
                values.append(float(energy))
                relaxation.append(int(evaluation))
            positions.append(minimum['x'])
            values.append(float(minimum['energy']))
            arr = np.array(positions)
            energies = np.array(values)
            dump(
                WORK / f'{label}_relaxation_trace.json',
                dict(
                    method='Record-lowering evaluations from endpoint L-BFGS minimization; all coordinates read back from original MOPAC inputs',
                    evaluations=relaxation,
                ),
            )
            arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(arr, axis=0) * np.sqrt(m.masses), axis=1))]
            idx = np.unique([int(np.argmin(np.abs(arclength - t))) for t in np.linspace(0, arclength[-1], n)])
            return arr[idx], energies[idx]

        ar, er = select(dr, xr, r)
        ap, ep = select(dp, xp, pdt)
        xs = np.concatenate([ar[::-1], ap[1:]])
        energies = np.r_[er[::-1], ep[1:]]
        tsidx = len(ar) - 1
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xs, axis=0) * np.sqrt(m.masses), axis=1))]
        coords = np.array([m.unpack(x) for x in xs])
        doc = np.linalg.norm(coords[:, m.o] - coords[:, m.c], axis=1)
        dccl = np.linalg.norm(coords[:, m.c] - coords[:, m.cl], axis=1)
        np.savez_compressed(
            WORK / 'path.npz',
            coords=coords,
            imaginary_frequency=vm['ts']['signed_frequencies_cm1'][0],
            energies=energies,
            relative=energies - energies[0],
            s=arc / arc[-1],
            arc=arc,
            ts_index=tsidx,
            d_oc=doc,
            d_ccl=dccl,
        )
        for i, xyz in enumerate(coords):
            m.write_pdb(WORK / f'path_{i:03d}.pdb', xyz)
        import csv

        with (WORK / 'profile.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    'image',
                    'normalized_mass_weighted_arc',
                    'arc_sqrtamu_A',
                    'energy_kcal_mol',
                    'relative_kcal_mol',
                    'O_C_A',
                    'C_Cl_A',
                    'state',
                ]
            )
            for i in range(len(xs)):
                writer.writerow(
                    [
                        i,
                        arc[i] / arc[-1],
                        arc[i],
                        energies[i],
                        energies[i] - energies[0],
                        doc[i],
                        dccl[i],
                        'TS' if i == tsidx else 'R' if i == 0 else 'P' if i == len(xs) - 1 else 'path',
                    ]
                )
        em = m.evaluate(xr['x'], field=False)[0]
        dump(
            WORK / 'summary.json',
            dict(
                system='Haloalkane dehalogenase DhlA / 1,2-dichloroethane, PDB 2DHC',
                reaction='First catalytic alkylation: Asp124 carboxylate + DCE -> Asp124-O-CH2-CH2-Cl + Cl-',
                engine='MOPAC 23.2.3 PM7 / Amber14 electrostatic embedding + Amber14/GAFF2.11 LJ',
                qm_atoms=len(m.xyz),
                mobile_qm_atoms=len(m.mobile),
                mm_atoms=len(m.mmidx),
                charge=-1,
                path_images=len(xs),
                ts_index=tsidx,
                barrier_kcal_mol=float(energies[tsidx] - energies[0]),
                reaction_energy_kcal_mol=float(energies[-1] - energies[0]),
                imaginary_frequency_cm1=vm['ts']['signed_frequencies_cm1'][0],
                reactant_label=r,
                product_label=pdt,
                embedding_energy_shift_kcal_mol=float(energies[0] - em),
                validation=vm,
                scope='Converged TS and two minima in 33 mobile QM Cartesian coordinates, with fixed protein and QM anchors. A single elementary alkylation step; not a full catalytic cycle or free-energy barrier.',
                path_method='Energy-decreasing mass-weighted steepest descent from both sides of the TS, followed by record-lowering evaluations from endpoint L-BFGS minimization. Nearest saved structures selected by arc length; no geometry or energy interpolation.',
            ),
        )
        print(
            'Path exported',
            len(xs),
            'images, barrier',
            energies[tsidx] - energies[0],
            'reaction',
            energies[-1] - energies[0],
            flush=True,
        )
    elif args.stage == 'validate':
        results = {}
        for label in ['reactant', 'ts', 'downhill_minus', 'downhill_plus']:
            d = np.load(WORK / f'{label}.npz')
            h, asym = m.hessian(d['x'], step=0.0005)
            w, v = np.linalg.eigh(h / np.sqrt(m.masses[:, None] * m.masses[None, :]))
            freq = np.sign(w) * np.sqrt(np.abs(w)) * 108.5913586
            e, g = m.evaluate(d['x'])
            results[label] = dict(
                energy_kcal_mol=e,
                max_gradient=float(np.max(np.abs(g))),
                gradient_norm=float(np.linalg.norm(g)),
                negative_modes=int(np.sum(w < -0.0001)),
                signed_frequencies_cm1=freq.tolist(),
                hessian_asymmetry_max=asym,
                rc_A=m.rc(d['x']),
            )
            np.savez_compressed(
                WORK / f'{label}_hessian.npz', hessian=h, eigenvalues=w, eigenvectors=v, frequencies_cm1=freq
            )
            print(label, results[label], flush=True)
        dump(WORK / 'stationary_validation.json', results)


if __name__ == '__main__':
    main()
