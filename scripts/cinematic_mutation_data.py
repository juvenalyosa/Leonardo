#!/usr/bin/env python3
"""Real alanine-mutation comparisons for molecular films 04 and 05.

Reuses the established fixed-protein PM7/Amber model without altering its data.
Only the named side chain changes. All shared atoms keep their source coordinates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENMM_CPU_THREADS'] = '2'
import cinematic_dhla_reaction as qm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cinematic_v2'
WORK = OUT / 'mutations'
OLD = ROOT / 'artifacts/cinematic_real'
ENV = Path('/opt/anaconda3/envs/protliginteract-full/bin')
SEEDS = [20260915, 20260916, 20260917]


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2) + '\n')


def mutate(source, stem, residue, expected, target):
    from openmm import NonbondedForce, XmlSerializer, unit
    from openmm.app import ForceField, Modeller, NoCutoff, PDBFile

    target.mkdir(parents=True, exist_ok=True)
    preparation = target / 'preparation.json'
    if preparation.exists():
        previous = json.loads(preparation.read_text())
        assert previous['source_sha256'] == hashlib.sha256((source / f'{stem}_amber.npz').read_bytes()).hexdigest()
        assert all(
            (target / name).exists()
            for name in [f'{stem}_amber.pdb', f'{stem}_amber.npz', f'{stem}_atoms.json', f'{stem}_system.xml']
        )
        print('Reusing prepared mutation', previous['mutation'], flush=True)
        return
    pdb = PDBFile(str(source / f'{stem}_amber.pdb'))
    original = np.load(source / f'{stem}_amber.npz')['xyz']
    before = {(a.residue.chain.id, a.residue.id, a.name): original[a.index] for a in pdb.topology.atoms()}
    mod = Modeller(pdb.topology, original * unit.angstrom)
    candidates = [r for r in mod.topology.residues() if r.id == str(residue) and r.chain.id == 'A']
    assert len(candidates) == 1 and candidates[0].name == expected
    res = candidates[0]
    mod.delete([a for a in res.atoms() if a.name not in {'N', 'H', 'CA', 'HA', 'C', 'O', 'CB'}])
    res = next(r for r in mod.topology.residues() if r.id == str(residue) and r.chain.id == 'A')
    res.name = 'ALA'
    ff = ForceField('amber14-all.xml')
    random.seed(20260915)
    variants = []
    for r in mod.topology.residues():
        names = {a.name for a in r.atoms()}
        variants.append(
            ('HIP' if 'HE2' in names else 'HID')
            if 'HD1' in names and r.name == 'HIS'
            else 'HIE'
            if r.name == 'HIS' and 'HE2' in names
            else None
        )
    mod.addHydrogens(ff, pH=7, variants=variants)
    system = ff.createSystem(mod.topology, nonbondedMethod=NoCutoff, constraints=None)
    xyz = np.asarray(mod.positions.value_in_unit(unit.angstrom))
    # Hydrogen completion may relax shared coordinates; restore the exact source
    # for every retained atom so the comparison changes only the side chain.
    for atom in mod.topology.atoms():
        key = (atom.residue.chain.id, atom.residue.id, atom.name)
        if key in before:
            xyz[atom.index] = before[key]
    mod.positions = xyz * unit.angstrom
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    pars = np.array(
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
    differences = [
        np.max(np.abs(xyz[a.index] - before[(a.residue.chain.id, a.residue.id, a.name)]))
        for a in mod.topology.atoms()
        if a.residue.id != str(residue)
    ]
    assert max(differences) < 1e-8, 'Unmutated coordinates must remain unchanged'
    with (target / f'{stem}_amber.pdb').open('w') as f:
        PDBFile.writeFile(mod.topology, mod.positions, f, keepIds=True)
    (target / f'{stem}_system.xml').write_text(XmlSerializer.serialize(system))
    np.savez_compressed(target / f'{stem}_amber.npz', xyz=xyz, params=pars, exceptions=np.array(exc))
    dump(target / f'{stem}_atoms.json', meta)
    dump(
        target / 'preparation.json',
        dict(
            mutation=f'{expected}{residue}ALA',
            chain='A',
            source=str(source.relative_to(ROOT)),
            source_sha256=hashlib.sha256((source / f'{stem}_amber.npz').read_bytes()).hexdigest(),
            protocol='Alanine sidechain truncation at CB; Amber14 hydrogen completion; shared coordinates fixed',
            unmutated_max_coordinate_change_A=float(max(differences)),
            total_charge_e=float(pars[:, 0].sum()),
            atom_count=len(meta),
            forcefield='Amber14',
            seed=20260915,
        ),
    )
    print('Prepared', expected, residue, 'ALA;', len(meta), 'atoms', flush=True)


def prepare():
    mutate(OLD / 'data', 'trypsin', 189, 'ASP', WORK / 'binding/data')
    mutate(OUT / 'data', 'dhla', 125, 'TRP', WORK / 'enzyme/data')
    for name in ['dce_xyz.npy', 'dce_crystal_h.sdf']:
        shutil.copy2(OUT / 'data' / name, WORK / 'enzyme/data' / name)


def run(args, folder, log):
    p = subprocess.run(list(map(str, args)), cwd=folder, capture_output=True, text=True, timeout=900)
    (folder / log).write_text(p.stdout + '\n' + p.stderr)
    if p.returncode:
        raise RuntimeError((folder / log).read_text()[-2000:])


def docking():
    from rdkit import Chem

    work = WORK / 'binding'
    crystal = Chem.MolFromPDBFile(str(OLD / 'data/crystal_benzamidine.pdb'), removeHs=True)
    query = Chem.MolFromSmarts('[#7]~[#6](~[#7])~[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~1')
    matches = crystal.GetSubstructMatches(query, uniquify=False, maxMatches=10000)
    reference = crystal.GetConformer().GetPositions()
    assert matches
    source_atoms = json.loads((OLD / 'data/trypsin_atoms.json').read_text())
    source_xyz = np.load(OLD / 'data/trypsin_amber.npz')['xyz']
    anchor = [a['index'] for a in source_atoms if a['resid'] == '189' and a['name'] in ['OD1', 'OD2']]
    rows = []
    for variant in ['WT', 'D189A']:
        folder = work / variant
        folder.mkdir(exist_ok=True)
        receptor = OLD / 'data/trypsin_amber.pdb' if variant == 'WT' else work / 'data/trypsin_amber.pdb'
        if variant == 'WT':
            shutil.copy2(OLD / 'docking/receptor.pdbqt', folder / 'receptor.pdbqt')
        else:
            run(
                [ENV / 'obabel', receptor, '-O', folder / 'receptor.pdbqt', '-xr', '--partialcharge', 'gasteiger'],
                folder,
                'receptor.log',
            )
        for seed in SEEDS:
            sdf = folder / f'poses_{seed}.sdf'
            if not sdf.exists():
                if variant == 'WT' and seed == SEEDS[0]:
                    shutil.copy2(OLD / 'docking/poses.sdf', sdf)
                    shutil.copy2(OLD / 'docking/smina.log', folder / f'smina_{seed}.log')
                else:
                    run(
                        [
                            ENV / 'smina',
                            '--receptor',
                            folder / 'receptor.pdbqt',
                            '--ligand',
                            OLD / 'docking/benzamidinium_input.sdf',
                            '--autobox_ligand',
                            OLD / 'data/crystal_benzamidine.pdb',
                            '--autobox_add',
                            '8',
                            '--exhaustiveness',
                            '16',
                            '--num_modes',
                            '9',
                            '--seed',
                            seed,
                            '--cpu',
                            '2',
                            '--out',
                            sdf,
                            '--log',
                            folder / f'smina_{seed}.log',
                        ],
                        folder,
                        f'console_{seed}.log',
                    )
            for rank, mol in enumerate(Chem.SDMolSupplier(str(sdf), removeHs=False), 1):
                assert mol is not None
                heavy = Chem.RemoveHs(mol)
                xyz = heavy.GetConformer().GetPositions()
                assert heavy.GetNumAtoms() == 9 and Chem.GetFormalCharge(heavy) == 1
                rmsd = min(float(np.sqrt(np.mean(np.sum((xyz - reference[list(m)]) ** 2, axis=1)))) for m in matches)
                nitrogens = [a.GetIdx() for a in heavy.GetAtoms() if a.GetSymbol() == 'N']
                distance = np.linalg.norm(xyz[nitrogens, None, :] - source_xyz[anchor][None, :, :], axis=-1)
                count = int(np.sum(distance <= 3.5)) if variant == 'WT' else 0
                Chem.MolToPDBFile(mol, str(folder / f'pose_{seed}_{rank:02d}.pdb'))
                rows.append(
                    dict(
                        variant=variant,
                        seed=seed,
                        rank=rank,
                        score_kcal_mol=float(mol.GetProp('minimizedAffinity')),
                        crystal_rmsd_A=rmsd,
                        asp189_N_O_contacts_3p5_A=count,
                        asp189_min_N_O_A=float(distance.min()) if variant == 'WT' else None,
                    )
                )
            print(
                'Docked',
                variant,
                seed,
                'best',
                next(r for r in rows if r['variant'] == variant and r['seed'] == seed and r['rank'] == 1),
                flush=True,
            )
    with (work / 'scores.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = {v: [r for r in rows if r['variant'] == v and r['rank'] == 1] for v in ['WT', 'D189A']}
    medians = {v: float(np.median([r['score_kcal_mol'] for r in rr])) for v, rr in best.items()}
    dump(
        work / 'summary.json',
        dict(
            system='Trypsin / benzamidinium; PDB 3PTB',
            mutation='D189A',
            engine='smina/Vina',
            seeds=SEEDS,
            exhaustiveness=16,
            ligand_charge=1,
            rows=rows,
            best_poses=best,
            median_best_score=medians,
            delta_median_score_kcal_mol=medians['D189A'] - medians['WT'],
            scope='Rigid receptors, identical ligand and search box. Seeds assess search variability only. Vina scores are not binding free energies. N-O contacts are geometric, not hydrogen-bond occupancy.',
        ),
    )


def model():
    qm.DATA = WORK / 'enzyme/data'
    qm.WORK = WORK / 'enzyme/W125A'
    return qm.Model()


def stationary(m, x, label, saddle=False):
    p = qm.WORK / f'{label}.npz'
    if p.exists():
        return np.load(p)['x']
    if not saddle:
        return m.minimum(x, label)
    for k in range(35):
        e, g = m.evaluate(x)
        h, asym = m.hessian(x)
        w, v = np.linalg.eigh(h)
        print('Mutant TS', k, 'E', e, 'max gradient', np.abs(g).max(), 'lowest eigenvalues', w[:3], flush=True)
        if np.max(np.abs(g)) < 0.0005 and np.sum(w < -1e-4) == 1:
            break
        eig = np.maximum(np.abs(w), 0.5)
        eig[0] *= -1
        dx = -(v @ ((v.T @ g) / eig))
        dx *= min(1, 0.12 / np.max(np.linalg.norm(dx.reshape(-1, 3), axis=1)))
        x += dx
    assert np.max(np.abs(g)) < 0.001 and np.sum(w < -1e-4) == 1
    np.savez_compressed(p, x=x, xyz=m.unpack(x), energy=e, gradient=g, hessian=h)
    m.write_pdb(qm.WORK / f'{label}.pdb', m.unpack(x))
    return x


def enzyme():
    m = model()
    old = OUT / 'qmmm'
    states = {}
    for label, source in [('reactant', 'downhill_minus'), ('ts', 'ts'), ('product', 'downhill_plus')]:
        states[label] = stationary(m, np.load(old / f'{source}.npz')['x'].copy(), label, label == 'ts')
    results = {}
    for label, x in states.items():
        h, asym = m.hessian(x, step=0.0005)
        w, v = np.linalg.eigh(h / np.sqrt(m.masses[:, None] * m.masses[None, :]))
        e, g = m.evaluate(x)
        freq = np.sign(w) * np.sqrt(np.abs(w)) * 108.5913586
        results[label] = dict(
            energy_kcal_mol=e,
            max_gradient=float(np.abs(g).max()),
            negative_modes=int(np.sum(w < -1e-4)),
            imaginary_frequency_cm1=float(freq[0]),
            rc_A=float(m.rc(x)),
            hessian_asymmetry=asym,
        )
        assert results[label]['max_gradient'] < 0.001 and results[label]['negative_modes'] == (
            1 if label == 'ts' else 0
        )
        np.savez_compressed(
            qm.WORK / f'{label}_hessian.npz', hessian=h, eigenvalues=w, eigenvectors=v, frequencies_cm1=freq
        )
        print('Validated', label, results[label], flush=True)
    # Relaxed reaction-coordinate scan: evaluate both directions outward from the
    # independently converged saddle, retaining exact optimized geometries.
    nodes = [(states['ts'], results['ts']['energy_kcal_mol'], 'TS')]
    for endpoint in ['reactant', 'product']:
        x = states['ts'].copy()
        targets = np.linspace(m.rc(x), m.rc(states[endpoint]), 23)[1:-1]
        for i, target in enumerate(targets):
            label = f'scan_{endpoint}_{i:02d}'
            p = qm.WORK / f'{label}.npz'
            x = np.load(p)['x'] if p.exists() else m.minimum(x, label, float(target), 450)
            d = np.load(p)
            projected = d['gradient'] - m.rc_grad(x) * np.dot(d['gradient'], m.rc_grad(x)) / np.dot(
                m.rc_grad(x), m.rc_grad(x)
            )
            assert abs(m.rc(x) - target) < 1e-5 and np.max(np.abs(projected)) < 0.01, label
            nodes.append((x.copy(), float(d['energy']), 'scan'))
        nodes.append((states[endpoint], results[endpoint]['energy_kcal_mol'], endpoint))
    nodes.sort(key=lambda t: m.rc(t[0]), reverse=True)
    coords = np.array([m.unpack(n[0]) for n in nodes])
    energies = np.array([n[1] for n in nodes])
    ts = next(i for i, n in enumerate(nodes) if n[2] == 'TS')
    doc = np.linalg.norm(coords[:, m.o] - coords[:, m.c], axis=1)
    dccl = np.linalg.norm(coords[:, m.c] - coords[:, m.cl], axis=1)
    assert np.argmax(energies) == ts
    np.savez_compressed(
        qm.WORK / 'path.npz',
        coords=coords,
        energies=energies,
        relative=energies - energies[0],
        ts_index=ts,
        d_oc=doc,
        d_ccl=dccl,
        rc=doc - dccl,
        imaginary_frequency=results['ts']['imaginary_frequency_cm1'],
    )
    with (qm.WORK / 'profile.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['image', 'rc_A', 'energy_kcal_mol', 'relative_kcal_mol', 'O_C_A', 'C_Cl_A', 'state'])
        for i, n in enumerate(nodes):
            w.writerow([i, doc[i] - dccl[i], energies[i], energies[i] - energies[0], doc[i], dccl[i], n[2]])
            m.write_pdb(qm.WORK / f'path_{i:03d}.pdb', coords[i])
    original = json.loads((old / 'summary.json').read_text())
    barrier = results['ts']['energy_kcal_mol'] - results['reactant']['energy_kcal_mol']
    dump(
        WORK / 'enzyme/summary.json',
        dict(
            system='DhlA / 1,2-dichloroethane; PDB 2DHC',
            mutation='W125A',
            WT_barrier_kcal_mol=original['barrier_kcal_mol'],
            mutant_barrier_kcal_mol=barrier,
            delta_barrier_kcal_mol=barrier - original['barrier_kcal_mol'],
            validation=results,
            mutant_path_images=len(nodes),
            engine=original['engine'],
            scope='Fixed-protein PM7/Amber14 potential-energy comparison. Identical QM region and charge. WT saved downhill path; mutant relaxed coordinate scan. Each curve zeroed at its own optimized reactant. Not a free-energy barrier or measured rate.',
            path_method='Two constrained relaxations outward from a Hessian-validated saddle in d(O-C)-d(C-Cl); unrestrained minima at both ends. No interpolated molecular coordinates or invented energies.',
        ),
    )
    print(
        'BARRIERS', original['barrier_kcal_mol'], barrier, 'DELTA', barrier - original['barrier_kcal_mol'], flush=True
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'docking', 'enzyme'])
    globals()[p.parse_args().stage]()
