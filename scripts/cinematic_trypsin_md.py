#!/usr/bin/env python3
"""A short real apo-trypsin trajectory, with exported aligned snapshots."""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ['OPENMM_CPU_THREADS'] = '2'
from openmm import GBSAOBCForce, LangevinMiddleIntegrator, NonbondedForce, Platform, XmlSerializer, unit
from openmm.app import DCDReporter, ForceField, HBonds, NoCutoff, PDBFile, Simulation, StateDataReporter
from openmm.app.internal.customgbforces import GBSAOBC2Force

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cinematic_real'
WORK = OUT / 'md'
WORK.mkdir(exist_ok=True)
(WORK / 'worker.pid').write_text(str(os.getpid()))
print('Reading topology', flush=True)
pdb = PDBFile(str(OUT / 'data/trypsin_amber.pdb'))
ff = ForceField('amber14-all.xml')
print('Creating force field', flush=True)
system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)
nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
gb = GBSAOBCForce()
gb.setSolventDielectric(78.5)
gb.setSoluteDielectric(1.0)
for i, (radius, scale) in enumerate(GBSAOBC2Force.getStandardParameters(pdb.topology)):
    gb.addParticle(nb.getParticleParameters(i)[0], radius, scale)
system.addForce(gb)
integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picosecond)
integrator.setRandomNumberSeed(20260915)
print('Creating CPU context / native OBC-II solvent', flush=True)
sim = Simulation(pdb.topology, system, integrator, Platform.getPlatformByName('CPU'), {'Threads': '2'})
sim.context.setPositions(pdb.positions)
start = time.perf_counter()
print('Minimizing', flush=True)
sim.minimizeEnergy(maxIterations=150)
print('Minimized', time.perf_counter() - start, flush=True)
with (WORK / 'minimized.pdb').open('w') as f:
    PDBFile.writeFile(pdb.topology, sim.context.getState(getPositions=True).getPositions(), f, keepIds=True)
sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 20260915)
sim.step(1000)
print('Equilibrated', time.perf_counter() - start, flush=True)
sim.reporters.append(DCDReporter(str(WORK / 'trajectory.dcd'), 50))
sim.reporters.append(
    StateDataReporter(str(WORK / 'state.csv'), 50, time=True, potentialEnergy=True, temperature=True, separator=',')
)
atoms = list(pdb.topology.atoms())
ca = np.array([a.index for a in atoms if a.name == 'CA'])
heavy = np.array([a.index for a in atoms if a.element.symbol != 'H'])
oi = next(a.index for a in atoms if a.residue.id == '195' and a.name == 'OG')
ni = next(a.index for a in atoms if a.residue.id == '57' and a.name == 'NE2')
ref = np.asarray(sim.context.getState(getPositions=True).getPositions().value_in_unit(unit.angstrom))
center = ref[ca].mean(0)
frames = []
rows = []
for i in range(120):
    sim.step(50)
    state = sim.context.getState(getPositions=True, getEnergy=True)
    xyz = np.asarray(state.getPositions().value_in_unit(unit.angstrom))
    mc = xyz[ca].mean(0)
    u, _, vt = np.linalg.svd((xyz[ca] - mc).T @ (ref[ca] - center))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    aligned = (xyz - mc) @ rot + center
    frames.append(aligned.astype(np.float32))
    rows.append(
        [
            i,
            (i + 1) * 0.1,
            float(np.sqrt(np.mean(np.sum((aligned[ca] - ref[ca]) ** 2, axis=1)))),
            float(np.linalg.norm(xyz[oi] - xyz[ni])),
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)),
        ]
    )
    if i % 20 == 0:
        print('MD snapshot', i, rows[-1], flush=True)
with (WORK / 'analysis.csv').open('w') as f:
    w = csv.writer(f)
    w.writerow(['frame', 'production_time_ps', 'ca_rmsd_A', 'Ser195_OG_His57_NE2_A', 'potential_kJ_mol'])
    w.writerows(rows)
np.savez_compressed(WORK / 'aligned.npz', xyz=np.asarray(frames), ca=ca, heavy=heavy, analysis=rows)
(WORK / 'system.xml').write_text(XmlSerializer.serialize(system))
(WORK / 'integrator.xml').write_text(XmlSerializer.serialize(integrator))
sim.saveCheckpoint(str(WORK / 'checkpoint.chk'))
summary = dict(
    system='Apo trypsin, PDB 3PTB',
    engine='OpenMM',
    force_field='Amber14 + OBC-II implicit solvent',
    platform='CPU / native GBSAOBCForce',
    atoms=len(atoms),
    seed=20260915,
    temperature_K=300,
    timestep_fs=2,
    equilibration_ps=2,
    production_ps=12,
    snapshot_interval_ps=0.1,
    frames=120,
    runtime_seconds=time.perf_counter() - start,
    scope='Short relaxation demonstration. Does not establish stability or converged equilibrium statistics. Ligand, waters and calcium removed.',
    alignment='Least-squares C-alpha fit to post-equilibration reference; exact saved coordinates, no trajectory morphing',
)
(WORK / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(summary, flush=True)
