#!/usr/bin/env python3
"""Independent provenance and coordinate checks for the three molecular films."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pymol2
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cinematic_real'
sys.path.insert(0, str(ROOT / 'scripts'))
from render_real_cinematic_videos import setup  # noqa: E402

report = {}
dock = json.loads((OUT / 'docking/summary.json').read_text())
assert len(dock['poses']) == 9
for i, m in enumerate(Chem.SDMolSupplier(str(OUT / 'docking/poses.sdf'), removeHs=False)):
    assert abs(float(m.GetProp('minimizedAffinity')) - dock['poses'][i]['score_kcal_mol']) < 1e-8
report['docking_scores_match_smina_sdf'] = True
query = Chem.MolFromSmarts('[#7]~[#6](~[#7])~[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~1')
crystal = Chem.MolFromPDBFile(str(OUT / 'data/crystal_benzamidine.pdb'), removeHs=True)
matches = crystal.GetSubstructMatches(query, uniquify=False)
ref = crystal.GetConformer().GetPositions()
for i, m in enumerate(Chem.SDMolSupplier(str(OUT / 'docking/poses.sdf'), removeHs=True)):
    xyz = m.GetConformer().GetPositions()
    rmsd = min(np.sqrt(np.mean(np.sum((xyz - ref[list(p)]) ** 2, axis=1))) for p in matches)
    assert abs(rmsd - dock['poses'][i]['crystal_rmsd_A']) < 1e-8
report['docking_rmsd_recomputed_without_alignment'] = True

q = np.load(OUT / 'qmmm/path.npz')
summary = json.loads((OUT / 'qmmm/summary.json').read_text())
assert q['coords'].shape == (33, 25, 3)
for i, e in enumerate(q['energies']):
    raw = (OUT / f'qmmm/path/image_{i:03d}/step.out').read_text()
    assert 'PM7' in raw and 'QMMM' in raw and 'SCF FIELD WAS ACHIEVED' in raw
    parsed = float(re.findall(r'FINAL HEAT OF FORMATION\s*=\s*([-+0-9.]+)\s*KCAL', raw)[-1])
    assert abs(parsed - e) < 1e-8
    lines = (OUT / f'qmmm/path/image_{i:03d}/mol.in').read_text().splitlines()
    assert sum(map(int, lines[1].split())) == 25 and len(lines[2:]) == 25
    assert all(len(row.split()) == 5 for row in lines[2:])
assert abs(summary['validation']['embedding_energy_shift_kcal_mol']) > 1
assert summary['validation']['full_enzyme_TS_validated'] is False
assert abs(summary['enzyme_total_charge'] - 6) < 1e-8
assert abs(np.load(OUT / 'qmmm/embedding.npz')['mm_charges'].sum() - 7) < 1e-8
assert np.allclose(q['relative'], q['energies'] - q['energies'][0])
report['qmmm_33_energies_match_raw_mopac_outputs'] = True
report['qmmm_field_enters_hamiltonian'] = True
report['qmmm_total_charge_conserved'] = True
report['qmmm_scope'] = 'Real electrostatic-embedding constrained scan; no validated enzyme TS.'

with pymol2.PyMOL() as pm:
    c = pm.cmd
    setup(c, 'docking')
    for rank in [1, 2, 3]:
        m = Chem.MolFromPDBFile(str(OUT / f'docking/pose_{rank:02d}.pdb'), removeHs=True)
        xyz = m.GetConformer().GetPositions()
        c.load_coords(xyz, 'ligand')
        assert np.allclose(c.get_coords('ligand'), xyz, atol=2e-6)
        names = [a.name for a in c.get_model('ligand').atom]
        expected = [a.GetPDBResidueInfo().GetName().strip() for a in m.GetAtoms()]
        assert names == expected
    setup(c, 'qmmm')
    expected = json.loads((OUT / 'qmmm/qm_atoms.json').read_text())
    assert [(a.name, a.resi) for a in c.get_model('quantum').atom] == [(a['name'], a['resid']) for a in expected]
    for i in [0, 16, 32]:
        c.load_coords(q['coords'][i], 'quantum')
        assert np.allclose(c.get_coords('quantum'), q['coords'][i], atol=2e-6)
report['pymol_atom_order_and_coordinates_match_saved_calculations'] = True

if (OUT / 'md/aligned.npz').exists():
    import MDAnalysis as mda

    md = np.load(OUT / 'md/aligned.npz')
    u = mda.Universe(str(OUT / 'md/minimized.pdb'), str(OUT / 'md/trajectory.dcd'))
    assert len(u.trajectory) == 120 and md['xyz'].shape == (120, 3220, 3)
    ca = u.select_atoms('name CA').indices
    for i in [0, 59, 119]:
        u.trajectory[i]
        raw = u.atoms.positions.copy()
        shown = md['xyz'][i]
        # Distances must be invariant under the visualization's rigid fit.
        assert np.allclose(
            np.linalg.norm(raw[ca[1:]] - raw[ca[:-1]], axis=1),
            np.linalg.norm(shown[ca[1:]] - shown[ca[:-1]], axis=1),
            atol=1e-5,
        )
    assert np.isfinite(md['analysis']).all()
    report['md_saved_frames_and_geometry_match_dcd'] = True
else:
    report['md_status'] = 'pending'

(OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
