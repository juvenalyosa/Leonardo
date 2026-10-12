#!/usr/bin/env python3
"""Cheap, auditable disulfide design on the 56-residue protein G B1 domain.

Geometry screening + restrained Amber14/OBC-II relaxation + Gaussian network
sensitivity analysis. This does not calculate folding free energy or melting T.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

os.environ['OPENMM_CPU_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cinematic_v2'
WORK = OUT / 'stability_design'
DATA = WORK / 'data'


def dump(p, obj):
    p.write_text(json.dumps(obj, indent=2) + '\n')


def source_residues():
    residues = {}
    for line in (DATA / '1PGA.pdb').read_text().splitlines():
        if line.startswith('ATOM  ') and line[21] == 'A' and line[16] in ' A':
            r = residues.setdefault(int(line[22:26]), {'resname': line[17:20]})
            r[line[12:16].strip()] = np.array([float(line[a:b]) for a,b in [(30,38),(38,46),(46,54)]])
    return residues


def sulfur(residue, theta):
    u = residue['CA']-residue['CB']; u /= np.linalg.norm(u)
    v = residue['N']-residue['CA']; v -= u*np.dot(u,v); v /= np.linalg.norm(v)
    w = np.cross(u,v)
    angle = np.deg2rad(114)
    return residue['CB'] + 1.81*(np.cos(angle)*u+np.sin(angle)*(np.cos(theta)*v+np.sin(theta)*w))


def angle(a,b,c):
    x=a-b; y=c-b
    return float(np.rad2deg(np.arccos(np.clip(np.dot(x,y)/np.linalg.norm(x)/np.linalg.norm(y),-1,1))))


def torsion(a,b,c,d):
    axis = c-b; axis /= np.linalg.norm(axis)
    x=a-b; x-=axis*np.dot(x,axis)
    y=d-c; y-=axis*np.dot(y,axis)
    return float(np.rad2deg(np.arctan2(np.dot(np.cross(axis,x),y),np.dot(x,y))))


def screen():
    start=time.perf_counter(); residues=source_residues(); rows=[]
    # Avoid replacing hydrophobic core residues in this inexpensive proposal.
    allowed={'SER','THR','ASN','GLN','ASP','GLU','LYS','ARG','ALA'}
    for i,a in residues.items():
        for j,b in residues.items():
            if j-i<12 or a['resname'] not in allowed or b['resname'] not in allowed or 'CB' not in a or 'CB' not in b:
                continue
            cb=np.linalg.norm(a['CB']-b['CB']); ca=np.linalg.norm(a['CA']-b['CA'])
            if not (3.2<cb<5.2 and 4<ca<7.5):
                continue
            obstacles=[]
            for k,r in residues.items():
                for atom,p in r.items():
                    if atom=='resname' or (k in [i,j] and atom not in {'N','C','O'}):
                        continue
                    obstacles.append(p)
            obstacles=np.array(obstacles)
            def objective(x):
                sa,sb=sulfur(a,x[0]),sulfur(b,x[1])
                d=np.linalg.norm(sa-sb)
                aa=angle(a['CB'],sa,sb); ab=angle(sa,sb,b['CB'])
                chi=torsion(a['CB'],sa,sb,b['CB'])
                clash=np.maximum(0,2.7-np.linalg.norm(np.array([sa,sb])[:,None,:]-obstacles[None,:,:],axis=-1))
                return ((d-2.03)/0.10)**2+((aa-104)/12)**2+((ab-104)/12)**2+((abs(chi)-90)/25)**2+float(np.sum(clash**2))*40
            candidates=[]
            for aa in np.linspace(-math.pi,math.pi,6,endpoint=False):
                for bb in np.linspace(-math.pi,math.pi,6,endpoint=False):
                    candidates.append(minimize(objective,[aa,bb],method='BFGS',options={'maxiter':120,'gtol':1e-5}))
            best=min(candidates,key=lambda r:r.fun)
            sa,sb=sulfur(a,best.x[0]),sulfur(b,best.x[1])
            rows.append(dict(residue_i=i,residue_j=j,original_i=a['resname'],original_j=b['resname'],
                sequence_span=j-i,geometry_penalty=float(best.fun),initial_SS_A=float(np.linalg.norm(sa-sb)),
                initial_CSS_angles_deg=[angle(a['CB'],sa,sb),angle(sa,sb,b['CB'])],
                initial_chi3_deg=torsion(a['CB'],sa,sb,b['CB']),sulfur_coordinates_A=[sa.tolist(),sb.tolist()]))
    rows.sort(key=lambda r:r['geometry_penalty'])
    dump(WORK/'screen.json',dict(candidates=rows,selected=rows[0],runtime_seconds=time.perf_counter()-start,
        method='All polar/small residue pairs, sequence separation >=12, CB 3.2–5.2 A and CA 4–7.5 A; 36 sulfur-rotamer starts per pair; geometry/clash penalty only, not folding energy.'))
    print(json.dumps(rows,indent=2),flush=True)


def prepare_model(variant,selection=None):
    from openmm import CustomExternalForce, GBSAOBCForce, NonbondedForce, Platform, VerletIntegrator, XmlSerializer, unit
    from openmm.app import ForceField, Modeller, NoCutoff, PDBFile, Simulation, element
    from openmm.app.internal.customgbforces import GBSAOBC2Force
    folder=WORK/variant; folder.mkdir(exist_ok=True)
    if (folder/'relaxation.npz').exists():
        print('Reusing',variant,flush=True); return
    source=DATA/'protein_heavy.pdb'
    if not source.exists():
        source.write_text('\n'.join(l for l in (DATA/'1PGA.pdb').read_text().splitlines()
                                    if l.startswith('ATOM  ') and l[21]=='A' and l[16] in ' A')+'\nTER\nEND\n')
    pdb=PDBFile(str(source)); mod=Modeller(pdb.topology,pdb.positions)
    selected=selection or json.loads((WORK/'screen.json').read_text())['selected']
    pair=[selected['residue_i'],selected['residue_j']]
    if variant!='original':
        changed=[r for r in mod.topology.residues() if int(r.id) in pair]
        mod.delete([a for r in changed for a in r.atoms() if a.name not in {'N','CA','C','O','OXT','CB'}])
        xyz=list(mod.positions.value_in_unit(unit.nanometer))
        # Insert each new sulfur while its residue is contiguous in the topology.
        from openmm.app import Topology
        top=Topology(); amap={}; ordered=[]; sulfurs=[]
        for chain in mod.topology.chains():
            nc=top.addChain(chain.id)
            for r in chain.residues():
                nr=top.addResidue('CYS' if int(r.id) in pair else r.name,nc,r.id)
                for a in r.atoms():
                    amap[a]=top.addAtom(a.name,a.element,nr); ordered.append(xyz[a.index])
                if int(r.id) in pair:
                    sg=top.addAtom('SG',element.sulfur,nr)
                    ordered.append(np.array(selected['sulfur_coordinates_A'][pair.index(int(r.id))])/10)
                    cb=next(a for a in r.atoms() if a.name=='CB')
                    top.addBond(amap[cb],sg);sulfurs.append(sg)
        for a,b in mod.topology.bonds(): top.addBond(amap[a],amap[b])
        top.addBond(*sulfurs)
        mod=Modeller(top,np.array(ordered)*unit.nanometer)
    ff=ForceField('amber14-all.xml'); random.seed(20260915)
    mod.addHydrogens(ff,pH=7)
    initial=np.asarray(mod.positions.value_in_unit(unit.angstrom))
    system=ff.createSystem(mod.topology,nonbondedMethod=NoCutoff,constraints=None)
    nb=next(f for f in system.getForces() if isinstance(f,NonbondedForce))
    gb=GBSAOBCForce(); gb.setSolventDielectric(78.5); gb.setSoluteDielectric(1)
    for i,(radius,scale) in enumerate(GBSAOBC2Force.getStandardParameters(mod.topology)):
        gb.addParticle(nb.getParticleParameters(i)[0],radius,scale)
    system.addForce(gb)
    restraint=CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k',100)
    for name in ['x0','y0','z0']: restraint.addPerParticleParameter(name)
    atoms=list(mod.topology.atoms())
    for a in atoms:
        if a.name in {'N','CA','C','O'}: restraint.addParticle(a.index,(initial[a.index]/10).tolist())
    system.addForce(restraint)
    sim=Simulation(mod.topology,system,VerletIntegrator(0.001),Platform.getPlatformByName('CPU'),{'Threads':'2'})
    sim.context.setPositions(mod.positions)
    frames=[]; energies=[]; start=time.perf_counter()
    for step in range(31):
        if step: sim.minimizeEnergy(tolerance=1*unit.kilojoule_per_mole/unit.nanometer,maxIterations=35)
        state=sim.context.getState(getPositions=True,getEnergy=True)
        frames.append(np.asarray(state.getPositions().value_in_unit(unit.angstrom)))
        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole))
    with (folder/'model.pdb').open('w') as f: PDBFile.writeFile(mod.topology,frames[-1]*unit.angstrom,f,keepIds=True)
    (folder/'system.xml').write_text(XmlSerializer.serialize(system))
    meta=[dict(index=a.index,name=a.name,resid=int(a.residue.id),resname=a.residue.name,element=a.element.symbol) for a in atoms]
    dump(folder/'atoms.json',meta)
    np.savez_compressed(folder/'relaxation.npz',xyz=np.array(frames),energy_kcal_mol=energies,
                        ca=[a.index for a in atoms if a.name=='CA'],heavy=[a.index for a in atoms if a.element.symbol!='H'])
    dump(folder/'preparation.json',dict(variant=variant,atoms=len(atoms),residues=mod.topology.getNumResidues(),
        disulfide_bonds=[[int(a.residue.id),int(b.residue.id)] for a,b in mod.topology.bonds() if a.name==b.name=='SG'],
        restrained_backbone_k_kJ_mol_nm2=100,force_field='Amber14 / OBC-II',runtime_seconds=time.perf_counter()-start,
        protocol='31 saved states: initial + 30 batches of up to 35 local minimization iterations. Backbone restrained; no MD. Potential energies are comparable only within each variant.'))
    print(variant,'relaxed',len(atoms),'atoms',time.perf_counter()-start,'seconds',flush=True)


def covariance(coords,pair,strength,cutoff=7.5):
    n=len(coords); distance=np.linalg.norm(coords[:,None,:]-coords[None,:,:],axis=-1)
    contacts=(distance<cutoff)&(distance>0)
    k=-contacts.astype(float); np.fill_diagonal(k,-k.sum(axis=1))
    i,j=pair
    # Additional cross-link stiffness is an explicit model parameter, not a fit.
    k[i,i]+=strength;k[j,j]+=strength;k[i,j]-=strength;k[j,i]-=strength
    values,vectors=np.linalg.eigh(k)
    assert abs(values[0])<1e-9 and values[1]>1e-6
    inverse=(vectors[:,1:]/values[None,1:])@vectors[:,1:].T
    return k,inverse,values,vectors


def select_relaxed():
    screen_data=json.loads((WORK/'screen.json').read_text());original=source_residues();results=[]
    for candidate in screen_data['candidates']:
        i,j=candidate['residue_i'],candidate['residue_j'];variant=f'candidate_{i}_{j}'
        prepare_model(variant,candidate)
        folder=WORK/variant;meta=json.loads((folder/'atoms.json').read_text());d=np.load(folder/'relaxation.npz');xyz=d['xyz'][-1]
        index=lambda r,n:next(a['index'] for a in meta if a['resid']==r and a['name']==n)
        a,b,c,e=[xyz[index(r,n)] for r,n in [(i,'CB'),(i,'SG'),(j,'SG'),(j,'CB')]]
        ss=float(np.linalg.norm(b-c));angles=[angle(a,b,c),angle(b,c,e)];chi=torsion(a,b,c,e)
        ref=np.array([original[a['resid']]['CA'] for a in meta if a['name']=='CA'])
        rms=float(np.sqrt(np.mean(np.sum((xyz[d['ca']]-ref)**2,axis=1))))
        passed=1.95<ss<2.15 and all(90<v<120 for v in angles) and 65<abs(chi)<115 and rms<.7
        results.append(dict(pair=[i,j],candidate_folder=variant,SS_A=ss,CSS_angles_deg=angles,chi3_deg=chi,
                            ca_displacement_rms_A=rms,passed_geometry_filter=passed))
    accepted=[r for r in results if r['passed_geometry_filter']]
    assert accepted,'No candidate passed the fixed structural acceptance criteria'
    best=min(accepted,key=lambda r:abs(abs(r['chi3_deg'])-90))
    screen_data['initial_geometry_selected']=screen_data.get('initial_geometry_selected',screen_data['selected'])
    screen_data['selected']=next(c for c in screen_data['candidates'] if [c['residue_i'],c['residue_j']]==best['pair'])
    screen_data['relaxed_candidates']=results
    screen_data['selection_protocol']='Choose after restrained relaxation: SS 1.95–2.15 A; CSS angles 90–120 deg; abs(chi3) 65–115 deg; CA RMS displacement <0.7 A. Rank passing candidates by abs(abs(chi3)-90).'
    dump(WORK/'screen.json',screen_data)
    target=WORK/'design'
    if target.exists():
        prep=json.loads((target/'preparation.json').read_text())
        if prep['disulfide_bonds']==[best['pair']]:
            print('Selected design already installed',best,flush=True);return
        oldpair=prep['disulfide_bonds'][0]
        target.rename(WORK/f'pilot_unselected_{oldpair[0]}_{oldpair[1]}')
    shutil.copytree(WORK/best['candidate_folder'],target)
    dump(target/'selection.json',best)
    print('Selected after relaxation',best,flush=True)


def analyze():
    s=json.loads((WORK/'screen.json').read_text())['selected']
    pair=[s['residue_i'],s['residue_j']]
    original=source_residues(); ids=list(original); coords=np.array([original[r]['CA'] for r in ids])
    indices=[ids.index(r) for r in pair]
    wt=covariance(coords,indices,0); design=covariance(coords,indices,10)
    # Common reference and contact map isolate the added elastic edge.
    normalization=float(np.sqrt(np.mean(np.diag(wt[1]))))
    mobility_wt=np.sqrt(np.diag(wt[1]))/normalization
    mobility_design=np.sqrt(np.diag(design[1]))/normalization
    sep=np.zeros(len(ids));sep[indices[0]]=1;sep[indices[1]]=-1
    pair_wt=float(sep@wt[1]@sep);pair_design=float(sep@design[1]@sep)
    sensitivity=[]
    for cutoff in [7,7.5,8]:
        reference=covariance(coords,indices,0,cutoff)[1]
        for strength in [5,10,20]:
            cov=covariance(coords,indices,strength,cutoff)[1]
            sensitivity.append(dict(cutoff_A=cutoff,extra_stiffness=strength,
                pair_relative_rms_reduction_percent=100*(1-np.sqrt((sep@cov@sep)/(sep@reference@sep))),
                whole_domain_rms_reduction_percent=100*(1-np.sqrt(np.trace(cov)/np.trace(reference)))))
    validation={}
    for variant in ['original','design']:
        d=np.load(WORK/variant/'relaxation.npz'); meta=json.loads((WORK/variant/'atoms.json').read_text())
        ca=d['xyz'][-1,d['ca']]; ref=np.array([original[a['resid']]['CA'] for a in meta if a['name']=='CA'])
        validation[variant]=dict(backbone_ca_displacement_rms_A=float(np.sqrt(np.mean(np.sum((ca-ref)**2,axis=1)))),
            max_energy_increase_kcal_mol=float(np.max(np.diff(d['energy_kcal_mol']))))
        assert validation[variant]['max_energy_increase_kcal_mol']<0.005
        assert np.isfinite(d['xyz']).all()
    d=np.load(WORK/'design/relaxation.npz'); meta=json.loads((WORK/'design/atoms.json').read_text());xyz=d['xyz'][-1]
    index=lambda r,name:next(a['index'] for a in meta if a['resid']==r and a['name']==name)
    ca,sa,sb,cb=[xyz[index(r,name)] for r,name in [(pair[0],'CB'),(pair[0],'SG'),(pair[1],'SG'),(pair[1],'CB')]]
    geometry=dict(SS_A=float(np.linalg.norm(sa-sb)),CSS_angles_deg=[angle(ca,sa,sb),angle(sa,sb,cb)],chi3_deg=torsion(ca,sa,sb,cb))
    assert 1.95<geometry['SS_A']<2.15
    assert all(90<a<120 for a in geometry['CSS_angles_deg'])
    assert 65<abs(geometry['chi3_deg'])<115
    assert validation['design']['backbone_ca_displacement_rms_A']<0.7
    # Positivity, connectivity and the rank-one update identity independently
    # check the Gaussian-network computation in the translation-free subspace.
    expected=wt[1]-(10*np.outer(wt[1]@sep,wt[1]@sep))/(1+10*pair_wt)
    assert np.max(np.abs(expected-design[1]))<1e-9
    assert np.linalg.eigvalsh(wt[1]-design[1]).min()>-1e-9
    assert all(r['pair_relative_rms_reduction_percent']>0 for r in sensitivity)
    with (WORK/'mobility.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['residue','original_relative_rms_mobility','design_relative_rms_mobility'])
        w.writerows(zip(ids,mobility_wt,mobility_design))
    with (WORK/'sensitivity.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(sensitivity[0]));w.writeheader();w.writerows(sensitivity)
    np.savez_compressed(WORK/'network.npz',coords=coords,resids=ids,original_covariance=wt[1],design_covariance=design[1],
        original_kirchhoff=wt[0],design_kirchhoff=design[0],original_mobility=mobility_wt,design_mobility=mobility_design,
        original_eigenvalues=wt[2],original_eigenvectors=wt[3],design_eigenvalues=design[2],design_eigenvectors=design[3])
    one={'LYS':'K','THR':'T','ASN':'N','GLU':'E','ALA':'A','SER':'S','GLN':'Q','ASP':'D','ARG':'R'}
    name=f"{one[s['original_i']]}{pair[0]}C / {one[s['original_j']]}{pair[1]}C"
    dump(WORK/'summary.json',dict(system='Protein G B1 domain (GB1); PDB 1PGA; chain A; 56 residues',design=name,
        residue_pair=pair,sequence_span=s['sequence_span'],geometry=geometry,structural_validation=validation,
        pair_relative_rms_reduction_percent=100*(1-np.sqrt(pair_design/pair_wt)),
        whole_domain_rms_reduction_percent=100*(1-np.sqrt(np.trace(design[1])/np.trace(wt[1]))),
        pair_variance_original=pair_wt,pair_variance_design=pair_design,sensitivity=sensitivity,
        source_sha256=hashlib.sha256((DATA/'1PGA.pdb').read_bytes()).hexdigest(),source_doi='10.1021/bi00181a032',
        network_model='Isotropic Gaussian network, CA cutoff 7.5 A, unit contact stiffness; disulfide adds a 10x contact spring. Identical reference coordinates. Unitless mobilities share WT RMS normalization.',
        scope='A structural proposal intended to increase rigidity. Local Amber minimization and coarse-grained harmonic predictions only; not experimental thermostability, folding free energy, activity or a validated production design.',
        checked=True))
    print('Design',name,'geometry',geometry,'local relative fluctuation reduction',100*(1-np.sqrt(pair_design/pair_wt)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['screen','original','design','select','analyze','all'])
    args=parser.parse_args()
    if args.stage in ['screen','all']:screen()
    for variant in ['original','design']:
        if args.stage==variant or (args.stage=='all' and variant=='original'):prepare_model(variant)
    if args.stage in ['select','all']:select_relaxed()
    if args.stage in ['analyze','all']:analyze()
