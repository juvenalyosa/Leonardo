#!/usr/bin/env python3
"""Small, auditable calculations for the white-background Leonardo films."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/cinematic_real"
DATA = OUT / "data"
ENV = Path("/opt/anaconda3/envs/protliginteract-full/bin")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENMM_CPU_THREADS"] = "2"
sys.path.insert(0, str(ROOT / "src"))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def run(args, cwd, log):
    t = time.perf_counter()
    p = subprocess.run(list(map(str, args)), cwd=cwd, capture_output=True, text=True, timeout=600)
    Path(log).write_text(p.stdout + "\n" + p.stderr)
    if p.returncode:
        raise RuntimeError(f"Command failed: {args}; see {log}")
    return time.perf_counter() - t


def prepare():
    """Amber14 charges on PDB-derived trypsin, including catalytic His57 HID."""
    from openmm import NonbondedForce, XmlSerializer, unit
    from openmm.app import ForceField, Modeller, NoCutoff, PDBFile

    DATA.mkdir(parents=True, exist_ok=True)
    source = ROOT / "examples/3ptb_ben/3PTB.pdb"
    lines = source.read_text().splitlines()
    (DATA / "3PTB.pdb").write_text(source.read_text())
    (DATA / "trypsin_heavy.pdb").write_text(
        "\n".join(x for x in lines if x.startswith(("ATOM  ", "TER", "SSBOND"))) + "\nEND\n"
    )
    (DATA / "crystal_benzamidine.pdb").write_text(
        "\n".join(x for x in lines if x.startswith("HETATM") and x[17:20] == "BEN") + "\nEND\n"
    )
    pdb = PDBFile(str(DATA / "trypsin_heavy.pdb"))
    ff = ForceField("amber14-all.xml")
    mod = Modeller(pdb.topology, pdb.positions)
    variants = ["HID" if r.name == "HIS" and r.id == "57" else None for r in mod.topology.residues()]
    random.seed(20260915)
    mod.addHydrogens(ff, pH=7.0, variants=variants)
    system = ff.createSystem(mod.topology, nonbondedMethod=NoCutoff, constraints=None)
    xyz = np.asarray(mod.positions.value_in_unit(unit.angstrom))
    atoms = list(mod.topology.atoms())
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    charges = np.array(
        [nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(len(atoms))]
    )
    with (DATA / "trypsin_amber.pdb").open("w") as f:
        PDBFile.writeFile(mod.topology, mod.positions, f, keepIds=True)
    (DATA / "trypsin_amber_system.xml").write_text(XmlSerializer.serialize(system))
    atom_meta = [
        dict(
            index=a.index,
            name=a.name,
            resname=a.residue.name,
            resid=a.residue.id,
            chain=a.residue.chain.id,
            element=a.element.symbol,
            atomic_number=a.element.atomic_number,
        )
        for a in atoms
    ]
    np.savez_compressed(DATA / "trypsin_amber.npz", xyz=xyz, charges=charges)
    dump(DATA / "trypsin_atoms.json", atom_meta)
    dump(
        DATA / "preparation.json",
        dict(
            source="RCSB PDB 3PTB; local example",
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            force_field="Amber14",
            pH=7.0,
            catalytic_histidine="HID57 (ND1 protonated; NE2 acceptor)",
            atoms=len(atoms),
            total_charge=float(charges.sum()),
            heavy_atoms="crystallographic; no coordinate morphing",
            scope="Apo trypsin for QM/MM; crystal inhibitor removed before enzyme model preparation. Waters and calcium removed.",
        ),
    )
    print("Prepared", len(atoms), "trypsin atoms", flush=True)


def docking():
    from rdkit import Chem
    from rdkit.Chem import AllChem


    work = OUT / "docking"
    work.mkdir(exist_ok=True)
    mol = Chem.MolFromSmiles("[NH2+]=C(N)c1ccccc1")
    # Generate independent initial geometry. Box alone uses the crystal ligand.
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=20260915)
    AllChem.MMFFOptimizeMolecule(mol)
    w = Chem.SDWriter(str(work / "benzamidinium_input.sdf"))
    w.write(mol)
    w.close()
    run(
        [
            ENV / "obabel",
            DATA / "trypsin_amber.pdb",
            "-O",
            work / "receptor.pdbqt",
            "-xr",
            "--partialcharge",
            "gasteiger",
        ],
        work,
        work / "receptor_preparation.log",
    )
    seconds = None
    if not (work / "poses.sdf").exists():
        seconds = run(
            [
                ENV / "smina",
                "--receptor",
                work / "receptor.pdbqt",
                "--ligand",
                work / "benzamidinium_input.sdf",
                "--autobox_ligand",
                DATA / "crystal_benzamidine.pdb",
                "--autobox_add",
                "8",
                "--exhaustiveness",
                "16",
                "--num_modes",
                "9",
                "--seed",
                "20260915",
                "--cpu",
                "2",
                "--out",
                work / "poses.sdf",
                "--log",
                work / "smina.log",
            ],
            work,
            work / "smina_console.log",
        )
    poses = list(Chem.SDMolSupplier(str(work / "poses.sdf"), removeHs=False))
    # Graph isomorphism, without rigid fitting, preserves docking displacement.
    crystal = Chem.MolFromPDBFile(str(DATA / "crystal_benzamidine.pdb"), removeHs=True)
    # Match by element/connectivity, ignoring aromatic bond perception in PDB.
    query = Chem.MolFromSmarts("[#7]~[#6](~[#7])~[#6]1~[#6]~[#6]~[#6]~[#6]~[#6]~1")
    matches = crystal.GetSubstructMatches(query, uniquify=False, maxMatches=10000)
    if not matches:
        raise RuntimeError("Cannot map crystal BEN atoms onto benzamidinium")
    ref = crystal.GetConformer().GetPositions()
    rows = []
    for i, pose in enumerate(poses):
        if pose is None:
            raise RuntimeError("Invalid SDF output")
        heavy = Chem.RemoveHs(pose)
        xyz = heavy.GetConformer().GetPositions()
        rmsd = min(float(np.sqrt(np.mean(np.sum((xyz - ref[list(m)]) ** 2, axis=1)))) for m in matches)
        score = float(pose.GetProp("minimizedAffinity"))
        rows.append(dict(rank=i + 1, score_kcal_mol=score, crystal_rmsd_A=rmsd))
        Chem.MolToPDBFile(pose, str(work / f"pose_{i+1:02d}.pdb"))
    with (work / "scores.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dump(
        work / "summary.json",
        dict(
            engine="smina 2020.12.10 / AutoDock Vina scoring",
            system="Trypsin / benzamidinium",
            pdb_id="3PTB",
            seed=20260915,
            exhaustiveness=16,
            cpu=2,
            ligand_charge=1,
            receptor="rigid; Amber-prepared hydrogens, Open Babel PDBQT",
            search_box="Crystal pocket + 8 Angstrom margin",
            runtime_seconds=seconds,
            poses=rows,
            scope="Pocket redocking; docking score is not a measured binding free energy; ranked poses are not a dynamical trajectory.",
        ),
    )
    print("Docking", rows, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "docking"])
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    {"prepare": prepare, "docking": docking}[a.stage]()


if __name__ == "__main__":
    main()
