#!/usr/bin/env python3
"""
Prepare 3PTB (trypsin) + BEN ligand and run a short OpenMM MD trajectory.
Outputs are intended for PyMOL trajectory testing.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from openff.toolkit.topology import Molecule
from openmm import LangevinMiddleIntegrator, Platform, app, unit
from openmm.app import CheckpointReporter, DCDReporter, Modeller, PDBFile, Simulation, StateDataReporter
from openmmforcefields.generators import SystemGenerator
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import resource_path

PROTEIN_RESN = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ASH",
    "CYM",
    "CYX",
    "GLH",
    "HID",
    "HIE",
    "HIP",
    "LYN",
}
WATER_RESN = {"HOH", "WAT", "DOD"}
ION_RESN = {"NA", "K", "CL", "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI"}


def pick_openmm_platform():
    for name, properties in (
        ("CUDA", {"Precision": "mixed"}),
        ("OpenCL", {"Precision": "mixed"}),
        ("CPU", {}),
    ):
        try:
            return Platform.getPlatformByName(name), dict(properties), str(name)
        except Exception:
            continue
    raise RuntimeError("No usable OpenMM platform was found.")


def choose_ligand(topology: app.Topology):
    candidates = []
    for residue in topology.residues():
        is_protein = residue.name in PROTEIN_RESN
        is_water = residue.name in WATER_RESN
        is_ion = residue.name in ION_RESN
        atom_count = sum(1 for _ in residue.atoms())
        if not is_protein and not is_water and not is_ion and atom_count >= 6:
            candidates.append((residue.chain.id, str(residue.id), residue.name, atom_count))

    if not candidates:
        raise RuntimeError("No suitable non-protein ligand found in input PDB")

    ben = [c for c in candidates if c[2] == "BEN"]
    if ben:
        return ben[0], candidates
    return sorted(candidates, key=lambda x: x[3], reverse=True)[0], candidates


def extract_ligand_pdb_block(pdb_path: Path, lig_chain: str, lig_resi: str, lig_resn: str) -> str:
    lines = []
    with pdb_path.open("r") as fh:
        for line in fh:
            if not (line.startswith("HETATM") or line.startswith("ATOM")):
                continue
            chain = line[21].strip()
            resi = line[22:26].strip()
            resn = line[17:20].strip()
            if chain == lig_chain and resi == str(lig_resi) and resn == lig_resn:
                lines.append(line.rstrip("\n"))
    if not lines:
        raise RuntimeError(f"Could not extract ligand {lig_resn} {lig_chain}:{lig_resi} from {pdb_path.name}")
    return "\n".join(lines) + "\nEND\n"


def rebuild_ben_with_hydrogens(lig_pdb_block: str, lig_chain: str, lig_resi: str, lig_resn: str):
    # Protonated benzamidine for physiological pH.
    template = Chem.MolFromSmiles("[NH2+]=C(N)c1ccccc1")
    if template is None:
        raise RuntimeError("Failed to build BEN template SMILES")

    crystal = Chem.MolFromPDBBlock(lig_pdb_block, removeHs=False)
    if crystal is None:
        raise RuntimeError("RDKit failed to parse ligand block from PDB")

    mapped = AllChem.AssignBondOrdersFromTemplate(template, crystal)
    mapped_h = Chem.AddHs(mapped, addCoords=True)

    for idx, atom in enumerate(mapped_h.GetAtoms(), start=1):
        info = atom.GetPDBResidueInfo()
        if info:
            atom_name = info.GetName().strip()
        elif atom.GetAtomicNum() == 1:
            atom_name = f"H{idx:02d}"
        else:
            atom_name = f"{atom.GetSymbol()}{idx}"

        atom_info = Chem.AtomPDBResidueInfo(
            f"{atom_name:<4s}",
            serialNumber=idx,
            altLoc="",
            residueName=f"{lig_resn:<3s}",
            residueNumber=int(lig_resi),
            chainId=f"{lig_chain:1s}",
            insertionCode="",
            occupancy=1.0,
            tempFactor=0.0,
        )
        atom.SetMonomerInfo(atom_info)

    lig_with_h_pdb = Chem.MolToPDBBlock(mapped_h)
    lig_pdb = PDBFile(io.StringIO(lig_with_h_pdb))
    return mapped_h, lig_pdb


def main():
    workdir = Path.cwd()
    pdb_path = resource_path("examples/3ptb_ben/3PTB.pdb")

    if not pdb_path.exists():
        raise FileNotFoundError(f"Missing input PDB: {pdb_path}")

    out_prefix = workdir / "3PTB_BEN_openmm"
    prepared_pdb = Path(str(out_prefix) + "_prepared.pdb")
    final_pdb = Path(str(out_prefix) + "_final.pdb")
    dcd_path = Path(str(out_prefix) + ".dcd")
    chk_path = Path(str(out_prefix) + ".chk")

    print(f"Loading {pdb_path.name} ...")
    pdb = PDBFile(str(pdb_path))

    chosen, candidates = choose_ligand(pdb.topology)
    lig_chain, lig_resi, lig_resn, lig_natoms = chosen
    print("Ligand candidates:", candidates)
    print(f"Selected ligand: {lig_resn} chain {lig_chain} resi {lig_resi} ({lig_natoms} atoms)")

    if lig_resn != "BEN":
        raise RuntimeError(f"This benchmark script expects BEN but detected {lig_resn}")

    # Keep protein only for initial protonation.
    modeller = Modeller(pdb.topology, pdb.positions)
    atoms_to_delete = []
    for residue in modeller.topology.residues():
        if residue.name not in PROTEIN_RESN:
            atoms_to_delete.extend(list(residue.atoms()))
    if atoms_to_delete:
        print(f"Removing {len(atoms_to_delete)} non-protein atoms from protein model")
        modeller.delete(atoms_to_delete)

    protein_ff = app.ForceField("amber14/protein.ff14SB.xml")
    print("Adding hydrogens to protein at pH 7.4 ...")
    modeller.addHydrogens(protein_ff, pH=7.4)

    # Rebuild BEN from crystal coordinates with explicit hydrogens and protonation.
    lig_block = extract_ligand_pdb_block(pdb_path, lig_chain, lig_resi, lig_resn)
    rdkit_ben, ben_pdb = rebuild_ben_with_hydrogens(lig_block, lig_chain, lig_resi, lig_resn)
    ben_h_count = sum(1 for a in rdkit_ben.GetAtoms() if a.GetAtomicNum() == 1)
    print(f"Rebuilt BEN with explicit hydrogens: {rdkit_ben.GetNumAtoms()} atoms ({ben_h_count} H)")

    print("Merging BEN into protonated protein topology ...")
    modeller.add(ben_pdb.topology, ben_pdb.positions)

    offmol = Molecule.from_smiles("[NH2+]=C(N)c1ccccc1", allow_undefined_stereo=True)
    offmol.name = lig_resn

    print("Initializing system generator (protein ff14SB + GAFF 2.11 ligand) ...")
    system_generator = SystemGenerator(
        forcefields=["amber14/protein.ff14SB.xml"],
        small_molecule_forcefield="gaff-2.11",
        molecules=[offmol],
        forcefield_kwargs={
            "constraints": app.HBonds,
            "rigidWater": True,
            "removeCMMotion": False,
        },
        nonperiodic_forcefield_kwargs={"nonbondedMethod": app.NoCutoff},
    )

    print(f"Writing prepared topology: {prepared_pdb.name}")
    with prepared_pdb.open("w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    print("Creating OpenMM System ...")
    system = system_generator.create_system(modeller.topology)

    integrator = LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picoseconds,
    )

    platform, platform_properties, platform_name = pick_openmm_platform()
    print(f"Using OpenMM platform: {platform_name}")
    if platform_properties:
        simulation = Simulation(modeller.topology, system, integrator, platform, platform_properties)
    else:
        simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)

    print("Minimizing energy ...")
    simulation.minimizeEnergy(maxIterations=1000)

    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)

    if dcd_path.exists():
        dcd_path.unlink()
    if chk_path.exists():
        chk_path.unlink()

    nvt_steps = 1000
    print(f"Running short NVT equilibration: {nvt_steps} steps ...")
    simulation.step(nvt_steps)

    simulation.reporters.append(DCDReporter(str(dcd_path), 100))
    simulation.reporters.append(CheckpointReporter(str(chk_path), 1000))
    simulation.reporters.append(
        StateDataReporter(
            sys.stdout,
            1000,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
    )

    nsteps = 5000  # 5 ps at 1 fs
    print(f"Running production dynamics: {nsteps} steps ...")
    simulation.step(nsteps)

    state = simulation.context.getState(getPositions=True)
    with final_pdb.open("w") as f:
        PDBFile.writeFile(simulation.topology, state.getPositions(), f, keepIds=True)

    print("Done.")
    print(f"Prepared PDB: {prepared_pdb}")
    print(f"Trajectory DCD: {dcd_path}")
    print(f"Checkpoint: {chk_path}")
    print(f"Final PDB: {final_pdb}")


if __name__ == "__main__":
    main()
