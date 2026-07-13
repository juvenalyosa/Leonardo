#!/usr/bin/env python3
"""
Validate the Leonardo AmberTools -> ffxml ligand parameterization path on 3PTB/BEN.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from openmm import CustomExternalForce, LangevinMiddleIntegrator, Platform, app, unit
from openmm.app import Modeller, PDBFile, Simulation
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import resource_path
from .openmm_qmmm_runtime import export_forcefield_xml_from_amber_prmtop

PROTEIN_RESN = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "CYM", "CYX", "GLH", "HID", "HIE", "HIP", "LYN",
}
WATER_RESN = {"HOH", "WAT", "DOD"}
ION_RESN = {"NA", "K", "CL", "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI"}
BEN_SMILES = "[NH2+]=C(N)c1ccccc1"
RING_ATOM_NAMES = ("C1", "C2", "C3", "C4", "C5", "C6")
RING_BONDS = tuple(zip(RING_ATOM_NAMES, RING_ATOM_NAMES[1:] + RING_ATOM_NAMES[:1]))


def run_checked(cmd, cwd):
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={proc.returncode}: {' '.join(str(x) for x in cmd)}\n"
            f"STDERR:\n{proc.stderr[-1200:]}\nSTDOUT:\n{proc.stdout[-1200:]}"
        )
    return proc


def resolve_tool_path(name: str, explicit: str = "") -> str:
    raw = str(explicit or "").strip()
    if raw and os.path.isfile(raw):
        return str(Path(raw).resolve())
    hit = shutil.which(str(name))
    if hit:
        return str(Path(hit).resolve())
    env_bin = Path(sys.executable).resolve().parent
    candidates = [
        env_bin / str(name),
        env_bin / "wrapped_progs" / str(name),
    ]
    if str(name).lower() == "tleap":
        candidates.append(env_bin / "teLeap")
    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())
    raise FileNotFoundError(f"Could not resolve required executable: {name}")


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


def build_simulation_with_fallback(topology, system, integrator):
    attempts = []
    for name, properties in (
        ("CUDA", {"Precision": "mixed"}),
        ("OpenCL", {"Precision": "mixed"}),
        ("CPU", {}),
        ("Reference", {}),
    ):
        try:
            platform = Platform.getPlatformByName(name)
        except Exception as e:
            attempts.append(f"{name}:unavailable:{e}")
            continue
        try:
            if properties:
                sim = Simulation(topology, system, integrator, platform, dict(properties))
            else:
                sim = Simulation(topology, system, integrator, platform)
            return sim, str(name)
        except Exception as e:
            attempts.append(f"{name}:init_failed:{e}")
            continue
    raise RuntimeError(" ; ".join(attempts[-4:]) or "No compatible OpenMM platform is available.")


def choose_ben(topology: app.Topology):
    candidates = []
    for residue in topology.residues():
        is_protein = residue.name in PROTEIN_RESN
        is_water = residue.name in WATER_RESN
        is_ion = residue.name in ION_RESN
        atom_count = sum(1 for _ in residue.atoms())
        if not is_protein and not is_water and not is_ion and atom_count >= 6:
            candidates.append((residue.chain.id, str(residue.id), residue.name, atom_count))
    ben = [row for row in candidates if row[2] == "BEN"]
    if ben:
        return ben[0]
    if candidates:
        return sorted(candidates, key=lambda row: row[3], reverse=True)[0]
    raise RuntimeError("No suitable ligand found in input PDB.")


def extract_component_pdb_block(pdb_path: Path, lig_chain: str, lig_resi: str, lig_resn: str) -> str:
    lines = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21].strip()
            resi = line[22:26].strip()
            resn = line[17:20].strip()
            if chain == lig_chain and resi == str(lig_resi) and resn == lig_resn:
                lines.append(line.rstrip("\n"))
    if not lines:
        raise RuntimeError(f"Could not extract {lig_resn} {lig_chain}:{lig_resi} from {pdb_path}.")
    return "\n".join(lines) + "\nEND\n"


def rebuild_ben_with_hydrogens(lig_pdb_block: str, lig_chain: str, lig_resi: str, lig_resn: str):
    template = Chem.MolFromSmiles(BEN_SMILES)
    if template is None:
        raise RuntimeError("Failed to build BEN template SMILES.")
    crystal = Chem.MolFromPDBBlock(lig_pdb_block, removeHs=False)
    if crystal is None:
        raise RuntimeError("RDKit failed to parse BEN crystal block.")
    mapped = AllChem.AssignBondOrdersFromTemplate(template, crystal)
    mapped_h = Chem.AddHs(mapped, addCoords=True)
    h_count = 1
    for idx, atom in enumerate(mapped_h.GetAtoms(), start=1):
        info = atom.GetPDBResidueInfo()
        atom_name = ""
        if info is not None and atom.GetAtomicNum() != 1:
            atom_name = str(info.GetName() or "").strip()
        if not atom_name:
            if atom.GetAtomicNum() == 1:
                atom_name = f"H{h_count:02d}"
                h_count += 1
            else:
                atom_name = f"{atom.GetSymbol()}{idx}"
        atom.SetMonomerInfo(
            Chem.AtomPDBResidueInfo(
                f"{atom_name:<4s}",
                serialNumber=int(idx),
                altLoc="",
                residueName=f"{lig_resn:<3s}",
                residueNumber=int(lig_resi),
                chainId=f"{lig_chain:1s}",
                insertionCode="",
                occupancy=1.0,
                tempFactor=0.0,
            )
        )
    return mapped_h


def write_component_files(mol, out_pdb: Path, out_sdf: Path):
    pdb_block = Chem.MolToPDBBlock(mol)
    out_pdb.write_text(str(pdb_block), encoding="utf-8")
    writer = Chem.SDWriter(str(out_sdf))
    writer.write(mol)
    writer.close()


def add_minimization_restraints(system, topology, positions, lig_chain, lig_resi, lig_resn):
    force = CustomExternalForce("0.5*k_scale*k_ref*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    force.addGlobalParameter("k_scale", 1.0)
    force.addPerParticleParameter("k_ref")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    counts = {"protein_backbone": 0, "protein_sidechain": 0, "ligand_heavy": 0}
    for atom in topology.atoms():
        pos = positions[atom.index]
        res = atom.residue
        elem = str(atom.element.symbol if atom.element is not None else "").strip().upper()
        if elem == "H":
            continue
        if res.name in PROTEIN_RESN:
            if atom.name in ("N", "CA", "C", "O", "OXT"):
                k_ref = 2400.0
                counts["protein_backbone"] += 1
            else:
                k_ref = 900.0
                counts["protein_sidechain"] += 1
        elif res.chain.id == lig_chain and str(res.id) == str(lig_resi) and res.name == lig_resn:
            k_ref = 700.0
            counts["ligand_heavy"] += 1
        else:
            continue
        try:
            x0, y0, z0 = pos.value_in_unit(unit.nanometer)
        except Exception:
            x0 = float(pos[0] / unit.nanometer)
            y0 = float(pos[1] / unit.nanometer)
            z0 = float(pos[2] / unit.nanometer)
        force.addParticle(
            int(atom.index),
            [float(k_ref), float(x0), float(y0), float(z0)],
        )
    system.addForce(force)
    return force, counts


def ring_bond_metrics(topology, positions, lig_chain, lig_resi, lig_resn):
    coord_map = {}
    for atom in topology.atoms():
        res = atom.residue
        if res.chain.id != lig_chain or str(res.id) != str(lig_resi) or res.name != lig_resn:
            continue
        coord_map[str(atom.name).strip().upper()] = positions[atom.index]
    lengths = {}
    for a1, a2 in RING_BONDS:
        p1 = coord_map.get(a1)
        p2 = coord_map.get(a2)
        if p1 is None or p2 is None:
            continue
        try:
            v1 = p1.value_in_unit(unit.angstrom)
            v2 = p2.value_in_unit(unit.angstrom)
            dx = float(v1[0] - v2[0])
            dy = float(v1[1] - v2[1])
            dz = float(v1[2] - v2[2])
        except Exception:
            dx = float((p1[0] - p2[0]) / unit.angstrom)
            dy = float((p1[1] - p2[1]) / unit.angstrom)
            dz = float((p1[2] - p2[2]) / unit.angstrom)
        lengths[f"{a1}-{a2}"] = math.sqrt(dx * dx + dy * dy + dz * dz)
    vals = list(lengths.values())
    return {
        "bond_lengths_ang": dict(lengths),
        "min_ang": float(min(vals)) if vals else float("nan"),
        "max_ang": float(max(vals)) if vals else float("nan"),
        "spread_ang": float(max(vals) - min(vals)) if vals else float("nan"),
        "mean_ang": float(sum(vals) / len(vals)) if vals else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description="Validate Leonardo BEN parameterization on 3PTB.")
    ap.add_argument("--pdb", default=str(resource_path("examples/3ptb_ben/3PTB.pdb")))
    ap.add_argument("--workdir", default="validation_3ptb_ben")
    ap.add_argument("--antechamber", default=os.environ.get("ANTECHAMBER_BIN", "antechamber"))
    ap.add_argument("--parmchk2", default=os.environ.get("PARMCHK2_BIN", "parmchk2"))
    ap.add_argument("--tleap", default=os.environ.get("TLEAP_BIN", "tleap"))
    ap.add_argument("--min-iters", type=int, default=600)
    args = ap.parse_args()

    pdb_path = Path(args.pdb).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    pdb = PDBFile(str(pdb_path))
    antechamber_bin = resolve_tool_path("antechamber", args.antechamber)
    parmchk2_bin = resolve_tool_path("parmchk2", args.parmchk2)
    tleap_bin = resolve_tool_path("tleap", args.tleap)
    lig_chain, lig_resi, lig_resn, lig_atoms = choose_ben(pdb.topology)
    if lig_resn != "BEN":
        raise RuntimeError(f"Expected BEN in 3PTB validation, got {lig_resn}.")

    ligand_block = extract_component_pdb_block(pdb_path, lig_chain, lig_resi, lig_resn)
    ben_mol = rebuild_ben_with_hydrogens(ligand_block, lig_chain, lig_resi, lig_resn)
    ben_pdb_path = workdir / "component_hydrated.pdb"
    ben_sdf_path = workdir / "component_hydrated.sdf"
    write_component_files(ben_mol, ben_pdb_path, ben_sdf_path)

    out_mol2 = workdir / "ligand_gaff2.mol2"
    out_frcmod = workdir / "ligand_gaff2.frcmod"
    out_prmtop = workdir / "ligand.prmtop"
    out_inpcrd = workdir / "ligand.inpcrd"
    out_ffxml = workdir / "ligand_openmm_ff.xml"
    leap_in = workdir / "tleap.in"
    leap_script = "\n".join(
        [
            "source leaprc.gaff2",
            f"LIG = loadmol2 {out_mol2.name}",
            f"loadamberparams {out_frcmod.name}",
            f"saveamberparm LIG {out_prmtop.name} {out_inpcrd.name}",
            "quit",
            "",
        ]
    )
    leap_in.write_text(leap_script, encoding="utf-8")

    run_checked(
        [
            antechamber_bin,
            "-i", str(ben_sdf_path),
            "-fi", "sdf",
            "-o", str(out_mol2),
            "-fo", "mol2",
            "-at", "gaff2",
            "-c", "bcc",
            "-nc", "1",
            "-dr", "no",
            "-s", "2",
        ],
        cwd=workdir,
    )
    run_checked(
        [
            parmchk2_bin,
            "-i", str(out_mol2),
            "-f", "mol2",
            "-o", str(out_frcmod),
        ],
        cwd=workdir,
    )
    run_checked([tleap_bin, "-f", str(leap_in)], cwd=workdir)

    export_out = export_forcefield_xml_from_amber_prmtop(
        prmtop_file=str(out_prmtop),
        out_xml=str(out_ffxml),
        residue_name_hint=lig_resn,
    )
    if not bool(export_out.get("ok", False)):
        raise RuntimeError(str(export_out.get("error", "ffxml export failed.")))

    modeller = Modeller(pdb.topology, pdb.positions)
    atoms_to_delete = []
    for residue in modeller.topology.residues():
        if residue.name not in PROTEIN_RESN:
            atoms_to_delete.extend(list(residue.atoms()))
    if atoms_to_delete:
        modeller.delete(atoms_to_delete)
    protein_ff = app.ForceField("amber14/protein.ff14SB.xml")
    modeller.addHydrogens(protein_ff, pH=7.4)
    ben_pdb = PDBFile(str(ben_pdb_path))
    modeller.add(ben_pdb.topology, ben_pdb.positions)
    prepared_pdb = workdir / "3PTB_BEN_prepared.pdb"
    with prepared_pdb.open("w", encoding="utf-8") as fh:
        PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)

    ff = app.ForceField("amber14/protein.ff14SB.xml", str(out_ffxml))
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=False,
    )
    restraint_force, restraint_counts = add_minimization_restraints(
        system,
        modeller.topology,
        modeller.positions,
        lig_chain=lig_chain,
        lig_resi=lig_resi,
        lig_resn=lig_resn,
    )

    integrator = LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picoseconds,
    )
    simulation, platform_name = build_simulation_with_fallback(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    state0 = simulation.context.getState(getEnergy=True, getPositions=True)
    e0 = float(state0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    ring0 = ring_bond_metrics(modeller.topology, state0.getPositions(), lig_chain, lig_resi, lig_resn)

    simulation.context.setParameter("k_scale", 1.0)
    simulation.minimizeEnergy(maxIterations=max(50, int(args.min_iters // 2)))
    simulation.context.setParameter("k_scale", 0.2)
    simulation.minimizeEnergy(maxIterations=max(50, int(args.min_iters // 4)))
    simulation.context.setParameter("k_scale", 0.0)
    simulation.minimizeEnergy(maxIterations=max(50, int(args.min_iters // 4)))

    statef = simulation.context.getState(getEnergy=True, getPositions=True)
    ef = float(statef.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    ringf = ring_bond_metrics(modeller.topology, statef.getPositions(), lig_chain, lig_resi, lig_resn)

    final_pdb = workdir / "3PTB_BEN_minimized.pdb"
    with final_pdb.open("w", encoding="utf-8") as fh:
        PDBFile.writeFile(simulation.topology, statef.getPositions(), fh, keepIds=True)

    mol2_charge_sum = 0.0
    with out_mol2.open("r", encoding="utf-8", errors="ignore") as fh:
        in_atom_block = False
        for raw in fh:
            line = raw.strip()
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom_block:
                break
            if not in_atom_block or not line:
                continue
            parts = line.split()
            if len(parts) >= 9:
                mol2_charge_sum += float(parts[8])

    audit = json.loads((Path(str(out_ffxml) + ".audit.json")).read_text(encoding="utf-8"))
    summary = {
        "pdb": str(pdb_path),
        "ligand": {"chain": lig_chain, "resi": lig_resi, "resn": lig_resn, "crystal_atoms": int(lig_atoms)},
        "platform": str(platform_name),
        "restraints": dict(restraint_counts),
        "energies_kj_mol": {
            "initial": float(e0),
            "final": float(ef),
            "delta": float(ef - e0),
        },
        "ring_before": dict(ring0),
        "ring_after": dict(ringf),
        "mol2_charge_sum_e": float(mol2_charge_sum),
        "ffxml_audit": dict(audit),
        "files": {
            "prepared_pdb": str(prepared_pdb),
            "final_pdb": str(final_pdb),
            "mol2": str(out_mol2),
            "frcmod": str(out_frcmod),
            "prmtop": str(out_prmtop),
            "inpcrd": str(out_inpcrd),
            "ffxml": str(out_ffxml),
        },
    }
    summary_path = workdir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Ligand: {lig_resn} {lig_chain}:{lig_resi}")
    print(f"Platform: {platform_name}")
    print(
        "Amber/ffxml terms:"
        f" atoms={audit['counts']['atoms']}"
        f" bonds={audit['counts']['bond_terms']}"
        f" angles={audit['counts']['angle_terms']}"
        f" torsions={audit['counts']['torsion_terms']}"
        f" charge_sum={float(audit['charge']['sum_e']):.4f}e"
        f" roundtrip={'ok' if bool((audit.get('roundtrip', {}) or {}).get('ok', False)) else 'failed'}"
    )
    print(
        f"Energy: initial={e0:.4f} kJ/mol final={ef:.4f} kJ/mol delta={ef - e0:.4f} kJ/mol"
    )
    print(
        f"BEN ring: mean_before={float(ring0['mean_ang']):.4f} A mean_after={float(ringf['mean_ang']):.4f} A "
        f"spread_after={float(ringf['spread_ang']):.4f} A"
    )
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
