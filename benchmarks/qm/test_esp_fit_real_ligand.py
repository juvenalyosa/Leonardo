#!/usr/bin/env python3
"""
End-to-end real-ligand ESP-fit test (Psi4 + MOL2 charge rewrite).

Workflow:
1) Load ligand (SDF/MOL2/PDB)
2) Convert to MOL2 (Open Babel if needed)
3) Generate ESP grid around ligand atoms
4) Run Psi4 GRID_ESP + constrained least-squares fit of atomic charges
5) Write new charged MOL2 and validate mapping
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from qm_smoke_tests import (
    psi4_generate_esp_grid,
    psi4_make_script,
    psi4_molecule_block,
    psi4_write_charged_mol2,
)


def mol2_atom_type_to_element(atom_type: str, atom_name: str) -> str:
    at = str(atom_type or "").strip()
    nm = str(atom_name or "").strip()
    if not at:
        at = nm
    head = at.split(".")[0].strip()
    head = re.sub(r"[^A-Za-z]", "", head)
    if not head:
        head = re.sub(r"[^A-Za-z]", "", nm)
    if not head:
        return "C"

    up = head.upper()
    if up.startswith("CL"):
        return "CL"
    if up.startswith("BR"):
        return "BR"
    if up.startswith("NA"):
        return "NA"
    if up.startswith("MG"):
        return "MG"
    if up.startswith("CA"):
        return "CA"
    if up.startswith("ZN"):
        return "ZN"
    if up.startswith("FE"):
        return "FE"
    if up.startswith("CU"):
        return "CU"
    if up.startswith("MN"):
        return "MN"

    if len(head) == 1:
        return head.upper()
    return (head[0] + head[1:].lower()).upper()


def collect_atoms_from_mol2(path: Path) -> list[dict]:
    atoms: list[dict] = []
    section = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("@<TRIPOS>"):
                section = line.strip().upper()
                continue
            if section != "@<TRIPOS>ATOM" or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                idx = int(parts[0])
                name = str(parts[1]).strip() or f"A{idx}"
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                atom_type = str(parts[5]).strip()
            except Exception:
                continue
            el = mol2_atom_type_to_element(atom_type, name)
            atoms.append(
                {
                    "index": idx,
                    "name": name,
                    "element": el,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
    atoms.sort(key=lambda a: int(a["index"]))
    return atoms


def infer_formal_charge(input_path: Path) -> int | None:
    try:
        from rdkit import Chem
    except Exception:
        return None

    p = str(input_path)
    ext = input_path.suffix.lower()
    mol = None
    try:
        if ext == ".sdf":
            supp = Chem.SDMolSupplier(p, removeHs=False)
            mol = supp[0] if supp and len(supp) > 0 else None
        elif ext == ".mol2":
            mol = Chem.MolFromMol2File(p, removeHs=False)
        elif ext in {".mol", ".mdl"}:
            mol = Chem.MolFromMolFile(p, removeHs=False)
        elif ext in {".pdb", ".ent"}:
            mol = Chem.MolFromPDBFile(p, removeHs=False)
    except Exception:
        mol = None

    if mol is None:
        return None
    try:
        return int(Chem.GetFormalCharge(mol))
    except Exception:
        return None


def ensure_mol2(input_ligand: Path, out_mol2: Path, obabel_bin: str | None) -> tuple[bool, str]:
    if input_ligand.suffix.lower() == ".mol2":
        shutil.copy2(str(input_ligand), str(out_mol2))
        return True, "input_mol2"

    if not obabel_bin:
        return False, "obabel_not_found"

    # First preserve geometry and just add hydrogens.
    cmd1 = [obabel_bin, str(input_ligand), "-O", str(out_mol2), "-h"]
    p1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p1.returncode == 0 and out_mol2.is_file() and out_mol2.stat().st_size > 0:
        return True, "obabel_addh"

    # Fallback: generate 3D with hydrogens.
    cmd2 = [obabel_bin, str(input_ligand), "-O", str(out_mol2), "--gen3d", "-h"]
    p2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p2.returncode == 0 and out_mol2.is_file() and out_mol2.stat().st_size > 0:
        return True, "obabel_gen3d_addh"

    msg = (
        f"obabel conversion failed. cmd1_rc={p1.returncode}, cmd2_rc={p2.returncode}; "
        f"stderr1={p1.stderr[-300:]}; stderr2={p2.stderr[-300:]}"
    )
    return False, msg


def main() -> int:
    ap = argparse.ArgumentParser(description="Run real-ligand Psi4 ESP-fit test.")
    ap.add_argument(
        "--ligand",
        default="examples/3ptb_ben/BEN.sdf",
        help="Input ligand file (SDF/MOL2/PDB).",
    )
    ap.add_argument("--out-dir", default="benchmarks/qm/esp_real", help="Output directory.")
    ap.add_argument("--method", default="hf", help="QM method (e.g., hf, wb97x-d).")
    ap.add_argument("--basis", default="6-31G*", help="Basis set.")
    ap.add_argument("--charge", type=int, default=None, help="Total charge override.")
    ap.add_argument("--multiplicity", type=int, default=1, help="Spin multiplicity.")
    ap.add_argument("--points-per-atom", type=int, default=32, help="ESP grid points per atom scale factor.")
    ap.add_argument("--threads", type=int, default=1, help="Psi4 threads.")
    ap.add_argument("--memory-gb", type=int, default=2, help="Psi4 memory in GB.")
    ap.add_argument("--timeout-sec", type=int, default=1800, help="Psi4 timeout seconds.")
    ap.add_argument("--psi4-bin", default="psi4", help="Psi4 executable.")
    ap.add_argument("--obabel-bin", default="obabel", help="Open Babel executable.")
    args = ap.parse_args()

    ligand = Path(args.ligand).resolve()
    if not ligand.is_file():
        print(f"ERROR: Ligand file not found: {ligand}")
        return 1

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    job_dir = out_root / (ligand.stem + "_espfit")
    job_dir.mkdir(parents=True, exist_ok=True)

    obabel_bin = shutil.which(args.obabel_bin)
    psi4_bin = shutil.which(args.psi4_bin)
    if not psi4_bin:
        print("ERROR: psi4 executable not found in PATH")
        return 1

    mol2_in = job_dir / "ligand_input.mol2"
    ok_m2, m2_note = ensure_mol2(ligand, mol2_in, obabel_bin)
    if not ok_m2:
        print(f"ERROR: Failed to prepare MOL2: {m2_note}")
        return 1

    atoms = collect_atoms_from_mol2(mol2_in)
    if not atoms:
        print("ERROR: No atoms parsed from prepared MOL2")
        return 1

    auto_charge = infer_formal_charge(ligand)
    total_charge = int(args.charge if args.charge is not None else (auto_charge if auto_charge is not None else 0))
    multiplicity = int(max(1, args.multiplicity))

    grid = psi4_generate_esp_grid(
        atoms,
        points_per_atom=max(8, int(args.points_per_atom)),
        scales=(1.4, 1.8),
        max_points=12000,
    )
    if grid.shape[0] < 20:
        print(f"ERROR: ESP grid too small: {grid.shape[0]} points")
        return 1
    grid_file = job_dir / "esp_grid.dat"
    np.savetxt(grid_file, grid, fmt="%.8f")
    # Psi4 GRID_ESP reads grid.dat by default.
    np.savetxt(job_dir / "grid.dat", grid, fmt="%.8f")

    script = psi4_make_script(
        {
            "operation": "esp_fit",
            "method": str(args.method),
            "basis": str(args.basis),
            "molecule_block": psi4_molecule_block(atoms, charge=total_charge, multiplicity=multiplicity),
            "memory_gb": int(max(1, args.memory_gb)),
            "threads": int(max(1, args.threads)),
            "geom_maxiter": 80,
            "implicit": False,
            "solvent": "Water",
            "esp_grid_file": "grid.dat",
            "charge": int(total_charge),
        }
    )

    psi4_in = job_dir / "psi4_input.dat"
    psi4_out = job_dir / "psi4_output.out"
    psi4_in.write_text(script, encoding="utf-8")

    cmdline = [psi4_bin, "-n", str(max(1, int(args.threads))), str(psi4_in), str(psi4_out)]
    proc = subprocess.run(
        cmdline,
        cwd=str(job_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(60, int(args.timeout_sec)),
        check=False,
    )

    result_json = job_dir / "psi4_result.json"
    if proc.returncode != 0 or not result_json.is_file():
        print(f"ERROR: Psi4 execution failed (rc={proc.returncode})")
        if proc.stderr.strip():
            print(proc.stderr[-1000:])
        return 1

    result = json.loads(result_json.read_text(encoding="utf-8"))
    if not bool(result.get("ok", False)):
        print("ERROR: Psi4 result marked failed")
        print(str(result.get("error", "unknown")))
        return 1

    charges_file = job_dir / str(result.get("esp_fit_charges_file", "esp_fit_charges.dat"))
    if not charges_file.is_file():
        print(f"ERROR: Missing fitted charges file: {charges_file}")
        return 1

    charges = np.loadtxt(charges_file, dtype=float).reshape(-1)
    out_mol2 = job_dir / "ligand_espfit.mol2"
    ok_write, msg_write = psi4_write_charged_mol2(mol2_in, charges, out_mol2)
    if not ok_write:
        print(f"ERROR: Failed writing charged MOL2: {msg_write}")
        return 1

    report = {
        "ligand": str(ligand),
        "prepared_mol2": str(mol2_in),
        "out_mol2": str(out_mol2),
        "psi4_input": str(psi4_in),
        "psi4_output": str(psi4_out),
        "charge_model": {
            "requested_total_charge": total_charge,
            "multiplicity": multiplicity,
            "n_atoms": int(len(atoms)),
            "n_grid_points": int(grid.shape[0]),
            "n_fitted_charges": int(charges.shape[0]),
            "fitted_charge_sum": float(np.sum(charges)),
            "esp_fit_rms_au": float(result.get("esp_fit_rms_au", np.nan)),
            "energy_hartree": float(result.get("energy_hartree", np.nan)),
        },
        "prep_note": m2_note,
    }
    rep_path = job_dir / "espfit_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("ESP-fit real ligand test: PASS")
    print(f"  Job dir: {job_dir}")
    print(f"  Ligand: {ligand.name}")
    print(f"  Method/Basis: {args.method}/{args.basis}")
    print(f"  Atoms: {len(atoms)} | Grid points: {grid.shape[0]}")
    print(f"  Fitted charges: {charges.shape[0]} | Sum(q): {float(np.sum(charges)):.6f}")
    print(f"  ESP RMS (au): {float(result.get('esp_fit_rms_au', np.nan)):.6e}")
    print(f"  Charged MOL2: {out_mol2}")
    print(f"  Report: {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
