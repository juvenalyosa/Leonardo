#!/usr/bin/env python3
"""
QM smoke tests for ProtLigInteract Psi4 workflow helpers.

This script is intentionally standalone (no PyMOL/Qt imports) so it can be
run in terminal before wiring behavior into the plugin UI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def psi4_element_token(element: str) -> str:
    e = str(element or "").strip().upper()
    if not e:
        return "C"
    if len(e) == 1:
        return e
    return e[0] + e[1:].lower()


def psi4_vdw_radius(element: str) -> float:
    e = str(element or "").strip().upper()
    table = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "F": 1.47,
        "P": 1.80,
        "S": 1.80,
        "CL": 1.75,
        "BR": 1.85,
        "I": 1.98,
        "NA": 2.27,
        "K": 2.75,
        "MG": 1.73,
        "CA": 2.31,
        "ZN": 1.39,
        "FE": 1.56,
        "CU": 1.40,
        "MN": 1.61,
    }
    if e in table:
        return float(table[e])
    if e and e[0] in table:
        return float(table[e[0]])
    return 1.70


def psi4_generate_esp_grid(
    atoms: Iterable[dict],
    points_per_atom: int = 36,
    scales: tuple[float, float] = (1.4, 1.8),
    max_points: int = 12000,
) -> np.ndarray:
    arr = np.array(
        [[float(a["x"]), float(a["y"]), float(a["z"])] for a in atoms], dtype=float
    )
    elems = [str(a["element"]).strip().upper() for a in atoms]
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)

    radii = np.array([psi4_vdw_radius(e) for e in elems], dtype=float)
    ppa = max(8, int(points_per_atom))

    def _sphere(n: int) -> list[tuple[float, float, float]]:
        golden = math.pi * (3.0 - math.sqrt(5.0))
        out = []
        n = max(4, int(n))
        for k in range(n):
            z = 1.0 - 2.0 * (float(k) + 0.5) / float(n)
            r = math.sqrt(max(0.0, 1.0 - z * z))
            phi = golden * float(k)
            out.append((r * math.cos(phi), r * math.sin(phi), z))
        return out

    pts: list[tuple[float, float, float]] = []
    nat = int(len(elems))
    for i in range(nat):
        c = arr[i]
        ri = radii[i]
        for sc in scales:
            shell = float(sc) * float(ri)
            n_shell = max(12, int(round(float(ppa) * float(sc))))
            for ux, uy, uz in _sphere(n_shell):
                p = c + shell * np.array([ux, uy, uz], dtype=float)
                keep = True
                for j in range(nat):
                    d = float(np.linalg.norm(p - arr[j]))
                    if d < (1.15 * float(radii[j])):
                        keep = False
                        break
                if keep:
                    pts.append((float(p[0]), float(p[1]), float(p[2])))
                    if len(pts) >= int(max_points):
                        break
            if len(pts) >= int(max_points):
                break
        if len(pts) >= int(max_points):
            break

    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.array(pts, dtype=float)


def psi4_molecule_block(atoms: Iterable[dict], charge: int = 0, multiplicity: int = 1) -> str:
    lines = [f"{int(charge)} {int(max(1, multiplicity))}"]
    for a in atoms:
        sym = psi4_element_token(a.get("element", a.get("symbol", "C")))
        x = float(a["x"])
        y = float(a["y"])
        z = float(a["z"])
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    lines.append("units angstrom")
    lines.append("no_reorient")
    lines.append("no_com")
    return "\n".join(lines)


def psi4_make_script(job: dict) -> str:
    op = str(job.get("operation", "optimize") or "optimize").strip().lower()
    method = str(job.get("method", "wb97x-d") or "wb97x-d").strip()
    basis = str(job.get("basis", "def2-SVP") or "def2-SVP").strip()
    mol_block = str(job.get("molecule_block", "") or "").strip()
    mem_gb = int(max(1, int(job.get("memory_gb", 8) or 8)))
    threads = int(max(1, int(job.get("threads", 1) or 1)))
    geom_maxiter = int(max(20, int(job.get("geom_maxiter", 100) or 100)))
    use_pcm = bool(job.get("implicit", False))
    solvent = str(job.get("solvent", "Water") or "Water").strip()
    irc_direction = str(job.get("irc_direction", "BOTH") or "BOTH").strip().upper()
    irc_points = int(max(4, int(job.get("irc_points", 20) or 20)))
    irc_step = float(max(0.01, float(job.get("irc_step", 0.20) or 0.20)))
    grid_file = str(job.get("esp_grid_file", "esp_grid.dat") or "esp_grid.dat").strip()
    total_charge = int(job.get("charge", 0) or 0)

    opts = {
        "basis": basis,
        "scf_type": "df",
        "maxiter": 200,
        "geom_maxiter": geom_maxiter,
        "e_convergence": 1e-8,
        "d_convergence": 1e-8,
    }
    if op == "ts":
        opts["opt_type"] = "TS"
    elif op == "irc":
        opts["opt_type"] = "IRC"
        opts["irc_direction"] = irc_direction
        opts["irc_points"] = irc_points
        opts["irc_step_size"] = irc_step

    pcm_block = (
        "Units = Angstrom\n"
        "Medium {\n"
        "  SolverType = IEFPCM\n"
        f"  Solvent = {solvent}\n"
        "}\n"
        "Cavity {\n"
        "  RadiiSet = UFF\n"
        "  Type = GePol\n"
        "  Area = 0.3\n"
        "  Scaling = False\n"
        "  Mode = Implicit\n"
        "}"
    )

    lines = []
    lines.append("import json, traceback")
    lines.append("import numpy as np")
    lines.append("import psi4")
    lines.append("result = {'ok': False, 'operation': " + repr(op) + "}")
    lines.append("")
    lines.append("def _mat_to_array_bohr(mat, n):")
    lines.append("    arr = np.zeros((int(n), 3), dtype=float)")
    lines.append("    for i in range(int(n)):")
    lines.append("        for j in range(3):")
    lines.append("            try:")
    lines.append("                arr[i, j] = float(mat.get(i, j))")
    lines.append("            except Exception:")
    lines.append("                arr[i, j] = float(np.asarray(mat)[i, j])")
    lines.append("    return arr")
    lines.append("")
    lines.append("def _write_xyz(path, mol):")
    lines.append("    nat = int(mol.natom())")
    lines.append("    mb = mol.geometry()")
    lines.append("    bohr = _mat_to_array_bohr(mb, nat)")
    lines.append("    ang = bohr * float(psi4.constants.bohr2angstroms)")
    lines.append("    syms = [str(mol.symbol(i)).strip() for i in range(nat)]")
    lines.append("    with open(path, 'w', encoding='utf-8') as fh:")
    lines.append("        fh.write(f'{nat}\\n')")
    lines.append("        fh.write('Generated by Psi4 via ProtLigInteract\\n')")
    lines.append("        for s, xyz in zip(syms, ang):")
    lines.append("            fh.write(f'{s} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\\n')")
    lines.append("")
    lines.append("def _collect_thermo_psivars():")
    lines.append("    keys = ['ZPVE','THERMAL ENERGY','ENTHALPY','GIBBS FREE ENERGY','ENTHALPY CORRECTION','GIBBS FREE ENERGY CORRECTION']")
    lines.append("    out = {}")
    lines.append("    for k in keys:")
    lines.append("        try:")
    lines.append("            out[k] = float(psi4.core.variable(k))")
    lines.append("        except Exception:")
    lines.append("            pass")
    lines.append("    return out")
    lines.append("")
    lines.append("try:")
    lines.append(f"    psi4.set_memory('{mem_gb} GB')")
    lines.append(f"    psi4.set_num_threads({threads})")
    lines.append("    psi4.core.set_output_file('psi4_run.out', False)")
    lines.append('    mol = psi4.geometry(r"""' + mol_block + '""")')
    if use_pcm:
        pcm_opts = dict(opts)
        pcm_opts["pcm"] = True
        pcm_opts["pcm_scf_type"] = "total"
        pcm_opts["pcm__input"] = pcm_block
        lines.append("    psi4.set_options(" + repr(pcm_opts) + ")")
    else:
        lines.append("    psi4.set_options(" + repr(opts) + ")")
    lines.append("    method_name = " + repr(method))
    lines.append("")
    if op == "optimize":
        lines.append("    e = psi4.optimize(method_name)")
        lines.append("    result['energy_hartree'] = float(e)")
    elif op == "frequencies":
        lines.append("    e, wfn = psi4.frequency(method_name, return_wfn=True)")
        lines.append("    result['energy_hartree'] = float(e)")
        lines.append("    result['thermo'] = _collect_thermo_psivars()")
    elif op == "stationary":
        lines.append("    eopt = psi4.optimize(method_name)")
        lines.append("    efreq, wfn = psi4.frequency(method_name, return_wfn=True)")
        lines.append("    result['energy_opt_hartree'] = float(eopt)")
        lines.append("    result['energy_freq_hartree'] = float(efreq)")
        lines.append("    result['thermo'] = _collect_thermo_psivars()")
    elif op == "ts":
        lines.append("    ets = psi4.optimize(method_name)")
        lines.append("    efreq, wfn = psi4.frequency(method_name, return_wfn=True)")
        lines.append("    result['energy_ts_hartree'] = float(ets)")
        lines.append("    result['energy_freq_hartree'] = float(efreq)")
        lines.append("    result['thermo'] = _collect_thermo_psivars()")
    elif op == "irc":
        lines.append("    efreq, wfn = psi4.frequency(method_name, return_wfn=True)")
        lines.append("    eirc = psi4.optimize(method_name)")
        lines.append("    result['energy_freq_hartree'] = float(efreq)")
        lines.append("    result['energy_irc_hartree'] = float(eirc)")
        lines.append("    result['thermo'] = _collect_thermo_psivars()")
    elif op == "esp_fit":
        lines.append("    e, wfn = psi4.properties(method_name, properties=['GRID_ESP'], return_wfn=True)")
        lines.append("    result['energy_hartree'] = float(e)")
        lines.append("    pts = np.loadtxt(" + repr(grid_file) + ", dtype=float)")
        lines.append("    pts = np.asarray(pts, dtype=float).reshape(-1, 3)")
        lines.append("    vals = np.asarray(wfn.oeprop.Vvals(), dtype=float).reshape(-1)")
        lines.append("    if vals.size != pts.shape[0]:")
        lines.append("        raise RuntimeError(f'GRID_ESP size mismatch: values={vals.size}, points={pts.shape[0]}')")
        lines.append("    amol = psi4.core.get_active_molecule()")
        lines.append("    amol.update_geometry()")
        lines.append("    nat = int(amol.natom())")
        lines.append("    atom_bohr = _mat_to_array_bohr(amol.geometry(), nat)")
        lines.append("    ang2bohr = 1.0 / float(psi4.constants.bohr2angstroms)")
        lines.append("    pts_bohr = pts * ang2bohr")
        lines.append("    diff = pts_bohr[:, None, :] - atom_bohr[None, :, :]")
        lines.append("    r = np.linalg.norm(diff, axis=2)")
        lines.append("    r = np.clip(r, 1.0e-8, None)")
        lines.append("    A = 1.0 / r")
        lines.append("    q_total = float(" + str(int(total_charge)) + ")")
        lines.append("    ATA = A.T @ A")
        lines.append("    ATV = A.T @ vals")
        lines.append("    n = int(ATA.shape[0])")
        lines.append("    K = np.zeros((n + 1, n + 1), dtype=float)")
        lines.append("    K[:n, :n] = 2.0 * ATA")
        lines.append("    K[:n, n] = 1.0")
        lines.append("    K[n, :n] = 1.0")
        lines.append("    b = np.zeros(n + 1, dtype=float)")
        lines.append("    b[:n] = 2.0 * ATV")
        lines.append("    b[n] = q_total")
        lines.append("    try:")
        lines.append("        sol = np.linalg.solve(K, b)")
        lines.append("    except Exception:")
        lines.append("        sol = np.linalg.lstsq(K, b, rcond=None)[0]")
        lines.append("    qfit = np.asarray(sol[:n], dtype=float)")
        lines.append("    resid = (A @ qfit) - vals")
        lines.append("    rms = float(np.sqrt(np.mean(resid * resid)))")
        lines.append("    np.savetxt('esp_fit_charges.dat', qfit, fmt='%.10f')")
        lines.append("    result['esp_fit_rms_au'] = rms")
        lines.append("    result['esp_fit_charges_file'] = 'esp_fit_charges.dat'")
    else:
        lines.append("    raise RuntimeError('Unsupported operation')")
    lines.append("    try:")
    lines.append("        fin = psi4.core.get_active_molecule()")
    lines.append("        fin.update_geometry()")
    lines.append("        _write_xyz('optimized.xyz', fin)")
    lines.append("        result['optimized_xyz'] = 'optimized.xyz'")
    lines.append("    except Exception as _geo_e:")
    lines.append("        result['geometry_warning'] = str(_geo_e)")
    lines.append("    result['ok'] = True")
    lines.append("except Exception as e:")
    lines.append("    result['error'] = str(e)")
    lines.append("    result['traceback'] = traceback.format_exc()")
    lines.append("finally:")
    lines.append("    with open('psi4_result.json', 'w', encoding='utf-8') as fh:")
    lines.append("        json.dump(result, fh, indent=2)")
    return "\n".join(lines) + "\n"


def psi4_write_charged_mol2(in_mol2: Path, charges: np.ndarray, out_mol2: Path) -> tuple[bool, str]:
    q = [float(v) for v in (list(charges) if charges is not None else [])]
    if not in_mol2.is_file():
        return False, "Input MOL2 not found."
    if len(q) <= 0:
        return False, "No fitted charges were provided."

    section = None
    out_lines = []
    q_idx = 0
    try:
        with in_mol2.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = str(raw).rstrip("\n")
                if line.startswith("@<TRIPOS>"):
                    section = line.strip().upper()
                    out_lines.append(line)
                    continue
                if section == "@<TRIPOS>ATOM" and line.strip():
                    parts = line.split()
                    if len(parts) >= 9 and q_idx < len(q):
                        try:
                            atom_id = int(parts[0])
                            atom_name = str(parts[1])
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])
                            atom_type = str(parts[5])
                            sub_id = int(float(parts[6]))
                            sub_name = str(parts[7])
                            qv = float(q[q_idx])
                            tail = ""
                            if len(parts) > 9:
                                tail = " " + " ".join(parts[9:])
                            newline = (
                                f"{atom_id:7d} {atom_name:<8} {x:10.4f} {y:10.4f} {z:10.4f} "
                                f"{atom_type:<8} {sub_id:4d} {sub_name:<8} {qv:10.6f}{tail}"
                            )
                            out_lines.append(newline)
                            q_idx += 1
                            continue
                        except Exception:
                            pass
                out_lines.append(line)
    except Exception as e:
        return False, f"Failed reading MOL2: {e}"

    if q_idx != len(q):
        return False, f"Charge count mismatch: fitted={len(q)} mapped={q_idx}"

    try:
        out_mol2.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    except Exception as e:
        return False, f"Failed writing charged MOL2: {e}"
    return True, ""


def parse_mol2_charges(path: Path) -> list[float]:
    section = None
    out = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("@<TRIPOS>"):
                section = line.strip().upper()
                continue
            if section == "@<TRIPOS>ATOM" and line.strip():
                parts = line.split()
                if len(parts) >= 9:
                    out.append(float(parts[8]))
    return out


def write_toy_mol2(path: Path) -> None:
    mol2 = """@<TRIPOS>MOLECULE
TOY
3 2 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1         0.0000     0.0000     0.0000 C.3       1 LIG       0.000000
      2 O1         1.2300     0.0000     0.0000 O.2       1 LIG       0.000000
      3 H1        -0.5400     0.9300     0.0000 H         1 LIG       0.000000
@<TRIPOS>BOND
     1    1    2 1
     2    1    3 1
"""
    path.write_text(mol2, encoding="utf-8")


def water_atoms() -> list[dict]:
    return [
        {"element": "O", "x": 0.000000, "y": 0.000000, "z": 0.000000},
        {"element": "H", "x": 0.957200, "y": 0.000000, "z": 0.000000},
        {"element": "H", "x": -0.239987, "y": 0.927297, "z": 0.000000},
    ]


def ethanol_like_atoms() -> list[dict]:
    return [
        {"element": "C", "x": 0.0000, "y": 0.0000, "z": 0.0000},
        {"element": "C", "x": 1.5200, "y": 0.0000, "z": 0.0000},
        {"element": "O", "x": 2.0900, "y": 1.2400, "z": 0.0000},
        {"element": "H", "x": -0.5400, "y": 0.9300, "z": 0.0000},
        {"element": "H", "x": -0.5400, "y": -0.9300, "z": 0.0000},
        {"element": "H", "x": 1.9800, "y": -0.9300, "z": 0.0000},
    ]


@dataclass
class TestResult:
    name: str
    ok: bool
    detail: str
    artifacts: dict


def test_grid_and_molecule_block(out_dir: Path) -> TestResult:
    atoms = ethanol_like_atoms()
    pts = psi4_generate_esp_grid(atoms, points_per_atom=24, max_points=6000)
    if pts.shape[0] < 100:
        return TestResult(
            name="grid_and_molecule_block",
            ok=False,
            detail=f"Too few grid points generated: {pts.shape[0]}",
            artifacts={},
        )
    grid_file = out_dir / "esp_grid_test.dat"
    np.savetxt(grid_file, pts, fmt="%.8f")
    mol_block = psi4_molecule_block(atoms, charge=0, multiplicity=1)
    if "units angstrom" not in mol_block or "no_reorient" not in mol_block:
        return TestResult(
            name="grid_and_molecule_block",
            ok=False,
            detail="Molecule block missing expected directives",
            artifacts={"grid_file": str(grid_file)},
        )
    return TestResult(
        name="grid_and_molecule_block",
        ok=True,
        detail=f"Generated {pts.shape[0]} ESP points",
        artifacts={"grid_file": str(grid_file)},
    )


def test_mol2_charge_write(out_dir: Path) -> TestResult:
    src = out_dir / "toy_input.mol2"
    dst = out_dir / "toy_charged.mol2"
    write_toy_mol2(src)
    q = np.array([0.125, -0.550, 0.425], dtype=float)
    ok, msg = psi4_write_charged_mol2(src, q, dst)
    if not ok:
        return TestResult("mol2_charge_write", False, msg, {"input": str(src)})
    charges = parse_mol2_charges(dst)
    if len(charges) != 3:
        return TestResult(
            "mol2_charge_write",
            False,
            f"Expected 3 charges in output, got {len(charges)}",
            {"output": str(dst)},
        )
    if max(abs(a - b) for a, b in zip(charges, q.tolist())) > 1e-6:
        return TestResult(
            "mol2_charge_write",
            False,
            "Output charges differ from requested charges",
            {"output": str(dst), "charges_out": charges},
        )
    return TestResult(
        "mol2_charge_write",
        True,
        "MOL2 charges written correctly",
        {"input": str(src), "output": str(dst), "charge_sum": float(np.sum(q))},
    )


def test_script_generation(out_dir: Path) -> TestResult:
    atoms = water_atoms()
    mol_block = psi4_molecule_block(atoms, charge=0, multiplicity=1)
    operations = ["optimize", "frequencies", "stationary", "ts", "irc", "esp_fit"]
    generated = []
    for op in operations:
        job = {
            "operation": op,
            "method": "hf",
            "basis": "sto-3g",
            "molecule_block": mol_block,
            "memory_gb": 1,
            "threads": 1,
            "geom_maxiter": 50,
            "implicit": False,
            "solvent": "Water",
            "irc_direction": "BOTH",
            "irc_points": 8,
            "irc_step": 0.20,
            "esp_grid_file": "esp_grid.dat",
            "charge": 0,
        }
        text = psi4_make_script(job)
        if "psi4_result.json" not in text or "method_name = 'hf'" not in text:
            return TestResult(
                "script_generation",
                False,
                f"Generated script for {op} is missing expected content",
                {},
            )
        p = out_dir / f"psi4_input_{op}.dat"
        p.write_text(text, encoding="utf-8")
        generated.append(str(p))
    return TestResult(
        "script_generation",
        True,
        f"Generated scripts for {len(operations)} operations",
        {"inputs": generated},
    )


def test_psi4_runtime_optimize(out_dir: Path, psi4_bin: str | None, timeout_s: int) -> TestResult:
    if not psi4_bin:
        return TestResult(
            "psi4_runtime_optimize",
            True,
            "SKIPPED: psi4 executable not found in PATH",
            {"skipped": True},
        )
    psi4_path = shutil.which(psi4_bin) if os.path.basename(psi4_bin) == psi4_bin else psi4_bin
    if not psi4_path or not os.path.isfile(psi4_path):
        return TestResult(
            "psi4_runtime_optimize",
            False,
            f"psi4 binary not found: {psi4_bin}",
            {},
        )

    job_dir = out_dir / "psi4_runtime_optimize"
    job_dir.mkdir(parents=True, exist_ok=True)
    atoms = water_atoms()
    inp = job_dir / "psi4_input.dat"
    out = job_dir / "psi4_output.out"
    script = psi4_make_script(
        {
            "operation": "optimize",
            "method": "hf",
            "basis": "sto-3g",
            "molecule_block": psi4_molecule_block(atoms, charge=0, multiplicity=1),
            "memory_gb": 1,
            "threads": 1,
            "geom_maxiter": 80,
            "implicit": False,
            "solvent": "Water",
            "charge": 0,
        }
    )
    inp.write_text(script, encoding="utf-8")

    cmdline = [str(psi4_path), "-n", "1", str(inp), str(out)]
    proc = subprocess.run(
        cmdline,
        cwd=str(job_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=max(60, int(timeout_s)),
        check=False,
    )
    result_json = job_dir / "psi4_result.json"
    if proc.returncode != 0:
        return TestResult(
            "psi4_runtime_optimize",
            False,
            f"Psi4 returned non-zero code {proc.returncode}",
            {
                "stdout_tail": proc.stdout[-1200:],
                "stderr_tail": proc.stderr[-1200:],
                "job_dir": str(job_dir),
            },
        )
    if not result_json.is_file():
        return TestResult(
            "psi4_runtime_optimize",
            False,
            "psi4_result.json was not produced",
            {"job_dir": str(job_dir)},
        )
    try:
        result = json.loads(result_json.read_text(encoding="utf-8"))
    except Exception as e:
        return TestResult(
            "psi4_runtime_optimize",
            False,
            f"Failed to parse psi4_result.json: {e}",
            {"job_dir": str(job_dir)},
        )
    if not bool(result.get("ok", False)):
        return TestResult(
            "psi4_runtime_optimize",
            False,
            f"Psi4 result marked failure: {result.get('error', 'unknown')}",
            {"job_dir": str(job_dir), "result": result},
        )
    return TestResult(
        "psi4_runtime_optimize",
        True,
        f"Energy={float(result.get('energy_hartree')):.8f} Eh",
        {"job_dir": str(job_dir), "result_json": str(result_json)},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run QM smoke tests for ProtLigInteract.")
    ap.add_argument(
        "--out-dir",
        default="benchmarks/qm/results",
        help="Directory for test artifacts and summary JSON.",
    )
    ap.add_argument(
        "--psi4-bin",
        default="psi4",
        help="Psi4 executable path or name (default: psi4).",
    )
    ap.add_argument(
        "--skip-psi4-runtime",
        action="store_true",
        help="Skip real Psi4 runtime test even if psi4 is available.",
    )
    ap.add_argument(
        "--timeout-sec",
        type=int,
        default=300,
        help="Timeout for the real Psi4 runtime test.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[TestResult] = []
    results.append(test_grid_and_molecule_block(out_dir))
    results.append(test_mol2_charge_write(out_dir))
    results.append(test_script_generation(out_dir))

    psi4_bin = None if args.skip_psi4_runtime else args.psi4_bin
    if psi4_bin:
        psi4_path = shutil.which(psi4_bin) if os.path.basename(psi4_bin) == psi4_bin else psi4_bin
        if not psi4_path:
            psi4_bin = None
    results.append(test_psi4_runtime_optimize(out_dir, psi4_bin, timeout_s=args.timeout_sec))

    summary = {
        "ok": all(r.ok for r in results),
        "python": sys.version.split()[0],
        "tests": [
            {"name": r.name, "ok": r.ok, "detail": r.detail, "artifacts": r.artifacts}
            for r in results
        ],
    }
    summary_json = out_dir / "qm_smoke_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("QM smoke tests")
    print(f"  Output dir: {out_dir}")
    for r in results:
        status = "PASS" if r.ok else "FAIL"
        print(f"  [{status}] {r.name}: {r.detail}")
    print(f"  Summary JSON: {summary_json}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
