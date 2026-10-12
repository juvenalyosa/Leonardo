#!/usr/bin/env python3
"""Generate three reproducible, cinematic Leonardo molecular-science videos.

The calculations are deliberately small enough for a workstation:
1. Crystal-pocket rigid redocking of benzamidine into trypsin (PDB 3PTB).
2. QST-style H + H2 exchange path with GFN2-xTB single-point energies.
3. Short implicit-solvent OpenMM trajectory of Trp-cage (PDB 1L2Y),
   analysed with MDAnalysis.

The rendered MP4s never relabel arbitrary animation as simulation.  Each one
shows its method in-frame and is accompanied by CSV/JSON source results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "cinematic_videos"
DATA = OUT / "data"
RESULTS = OUT / "results"
FRAMES = OUT / "frames"
VIDEOS = OUT / "videos"

BG = "#040711"
PANEL = "#0a1222"
WHITE = "#f2f5f8"
MUTED = "#91a0b6"
CYAN = "#32d5e6"
BLUE = "#3978ff"
GOLD = "#e3b341"
MAGENTA = "#d05bd3"
RED = "#ff5c70"
GREEN = "#57d38c"
FPS = 15
N_FRAMES = 120
HARTREE_TO_KCAL = 627.509474


@dataclass
class Atom:
    record: str
    serial: int
    name: str
    resname: str
    chain: str
    resid: int
    xyz: np.ndarray
    element: str


def ensure_dirs() -> None:
    for path in (DATA, RESULTS, FRAMES, VIDEOS):
        path.mkdir(parents=True, exist_ok=True)


def parse_pdb_first_model(path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        alt = line[16:17]
        if alt not in (" ", "A"):
            continue
        raw_element = line[76:78].strip() if len(line) >= 78 else ""
        atom_name = line[12:16].strip()
        element = (raw_element or re.sub(r"[^A-Za-z]", "", atom_name)[:1]).upper()
        atoms.append(
            Atom(
                record=line[0:6].strip(),
                serial=int(line[6:11]),
                name=atom_name,
                resname=line[17:20].strip(),
                chain=line[21:22].strip(),
                resid=int(line[22:26]),
                xyz=np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                element=element,
            )
        )
    return atoms


def rotation_matrix(angles: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", angles).as_matrix()


def infer_bonds(coords: np.ndarray, elements: list[str], scale: float = 1.23) -> list[tuple[int, int]]:
    cov = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05, "P": 1.07}
    bonds: list[tuple[int, int]] = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            limit = scale * (cov.get(elements[i], 0.76) + cov.get(elements[j], 0.76))
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.45 < distance <= limit:
                bonds.append((i, j))
    return bonds


def atom_charge(atom: Atom, ligand: bool = False) -> float:
    if ligand:
        if atom.element == "N":
            return 0.38
        if atom.element == "O":
            return -0.30
        return 0.0
    if atom.resname in {"ASP", "GLU"} and atom.element == "O" and atom.name.startswith(("OD", "OE")):
        return -0.55
    if atom.resname in {"LYS", "ARG", "HIS"} and atom.element == "N":
        return 0.35
    if atom.element == "O":
        return -0.18
    if atom.element == "N":
        return 0.14
    return 0.0


def docking_score(
    pose: np.ndarray,
    ligand_atoms: list[Atom],
    pocket_atoms: list[Atom],
) -> float:
    rec = np.vstack([a.xyz for a in pocket_atoms])
    distances = np.linalg.norm(pose[:, None, :] - rec[None, :, :], axis=2)
    distances = np.clip(distances, 0.75, 20.0)
    vdw = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}
    lrad = np.array([vdw.get(a.element, 1.70) for a in ligand_atoms])[:, None]
    rrad = np.array([vdw.get(a.element, 1.70) for a in pocket_atoms])[None, :]
    contact = lrad + rrad
    overlap = np.maximum(0.0, 0.80 * contact - distances)
    repulsion = 4.0 * np.sum(overlap**2)
    surface = -0.080 * np.sum(np.exp(-((distances - 3.55) / 0.72) ** 2))
    le = np.array([a.element in {"N", "O", "S"} for a in ligand_atoms])[:, None]
    relem = np.array([a.element in {"N", "O", "S"} for a in pocket_atoms])[None, :]
    polar = -0.38 * np.sum((le & relem) * np.exp(-((distances - 2.90) / 0.48) ** 2))
    lq = np.array([atom_charge(a, ligand=True) for a in ligand_atoms])[:, None]
    rq = np.array([atom_charge(a, ligand=False) for a in pocket_atoms])[None, :]
    electrostatic = 3.2 * np.sum(lq * rq / distances)
    return float(repulsion + surface + polar + electrostatic)


def write_pdb_atoms(path: Path, atoms: list[Atom], coords: np.ndarray, record: str = "HETATM") -> None:
    lines = []
    for idx, (atom, xyz) in enumerate(zip(atoms, coords), 1):
        lines.append(
            f"{record:<6}{idx:5d} {atom.name:>4s} {atom.resname:>3s} {atom.chain or 'A':1s}"
            f"{atom.resid:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
            f"  1.00 20.00          {atom.element:>2s}"
        )
    path.write_text("\n".join(lines) + "\nEND\n")


def run_docking() -> dict:
    print("[1/6] Crystal-pocket redocking: 3PTB / benzamidine", flush=True)
    atoms = parse_pdb_first_model(DATA / "3PTB.pdb")
    ligand = [a for a in atoms if a.record == "HETATM" and a.resname == "BEN"]
    receptor = [a for a in atoms if a.record == "ATOM"]
    if len(ligand) != 9:
        raise RuntimeError(f"Expected 9 BEN atoms in 3PTB; found {len(ligand)}")
    native = np.vstack([a.xyz for a in ligand])
    center = native.mean(axis=0)
    pocket = [a for a in receptor if np.linalg.norm(a.xyz - center) <= 10.0]
    base = native - center

    def pose_from_vector(vector: np.ndarray) -> np.ndarray:
        return base @ rotation_matrix(vector[:3]).T + center + vector[3:]

    trace: list[float] = []

    def objective(vector: np.ndarray) -> float:
        return docking_score(pose_from_vector(vector), ligand, pocket)

    def callback(vector: np.ndarray, convergence: float = 0.0) -> bool:
        trace.append(objective(vector))
        return False

    rng = np.random.default_rng(1738)
    population = np.column_stack(
        [rng.uniform(-math.pi, math.pi, 72) for _ in range(3)]
        + [rng.uniform(-4.5, 4.5, 72) for _ in range(3)]
    )
    result = differential_evolution(
        objective,
        [(-math.pi, math.pi)] * 3 + [(-4.5, 4.5)] * 3,
        seed=1738,
        maxiter=140,
        popsize=12,
        tol=1e-4,
        polish=True,
        updating="immediate",
        workers=1,
        callback=callback,
        init=population,
    )
    pose = pose_from_vector(result.x)
    rmsd = float(np.sqrt(np.mean(np.sum((pose - native) ** 2, axis=1))))
    trace = np.minimum.accumulate(np.array(trace or [result.fun], dtype=float)).tolist()
    write_pdb_atoms(RESULTS / "docking_best_pose.pdb", ligand, pose)
    with (RESULTS / "docking_convergence.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "best_nonbonded_score_au"])
        writer.writerows(enumerate(trace, 1))
    payload = {
        "system": "trypsin-benzamidine",
        "pdb_id": "3PTB",
        "protocol": "rigid-body crystal-pocket redocking; differential evolution",
        "score": "soft van der Waals + polar contact + approximate electrostatics",
        "seed": 1738,
        "pocket_atoms": len(pocket),
        "ligand_heavy_atoms": len(ligand),
        "iterations": len(trace),
        "best_score_au": float(result.fun),
        "heavy_atom_rmsd_to_crystal_A": rmsd,
        "success": bool(result.success),
        "optimizer_message": str(result.message),
    }
    (RESULTS / "docking_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    np.savez_compressed(
        RESULTS / "docking_render_data.npz",
        native=native,
        pose=pose,
        ligand_elements=np.array([a.element for a in ligand]),
        pocket_xyz=np.vstack([a.xyz for a in pocket]),
        pocket_elements=np.array([a.element for a in pocket]),
        ca_xyz=np.vstack([a.xyz for a in receptor if a.name == "CA"]),
        trace=np.asarray(trace),
    )
    return payload


def qst_geometry(progress: float) -> np.ndarray:
    left_distance = (1.0 - progress) * 1.80 + progress * 0.74
    right_distance = (1.0 - progress) * 0.74 + progress * 1.80
    return np.array([[-left_distance, 0.0, 0.0], [0.0, 0.0, 0.0], [right_distance, 0.0, 0.0]])


def run_qst() -> dict:
    print("[2/6] QST-style H + H2 path: GFN2-xTB energies", flush=True)
    xtb = shutil.which("xtb")
    if not xtb:
        raise RuntimeError("xtb is required for the quantum reaction-path video")
    qst_dir = RESULTS / "qst_xtb"
    qst_dir.mkdir(parents=True, exist_ok=True)
    progress_values = np.linspace(0.0, 1.0, 25)
    energies_h = []
    geometries = []
    energy_pattern = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "2"
    for idx, progress in enumerate(progress_values):
        geometry = qst_geometry(float(progress))
        geometries.append(geometry)
        work = qst_dir / f"image_{idx:02d}"
        work.mkdir(parents=True, exist_ok=True)
        xyz = work / "h3.xyz"
        xyz.write_text(
            "3\n"
            f"QST synchronous-transit image {idx}; neutral doublet\n"
            + "\n".join(f"H {x:.8f} {y:.8f} {z:.8f}" for x, y, z in geometry)
            + "\n"
        )
        completed = subprocess.run(
            [xtb, xyz.name, "--sp", "--gfn", "2", "--uhf", "1"],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        log = completed.stdout + "\n" + completed.stderr
        (work / "xtb.log").write_text(log)
        match = energy_pattern.search(log)
        if completed.returncode or not match:
            raise RuntimeError(f"xTB failed for QST image {idx}: {log[-1200:]}")
        energies_h.append(float(match.group(1)))
    energies_h = np.asarray(energies_h)
    relative = (energies_h - energies_h.min()) * HARTREE_TO_KCAL
    barrier = float(relative.max())
    ts_index = int(np.argmax(relative))
    with (RESULTS / "qst_energy_profile.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "reaction_coordinate", "energy_hartree", "relative_kcal_mol"])
        for idx, row in enumerate(zip(progress_values, energies_h, relative)):
            writer.writerow([idx, *row])
    np.savez_compressed(
        RESULTS / "qst_render_data.npz",
        reaction_coordinate=progress_values,
        energies_hartree=energies_h,
        relative_kcal=relative,
        geometries=np.asarray(geometries),
    )
    payload = {
        "reaction": "H(a) + H(b)-H(c) -> H(a)-H(b) + H(c)",
        "path": "25-image synchronous transit (QST-style)",
        "electronic_structure": "GFN2-xTB single-point; charge 0; doublet (UHF=1)",
        "xtb_version": subprocess.run([xtb, "--version"], capture_output=True, text=True).stdout.strip(),
        "barrier_kcal_mol_relative_to_path_minimum": barrier,
        "highest_energy_image": ts_index,
        "note": "This is a QST-style scanned path, not a converged saddle-point frequency calculation.",
    }
    (RESULTS / "qst_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def kabsch(mobile: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    cov = (mobile - mobile_center).T @ (reference - reference_center)
    u, _, vt = np.linalg.svd(cov)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation, mobile_center, reference_center


def run_md() -> dict:
    print("[3/6] OpenMM dynamics: Trp-cage / implicit solvent", flush=True)
    import MDAnalysis as mda
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import DCDReporter, ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation, StateDataReporter

    pdb = PDBFile(str(DATA / "1L2Y.pdb"))
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7.0)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        hydrogenMass=3.0 * unit.amu,
    )
    integrator = LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    platform = Platform.getPlatformByName("CPU")
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=800)
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 20260915)
    simulation.step(1000)

    prepared_pdb = RESULTS / "trpcage_prepared.pdb"
    with prepared_pdb.open("w") as handle:
        PDBFile.writeFile(modeller.topology, simulation.context.getState(getPositions=True).getPositions(), handle)
    dcd = RESULTS / "trpcage_trajectory.dcd"
    state_csv = RESULTS / "trpcage_state.csv"
    simulation.reporters.append(DCDReporter(str(dcd), 100))
    simulation.reporters.append(
        StateDataReporter(
            str(state_csv),
            100,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            separator=",",
        )
    )
    simulation.step(12000)

    universe = mda.Universe(str(prepared_pdb), str(dcd))
    ca = universe.select_atoms("protein and name CA")
    heavy = universe.select_atoms("protein and not name H*")
    trp = universe.select_atoms("resname TRP and not name H*")
    ca_indices = ca.indices
    heavy_indices = heavy.indices
    trp_indices_global = set(int(i) for i in trp.indices)
    trp_indices_heavy = np.array([i for i, global_i in enumerate(heavy_indices) if int(global_i) in trp_indices_global])
    reference = None
    ca_frames = []
    heavy_frames = []
    rmsd_values = []
    rg_values = []
    for ts in universe.trajectory:
        current_ca = ca.positions.astype(float).copy()
        current_heavy = heavy.positions.astype(float).copy()
        if reference is None:
            reference = current_ca.copy()
        rotation, mobile_center, reference_center = kabsch(current_ca, reference)
        aligned_ca = (current_ca - mobile_center) @ rotation + reference_center
        aligned_heavy = (current_heavy - mobile_center) @ rotation + reference_center
        ca_frames.append(aligned_ca)
        heavy_frames.append(aligned_heavy)
        rmsd_values.append(float(np.sqrt(np.mean(np.sum((aligned_ca - reference) ** 2, axis=1)))))
        centered = aligned_ca - aligned_ca.mean(axis=0)
        rg_values.append(float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))))

    with state_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    potential_key = next(k for k in rows[0] if "Potential Energy" in k)
    temperature_key = next(k for k in rows[0] if "Temperature" in k)
    time_key = next(k for k in rows[0] if "Time" in k)
    energies = np.array([float(row[potential_key]) for row in rows])
    temperatures = np.array([float(row[temperature_key]) for row in rows])
    times = np.array([float(row[time_key]) for row in rows])
    rmsd_values = np.asarray(rmsd_values)
    rg_values = np.asarray(rg_values)
    with (RESULTS / "trpcage_analysis.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time_ps", "ca_rmsd_A", "ca_radius_gyration_A", "potential_energy_kJ_mol", "temperature_K"])
        for idx in range(len(times)):
            writer.writerow([idx, times[idx], rmsd_values[idx], rg_values[idx], energies[idx], temperatures[idx]])
    np.savez_compressed(
        RESULTS / "md_render_data.npz",
        ca_frames=np.asarray(ca_frames),
        heavy_frames=np.asarray(heavy_frames),
        heavy_elements=np.array([atom.element for atom in heavy]),
        trp_indices_heavy=trp_indices_heavy,
        times_ps=times,
        rmsd_A=rmsd_values,
        rg_A=rg_values,
        potential_kj_mol=energies,
        temperature_K=temperatures,
    )
    payload = {
        "system": "Trp-cage miniprotein TC5b",
        "pdb_id": "1L2Y (model 1)",
        "engine": "OpenMM",
        "force_field": "Amber14 protein + GBN2 implicit solvent",
        "integrator": "LangevinMiddle, 300 K, friction 1/ps, timestep 2 fs",
        "equilibration_ps": 2.0,
        "production_ps": 24.0,
        "total_simulated_ps": float(times[-1]) if len(times) else 0.0,
        "saved_interval_ps": 0.2,
        "atoms": int(universe.atoms.n_atoms),
        "frames": int(len(times)),
        "final_ca_rmsd_A": float(rmsd_values[-1]),
        "mean_temperature_K": float(temperatures.mean()),
        "analysis": "MDAnalysis; Kabsch-aligned C-alpha RMSD and radius of gyration",
        "platform": platform.getName(),
    }
    (RESULTS / "md_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def ease(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def normalize_xyz(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = coords.mean(axis=0)
    centered = coords - center
    scale = float(np.max(np.linalg.norm(centered, axis=1))) or 1.0
    return centered, center, scale


def style_3d(ax) -> None:
    ax.set_facecolor((0, 0, 0, 0))
    ax.set_axis_off()
    ax.grid(False)
    ax.set_proj_type("persp", focal_length=0.8)


def add_brand(fig, chapter: str, frame: int) -> None:
    fig.text(0.045, 0.932, "LEONARDO", color=WHITE, fontsize=19, fontweight="bold", ha="left", va="center")
    fig.text(0.183, 0.932, "MOLECULAR DISCOVERY", color=CYAN, fontsize=8.5, fontweight="bold", ha="left", va="center")
    fig.text(0.955, 0.932, chapter.upper(), color=MUTED, fontsize=8.5, fontweight="bold", ha="right", va="center")
    fig.lines.append(plt.Line2D([0.045, 0.955], [0.905, 0.905], transform=fig.transFigure, color="#21304a", lw=0.7))
    fig.lines.append(plt.Line2D([0.045, 0.045 + 0.91 * frame / (N_FRAMES - 1)], [0.067, 0.067], transform=fig.transFigure, color=CYAN, lw=2.0, alpha=0.8))


def cinematic_title(fig, title: str, subtitle: str, frame: int) -> None:
    alpha = min(1.0, frame / 12.0)
    fig.text(0.055, 0.835, title, color=WHITE, fontsize=30, fontweight="bold", alpha=alpha, ha="left", va="top", path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
    fig.text(0.057, 0.777, subtitle, color=MUTED, fontsize=10.5, alpha=alpha, ha="left", va="top")


def colored_bonds(ax, coords: np.ndarray, bonds: list[tuple[int, int]], color: str = "#d8e0ec", lw: float = 2.2, alpha: float = 0.9) -> None:
    for i, j in bonds:
        xyz = np.vstack([coords[i], coords[j]])
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=lw, alpha=alpha, solid_capstyle="round")


ELEMENT_COLORS = {"C": "#a8b2c1", "N": "#4d88ff", "O": "#ff5b69", "S": "#ffd24a", "H": "#f7f9fc", "P": "#ff9d4a"}
ELEMENT_SIZES = {"C": 150, "N": 175, "O": 175, "S": 210, "H": 115, "P": 210}


def draw_atoms(ax, coords: np.ndarray, elements: list[str], alpha: float = 1.0, scale: float = 1.0, glow: bool = True) -> None:
    colors = [ELEMENT_COLORS.get(str(e), "#bec7d4") for e in elements]
    sizes = [ELEMENT_SIZES.get(str(e), 150) * scale for e in elements]
    if glow:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=np.asarray(sizes) * 2.4, c=colors, alpha=0.07 * alpha, depthshade=False, linewidth=0)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=sizes, c=colors, alpha=alpha, depthshade=True, edgecolors="#f5f8ff", linewidths=0.25)


def make_panel(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#263550")
        spine.set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)
    return ax


def save_frame(fig, target: Path) -> None:
    fig.savefig(target, dpi=100, facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
    plt.close(fig)


def render_docking(summary: dict) -> Path:
    print("[4/6] Rendering video 1: docking", flush=True)
    data = np.load(RESULTS / "docking_render_data.npz")
    target_dir = FRAMES / "01_docking"
    target_dir.mkdir(parents=True, exist_ok=True)
    native_center = data["native"].mean(axis=0)
    pocket_raw = data["pocket_xyz"] - native_center
    scale = float(np.max(np.linalg.norm(pocket_raw, axis=1)))
    pocket = pocket_raw / scale
    ca = (data["ca_xyz"] - native_center) / scale
    local_ca = np.linalg.norm(ca, axis=1) < 1.48
    pose = (data["pose"] - native_center) / scale
    elements = data["ligand_elements"].tolist()
    bonds = infer_bonds(pose, elements, scale=1.23 / scale)
    trace = data["trace"]
    trace_x = np.arange(1, len(trace) + 1)
    for frame in range(N_FRAMES):
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
        ax = fig.add_axes([0.00, 0.03, 0.72, 0.88], projection="3d")
        style_3d(ax)
        for idx in range(len(ca) - 1):
            if local_ca[idx] and local_ca[idx + 1] and np.linalg.norm(ca[idx + 1] - ca[idx]) < 0.65:
                segment = ca[idx : idx + 2]
                ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=CYAN, lw=1.7, alpha=0.42)
        ax.scatter(pocket[:, 0], pocket[:, 1], pocket[:, 2], s=8, c="#5b84b8", alpha=0.18, depthshade=False)
        dock_t = ease((frame - 18) / 65.0)
        start_offset = np.array([0.78, -0.58, 0.48])
        current = pose + (1.0 - dock_t) * start_offset
        colored_bonds(ax, current, bonds, lw=2.5)
        draw_atoms(ax, current, elements, scale=1.1)
        if dock_t > 0.82:
            d = np.linalg.norm(current[:, None, :] - pocket[None, :, :], axis=2)
            contacts = np.argwhere(d < (3.4 / scale))
            for li, ri in contacts[:14]:
                pts = np.vstack([current[li], pocket[ri]])
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=GOLD, lw=0.8, ls="--", alpha=0.45)
        ax.view_init(elev=18 + 7 * math.sin(frame / 25), azim=28 + frame * 0.44)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        add_brand(fig, "01 / REAL DOCKING", frame)
        cinematic_title(fig, "Docking. Resuelto.", "Benzamidina en tripsina · PDB 3PTB · búsqueda rígida reproducible", frame)
        panel = make_panel(fig, [0.735, 0.205, 0.225, 0.275])
        shown = max(1, int(len(trace) * min(1.0, frame / 82.0)))
        panel.plot(trace_x[:shown], trace[:shown], color=CYAN, lw=1.8)
        panel.fill_between(trace_x[:shown], trace[:shown], np.max(trace), color=CYAN, alpha=0.08)
        panel.set_title("CONVERGENCIA DEL DOCKING", loc="left", color=WHITE, fontsize=8.5, fontweight="bold", pad=10)
        panel.set_ylabel("score no enlazante (u.a.)", color=MUTED, fontsize=7)
        panel.grid(color="#24324a", alpha=0.35, lw=0.5)
        fig.text(0.742, 0.145, f"BEST SCORE  {summary['best_score_au']:.2f} u.a.", color=GOLD, fontsize=10, fontweight="bold")
        fig.text(0.742, 0.111, f"RMSD vs cristal  {summary['heavy_atom_rmsd_to_crystal_A']:.2f} Å", color=WHITE, fontsize=8.5)
        fig.text(0.055, 0.105, "POSE SEARCH", color=GOLD if dock_t < 1 else GREEN, fontsize=9, fontweight="bold")
        fig.text(0.055, 0.080, "soft vdW  ·  polar contacts  ·  electrostatics", color=MUTED, fontsize=8)
        save_frame(fig, target_dir / f"frame_{frame:04d}.png")
    return encode_video(target_dir, VIDEOS / "01_leonardo_docking_cinematic.mp4", tone=55)


def render_qst(summary: dict) -> Path:
    print("[5/6] Rendering video 2: QST reaction path", flush=True)
    data = np.load(RESULTS / "qst_render_data.npz")
    rc = data["reaction_coordinate"]
    energy = data["relative_kcal"]
    geometries = data["geometries"]
    target_dir = FRAMES / "02_qst"
    target_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(N_FRAMES):
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
        ax = fig.add_axes([0.01, 0.05, 0.68, 0.84], projection="3d")
        style_3d(ax)
        scan_t = ease((frame - 12) / 90.0)
        position = scan_t * (len(geometries) - 1)
        lo = int(np.floor(position))
        hi = min(lo + 1, len(geometries) - 1)
        frac = position - lo
        geometry = geometries[lo] * (1 - frac) + geometries[hi] * frac
        angle = 0.20 * math.sin(frame / 22)
        rot = Rotation.from_euler("yz", [angle, 0.22 * math.sin(frame / 31)]).as_matrix()
        geometry = geometry @ rot.T
        for i, j in ((0, 1), (1, 2)):
            dist = np.linalg.norm(geometry[i] - geometry[j])
            bond_alpha = float(np.clip((1.75 - dist) / 0.75, 0.08, 1.0))
            pts = np.vstack([geometry[i], geometry[j]])
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=CYAN, lw=7, alpha=0.12 * bond_alpha)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=WHITE, lw=2.0, alpha=bond_alpha)
        draw_atoms(ax, geometry, ["H", "H", "H"], scale=2.2)
        ax.view_init(elev=18, azim=30 + frame * 0.36)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_zlim(-2.4, 2.4)
        add_brand(fig, "02 / QUANTUM REACTION PATH", frame)
        cinematic_title(fig, "Una reacción, cuadro a cuadro.", "Intercambio H + H₂ · ruta QST · energía electrónica GFN2-xTB", frame)
        panel = make_panel(fig, [0.70, 0.215, 0.26, 0.31])
        shown_x = float(rc[lo] * (1 - frac) + rc[hi] * frac)
        shown_e = float(energy[lo] * (1 - frac) + energy[hi] * frac)
        panel.plot(rc, energy, color="#44526a", lw=1.2, alpha=0.65)
        mask = rc <= shown_x
        panel.plot(rc[mask], energy[mask], color=MAGENTA, lw=2.4)
        panel.scatter([shown_x], [shown_e], s=38, c=GOLD, edgecolors=WHITE, linewidths=0.6, zorder=5)
        panel.set_title("PERFIL DE ENERGÍA CUÁNTICA", loc="left", color=WHITE, fontsize=8.5, fontweight="bold", pad=10)
        panel.set_ylabel("ΔE (kcal/mol)", color=MUTED, fontsize=7)
        panel.grid(color="#24324a", alpha=0.35, lw=0.5)
        fig.text(0.708, 0.151, f"BARRERA ESCANEADA  {summary['barrier_kcal_mol_relative_to_path_minimum']:.1f} kcal/mol", color=GOLD, fontsize=9, fontweight="bold")
        fig.text(0.708, 0.116, "25 puntos · carga 0 · doblete", color=MUTED, fontsize=8)
        stage = "REACTIVOS" if scan_t < 0.32 else "REGIÓN DE TRANSICIÓN" if scan_t < 0.68 else "PRODUCTOS"
        fig.text(0.055, 0.102, stage, color=CYAN if stage != "REGIÓN DE TRANSICIÓN" else GOLD, fontsize=9, fontweight="bold")
        save_frame(fig, target_dir / f"frame_{frame:04d}.png")
    return encode_video(target_dir, VIDEOS / "02_leonardo_qst_cinematic.mp4", tone=73)


def render_md(summary: dict) -> Path:
    print("[6/6] Rendering video 3: OpenMM dynamics", flush=True)
    data = np.load(RESULTS / "md_render_data.npz")
    ca_frames = data["ca_frames"]
    heavy_frames = data["heavy_frames"]
    trp_idx = data["trp_indices_heavy"]
    times = data["times_ps"]
    rmsd = data["rmsd_A"]
    energy = data["potential_kj_mol"]
    center = ca_frames[0].mean(axis=0)
    scale = float(np.max(np.linalg.norm(ca_frames[0] - center, axis=1)))
    target_dir = FRAMES / "03_md"
    target_dir.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("leonardo", [CYAN, BLUE, MAGENTA])
    for frame in range(N_FRAMES):
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
        ax = fig.add_axes([0.00, 0.03, 0.71, 0.88], projection="3d")
        style_3d(ax)
        trajectory_index = min(frame, len(ca_frames) - 1)
        ca = (ca_frames[trajectory_index] - center) / scale
        heavy = (heavy_frames[trajectory_index] - center) / scale
        ax.scatter(heavy[:, 0], heavy[:, 1], heavy[:, 2], s=6, c="#83a1c7", alpha=0.12, depthshade=False)
        for idx in range(len(ca) - 1):
            segment = ca[idx : idx + 2]
            color = cmap(idx / max(1, len(ca) - 2))
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=7, alpha=0.12)
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, lw=2.2, alpha=0.95)
        trp = heavy[trp_idx]
        ax.scatter(trp[:, 0], trp[:, 1], trp[:, 2], s=46, c=GOLD, alpha=0.90, edgecolors=WHITE, linewidths=0.25)
        ax.view_init(elev=17 + 6 * math.sin(frame / 30), azim=22 + frame * 0.52)
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-1.08, 1.08)
        ax.set_zlim(-1.08, 1.08)
        add_brand(fig, "03 / MOLECULAR DYNAMICS", frame)
        cinematic_title(fig, "Movimiento molecular real.", "Trp-cage · OpenMM · Amber14 + GBN2 · 300 K", frame)
        panel = make_panel(fig, [0.72, 0.205, 0.24, 0.26])
        shown = trajectory_index + 1
        panel.plot(times[:shown], rmsd[:shown], color=CYAN, lw=2.0, label="RMSD Cα")
        panel.set_title("ESTABILIDAD ESTRUCTURAL", loc="left", color=WHITE, fontsize=8.5, fontweight="bold", pad=10)
        panel.set_xlabel("tiempo (ps)", color=MUTED, fontsize=7)
        panel.set_ylabel("RMSD Cα (Å)", color=MUTED, fontsize=7)
        panel.grid(color="#24324a", alpha=0.35, lw=0.5)
        energy_panel = make_panel(fig, [0.72, 0.082, 0.24, 0.09])
        energy_panel.plot(times[:shown], energy[:shown], color=GOLD, lw=1.2)
        energy_panel.set_yticks([])
        energy_panel.set_xlabel("ENERGÍA POTENCIAL", color=MUTED, fontsize=6.5, labelpad=2)
        fig.text(0.055, 0.105, f"{times[trajectory_index]:05.1f} ps", color=GOLD, fontsize=11, fontweight="bold")
        fig.text(0.055, 0.078, f"RMSD Cα  {rmsd[trajectory_index]:.2f} Å   ·   {summary['atoms']} átomos", color=MUTED, fontsize=8)
        save_frame(fig, target_dir / f"frame_{frame:04d}.png")
    return encode_video(target_dir, VIDEOS / "03_leonardo_openmm_cinematic.mp4", tone=43)


def encode_video(frame_dir: Path, target: Path, tone: int) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the videos")
    duration = N_FRAMES / FPS
    audio_filter = (
        f"sine=frequency={tone}:sample_rate=48000:duration={duration},"
        f"volume=0.028,afade=t=in:st=0:d=1.2,afade=t=out:st={duration - 1.2}:d=1.2"
    )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(FPS),
        "-i",
        str(frame_dir / "frame_%04d.png"),
        "-f",
        "lavfi",
        "-i",
        audio_filter,
        "-vf",
        "scale=1920:1080:flags=lanczos,format=yuv420p",
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(command, check=True)
    return target


def write_manifest(summaries: dict, videos: list[Path]) -> None:
    import hashlib

    video_rows = []
    for path in videos:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        video_rows.append({"file": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest})
    manifest = {
        "title": "Leonardo molecular cinema campaign",
        "render": {"resolution": "1920x1080", "fps": 30, "duration_seconds_each": N_FRAMES / FPS},
        "scientific_results": summaries,
        "videos": video_rows,
        "disclosure": [
            "Docking is a reproducible rigid-body crystal-pocket search with a documented nonbonded score; it is not an experimental affinity.",
            "The QST video is a synchronous-transit scan with GFN2-xTB single-point energies, not a converged saddle-point frequency calculation.",
            "The MD video uses the actual saved OpenMM coordinates and MDAnalysis measurements from a short illustrative trajectory.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-only", action="store_true", help="Run calculations but do not render videos")
    parser.add_argument("--render-only", action="store_true", help="Render from existing result files")
    parser.add_argument("--keep-frames", action="store_true", help="Keep the intermediate PNG frame sequences")
    args = parser.parse_args()
    if args.compute_only and args.render_only:
        parser.error("--compute-only and --render-only cannot be combined")
    ensure_dirs()
    summaries: dict[str, dict] = {}
    if not args.render_only:
        summaries["docking"] = run_docking()
        summaries["qst"] = run_qst()
        summaries["molecular_dynamics"] = run_md()
    else:
        summaries = {
            "docking": json.loads((RESULTS / "docking_summary.json").read_text()),
            "qst": json.loads((RESULTS / "qst_summary.json").read_text()),
            "molecular_dynamics": json.loads((RESULTS / "md_summary.json").read_text()),
        }
    if args.compute_only:
        write_manifest(summaries, [])
        return 0
    videos = [
        render_docking(summaries["docking"]),
        render_qst(summaries["qst"]),
        render_md(summaries["molecular_dynamics"]),
    ]
    write_manifest(summaries, videos)
    if not args.keep_frames and FRAMES.exists():
        shutil.rmtree(FRAMES)
    print("Created:", flush=True)
    for path in videos:
        print(f"  {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
