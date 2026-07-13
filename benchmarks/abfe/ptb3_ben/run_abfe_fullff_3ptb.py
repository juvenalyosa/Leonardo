#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from openmm import unit as u
from openmm import (
    System,
    CustomExternalForce,
    CustomNonbondedForce,
    HarmonicBondForce,
    HarmonicAngleForce,
    PeriodicTorsionForce,
    RBTorsionForce,
    NonbondedForce,
    LangevinMiddleIntegrator,
    LangevinIntegrator,
    LocalEnergyMinimizer,
    Context,
    Platform,
)
from openmm import app
from openmm.app import PDBFile, AmberPrmtopFile, DCDReporter

NB_PARAMS = {
    "H": (0.120, 0.065),
    "C": (0.340, 0.276),
    "N": (0.325, 0.200),
    "O": (0.300, 0.210),
    "S": (0.360, 1.046),
    "P": (0.380, 0.837),
    "F": (0.305, 0.250),
    "CL": (0.347, 1.108),
    "BR": (0.373, 1.380),
    "I": (0.396, 1.670),
    "ZN": (0.220, 0.080),
    "MG": (0.200, 0.060),
    "CA": (0.240, 0.060),
    "FE": (0.220, 0.090),
    "MN": (0.230, 0.070),
    "CU": (0.220, 0.090),
    "NI": (0.220, 0.090),
    "CO": (0.220, 0.090),
    "NA": (0.250, 0.030),
    "K": (0.280, 0.020),
}

MASS_MAP = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
    "F": 18.998,
    "CL": 35.45,
    "BR": 79.904,
    "I": 126.90,
    "NA": 22.99,
    "K": 39.098,
    "MG": 24.305,
    "CA": 40.078,
    "ZN": 65.38,
    "FE": 55.845,
    "MN": 54.938,
    "CU": 63.546,
    "NI": 58.693,
    "CO": 58.933,
}

POS_RES = {"LYS", "ARG"}
NEG_RES = {"ASP", "GLU"}
HIS_RES = {"HIS", "HID", "HIE", "HIP", "HSD", "HSE", "HSP"}


def infer_element(atom: app.topology.Atom) -> str:
    try:
        e = atom.element.symbol.strip().upper()
        if e:
            return e
    except Exception:
        pass
    name = str(atom.name or "").strip().upper()
    for two in ("CL", "BR", "NA", "ZN", "MG", "CA", "FE", "MN", "CU", "NI", "CO"):
        if name.startswith(two):
            return two
    return name[0] if name else "C"


def nb_params(el: str) -> tuple[float, float]:
    e = str(el or "").strip().upper()
    if e in NB_PARAMS:
        return NB_PARAMS[e]
    if e and e[0] in NB_PARAMS:
        return NB_PARAMS[e[0]]
    return NB_PARAMS["C"]


def mass_for_element(el: str) -> float:
    e = str(el or "").strip().upper()
    if e in MASS_MAP:
        return float(MASS_MAP[e])
    if e and e[0] in MASS_MAP:
        return float(MASS_MAP[e[0]])
    return 12.011


def estimate_protein_charge(resn: str, atom_name: str, element: str) -> float:
    resn_u = str(resn).strip().upper()
    an = str(atom_name).strip().upper()
    el = str(element).strip().upper()

    if resn_u in POS_RES:
        if resn_u == "LYS" and an == "NZ":
            return 0.8
        if resn_u == "ARG" and an in {"NE", "NH1", "NH2", "CZ"}:
            return 0.5
    if resn_u in NEG_RES and an.startswith(("OD", "OE")):
        return -0.6
    if resn_u in HIS_RES and an in {"ND1", "NE2"}:
        return 0.2

    if an == "N":
        return 0.2
    if an == "O":
        return -0.2

    if el == "N":
        return -0.05
    if el == "O":
        return -0.10
    return 0.0


def load_mol2_charges(mol2_file: Path) -> dict[str, float]:
    out = {}
    sec = None
    for raw in mol2_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@<TRIPOS>"):
            sec = line.upper()
            continue
        if sec == "@<TRIPOS>ATOM":
            parts = line.split()
            if len(parts) < 9:
                continue
            nm = parts[1].strip().upper()
            try:
                q = float(parts[8])
            except Exception:
                q = 0.0
            out[nm] = q
    return out


def pick_platform() -> tuple[Platform, dict, str]:
    for name in ("CUDA", "OpenCL", "Metal", "CPU"):
        try:
            p = Platform.getPlatformByName(name)
            props = {}
            if name in {"CUDA", "OpenCL"}:
                props = {"Precision": "mixed"}
            return p, props, name
        except Exception:
            continue
    raise RuntimeError("No OpenMM platform available")


def build_context_with_fallback(system: System, integrator):
    last_err = None
    for name in ("CUDA", "OpenCL", "Metal", "CPU"):
        try:
            platform = Platform.getPlatformByName(name)
        except Exception as e:
            last_err = e
            continue
        props = {}
        if name in {"CUDA", "OpenCL"}:
            props = {"Precision": "mixed"}
        try:
            if props:
                ctx = Context(system, integrator, platform, props)
            else:
                ctx = Context(system, integrator, platform)
            return ctx, platform, props, name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No compatible OpenMM platform for this system: {last_err}")


def inject_ligand_terms_from_prmtop(system: System, local_lig: list[int], local_infos: list[dict], prmtop_file: Path) -> dict:
    prmtop = AmberPrmtopFile(str(prmtop_file))
    prm_atoms = list(prmtop.topology.atoms())
    prm_names = [str(a.name or "").strip().upper() for a in prm_atoms]
    loc_names = [str(local_infos[i]["name"] or "").strip().upper() for i in local_lig]

    if len(prm_names) != len(loc_names):
        raise RuntimeError(f"Ligand atom count mismatch: prmtop={len(prm_names)} vs structure={len(loc_names)}")
    if prm_names != loc_names:
        raise RuntimeError("Ligand atom names/order mismatch between prmtop and complex")

    param_to_local = {i: int(local_lig[i]) for i in range(len(local_lig))}
    lig_sys = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )

    counts = {"bond": 0, "angle": 0, "dihedral": 0, "rb_torsion": 0, "nonbonded_exceptions": 0}

    for force in lig_sys.getForces():
        if isinstance(force, HarmonicBondForce):
            f = HarmonicBondForce()
            for ii in range(force.getNumBonds()):
                a, b, length, kval = force.getBondParameters(ii)
                f.addBond(param_to_local[int(a)], param_to_local[int(b)], length, kval)
                counts["bond"] += 1
            if f.getNumBonds() > 0:
                system.addForce(f)
        elif isinstance(force, HarmonicAngleForce):
            f = HarmonicAngleForce()
            for ii in range(force.getNumAngles()):
                a, b, c, theta, kval = force.getAngleParameters(ii)
                f.addAngle(param_to_local[int(a)], param_to_local[int(b)], param_to_local[int(c)], theta, kval)
                counts["angle"] += 1
            if f.getNumAngles() > 0:
                system.addForce(f)
        elif isinstance(force, PeriodicTorsionForce):
            f = PeriodicTorsionForce()
            for ii in range(force.getNumTorsions()):
                a, b, c, d, per, phase, kval = force.getTorsionParameters(ii)
                f.addTorsion(param_to_local[int(a)], param_to_local[int(b)], param_to_local[int(c)], param_to_local[int(d)], per, phase, kval)
                counts["dihedral"] += 1
            if f.getNumTorsions() > 0:
                system.addForce(f)
        elif isinstance(force, RBTorsionForce):
            f = RBTorsionForce()
            for ii in range(force.getNumTorsions()):
                a, b, c, d, c0, c1, c2, c3, c4, c5 = force.getTorsionParameters(ii)
                f.addTorsion(param_to_local[int(a)], param_to_local[int(b)], param_to_local[int(c)], param_to_local[int(d)], c0, c1, c2, c3, c4, c5)
                counts["rb_torsion"] += 1
            if f.getNumTorsions() > 0:
                system.addForce(f)
            elif isinstance(force, NonbondedForce):
                # Skip intraligand nonbonded transfer here to keep ABFE-lite stable/fast
                # with separate custom protein-ligand interaction forces.
                continue

    return counts


def main():
    ap = argparse.ArgumentParser(description="ABFE-like full-ligand-FF validation on 3PTB/BEN")
    ap.add_argument("--pdb", default="3PTB_BEN_bound_openmm_prepared.pdb")
    ap.add_argument("--prmtop", default="benchmarks/abfe/ptb3_ben/BEN_from_complex.prmtop")
    ap.add_argument("--mol2", default="benchmarks/abfe/ptb3_ben/BEN_from_complex_gaff2.mol2")
    ap.add_argument("--sim-ns", type=float, default=3.0)
    ap.add_argument("--sample-every-ps", type=float, default=2.0)
    ap.add_argument("--temperature-k", type=float, default=300.0)
    ap.add_argument("--pull-distance-a", type=float, default=12.0)
    ap.add_argument("--pull-k", type=float, default=800.0)
    ap.add_argument("--out-prefix", default="benchmarks/abfe/ptb3_ben/abfe_fullff_3ptb")
    ap.add_argument("--minimize-iters", type=int, default=1200)
    ap.add_argument("--eq-ps", type=float, default=20.0)
    args = ap.parse_args()

    pdb_path = Path(args.pdb)
    prmtop_path = Path(args.prmtop)
    mol2_path = Path(args.mol2)
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)
    if not prmtop_path.exists():
        raise FileNotFoundError(prmtop_path)
    if not mol2_path.exists():
        raise FileNotFoundError(mol2_path)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_dcd = Path(str(out_prefix) + ".dcd")
    out_pdb = Path(str(out_prefix) + "_final.pdb")

    pdb = PDBFile(str(pdb_path))
    top = pdb.topology
    xyz0 = np.array(pdb.positions.value_in_unit(u.nanometer), dtype=float)

    atoms = list(top.atoms())
    n_atoms = len(atoms)

    local_infos = []
    local_lig = []
    local_lig_heavy = []
    local_prot = []
    local_prot_heavy = []

    lig_charge_map = load_mol2_charges(mol2_path)

    sigmas = np.zeros(n_atoms, dtype=float)
    epsilons = np.zeros(n_atoms, dtype=float)
    charges = np.zeros(n_atoms, dtype=float)
    masses = np.zeros(n_atoms, dtype=float)

    for i, a in enumerate(atoms):
        res = a.residue
        resn = str(res.name).strip().upper()
        name = str(a.name).strip()
        el = infer_element(a)
        origin = "ligand" if resn == "BEN" else "protein"

        info = {
            "name": name,
            "resn": resn,
            "chain": str(res.chain.id),
            "resi": str(res.id),
            "element": el,
            "origin": origin,
        }
        local_infos.append(info)

        if origin == "ligand":
            local_lig.append(i)
            if el != "H":
                local_lig_heavy.append(i)
        else:
            local_prot.append(i)
            if el != "H":
                local_prot_heavy.append(i)

        sigma, eps = nb_params(el)
        sigmas[i] = sigma
        epsilons[i] = eps
        masses[i] = mass_for_element(el)

        if origin == "ligand":
            q = float(lig_charge_map.get(name.strip().upper(), 0.0))
        else:
            q = estimate_protein_charge(resn, name, el)
        charges[i] = q

    if not local_lig or not local_prot:
        raise RuntimeError("Failed to identify ligand/protein atoms")

    print(f"Atoms total: {n_atoms} | protein={len(local_prot)} | ligand={len(local_lig)}")

    system = System()
    for m in masses:
        system.addParticle(max(1.0, float(m)) * u.dalton)

    # Full ligand FF from prmtop.
    ff_counts = inject_ligand_terms_from_prmtop(system, local_lig, local_infos, prmtop_path)
    print("Ligand FF terms from prmtop:", ff_counts)

    # Protein positional restraints.
    rest = CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    rest.addPerParticleParameter("k")
    rest.addPerParticleParameter("x0")
    rest.addPerParticleParameter("y0")
    rest.addPerParticleParameter("z0")

    n_rest = 0
    for i in local_prot:
        info = local_infos[i]
        el = str(info["element"]).upper()
        if el == "H":
            continue
        an = str(info["name"]).strip().upper()
        k_rest = 3600.0 if an in ("N", "CA", "C", "O") else 1300.0
        x0, y0, z0 = xyz0[i]
        rest.addParticle(int(i), [float(k_rest), float(x0), float(y0), float(z0)])
        n_rest += 1
    system.addForce(rest)
    print("Protein restrained atoms:", n_rest)

    # Temporary ligand hold for minimization.
    hold = CustomExternalForce("0.5*k_hold*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    hold.addGlobalParameter("k_hold", 0.0)
    hold.addPerParticleParameter("x0")
    hold.addPerParticleParameter("y0")
    hold.addPerParticleParameter("z0")
    for i in local_lig_heavy:
        x0, y0, z0 = xyz0[i]
        hold.addParticle(int(i), [float(x0), float(y0), float(z0)])
    system.addForce(hold)

    # Protein-ligand interaction terms (same style as plugin ABFE-lite).
    lj = CustomNonbondedForce("4*sqrt(epsilon1*epsilon2)*((0.5*(sigma1+sigma2)/r)^12-(0.5*(sigma1+sigma2)/r)^6)")
    lj.addPerParticleParameter("sigma")
    lj.addPerParticleParameter("epsilon")
    lj.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    lj.setCutoffDistance(1.2 * u.nanometer)
    lj.addInteractionGroup(set(int(i) for i in local_prot), set(int(i) for i in local_lig))
    lj.setForceGroup(1)

    coul = CustomNonbondedForce("138.935456*q1*q2/r")
    coul.addPerParticleParameter("q")
    coul.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    coul.setCutoffDistance(1.2 * u.nanometer)
    coul.addInteractionGroup(set(int(i) for i in local_prot), set(int(i) for i in local_lig))
    coul.setForceGroup(2)

    for i in range(n_atoms):
        lj.addParticle([float(sigmas[i]), float(epsilons[i])])
        coul.addParticle([float(charges[i])])
    system.addForce(lj)
    system.addForce(coul)

    # Steering pull/hold/return.
    lig_heavy_idx = np.asarray(local_lig_heavy if local_lig_heavy else local_lig, dtype=int)
    prot_heavy_idx = np.asarray(local_prot_heavy if local_prot_heavy else local_prot, dtype=int)

    lig_com0 = np.mean(xyz0[lig_heavy_idx], axis=0)
    prot_com0 = np.mean(xyz0[prot_heavy_idx], axis=0)
    away = lig_com0 - prot_com0
    n_away = np.linalg.norm(away)
    if n_away < 1e-8:
        away = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        away = away / n_away

    pull_dist_nm = float(args.pull_distance_a) * 0.1
    pull_vec = away * pull_dist_nm
    pull_particles_all = [int(i) for i in lig_heavy_idx.tolist()]
    pull_particles = list(pull_particles_all)
    if len(pull_particles) > 3:
        d0 = np.linalg.norm(xyz0[pull_particles] - lig_com0[None, :], axis=1)
        keep = np.argsort(d0)[:3]
        pull_particles = [pull_particles[int(k)] for k in keep]

    k_total = max(200.0, float(args.pull_k))
    k_per_atom = k_total / float(max(1, len(pull_particles)))
    steer = CustomExternalForce("0.5*k_pull*((x-(x0+lam*dx))^2 + (y-(y0+lam*dy))^2 + (z-(z0+lam*dz))^2)")
    steer.addGlobalParameter("k_pull", float(k_per_atom))
    steer.addGlobalParameter("lam", 0.0)
    steer.addPerParticleParameter("x0")
    steer.addPerParticleParameter("y0")
    steer.addPerParticleParameter("z0")
    steer.addPerParticleParameter("dx")
    steer.addPerParticleParameter("dy")
    steer.addPerParticleParameter("dz")
    for i in pull_particles:
        x0, y0, z0 = xyz0[i]
        steer.addParticle(i, [float(x0), float(y0), float(z0), float(pull_vec[0]), float(pull_vec[1]), float(pull_vec[2])])
    steer.setForceGroup(3)
    system.addForce(steer)

    site_mask = []
    for i in prot_heavy_idx.tolist():
        d = np.min(np.linalg.norm(xyz0[int(i)] - xyz0[lig_heavy_idx], axis=1))
        if d <= 0.65:
            site_mask.append(int(i))
    if site_mask:
        p_idx0 = np.asarray(site_mask, dtype=int)
        d0 = np.linalg.norm(xyz0[p_idx0] - lig_com0[None, :], axis=1)
        idx = np.argsort(d0)[: min(96, len(d0))]
        site_mask = [int(p_idx0[int(k)]) for k in idx]
    else:
        d_all = np.linalg.norm(xyz0[prot_heavy_idx] - lig_com0[None, :], axis=1)
        idx = np.argsort(d_all)[: min(64, len(d_all))]
        site_mask = [int(prot_heavy_idx[int(k)]) for k in idx]
    site_anchor_idx = np.asarray(sorted(set(int(i) for i in site_mask)), dtype=int)
    if site_anchor_idx.size <= 0:
        raise RuntimeError('Failed to define site anchors')
    site_com0 = np.mean(xyz0[site_anchor_idx], axis=0)
    dv0 = xyz0[site_anchor_idx][:, None, :] - xyz0[lig_heavy_idx][None, :, :]
    d0_site_min_a = float(np.min(np.linalg.norm(dv0, axis=2)) * 10.0)
    d0_site_com_a = float(np.linalg.norm(lig_com0 - site_com0) * 10.0)
    print(f'Site anchors: n={site_anchor_idx.size} initial_site_min={d0_site_min_a:.3f} A initial_site_com={d0_site_com_a:.3f} A')

    timestep_ps = 0.001
    try:
        integ = LangevinMiddleIntegrator(float(args.temperature_k) * u.kelvin, 5.0 / u.picosecond, timestep_ps * u.picoseconds)
    except Exception:
        integ = LangevinIntegrator(float(args.temperature_k) * u.kelvin, 5.0 / u.picosecond, timestep_ps * u.picoseconds)

    ctx, platform, props, pname = build_context_with_fallback(system, integ)
    print(f"OpenMM platform: {pname} props={props}")

    ctx.setPositions(xyz0 * u.nanometer)
    ctx.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin)

    # Minimization with temporary ligand hold.
    max_it = max(0, int(args.minimize_iters))
    if max_it > 0:
        ctx.setParameter("k_hold", 1800.0)
        LocalEnergyMinimizer.minimize(ctx, maxIterations=max_it)
        ctx.setParameter("k_hold", 0.0)
        LocalEnergyMinimizer.minimize(ctx, maxIterations=max(200, max_it // 2))

    # Build simulation wrapper for reporter convenience with a fresh integrator.
    min_positions = ctx.getState(getPositions=True).getPositions()
    try:
        integ_run = LangevinMiddleIntegrator(
            float(args.temperature_k) * u.kelvin,
            5.0 / u.picosecond,
            timestep_ps * u.picoseconds,
        )
    except Exception:
        integ_run = LangevinIntegrator(
            float(args.temperature_k) * u.kelvin,
            5.0 / u.picosecond,
            timestep_ps * u.picoseconds,
        )

    sim = app.Simulation(top, system, integ_run, platform, props)
    sim.context.setPositions(min_positions)
    sim.context.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin)

    if out_dcd.exists():
        out_dcd.unlink()
    stride = max(1, int(round(float(args.sample_every_ps) / timestep_ps)))
    sim.reporters.append(DCDReporter(str(out_dcd), stride, enforcePeriodicBox=False))

    # Equilibration.
    eq_ps = max(0.0, float(args.eq_ps))
    if eq_ps > 0.0:
        eq_steps = max(1, int(round(eq_ps / timestep_ps)))
        sim.step(eq_steps)

    sim_ns = max(0.2, float(args.sim_ns))
    prod_steps = int(round((sim_ns * 1000.0) / timestep_ps))
    n_samples = max(1, prod_steps // stride)

    # pull/hold/return schedule
    lam = np.zeros(n_samples, dtype=float)
    if n_samples == 1:
        lam[0] = 1.0
    else:
        n_pull = max(1, int(round(0.45 * n_samples)))
        n_hold = max(1, int(round(0.10 * n_samples)))
        n_ret = max(1, n_samples - n_pull - n_hold)
        while n_pull + n_hold + n_ret > n_samples:
            if n_hold > 1:
                n_hold -= 1
            elif n_pull > 1:
                n_pull -= 1
            elif n_ret > 1:
                n_ret -= 1
            else:
                break
        while n_pull + n_hold + n_ret < n_samples:
            n_ret += 1
        lam_pull = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, n_pull, endpoint=True, dtype=float)))
        lam_hold = np.ones(n_hold, dtype=float)
        lam_ret = 0.5 * (1.0 + np.cos(np.linspace(0.0, math.pi, n_ret, endpoint=True, dtype=float)))
        lam = np.concatenate([lam_pull, lam_hold, lam_ret], axis=0)[:n_samples]

    print(
        f"ABFE run: sim_ns={sim_ns:.2f}, stride={stride} steps ({args.sample_every_ps:.2f} ps), "
        f"samples={n_samples}, pull_atoms={len(pull_particles)}, k_total={k_total:.1f}, k_per_atom={k_per_atom:.1f}"
    )

    energies = []
    dists = []
    dists_site_min = []
    dists_site_com = []
    times = []

    lig_mass = np.asarray([max(1.0, masses[i]) for i in lig_heavy_idx], dtype=float)

    for sidx in range(n_samples):
        sim.context.setParameter("lam", float(lam[sidx]))
        sim.step(stride)

        st_pos = sim.context.getState(getPositions=True)
        xyz = np.array(st_pos.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=float)

        st_lj = sim.context.getState(getEnergy=True, groups=(1 << 1))
        st_c = sim.context.getState(getEnergy=True, groups=(1 << 2))
        e_lj = float(st_lj.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        e_c = float(st_c.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        e_t = e_lj + e_c

        lxyz = xyz[lig_heavy_idx]
        com = np.average(lxyz, axis=0, weights=lig_mass)
        rc_com_a = float(np.linalg.norm(com - site_com0) * 10.0)

        dv_site = xyz[site_anchor_idx][:, None, :] - xyz[lig_heavy_idx][None, :, :]
        rc_site_min_a = float(np.min(np.linalg.norm(dv_site, axis=2)) * 10.0)

        # Primary reactive coordinate aligned with plugin fix.
        rc_a = float(rc_site_min_a)

        energies.append(e_t)
        dists.append(rc_a)
        dists_site_min.append(rc_site_min_a)
        dists_site_com.append(rc_com_a)
        times.append((sidx + 1) * float(args.sample_every_ps))

        if (sidx + 1) % 50 == 0 or sidx == 0 or sidx == n_samples - 1:
            print(
                f"sample {sidx+1:4d}/{n_samples}  t={times[-1]:8.1f} ps  E={e_t:10.3f} kJ/mol  "
                f"rc(min/site)={rc_site_min_a:7.3f} A  rc(com/site)={rc_com_a:7.3f} A"
            )

    state = sim.context.getState(getPositions=True)
    with out_pdb.open("w") as fh:
        PDBFile.writeFile(sim.topology, state.getPositions(), fh, keepIds=True)

    arr_e = np.asarray(energies, dtype=float)
    arr_d = np.asarray(dists, dtype=float)
    arr_d_site_min = np.asarray(dists_site_min, dtype=float)
    arr_d_site_com = np.asarray(dists_site_com, dtype=float)
    if np.any(~np.isfinite(arr_e)):
        print("ERROR: Non-finite energies detected")
    if np.any(~np.isfinite(arr_d)):
        print("ERROR: Non-finite distances detected")

    n_bound = int(np.sum(arr_d <= 4.5))
    n_unbound = int(np.sum(arr_d > 4.5))
    print("\n=== ABFE Validation Summary ===")
    print(f"Frames: {len(arr_e)}")
    print(f"Energy (kJ/mol): min={np.min(arr_e):.3f} max={np.max(arr_e):.3f} mean={np.mean(arr_e):.3f}")
    print(
        f"Reactive distance site-min (A): min={np.min(arr_d_site_min):.3f} "
        f"max={np.max(arr_d_site_min):.3f} mean={np.mean(arr_d_site_min):.3f}"
    )
    print(
        f"Diagnostic COM->site (A): min={np.min(arr_d_site_com):.3f} "
        f"max={np.max(arr_d_site_com):.3f} mean={np.mean(arr_d_site_com):.3f}"
    )
    print(f"Bound/unbound (4.5A site-min cutoff): {n_bound}/{n_unbound}")
    print(f"DCD: {out_dcd}")
    print(f"Final PDB: {out_pdb}")


if __name__ == "__main__":
    main()
