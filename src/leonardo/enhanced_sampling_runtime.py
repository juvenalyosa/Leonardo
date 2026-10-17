"""
Enhanced sampling on real systems with OpenMM: umbrella sampling, well-tempered metadynamics, accelerated MD
(aMD) and Gaussian accelerated MD (GaMD), each with its analysis (PMF/free-energy surface, reweighting,
sampling diagnostics) and plots.

Everything runs on the GPU OpenMM picks (CUDA, HIP, OpenCL/Metal; CPU otherwise): the collective variables are
OpenMM custom forces, the umbrella restraint a CustomCVForce whose centre and spring are context parameters
(all windows share one context), metadynamics OpenMM's Metadynamics class, and aMD/GaMD a Langevin
CustomIntegrator with the boost inside its force term (OpenMM compiles all of these into GPU kernels).

The job runs in its own process, like the membrane jobs:

    python -m leonardo.enhanced_sampling_runtime job.json

job.json: {"method": "umbrella|metadynamics|amd|gamd", "pdb": ..., "out_dir": ..., "result_json": ...,
"system": {...}, "cvs": [{"name", "type": distance|angle|dihedral|rmsd, "atoms": [[[chain, resi, icode, name],
...], ...], ...}], plus the method's block}. Progress lines start with "PROGRESS ".

Short runs give qualitative answers: every result carries its sampling diagnostics (window overlap,
transitions, hills, boost anharmonicity) and says it is not converged.
"""
import json
import math
import os
import sys
import time
import traceback

import numpy as np

KB_KJ_MOL_K = 0.0083144626           # Boltzmann constant, kJ/mol/K
KJ_PER_KCAL = 4.184                  # every kcal parameter (GaMD sigma0, aMD offsets) is converted here only
DIHEDRAL_GROUP = 1                   # torsion forces: the dihedral boost of aMD/GaMD acts on this force group
BIAS_GROUP = 2                       # CV, restraint and metadynamics bias forces


def emit(msg):
    print(f"PROGRESS {msg}", flush=True)


def _import_sibling(name):
    try:
        return __import__(f"leonardo.{name}", fromlist=[name])
    except ImportError:
        return __import__(name)


# --------------------------------------------------------------------------------------------- system
def build_system(pdb_path, solvent="implicit", family="amber14", ph=7.0, keep_waters=False, remove_resn=(),
                 padding_nm=1.0, ionic_strength_molar=0.15, temperature_k=300.0, pressure_bar=1.0):
    """Leonardo's component-aware preparation (ligands, cofactors, metals) and an OpenMM system:
    implicit GBn2 (amber14 only), explicit TIP3P with PME and a barostat, or gas phase.
    Returns (topology, system, positions, info)."""
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    cl = _import_sibling("component_library_runtime")
    prep = cl.prepare_system_for_solvation(pdb_path, family=family, ph=ph, keep_waters=keep_waters,
                                           remove_resn=tuple(remove_resn or ()), remove_additives=True)
    if not prep.get("ok"):
        raise RuntimeError(f"system preparation failed: {prep.get('error') or prep.get('errors') or prep}")
    modeller, ff = prep["modeller"], prep["forcefield"]
    solvent = str(solvent or "implicit").lower()
    if solvent == "explicit":
        # PME needs every box edge >= 2 x cutoff: a small solute (a dipeptide, a ligand) with the usual padding
        # gives a box OpenMM rejects, so such boxes are enlarged to 2.4 nm
        xyz = np.asarray(modeller.positions.value_in_unit(unit.nanometer), float)
        edge = float(np.max(xyz.max(0) - xyz.min(0))) + 2.0 * float(padding_nm)
        if edge < 2.4:
            modeller.addSolvent(ff, boxSize=mm.Vec3(2.4, 2.4, 2.4) * unit.nanometer,
                                ionicStrength=float(ionic_strength_molar) * unit.molar)
        else:
            modeller.addSolvent(ff, padding=float(padding_nm) * unit.nanometer,
                                ionicStrength=float(ionic_strength_molar) * unit.molar)
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                                 constraints=app.HBonds)
        system.addForce(mm.MonteCarloBarostat(float(pressure_bar) * unit.bar, float(temperature_k) * unit.kelvin))
    elif solvent == "implicit":
        if not str(family).lower().startswith("amber"):
            raise RuntimeError("implicit solvent (GBn2) is available with the amber14 force field only; "
                               "use explicit solvent with charmm36")
        ff.loadFile("implicit/gbn2.xml")
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.CutoffNonPeriodic,
                                 nonbondedCutoff=2.0 * unit.nanometer, constraints=app.HBonds)
    else:
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    for force in system.getForces():
        if isinstance(force, (mm.PeriodicTorsionForce, mm.CMAPTorsionForce, mm.RBTorsionForce)):
            force.setForceGroup(DIHEDRAL_GROUP)
        elif force.getForceGroup() in (DIHEDRAL_GROUP, BIAS_GROUP):
            force.setForceGroup(0)
    info = {"n_atoms": int(modeller.topology.getNumAtoms()), "solvent": solvent, "family": family,
            "n_residues_solute": int(sum(1 for r in modeller.topology.residues()
                                         if r.name not in ("HOH", "WAT", "NA", "CL", "K", "Na+", "Cl-")))}
    return modeller.topology, system, modeller.positions, info


def _atom_key(chain, resi, icode, name):
    return (str(chain or "").strip(), str(resi).strip(), str(icode or "").strip(), str(name).strip())


def map_atom_keys(topology, groups):
    """[[(chain, resi, icode, name), ...], ...] -> [[topology index, ...], ...]. Hydrogens are added and
    residues renamed (HIS->HIE, CYS->CYX) between PyMOL and OpenMM: heavy-atom keys survive both. A key that is
    not found is an error (a CV on the wrong atoms is worse than no CV)."""
    index = {}
    loose = {}
    for atom in topology.atoms():
        res = atom.residue
        key = _atom_key(res.chain.id, res.id, getattr(res, "insertionCode", ""), atom.name)
        index[key] = atom.index
        loose.setdefault((key[1], key[2], key[3]), []).append(atom.index)
    out, missing = [], []
    for group in groups:
        idx = []
        for key in group:
            k = _atom_key(*key)
            if k in index:
                idx.append(index[k])
            elif len(loose.get(k[1:], [])) == 1:          # chain ids blanked or renamed (tleap, PDBFixer)
                idx.append(loose[k[1:]][0])
            else:
                missing.append("/".join(x for x in k if x))
        out.append(idx)
    if missing:
        raise RuntimeError(f"{len(missing)} CV atom(s) not found in the prepared system: {', '.join(missing[:8])}")
    if any(not g for g in out):
        raise RuntimeError("a CV group has no atoms")
    return out


# ------------------------------------------------------------------------------------ collective variables
_CV_ARITY = {"distance": 2, "angle": 3, "dihedral": 4}
_CV_EXPR = {"distance": "distance(g1,g2)", "angle": "angle(g1,g2,g3)", "dihedral": "dihedral(g1,g2,g3,g4)"}


def cv_force(cv, groups, positions_nm=None, periodic=False):
    """An OpenMM force whose energy is the CV value (nm or rad): centroid distance/angle/dihedral of atom
    groups (mass-weighted), or the RMSD of group 1 from the starting positions."""
    import openmm as mm

    typ = str(cv.get("type", "distance")).lower()
    if typ == "rmsd":
        ref = np.asarray(positions_nm, float)
        force = mm.RMSDForce([mm.Vec3(*p) for p in ref], [int(i) for i in groups[0]])
    else:
        n = _CV_ARITY[typ]
        if len(groups) != n:
            raise RuntimeError(f"a {typ} CV needs {n} atom groups, got {len(groups)}")
        force = mm.CustomCentroidBondForce(n, _CV_EXPR[typ])
        for g in groups:
            force.addGroup([int(i) for i in g])
        force.addBond(list(range(n)), [])
        force.setUsesPeriodicBoundaryConditions(bool(periodic))
    force.setForceGroup(BIAS_GROUP)
    return force


def cv_values_numpy(cv_type, groups, positions_nm, masses, box_nm=None, reference_nm=None):
    """The same CVs from coordinates (for aMD/GaMD frames, which carry no CV force)."""
    pos = np.asarray(positions_nm, float)
    mass = np.asarray(masses, float)

    def centroid(g):
        w = mass[g]
        return (pos[g] * w[:, None]).sum(0) / w.sum()

    def wrap(d):
        if box_nm is None:
            return d
        box = np.asarray(box_nm, float)
        return d - box * np.round(d / box)

    typ = str(cv_type).lower()
    if typ == "rmsd":
        g = np.asarray(groups[0], int)
        a = pos[g] - pos[g].mean(0)
        b = np.asarray(reference_nm, float)[g]
        b = b - b.mean(0)
        u, s, vt = np.linalg.svd(a.T @ b)
        d = np.sign(np.linalg.det(u @ vt))
        r = u @ np.diag([1, 1, d]) @ vt
        return float(np.sqrt(((a @ r - b) ** 2).sum(1).mean()))
    c = [centroid(np.asarray(g, int)) for g in groups]
    if typ == "distance":
        return float(np.linalg.norm(wrap(c[1] - c[0])))
    if typ == "angle":
        v1, v2 = wrap(c[0] - c[1]), wrap(c[2] - c[1])
        return float(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
    b1, b2, b3 = wrap(c[1] - c[0]), wrap(c[2] - c[1]), wrap(c[3] - c[2])
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    # OpenMM's sign convention (the dihedral() of its custom forces), so boosted-MD PMFs match umbrella/metaD
    return float(np.arctan2(np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2)), np.dot(n1, n2)))


def _periodic_delta(a, b, periodic):
    d = np.asarray(a, float) - np.asarray(b, float)
    return (d + np.pi) % (2 * np.pi) - np.pi if periodic else d


# ------------------------------------------------------------------------------------------ simulation
def _platform_helpers():
    mr = _import_sibling("membrane_runtime")
    return mr._simulation, mr.simulation_platform


def make_simulation(topology, system, integrator, positions, temperature_k, seed=0, minimize_steps=500):
    import openmm.app as app
    from openmm import unit

    simulation_fn, platform_fn = _platform_helpers()
    sim = simulation_fn(app, __import__("openmm"), topology, system, integrator)
    sim.context.setPositions(positions)
    if minimize_steps:
        sim.minimizeEnergy(maxIterations=int(minimize_steps))
    sim.context.setVelocitiesToTemperature(float(temperature_k) * unit.kelvin, int(seed) or 1)
    return sim, platform_fn(sim)


def langevin_integrator(temperature_k, friction_ps, timestep_fs, seed=0):
    import openmm as mm
    from openmm import unit

    integ = mm.LangevinMiddleIntegrator(float(temperature_k) * unit.kelvin, float(friction_ps) / unit.picosecond,
                                        float(timestep_fs) * unit.femtosecond)
    if seed:
        integ.setRandomNumberSeed(int(seed))
    return integ


def _ns_per_day(steps, timestep_fs, seconds):
    return round(steps * timestep_fs * 1e-6 / max(seconds, 1e-9) * 86400.0, 2)


# -------------------------------------------------------------------------------------- boosted dynamics
def boost_energy(kind, V, E, k_or_alpha):
    """Boost potential ΔV(V) (same units as V): GaMD 0.5 k (E-V)^2, aMD (E-V)^2 / (alpha + E-V); 0 when V >= E."""
    x = np.maximum(np.asarray(E, float) - np.asarray(V, float), 0.0)
    if kind == "gamd":
        return 0.5 * k_or_alpha * x ** 2
    return x ** 2 / (k_or_alpha + x)


def boost_force_scale(kind, V, E, k_or_alpha):
    """1 + dΔV/dV, the factor that multiplies the forces of the boosted potential."""
    x = np.maximum(np.asarray(E, float) - np.asarray(V, float), 0.0)
    if kind == "gamd":
        return 1.0 - k_or_alpha * x
    return k_or_alpha ** 2 / (k_or_alpha + x) ** 2


def boosted_langevin_integrator(kind, temperature_k, friction_ps, timestep_fs, total=None, dihedral=None, seed=0):
    """Langevin-middle integrator on V* = V + ΔV_total(V) + ΔV_dihedral(V_dih) (dual boost when both are given).
    total/dihedral: (E, k) for GaMD or (E, alpha) for aMD, kJ/mol. The force is
    f* = f (1 + ΔV_t'(V)) + f_dih ΔV_d'(V_dih), with f_dih the forces of force group DIHEDRAL_GROUP.
    OpenMM's own aMD integrators are NVE: this one keeps the temperature."""
    import openmm as mm
    from openmm import unit

    dt = float(timestep_fs) * 1e-3                                   # ps
    integ = mm.CustomIntegrator(dt * unit.picoseconds)
    kt = KB_KJ_MOL_K * float(temperature_k)
    a = math.exp(-float(friction_ps) * dt)
    integ.addGlobalVariable("kT", kt)
    integ.addGlobalVariable("a", a)
    integ.addGlobalVariable("b", math.sqrt(1.0 - a * a))
    for name, pair in (("t", total), ("d", dihedral)):
        on = pair is not None
        integ.addGlobalVariable(f"E{name}", float(pair[0]) if on else -1.0e30)
        integ.addGlobalVariable(f"p{name}", float(pair[1]) if on else 1.0)
        integ.addGlobalVariable(f"x{name}", 0.0)
        integ.addGlobalVariable(f"s{name}", 1.0)
    integ.addPerDofVariable("x1", 0.0)
    integ.addPerDofVariable("fd", 0.0)
    if seed:
        integ.setRandomNumberSeed(int(seed))
    integ.addUpdateContextState()
    integ.addComputeGlobal("xt", "max(0, Et - energy)")
    integ.addComputeGlobal("xd", f"max(0, Ed - energy{DIHEDRAL_GROUP})")
    if kind == "gamd":
        integ.addComputeGlobal("st", "1 - pt*xt")                      # 1 + dΔV_t/dV
        integ.addComputeGlobal("sd", "-pd*xd")                         # dΔV_d/dV_d
    else:
        integ.addComputeGlobal("st", "pt^2/(pt + xt)^2")
        integ.addComputeGlobal("sd", "-(xd^2 + 2*pd*xd)/(pd + xd)^2")
    # one computation cannot read two force groups: keep the dihedral forces in a per-DOF variable first
    integ.addComputePerDof("fd", f"f{DIHEDRAL_GROUP}")
    integ.addComputePerDof("v", "v + dt*(f*st + fd*sd)/m")
    integ.addConstrainVelocities()
    integ.addComputePerDof("x", "x + 0.5*dt*v")
    integ.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
    integ.addComputePerDof("x", "x + 0.5*dt*v")
    integ.addComputePerDof("x1", "x")
    integ.addConstrainPositions()
    integ.addComputePerDof("v", "v + (x - x1)/dt")
    return integ


def gamd_parameters(v_samples, sigma0_kcal=6.0):
    """GaMD lower-bound parameters (Miao et al. 2015) from conventional-MD potentials (kJ/mol):
    E = Vmax, k0 = min(1, sigma0/sigmaV * (Vmax-Vmin)/(Vmax-Vavg)), k = k0/(Vmax-Vmin)."""
    v = np.asarray(v_samples, float)
    vmax, vmin, vavg, sd = float(v.max()), float(v.min()), float(v.mean()), float(v.std())
    sigma0 = float(sigma0_kcal) * KJ_PER_KCAL
    span = max(vmax - vmin, 1e-6)
    k0 = min(1.0, (sigma0 / max(sd, 1e-9)) * span / max(vmax - vavg, 1e-9))
    return {"E": vmax, "k": k0 / span, "k0": k0, "Vmax": vmax, "Vmin": vmin, "Vavg": vavg, "sigmaV": sd,
            "sigma0_kj": sigma0}


def amd_parameters(v_total_avg, v_dih_avg, n_atoms, n_residues, level=1.0):
    """aMD dual-boost parameters (Pierce et al. 2012), kcal-based rules converted to kJ/mol:
    E_dih = <V_dih> + 3.5 N_res, alpha_dih = 3.5 N_res / 5; E_tot = <V_tot> + 0.175 N_atoms, alpha_tot = 0.175 N_atoms.
    `level` scales the offsets (0.5 = milder, 2 = stronger)."""
    lv = float(level)
    e_d = 3.5 * n_residues * KJ_PER_KCAL * lv
    e_t = 0.175 * n_atoms * KJ_PER_KCAL * lv
    return {"total": (float(v_total_avg) + e_t, max(e_t, 1e-3)),
            "dihedral": (float(v_dih_avg) + e_d, max(e_d / 5.0, 1e-3))}


def reweight_pmf(cv, dv, temperature_k, edges, periodic=False, min_count=10):
    """PMF along a CV from boosted frames (ΔV in kJ/mol): exponential reweighting and the 2nd-order cumulant
    expansion (the GaMD default; exact for a Gaussian ΔV). Bins with fewer than `min_count` frames are NaN."""
    beta = 1.0 / (KB_KJ_MOL_K * float(temperature_k))
    cv = np.asarray(cv, float)
    dv = np.asarray(dv, float)
    idx = np.digitize(cv, edges) - 1
    nb = len(edges) - 1
    counts = np.zeros(nb)
    exp_w = np.zeros(nb)
    c1 = np.full(nb, np.nan)
    c2 = np.full(nb, np.nan)
    shift = dv.max() if dv.size else 0.0
    for b in range(nb):
        sel = dv[idx == b]
        counts[b] = sel.size
        if sel.size:
            exp_w[b] = np.exp(beta * (sel - shift)).sum()
            c1[b] = sel.mean()
            c2[b] = sel.var()
    ok = counts >= min_count
    p_star = counts / max(counts.sum(), 1)
    pmf_cum = np.full(nb, np.nan)
    pmf_exp = np.full(nb, np.nan)
    pmf_raw = np.full(nb, np.nan)
    pmf_raw[ok] = -np.log(p_star[ok]) / beta
    pmf_cum[ok] = -np.log(p_star[ok]) / beta - (c1[ok] + 0.5 * beta * c2[ok])
    pmf_exp[ok] = -np.log(exp_w[ok]) / beta
    for arr in (pmf_cum, pmf_exp, pmf_raw):
        if np.any(np.isfinite(arr)):
            arr -= np.nanmin(arr)
    centers = 0.5 * (np.asarray(edges[:-1]) + np.asarray(edges[1:]))
    return {"centers": centers, "counts": counts, "pmf_cumulant2": pmf_cum, "pmf_exponential": pmf_exp,
            "pmf_unweighted": pmf_raw}


def boost_anharmonicity(dv, bins=50):
    """Miao's anharmonicity γ = S_gauss - S of the ΔV distribution (0 for a Gaussian; < ~0.01 is considered
    accurate for cumulant reweighting)."""
    dv = np.asarray(dv, float)
    if dv.size < 10 or dv.std() <= 0:
        return 0.0
    hist, edges = np.histogram(dv, bins=bins, density=True)
    width = edges[1] - edges[0]
    p = hist[hist > 0]
    s = -np.sum(p * np.log(p)) * width
    s_max = 0.5 * math.log(2 * math.pi * math.e * dv.var())
    return float(s_max - s)


# ------------------------------------------------------------------------------------------------ plots
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _cv_label(cv):
    unit = {"distance": "nm", "rmsd": "nm"}.get(str(cv.get("type")), "rad")
    return f"{cv.get('name', 'CV')} ({unit})"


def plot_profile(path, x, curves, xlabel, title, ylabel="Free energy (kJ/mol)"):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, y, err in curves:
        y = np.asarray(y, float)
        ax.plot(x, y, lw=2, label=label)
        if err is not None:
            e = np.asarray(err, float)
            ax.fill_between(x, y - e, y + e, alpha=0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(curves) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_timeseries(path, t_ps, series, xlabel="time (ps)", title=""):
    plt = _plt()
    fig, axes = plt.subplots(len(series), 1, figsize=(6.4, 1.9 * len(series) + 0.6), sharex=True, squeeze=False)
    for ax, (label, y) in zip(axes[:, 0], series):
        ax.plot(t_ps, y, lw=0.8)
        ax.set_ylabel(label)
    axes[-1, 0].set_xlabel(xlabel)
    if title:
        axes[0, 0].set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_histograms(path, samples, centers, xlabel, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(samples):
        if len(s):
            ax.hist(s, bins=30, alpha=0.6, color=cmap(i / max(1, len(samples) - 1)), histtype="stepfilled")
    for c in centers:
        ax.axvline(c, color="k", lw=0.4, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("samples")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_fes_2d(path, x, y, fes, xlabel, ylabel, title, traj=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    f = np.asarray(fes, float)
    cs = ax.contourf(x, y, f.T, levels=24, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="Free energy (kJ/mol)")
    if traj is not None and len(traj):
        tr = np.asarray(traj, float)
        ax.plot(tr[:, 0], tr[:, 1], ".", ms=1.2, color="w", alpha=0.35)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_boost_distribution(path, dv, title):
    plt = _plt()
    dv = np.asarray(dv, float)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist(dv, bins=50, density=True, alpha=0.7, label="ΔV")
    if dv.std() > 0:
        xs = np.linspace(dv.min(), dv.max(), 200)
        ax.plot(xs, np.exp(-0.5 * ((xs - dv.mean()) / dv.std()) ** 2) / (dv.std() * math.sqrt(2 * math.pi)),
                lw=2, label="Gaussian")
    ax.set_xlabel("boost potential ΔV (kJ/mol)")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_csv(path, header, rows):
    with open(path, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else
                              (f"{v:.6g}" if isinstance(v, float) else str(v)) for v in r) + "\n")
    return path


# ------------------------------------------------------------------------------------------- the methods
class _Job:
    """Shared set-up: system, CV atom groups, positions, temperature."""

    def __init__(self, spec):
        from openmm import unit

        self.spec = spec
        self.out = spec["out_dir"]
        os.makedirs(self.out, exist_ok=True)
        sysopt = dict(spec.get("system") or {})
        self.T = float(spec.get("temperature_k", 300.0))
        self.dt_fs = float(spec.get("timestep_fs", 2.0))
        self.friction = float(spec.get("friction_ps", 1.0))
        self.seed = int(spec.get("seed", 2026))
        emit("preparing the system")
        self.topology, self.system, self.positions, self.info = build_system(
            spec["pdb"], solvent=sysopt.get("solvent", "implicit"), family=sysopt.get("family", "amber14"),
            ph=float(sysopt.get("ph", 7.0)), keep_waters=bool(sysopt.get("keep_waters", False)),
            remove_resn=sysopt.get("remove_resn") or (), padding_nm=float(sysopt.get("padding_nm", 1.0)),
            ionic_strength_molar=float(sysopt.get("ionic_strength_molar", 0.15)), temperature_k=self.T)
        self.periodic_box = self.info["solvent"] == "explicit"
        self.cvs = []
        pos_nm = np.asarray(self.positions.value_in_unit(unit.nanometer), float)
        self.reference_nm = pos_nm
        for cv in spec.get("cvs") or []:
            groups = map_atom_keys(self.topology, cv["atoms"])
            self.cvs.append(dict(cv, indices=groups, n_atoms_per_group=[len(g) for g in groups],
                                 periodic=bool(cv.get("periodic", str(cv.get("type")) == "dihedral"))))
        self.masses = np.array([self.system.getParticleMass(i).value_in_unit(unit.dalton)
                                for i in range(self.system.getNumParticles())])
        self.beta = 1.0 / (KB_KJ_MOL_K * self.T)

    def cv_force(self, cv):
        return cv_force(cv, cv["indices"], self.reference_nm, periodic=self.periodic_box)

    def base_result(self, method, platform, steps, seconds):
        return {"ok": True, "method": method, **platform, "ns_per_day": _ns_per_day(steps, self.dt_fs, seconds),
                "simulated_ns": round(steps * self.dt_fs * 1e-6, 4), "n_atoms": self.info["n_atoms"],
                "solvent": self.info["solvent"], "force_field": self.info["family"], "temperature_k": self.T,
                "cvs": [{"name": c.get("name"), "type": c.get("type"), "atoms_per_group": c["n_atoms_per_group"],
                         "periodic": c["periodic"]} for c in self.cvs],
                "out_dir": self.out,
                "note": "Short run: a qualitative profile with its sampling diagnostics, not a converged free energy."}


def run_umbrella(spec):
    """Umbrella sampling along one CV: windows share one context (the restraint centre and spring are context
    parameters), swept outward from the window nearest the starting value; PMF by MBAR (pymbar FES) with
    analytical uncertainties (WHAM if pymbar fails); histogram overlap between neighbours is reported."""
    import openmm as mm

    job = _Job(spec)
    if len(job.cvs) != 1:
        raise RuntimeError("umbrella sampling needs exactly one CV")
    cv = job.cvs[0]
    u = dict(spec.get("umbrella") or {})
    centers = list(u.get("centers") or np.linspace(float(u["start"]), float(u["end"]), int(u["windows"])))
    k = float(u.get("k", 1000.0))
    eq_steps, prod_steps = int(u.get("equil_steps", 2000)), int(u.get("prod_steps", 5000))
    stride = max(1, int(u.get("stride", 50)))
    periodic = cv["periodic"]
    restraint = mm.CustomCVForce("0.5*k_umb*d^2; d=" + ("min(abs(s - c0), 2*pi - abs(s - c0)); pi=3.141592653589793"
                                                       if periodic else "s - c0"))
    restraint.addGlobalParameter("k_umb", k)
    restraint.addGlobalParameter("c0", float(centers[0]))
    restraint.addCollectiveVariable("s", job.cv_force(cv))
    restraint.setForceGroup(BIAS_GROUP)
    job.system.addForce(restraint)
    sim, plat = make_simulation(job.topology, job.system, langevin_integrator(job.T, job.friction, job.dt_fs, job.seed),
                                job.positions, job.T, job.seed)
    ctx = sim.context
    x0 = float(restraint.getCollectiveVariableValues(ctx)[0])
    start = int(np.argmin(np.abs(_periodic_delta(np.asarray(centers), x0, periodic))))
    order = list(range(start, len(centers))) + list(range(start - 1, -1, -1))
    samples = [None] * len(centers)
    saved = None
    t0 = time.time()
    steps = 0
    for n, w in enumerate(order):
        if w == start - 1 and saved is not None:
            ctx.setState(saved)                         # the downward sweep starts from the first window
        ctx.setParameter("c0", float(centers[w]))
        ctx.setParameter("k_umb", k)
        emit(f"window {n + 1}/{len(centers)} at {centers[w]:.3f}")
        sim.step(eq_steps)
        if w == start:
            saved = ctx.getState(getPositions=True, getVelocities=True, getParameters=True)
        vals = []
        for _ in range(prod_steps // stride):
            sim.step(stride)
            vals.append(float(restraint.getCollectiveVariableValues(ctx)[0]))
        samples[w] = np.asarray(vals)
        steps += eq_steps + prod_steps
    seconds = time.time() - t0
    emit("analysis")
    res = job.base_result("umbrella", plat, steps, seconds)
    res.update(analyze_umbrella(samples, centers, k, job.T, periodic, cv, job.out))
    res.update(windows=len(centers), force_constant=k, samples_per_window=prod_steps // stride)
    return res


def analyze_umbrella(samples, centers, k, temperature_k, periodic, cv, out_dir, nbins=None):
    beta = 1.0 / (KB_KJ_MOL_K * float(temperature_k))
    centers = np.asarray(centers, float)
    x = np.concatenate(samples)
    n_k = np.array([len(s) for s in samples])
    u_kn = np.array([0.5 * beta * k * _periodic_delta(x, c, periodic) ** 2 for c in centers])
    lo, hi = (min(x.min(), centers.min()), max(x.max(), centers.max()))
    if periodic:
        lo, hi = -np.pi, np.pi
    nbins = int(nbins or max(20, min(60, len(centers) * 3)))
    edges = np.linspace(lo, hi, nbins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    pmf, err, method = None, None, "MBAR (pymbar FES)"
    try:
        from pymbar import FES

        fes = FES(u_kn, n_k, verbose=False)
        # u_n: reduced potential of each sample in the target (unbiased) state; the biases are all in u_kn
        fes.generate_fes(np.zeros(x.size), x, fes_type="histogram", histogram_parameters={"bin_edges": edges})
        vis = mid[(np.histogram(x, edges)[0] > 0)]
        r = fes.get_fes(vis, reference_point="from-lowest", uncertainty_method="analytical")
        f = np.full(nbins, np.nan)
        e = np.full(nbins, np.nan)
        f[np.histogram(x, edges)[0] > 0] = np.asarray(r["f_i"]) / beta
        e[np.histogram(x, edges)[0] > 0] = np.asarray(r["df_i"]) / beta
        pmf, err = f, e
        try:
            overlap = fes.mbar.compute_overlap()
            min_overlap = float(np.min(np.diag(np.asarray(overlap["matrix"]), 1)))
        except Exception:
            min_overlap = float("nan")
    except Exception as exc:
        pmf, err, method = _wham_1d(samples, centers, k, temperature_k, periodic, edges), None, f"WHAM ({exc})"
        min_overlap = float("nan")
    hist = [np.histogram(s, edges)[0].astype(float) for s in samples]
    neigh = []
    for a, b in zip(hist[:-1], hist[1:]):
        if a.sum() and b.sum():
            neigh.append(float(np.minimum(a / a.sum(), b / b.sum()).sum()))
    xl = _cv_label(cv)
    files = {
        "pmf_png": plot_profile(os.path.join(out_dir, "umbrella_pmf.png"), mid, [("PMF", pmf, err)], xl,
                                "Umbrella sampling PMF (MBAR)"),
        "histograms_png": plot_histograms(os.path.join(out_dir, "umbrella_histograms.png"), samples, centers, xl,
                                          "Window histograms"),
        "pmf_csv": _write_csv(os.path.join(out_dir, "umbrella_pmf.csv"), ["cv", "pmf_kj_mol", "sd_kj_mol"],
                              [(float(a), float(b), (float(c) if err is not None else None))
                               for a, b, c in zip(mid, pmf, err if err is not None else [None] * len(mid))]),
        "samples_csv": _write_csv(os.path.join(out_dir, "umbrella_samples.csv"), ["window", "center", "cv"],
                                  [(i, float(centers[i]), float(v)) for i, s in enumerate(samples) for v in s]),
    }
    finite = np.isfinite(pmf)
    warn = []
    if neigh and min(neigh) < 0.05:
        warn.append(f"poor overlap between neighbouring windows (min {min(neigh):.2f}): add windows or lower k")
    return {"pmf_method": method, "pmf": {"cv": mid[finite].round(5).tolist(), "kj_mol": pmf[finite].round(3).tolist()},
            "pmf_range_kj_mol": round(float(np.nanmax(pmf) - np.nanmin(pmf)), 2) if finite.any() else None,
            "pmf_minimum_at": round(float(mid[np.nanargmin(pmf)]), 4) if finite.any() else None,
            "min_neighbour_overlap": round(min(neigh), 3) if neigh else None,
            "min_mbar_overlap": None if not np.isfinite(min_overlap) else round(min_overlap, 4),
            "files": files, "warnings": warn}


def _wham_1d(samples, centers, k, temperature_k, periodic, edges, tol=1e-7, it=20000):
    beta = 1.0 / (KB_KJ_MOL_K * float(temperature_k))
    mid = 0.5 * (edges[:-1] + edges[1:])
    h = np.array([np.histogram(s, edges)[0] for s in samples], float)
    n = h.sum(1)
    bias = np.array([0.5 * k * _periodic_delta(mid, c, periodic) ** 2 for c in centers]) * beta
    f = np.zeros(len(centers))
    for _ in range(it):
        p = h.sum(0) / np.maximum((n[:, None] * np.exp(f[:, None] - bias)).sum(0), 1e-300)
        f_new = -np.log(np.maximum((p[None, :] * np.exp(-bias)).sum(1), 1e-300))
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    with np.errstate(divide="ignore"):
        pmf = -np.log(p) / beta
    pmf[~np.isfinite(pmf)] = np.nan
    return pmf - np.nanmin(pmf)


def run_metadynamics(spec):
    """Well-tempered metadynamics on 1-3 CVs with OpenMM's Metadynamics (bias grid on the GPU). The free energy
    is Metadynamics.getFreeEnergy() (already rescaled by bias_factor/(bias_factor-1)); convergence is shown by
    the surfaces at 50/75/100 % of the run."""
    import openmm.app as app
    from openmm import unit

    job = _Job(spec)
    m = dict(spec.get("metadynamics") or {})
    variables = []
    for cv in job.cvs:
        if cv["periodic"]:
            lo, hi = -math.pi, math.pi                  # periodic CVs span the full circle
        else:
            rng = cv.get("range") or [cv.get("min"), cv.get("max")]
            if rng[0] is None or rng[1] is None:
                raise RuntimeError(f"CV {cv.get('name')}: metadynamics needs its range (min, max)")
            lo, hi = float(rng[0]), float(rng[1])
        width = float(cv.get("bias_width", (hi - lo) / 30.0))
        grid = int(cv.get("grid_width") or 0) or None
        variables.append(app.BiasVariable(job.cv_force(cv), lo, hi, width, periodic=cv["periodic"],
                                          gridWidth=grid))
    if not 1 <= len(variables) <= 3:
        raise RuntimeError("metadynamics needs 1-3 CVs")
    freq = int(m.get("frequency", 500))
    total = int(m.get("total_steps", 50000))
    stride = max(1, int(m.get("stride", freq)))
    meta = app.Metadynamics(job.system, variables, job.T * unit.kelvin, float(m.get("bias_factor", 10.0)),
                            float(m.get("height_kj_mol", 1.2)) * unit.kilojoules_per_mole, freq)
    sim, plat = make_simulation(job.topology, job.system, langevin_integrator(job.T, job.friction, job.dt_fs, job.seed),
                                job.positions, job.T, job.seed)
    traj, times, snaps = [], [], {}
    t0 = time.time()
    done = 0
    marks = {int(total * 0.5): "50%", int(total * 0.75): "75%"}
    while done < total:
        n = min(stride, total - done)
        meta.step(sim, n)
        done += n
        traj.append([float(v) for v in meta.getCollectiveVariables(sim)])
        times.append(done * job.dt_fs * 1e-3)
        for mark, label in list(marks.items()):
            if done >= mark:
                snaps[label] = np.asarray(meta.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole), float).copy()
                marks.pop(mark)
        if len(traj) % 20 == 0:
            emit(f"{100 * done // total}% ({len(traj)} CV samples)")
    seconds = time.time() - t0
    fes = np.asarray(meta.getFreeEnergy().value_in_unit(unit.kilojoules_per_mole), float)
    snaps["100%"] = fes
    emit("analysis")
    res = job.base_result("metadynamics", plat, total, seconds)
    res.update(analyze_metadynamics(variables, job.cvs, fes, snaps, np.asarray(traj), np.asarray(times), job.out))
    res.update(hills=int(total // freq), bias_factor=float(m.get("bias_factor", 10.0)),
               height_kj_mol=float(m.get("height_kj_mol", 1.2)))
    return res


def _grid(var):
    n = var.gridWidth
    return np.linspace(var.minValue, var.maxValue, n)


def analyze_metadynamics(variables, cvs, fes, snaps, traj, times, out_dir):
    files = {}
    fes = fes - np.nanmin(fes)
    out = {}
    if len(variables) == 1:
        x = _grid(variables[0])
        curves = [(f"{lab} of run", np.asarray(s, float).reshape(-1) - np.nanmin(s), None) for lab, s in snaps.items()]
        files["fes_png"] = plot_profile(os.path.join(out_dir, "metadynamics_fes.png"), x, curves, _cv_label(cvs[0]),
                                        "Well-tempered metadynamics free energy")
        files["fes_csv"] = _write_csv(os.path.join(out_dir, "metadynamics_fes.csv"), ["cv", "fes_kj_mol"],
                                      [(float(a), float(b)) for a, b in zip(x, fes.reshape(-1))])
        out["fes"] = {"cv": x.round(5).tolist(), "kj_mol": fes.reshape(-1).round(3).tolist()}
        out["fes_minimum_at"] = round(float(x[int(np.argmin(fes))]), 4)
        visited = (x >= np.nanmin(traj[:, 0])) & (x <= np.nanmax(traj[:, 0]))
        s75 = np.asarray(snaps.get("75%", fes), float).reshape(-1)
        s75 = s75 - np.nanmin(s75)
        out["fes_change_last_quarter_kj_mol"] = round(float(np.sqrt(np.nanmean((fes.reshape(-1) - s75)[visited] ** 2))), 3)
    else:
        x, y = _grid(variables[0]), _grid(variables[1])
        f2 = fes if fes.ndim == 2 else fes.reshape(len(y), len(x))
        f2 = np.asarray(f2).T if f2.shape == (len(y), len(x)) else np.asarray(f2)
        files["fes_png"] = plot_fes_2d(os.path.join(out_dir, "metadynamics_fes2d.png"), x, y, f2,
                                       _cv_label(cvs[0]), _cv_label(cvs[1]), "Well-tempered metadynamics free energy",
                                       traj=traj[:, :2])
        files["fes_csv"] = _write_csv(os.path.join(out_dir, "metadynamics_fes2d.csv"), ["cv1", "cv2", "fes_kj_mol"],
                                      [(float(a), float(b), float(f2[i, j])) for i, a in enumerate(x)
                                       for j, b in enumerate(y)])
        i, j = np.unravel_index(int(np.nanargmin(f2)), f2.shape)
        out["fes_minimum_at"] = [round(float(x[i]), 4), round(float(y[j]), 4)]
    files["cv_png"] = plot_timeseries(os.path.join(out_dir, "metadynamics_cv.png"), times,
                                      [(_cv_label(c), traj[:, i]) for i, c in enumerate(cvs)],
                                      title="Collective variables during metadynamics")
    files["cv_csv"] = _write_csv(os.path.join(out_dir, "metadynamics_cv.csv"),
                                 ["time_ps"] + [c.get("name", f"cv{i + 1}") for i, c in enumerate(cvs)],
                                 [(float(t), *[float(v) for v in row]) for t, row in zip(times, traj)])
    out["cv_sampled_range"] = [[round(float(np.min(traj[:, i])), 4), round(float(np.max(traj[:, i])), 4)]
                               for i in range(traj.shape[1])]
    out["files"] = files
    return out


def run_boosted(spec, kind):
    """aMD or GaMD: a short conventional MD gives the potential statistics, then the boosted run; the PMF along
    each CV is reweighted (cumulant 2nd order and exponential)."""
    from openmm import unit

    job = _Job(spec)
    b = dict(spec.get(kind) or spec.get("boost") or {})
    cmd_steps = int(b.get("cmd_steps", 10000))
    equil_steps = int(b.get("equil_steps", 2000))
    prod_steps = int(b.get("prod_steps", 50000))
    stride = max(1, int(b.get("stride", 100)))
    mode = str(b.get("boost", "dual")).lower()
    sim, plat = make_simulation(job.topology, job.system, langevin_integrator(job.T, job.friction, job.dt_fs, job.seed),
                                job.positions, job.T, job.seed)
    emit("conventional MD for the potential statistics")
    vt, vd = [], []
    t0 = time.time()
    for _ in range(max(1, cmd_steps // stride)):
        sim.step(stride)
        vt.append(sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
        vd.append(sim.context.getState(getEnergy=True, groups={DIHEDRAL_GROUP})
                  .getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole))
    vt, vd = np.asarray(vt), np.asarray(vd)
    if kind == "gamd":
        sigma0 = float(b.get("sigma0_kcal", 6.0))
        pt, pd = gamd_parameters(vt, sigma0), gamd_parameters(vd, sigma0)
        total, dih = (pt["E"], pt["k"]), (pd["E"], pd["k"])
        params = {"total": pt, "dihedral": pd}
    else:
        p = amd_parameters(vt.mean(), vd.mean(), job.info["n_atoms"], job.info["n_residues_solute"],
                           float(b.get("level", 1.0)))
        total, dih = p["total"], p["dihedral"]
        params = {"total": {"E": total[0], "alpha": total[1]}, "dihedral": {"E": dih[0], "alpha": dih[1]}}
    if mode == "total":
        dih = None
    elif mode == "dihedral":
        total = None
    state = sim.context.getState(getPositions=True, getVelocities=True)
    integ = boosted_langevin_integrator(kind, job.T, job.friction, job.dt_fs, total=total, dihedral=dih, seed=job.seed)
    simulation_fn, platform_fn = _platform_helpers()
    import openmm.app as app

    bsim = simulation_fn(app, __import__("openmm"), job.topology, job.system, integ)
    plat = platform_fn(bsim)
    bsim.context.setState(state)
    emit("boosted equilibration")
    bsim.step(equil_steps)
    rows = []
    box = None
    for i in range(max(1, prod_steps // stride)):
        bsim.step(stride)
        st = bsim.context.getState(getEnergy=True, getPositions=True)
        v_tot = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        v_dih = bsim.context.getState(getEnergy=True, groups={DIHEDRAL_GROUP}).getPotentialEnergy() \
            .value_in_unit(unit.kilojoules_per_mole)
        dv = (float(boost_energy(kind, v_tot, *total)) if total else 0.0) + \
             (float(boost_energy(kind, v_dih, *dih)) if dih else 0.0)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if job.periodic_box:
            box = np.diag(st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer))
        cvs = [cv_values_numpy(c["type"], c["indices"], pos, job.masses, box, job.reference_nm) for c in job.cvs]
        rows.append([(i + 1) * stride * job.dt_fs * 1e-3, v_tot, v_dih, dv] + cvs)
        if (i + 1) % 50 == 0:
            emit(f"{100 * (i + 1) * stride // prod_steps}% of the boosted run")
    seconds = time.time() - t0
    steps = cmd_steps + equil_steps + prod_steps
    emit("analysis")
    res = job.base_result(kind, plat, steps, seconds)
    res["boost"] = mode
    res["parameters_kj_mol"] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                for k, v in params.items()}
    res.update(analyze_boosted(kind, np.asarray(rows, float), job.cvs, job.T, job.out,
                               bins=int(b.get("bins", 40)), min_count=int(b.get("min_count", 10))))
    return res


def analyze_boosted(kind, rows, cvs, temperature_k, out_dir, bins=40, min_count=10):
    t, dv = rows[:, 0], rows[:, 3]
    files = {"timeseries_csv": _write_csv(os.path.join(out_dir, f"{kind}_frames.csv"),
                                          ["time_ps", "V_total_kj_mol", "V_dihedral_kj_mol", "dV_kj_mol"]
                                          + [c.get("name", f"cv{i + 1}") for i, c in enumerate(cvs)], rows.tolist())}
    gamma = boost_anharmonicity(dv)
    files["boost_png"] = plot_boost_distribution(os.path.join(out_dir, f"{kind}_boost.png"), dv,
                                                 f"{kind.upper()} boost potential (anharmonicity {gamma:.3f})")
    out = {"frames": int(len(rows)), "boost_mean_kj_mol": round(float(dv.mean()), 3),
           "boost_sd_kj_mol": round(float(dv.std()), 3), "boost_max_kj_mol": round(float(dv.max()), 3),
           "boost_anharmonicity": round(gamma, 4), "pmfs": []}
    warn = []
    if dv.std() * (1.0 / (KB_KJ_MOL_K * temperature_k)) > 3.0:
        warn.append("ΔV fluctuations exceed ~3 kT: cumulant reweighting is noisy; lower the boost (sigma0/level)")
    if gamma > 0.01:
        warn.append(f"ΔV distribution is anharmonic (γ={gamma:.3f} > 0.01): treat the reweighted PMF as qualitative")
    series = [("ΔV (kJ/mol)", dv)]
    for i, cv in enumerate(cvs):
        x = rows[:, 4 + i]
        series.append((_cv_label(cv), x))
        if cv["periodic"]:
            edges = np.linspace(-np.pi, np.pi, bins + 1)
        else:
            lo, hi = float(np.min(x)), float(np.max(x))
            pad = 0.02 * (hi - lo + 1e-9)
            edges = np.linspace(lo - pad, hi + pad, bins + 1)
        rw = reweight_pmf(x, dv, temperature_k, edges, periodic=cv["periodic"], min_count=min_count)
        name = cv.get("name", f"cv{i + 1}")
        png = plot_profile(os.path.join(out_dir, f"{kind}_pmf_{i + 1}.png"), rw["centers"],
                           [("cumulant (2nd order)", rw["pmf_cumulant2"], None),
                            ("exponential", rw["pmf_exponential"], None),
                            ("unreweighted (boosted)", rw["pmf_unweighted"], None)],
                           _cv_label(cv), f"{kind.upper()} reweighted PMF along {name}")
        csv = _write_csv(os.path.join(out_dir, f"{kind}_pmf_{i + 1}.csv"),
                         ["cv", "frames", "pmf_cumulant2_kj_mol", "pmf_exponential_kj_mol", "pmf_unweighted_kj_mol"],
                         [(float(a), int(c), float(p), float(q), float(r)) for a, c, p, q, r in
                          zip(rw["centers"], rw["counts"], rw["pmf_cumulant2"], rw["pmf_exponential"],
                              rw["pmf_unweighted"])])
        fin = np.isfinite(rw["pmf_cumulant2"])
        out["pmfs"].append({"cv": name, "png": png, "csv": csv,
                            "pmf_cumulant2": {"cv": rw["centers"][fin].round(5).tolist(),
                                              "kj_mol": rw["pmf_cumulant2"][fin].round(3).tolist()},
                            "minimum_at": round(float(rw["centers"][np.nanargmin(rw["pmf_cumulant2"])]), 4)
                            if fin.any() else None,
                            "sampled_range": [round(float(x.min()), 4), round(float(x.max()), 4)]})
    files["timeseries_png"] = plot_timeseries(os.path.join(out_dir, f"{kind}_timeseries.png"), t, series,
                                              title=f"{kind.upper()} production")
    out["files"] = files
    out["warnings"] = warn
    return out


# ------------------------------------------------------------------------------------------- OpenMM-ML
ANI_ELEMENTS = {"ani2x": {"H", "C", "N", "O", "F", "S", "Cl"}, "ani1ccx": {"H", "C", "N", "O"},
                "ani1x": {"H", "C", "N", "O"}}


def ml_python():
    """Python of the separate 'leonardo-ml' environment (openmm-torch/openmm-ml/torchani conflict with the main
    one): LEONARDO_ML_PYTHON, else a sibling conda environment named leonardo-ml."""
    env = os.environ.get("LEONARDO_ML_PYTHON", "")
    if env and os.path.isfile(env):
        return env
    exe = "python.exe" if os.name == "nt" else os.path.join("bin", "python")
    for base in (os.path.dirname(sys.prefix), os.path.join(sys.prefix, "envs")):
        cand = os.path.join(base, "leonardo-ml", exe)
        if os.path.isfile(cand):
            return cand
    return ""


def run_openmm_ml(spec):
    """MD with an ANI potential (openmm-ml) for the ligand inside the classical system (mixed ML/MM) or for a whole
    small molecule (ml_only). The classical system is prepared here; the dynamics run in the leonardo-ml
    environment (openmm_ml_runtime)."""
    import subprocess

    import openmm as mm
    import openmm.app as app

    ml = dict(spec.get("openmm_ml") or {})
    mode = str(ml.get("mode", "mixed"))
    model = str(ml.get("model", "ani2x"))
    py = ml_python()
    if not py:
        raise RuntimeError("OpenMM-ML needs Leonardo's 'leonardo-ml' environment (openmm-ml + torchani), which is "
                           "not installed: rerun the installer with the ML option, or set LEONARDO_ML_PYTHON")
    job = _Job(spec)
    atoms = list(job.topology.atoms())
    if mode == "mixed":
        idx = set(map_atom_keys(job.topology, [ml["atoms"]])[0])
        residues = {a.residue.index for a in atoms if a.index in idx}
        ml_atoms = [a.index for a in atoms if a.residue.index in residues]   # whole residues, with their hydrogens
    else:
        if len(atoms) > 300:
            raise RuntimeError(f"ml_only treats every atom with the network: {len(atoms)} atoms is a complex, use "
                               "mixed ML/MM with the ligand as the ML region")
        ml_atoms = [a.index for a in atoms]
    charge = 0.0
    for force in job.system.getForces():
        if isinstance(force, mm.NonbondedForce):
            from openmm import unit as _u

            charge = sum(force.getParticleParameters(i)[0].value_in_unit(_u.elementary_charge) for i in ml_atoms)
    if abs(charge) > 0.5 and not bool(ml.get("allow_charged", False)):
        # ANI models are trained on neutral molecules: a charged region (e.g. benzamidinium) falls apart
        raise RuntimeError(f"the ML region has a net charge of {charge:+.0f} e; {model} was trained on neutral "
                           "molecules only and gives wrong forces for ions (a benzamidinium test came apart). Use a "
                           "neutral ligand or the classical force field")
    allowed = ANI_ELEMENTS.get(model, ANI_ELEMENTS["ani2x"])
    bad = sorted({a.element.symbol for a in atoms if a.index in set(ml_atoms) and a.element.symbol not in allowed})
    if bad:
        raise RuntimeError(f"{model} has no parameters for {', '.join(bad)} (it covers {', '.join(sorted(allowed))})")
    xml = os.path.join(job.out, "mm_system.xml")
    with open(xml, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(job.system))
    pdb = os.path.join(job.out, "prepared.pdb")
    with open(pdb, "w") as fh:
        app.PDBFile.writeFile(job.topology, job.positions, fh, keepIds=True)
    ml_job = {"system_xml": xml, "pdb": pdb, "ml_atoms": ml_atoms, "mode": mode, "model": model,
              "steps": int(ml.get("steps", 5000)), "stride": int(ml.get("stride", 50)), "temperature_k": job.T,
              "timestep_fs": float(spec.get("ml_timestep_fs", 1.0)), "minimize_steps": int(ml.get("minimize_steps", 200)),
              "seed": job.seed, "result_json": os.path.join(job.out, "ml_result.json")}
    job_path = os.path.join(job.out, "ml_job.json")
    with open(job_path, "w") as fh:
        json.dump(ml_job, fh)
    env = dict(os.environ)
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    emit(f"running {model} ({mode}) in the leonardo-ml environment")
    proc = subprocess.Popen([py, "-m", "leonardo.openmm_ml_runtime", job_path], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env, cwd=job.out)
    tail = []
    for line in proc.stdout:
        if line.startswith("PROGRESS"):
            print(line.rstrip(), flush=True)
        else:
            tail = (tail + [line.rstrip()])[-30:]
    proc.wait()
    try:
        with open(ml_job["result_json"]) as fh:
            res = json.load(fh)
    except Exception:
        res = {"ok": False, "error": "\n".join(tail)[-1500:] or "the ML run wrote no result"}
    if not res.get("ok"):
        raise RuntimeError(f"OpenMM-ML run failed: {res.get('error')}")
    rows = np.asarray(res.pop("rows"), float)
    files = {"timeseries_csv": _write_csv(os.path.join(job.out, "openmm_ml_timeseries.csv"),
                                          ["time_ps", "potential_kj_mol", "kinetic_kj_mol", "temperature_k",
                                           "ml_region_rmsd_A"], rows.tolist()),
             "timeseries_png": plot_timeseries(os.path.join(job.out, "openmm_ml_timeseries.png"), rows[:, 0],
                                               [("Potential (kJ/mol)", rows[:, 1]), ("Temperature (K)", rows[:, 3]),
                                                ("ML-region RMSD (Å)", rows[:, 4])],
                                               title=f"{model} {mode} MD")}
    out = {"ok": True, "method": "openmm_ml", **res, "precision": "", "solvent": job.info["solvent"],
           "force_field": job.info["family"], "simulated_ns": round(res["steps"] * res["timestep_fs"] * 1e-6, 5),
           "files": files, "out_dir": job.out,
           "note": "Short ML/MM dynamics: a stability check of the ML potential in this system, not a free energy."}
    warn = []
    if abs(out["temperature_mean_k"] - job.T) > 20:
        warn.append(f"mean temperature {out['temperature_mean_k']} K is far from {job.T} K")
    out["warnings"] = warn
    return out


METHODS = {"umbrella": run_umbrella, "metadynamics": run_metadynamics,
           "amd": lambda s: run_boosted(s, "amd"), "gamd": lambda s: run_boosted(s, "gamd"),
           "openmm_ml": run_openmm_ml}


def run_job(spec):
    method = str(spec.get("method", "")).lower()
    if method not in METHODS:
        raise RuntimeError(f"unknown method {method!r}; one of {sorted(METHODS)}")
    return METHODS[method](spec)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    spec = json.load(open(argv[0]))
    out_json = spec.get("result_json") or os.path.join(spec.get("out_dir", "."), "result.json")
    try:
        res = run_job(spec)
    except Exception as exc:
        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}
    with open(out_json, "w") as fh:
        json.dump(_jsonable(res), fh, indent=1)
    emit("done" if res.get("ok") else f"failed: {res.get('error')}")
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
