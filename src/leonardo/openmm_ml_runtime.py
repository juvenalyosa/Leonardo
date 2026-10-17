"""
OpenMM-ML runs: molecular dynamics with a machine-learned potential (ANI-2x / ANI-1ccx through openmm-ml) for a
whole small molecule, or for a region (a ligand) embedded in the classical force field of the rest (mixed ML/MM).

This file runs in the separate ``leonardo-ml`` conda environment (openmm, openmm-torch, openmm-ml, torchani:
PyTorch conflicts with Leonardo's main environment), so it imports nothing but OpenMM, openmm-ml and NumPy.
Leonardo's main environment prepares the classical system (enhanced_sampling_runtime.run_openmm_ml) and calls

    <leonardo-ml python> -m leonardo.openmm_ml_runtime job.json

job.json: {"system_xml", "pdb", "ml_atoms": [indices], "mode": "mixed|ml_only", "model": "ani2x", "steps",
"stride", "temperature_k", "timestep_fs", "minimize_steps", "seed", "result_json"}.

The network runs through openmm-torch: on CUDA GPUs when OpenMM picks CUDA, on the CPU otherwise (openmm-torch
has no OpenCL/Metal kernels). The pretrained weights ship with TorchANI and are downloaded to
~/.local/share/Torchani on first use (the installer fetches them ahead of time).
"""
import json
import os
import sys
import time
import traceback

import numpy as np

KB_KJ_MOL_K = 0.0083144626


def emit(msg):
    print(f"PROGRESS {msg}", flush=True)


def _compute_ani(state, model, species, pbc, indices):
    """openmm-ml's ANI force callback with the units right. openmm-ml 1.7 differentiates the energy with respect
    to positions in angstrom and hands that gradient (kJ/mol/A) to OpenMM, which reads kJ/mol/nm: its ANI forces
    are 10x too weak (alanine dipeptide fell apart within 10 ps at 300 K, and NVE did not conserve energy)."""
    import numpy as np
    import torch
    import torchani
    from openmm import unit

    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    n_atoms = positions.shape[0]
    if indices is not None:
        positions = positions[indices]
    positions = torch.tensor(positions, dtype=torch.float32, device=species.device)
    if pbc is None:
        box = None
    else:
        box = torch.tensor(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom),
                           dtype=torch.float32, device=species.device)
        for k in (2, 1, 0):
            positions = positions - torch.outer(torch.floor(positions[:, k] / box[k, k]), box[k])
    positions = positions.unsqueeze(0)
    positions.requires_grad_(True)
    _, energy = model((species, positions), cell=box, pbc=pbc)
    energy = energy * torchani.units.hartree2kjoulemol(1)
    energy.backward()
    forces = (-positions.grad[0]).detach().cpu().numpy().astype(np.float64) * 10.0     # kJ/mol/A -> kJ/mol/nm
    if indices is not None:
        full = np.zeros((n_atoms, 3))
        full[indices] = forces
        forces = full
    return float(energy.detach().cpu().numpy().reshape(-1)[0]), forces


def _patch_openmmml_ani():
    """Use _compute_ani for every ANI force openmm-ml builds (the callback is looked up when the force is made)."""
    try:
        from openmmml.models import anipotential

        if hasattr(anipotential, "_computeANI"):
            anipotential._computeANI = _compute_ani
            return True
    except Exception:
        pass
    return False


def check_ani_forces(system, positions_nm, model_name="ani2x", atoms=None):
    """Forces of the ML system against TorchANI in double precision (same coordinates); returns the force ratio
    and direction cosine over the ML atoms. A run stops if they disagree (wrong units, wrong atoms)."""
    import numpy as np
    import openmm as mm
    import torch
    import torchani
    from openmm import unit

    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(positions_nm)
    f_all = ctx.getState(getForces=True, groups={_ML_GROUP}).getForces(asNumpy=True) \
        .value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    idx = list(atoms) if atoms is not None else list(range(system.getNumParticles()))
    model = (torchani.models.ANI2x if model_name == "ani2x" else torchani.models.ANI1ccx)(periodic_table_index=True)
    model = model.double()
    numbers = [_ATOMIC_NUMBERS[i] for i in idx]
    species = torch.tensor([numbers])
    xyz = torch.tensor(np.asarray(positions_nm.value_in_unit(unit.nanometer) if hasattr(positions_nm, "value_in_unit")
                                  else positions_nm, float)[idx] * 10.0, dtype=torch.float64).unsqueeze(0)
    xyz.requires_grad_(True)
    _, e = model((species, xyz))
    (e * torchani.units.hartree2kjoulemol(1)).backward()
    ref = (-xyz.grad[0]).numpy() * 10.0
    got = np.asarray(f_all, float)[idx]
    ratio = float(np.linalg.norm(got) / max(np.linalg.norm(ref), 1e-12))
    cosine = float((got * ref).sum() / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-12))
    return ratio, cosine


_ML_GROUP = 30
_ATOMIC_NUMBERS = []


def _context(mm, system, make_integrator, prefer=None):
    """CUDA first (TorchForce on the GPU), then the CPU; returns (context, integrator, platform, errors)."""
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    order = ([prefer] if prefer in names else []) + [p for p in ("CUDA", "CPU", "Reference") if p in names and p != prefer]
    errors = []
    for name in order:
        props = {"Precision": "mixed"} if name == "CUDA" else {}
        integ = make_integrator()
        try:
            return mm.Context(system, integ, mm.Platform.getPlatformByName(name), props), integ, name, errors
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("no OpenMM platform could run the ML system: " + "; ".join(errors))


def run(job):
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    from openmmml import MLPotential

    emit("loading the system")
    with open(job["system_xml"]) as fh:
        mm_system = mm.XmlSerializer.deserialize(fh.read())
    pdb = app.PDBFile(job["pdb"])
    ml_atoms = [int(i) for i in job.get("ml_atoms") or []]
    mode = str(job.get("mode", "mixed"))
    model = str(job.get("model", "ani2x"))
    emit(f"loading {model}")
    patched = _patch_openmmml_ani() if model.startswith("ani") else False
    potential = MLPotential(model)
    if mode == "ml_only":
        system = potential.createSystem(pdb.topology)       # the whole (small) molecule in vacuum
        n_ml = system.getNumParticles()
        ml_atoms = list(range(n_ml))
    else:
        if not ml_atoms:
            raise RuntimeError("mixed ML/MM needs the ML region (ml_atoms)")
        system = potential.createMixedSystem(pdb.topology, mm_system, ml_atoms)
        n_ml = len(ml_atoms)
    # the ML force alone in its own group, so it can be checked against TorchANI before any dynamics
    _ATOMIC_NUMBERS[:] = [a.element.atomic_number for a in pdb.topology.atoms()]
    for f in system.getForces():
        if type(f).__name__ in ("PythonForce", "TorchForce"):
            f.setForceGroup(_ML_GROUP)
    force_check = None
    if model.startswith("ani"):
        ratio, cosine = check_ani_forces(system, pdb.positions, model, ml_atoms or None)
        force_check = {"ratio_to_torchani": round(ratio, 4), "cosine": round(cosine, 5), "patched": patched}
        if abs(ratio - 1.0) > 0.02 or cosine < 0.999:
            raise RuntimeError(f"the {model} forces disagree with TorchANI (ratio {ratio:.3f}, cosine {cosine:.4f}): "
                               "refusing to run")
    T = float(job.get("temperature_k", 300.0))
    dt = float(job.get("timestep_fs", 1.0))
    def make_integrator():
        integ = mm.LangevinMiddleIntegrator(T * unit.kelvin, 1.0 / unit.picosecond, dt * unit.femtosecond)
        integ.setRandomNumberSeed(int(job.get("seed", 2026)))
        return integ

    ctx, integ, platform, errors = _context(mm, system, make_integrator, prefer=job.get("platform"))
    ctx.setPositions(pdb.positions)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    if int(job.get("minimize_steps", 200)):
        emit("minimizing")
        mm.LocalEnergyMinimizer.minimize(ctx, 10.0, int(job.get("minimize_steps", 200)))
    ctx.setVelocitiesToTemperature(T * unit.kelvin, int(job.get("seed", 2026)))
    steps, stride = int(job.get("steps", 2000)), max(1, int(job.get("stride", 50)))
    n_dof = 3 * system.getNumParticles() - system.getNumConstraints() - 3
    ref = None
    rows = []
    t0 = time.time()
    for i in range(max(1, steps // stride)):
        integ.step(stride)
        st = ctx.getState(getEnergy=True, getPositions=True)
        pe = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        ke = st.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        pos = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        region = pos[ml_atoms] if ml_atoms else pos
        if ref is None:
            ref = region - region.mean(0)
        a = region - region.mean(0)
        u, _s, vt = np.linalg.svd(a.T @ ref)
        r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
        rmsd = float(np.sqrt(((a @ r - ref) ** 2).sum(1).mean()))
        rows.append([(i + 1) * stride * dt * 1e-3, pe, ke, 2 * ke / (n_dof * KB_KJ_MOL_K), rmsd * 10.0])
        if not np.isfinite(pe):
            raise RuntimeError(f"the ML/MM energy became non-finite at {rows[-1][0]:.3f} ps")
        if (i + 1) % 20 == 0:
            emit(f"{100 * (i + 1) * stride // steps}% of the ML MD")
    seconds = time.time() - t0
    arr = np.asarray(rows)
    half = arr[len(arr) // 2:]
    return {"ok": True, "platform": platform, "platform_errors": errors, "model": model, "mode": mode,
            "n_atoms": int(system.getNumParticles()), "n_ml_atoms": int(n_ml),
            "energy_start_kj_mol": round(e0, 3), "steps": steps, "timestep_fs": dt,
            "ns_per_day": round(steps * dt * 1e-6 / max(seconds, 1e-9) * 86400.0, 3),
            "temperature_mean_k": round(float(half[:, 3].mean()), 2), "temperature_sd_k": round(float(half[:, 3].std()), 2),
            "ml_region_rmsd_max_A": round(float(arr[:, 4].max()), 3), "rows": arr.tolist(),
            "force_check": force_check,
            "openmm_version": mm.__version__}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    job = json.load(open(argv[0]))
    out = job.get("result_json") or os.path.join(os.path.dirname(os.path.abspath(argv[0])), "ml_result.json")
    try:
        res = run(job)
    except Exception as exc:
        res = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-3000:]}
    with open(out, "w") as fh:
        json.dump(res, fh)
    emit("done" if res.get("ok") else f"failed: {res.get('error')}")
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
