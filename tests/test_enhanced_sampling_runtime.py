import json
import os

import numpy as np
import pytest

from leonardo import enhanced_sampling_runtime as es

ALA2 = os.path.join(os.path.dirname(__file__), "data", "alanine_dipeptide.pdb")
PHI = [[["", "1", "", "C"]], [["", "2", "", "N"]], [["", "2", "", "CA"]], [["", "2", "", "C"]]]
PSI = [[["", "2", "", "N"]], [["", "2", "", "CA"]], [["", "2", "", "C"]], [["", "3", "", "N"]]]


def _toy_system():
    """Four particles: harmonic bonds (group 0) and one torsion (the dihedral boost group)."""
    import openmm as mm

    system = mm.System()
    for m in (12.0, 14.0, 12.0, 16.0):
        system.addParticle(m)
    bonds = mm.HarmonicBondForce()
    for i in range(3):
        bonds.addBond(i, i + 1, 0.15, 2.0e4)
    torsion = mm.PeriodicTorsionForce()
    torsion.addTorsion(0, 1, 2, 3, 3, 0.4, 6.0)
    torsion.addTorsion(0, 1, 2, 3, 1, 1.1, 9.0)
    torsion.setForceGroup(es.DIHEDRAL_GROUP)
    system.addForce(bonds)
    system.addForce(torsion)
    pos = np.array([[0.0, 0.02, 0.0], [0.16, 0.0, 0.01], [0.2, 0.14, -0.02], [0.33, 0.17, 0.09]])
    return system, pos


def _energies(system, pos):
    import openmm as mm
    from openmm import unit

    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos)
    v = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    vd = ctx.getState(getEnergy=True, groups={es.DIHEDRAL_GROUP}).getPotentialEnergy() \
        .value_in_unit(unit.kilojoules_per_mole)
    return v, vd


@pytest.mark.parametrize("kind", ["gamd", "amd"])
def test_boosted_integrator_force_is_the_gradient_of_the_boosted_potential(kind):
    # the sign, chain rule and dual-boost bookkeeping, checked before any MD: one frictionless step from rest
    # gives v = dt F*/m, and F* must be -grad of V + dV_t(V) + dV_d(V_d) by finite differences
    import openmm as mm
    from openmm import unit

    system, pos = _toy_system()
    v0, vd0 = _energies(system, pos)
    total = (v0 + 8.0, 0.05 if kind == "gamd" else 6.0)
    dih = (vd0 + 5.0, 0.08 if kind == "gamd" else 3.0)

    def boosted(p):
        v, vd = _energies(system, p)
        return v + es.boost_energy(kind, v, *total) + es.boost_energy(kind, vd, *dih)

    integ = es.boosted_langevin_integrator(kind, 300.0, 0.0, 1.0, total=total, dihedral=dih)   # 1 fs = 0.001 ps
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos)
    ctx.setVelocities(np.zeros_like(pos))
    integ.step(1)
    vel = ctx.getState(getVelocities=True).getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(4)])
    f_int = vel * masses[:, None] / 0.001
    h = 1e-6
    f_fd = np.zeros_like(pos)
    for i in range(4):
        for d in range(3):
            p1, p2 = pos.copy(), pos.copy()
            p1[i, d] += h
            p2[i, d] -= h
            f_fd[i, d] = -(boosted(p1) - boosted(p2)) / (2 * h)
    assert np.allclose(f_int, f_fd, rtol=2e-3, atol=0.5), (f_int, f_fd)
    # the boost must change the forces, otherwise the comparison proves nothing
    assert not np.allclose(f_int, _plain_forces(system, pos), atol=1.0)


def _plain_forces(system, pos):
    import openmm as mm
    from openmm import unit

    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos)
    return ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)


def test_gamd_parameters_use_sigma0_in_kj():
    v = np.array([-100.0, -95.0, -90.0, -92.0, -97.0])
    p = es.gamd_parameters(v, sigma0_kcal=6.0)
    assert p["sigma0_kj"] == pytest.approx(25.104)
    assert p["E"] == -90.0
    k0 = min(1.0, 25.104 / v.std() * 10.0 / (-90.0 - v.mean()))
    assert p["k0"] == pytest.approx(k0) and p["k"] == pytest.approx(k0 / 10.0)


def test_amd_parameters_follow_pierce_rules_in_kj():
    p = es.amd_parameters(-500.0, 40.0, n_atoms=22, n_residues=3)
    assert p["dihedral"][0] == pytest.approx(40.0 + 3.5 * 3 * 4.184)
    assert p["dihedral"][1] == pytest.approx(3.5 * 3 * 4.184 / 5)
    assert p["total"] == pytest.approx((-500.0 + 0.175 * 22 * 4.184, 0.175 * 22 * 4.184))


def test_reweighting_recovers_the_unbiased_pmf():
    rng = np.random.default_rng(1)
    T = 300.0
    beta = 1.0 / (es.KB_KJ_MOL_K * T)
    true = lambda x: 4.0 * (x ** 2 - 1.0) ** 2              # double well, kJ/mol (barrier 4 at x = 0)
    boost = lambda x: 0.7 * np.maximum(6.0 - true(x), 0.0)  # GaMD-like: raises the wells, flattens the barrier
    xs = np.linspace(-1.8, 1.8, 4001)
    p_boosted = np.exp(-beta * (true(xs) + boost(xs)))      # sampled on V + dV
    x = rng.choice(xs, size=200000, p=p_boosted / p_boosted.sum())
    edges = np.linspace(-1.6, 1.6, 33)
    rw = es.reweight_pmf(x, boost(x), T, edges, min_count=50)
    c = rw["centers"]
    ref = true(c) - true(c).min()
    for key in ("pmf_cumulant2", "pmf_exponential"):
        ok = np.isfinite(rw[key])
        assert np.nanmax(np.abs(rw[key][ok] - ref[ok] - np.nanmean(rw[key][ok] - ref[ok]))) < 0.6, key


def _umbrella_samples(free, centers, k, T, periodic=False, n=3000, seed=3):
    """Exact samples of each window: p_k(x) ∝ exp(-β(F(x) + k/2 d(x, c_k)^2)) on a fine grid."""
    rng = np.random.default_rng(seed)
    beta = 1.0 / (es.KB_KJ_MOL_K * T)
    lo, hi = (-np.pi, np.pi) if periodic else (centers[0] - 0.4, centers[-1] + 0.4)
    xs = np.linspace(lo, hi, 20001)
    out = []
    for c in centers:
        w = np.exp(-beta * (free(xs) + 0.5 * k * es._periodic_delta(xs, c, periodic) ** 2))
        out.append(rng.choice(xs, size=n, p=w / w.sum()))
    return out


def test_umbrella_analysis_recovers_a_double_well(tmp_path):
    # a flat landscape cannot tell a correct MBAR set-up from a wrong one (the bug this replaced passed it)
    T, k = 300.0, 1500.0
    free = lambda x: 12.0 * ((x - 0.6) ** 2 / 0.04 - 1.0) ** 2        # wells at 0.4 and 0.8 nm, barrier 12
    centers = np.linspace(0.3, 0.9, 25)
    samples = _umbrella_samples(free, centers, k, T)
    out = es.analyze_umbrella(samples, centers, k, T, False, {"name": "d", "type": "distance"}, str(tmp_path), nbins=40)
    xs, pmf = np.array(out["pmf"]["cv"]), np.array(out["pmf"]["kj_mol"])
    inner = (xs > 0.33) & (xs < 0.87)
    ref = free(xs[inner]) - free(xs[inner]).min()
    assert out["pmf_method"].startswith("MBAR")
    assert np.max(np.abs(pmf[inner] - ref)) < 1.5, (out["pmf_method"], np.round(pmf[inner] - ref, 2))
    assert out["min_neighbour_overlap"] > 0.1 and not out["warnings"]
    edges = np.linspace(0.3, 0.9, 41)
    wham = es._wham_1d(samples, centers, k, T, False, edges)
    mid = 0.5 * (edges[:-1] + edges[1:])
    ok = (mid > 0.33) & (mid < 0.87)
    assert np.nanmax(np.abs(wham[ok] - (free(mid[ok]) - free(mid[ok]).min()))) < 1.5
    assert os.path.isfile(out["files"]["pmf_png"]) and os.path.isfile(out["files"]["histograms_png"])


def test_umbrella_analysis_periodic_torsion(tmp_path):
    T, k = 300.0, 150.0
    free = lambda x: 6.0 * (1.0 + np.cos(2.0 * x))                     # minima at ±π/2, maxima at 0 and π
    centers = np.linspace(-np.pi, np.pi, 25)[:-1]
    samples = _umbrella_samples(free, centers, k, T, periodic=True)
    out = es.analyze_umbrella(samples, centers, k, T, True, {"name": "phi", "type": "dihedral"}, str(tmp_path), nbins=36)
    xs, pmf = np.array(out["pmf"]["cv"]), np.array(out["pmf"]["kj_mol"])
    assert np.max(np.abs(pmf - (free(xs) - free(xs).min()))) < 1.5


def test_map_atom_keys_tolerates_blank_chains_and_refuses_missing_atoms():
    import openmm.app as app

    top = app.PDBFile(ALA2).topology
    idx = es.map_atom_keys(top, PHI)
    assert [len(g) for g in idx] == [1, 1, 1, 1]
    with pytest.raises(RuntimeError, match="not found"):
        es.map_atom_keys(top, [[["A", "2", "", "XX"]]])


def test_cv_values_numpy_match_the_openmm_cv_force():
    import openmm as mm
    from openmm import unit

    system, pos = _toy_system()
    cv = es.cv_force({"type": "dihedral"}, [[0], [1], [2], [3]])
    cv.setForceGroup(5)
    system.addForce(cv)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos)
    phi = ctx.getState(getEnergy=True, groups={5}).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(4)]
    assert es.cv_values_numpy("dihedral", [[0], [1], [2], [3]], pos, masses) == pytest.approx(phi, abs=1e-5)


def _spec(tmp_path, method, block, cvs):
    return {"method": method, "pdb": ALA2, "out_dir": str(tmp_path / method), "seed": 7,
            "system": {"solvent": "implicit", "family": "amber14"}, "temperature_k": 300.0, "timestep_fs": 2.0,
            "cvs": cvs, method: block}


@pytest.fixture
def cpu_platform(monkeypatch):
    monkeypatch.setenv("LEONARDO_OPENMM_PLATFORM", "CPU")


def test_umbrella_runs_on_alanine_dipeptide(tmp_path, cpu_platform):
    spec = _spec(tmp_path, "umbrella", {"start": -2.6, "end": -0.6, "windows": 5, "k": 200.0,
                                        "equil_steps": 200, "prod_steps": 400, "stride": 20},
                 [{"name": "phi", "type": "dihedral", "atoms": PHI}])
    res = es.run_job(spec)
    assert res["ok"] and res["platform"] == "CPU" and res["windows"] == 5
    assert len(res["pmf"]["cv"]) > 3 and os.path.isfile(res["files"]["pmf_png"])


def test_metadynamics_runs_on_alanine_dipeptide_in_2d(tmp_path, cpu_platform):
    spec = _spec(tmp_path, "metadynamics", {"bias_factor": 6.0, "height_kj_mol": 1.0, "frequency": 100,
                                            "total_steps": 2000, "stride": 100},
                 [{"name": "phi", "type": "dihedral", "atoms": PHI, "bias_width": 0.35, "grid_width": 40},
                  {"name": "psi", "type": "dihedral", "atoms": PSI, "bias_width": 0.35, "grid_width": 40}])
    res = es.run_job(spec)
    assert res["ok"] and res["hills"] == 20 and len(res["fes_minimum_at"]) == 2
    assert os.path.isfile(res["files"]["fes_png"]) and os.path.isfile(res["files"]["cv_png"])


@pytest.mark.parametrize("kind", ["gamd", "amd"])
def test_boosted_md_runs_and_reweights_on_alanine_dipeptide(tmp_path, cpu_platform, kind):
    spec = _spec(tmp_path, kind, {"cmd_steps": 1000, "equil_steps": 200, "prod_steps": 2000, "stride": 20,
                                  "bins": 18, "min_count": 3},
                 [{"name": "phi", "type": "dihedral", "atoms": PHI}])
    res = es.run_job(spec)
    assert res["ok"] and res["frames"] == 100 and res["boost_mean_kj_mol"] > 0
    assert res["pmfs"][0]["pmf_cumulant2"]["cv"] and os.path.isfile(res["pmfs"][0]["png"])
    assert set(res["parameters_kj_mol"]) == {"total", "dihedral"}


def test_cli_writes_the_result_and_reports_errors(tmp_path, cpu_platform):
    spec = {"method": "nope", "out_dir": str(tmp_path), "result_json": str(tmp_path / "r.json")}
    path = tmp_path / "job.json"
    path.write_text(json.dumps(spec))
    assert es.main([str(path)]) == 1
    assert "unknown method" in json.loads((tmp_path / "r.json").read_text())["error"]


def test_pymol_ter_before_the_nme_cap_does_not_strip_it():
    # PyMOL writes TER before NME (not polymer to it): OpenMM made it a lone chain, Leonardo dropped it and the
    # dipeptide became ACE-ALA(-) with OXT; every enhanced-sampling run on it silently simulated the anion
    import openmm as mm
    from openmm import unit

    from leonardo import component_library_runtime as cl

    pymol_pdb = os.path.join(os.path.dirname(__file__), "data", "alanine_dipeptide_pymol_ter.pdb")
    prep = cl.prepare_system_for_solvation(pymol_pdb, family="amber14")
    assert prep["ok"] and prep["rejoined_caps"] == ["NME3"] and not prep["removed_isolated_residues"]
    top = prep["modeller"].topology
    assert [r.name for r in top.residues()] == ["ACE", "ALA", "NME"]
    assert "OXT" not in {a.name for a in top.atoms()}
    system = prep["forcefield"].createSystem(top)
    nb = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
    charge = sum(nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge) for i in range(nb.getNumParticles()))
    assert abs(charge) < 1e-6
