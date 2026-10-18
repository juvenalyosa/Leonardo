"""OpenMMContextProcess: the Context in a child process behaves like an in-process one (CPU platform)."""
import numpy as np
import pytest

mm = pytest.importorskip("openmm")
psutil = pytest.importorskip("psutil")
unit = pytest.importorskip("openmm.unit")

from leonardo.openmm_context_process import OpenMMContextProcess
from leonardo.openmm_dynamics_runtime import (
    openmm_make_context,
    openmm_minimize,
    openmm_step_synced,
    openmm_update_force_in_context,
)


def _system(n=40, seed=3):
    rng = np.random.default_rng(seed)
    system = mm.System()
    for _ in range(n):
        system.addParticle(12.0)
    pos = rng.uniform(0.0, 2.0, (n, 3))
    lj = mm.CustomNonbondedForce("4*0.4*((0.3/r)^12-(0.3/r)^6)")
    for _ in range(n):
        lj.addParticle([])
    lj.setNonbondedMethod(lj.CutoffNonPeriodic)
    lj.setCutoffDistance(1.0)
    lj.setForceGroup(1)
    system.addForce(lj)
    pull = mm.CustomExternalForce("0.5*k_pull*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    pull.addGlobalParameter("k_pull", 50.0)
    for p in ("x0", "y0", "z0"):
        pull.addPerParticleParameter(p)
    for i in range(5):
        pull.addParticle(i, [1.0, 1.0, 1.0])
    pull.setForceGroup(3)
    system.addForce(pull)
    return system, pull, pos


def _cpu():
    return mm.Platform.getPlatformByName("CPU"), {"Threads": "1"}


def _run(context, integrator, pos, steps):
    context.setPositions(pos * unit.nanometer)
    openmm_minimize(context, 50)
    context.setVelocitiesToTemperature(300.0 * unit.kelvin, 7)
    openmm_step_synced(integrator, context, steps, sync_every=100)
    st = context.getState(getPositions=True, getEnergy=True)
    return st.getPositions(asNumpy=True).value_in_unit(unit.nanometer), st


def _children_alive(proxy):
    return [pid for pid in proxy.child_pids if psutil.pid_exists(pid)
            and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE]


def test_proxy_matches_an_in_process_context():
    system, _pull, pos = _system()
    platform, props = _cpu()
    integ_a = mm.LangevinMiddleIntegrator(300, 5, 0.002)
    integ_a.setRandomNumberSeed(11)
    x_ref, st_ref = _run(mm.Context(system, integ_a, platform, props), integ_a, pos, 600)
    integ_b = mm.LangevinMiddleIntegrator(300, 5, 0.002)
    integ_b.setRandomNumberSeed(11)
    proxy = OpenMMContextProcess(system, integ_b, "CPU", props)
    try:
        assert proxy.getPlatform().getName() == "CPU"
        x, st = _run(proxy, integ_b, pos, 600)
        assert isinstance(st, mm.State)
        assert np.allclose(x, x_ref, atol=1e-4)
        e_ref = st_ref.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        assert st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == pytest.approx(e_ref, abs=1e-3)
    finally:
        proxy.close()
    assert not _children_alive(proxy)


def test_proxy_recycles_its_child_and_keeps_the_run_going():
    system, _pull, pos = _system()
    integ = mm.LangevinMiddleIntegrator(300, 5, 0.002)
    integ.setRandomNumberSeed(5)
    proxy = OpenMMContextProcess(system, integ, "CPU", {"Threads": "1"}, rss_cap_mb=0.0)   # recycle every call
    try:
        proxy.setPositions(pos * unit.nanometer)
        openmm_minimize(proxy, 50)
        proxy.setParameter("k_pull", 80.0)
        proxy.setVelocitiesToTemperature(300.0 * unit.kelvin, 3)
        t0 = proxy.getState().getTime().value_in_unit(unit.picosecond)
        for _ in range(10):
            openmm_step_synced(integ, proxy, 100)
        st = proxy.getState(getPositions=True, getEnergy=True, getParameters=True)
        assert proxy.recycles >= 9 and len(set(proxy.child_pids)) == proxy.recycles + 1
        assert st.getTime().value_in_unit(unit.picosecond) == pytest.approx(t0 + 2.0, abs=1e-6)
        assert st.getParameters()["k_pull"] == pytest.approx(80.0)
        assert np.isfinite(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
        assert np.all(np.isfinite(st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))
    finally:
        proxy.close()
    assert not _children_alive(proxy)


def test_force_parameter_updates_reach_the_child():
    system, pull, pos = _system()
    platform, props = _cpu()
    ref_integ = mm.VerletIntegrator(0.001)
    ref = mm.Context(system, ref_integ, platform, props)
    proxy = OpenMMContextProcess(system, mm.VerletIntegrator(0.001), "CPU", props)
    try:
        for ctx in (ref, proxy):
            ctx.setPositions(pos * unit.nanometer)
        for i in range(5):
            pull.setParticleParameters(i, i, [0.2, 1.7, 0.4])
        openmm_update_force_in_context(pull, ref)
        openmm_update_force_in_context(pull, proxy)

        def e3(c):
            return c.getState(getEnergy=True, groups=1 << 3).getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole)

        assert e3(proxy) == pytest.approx(e3(ref), rel=1e-6)
        assert e3(proxy) > 1.0
    finally:
        proxy.close()


def test_factory_uses_the_child_only_when_wanted(monkeypatch):
    system, _pull, _pos = _system()
    platform, props = _cpu()
    monkeypatch.setenv("LEONARDO_OPENMM_CONTEXT_PROCESS", "0")
    ctx = openmm_make_context(system, mm.VerletIntegrator(0.001), platform, props)
    assert isinstance(ctx, mm.Context)
    monkeypatch.setenv("LEONARDO_OPENMM_CONTEXT_PROCESS", "1")
    proxy = openmm_make_context(system, mm.VerletIntegrator(0.001), platform, props)
    try:
        assert isinstance(proxy, OpenMMContextProcess)
    finally:
        proxy.close()


def test_worker_errors_come_back_as_exceptions():
    system, _pull, _pos = _system()
    proxy = OpenMMContextProcess(system, mm.VerletIntegrator(0.001), "CPU", {"Threads": "1"})
    try:
        with pytest.raises(RuntimeError, match="context process error"):
            proxy.setParameter("no_such_parameter", 1.0)
        proxy.setParameter("k_pull", 3.0)                 # still usable after an error
        assert proxy.getParameter("k_pull") == pytest.approx(3.0)
    finally:
        proxy.close()
