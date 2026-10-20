import os

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from leonardo import simulation_health_runtime as h  # noqa: E402
from leonardo.agent import checks  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "prep")


def _pdb(name):
    return app.PDBFile(os.path.join(DATA, name))


def _codes(findings, level=None):
    return {f["code"] for f in findings if level is None or f["level"] == level}


def _xyz(pdb):
    return np.array(pdb.positions.value_in_unit(unit.nanometer))


def test_crystal_structures_have_no_red_findings():
    for name in ("2LZT.pdb", "7RSA.pdb", "1ATP.pdb"):    # 1ATP: phosphorylated SEP/TPO in the chain
        found = h.check_pdb(os.path.join(DATA, name))
        assert not _codes(found, "red"), (name, h.format_findings(found))
        assert "d_residue" not in _codes(found), name


def test_gap_bonded_across_is_red():
    found = h.check_pdb(os.path.join(DATA, "1M17.pdb"))       # 12 residues missing in the activation loop
    red = [f for f in found if f["code"] == "bonded_gap"]
    assert red and red[0]["level"] == "red" and "prepare_protein" in red[0]["fix"]


def test_missing_atoms_overlap_and_chirality_are_detected():
    pdb = _pdb("2LZT.pdb")
    m = app.Modeller(pdb.topology, pdb.positions)
    res = [r for r in m.topology.residues() if r.name == "LYS"][0]
    m.delete([a for a in res.atoms() if a.name == "NZ"])
    xyz = np.array(m.positions.value_in_unit(unit.nanometer))
    # mirror the CB of a valine through the N-CA-C plane: a D-residue
    val = [r for r in m.topology.residues() if r.name == "VAL"][0]
    idx = {a.name: a.index for a in val.atoms()}
    n, ca, c, cb = (xyz[idx[k]] for k in ("N", "CA", "C", "CB"))
    normal = np.cross(n - ca, c - ca)
    normal /= np.linalg.norm(normal)
    xyz[idx["CB"]] = cb - 2 * np.dot(cb - ca, normal) * normal
    # put a leucine's CD1 on another residue's atom
    leu = [r for r in m.topology.residues() if r.name == "LEU"][0]
    other = [r for r in m.topology.residues() if r.name == "TRP"][0]
    xyz[[a.index for a in leu.atoms() if a.name == "CD1"][0]] = xyz[[a.index for a in other.atoms()][5]] + 0.003
    found = h.structure_findings(m.topology, xyz)
    assert {"missing_atoms", "overlapping_atoms"} <= _codes(found, "red")
    d = [f for f in found if f["code"] == "d_residue"][0]
    assert any(f"VAL{val.id}" in w for w in d["where"])


def _system(pdb_name="2LZT.pdb", hmr=False, periodic=False):
    pdb = _pdb(pdb_name)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml" if periodic else "implicit/gbn2.xml")
    m = app.Modeller(pdb.topology, pdb.positions)
    m.delete([r for r in m.topology.residues() if r.name not in h.AMINO])      # waters, nitrate
    m.addHydrogens(ff)
    kw = {"constraints": app.HBonds}
    if hmr:
        kw["hydrogenMass"] = 3.0 * unit.amu
    if periodic:
        m.addSolvent(ff, padding=0.3 * unit.nanometer, neutralize=False)
        system = ff.createSystem(m.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer, **kw)
    else:
        system = ff.createSystem(m.topology, nonbondedMethod=app.NoCutoff, **kw)
    return m, system


def test_timestep_rules():
    m, system = _system()
    assert "timestep" in _codes(h.system_findings(system, m.topology, m.positions, timestep_fs=4.0), "red")
    assert "timestep" not in _codes(h.system_findings(system, m.topology, m.positions, timestep_fs=2.0))
    m, system = _system(hmr=True)
    assert "timestep" not in _codes(h.system_findings(system, m.topology, m.positions, timestep_fs=4.0))


def test_fractional_charge_is_red_and_charged_box_yellow():
    m, system = _system()
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    q, s, e = nb.getParticleParameters(0)
    nb.setParticleParameters(0, q + 0.3 * unit.elementary_charge, s, e)
    assert "fractional_charge" in _codes(h.system_findings(system, m.topology, m.positions), "red")
    m, system = _system(periodic=True)                   # lysozyme is +8, not neutralized, 3 A padding
    found = h.system_findings(system, m.topology, m.positions, mode="explicit", cutoff_nm=1.0)
    assert "charged_box" in _codes(found, "yellow")
    assert "box_small" in _codes(found)


def _peptide_sim(n_res=8, dt_fs=2.0):
    pdb = _pdb("2LZT.pdb")
    m = app.Modeller(pdb.topology, pdb.positions)
    keep = [r for r in m.topology.residues() if r.name in h.AMINO][-n_res:]          # the C-terminal residues: OXT is in the file
    m.delete([r for r in m.topology.residues() if r not in keep])
    ff = app.ForceField("amber14-all.xml")
    m.addHydrogens(ff)
    system = ff.createSystem(m.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, dt_fs * unit.femtoseconds)
    sim = app.Simulation(m.topology, system, integ, openmm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions(m.positions)
    return m, system, sim


def test_monitor_healthy_run_and_ligand_drift():
    m, system, sim = _peptide_sim()
    sim.minimizeEnergy(maxIterations=200)
    tail = [a.index for a in list(m.topology.residues())[-1].atoms()]         # stands in for a ligand
    mon = h.MDMonitor(sim.context, system, m.topology, temperature_K=300, ligand_atoms=tail)
    mon.set_reference()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    for _ in range(3):
        sim.step(100)
        sample = mon.check("production", 300, steps=100)
    assert 100 < sample["T_K"] < 600 and "ca_rmsd_A" in sample
    rep = mon.finish()
    assert rep["checks"] == 3 and not _codes(rep["findings"], "red")
    # carry the "ligand" 10 A away: flagged, not fatal
    xyz = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    xyz[tail] += np.array([1.0, 0.0, 0.0])
    sim.context.setPositions(xyz)
    mon.check("production", 300)
    assert "ligand_left" in _codes(mon.findings, "yellow")
    assert h.summarize(mon.findings)["confidence"] == "medium"


def test_monitor_stops_on_huge_forces_and_explosions():
    m, system, sim = _peptide_sim()
    xyz = np.array(m.positions.value_in_unit(unit.nanometer))
    heavy = [a.index for a in m.topology.atoms() if a.element.symbol != "H"]
    xyz[heavy[20]] = xyz[heavy[40]] + 0.005                    # two atoms nearly on top of each other
    sim.context.setPositions(xyz)
    mon = h.MDMonitor(sim.context, system, m.topology)
    with pytest.raises(h.SimulationHealthError) as err:
        mon.set_reference()
    assert err.value.finding["code"] in ("huge_force", "nan_start") and err.value.finding["level"] == "red"

    m, system, sim = _peptide_sim(dt_fs=2.0)
    sim.minimizeEnergy(maxIterations=200)
    mon = h.MDMonitor(sim.context, system, m.topology, temperature_K=300, timestep_fs=2.0)
    mon.set_reference()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    sim.step(50)
    mon.check("production", 300, steps=50)
    sim.integrator.setStepSize(25 * unit.femtoseconds)
    item = None
    try:
        for _ in range(40):
            sim.step(20)
            mon.check("production", 300, steps=20)
    except h.SimulationHealthError as e:
        item = e.finding
    except Exception as e:                                     # OpenMM: "Particle coordinate is NaN"
        item = mon.explain_exception(e, "production")
    assert item is not None and item["level"] == "red"
    assert item["code"] in ("nan", "runaway_temperature")
    assert item["where"], "the fastest atoms point at the residue responsible"
    assert h.summarize(mon.findings, stopped=item)["confidence"] == "low"


def test_agent_checks_report_health():
    r = {"health": {"confidence": "low", "findings": [h.finding("red", "nan", "The simulation exploded (NaN)",
                                                                "coordinates became NaN", ["A:LYS1 NZ"], "minimize")]}}
    lines = checks.post_checks("run_md", {}, r)
    assert any("exploded" in w and "A:LYS1" in w for w in lines)
    assert any("LOW" in w for w in lines)


@pytest.fixture()
def agent(tmp_path):
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    app_qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pymol import cmd

    from leonardo.agent.tools import AgentContext, build_registry

    cmd.reinitialize()
    ctx = AgentContext(dlg=None)

    def workspace(sub=""):
        path = os.path.join(str(tmp_path), sub)
        os.makedirs(path, exist_ok=True)
        return path

    ctx.workspace = workspace
    yield ctx, build_registry(ctx), cmd
    del app_qt


def test_agent_check_and_md_guard_on_objects_saved_from_pymol(agent):
    from leonardo.agent.registry import ToolError

    ctx, reg, cmd = agent
    assert reg.get("check_simulation_system").permission == "read"
    cmd.load(os.path.join(DATA, "1M17.pdb"), "egfr")
    out = reg.call("check_simulation_system", {"object": "egfr", "timestep_fs": 4.0}, ctx)
    assert out["confidence"] == "low" and "bonded_gap" in _codes(out["findings"], "red")
    assert "hmr" in _codes(out["findings"], "info")
    with pytest.raises(ToolError, match="unphysical"):
        reg.call("run_md", {"object": "egfr"}, ctx)
    # residues copied on top of themselves (PyMOL bonds them by distance, the CONECT records carry it)
    cmd.load(os.path.join(DATA, "2LZT.pdb"), "lyz")
    cmd.create("dup", "lyz and resi 20-22")
    cmd.alter("dup", "chain='B'")
    cmd.create("both", "lyz or dup")
    cmd.bond("both and chain A and resi 21 and name CA", "both and chain B and resi 21 and name CA")
    out = reg.call("check_simulation_system", {"object": "both"}, ctx)
    assert "overlapping_atoms" in _codes(out["findings"], "red")
    out = reg.call("check_simulation_system", {"object": "lyz"}, ctx)
    assert out["confidence"] in ("high", "medium") and not _codes(out["findings"], "red")
