import numpy as np
import pytest
from abfe_runtime import (
    ABFE_TUNNEL_ENERGY_EXPRESSION,
    abfe_tunnel_wall_energy,
    build_abfe_memory_guard_decision,
    build_abfe_pymol_state_budget,
    build_abfe_tunnel_restraint_payload,
    openmm_cpu_thread_budget,
)


def _funnel(**kw):
    args = dict(origin_nm=[0.0, 0.0, 0.0], direction=[0.0, 0.0, 2.0], radius_a=3.5, shape="funnel")
    args.update(kw)
    return build_abfe_tunnel_restraint_payload(**args)


def test_funnel_is_wide_at_the_site_and_narrow_outside():
    p = _funnel()
    assert p["ok"] and p["shape"] == "funnel"
    assert p["tan_alpha"] > 0.0 and 0.0 < p["zcc_nm"] < p["zmax_nm"]
    off_axis = 0.6  # nm, beyond the 0.35 nm cylinder
    assert abfe_tunnel_wall_energy([off_axis, 0.0, 0.0], p) == 0.0  # inside the cone
    assert abfe_tunnel_wall_energy([off_axis, 0.0, p["zcc_nm"] + 0.3], p) > 0.0  # outside the cylinder
    assert abfe_tunnel_wall_energy([0.0, 0.0, p["zmax_nm"] + 0.2], p) > 0.0  # axial cap
    assert abfe_tunnel_wall_energy([0.0, 0.0, 0.5 * p["zmax_nm"]], p) == 0.0  # on the axis


def test_cylinder_shape_keeps_the_previous_wall():
    p = build_abfe_tunnel_restraint_payload(origin_nm=[0, 0, 0], direction=[1, 0, 0], radius_a=3.5)
    assert p["shape"] == "cylinder" and p["tan_alpha"] == 0.0
    assert abfe_tunnel_wall_energy([0.0, 0.6, 0.0], p) > 0.0
    assert "tan_a" in p["energy_expression"] and p["energy_expression"] == ABFE_TUNNEL_ENERGY_EXPRESSION


def test_unknown_shape_falls_back_to_cylinder():
    p = build_abfe_tunnel_restraint_payload(origin_nm=[0, 0, 0], direction=[1, 0, 0], shape="cone?")
    assert p["shape"] == "cylinder"
    assert any(n.startswith("unknown_shape_defaulted") for n in p["notes"])


def test_funnel_expression_matches_numpy_mirror_in_openmm():
    mm = pytest.importorskip("openmm")
    p = _funnel()
    system = mm.System()
    system.addParticle(12.0)
    force = mm.CustomCentroidBondForce(1, p["energy_expression"])
    params = {
        "k_cyl": p["k_radial_kj_mol_nm2"], "k_axial": p["k_axial_kj_mol_nm2"], "r0": p["radius_nm"],
        "zmin": p["zmin_nm"], "zmax": p["zmax_nm"], "tan_a": p["tan_alpha"], "zcc": p["zcc_nm"],
    }
    for name, val in params.items():
        force.addGlobalParameter(name, float(val))
    for name, val in zip(("ox", "oy", "oz"), p["origin_nm"]):
        force.addGlobalParameter(name, float(val))
    for name, val in zip(("ux", "uy", "uz"), p["direction"]):
        force.addGlobalParameter(name, float(val))
    force.addBond([force.addGroup([0])], [])
    system.addForce(force)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    for pos in ([0.6, 0.0, 0.0], [0.6, 0.0, 1.2], [0.0, 0.0, 3.0], [0.2, -0.1, -0.6]):
        ctx.setPositions([mm.Vec3(*pos)])
        e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)
        assert e == pytest.approx(abfe_tunnel_wall_energy(pos, p), rel=1e-5, abs=1e-6)


def test_memory_guard_aborts_on_low_available_memory_only():
    low = build_abfe_memory_guard_decision(available_mb=900.0, total_mb=16384.0)
    assert low["should_abort"] and low["reasons"][0].startswith("available_memory_low")
    ok = build_abfe_memory_guard_decision(available_mb=6000.0, total_mb=16384.0)
    assert not ok["should_abort"]
    # 8 % of 64 GB is above the absolute floor
    assert build_abfe_memory_guard_decision(available_mb=4000.0, total_mb=65536.0)["should_abort"]
    assert not build_abfe_memory_guard_decision(available_mb=None, total_mb=None)["should_abort"]
    assert build_abfe_memory_guard_decision(available_mb=8000.0, swap_used_mb=9000.0)["should_abort"]


def test_pymol_state_budget_scales_with_system_size():
    small = build_abfe_pymol_state_budget(n_atoms=500)
    big = build_abfe_pymol_state_budget(n_atoms=20000)
    assert small["max_states"] == 1500 and not small["limited_by_memory"]
    assert big["limited_by_memory"] and big["max_states"] < 200
    assert big["estimated_mb"] <= 384.0 + 1e-6


def test_cpu_thread_budget_leaves_cores_free_and_honors_env():
    assert openmm_cpu_thread_budget(cpu_count=10, env_value="") == 8
    assert openmm_cpu_thread_budget(cpu_count=2, env_value="") == 1
    assert openmm_cpu_thread_budget(cpu_count=10, env_value="3") == 3
    assert openmm_cpu_thread_budget(cpu_count=10, env_value="nope") == 8
    assert np.isfinite(openmm_cpu_thread_budget())


def test_funnel_axis_prefers_the_open_side_of_a_half_shell():
    from abfe_runtime import build_abfe_funnel_axis_payload

    # protein atoms fill the half-space z < 0.3 nm around a ligand at the origin: the only
    # way out is +z
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2.0, 2.0, size=(4000, 3))
    prot = pts[(pts[:, 2] < 0.3) & (np.linalg.norm(pts, axis=1) > 0.45)]
    lig = np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
    coords = np.vstack([prot, lig])
    payload = build_abfe_funnel_axis_payload(
        coords_nm=coords,
        protein_heavy_idx=list(range(len(prot))),
        ligand_heavy_idx=[len(prot), len(prot) + 1],
    )
    assert payload["ok"] and payload["blocked_atoms"] == 0
    assert payload["direction"][2] > 0.8


def test_state_machine_returns_when_out_of_site_but_stuck_on_surface():
    from abfe_runtime import (
        build_abfe_adaptive_steering_state_payload,
        build_abfe_adaptive_steering_update_decision,
        configure_abfe_steering_cycles,
    )

    state = build_abfe_adaptive_steering_state_payload(enabled=True, steering_mode="pull_hold_return")
    state = configure_abfe_steering_cycles(state, n_cycles=3, n_samples=150, warmup_samples=0)
    state["phase"] = "pull"
    timeout = int(state["out_of_site_timeout_samples"])
    site, mind, contacts, energy = [], [], [], []
    phase = "pull"
    # decisions are only taken every few samples (warm-up history, update interval)
    for s in range(8 * timeout + 20):
        # 20 A from the site but in contact with the surface (2.4 A): never "far unbound"
        site.append(20.0)
        mind.append(2.4)
        contacts.append(8.0)
        energy.append(-100.0)
        dec = build_abfe_adaptive_steering_update_decision(
            adaptive_state=state,
            sample_index=s,
            lam_value=1.0,
            current_pull_k_kj_nm2=float(state.get("current_pull_k_kj_nm2", 800.0)),
            min_dist_a=list(mind),
            reactive_dist_a=list(site),
            site_min_dist_a=list(site),
            site_com_dist_a=list(site),
            contact_count_4a=list(contacts),
            total_e_kj=list(energy),
        )
        state = dict(dec["adaptive_state"])
        phase = str(state.get("phase"))
        if phase == "return":
            assert dec["reason"] == "out_of_site_timeout_enter_return"
            break
    assert phase == "return"
