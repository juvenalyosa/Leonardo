import numpy as np
import pytest

abfe = pytest.importorskip("abfe_runtime")


def _lambda_applied(state, lam_schedule_value):
    """The phase -> lambda mapping of the ABFE loop (code_v2 / run_abfe_benchmark_probe_check)."""
    phase = str(state.get("phase", "pull"))
    if phase == "bound_settle":
        return float(state.get("bound_settle_lambda_value", 0.1))
    if phase == "probe":
        lam = float(state.get("probe_phase_lambda_value", lam_schedule_value))
    elif phase == "detached_settle":
        return float(state.get("detached_settle_lambda_value", 0.05))
    elif phase == "far_hold":
        return float(state.get("hold_lambda_value", 1.0) or 1.0)
    else:
        lam = float(lam_schedule_value)
    if state.get("state_driven_lambda") and phase in {"pull", "return", "bound_hold"}:
        lam = float(state.get("cycle_lambda", 0.0))
    return lam


def _run(n_cycles, n_samples=900, steering_mode="pull_hold_return", probe=False, n_dirs=1):
    state = abfe.build_abfe_adaptive_steering_state_payload(
        enabled=True, steering_mode=steering_mode, initial_pull_k_kj_nm2=800.0, n_direction_candidates=n_dirs,
        ligand_heavy_atoms=20, enable_probe_phase=probe, sample_every_ps=2.0,
    )
    plan = abfe.build_abfe_steering_lambda_schedule(n_samples=n_samples, steering_mode=steering_mode, steering_enabled=True)
    state = abfe.configure_abfe_steering_cycles(state, n_cycles=n_cycles, n_samples=n_samples,
                                                warmup_samples=int(plan["n_warm"]))
    k = float(state["current_pull_k_kj_nm2"])
    pos = 0.0                                   # ligand COM distance from the site (A)
    hist = {"site": [], "min": [], "contacts": [], "e": []}
    phases, ks, returns_k = [], [], []
    for s in range(n_samples):
        lam = _lambda_applied(state, plan["lambda_schedule"][s])
        target = 14.0 * lam
        if k >= 100.0:                          # the restraint moves the ligand toward its target
            pos += 0.5 * (target - pos)
        hist["site"].append(pos)
        hist["min"].append(float(np.clip(2.6 + 0.35 * pos, 2.6, 8.0)))
        hist["contacts"].append(float(max(0.0, 14.0 - 1.4 * pos)))
        hist["e"].append(-50.0)
        dec = abfe.build_abfe_adaptive_steering_update_decision(
            adaptive_state=state, sample_index=s, lam_value=lam, current_pull_k_kj_nm2=k,
            direction_count=n_dirs, min_dist_a=hist["min"], reactive_dist_a=hist["min"],
            site_min_dist_a=hist["min"], site_com_dist_a=hist["site"], contact_count_4a=hist["contacts"],
            total_e_kj=hist["e"],
        )
        prev_phase = state.get("phase")
        state = dec["adaptive_state"]
        if dec.get("update_pull_k"):
            k = float(dec["pull_k_kj_nm2"])
        if state.get("phase") == "return" and prev_phase != "return":
            returns_k.append(k)
        phases.append(state.get("phase"))
        ks.append(k)
    return state, phases, ks, returns_k, hist


def test_ligand_exits_and_reenters_for_every_cycle():
    state, phases, _ks, returns_k, hist = _run(n_cycles=3)
    assert state["cycles_completed"] == 3 and state["phase"] == "bound_hold"
    # three exits (far from the site) separated by returns to the bound region
    far = np.asarray(hist["site"]) >= state["far_unbound_site_com_a"]
    exits = int(np.sum(far[1:] & ~far[:-1]))
    assert exits == 3
    assert hist["site"][-1] <= state["bound_site_com_threshold_a"]
    # the return leg runs with the full pull force (not the ~1% detached-settle force)
    assert len(returns_k) == 3 and all(k >= 0.99 * state["initial_pull_k_kj_nm2"] for k in returns_k)


def test_single_cycle_and_probe_phase_still_rebind():
    state, phases, *_ = _run(n_cycles=1, probe=True, n_dirs=3)
    assert state["cycles_completed"] == 1 and "return" in phases and phases[-1] == "bound_hold"


def test_pull_hold_mode_is_unchanged_exit_only():
    state, phases, *_ = _run(n_cycles=3, steering_mode="pull_hold")
    assert not state.get("state_driven_lambda") and "return" not in phases
    assert phases[-1] in {"far_hold", "detached_settle"}


def test_return_boosts_force_when_the_ligand_does_not_rebind():
    state = abfe.build_abfe_adaptive_steering_state_payload(
        enabled=True, steering_mode="pull_hold_return", initial_pull_k_kj_nm2=800.0, ligand_heavy_atoms=20,
        sample_every_ps=2.0,
    )
    state = abfe.configure_abfe_steering_cycles(state, n_cycles=2, n_samples=600, warmup_samples=0)
    state.update(phase="return", cycle_lambda=0.0, last_update_sample=-999)
    stuck = [9.0] * 40                         # lambda already 0 but the ligand stays out
    k = 800.0
    boosted = False
    for s in range(200, 200 + 4 * state["return_timeout_samples"], state["update_interval_samples"]):
        dec = abfe.build_abfe_adaptive_steering_update_decision(
            adaptive_state=state, sample_index=s, lam_value=0.0, current_pull_k_kj_nm2=k,
            min_dist_a=[4.0] * 40, reactive_dist_a=[4.0] * 40, site_min_dist_a=[4.0] * 40,
            site_com_dist_a=stuck, contact_count_4a=[2.0] * 40, total_e_kj=[-10.0] * 40,
        )
        state = dec["adaptive_state"]
        if dec["reason"] == "return_stalled_increase_force":
            boosted = True
            assert dec["pull_k_kj_nm2"] > k
            k = dec["pull_k_kj_nm2"]
    assert boosted and state["phase"] == "return" and state["cycles_completed"] == 0


def test_cycle_configuration_scales_with_run_length():
    base = abfe.build_abfe_adaptive_steering_state_payload(enabled=True, steering_mode="pull_hold_return")
    short = abfe.configure_abfe_steering_cycles(base, n_cycles=3, n_samples=300, warmup_samples=18)
    long = abfe.configure_abfe_steering_cycles(base, n_cycles=3, n_samples=3000, warmup_samples=180)
    assert short["cycle_ramp_samples"] < long["cycle_ramp_samples"]
    assert short["state_driven_lambda"] and short["cycles_target"] == 3
    off = abfe.configure_abfe_steering_cycles(
        abfe.build_abfe_adaptive_steering_state_payload(enabled=True, steering_mode="pull_hold"), n_cycles=3)
    assert off["state_driven_lambda"] is False and off["cycles_target"] == 1
