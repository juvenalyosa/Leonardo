"""Free-energy bookkeeping of ABFE-lite: which frames define V_eff, and which CI goes with which DeltaG."""
import contextlib
import math

import numpy as np

from leonardo.abfe_runtime import (
    build_abfe_bias_reweighted_population_payload,
    build_abfe_free_energy_report_payload,
    build_abfe_post_sampling_metrics,
)
from leonardo.agent.tools import trajectory


def _bound_then_unbound(n_bound=12, n_unbound=12):
    """Bound COM at the origin, unbound COM in a 0.2 nm cube 1.5 nm away along x."""
    rng = np.random.default_rng(3)
    n = n_bound + n_unbound
    com = np.zeros((n, 3))
    com[:n_bound] = rng.uniform(-0.05, 0.05, size=(n_bound, 3))
    com[n_bound:] = 1.5 + rng.uniform(0.0, 0.2, size=(n_unbound, 3))
    com[n_bound:, 1:] -= 1.5
    site = np.r_[np.full(n_bound, 2.0), np.full(n_unbound, 14.0)]
    mins = np.r_[np.full(n_bound, 2.5), np.full(n_unbound, 6.0)]
    contacts = np.r_[np.full(n_bound, 12.0), np.full(n_unbound, 0.0)]
    return dict(
        frames=list(range(1, n + 1)),
        total_e=list(np.r_[np.full(n_bound, -80.0), np.full(n_unbound, -1.0)]),
        lj_e=[0.0] * n,
        coul_e=[0.0] * n,
        bias_e=[0.0] * n,
        min_dist=list(mins),
        site_com_dist=list(site),
        protein_contact_count_4a=list(contacts),
        lig_com=com.tolist(),
        far_hold_stable_mask=[False] * n_bound + [True] * n_unbound,
        temperature_k=300.0,
    ), com[n_bound:]


def test_v_eff_is_the_unbound_volume_not_the_exit_path():
    kwargs, unbound_com = _bound_then_unbound()
    payload = build_abfe_post_sampling_metrics(**kwargs)
    assert payload["n_bound"] > 0 and payload["n_unbound"] > 0
    span = np.maximum(unbound_com.max(axis=0) - unbound_com.min(axis=0), 0.2)
    expected = float(np.prod(span))
    # before the fix the box spanned bound + unbound frames (~1.7 x 0.2 x 0.2 nm^3)
    assert math.isclose(float(payload["v_eff_nm3"]), expected, rel_tol=1e-9)
    kbt = 0.008314462618 * 300.0
    assert math.isclose(
        float(payload["standard_state_correction_kj"]), kbt * math.log(1.66054 / expected), rel_tol=1e-9
    )


def test_v_eff_falls_back_to_all_state_frames_without_unbound_frames():
    kwargs, _ = _bound_then_unbound(n_bound=6, n_unbound=1)
    payload = build_abfe_post_sampling_metrics(**kwargs)
    assert np.isfinite(float(payload["v_eff_nm3"]))


def _reweight_payload():
    rng = np.random.default_rng(11)
    n = 80
    bound = np.zeros(n, dtype=bool)
    bound[:50] = True
    unbound = ~bound
    bias = rng.uniform(0.0, 1.0, size=n)
    return build_abfe_bias_reweighted_population_payload(
        bound_valid_mask=bound, unbound_valid_mask=unbound, bias_valid=bias,
        temperature_k=300.0, v_eff_nm3=0.05, n_boot=300,
    )


def test_report_labels_reweight_ci_as_state_and_gives_the_standard_ci():
    rw = _reweight_payload()
    corr = float(rw["standard_state_correction_kj"])
    assert corr > 5.0  # small V_eff -> CI(state) cannot bracket DeltaG(standard)
    assert not (rw["ci_low"] <= rw["dg_reweighted_standard_kj"] <= rw["ci_high"])
    rep = build_abfe_free_energy_report_payload(
        n_dist_valid=80, n_bound=50, n_unbound=30,
        dg_reweighted=rw["dg_reweighted_kj"], dg_reweighted_state=rw["dg_reweighted_state_kj"],
        dg_reweighted_standard=rw["dg_reweighted_standard_kj"],
        standard_state_correction_kj=corr, c_eff_m=rw["c_eff_m"],
        kd_app_reweighted_m=rw["kd_app_reweighted_m"], p_kd_reweighted=rw["p_kd_reweighted"],
        reweight_ci_low=rw["ci_low"], reweight_ci_high=rw["ci_high"],
    )
    line = rep["reweight_line"]
    assert "CI95(state)=" in line
    lo, hi = rw["ci_low"] + corr, rw["ci_high"] + corr
    assert lo <= rw["dg_reweighted_standard_kj"] <= hi
    assert f"CI95(app,std)=[{lo:.3f}, {hi:.3f}]" in line


class _Reg:
    def __init__(self):
        self.tools = {}

    def tool(self, name, *a, **k):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


class _Dlg:
    def _run_abfe_lite_local_md(self, *a, **k):
        return {}

    def _run_trajectory_free_energy_calculation(self, source=""):
        return None

    def _abfe_policy_mode(self, source=""):
        return "heuristic"


class _Ctx:
    def __init__(self):
        self.dlg = _Dlg()

    def scene(self, *a, **k):
        return "scene"


def _collect_for(monkeypatch, report_kwargs):
    """Run the run_abfe tool with a fake dialog whose 'run' produces one report with report_kwargs."""
    reg = _Reg()
    trajectory.register(reg)
    jobs = {}
    monkeypatch.setattr(trajectory, "_use_object", lambda *a, **k: 1)
    monkeypatch.setattr(
        trajectory, "start_deferred_job",
        lambda ctx, kind, run, collect, description="": jobs.update(run=run, collect=collect),
    )

    @contextlib.contextmanager
    def fake_capture(dlg, name, sink):
        sink.append({"kwargs": dict(report_kwargs), "result": {}})
        yield sink

    monkeypatch.setattr(trajectory, "_capture_global", fake_capture)
    reg.tools["run_abfe"](_Ctx(), "cplx")
    jobs["run"]()
    return jobs["collect"]({"error": None})


def test_run_abfe_reports_the_ci_of_the_selected_estimate(monkeypatch):
    kw = dict(dg_selected=-20.0, selected_estimator="bias_reweighted_hold_windows",
              dg_exp=-150.0, ci_low=-160.0, ci_high=-140.0,
              reweight_ci_low=-30.0, reweight_ci_high=-26.0,
              standard_state_correction_kj=8.0, dg_selected_standard=-20.0, dg_selected_state=-28.0,
              production_reliability="qualitative")
    out = _collect_for(monkeypatch, kw)
    assert out["ci_low"] == -22.0 and out["ci_high"] == -18.0
    assert out["dg_exp_ci_low"] == -160.0 and out["dg_exp_ci_high"] == -140.0
    assert "CI -22.0 to -18.0" in out["_card"]
    assert "-160" not in out["_card"]


def test_run_abfe_gives_no_ci_for_rescored_estimates(monkeypatch):
    kw = dict(dg_selected=-35.0, selected_estimator="gbsa_obc2_hold_window_blockexpavg",
              dg_exp=-150.0, ci_low=-160.0, ci_high=-140.0,
              reweight_ci_low=-30.0, reweight_ci_high=-26.0, standard_state_correction_kj=8.0)
    out = _collect_for(monkeypatch, kw)
    assert "ci_low" not in out and "ci_high" not in out
    assert out["dg_exp_ci_low"] == -160.0
    assert "no CI for this estimator" in out["_card"]
    assert "-160" not in out["_card"]
