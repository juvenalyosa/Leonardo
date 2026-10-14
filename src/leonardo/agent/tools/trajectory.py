"""Trajectory analysis (RMSD/RMSF, interaction occupancy, interaction energies) and ABFE-lite
(steered ligand exit through a guided tunnel, population-based binding free energy)."""

import contextlib
import math
import sys

from .. import presentation
from ..registry import ToolError
from . import _headless
from ._dialogjob import start_deferred_job
from ._widgets import jsonable, set_combo, set_value


def _cmd():
    from pymol import cmd

    return cmd


@contextlib.contextmanager
def _capture_global(dlg, name, sink):
    """Record the return value (and kwargs) of a function Leonardo calls through a module global."""
    module = sys.modules[type(dlg).__module__]
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        out = original(*args, **kwargs)
        sink.append({"kwargs": kwargs, "result": out})
        return out
    setattr(module, name, wrapper)
    try:
        yield sink
    finally:
        setattr(module, name, original)


def _as_list(values):
    """Leonardo returns lists or numpy arrays (whose truth value is ambiguous)."""
    if values is None:
        return []
    return list(values.tolist() if hasattr(values, "tolist") else values)


def _use_object(dlg, obj, ligand=None):
    cmd = _cmd()
    if obj not in cmd.get_names("objects"):
        raise ToolError(f"no object named {obj!r}")
    try:
        dlg._openmm_sync_object_to_analysis_context(obj)
    except Exception:
        dlg.loaded_object = obj
        dlg._traj_topology_object = obj
    dlg.loaded_object = obj
    dlg._traj_topology_object = obj
    if ligand:
        parts = dict(ligand)
        dlg.ligand_chain.setText(str(parts.get("chain", "")))
        dlg.ligand_resi.setText(str(parts.get("resi", "")))
        dlg.ligand_resn.setText(str(parts.get("resn", "")))
    return cmd.count_states(obj)


_POLICY_ALIASES = {"heuristic": "heuristic", "probe": "heuristic", "probe_control": "heuristic",
                   "probe+control": "heuristic", "bandit": "bandit", "safe_rl": "bandit"}


def _set_abfe_policy(dlg, policy):
    """The ABFE steering-policy combo (Probe + Control / Bandit); without the combo, Leonardo's setter."""
    mode = _POLICY_ALIASES.get(str(policy or "").strip().lower())
    if mode is None:
        raise ToolError(f"steering_policy must be one of {sorted(set(_POLICY_ALIASES.values()))}")
    if getattr(dlg, "cmb_abfe_policy_mode", None) is not None:
        set_combo(dlg, "cmb_abfe_policy_mode", mode)
    elif hasattr(dlg, "_set_abfe_policy_mode"):
        dlg._set_abfe_policy_mode(mode)
    return mode


def _abfe_policy_used(dlg, replicas_out=()):
    used = [str(r.get("policy_mode")) for r in replicas_out or () if r.get("policy_mode")]
    if used:
        return used[-1]
    try:
        return str(dlg._abfe_policy_mode(source="traj") or "heuristic")
    except Exception:
        return ""


_LIGAND = {"type": "object", "properties": {"chain": {"type": "string"}, "resi": {"type": "string"},
                                            "resn": {"type": "string"}},
           "description": "ligand residue; auto-detected when omitted"}


def register(reg):
    @reg.tool("trajectory_rmsd_rmsf", "RMSD over time (after fitting every frame on the first) and per-residue "
              "RMSF of a trajectory object (for example the output of run_md). Colors the protein by RMSF.",
              {"properties": {"object": {"type": "string"},
                              "selection": {"type": "string", "description": "atoms to fit/measure; default Cα"}},
               "required": ["object"]},
              permission="compute_light", title="RMSD / RMSF")
    def trajectory_rmsd_rmsf(ctx, object, selection="polymer and name CA"):
        cmd = _cmd()
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        n = cmd.count_states(object)
        if n < 2:
            raise ToolError(f"{object} has a single state; load or run a trajectory first")
        sel = f"({object}) and ({selection})"
        rmsd = [float(v) for v in cmd.intra_fit(sel, 1)]
        rmsd[0] = 0.0
        coords = {}
        for state in range(1, n + 1):
            space = {"rows": []}
            cmd.iterate_state(state, sel, "rows.append((chain, resi, resn, name, alt, x, y, z))", space=space)
            for ch, resi, resn, aname, alt, x, y, z in space["rows"]:
                coords.setdefault((ch, resi, resn), {}).setdefault((aname, alt), []).append((x, y, z))
        rmsf = []
        # RMSF of each atom about its own mean position, averaged over the residue's selected atoms
        for (ch, resi, resn), atoms in coords.items():
            vals = []
            for xyz in atoms.values():
                m = [sum(c[i] for c in xyz) / len(xyz) for i in range(3)]
                vals.append(math.sqrt(sum(sum((c[i] - m[i]) ** 2 for i in range(3)) for c in xyz) / len(xyz)))
            rmsf.append({"chain": ch, "resi": resi, "resn": resn, "rmsf_A": round(sum(vals) / len(vals), 2)})
        cmd.alter(f"({object})", "b = 0.0")
        for r in rmsf:
            chain = f"chain {r['chain']}" if r["chain"] else "chain ''"
            cmd.alter(f"({object}) and {chain} and resi {r['resi']}", f"b = {r['rmsf_A']}")
        cmd.spectrum("b", "blue_white_red", f"({object}) and polymer")
        presentation.focus(object)
        top = sorted(rmsf, key=lambda r: -r["rmsf_A"])[:12]
        step = max(1, len(rmsd) // 50)
        ctx.__dict__.setdefault("plots", {})["rmsd_rmsf"] = {"object": object, "rmsd": rmsd, "rmsf": rmsf}
        return {"object": object, "frames": n, "rmsd_A": [round(v, 2) for v in rmsd[::step]],
                "rmsd_final_A": round(rmsd[-1], 2), "rmsd_max_A": round(max(rmsd), 2),
                "rmsf_mean_A": round(sum(r["rmsf_A"] for r in rmsf) / len(rmsf), 2), "most_flexible": top,
                "scene": ctx.scene(f"{object}: colored by RMSF (blue rigid, red flexible)"),
                "_card": f"{n} frames; final RMSD **{rmsd[-1]:.2f} Å** (max {max(rmsd):.2f}); most flexible "
                         + ", ".join(f"{r['resn']}{r['resi']} {r['rmsf_A']} Å" for r in top[:4]) + "."}

    @reg.tool("trajectory_distance", "Distance between two atoms in every frame of a trajectory object (for example "
              "a catalytic Ser OG - His NE2 distance along run_md): the series, mean, standard deviation, range "
              "and the fraction of frames within a cutoff.",
              {"properties": {"object": {"type": "string"},
                              "atom_a": {"type": "string", "description": "single-atom selection inside the "
                                                                          "object, e.g. 'resi 195 and name OG'"},
                              "atom_b": {"type": "string", "description": "single-atom selection inside the object"},
                              "cutoff_A": {"type": "number", "minimum": 0.5, "maximum": 20,
                                           "description": "report the fraction of frames at or below this "
                                                          "distance (default 3.5 A, an H-bond)"}},
               "required": ["object", "atom_a", "atom_b"]},
              permission="compute_light", title="Distance along a trajectory")
    def trajectory_distance(ctx, object, atom_a, atom_b, cutoff_A=3.5):
        cmd = _cmd()
        if object not in cmd.get_names("objects"):
            raise ToolError(f"no object named {object!r}")
        sel_a, sel_b = f"({object}) and ({atom_a})", f"({object}) and ({atom_b})"
        for label, sel in (("atom_a", sel_a), ("atom_b", sel_b)):
            if cmd.count_atoms(sel) != 1:
                raise ToolError(f"{label} must select exactly one atom ({cmd.count_atoms(sel)} selected)")
        n = cmd.count_states(object)
        series = [float(cmd.get_distance(sel_a, sel_b, state=s)) for s in range(1, n + 1)]
        mean = sum(series) / len(series)
        sd = math.sqrt(sum((v - mean) ** 2 for v in series) / len(series))
        within = sum(1 for v in series if v <= float(cutoff_A)) / len(series)
        step = max(1, len(series) // 60)
        ctx.__dict__.setdefault("plots", {})["trajectory_distance"] = {"object": object, "series": series}
        return {"object": object, "frames": n, "atom_a": atom_a, "atom_b": atom_b,
                "distance_A": [round(v, 2) for v in series[::step]], "mean_A": round(mean, 3), "sd_A": round(sd, 3),
                "min_A": round(min(series), 2), "max_A": round(max(series), 2), "cutoff_A": float(cutoff_A),
                "fraction_within_cutoff": round(within, 3),
                "_card": f"{n} frames: **{mean:.2f} ± {sd:.2f} Å** (range {min(series):.2f}–{max(series):.2f}); "
                         f"{within:.0%} of frames ≤ {float(cutoff_A):g} Å."}

    @reg.tool("trajectory_interaction_occupancy", "Protein-ligand interactions along a trajectory: how often each "
              "contact (H-bond, salt bridge, hydrophobic, pi...) is present across the frames.",
              {"properties": {"object": {"type": "string"}, "ligand": _LIGAND,
                              "frame_step": {"type": "integer", "minimum": 1}}, "required": ["object"]},
              permission="compute_light", title="Interaction occupancy")
    def trajectory_interaction_occupancy(ctx, object, ligand=None, frame_step=1):
        dlg = ctx.dlg
        if _use_object(dlg, object, ligand) < 2:
            raise ToolError(f"{object} has a single state")
        set_value(dlg, "traj_step", int(frame_step))
        sink = []

        def run():
            with _capture_global(dlg, "trajectory_run_interaction_analysis", sink):
                dlg._run_trajectory_analysis()

        def collect(outcome):
            if outcome["error"]:
                raise ToolError(outcome["error"])
            if not sink or not sink[-1]["result"].get("ok"):
                raise ToolError("no interactions found along the trajectory")
            res = sink[-1]["result"]
            labels = list(res.get("top_labels") or [])
            freqs = [round(float(f), 1) for f in list(res.get("top_freqs") or [])]
            table = [{"interaction": lbl, "occupancy_pct": f} for lbl, f in zip(labels, freqs)]
            ctx.__dict__.setdefault("plots", {})["occupancy"] = {"object": object, "table": table,
                                                               "frames": res.get("frames_analyzed")}
            return {"object": object, "frames_analyzed": res.get("frames_analyzed"), "top_interactions": table,
                    "_card": f"{res.get('frames_analyzed')} frames; most persistent: "
                             + ", ".join(f"{t['interaction']} {t['occupancy_pct']}%" for t in table[:4]) + "."}

        return start_deferred_job(ctx, "occupancy", run, collect, description=f"interaction occupancy {object}")

    @reg.tool("trajectory_interaction_energy", "Protein-ligand interaction energy per frame (Lennard-Jones + "
              "Coulomb, OpenMM when available) and minimum distance along a trajectory.",
              {"properties": {"object": {"type": "string"}, "ligand": _LIGAND,
                              "local_cutoff_A": {"type": "number", "minimum": 4, "maximum": 20,
                                                 "description": "only protein atoms within this distance"}},
               "required": ["object"]},
              permission="compute_light", title="Interaction energy")
    def trajectory_interaction_energy(ctx, object, ligand=None, local_cutoff_A=None):
        dlg = ctx.dlg
        if _use_object(dlg, object, ligand) < 2:
            raise ToolError(f"{object} has a single state")
        with _headless.quiet() as q:
            res = dlg._run_trajectory_energy_discrimination(collect_only=True, local_mode=local_cutoff_A is not None,
                                                            local_cutoff_a=float(local_cutoff_A or 8.0))
        frames = _as_list(res.get("frames")) if isinstance(res, dict) else []
        if not frames:
            raise ToolError("; ".join(m["text"] for m in q.messages) or "energy analysis failed")

        def stats(key):
            vals = [float(v) for v in _as_list(res.get(key)) if v is not None and math.isfinite(float(v))]
            if not vals:
                return None
            return {"mean": round(sum(vals) / len(vals), 1), "min": round(min(vals), 1), "max": round(max(vals), 1)}
        ctx.__dict__.setdefault("plots", {})["interaction_energy"] = {
            "object": object, "frames": frames, "total": _as_list(res.get("total_energy_kj")),
            "lj": _as_list(res.get("lj_energy_kj")), "coulomb": _as_list(res.get("coul_energy_kj")),
            "min_distance": _as_list(res.get("min_dist_a"))}
        out = {"object": object, "frames": len(frames), "energy_mode": res.get("energy_mode"),
               "total_kj_mol": stats("total_energy_kj"), "lj_kj_mol": stats("lj_energy_kj"),
               "coulomb_kj_mol": stats("coul_energy_kj"), "min_distance_A": stats("min_dist_a")}
        t = out["total_kj_mol"] or {}
        out["_card"] = (f"Interaction energy over {out['frames']} frames: mean **{t.get('mean')} kJ/mol** "
                        f"(range {t.get('min')} to {t.get('max')}); an interaction energy, not a binding ΔG.")
        return out

    @reg.tool("run_abfe", "ABFE-lite / guided exit: steered MD pulls the ligand out of the site through a guided "
              "tunnel (protein restrained), holds and optionally returns it, over several replicas; the bound/"
              "unbound populations give an apparent binding free energy with a confidence interval. Needs a "
              "protein-ligand complex with a parameterized ligand (parametrize_ligand + export_ligand_files "
              "openmm_xml). Use at least 1 ns (shorter runs have no production window) and 3+ replicas for a "
              "reportable estimate. Tens of minutes to hours.",
              {"properties": {
                  "object": {"type": "string", "description": "complex (docked pose complex or MD output)"},
                  "ligand": _LIGAND,
                  "sim_ns": {"type": "number", "minimum": 0.2, "maximum": 200},
                  "replicas": {"type": "integer", "minimum": 1, "maximum": 12},
                  "steering": {"type": "string", "enum": ["pull_hold", "pull_hold_return"]},
                  "tunnel_radius_A": {"type": "number", "minimum": 1.0, "maximum": 10.0},
                  "steering_policy": {"type": "string", "enum": ["heuristic", "bandit"],
                                      "description": "adaptive steering controller: 'heuristic' = Leonardo's "
                                                     "'Probe + Control' (default), 'bandit' = experimental bandit "
                                                     "policy; omitted = keep the current GUI choice"},
              }, "required": ["object"]},
              permission="compute_heavy", title="ABFE-lite (guided exit)")
    def run_abfe(ctx, object, ligand=None, sim_ns=3.0, replicas=3, steering="pull_hold_return", tunnel_radius_A=3.5,
                 steering_policy=None):
        dlg = ctx.dlg
        _use_object(dlg, object, ligand)
        if steering_policy:
            _set_abfe_policy(dlg, steering_policy)
        xml = str(getattr(dlg, "_parameterized_ligand_xml", "") or "")
        if xml and hasattr(dlg, "edit_abfe_ligand_xml") and not dlg.edit_abfe_ligand_xml.text().strip():
            dlg.edit_abfe_ligand_xml.setText(xml)
        set_combo(dlg, "cmb_abfe_steering_mode", steering, required=False)
        if hasattr(dlg, "cb_abfe_guide_tunnel"):
            dlg.cb_abfe_guide_tunnel.setChecked(True)
        if hasattr(dlg, "spin_abfe_guide_tunnel_radius"):
            dlg.spin_abfe_guide_tunnel_radius.setValue(float(tunnel_radius_A))
        replicas_out, reports = [], []
        original_md = dlg._run_abfe_lite_local_md

        def md(*args, **kwargs):
            out = original_md(*args, **kwargs)
            if isinstance(out, dict):
                if "policy_mode" not in out and kwargs.get("policy_mode"):
                    out = dict(out, policy_mode=kwargs.get("policy_mode"))
                replicas_out.append(out)
            return out

        def run():
            _headless.preset_prompts({"ABFE-lite Simulation Length": float(sim_ns),
                                      "ABFE-lite Replicas": int(replicas)})
            dlg._run_abfe_lite_local_md = md
            try:
                with _capture_global(dlg, "abfe_build_free_energy_report_payload", reports):
                    dlg._run_trajectory_free_energy_calculation(source="traj")
            finally:
                dlg._run_abfe_lite_local_md = original_md
                _headless.preset_prompts({})

        def collect(outcome):
            if outcome["error"]:
                raise ToolError(outcome["error"])
            if not reports:
                raise ToolError("ABFE did not produce a result (see Leonardo's messages)")
            kw = dict(reports[-1]["kwargs"])
            keys = ("dg_selected", "selected_estimator", "ci_low", "ci_high", "dg_exp", "dg_pop", "dg_cum",
                    "dg_reweighted_standard", "kd_selected_m", "production_reliability", "n_bound", "n_unbound")
            out = {k: jsonable(kw.get(k)) for k in keys if kw.get(k) is not None}
            report = reports[-1]["result"] if isinstance(reports[-1]["result"], dict) else {}
            if report.get("message_text"):
                out["report"] = str(report["message_text"])[:3000]
            out.update({"energy_units": "kJ/mol", "object": object, "sim_ns": sim_ns, "replicas": len(replicas_out) or replicas,
                        "steering_policy": _abfe_policy_used(dlg, replicas_out),
                        "per_replica": [jsonable({k: r.get(k) for k in ("dg_pop_kj", "p_bound", "kd_app_m",
                                                                        "policy_mode")})
                                        for r in replicas_out]})
            ctx.__dict__.setdefault("plots", {})["abfe"] = {
                "object": object, "report": out.get("report", ""), "dg": out.get("dg_selected"),
                "replicas": [{"frames": _as_list(r.get("frames")), "distance": _as_list(r.get("distance_a")),
                              "energy": _as_list(r.get("total_energy_kj")),
                              "bound_cutoff": r.get("bound_cutoff_a")} for r in replicas_out]}
            traj = next((r.get("abfe_traj_object") for r in reversed(replicas_out) if r.get("abfe_traj_object")), "")
            if traj and traj in _cmd().get_names("objects"):
                presentation.focus(traj, buffer=6.0)
                out["trajectory_object"] = traj
            out["scene"] = ctx.scene(f"ABFE-lite of {object}")
            dg = out.get("dg_selected")
            if dg is not None:
                out["_card"] = (f"ABFE-lite **ΔG ≈ {dg} kJ/mol** ({out.get('selected_estimator')}; CI "
                                f"{out.get('ci_low')} to {out.get('ci_high')}); reliability "
                                f"{out.get('production_reliability')}. Apparent value from a guided-exit protocol, "
                                "not alchemical ABFE.")
            else:
                out["_card"] = (f"ABFE-lite ran ({out.get('replicas')} replica(s), {sim_ns} ns) but gave **no binding "
                                f"estimate** (estimator {out.get('selected_estimator')}; bound frames "
                                f"{out.get('n_bound')}, unbound {out.get('n_unbound')}). Diagnostic only: more replicas "
                                "and longer runs are needed.")
            return out

        return start_deferred_job(ctx, "abfe", run, collect, description=f"ABFE-lite {object}")
