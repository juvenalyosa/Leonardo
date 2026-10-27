"""Hardware awareness for the assistant: what the machine can do (system_info) and how long a
calculation will take on it (estimate_cost, also shown on every heavy-step confirmation)."""

import os

from ..registry import ToolError


def _hw():
    from ... import hardware_runtime

    return hardware_runtime


def _cmd():
    from pymol import cmd

    return cmd


def _profile(ctx):
    hw = _hw()
    prof = getattr(ctx.dlg, "_hw_profile", None) if ctx.dlg is not None else None
    return prof or hw._load_cache() or hw.detect(benchmark=False)


def _symbols(selection):
    cmd = _cmd()
    model = cmd.get_model(selection)
    return [a.symbol for a in model.atom]


def _count(selection):
    try:
        return int(_cmd().count_atoms(selection))
    except Exception:
        return 0


def _md_atoms(obj, solvent, padding_a=10.0):
    try:
        return _hw().md_system_atoms(_cmd(), obj, solvent, padding_a)
    except Exception:
        return 0


def estimate(ctx, tool, args):
    """{'minutes': float, 'text': str, 'advice': str} for a tool call, or None when not applicable."""
    hw, prof = _hw(), _profile(ctx)
    a = dict(args or {})
    thr = prof.get("threads") or {}
    if tool == "run_md":
        solvent = a.get("solvent", "implicit")
        ns = float(a.get("production_ns", 0.1)) * 1.1 + 0.014
        return hw.estimate_md(_md_atoms(a.get("object", ""), solvent), ns, solvent, prof=prof)
    if tool == "minimize_structure":
        n = _md_atoms(a.get("object", ""), a.get("solvent", "implicit"))
        m = 0.3 + n * int(a.get("max_iterations", 2000)) * 2e-6
        return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} ({n} atoms with hydrogens)"}
    if tool == "run_abfe":
        ns = float(a.get("sim_ns", 3.0)) * int(a.get("replicas", 3)) + 0.3
        est = hw.estimate_md(_md_atoms(a.get("object", ""), "implicit"), ns, "implicit", prof=prof)
        est["text"] = f"{est['text']} for {a.get('replicas', 3)} replica(s) of {a.get('sim_ns', 3.0)} ns"
        return est
    if tool == "run_enhanced_sampling":
        return _enhanced_sampling_estimate(hw, prof, a)
    if tool in ("run_psi4", "run_uv_spectrum"):
        sel = a.get("selection") or a.get("object") or a.get("reactant") or ""
        region = a.get("region", "ligand")
        if tool == "run_psi4" and sel and region == "ligand":
            sel = f"({sel}) and not polymer and not (solvent and resn HOH+WAT+H2O+DOD+TIP+TIP3+TIP4+TP3+T3P+SOL+SPC)"
        if not sel:
            return None
        syms = _symbols(sel)
        if not any(s in ("H", "h") for s in syms):
            heavy = len(syms)
            syms = syms + ["H"] * heavy                  # hydrogens added/implicit
        op = a.get("operation", "energy") if tool == "run_psi4" else "energy"
        est = hw.estimate_psi4(syms, op, a.get("method", "b3lyp"), a.get("basis", "def2-svp"),
                               threads=a.get("threads") or thr.get("psi4"), prof=prof)
        if tool == "run_uv_spectrum":
            k = max(1.0, int(a.get("states", 10)) / 3.0)
            est["minutes"] = round(est["minutes"] * 4 * k, 1)
            est["text"] = f"~{hw._fmt(est['minutes'])} (SCF + {a.get('states', 10)} TDDFT states)"
        return est
    if tool == "run_mopac":
        n = _count(a.get("selection", "")) or 30
        return hw.estimate_mopac(n * (2 if _count(f"({a.get('selection', '')}) and elem H") == 0 else 1),
                                 bool(a.get("frequencies", True)), prof)
    if tool == "run_qmmm":
        op = a.get("operation", "probe")
        if op.startswith("mopac_"):
            return {"minutes": 1.0, "text": "~seconds to a few minutes (MOPAC on the QM atoms)"}
        per_call = 0.05 * float(a.get("max_qm_atoms", 60) or 60)             # s per MOPAC call
        calls = {"probe": 1, "electrostatic": 1, "optimization": 60, "thermochemistry": 80,
                 "reaction_scan": int(a.get("scan_points", 9)) * 150, "transition_state": 400}.get(op, 20)
        m = 0.5 + per_call * calls / 60.0
        return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} ({calls} MOPAC calls on {thr.get('mopac')} cores)",
                "advice": "long QM/MM: fewer scan points or a smaller QM region" if m > 60 else ""}
    if tool in ("run_docking", "run_redocking"):
        reps = int(a.get("repeats", 1))
        exh = int(a.get("exhaustiveness", 16 if tool == "run_docking" else 8))
        m = reps * (0.3 + exh * 0.08) * 8.0 / max(1, int(thr.get("smina") or 4))
        n_ens = int(a.get("ensemble_structures", 0) or 0)
        if n_ens:
            m = m * (1 + n_ens) + _mc_minutes(prof, n_ens)
            return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} (Monte Carlo ensemble of {n_ens} + redocking "
                                                   f"into {n_ens + 1} receptors)"}
        return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} (smina on {thr.get('smina')} cores)"}
    if tool == "run_cluster_reaction":
        atoms = list((getattr(ctx.dlg, "_psi4_cluster_state", {}) or {}).get("atoms") or [])
        if not atoms:
            return {"minutes": 0, "text": "build the cluster first to estimate"}
        syms = [str(x.get("element", "C")) for x in atoms]
        est = hw.estimate_psi4(syms, "ts_qst3", a.get("method", "b3lyp"), a.get("basis", "def2-svp"),
                               threads=a.get("threads") or thr.get("psi4"), prof=prof)
        m = est["minutes"] * (2.5 if a.get("irc", True) else 1.5)
        return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} for a {len(atoms)}-atom cluster (TS search"
                                               + (" + IRC" if a.get("irc", True) else "") + ")",
                "advice": "large: consider MOPAC (run_qmmm mopac_reaction_path) to locate the TS first, a smaller "
                          "cluster or basis" if m > 120 else ""}
    if tool == "generate_mc_ensemble":
        n = int(a.get("n_structures", 10))
        m = _mc_minutes(prof, n)
        return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} for {n} structures on "
                                               f"{(prof.get('md_platform') or {}).get('name', 'CPU')}"}
    if tool == "run_virtual_screening":
        n = len(a.get("compounds") or []) or int(a.get("max_ligands", 250))
        m = n * (0.1 + int(a.get("exhaustiveness", 8)) * 0.03) * 8.0 / max(1, int(thr.get("smina") or 4))
        return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} for {n} compounds"}
    if tool == "run_lead_optimization":
        n = int(a.get("max_analogs", 40) or 40)
        m = n * 0.15 * 8.0 / max(1, int(thr.get("smina") or 4))
        return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} for up to {n} analogs"}
    if tool in ("build_homology_model", "model_mutant", "repair_structure_gaps"):
        if tool == "model_mutant" and a.get("method") == "local":
            return {"minutes": 0.1, "text": "~seconds (only the mutated side chains are rebuilt)"}
        n_res = len(str(a.get("sequence", "")))
        if not n_res and a.get("object"):
            # mutants and gap repairs model one chain of a loaded structure (there is no sequence argument)
            chain = str(a.get("chain") or "")
            n_res = _count(f"({a['object']}) and polymer.protein and name CA"
                           + (f" and chain {chain}" if chain else ""))
        n_res = n_res or 300
        m = 0.2 + int(a.get("num_models", 5 if tool == "build_homology_model" else 3)) * n_res * 0.0007
        return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} (MODELLER)"}
    if tool == "predict_structure_alphafold":
        md = prof.get("md_platform") or {}
        n = len(str(a.get("sequence", "")))
        gpus = [g for g in prof.get("gpus") or [] if g.get("vendor") == "NVIDIA"]
        if gpus:
            m = 2 + n * 0.02 * int(a.get("num_models", 5))
            return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} on {gpus[0]['name']}"}
        return {"minutes": 0, "text": "no CUDA GPU here: Google Colab hand-off (the run happens in the browser)",
                "advice": "" if md.get("is_gpu") else ""}
    if tool == "animate_reaction":
        frames = int(a.get("max_frames", 24))
        m = frames * (0.02 if a.get("orbital_program", "mopac") == "mopac" else 0.5) + 0.1
        return {"minutes": round(m, 1), "text": f"~{hw._fmt(m)} ({frames} frames)"}
    return None


def _enhanced_sampling_estimate(hw, prof, a):
    """action='run' only. Costs measured on an Apple M4 (OpenCL): metadynamics ~plain MD, aMD/GaMD ~2x (custom
    integrator with group energies every step), ANI ~150/n_ML_atoms ns/day on the CPU (~10x on a CUDA GPU)."""
    if a.get("action") != "run":
        return None
    mode = a.get("mode", "umbrella")
    solvent = a.get("solvent", "implicit")
    n = _md_atoms(a.get("object", ""), solvent)
    if mode == "umbrella":
        windows = int(a.get("windows") or 24)
        per = float(a.get("production_ns_per_window") or 0.4) + float(a.get("equilibration_ps_per_window") or 100) / 1000
        est = hw.estimate_md(n, windows * per, solvent, prof=prof)
        est["text"] = f"{est['text']} for {windows} windows of {per:g} ns"
        return est
    if mode == "metadynamics":
        ns = float(a.get("production_ns") or 1.0)
        est = hw.estimate_md(n, ns * 1.1, solvent, prof=prof)
        est["text"] = f"{est['text']} for {ns:g} ns of metadynamics"
        return est
    if mode in ("amd", "gamd"):
        ns = float(a.get("production_ns") or 1.0) * 1.1 + float(a.get("cmd_ns") or 0.05)
        est = hw.estimate_md(n, ns * 2.0, solvent, prof=prof)
        est["text"] = f"{est['text']} for {ns:g} ns of {mode.upper()} (boosted steps cost ~2x)"
        return est
    if mode == "openmm_ml":
        ns = float(a.get("production_ns") or 0.1)
        sel = a.get("ml_region_selection") or "organic"
        n_ml = n if a.get("ml_mode", "ml_only") == "ml_only" else max(1, _count(f"({a.get('object', '')}) and ({sel})") * 2)
        cuda = (prof.get("md_platform") or {}).get("name") == "CUDA"
        ns_day = (1500.0 if cuda else 150.0) / max(n_ml, 1)
        minutes = ns / ns_day * 1440.0 + 0.5
        if a.get("ml_mode") == "mixed" and not cuda:
            minutes += hw.estimate_md(n, ns, solvent, platform_name="CPU", prof=prof).get("minutes", 0)
        return {"minutes": round(minutes, 1),
                "text": f"~{hw._fmt(minutes)} ({n_ml} ML atoms on the {'GPU' if cuda else 'CPU'}, {ns:g} ns)",
                "advice": "" if cuda else "ANI runs on the CPU here (openmm-torch has no OpenCL kernels); keep runs short"}
    return None


def _mc_minutes(prof, n_structures):
    """Monte Carlo side-chain sampling: ~22 moves per saved structure, one 100-step minimization each."""
    per_move = 0.04 if (prof.get("md_platform") or {}).get("is_gpu") else 0.3
    return (15.0 + n_structures * 22 * per_move) / 60.0


ESTIMATED_TOOLS = ("run_md", "minimize_structure", "run_abfe", "run_psi4", "run_uv_spectrum", "run_mopac",
                   "run_qmmm", "run_docking", "run_redocking", "run_virtual_screening", "run_lead_optimization",
                   "build_homology_model", "model_mutant", "repair_structure_gaps", "predict_structure_alphafold",
                   "animate_reaction", "generate_mc_ensemble", "run_cluster_reaction", "run_enhanced_sampling")


def register(reg):
    @reg.tool("system_info", "The computer Leonardo runs on and what it means for calculations: CPU cores "
              "(performance/efficiency), RAM, GPUs, the OpenMM platform MD uses (benchmarked: GPU vs CPU speed), "
              "threads used for Psi4/MOPAC/smina, MD speeds measured on this machine, and which programs are "
              "installed (OpenMM, Psi4, MOPAC, MODELLER, smina, fpocket, antechamber, ColabFold...). Use it to "
              "choose sizes and to warn the user about long runs.",
              {"properties": {"refresh": {"type": "boolean", "description": "re-run detection and the GPU benchmark"}}},
              permission="read", title="System info")
    def system_info(ctx, refresh=False):
        import shutil

        hw = _hw()
        prof = hw.detect(benchmark=True, refresh=bool(refresh))
        if ctx.dlg is not None:
            ctx.dlg._hw_profile = prof
            try:
                ctx.dlg._update_hardware_label()
            except Exception:
                pass
        tools = {name: bool(shutil.which(name)) for name in
                 ("smina", "fpocket", "obabel", "antechamber", "parmchk2", "tleap", "mopac", "psi4", "colabfold_batch",
                  "s-dftd3", "dftd4", "nvidia-smi")}
        try:
            import modeller  # noqa: F401

            tools["modeller"] = True
        except Exception:
            tools["modeller"] = False
        md = prof.get("md_platform") or {}
        out = {"cpu": prof.get("cpu"), "gpus": prof.get("gpus"), "openmm_platforms": prof.get("openmm_platforms"),
               "openmm_benchmark_steps_per_s": {k: round(v.get("steps_per_s", 0)) if isinstance(v, dict)
                                                and v.get("steps_per_s") else v.get("error") if isinstance(v, dict)
                                                else v for k, v in (prof.get("openmm_benchmark") or {}).items()},
               "md_platform": md, "threads": prof.get("threads"), "measured_speeds": prof.get("calibration"),
               "programs": tools,
               "guidance": [
                   f"MD runs on {md.get('name')}" + (f", {md.get('speedup_vs_cpu')}x faster than the CPU"
                                                     if md.get("is_gpu") else " (no faster GPU platform)"),
                   f"Psi4 uses {(prof.get('threads') or {}).get('psi4')} threads (performance cores); "
                   "more threads on efficiency cores barely help",
                   "long MD (>30 min estimated) is worth a CUDA GPU; QM and QM/MM scale with CPU cores"]}
        out["_card"] = hw.summary(prof) + "."
        return out

    @reg.tool("estimate_cost", "Estimate how long a calculation will take on this computer before running it "
              "(MD, ABFE, Psi4, TDDFT, MOPAC, QM/MM, docking, screening, MODELLER, AlphaFold, reaction "
              "animations), with advice when it is long. Give the tool name and the arguments you plan to use.",
              {"properties": {"tool": {"type": "string", "enum": list(ESTIMATED_TOOLS)},
                              "args": {"type": "object"}}, "required": ["tool"]},
              permission="read", title="Time estimate")
    def estimate_cost(ctx, tool, args=None):
        try:
            est = estimate(ctx, tool, args or {})
        except Exception as exc:
            raise ToolError(f"cannot estimate {tool}: {exc}") from exc
        if not est:
            return {"tool": tool, "text": "no estimate for this call", "_card": "No estimate."}
        return dict(est, tool=tool, _card=f"{tool}: {est.get('text')}"
                    + (f" — {est['advice']}" if est.get("advice") else ""))


def benchmark_cached():
    return os.path.isfile(_hw().CACHE_PATH)


def recommended(ctx, key, default):
    """Thread/memory setting for this machine (hardware profile), e.g. recommended(ctx, 'psi4', 4)."""
    try:
        return int((_profile(ctx).get("threads") or {}).get(key) or default)
    except Exception:
        return default
