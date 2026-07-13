import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from .ligand_vib_refine_runtime import (
    DEFAULT_PSO_BOUNDS,  # noqa: F401 - compatibility re-export
    DEFAULT_SCALE_VECTOR,  # noqa: F401 - compatibility re-export
    _deserialize_scaled_system,
    build_ligand_system_payload,
    build_mode_comparison_rows,
    build_qm_optimization_stages,
    build_scale_search_config,
    evaluate_scale_vector,
    export_refined_forcefield_xml,
    load_first_sdf_record_atoms,
    load_mol2_full_records,
    make_summary_payload,
    make_summary_plots,
    qm_result_matches_stage,
    regenerate_refined_amber_files,
    run_bounded_pso,
    run_mopac_stationary_from_mol2,
    run_mopac_stationary_from_sdf,
    run_mopac_stationary_from_xyz,  # noqa: F401 - compatibility re-export
    run_psi4_frequency_from_xyz,
    run_psi4_stationary_from_mol2,
    run_psi4_stationary_from_sdf,
    run_psi4_stationary_from_xyz,
    validate_qm_start_sdf_matches_template_mol2,
    write_json,
    write_markdown_report,
    write_mode_comparison_csv,
    write_mol2_with_coords,
    write_refined_frcmod,
)
from .psi4_runtime import psi4_guess_multiplicity_from_atoms, psi4_load_vib_data


def _run_checked(cmd, cwd):
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={proc.returncode}: {' '.join(str(x) for x in cmd)}\n"
            f"STDERR:\n{proc.stderr[-2000:]}\nSTDOUT:\n{proc.stdout[-2000:]}"
        )
    return proc


def _resolve_tool_path(name, explicit=""):
    raw = str(explicit or "").strip()
    if raw and os.path.isfile(raw):
        return str(Path(raw).resolve())
    hit = shutil.which(str(name))
    if hit:
        return str(Path(hit).resolve())
    env_bin = Path(sys.executable).resolve().parent
    candidates = [env_bin / str(name), env_bin / "wrapped_progs" / str(name)]
    if str(name).lower() == "tleap":
        candidates.append(env_bin / "teLeap")
    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())
    raise FileNotFoundError(f"Could not resolve required executable: {name}")


def _bootstrap_template_from_sdf(
    sdf_file,
    out_dir,
    antechamber_bin="antechamber",
    parmchk2_bin="parmchk2",
    tleap_bin="tleap",
    residue_name="LIG",
):
    from rdkit import Chem

    workdir = Path(out_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    sdf_path = Path(str(sdf_file)).resolve()
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = None
    for one in suppl:
        if one is not None:
            mol = Chem.Mol(one)
            break
    if mol is None:
        raise RuntimeError(f"Failed to read a molecule from bootstrap SDF: {sdf_path}")
    formal_charge = int(sum(int(at.GetFormalCharge()) for at in mol.GetAtoms()))
    mol2_path = workdir / "ligand_gaff2.mol2"
    frcmod_path = workdir / "ligand_gaff2.frcmod"
    prmtop_path = workdir / "ligand.prmtop"
    inpcrd_path = workdir / "ligand.inpcrd"
    tleap_in = workdir / "tleap.in"
    _run_checked(
        [
            str(antechamber_bin),
            "-i",
            str(sdf_path),
            "-fi",
            "sdf",
            "-o",
            str(mol2_path),
            "-fo",
            "mol2",
            "-at",
            "gaff2",
            "-c",
            "bcc",
            "-nc",
            str(int(formal_charge)),
            "-dr",
            "no",
            "-s",
            "2",
        ],
        cwd=workdir,
    )
    _run_checked(
        [
            str(parmchk2_bin),
            "-i",
            str(mol2_path),
            "-f",
            "mol2",
            "-o",
            str(frcmod_path),
        ],
        cwd=workdir,
    )
    tleap_script = "\n".join(
        [
            "source leaprc.gaff2",
            f"{str(residue_name)[:3]} = loadmol2 {mol2_path.name}",
            f"loadamberparams {frcmod_path.name}",
            f"saveamberparm {str(residue_name)[:3]} {prmtop_path.name} {inpcrd_path.name}",
            "quit",
            "",
        ]
    )
    tleap_in.write_text(tleap_script, encoding="utf-8")
    _run_checked([str(tleap_bin), "-f", str(tleap_in)], cwd=workdir)
    return {
        "ok": True,
        "sdf": str(sdf_path),
        "formal_charge": int(formal_charge),
        "mol2": str(mol2_path),
        "frcmod": str(frcmod_path),
        "prmtop": str(prmtop_path),
        "inpcrd": str(inpcrd_path),
    }


def _looks_like_placeholder(path_value):
    txt = str(path_value or "").strip().lower()
    if not txt:
        return False
    return txt.startswith("your_") or txt.startswith("real_") or txt in {
        "your_ligand.mol2",
        "your_ligand.frcmod",
        "your_ligand.prmtop",
        "your_ligand.inpcrd",
        "your_outdir",
        "real_ligand.mol2",
        "real_ligand.frcmod",
        "real_ligand.prmtop",
        "real_ligand.inpcrd",
        "real_output_dir",
    }


def _validate_inputs(args):
    bootstrap_sdf = str(getattr(args, "bootstrap_template_sdf", "") or "").strip()
    required_files = {}
    if not bootstrap_sdf:
        required_files = {
            "mol2": str(args.mol2),
            "prmtop": str(args.prmtop),
            "inpcrd": str(args.inpcrd),
        }
    optional_files = {"frcmod": str(args.frcmod or "").strip()}
    extra_files = {
        "qm-start-sdf": str(args.qm_start_sdf or "").strip(),
        "bootstrap-template-sdf": bootstrap_sdf,
    }
    for label, path in required_files.items():
        if _looks_like_placeholder(path):
            raise ValueError(f"`--{label}` still uses the placeholder value `{path}`. Replace it with a real file path.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"`--{label}` file was not found: {path}")
    for label, path in optional_files.items():
        if not path:
            continue
        if _looks_like_placeholder(path):
            raise ValueError(f"`--{label}` still uses the placeholder value `{path}`. Replace it with a real file path.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"`--{label}` file was not found: {path}")
    for label, path in extra_files.items():
        if not path:
            continue
        if _looks_like_placeholder(path):
            raise ValueError(f"`--{label}` still uses the placeholder value `{path}`. Replace it with a real file path.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"`--{label}` file was not found: {path}")
    if bootstrap_sdf:
        return
    mol2_payload = load_mol2_full_records(str(args.mol2))
    atoms = list(mol2_payload.get("atoms", []) or [])
    if not atoms:
        raise ValueError(f"`--mol2` does not contain any readable MOL2 atoms: {args.mol2}")
    if str(args.qm_start_sdf or "").strip():
        validate_qm_start_sdf_matches_template_mol2(str(args.qm_start_sdf), str(args.mol2))


def _is_opt_convergence_failure(qm_result):
    txt = str((qm_result or {}).get("error", "") or "").lower()
    return ("could not converge geometry optimization" in txt) or ("energy has increased in a minimization" in txt)


def main():
    ap = argparse.ArgumentParser(description="Refine ligand bonded/LJ parameters against QM normal modes with fixed template charges.")
    ap.add_argument("--mol2", default="", help="Template ligand MOL2 with final charges.")
    ap.add_argument("--prmtop", default="", help="Ligand Amber prmtop.")
    ap.add_argument("--inpcrd", default="", help="Ligand Amber inpcrd.")
    ap.add_argument("--out-dir", required=True, help="Output working directory.")
    ap.add_argument("--frcmod", default="", help="Optional input frcmod path for provenance.")
    ap.add_argument("--qm-start-sdf", default="", help="Optional ligand-only SDF used only for the initial QM geometry source.")
    ap.add_argument("--qm-engine", choices=["psi4", "mopac"], default="psi4")
    ap.add_argument("--bootstrap-template-sdf", default="", help="Optional SDF used to build missing template mol2/frcmod/prmtop/inpcrd via AmberTools.")
    ap.add_argument("--method", default="wb97x-d")
    ap.add_argument("--basis", default="def2-SVP")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--memory-gb", type=int, default=4)
    ap.add_argument("--geom-maxiter", type=int, default=120)
    ap.add_argument("--swarm-size", type=int, default=8)
    ap.add_argument("--n-iter", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qm-stage", choices=["stationary", "optimize_only"], default="stationary")
    ap.add_argument("--qm-opt-staging", choices=["auto", "off"], default="auto")
    ap.add_argument("--preopt-method", default="hf")
    ap.add_argument("--preopt-basis", default="sto-3g")
    ap.add_argument("--midopt-method", default="b3lyp")
    ap.add_argument("--midopt-basis", default="6-31g(d)")
    ap.add_argument("--disable-preopt-fallback", action="store_true")
    ap.add_argument("--fit-mode", choices=["global", "typed_hybrid"], default="typed_hybrid")
    ap.add_argument("--max-typed-bond-types", type=int, default=4)
    ap.add_argument("--max-typed-angle-types", type=int, default=4)
    ap.add_argument("--max-typed-torsion-types", type=int, default=6)
    ap.add_argument("--max-typed-lj-types", type=int, default=4)
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--mopac-exe", default=os.environ.get("MOPAC_BIN", "mopac"))
    ap.add_argument("--antechamber-bin", default=os.environ.get("ANTECHAMBER_BIN", "antechamber"))
    ap.add_argument("--parmchk2-bin", default=os.environ.get("PARMCHK2_BIN", "parmchk2"))
    ap.add_argument("--tleap-bin", default="tleap")
    args = ap.parse_args()
    _validate_inputs(args)

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    bootstrap_sdf = str(args.bootstrap_template_sdf or "").strip()
    bootstrap_payload = {}
    if bootstrap_sdf:
        antechamber_bin = _resolve_tool_path("antechamber", args.antechamber_bin)
        parmchk2_bin = _resolve_tool_path("parmchk2", args.parmchk2_bin)
        tleap_bin = _resolve_tool_path("tleap", args.tleap_bin)
        bootstrap_dir = os.path.join(out_dir, "bootstrap_template")
        bootstrap_payload = _bootstrap_template_from_sdf(
            sdf_file=bootstrap_sdf,
            out_dir=bootstrap_dir,
            antechamber_bin=antechamber_bin,
            parmchk2_bin=parmchk2_bin,
            tleap_bin=tleap_bin,
        )
        args.mol2 = str(bootstrap_payload["mol2"])
        args.frcmod = str(bootstrap_payload["frcmod"])
        args.prmtop = str(bootstrap_payload["prmtop"])
        args.inpcrd = str(bootstrap_payload["inpcrd"])
        if not str(args.qm_start_sdf or "").strip():
            args.qm_start_sdf = str(bootstrap_payload["sdf"])
        validate_qm_start_sdf_matches_template_mol2(str(args.qm_start_sdf), str(args.mol2))
    summary_path = os.path.join(out_dir, "refinement_summary.json")
    report_path = os.path.join(out_dir, "refinement_report.md")
    mode_csv = os.path.join(out_dir, "mode_comparison.csv")
    plot_png = os.path.join(out_dir, "qm_ff_vibrational_summary.png")
    refined_mol2 = os.path.join(out_dir, "ligand_refined.mol2")
    refined_frcmod = os.path.join(out_dir, "ligand_refined.frcmod")
    summary = {
        "ok": False,
        "inputs": {
            "mol2": os.path.abspath(args.mol2),
            "frcmod": os.path.abspath(args.frcmod) if str(args.frcmod or "").strip() else "",
            "qm_start_sdf": os.path.abspath(args.qm_start_sdf) if str(args.qm_start_sdf or "").strip() else "",
            "bootstrap_template_sdf": os.path.abspath(bootstrap_sdf) if bootstrap_sdf else "",
            "prmtop": os.path.abspath(args.prmtop),
            "inpcrd": os.path.abspath(args.inpcrd),
            "qm_engine": str(args.qm_engine),
            "method": str(args.method),
            "basis": str(args.basis),
            "mopac_exe": str(args.mopac_exe),
            "geom_maxiter": int(args.geom_maxiter),
            "swarm_size": int(args.swarm_size),
            "n_iter": int(args.n_iter),
            "seed": int(args.seed),
            "qm_stage": str(args.qm_stage),
            "qm_opt_staging": str(args.qm_opt_staging),
            "preopt_method": str(args.preopt_method),
            "preopt_basis": str(args.preopt_basis),
            "midopt_method": str(args.midopt_method),
            "midopt_basis": str(args.midopt_basis),
            "fit_mode": str(args.fit_mode),
        },
    }
    if bootstrap_payload:
        summary["bootstrap_template"] = dict(bootstrap_payload)
    try:
        template_mol2_payload = load_mol2_full_records(str(args.mol2))
        template_atoms = list(template_mol2_payload.get("atoms", []) or [])
        template_charge = int(np.round(np.sum([float(a.get("charge", 0.0)) for a in template_atoms]))) if template_atoms else 0
        template_multiplicity = int(psi4_guess_multiplicity_from_atoms(template_atoms, template_charge)) if template_atoms else 1
        qm_start_sdf = str(args.qm_start_sdf or "").strip()
        if qm_start_sdf:
            sdf_info = load_first_sdf_record_atoms(qm_start_sdf)
            summary["qm_start_geometry"] = {
                "source": "sdf",
                "file": os.path.abspath(qm_start_sdf),
                "name": str(sdf_info.get("name", "") or ""),
                "atom_count": int(len(list(sdf_info.get("atoms", []) or []))),
            }
        else:
            summary["qm_start_geometry"] = {"source": "mol2", "file": os.path.abspath(args.mol2)}
        qm_dir = os.path.join(out_dir, "qm_stationary")
        qm_result_path = os.path.join(qm_dir, "psi4_result.json")
        target_requires_frequency = str(args.qm_stage) == "stationary"
        qm_engine = str(args.qm_engine or "psi4").strip().lower()
        qm_basis_label = "SEMIEMPIRICAL" if qm_engine == "mopac" else str(args.basis)
        if qm_engine == "mopac":
            stage_specs = [{"label": "high", "method": str(args.method).upper(), "basis": str(qm_basis_label)}]
        else:
            stage_specs = build_qm_optimization_stages(
                target_method=args.method,
                target_basis=args.basis,
                low_method=args.preopt_method,
                low_basis=args.preopt_basis,
                medium_method=args.midopt_method,
                medium_basis=args.midopt_basis,
                staging_mode=args.qm_opt_staging,
            )
        summary["qm_optimization_stage_plan"] = list(stage_specs)
        summary["qm_optimization_stages"] = []
        qm_result = None
        cached = None
        if os.path.isfile(qm_result_path):
            try:
                with open(qm_result_path, "r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                if qm_result_matches_stage(
                    cached,
                    method=(str(args.method).upper() if qm_engine == "mopac" else args.method),
                    basis=qm_basis_label,
                    require_frequency=target_requires_frequency,
                ):
                    qm_result = cached
            except Exception:
                cached = None
                qm_result = None
        if qm_result is None and qm_engine == "mopac":
            stage_out_dir = qm_dir
            stage_result = None
            if qm_start_sdf:
                stage_result = run_mopac_stationary_from_sdf(
                    sdf_file=qm_start_sdf,
                    out_dir=stage_out_dir,
                    method=str(args.method).upper(),
                    charge=int(template_charge),
                    multiplicity=int(template_multiplicity),
                    mopac_executable=str(args.mopac_exe or "mopac"),
                    run_frequency=target_requires_frequency,
                )
            else:
                stage_result = run_mopac_stationary_from_mol2(
                    mol2_file=args.mol2,
                    out_dir=stage_out_dir,
                    method=str(args.method).upper(),
                    charge=int(template_charge),
                    multiplicity=int(template_multiplicity),
                    mopac_executable=str(args.mopac_exe or "mopac"),
                    run_frequency=target_requires_frequency,
                )
            summary["qm_optimization_stages"].append(
                {
                    "label": "high",
                    "method": str(args.method).upper(),
                    "basis": str(qm_basis_label),
                    "out_dir": str(stage_out_dir),
                    "used_cache": False,
                    "ok": bool((stage_result or {}).get("ok", False)),
                    "operation": str((stage_result or {}).get("operation", "") or ""),
                    "error": str((stage_result or {}).get("error", "") or ""),
                }
            )
            qm_result = stage_result
        elif qm_result is None:
            prev_xyz = ""
            prev_charge = None
            prev_multiplicity = None
            for idx, spec in enumerate(stage_specs):
                is_final = idx == (len(stage_specs) - 1)
                require_frequency = bool(is_final and target_requires_frequency)
                stage_out_dir = qm_dir if is_final else os.path.join(out_dir, f"qm_stage_{int(idx) + 1:02d}_{str(spec['label'])}")
                stage_result_path = os.path.join(stage_out_dir, "psi4_result.json")
                stage_cached = None
                stage_result = None
                stage_used_cache = False
                if os.path.isfile(stage_result_path):
                    try:
                        with open(stage_result_path, "r", encoding="utf-8") as fh:
                            stage_cached = json.load(fh)
                    except Exception:
                        stage_cached = None
                stage_opt_xyz = os.path.join(stage_out_dir, "optimized.xyz")
                if isinstance(stage_cached, dict) and qm_result_matches_stage(
                    stage_cached,
                    method=spec["method"],
                    basis=spec["basis"],
                    require_frequency=require_frequency,
                ):
                    stage_result = stage_cached
                    stage_used_cache = True
                elif (
                    require_frequency
                    and isinstance(stage_cached, dict)
                    and qm_result_matches_stage(stage_cached, method=spec["method"], basis=spec["basis"], require_frequency=False)
                    and os.path.isfile(stage_opt_xyz)
                ):
                    stage_result = run_psi4_frequency_from_xyz(
                        xyz_file=stage_opt_xyz,
                        out_dir=stage_out_dir,
                        method=spec["method"],
                        basis=spec["basis"],
                        charge=int(stage_cached.get("charge", 0) or 0),
                        multiplicity=int(stage_cached.get("multiplicity", 1) or 1),
                        threads=args.threads,
                        memory_gb=args.memory_gb,
                    )
                elif prev_xyz and os.path.isfile(prev_xyz):
                    stage_result = run_psi4_stationary_from_xyz(
                        xyz_file=prev_xyz,
                        out_dir=stage_out_dir,
                        method=spec["method"],
                        basis=spec["basis"],
                        charge=int(prev_charge if prev_charge is not None else 0),
                        multiplicity=int(prev_multiplicity if prev_multiplicity is not None else 1),
                        threads=args.threads,
                        memory_gb=args.memory_gb,
                        run_frequency=require_frequency,
                        geom_maxiter=(max(180, int(args.geom_maxiter) + 60) if is_final else max(140, int(args.geom_maxiter))),
                    )
                elif qm_start_sdf:
                    stage_result = run_psi4_stationary_from_sdf(
                        sdf_file=qm_start_sdf,
                        out_dir=stage_out_dir,
                        method=spec["method"],
                        basis=spec["basis"],
                        charge=int(template_charge),
                        multiplicity=int(template_multiplicity),
                        threads=args.threads,
                        memory_gb=args.memory_gb,
                        run_frequency=require_frequency,
                        geom_maxiter=(max(180, int(args.geom_maxiter) + 60) if is_final else max(140, int(args.geom_maxiter))),
                    )
                else:
                    stage_result = run_psi4_stationary_from_mol2(
                        mol2_file=args.mol2,
                        out_dir=stage_out_dir,
                        method=spec["method"],
                        basis=spec["basis"],
                        threads=args.threads,
                        memory_gb=args.memory_gb,
                        run_frequency=require_frequency,
                        geom_maxiter=(max(180, int(args.geom_maxiter) + 60) if is_final else max(140, int(args.geom_maxiter))),
                    )
                if (
                    is_final
                    and (not bool(stage_result.get("ok", False)))
                    and (not bool(args.disable_preopt_fallback))
                    and str(args.qm_opt_staging) == "off"
                    and _is_opt_convergence_failure(stage_result)
                    and ((str(spec["method"]).lower(), str(spec["basis"]).lower()) != (str(args.preopt_method).lower(), str(args.preopt_basis).lower()))
                ):
                    preopt_dir = os.path.join(out_dir, "qm_preopt")
                    if qm_start_sdf:
                        preopt = run_psi4_stationary_from_sdf(
                            sdf_file=qm_start_sdf,
                            out_dir=preopt_dir,
                            method=args.preopt_method,
                            basis=args.preopt_basis,
                            charge=int(template_charge),
                            multiplicity=int(template_multiplicity),
                            threads=args.threads,
                            memory_gb=args.memory_gb,
                            run_frequency=False,
                            geom_maxiter=max(160, int(args.geom_maxiter)),
                        )
                    else:
                        preopt = run_psi4_stationary_from_mol2(
                            mol2_file=args.mol2,
                            out_dir=preopt_dir,
                            method=args.preopt_method,
                            basis=args.preopt_basis,
                            threads=args.threads,
                            memory_gb=args.memory_gb,
                            run_frequency=False,
                            geom_maxiter=max(160, int(args.geom_maxiter)),
                        )
                    summary["preopt_fallback"] = {
                        "attempted": True,
                        "method": str(args.preopt_method),
                        "basis": str(args.preopt_basis),
                        "ok": bool(preopt.get("ok", False)),
                        "error": str(preopt.get("error", "") or ""),
                    }
                    preopt_xyz = os.path.join(preopt_dir, str(preopt.get("optimized_xyz", "optimized.xyz")))
                    if bool(preopt.get("ok", False)) and os.path.isfile(preopt_xyz):
                        stage_result = run_psi4_stationary_from_xyz(
                            xyz_file=preopt_xyz,
                            out_dir=stage_out_dir,
                            method=spec["method"],
                            basis=spec["basis"],
                            charge=int(preopt.get("charge", 0) or 0),
                            multiplicity=int(preopt.get("multiplicity", 1) or 1),
                            threads=args.threads,
                            memory_gb=args.memory_gb,
                            run_frequency=require_frequency,
                            geom_maxiter=max(180, int(args.geom_maxiter) + 60),
                        )
                summary["qm_optimization_stages"].append(
                    {
                        "label": str(spec["label"]),
                        "method": str(spec["method"]),
                        "basis": str(spec["basis"]),
                        "out_dir": str(stage_out_dir),
                        "used_cache": bool(stage_used_cache),
                        "ok": bool((stage_result or {}).get("ok", False)),
                        "operation": str((stage_result or {}).get("operation", "") or ""),
                        "error": str((stage_result or {}).get("error", "") or ""),
                    }
                )
                if not bool(stage_result.get("ok", False)):
                    qm_result = stage_result
                    break
                prev_xyz = os.path.join(stage_out_dir, str(stage_result.get("optimized_xyz", "optimized.xyz")))
                prev_charge = int(stage_result.get("charge", 0) or 0)
                prev_multiplicity = int(stage_result.get("multiplicity", 1) or 1)
                if is_final:
                    qm_result = stage_result
        if not bool(qm_result.get("ok", False)):
            raise RuntimeError(f"QM stationary job failed: {qm_result.get('error', 'unknown_error')}")
        if str(args.qm_stage) == "optimize_only":
            summary.update(
                {
                    "ok": True,
                    "qm_stage": "optimize_only",
                    "qm_result": qm_result,
                    "note": "Geometry optimization completed. Re-run with --qm-stage stationary to compute frequencies and fit parameters.",
                    "optimized_xyz": os.path.join(qm_dir, str(qm_result.get("optimized_xyz", "optimized.xyz"))),
                }
            )
            lines = [
                "# Ligand QM Optimization",
                "",
                "- Stage: `optimize_only`",
                f"- Method: `{str(qm_result.get('method', args.method) or args.method)}`",
                f"- Basis: `{str(qm_result.get('basis', qm_basis_label) or qm_basis_label)}`",
                f"- Optimized XYZ: `{summary['optimized_xyz']}`",
                f"- Charge: `{int(qm_result.get('charge', 0) or 0)}`",
                f"- Multiplicity: `{int(qm_result.get('multiplicity', 1) or 1)}`",
                "",
                "Run again with `--qm-stage stationary` to compute frequencies and fit MM parameters from the cached optimized geometry.",
            ]
            write_json(summary_path, summary)
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            print(f"Wrote summary: {summary_path}")
            print(f"Wrote report:  {report_path}")
            return
        qm_vib = psi4_load_vib_data(qm_result, qm_dir)
        qm_coords = np.asarray(qm_vib.get("eq_geom_ang", []), dtype=float)
        if qm_coords.ndim != 2 or qm_coords.shape[1] != 3:
            raise RuntimeError("QM vibrational data did not provide equilibrium geometry")
        if int(np.sum(np.abs(np.asarray(qm_vib.get("freq_imag", []), dtype=float)) > 1.0e-3)) > 0:
            raise RuntimeError("QM stationary point has imaginary frequencies")
        sys_payload = build_ligand_system_payload(
            prmtop_file=args.prmtop,
            inpcrd_file=args.inpcrd,
            mol2_file=args.mol2,
        )
        fit_config = build_scale_search_config(
            sys_payload,
            fit_mode=args.fit_mode,
            max_typed_bond_types=args.max_typed_bond_types,
            max_typed_angle_types=args.max_typed_angle_types,
            max_typed_torsion_types=args.max_typed_torsion_types,
            max_typed_lj_types=args.max_typed_lj_types,
        )
        base_metrics = evaluate_scale_vector(
            system_xml=sys_payload["system_xml"],
            qm_coords_ang=qm_coords,
            qm_vib_data=qm_vib,
            masses_da=sys_payload["masses_da"],
            charges_e=sys_payload["charges_e"],
            scales=fit_config["initial"],
            typed_plan=fit_config["typed_plan"],
            platform_name=args.platform,
        )
        if not bool(base_metrics.get("ok", False)):
            raise RuntimeError(f"Initial FF evaluation failed: {base_metrics.get('error', 'unknown_error')}")

        def objective(scale_dict):
            metrics = evaluate_scale_vector(
                system_xml=sys_payload["system_xml"],
                qm_coords_ang=qm_coords,
                qm_vib_data=qm_vib,
                masses_da=sys_payload["masses_da"],
                charges_e=sys_payload["charges_e"],
                scales=scale_dict,
                typed_plan=fit_config["typed_plan"],
                platform_name=args.platform,
            )
            return float(metrics.get("objective", np.inf))

        pso_result = run_bounded_pso(
            objective_fn=objective,
            bounds=fit_config["bounds"],
            initial=fit_config["initial"],
            swarm_size=args.swarm_size,
            n_iter=args.n_iter,
            seed=args.seed,
        )
        best_scales = dict(pso_result.get("best_scales", {}) or {})
        refined_metrics = evaluate_scale_vector(
            system_xml=sys_payload["system_xml"],
            qm_coords_ang=qm_coords,
            qm_vib_data=qm_vib,
            masses_da=sys_payload["masses_da"],
            charges_e=sys_payload["charges_e"],
            scales=best_scales,
            typed_plan=fit_config["typed_plan"],
            platform_name=args.platform,
        )
        if not bool(refined_metrics.get("ok", False)):
            raise RuntimeError(f"Refined FF evaluation failed: {refined_metrics.get('error', 'unknown_error')}")
        write_mol2_with_coords(args.mol2, refined_metrics["mm_coords_ang"], refined_mol2)
        refined_system = _deserialize_scaled_system(sys_payload["system_xml"], best_scales, typed_plan=fit_config["typed_plan"])
        write_refined_frcmod(refined_frcmod, refined_system, args.mol2)
        amber_payload = regenerate_refined_amber_files(
            mol2_file=refined_mol2,
            frcmod_file=refined_frcmod,
            out_dir=out_dir,
            tleap_bin=args.tleap_bin,
        )
        xml_payload = {}
        if bool(amber_payload.get("ok", False)):
            refined_xml = os.path.join(out_dir, "ligand_refined.xml")
            xml_payload = export_refined_forcefield_xml(amber_payload["prmtop"], refined_xml)
        stage_plan = list(summary.get("qm_optimization_stage_plan", []) or [])
        stage_history = list(summary.get("qm_optimization_stages", []) or [])
        refined_paths = {
            "refined_mol2": refined_mol2,
            "refined_frcmod": refined_frcmod,
            "refined_prmtop": str(amber_payload.get("prmtop", "") or "") if bool(amber_payload.get("ok", False)) else "",
            "refined_inpcrd": str(amber_payload.get("inpcrd", "") or "") if bool(amber_payload.get("ok", False)) else "",
            "refined_xml": os.path.join(out_dir, "ligand_refined.xml") if bool((xml_payload or {}).get("ok", False)) else "",
        }
        summary = make_summary_payload(
            qm_result=qm_result,
            qm_vib_data=qm_vib,
            base_metrics=base_metrics,
            refined_metrics=refined_metrics,
            pso_result=pso_result,
            refined_paths=refined_paths,
        )
        summary["qm_optimization_stage_plan"] = stage_plan
        summary["qm_optimization_stages"] = stage_history
        summary["fit_mode"] = str(fit_config.get("fit_mode", args.fit_mode))
        summary["selected_typed_parameters"] = dict(fit_config.get("selected_typed_parameters", {}) or {})
        summary["amber_regeneration"] = amber_payload
        summary["xml_export"] = xml_payload
        mode_rows = build_mode_comparison_rows(refined_metrics, refined_metrics, base_metrics)
        write_mode_comparison_csv(mode_csv, mode_rows)
        summary["mode_comparison_csv"] = mode_csv
        write_markdown_report(report_path, summary, mode_rows)
        summary["report_md"] = report_path
        make_summary_plots(
            out_png=plot_png,
            summary=summary,
            mode_rows=mode_rows,
            qm_ir_grid=refined_metrics.get("ir_grid_cm", []),
            qm_ir_curve=refined_metrics.get("qm_ir_broadened", []),
            base_ir_curve=base_metrics.get("mm_ir_broadened", []),
            refined_ir_curve=refined_metrics.get("mm_ir_broadened", []),
        )
        summary["plot_png"] = plot_png
        summary["ok"] = True
        write_json(summary_path, summary)
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        write_json(summary_path, summary)
        raise
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report:  {report_path}")
    print(f"Wrote plot:    {plot_png}")


if __name__ == "__main__":
    main()
