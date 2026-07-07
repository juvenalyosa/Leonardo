#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from openmm import (
    Context,
    CustomExternalForce,
    CustomNonbondedForce,
    LangevinIntegrator,
    LangevinMiddleIntegrator,
    LocalEnergyMinimizer,
    Platform,
    System,
)
from openmm import app
from openmm import unit as u
from openmm.app import DCDReporter, Modeller, PDBFile

from abfe_runtime import build_abfe_adaptive_steering_state_payload
from abfe_runtime import build_abfe_surface_detach_direction_payload
from abfe_runtime import build_abfe_adaptive_steering_update_decision
from abfe_runtime import build_abfe_steering_lambda_schedule
from abfe_runtime import build_abfe_steering_policy_state_payload
from abfe_runtime import build_abfe_steering_policy_observation_payload
from abfe_runtime import build_abfe_steering_bandit_action_payload
from abfe_runtime import build_abfe_steering_policy_safe_action_payload
from abfe_runtime import build_abfe_steering_policy_trace_payload
from abfe_runtime import build_abfe_steering_policy_summary_payload
from abfe_runtime import build_abfe_post_sampling_metrics
from abfe_runtime import build_abfe_free_energy_numeric_payload
from abfe_runtime import build_abfe_replica_aggregate_payload
from abfe_runtime import compute_runtime_pocket_descriptors
from validate_3ptb_ben_parameterization import ION_RESN
from test_abfe_3ptb_ben.run_abfe_fullff_3ptb import (
    estimate_protein_charge,
    infer_element,
    inject_ligand_terms_from_prmtop,
    load_mol2_charges,
    mass_for_element,
    nb_params,
)


ION_CHARGES = {
    "NA": 1.0,
    "K": 1.0,
    "CA": 2.0,
    "MG": 2.0,
    "ZN": 2.0,
    "MN": 2.0,
    "FE": 2.0,
    "CU": 2.0,
    "NI": 2.0,
    "CO": 2.0,
    "CL": -1.0,
}
BEN_N_ATOMS = {"N1", "N2"}
ASP189_O_ATOMS = {"OD1", "OD2"}
KNOWN_POCKET = {
    ("ASP", "A", "189"),
    ("SER", "A", "190"),
    ("GLN", "A", "192"),
    ("SER", "A", "195"),
    ("TRP", "A", "215"),
    ("GLY", "A", "216"),
    ("SER", "A", "217"),
    ("GLY", "A", "219"),
    ("CYS", "A", "220"),
    ("GLY", "A", "226"),
}


def run_checked(cmd, cwd: Path) -> None:
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


def parameterize_from_raw_3ptb(repo_root: Path, pdb_path: Path, workdir: Path, min_iters: int) -> dict:
    param_dir = workdir / "parameterization"
    param_dir.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            sys.executable,
            str(repo_root / "validate_3ptb_ben_parameterization.py"),
            "--pdb",
            str(pdb_path),
            "--workdir",
            str(param_dir),
            "--min-iters",
            str(int(min_iters)),
        ],
        cwd=repo_root,
    )
    summary_path = param_dir / "validation_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def extract_ion_block(pdb_path: Path) -> str:
    lines = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            resn = line[17:20].strip().upper()
            el = line[76:78].strip().upper()
            if resn in ION_RESN or el in ION_CHARGES:
                lines.append(line.rstrip("\n"))
    if not lines:
        return ""
    return "\n".join(lines) + "\nEND\n"


def build_complex_with_ions(prepared_pdb: Path, raw_pdb: Path, out_pdb: Path) -> dict:
    pdb = PDBFile(str(prepared_pdb))
    modeller = Modeller(pdb.topology, pdb.positions)
    ion_block = extract_ion_block(raw_pdb)
    ion_records = []
    if ion_block:
        ion_pdb = PDBFile(io.StringIO(ion_block))
        modeller.add(ion_pdb.topology, ion_pdb.positions)
        for residue in ion_pdb.topology.residues():
            ion_records.append(
                {
                    "resn": str(residue.name).strip().upper(),
                    "chain": str(residue.chain.id).strip(),
                    "resi": str(residue.id).strip(),
                    "atoms": int(sum(1 for _ in residue.atoms())),
                }
            )
    with out_pdb.open("w", encoding="utf-8") as fh:
        PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)
    return {
        "complex_pdb": str(out_pdb),
        "ions": ion_records,
    }


def load_mol2_atom_names(mol2_file: Path) -> list[str]:
    out = []
    section = ""
    for raw in mol2_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@<TRIPOS>"):
            section = line.upper()
            continue
        if section == "@<TRIPOS>ATOM":
            parts = line.split()
            if len(parts) >= 2:
                out.append(str(parts[1]).strip().upper())
    return out


def ion_charge(resn: str, atom_name: str, element: str) -> float:
    _ = atom_name
    key = str(resn or element or "").strip().upper()
    if key in ION_CHARGES:
        return float(ION_CHARGES[key])
    e = str(element or "").strip().upper()
    return float(ION_CHARGES.get(e, 0.0))


def build_context_with_platform_fallback(system: System, integrator, platform_override: str = ""):
    """Create an OpenMM Context using the fastest safe platform available.

    The ABFE-lite validation run is intentionally small enough for a local machine, but
    a 4 ns guided-exit trajectory is still expensive on CPU.  We try GPU-like
    platforms first and fall back to CPU/Reference if a platform is unavailable
    or rejects this system/force combination.
    """
    requested = str(platform_override or os.environ.get("LEONARDO_OPENMM_PLATFORM", "") or "").strip()
    ordered = []
    if requested:
        ordered.append(requested)
    ordered.extend(["CUDA", "OpenCL", "Metal", "CPU", "Reference"])

    seen = set()
    last_error = None
    for name in ordered:
        if str(name) in seen:
            continue
        seen.add(str(name))
        try:
            platform = Platform.getPlatformByName(name)
        except Exception as exc:
            last_error = exc
            continue
        props = {}
        if str(name) in {"CUDA", "OpenCL"}:
            props = {"Precision": "mixed"}
        try:
            if props:
                return Context(system, integrator, platform, props), platform, str(name), props
            return Context(system, integrator, platform), platform, str(name), props
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"No compatible OpenMM platform available: {last_error}")


def build_context_cpu_only(system: System, integrator):
    """Compatibility wrapper for older benchmark helpers that expect CPU only."""
    ctx, platform, platform_name, _props = build_context_with_platform_fallback(
        system,
        integrator,
        platform_override="CPU",
    )
    return ctx, platform, platform_name


def rank_pull_directions(
    coords_nm,
    protein_ids,
    ligand_ids,
    protein_heavy_ids=None,
    ligand_heavy_ids=None,
):
    prot_idx = np.asarray(protein_heavy_ids if protein_heavy_ids else protein_ids, dtype=int)
    lig_idx = np.asarray(ligand_heavy_ids if ligand_heavy_ids else ligand_ids, dtype=int)
    if prot_idx.size == 0:
        prot_idx = np.asarray(protein_ids, dtype=int)
    if lig_idx.size == 0:
        lig_idx = np.asarray(ligand_ids, dtype=int)
    if prot_idx.size == 0 or lig_idx.size == 0:
        return [{
            "direction": np.array([1.0, 0.0, 0.0], dtype=float),
            "clearance_nm": np.nan,
            "collision_penalty": np.nan,
            "alignment": np.nan,
            "score": np.nan,
        }]

    xyz = np.asarray(coords_nm, dtype=float)
    lig_com = np.mean(xyz[lig_idx], axis=0)
    prot_com = np.mean(xyz[prot_idx], axis=0)
    away = np.asarray(lig_com - prot_com, dtype=float)
    away_norm = float(np.linalg.norm(away))
    if away_norm < 1.0e-9:
        away = np.array([1.0, 0.0, 0.0], dtype=float)
        away_norm = 1.0
    away_u = away / away_norm
    rel = np.asarray(xyz[prot_idx] - lig_com[None, :], dtype=float)
    if rel.size == 0:
        return [{
            "direction": np.asarray(away_u, dtype=float),
            "clearance_nm": np.nan,
            "collision_penalty": 0.0,
            "alignment": 1.0,
            "score": np.nan,
        }]

    dirs = []

    def _add_dir(vec):
        vv = np.asarray(vec, dtype=float)
        nn = float(np.linalg.norm(vv))
        if nn < 1.0e-8:
            return
        uu = vv / nn
        for d0 in dirs:
            if abs(float(np.dot(uu, d0))) > 0.985:
                return
        dirs.append(uu)

    _add_dir(away_u)
    for v in (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ):
        _add_dir(v)

    drel = np.linalg.norm(rel, axis=1)
    nearest = np.argsort(drel)[: min(24, int(drel.size))]
    for idx in nearest:
        _add_dir(-rel[int(idx)])

    n_sphere = 24
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    for k in range(n_sphere):
        z = 1.0 - (2.0 * (k + 0.5) / n_sphere)
        r = max(0.0, 1.0 - z * z) ** 0.5
        phi = 2.0 * math.pi * ((k / golden) % 1.0)
        _add_dir((r * math.cos(phi), r * math.sin(phi), z))

    if not dirs:
        return [{
            "direction": np.asarray(away_u, dtype=float),
            "clearance_nm": np.nan,
            "collision_penalty": np.nan,
            "alignment": 1.0,
            "score": np.nan,
        }]

    vdw_axis_nm = 0.22
    close_axis_nm = 0.30
    ranked = []
    for d in dirs:
        fwd = np.dot(rel, d)
        ahead = fwd > 0.05
        if not np.any(ahead):
            clearance_nm = 2.5
            penalty = 0
        else:
            ra = rel[ahead]
            fa = fwd[ahead]
            lat2 = np.sum(ra * ra, axis=1) - np.square(fa)
            lat2 = np.maximum(lat2, 0.0)
            collide = lat2 <= (vdw_axis_nm ** 2)
            near_axis = np.logical_and(fa < 0.60, lat2 <= (close_axis_nm ** 2))
            penalty = int(np.sum(near_axis))
            if np.any(collide):
                clearance_nm = float(np.min(fa[collide]))
            else:
                clearance_nm = float(np.percentile(fa, 20.0)) if fa.size else 2.5
            clearance_nm = float(np.clip(clearance_nm, 0.05, 2.5))
        align = float(np.dot(d, away_u))
        score = (3.0 * clearance_nm) + (0.8 * align) - (0.08 * float(penalty))
        ranked.append(
            {
                "direction": np.asarray(d, dtype=float),
                "clearance_nm": float(clearance_nm),
                "collision_penalty": float(penalty),
                "alignment": float(align),
                "score": float(score),
            }
        )
    ranked = sorted(
        list(ranked or []),
        key=lambda row: (
            -float(row.get("score", -1.0e18) if np.isfinite(float(row.get("score", np.nan))) else -1.0e18),
            -float(row.get("clearance_nm", -1.0) if np.isfinite(float(row.get("clearance_nm", np.nan))) else -1.0),
            float(row.get("collision_penalty", np.inf) if np.isfinite(float(row.get("collision_penalty", np.nan))) else np.inf),
            -float(row.get("alignment", -1.0) if np.isfinite(float(row.get("alignment", np.nan))) else -1.0),
        ),
    )
    return ranked or [{
        "direction": np.asarray(away_u, dtype=float),
        "clearance_nm": np.nan,
        "collision_penalty": np.nan,
        "alignment": 1.0,
        "score": np.nan,
    }]


def pair_energy_components(
    receptor_idx,
    ligand_idx,
    xyz_nm,
    sigmas,
    epsilons,
    charges,
    cutoff_nm=1.2,
):
    e_lj = 0.0
    e_c = 0.0
    cutoff = float(cutoff_nm)
    sigma_avg = 0.0
    for i in list(receptor_idx):
        ii = int(i)
        for j in list(ligand_idx):
            jj = int(j)
            dv = np.asarray(xyz_nm[ii], dtype=float) - np.asarray(xyz_nm[jj], dtype=float)
            r = float(np.linalg.norm(dv))
            if (not np.isfinite(r)) or r <= 1.0e-8 or r > cutoff:
                continue
            sigma_avg = 0.5 * (float(sigmas[ii]) + float(sigmas[jj]))
            eps_mix = math.sqrt(max(0.0, float(epsilons[ii]) * float(epsilons[jj])))
            sr = sigma_avg / r
            e_lj += 4.0 * eps_mix * ((sr ** 12) - (sr ** 6))
            e_c += 138.935456 * float(charges[ii]) * float(charges[jj]) / r
    return float(e_lj), float(e_c)


def contact_occupancies(
    atoms,
    xyz_series_nm,
    ligand_heavy_idx,
    bound_mask,
    cutoff_a=4.0,
):
    counts = defaultdict(int)
    total = 0
    ligand_heavy = np.asarray(list(ligand_heavy_idx), dtype=int)
    cutoff_nm = float(cutoff_a) * 0.1
    for frame_idx, is_bound in enumerate(list(bound_mask)):
        if not bool(is_bound):
            continue
        total += 1
        xyz = np.asarray(xyz_series_nm[int(frame_idx)], dtype=float)
        seen = set()
        for atom in atoms:
            if atom["origin"] != "protein":
                continue
            idx = int(atom["index"])
            if atom["element"] == "H":
                continue
            dmin = float(np.min(np.linalg.norm(xyz[idx] - xyz[ligand_heavy], axis=1)))
            if dmin <= cutoff_nm:
                key = (atom["resn"], atom["chain"], atom["resi"])
                if key not in seen:
                    counts[key] += 1
                    seen.add(key)
    rows = []
    denom = float(max(1, total))
    for key, value in counts.items():
        rows.append(
            {
                "resn": str(key[0]),
                "chain": str(key[1]),
                "resi": str(key[2]),
                "bound_contact_occupancy": float(value / denom),
                "bound_contact_frames": int(value),
            }
        )
    rows.sort(key=lambda row: row["bound_contact_occupancy"], reverse=True)
    return rows, int(total)


def asp189_amidinium_occupancy(atoms, xyz_series_nm, bound_mask, cutoff_a=3.5):
    ben_n = [int(a["index"]) for a in atoms if a["resn"] == "BEN" and a["name"].upper() in BEN_N_ATOMS]
    asp_o = [
        int(a["index"])
        for a in atoms
        if a["resn"] == "ASP" and a["chain"] == "A" and a["resi"] == "189" and a["name"].upper() in ASP189_O_ATOMS
    ]
    total = 0
    hits = 0
    cutoff_nm = float(cutoff_a) * 0.1
    if not ben_n or not asp_o:
        return 0.0, 0, 0
    for frame_idx, is_bound in enumerate(list(bound_mask)):
        if not bool(is_bound):
            continue
        total += 1
        xyz = np.asarray(xyz_series_nm[int(frame_idx)], dtype=float)
        best = float("inf")
        for ii in ben_n:
            for jj in asp_o:
                best = min(best, float(np.linalg.norm(xyz[ii] - xyz[jj])))
        if best <= cutoff_nm:
            hits += 1
    occ = float(hits / float(max(1, total)))
    return occ, int(hits), int(total)


def summarize_checks(
    parameterization_summary,
    ion_info,
    arr_total,
    arr_lj,
    arr_coul,
    arr_metal_total,
    arr_rc_min,
    arr_rc_com,
    bound_mask,
    unbound_mask,
    asp189_occ,
    contact_rows,
):
    arr_rc_min = np.asarray(arr_rc_min, dtype=float)
    arr_rc_com = np.asarray(arr_rc_com, dtype=float)
    bound_mask = np.asarray(bound_mask, dtype=bool)
    unbound_mask = np.asarray(unbound_mask, dtype=bool)

    def _mean_if(mask, arr):
        sel = np.asarray(arr, dtype=float)[np.asarray(mask, dtype=bool)]
        if sel.size <= 0:
            return float("nan")
        return float(np.mean(sel))

    mean_bound_total = _mean_if(bound_mask, arr_total)
    mean_unbound_total = _mean_if(unbound_mask, arr_total)
    mean_bound_coul = _mean_if(bound_mask, arr_coul)
    mean_bound_metal = _mean_if(bound_mask, arr_metal_total)
    delta_bound_unbound = mean_bound_total - mean_unbound_total

    top5 = [
        (str(row["resn"]), str(row["chain"]), str(row["resi"]))
        for row in list(contact_rows or [])[:5]
    ]
    known_hits = [row for row in list(contact_rows or [])[:8] if (row["resn"], row["chain"], row["resi"]) in KNOWN_POCKET]

    charge_sum = float(
        (((parameterization_summary.get("ffxml_audit", {}) or {}).get("charge", {}) or {}).get("sum_e", 0.0))
    )
    ring_spread = float(((parameterization_summary.get("ring_after", {}) or {}).get("spread_ang", float("nan"))))
    metal_distance = float(((ion_info or {}).get("min_metal_ligand_distance_ang", float("nan"))))

    physics = {
        "parameterized_charge_close_to_plus_one": bool(abs(charge_sum - 1.0) <= 0.1),
        "ligand_ring_geometry_preserved": bool(np.isfinite(ring_spread) and ring_spread <= 0.05),
        "steering_reaches_unbound_region": bool(np.sum(unbound_mask) > 0),
        "bound_state_more_favorable_than_unbound": bool(np.isfinite(delta_bound_unbound) and delta_bound_unbound <= -25.0),
        "bound_state_coulomb_is_attractive": bool(np.isfinite(mean_bound_coul) and mean_bound_coul < 0.0),
        "metal_does_not_dominate_binding_energy": bool(
            np.isfinite(mean_bound_metal)
            and np.isfinite(mean_bound_total)
            and abs(mean_bound_metal) <= 5.0
            and abs(mean_bound_metal) <= 0.10 * max(1.0, abs(mean_bound_total))
            and np.isfinite(metal_distance)
            and metal_distance >= 10.0
        ),
    }
    biology = {
        "asp189_amidinium_contact_persistent": bool(float(asp189_occ) >= 0.5),
        "asp189_ranked_among_top_contacts": bool(("ASP", "A", "189") in top5),
        "known_s1_pocket_residues_enriched": bool(len(known_hits) >= 3),
    }
    return {
        "bound_mask_count": int(np.sum(bound_mask)),
        "unbound_mask_count": int(np.sum(unbound_mask)),
        "mean_bound_total_kj_mol": float(mean_bound_total),
        "mean_unbound_total_kj_mol": float(mean_unbound_total),
        "delta_bound_minus_unbound_kj_mol": float(delta_bound_unbound),
        "mean_bound_coul_kj_mol": float(mean_bound_coul),
        "mean_bound_metal_total_kj_mol": float(mean_bound_metal),
        "physics_checks": physics,
        "biology_checks": biology,
        "overall_pass": bool(all(physics.values()) and all(biology.values())),
    }


def classify_population_masks(
    arr_site_min_a,
    arr_site_com_a,
    protein_contact_count_4a,
    far_hold_stable_mask,
    bound_site_com_threshold_a=4.0,
    bound_max_min_dist_a=3.5,
    bound_min_contact_count_4a=8,
    unbound_site_com_threshold_a=10.0,
    unbound_min_dist_a=3.5,
    unbound_max_contact_count_4a=4,
):
    site_min = np.asarray(arr_site_min_a, dtype=float)
    site_com = np.asarray(arr_site_com_a, dtype=float)
    contact_arr = np.asarray(protein_contact_count_4a if protein_contact_count_4a is not None else [], dtype=float)
    far_hold = np.asarray(far_hold_stable_mask if far_hold_stable_mask is not None else [], dtype=bool)
    base = np.logical_and(np.isfinite(site_min), np.isfinite(site_com))
    bound_mask = np.logical_and.reduce(
        [
            base,
            np.asarray(site_com <= float(bound_site_com_threshold_a), dtype=bool),
            np.asarray(site_min <= float(bound_max_min_dist_a), dtype=bool),
        ]
    )
    if contact_arr.size == site_min.size and np.any(np.isfinite(contact_arr)):
        bound_mask = np.logical_and(
            bound_mask,
            np.asarray(contact_arr >= float(bound_min_contact_count_4a), dtype=bool),
        )
    unbound_mask = np.logical_and.reduce(
        [
            base,
            np.asarray(site_com >= float(unbound_site_com_threshold_a), dtype=bool),
            np.asarray(site_min >= float(unbound_min_dist_a), dtype=bool),
        ]
    )
    if contact_arr.size == site_min.size and np.any(np.isfinite(contact_arr)):
        unbound_mask = np.logical_and(
            unbound_mask,
            np.asarray(contact_arr <= float(unbound_max_contact_count_4a), dtype=bool),
        )
    if far_hold.size == site_min.size:
        unbound_mask = np.logical_and(unbound_mask, far_hold)
    intermediate_mask = np.logical_and(base, np.logical_not(np.logical_or(bound_mask, unbound_mask)))
    return np.asarray(bound_mask, dtype=bool), np.asarray(unbound_mask, dtype=bool), np.asarray(intermediate_mask, dtype=bool)


def population_free_energy_metrics(
    *,
    frames,
    times_ps,
    lam_hist,
    bias_energy_kj,
    lj_energy_kj,
    total_energy_kj,
    coul_energy_kj,
    protein_min_dist_ang,
    rc_site_min_ang,
    rc_site_com_ang,
    protein_contact_count_4a,
    far_hold_stable_mask,
    detached_settle_mask=None,
    ligand_com_nm,
    temperature_k,
    bootstrap_seed=20260312,
    hold_config=None,
    rescored_total_valid=None,
    rescored_energy_label="",
):
    frames_i = [] if frames is None else list(frames)
    times_i = [] if times_ps is None else list(times_ps)
    lam_i = [] if lam_hist is None else list(lam_hist)
    bias_i = [] if bias_energy_kj is None else list(bias_energy_kj)
    lj_i = [] if lj_energy_kj is None else list(lj_energy_kj)
    total_i = [] if total_energy_kj is None else list(total_energy_kj)
    coul_i = [] if coul_energy_kj is None else list(coul_energy_kj)
    protein_min_i = [] if protein_min_dist_ang is None else list(protein_min_dist_ang)
    rc_site_min_i = [] if rc_site_min_ang is None else list(rc_site_min_ang)
    rc_site_com_i = [] if rc_site_com_ang is None else list(rc_site_com_ang)
    contact_i = [] if protein_contact_count_4a is None else list(protein_contact_count_4a)
    far_hold_i = [] if far_hold_stable_mask is None else list(far_hold_stable_mask)
    detached_i = [] if detached_settle_mask is None else list(detached_settle_mask)
    lig_com_i = [] if ligand_com_nm is None else list(ligand_com_nm)
    hold_cfg = dict(hold_config or {})
    post = build_abfe_post_sampling_metrics(
        frames=frames_i,
        times_ps=times_i,
        lam_hist=lam_i,
        bias_e=bias_i,
        lj_e=lj_i,
        coul_e=coul_i,
        total_e=total_i,
        min_dist=protein_min_i,
        reactive_dist=[],
        site_min_dist=rc_site_min_i,
        site_com_dist=rc_site_com_i,
        protein_contact_count_4a=contact_i,
        lig_com=lig_com_i,
        far_hold_stable_mask=far_hold_i,
        detached_settle_mask=detached_i,
        detached_settle_max_lambda=float(hold_cfg.get("detached_settle_max_lambda", 0.25) or 0.25),
        bound_hold_min_run_samples=int(hold_cfg.get("bound_hold_min_run_samples", 4) or 4),
        unbound_hold_min_run_samples=int(hold_cfg.get("unbound_hold_min_run_samples", 4) or 4),
        bound_hold_max_gap_samples=int(hold_cfg.get("bound_hold_max_gap_samples", 0) or 0),
        unbound_hold_max_gap_samples=int(hold_cfg.get("unbound_hold_max_gap_samples", 3) or 3),
        temperature_k=float(temperature_k),
        local_cutoff_nm=1.0,
    )
    numeric = build_abfe_free_energy_numeric_payload(
        frames_valid=np.asarray(post.get("frames_valid", []), dtype=int),
        bias_valid=np.asarray(post.get("bias_valid", []), dtype=float),
        total_valid=np.asarray(post.get("total_valid", []), dtype=float),
        rescored_total_valid=np.asarray(rescored_total_valid if rescored_total_valid is not None else [], dtype=float),
        rescored_energy_label=str(rescored_energy_label or ""),
        distance_valid=np.asarray(post.get("distance_valid", []), dtype=float),
        coul_valid=np.asarray(post.get("coul_valid", []), dtype=float),
        bound_valid_mask=np.asarray(post.get("bound_valid_mask", []), dtype=bool),
        unbound_valid_mask=np.asarray(post.get("unbound_valid_mask", []), dtype=bool),
        bound_cutoff_a=float(post.get("bound_cutoff_a", np.nan)),
        n_bound=int(post.get("n_bound", 0) or 0),
        n_dist_valid=int(post.get("n_dist_valid", 0) or 0),
        n_bound_raw=int(post.get("n_bound_raw", post.get("n_bound", 0)) or 0),
        n_unbound_raw=int(post.get("n_unbound_raw", post.get("n_unbound", 0)) or 0),
        n_bound_hold=int(post.get("n_bound_hold", post.get("n_bound", 0)) or 0),
        n_unbound_hold=int(post.get("n_unbound_hold", post.get("n_unbound", 0)) or 0),
        estimator_window_mode=str(post.get("estimator_window_mode", "state_fallback") or "state_fallback"),
        use_count_fraction=bool(int(post.get("n_dist_valid", 0) or 0) > 0),
        temperature_k=float(temperature_k),
        v_eff_nm3=float(post.get("v_eff_nm3", np.nan)),
        bootstrap_seed=int(bootstrap_seed),
        n_boot=200,
    )
    state_diag = dict(post.get("state_coverage_diagnostic", {}) or {})
    summary = {
        "n_valid": int(post.get("n_dist_valid", 0) or 0),
        "n_bound": int(post.get("n_bound", 0) or 0),
        "n_unbound": int(post.get("n_unbound", 0) or 0),
        "n_bound_raw": int(post.get("n_bound_raw", post.get("n_bound", 0)) or 0),
        "n_unbound_raw": int(post.get("n_unbound_raw", post.get("n_unbound", 0)) or 0),
        "n_bound_hold": int(post.get("n_bound_hold", post.get("n_bound", 0)) or 0),
        "n_unbound_hold": int(post.get("n_unbound_hold", post.get("n_unbound", 0)) or 0),
        "estimator_window_mode": str(post.get("estimator_window_mode", "state_fallback") or "state_fallback"),
        "p_bound": float(post.get("p_bound", np.nan)),
        "p_unbound": float(post.get("p_unbound", np.nan)),
        "dg_pop_kj_mol": float(post.get("dg_pop_kj", np.nan)),
        "p_bound_jeff": float(post.get("p_bound_jeff", np.nan)),
        "p_unbound_jeff": float(post.get("p_unbound_jeff", np.nan)),
        "dg_pop_jeff_kj_mol": float(post.get("dg_pop_jeff_kj", np.nan)),
        "v_eff_nm3": float(post.get("v_eff_nm3", np.nan)),
        "kd_app_m": float(post.get("kd_app_m", np.nan)),
        "p_kd_app": float(post.get("p_kd_app", np.nan)),
        "kd_app_jeff_m": float(post.get("kd_app_jeff_m", np.nan)),
        "p_kd_jeff": float(post.get("p_kd_jeff", np.nan)),
        "p_bound_reweighted": float(numeric.get("p_bound_reweighted", np.nan)),
        "p_unbound_reweighted": float(numeric.get("p_unbound_reweighted", np.nan)),
        "dg_reweighted_kj_mol": float(numeric.get("dg_reweighted_kj", np.nan)),
        "dg_reweighted_state_kj_mol": float(numeric.get("dg_reweighted_state_kj", numeric.get("dg_reweighted_kj", np.nan))),
        "dg_reweighted_standard_kj_mol": float(numeric.get("dg_reweighted_standard_kj", np.nan)),
        "standard_state_correction_kj_mol": float(numeric.get("standard_state_correction_kj", np.nan)),
        "c_eff_m": float(numeric.get("c_eff_m", post.get("c_eff_m", np.nan))),
        "kd_app_reweighted_m": float(numeric.get("kd_app_reweighted_m", np.nan)),
        "kd_standard_reweighted_m": float(numeric.get("kd_standard_reweighted_m", numeric.get("kd_app_reweighted_m", np.nan))),
        "p_kd_reweighted": float(numeric.get("p_kd_reweighted", np.nan)),
        "reweight_ci_low_kj_mol": float(numeric.get("reweight_ci_low", np.nan)),
        "reweight_ci_high_kj_mol": float(numeric.get("reweight_ci_high", np.nan)),
        "reweight_ess_total": float(numeric.get("reweight_ess_total", np.nan)),
        "reweight_ess_bound": float(numeric.get("reweight_ess_bound", np.nan)),
        "reweight_ess_unbound": float(numeric.get("reweight_ess_unbound", np.nan)),
        "reweight_clip_percentile_used": float(numeric.get("reweight_clip_percentile_used", np.nan)),
        "reweight_confidence_quality": str(numeric.get("reweight_confidence_quality", "low")),
        "reweight_confidence_note": str(numeric.get("reweight_confidence_note", "")),
        "reweight_confidence_ok": bool(numeric.get("reweight_confidence_ok", False)),
        "dg_selected_kj_mol": float(numeric.get("dg_selected_kj", np.nan)),
        "dg_selected_state_kj_mol": float(numeric.get("dg_selected_state_kj", np.nan)),
        "dg_selected_standard_kj_mol": float(numeric.get("dg_selected_standard_kj", numeric.get("dg_selected_kj", np.nan))),
        "kd_selected_m": float(numeric.get("kd_selected_m", np.nan)),
        "kd_selected_standard_m": float(numeric.get("kd_selected_standard_m", numeric.get("kd_selected_m", np.nan))),
        "p_kd_selected": float(numeric.get("p_kd_selected", np.nan)),
        "selected_estimator": str(numeric.get("selected_estimator", "gated_off")),
        "estimate_kind": str(numeric.get("estimate_kind", "")),
        "production_reliability": str(numeric.get("production_reliability", "")),
        "production_note": str(numeric.get("production_note", "")),
        "binding_estimate_allowed": bool(post.get("binding_estimate_allowed", False)),
        "binding_estimate_status": str(post.get("binding_estimate_status", state_diag.get("status", "")) or ""),
        "binding_estimate_note": str(post.get("binding_estimate_note", state_diag.get("note", "")) or ""),
        "state_coverage_diagnostic": dict(state_diag),
        "max_exit_coordinate_a": float(state_diag.get("max_exit_coordinate_a", np.nan)),
        "max_min_protein_ligand_distance_a": float(state_diag.get("max_min_protein_ligand_distance_a", np.nan)),
        "min_contact_count_4a": float(state_diag.get("min_contact_count_4a", np.nan)),
        "contact_count_at_max_exit_4a": float(state_diag.get("contact_count_at_max_exit_4a", np.nan)),
        "min_dist_at_max_exit_a": float(state_diag.get("min_dist_at_max_exit_a", np.nan)),
        "far_hold_frames": int(state_diag.get("far_hold_frames", 0) or 0),
        "rescored_energy_label": str(numeric.get("rescored_energy_label", "")),
        "dg_rescored_state_expavg_kj_mol": float(numeric.get("dg_rescored_state_expavg_kj", np.nan)),
        "dg_rescored_state_mean_kj_mol": float(numeric.get("dg_rescored_state_mean_kj", np.nan)),
    }
    return {
        "summary": summary,
        "post_payload": dict(post or {}),
        "numeric_payload": dict(numeric or {}),
    }


def write_markdown_report(out_path: Path, summary: dict) -> None:
    checks = summary["assessment"]
    physics = checks["physics_checks"]
    biology = checks["biology_checks"]
    top_contacts = summary["top_bound_contacts"][:8]
    free_energy = summary.get("free_energy_proxy", {}) or {}
    state_diag = dict(free_energy.get("state_coverage_diagnostic", {}) or {})

    def _fmt_or_na(value, fmt):
        try:
            vv = float(value)
        except Exception:
            return "not estimable"
        return fmt.format(vv) if np.isfinite(vv) else "not estimable"

    def _fmt_unit_or_na(value, fmt, unit):
        txt = _fmt_or_na(value, fmt)
        return txt if txt == "not estimable" else f"{txt} {unit}"

    lines = [
        "# 3PTB/BEN Metal-Preserving ABFE Check",
        "",
        f"Overall pass: {'yes' if checks['overall_pass'] else 'no'}",
        "",
        "## System",
        "",
        f"- Raw PDB: `{summary['inputs']['raw_pdb']}`",
        f"- Complex PDB used for ABFE: `{summary['files']['complex_with_ions_pdb']}`",
        f"- Metal ions retained: `{summary['system']['ion_residues']}`",
        f"- Min metal-ligand distance: `{summary['system']['min_metal_ligand_distance_ang']:.3f} A`",
        "",
        "## Energetics",
        "",
        f"- Bound mean total interaction: `{checks['mean_bound_total_kj_mol']:.3f} kJ/mol`",
        f"- Unbound mean total interaction: `{checks['mean_unbound_total_kj_mol']:.3f} kJ/mol`",
        f"- Bound minus unbound: `{checks['delta_bound_minus_unbound_kj_mol']:.3f} kJ/mol`",
        f"- Bound mean Coulomb term: `{checks['mean_bound_coul_kj_mol']:.3f} kJ/mol`",
        f"- Bound mean metal contribution: `{checks['mean_bound_metal_total_kj_mol']:.3f} kJ/mol`",
        f"- Estimator window mode: `{str(free_energy.get('estimator_window_mode', 'state_fallback'))}`",
        f"- Raw state counts: `bound={int(free_energy.get('n_bound_raw', 0))}`, `unbound={int(free_energy.get('n_unbound_raw', 0))}`",
        f"- Hold-window counts: `bound={int(free_energy.get('n_bound_hold', 0))}`, `unbound={int(free_energy.get('n_unbound_hold', 0))}`",
        f"- Binding estimate status: `{str(free_energy.get('binding_estimate_status', 'unknown'))}`",
        f"- Binding estimate note: `{str(free_energy.get('binding_estimate_note', ''))}`",
        f"- Max exit coordinate: `{_fmt_or_na(state_diag.get('max_exit_coordinate_a', np.nan), '{:.3f}')} A`",
        f"- Min distance at max exit: `{_fmt_or_na(state_diag.get('min_dist_at_max_exit_a', np.nan), '{:.3f}')} A`",
        f"- Contact count at max exit: `{_fmt_or_na(state_diag.get('contact_count_at_max_exit_4a', np.nan), '{:.1f}')}`",
        f"- Far-hold frames: `{int(state_diag.get('far_hold_frames', free_energy.get('far_hold_frames', 0)) or 0)}`",
        f"- Selected estimator: `{str(free_energy.get('selected_estimator', 'gated_off'))}` | confidence=`{str(free_energy.get('reweight_confidence_quality', 'low'))}`",
        f"- Selected note: `{str(free_energy.get('reweight_confidence_note', ''))}`",
        f"- Estimate kind: `{str(free_energy.get('estimate_kind', ''))}` | reliability=`{str(free_energy.get('production_reliability', ''))}`",
        f"- Reliability note: `{str(free_energy.get('production_note', ''))}`",
        f"- State-ratio ΔG: `{_fmt_unit_or_na(free_energy.get('dg_reweighted_state_kj_mol', free_energy.get('dg_reweighted_kj_mol', np.nan)), '{:.3f}', 'kJ/mol')}`",
        f"- Standard-state correction: `{_fmt_unit_or_na(free_energy.get('standard_state_correction_kj_mol', np.nan), '{:.3f}', 'kJ/mol')}` "
        f"(Ceff=`{_fmt_unit_or_na(free_energy.get('c_eff_m', np.nan), '{:.3e}', 'M')}`)",
        f"- Selected ΔG(app, standard): `{_fmt_unit_or_na(free_energy.get('dg_selected_standard_kj_mol', free_energy.get('dg_selected_kj_mol', np.nan)), '{:.3f}', 'kJ/mol')}`",
        f"- Selected Kd(app): `{_fmt_unit_or_na(free_energy.get('kd_selected_standard_m', free_energy.get('kd_selected_m', np.nan)), '{:.3e}', 'M')}`",
        f"- Bias-reweighted ΔG(state): `{_fmt_unit_or_na(free_energy.get('dg_reweighted_state_kj_mol', free_energy.get('dg_reweighted_kj_mol', np.nan)), '{:.3f}', 'kJ/mol')}`",
        f"- Bias-reweighted ΔG(standard): `{_fmt_unit_or_na(free_energy.get('dg_reweighted_standard_kj_mol', free_energy.get('dg_selected_standard_kj_mol', np.nan)), '{:.3f}', 'kJ/mol')}`",
        f"- Bias-reweighted Kd(app): `{_fmt_unit_or_na(free_energy.get('kd_standard_reweighted_m', free_energy.get('kd_app_reweighted_m', np.nan)), '{:.3e}', 'M')}`",
        f"- Bias-reweighted ESS(total): `{float(free_energy.get('reweight_ess_total', np.nan)):.2f}`",
        f"- Population diagnostic ΔG: `{_fmt_unit_or_na(free_energy.get('dg_pop_kj_mol', np.nan), '{:.3f}', 'kJ/mol')}`",
        f"- Jeffreys-smoothed diagnostic: `{_fmt_unit_or_na(free_energy.get('dg_pop_jeff_kj_mol', np.nan), '{:.3f}', 'kJ/mol')}` "
        "(pseudo-count only; not reported as BFE/Kd when one state is missing)",
        "",
        "## Biology",
        "",
        f"- Asp189 amidinium occupancy: `{summary['biology']['asp189_amidinium_contact_occupancy']:.3f}`",
        "- Top bound contacts:",
    ]
    for row in top_contacts:
        lines.append(
            f"  - {row['resn']} {row['chain']} {row['resi']}: occupancy={row['bound_contact_occupancy']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "- Physics:",
        ]
    )
    for key, value in physics.items():
        lines.append(f"  - {key}: {'pass' if value else 'fail'}")
    lines.append("- Biology:")
    for key, value in biology.items():
        lines.append(f"  - {key}: {'pass' if value else 'fail'}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_replica_runtime_row(summary: dict) -> dict:
    time_rows = [dict(r or {}) for r in list(summary.get("timeseries", []) or [])]
    has_estimator_flags = any(("estimator_bound_like" in r) or ("estimator_unbound_like" in r) for r in time_rows)
    bound_flag_key = "estimator_bound_like" if has_estimator_flags else "bound_like"
    unbound_flag_key = "estimator_unbound_like" if has_estimator_flags else "unbound_like"
    rows = [
        dict(r or {})
        for r in time_rows
        if bool((r or {}).get(bound_flag_key, False)) or bool((r or {}).get(unbound_flag_key, False))
    ]
    free_energy = dict(summary.get("free_energy_proxy", {}) or {})
    inputs = dict(summary.get("inputs", {}) or {})
    return {
        "replica_index": int(inputs.get("replica_index", 1) or 1),
        "random_seed": int(inputs.get("random_seed", 0) or 0),
        "total_valid": np.asarray([float(r.get("e_total_kj_mol", np.nan)) for r in rows], dtype=float),
        "distance_valid": np.asarray([float(r.get("protein_min_dist_ang", np.nan)) for r in rows], dtype=float),
        "coul_valid": np.asarray([float(r.get("e_coul_kj_mol", np.nan)) for r in rows], dtype=float),
        "bias_valid": np.asarray([float(r.get("e_bias_kj_mol", np.nan)) for r in rows], dtype=float),
        "bound_valid_mask": np.asarray([bool(r.get(bound_flag_key, False)) for r in rows], dtype=bool),
        "unbound_valid_mask": np.asarray([bool(r.get(unbound_flag_key, False)) for r in rows], dtype=bool),
        "bound_cutoff_a": 4.5,
        "n_bound": int(free_energy.get("n_bound", 0) or 0),
        "n_unbound": int(free_energy.get("n_unbound", 0) or 0),
        "n_dist_valid": int(free_energy.get("n_valid", 0) or 0),
        "n_bound_raw": int(free_energy.get("n_bound_raw", free_energy.get("n_bound", 0)) or 0),
        "n_unbound_raw": int(free_energy.get("n_unbound_raw", free_energy.get("n_unbound", 0)) or 0),
        "n_bound_hold": int(free_energy.get("n_bound_hold", free_energy.get("n_bound", 0)) or 0),
        "n_unbound_hold": int(free_energy.get("n_unbound_hold", free_energy.get("n_unbound", 0)) or 0),
        "estimator_window_mode": str(free_energy.get("estimator_window_mode", "state_fallback") or "state_fallback"),
        "dg_pop_kj": float(free_energy.get("dg_pop_kj_mol", np.nan)),
        "p_bound": float(free_energy.get("p_bound", np.nan)),
        "p_unbound": float(free_energy.get("p_unbound", np.nan)),
        "v_eff_nm3": float(free_energy.get("v_eff_nm3", np.nan)),
        "sim_ns": float(inputs.get("sim_ns", np.nan)),
        "sample_every_ps": float(inputs.get("sample_every_ps", np.nan)),
    }


def write_replica_markdown_report(out_path: Path, summary: dict) -> None:
    agg = dict(summary.get("aggregate_free_energy", {}) or {})
    rows = list(summary.get("replicas", []) or [])

    def _fmt_or_na(value, fmt):
        try:
            vv = float(value)
        except Exception:
            return "not estimable"
        return fmt.format(vv) if np.isfinite(vv) else "not estimable"

    def _fmt_unit_or_na(value, fmt, unit):
        txt = _fmt_or_na(value, fmt)
        return txt if txt == "not estimable" else f"{txt} {unit}"

    lines = [
        "# 3PTB/BEN Replica Aggregate Check",
        "",
        f"Completed replicas: `{int(summary.get('completed_replica_count', 0))}` / `{int(summary.get('requested_replica_count', 0))}`",
        f"All replicas passed built-in checks: `{'yes' if bool(summary.get('all_replicas_pass', False)) else 'no'}`",
        "",
        "## Aggregate",
        "",
        f"- Replica convergence: `{str(agg.get('convergence_quality', 'unknown'))}`",
        f"- Mean ΔG(exp): `{float(agg.get('dg_exp_mean', np.nan)):.3f} kJ/mol`",
        f"- ΔG(exp) SEM: `{float(agg.get('dg_exp_sem', np.nan)):.3f} kJ/mol`",
        f"- ΔG(exp) span: `{float(agg.get('dg_exp_span', np.nan)):.3f} kJ/mol`",
        f"- Selected estimator: `{str(agg.get('selected_estimator', 'gated_off'))}` | confidence=`{str(agg.get('reweight_confidence_quality', 'low'))}`",
        f"- Selected note: `{str(agg.get('reweight_confidence_note', ''))}`",
        f"- Selected replicas: `{int(agg.get('selected_replica_count', 0))}` / `{int(agg.get('replica_count', 0))}`",
        f"- Selected ΔG: `{_fmt_unit_or_na(agg.get('dg_selected_kj', np.nan), '{:.3f}', 'kJ/mol')}`",
        f"- Selected Kd(app): `{_fmt_unit_or_na(agg.get('kd_selected_m', np.nan), '{:.3e}', 'M')}`",
        f"- Bias-reweighted ΔG: `{_fmt_unit_or_na(agg.get('dg_reweighted_kj', np.nan), '{:.3f}', 'kJ/mol')}`",
        f"- Bias-reweighted Kd(app): `{_fmt_unit_or_na(agg.get('kd_app_reweighted_m', np.nan), '{:.3e}', 'M')}`",
        f"- Bias-reweighted ESS(total): `{float(agg.get('reweight_ess_total', np.nan)):.2f}`",
        f"- Aggregate Jeffreys diagnostic: `{_fmt_unit_or_na(agg.get('dg_pop_jeff_kj', np.nan), '{:.3f}', 'kJ/mol')}` "
        "(pseudo-count only; not reported as BFE/Kd when one state is missing)",
        f"- Unbound coverage fraction: `{float(agg.get('unbound_coverage_fraction', np.nan)):.3f}`",
        "",
        "## Replica Rows",
        "",
    ]
    for row in rows:
        lines.append(
            f"- replica {int(row.get('replica_index', 0))}: "
            f"seed={int(row.get('random_seed', 0))}, "
            f"pass={bool(row.get('overall_pass', False))}, "
            f"bound={int(row.get('n_bound', 0))}, "
            f"unbound={int(row.get('n_unbound', 0))}, "
            f"bound_mean={float(row.get('mean_bound_total_kj_mol', np.nan)):.3f}, "
            f"unbound_mean={float(row.get('mean_unbound_total_kj_mol', np.nan)):.3f}, "
            f"selected={str(row.get('selected_estimator', 'gated_off'))}, "
            f"ΔG(sel)={float(row.get('dg_selected_kj_mol', np.nan)):.3f}, "
            f"ΔG(rw)={float(row.get('dg_reweighted_kj_mol', np.nan)):.3f}, "
            f"ΔG(pop,Jeff)={float(row.get('dg_pop_jeff_kj_mol', np.nan)):.3f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_single_run_command(args, script_path: Path, raw_pdb: Path, workdir: Path, random_seed: int, replica_index: int) -> list[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--pdb",
        str(raw_pdb),
        "--workdir",
        str(workdir),
        "--sim-ns",
        str(float(args.sim_ns)),
        "--sample-every-ps",
        str(float(args.sample_every_ps)),
        "--temperature-k",
        str(float(args.temperature_k)),
        "--pull-distance-a",
        str(float(args.pull_distance_a)),
        "--pull-k",
        str(float(args.pull_k)),
        "--minimize-iters",
        str(int(args.minimize_iters)),
        "--param-min-iters",
        str(int(args.param_min_iters)),
        "--eq-ps",
        str(float(args.eq_ps)),
        "--steering-mode",
        str(args.steering_mode),
        "--policy-mode",
        str(args.policy_mode),
        "--platform",
        str(args.platform or ""),
        "--random-seed",
        str(int(random_seed)),
        "--replica-index",
        str(int(replica_index)),
        "--replica-count",
        str(int(max(1, int(args.replicas)))),
        "--replicas",
        "1",
    ]
    if bool(args.fixed_steering):
        cmd.append("--fixed-steering")
    if str(args.policy_config_json or "").strip():
        cmd.extend(["--policy-config-json", str(args.policy_config_json)])
    return cmd


def run_replica_batch(args, repo_root: Path, raw_pdb: Path, workdir: Path) -> dict:
    script_path = Path(__file__).resolve()
    env = dict(os.environ)
    root_txt = str(repo_root)
    py_path = str(env.get("PYTHONPATH", "") or "").strip()
    env["PYTHONPATH"] = root_txt if not py_path else (root_txt + os.pathsep + py_path)
    summary_rows = []
    summary_paths = []
    failures = []
    for rep_idx in range(1, int(max(1, args.replicas)) + 1):
        rep_seed = int(args.seed0 + (rep_idx - 1)) if int(args.seed0) > 0 else 0
        rep_workdir = workdir / f"replica_{rep_idx:02d}"
        rep_workdir.mkdir(parents=True, exist_ok=True)
        cmd = build_single_run_command(args, script_path, raw_pdb, rep_workdir, rep_seed, rep_idx)
        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(repo_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.stdout.strip():
            print(proc.stdout.strip())
        if proc.returncode != 0:
            failures.append(
                {
                    "replica_index": int(rep_idx),
                    "random_seed": int(rep_seed),
                    "returncode": int(proc.returncode),
                    "stderr_tail": str(proc.stderr[-2000:]),
                }
            )
            continue
        summary_path = rep_workdir / "metal_biology_summary.json"
        if not summary_path.is_file():
            failures.append(
                {
                    "replica_index": int(rep_idx),
                    "random_seed": int(rep_seed),
                    "returncode": 0,
                    "stderr_tail": "missing metal_biology_summary.json",
                }
            )
            continue
        summary_obj = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_rows.append(dict(summary_obj))
        summary_paths.append(str(summary_path))

    if not summary_rows:
        raise RuntimeError(f"All replica runs failed: {json.dumps(failures, indent=2)}")

    replica_runtime_rows = [build_replica_runtime_row(row) for row in summary_rows]
    aggregate = build_abfe_replica_aggregate_payload(
        replica_results=list(replica_runtime_rows),
        requested_replica_count=int(args.replicas),
        temperature_k=float(args.temperature_k),
        bootstrap_seed=int(max(1, int(args.seed0) or 20260312)),
        n_boot=300,
    )
    if not bool(aggregate.get("ok", False)):
        raise RuntimeError(f"Replica aggregation failed: {aggregate}")

    replica_rows = []
    for row in summary_rows:
        assessment = dict(row.get("assessment", {}) or {})
        free_energy = dict(row.get("free_energy_proxy", {}) or {})
        inputs = dict(row.get("inputs", {}) or {})
        system = dict(row.get("system", {}) or {})
        biology = dict(row.get("biology", {}) or {})
        replica_rows.append(
            {
                "replica_index": int(inputs.get("replica_index", 1) or 1),
                "random_seed": int(inputs.get("random_seed", 0) or 0),
                "overall_pass": bool(assessment.get("overall_pass", False)),
                "n_bound": int(free_energy.get("n_bound", 0) or 0),
                "n_unbound": int(free_energy.get("n_unbound", 0) or 0),
                "mean_bound_total_kj_mol": float(assessment.get("mean_bound_total_kj_mol", np.nan)),
                "mean_unbound_total_kj_mol": float(assessment.get("mean_unbound_total_kj_mol", np.nan)),
                "delta_bound_minus_unbound_kj_mol": float(assessment.get("delta_bound_minus_unbound_kj_mol", np.nan)),
                "dg_reweighted_kj_mol": float(free_energy.get("dg_reweighted_kj_mol", np.nan)),
                "kd_app_reweighted_m": float(free_energy.get("kd_app_reweighted_m", np.nan)),
                "dg_selected_kj_mol": float(free_energy.get("dg_selected_kj_mol", np.nan)),
                "kd_selected_m": float(free_energy.get("kd_selected_m", np.nan)),
                "selected_estimator": str(free_energy.get("selected_estimator", "gated_off")),
                "reweight_confidence_quality": str(free_energy.get("reweight_confidence_quality", "low")),
                "dg_pop_jeff_kj_mol": float(free_energy.get("dg_pop_jeff_kj_mol", np.nan)),
                "kd_app_jeff_m": float(free_energy.get("kd_app_jeff_m", np.nan)),
                "asp189_amidinium_contact_occupancy": float(biology.get("asp189_amidinium_contact_occupancy", np.nan)),
                "min_metal_ligand_distance_ang": float(system.get("min_metal_ligand_distance_ang", np.nan)),
                "far_hold_frames": int(system.get("far_hold_frames", 0) or 0),
                "summary_json": str(Path(row.get("files", {}).get("final_pdb", "")).resolve().parent / "metal_biology_summary.json"),
            }
        )
    replica_rows.sort(key=lambda row: int(row.get("replica_index", 0)))

    aggregate_summary = {
        "inputs": {
            "raw_pdb": str(raw_pdb),
            "requested_replica_count": int(args.replicas),
            "seed0": int(args.seed0),
            "sim_ns": float(args.sim_ns),
            "sample_every_ps": float(args.sample_every_ps),
            "temperature_k": float(args.temperature_k),
            "pull_distance_a": float(args.pull_distance_a),
            "pull_k": float(args.pull_k),
            "steering_mode": str(args.steering_mode),
            "policy_mode": str(args.policy_mode),
            "policy_config_json": str(args.policy_config_json or ""),
            "adaptive_steering": bool(not args.fixed_steering),
        },
        "completed_replica_count": int(len(summary_rows)),
        "requested_replica_count": int(args.replicas),
        "failed_replica_count": int(len(failures)),
        "all_replicas_pass": bool(all(bool(r.get("overall_pass", False)) for r in replica_rows)),
        "aggregate_free_energy": dict(aggregate.get("summary", {}) or {}),
        "replicas": list(replica_rows),
        "failed_replicas": list(failures),
        "replica_summary_paths": list(summary_paths),
    }
    summary_path = workdir / "metal_biology_summary.json"
    report_path = workdir / "metal_biology_report.md"
    summary_path.write_text(json.dumps(aggregate_summary, indent=2), encoding="utf-8")
    write_replica_markdown_report(report_path, aggregate_summary)
    print("3PTB/BEN replica aggregate check")
    print(
        f"Replicas completed: {aggregate_summary['completed_replica_count']}/{aggregate_summary['requested_replica_count']} | "
        f"all_pass={aggregate_summary['all_replicas_pass']}"
    )
    print(
        f"Aggregate ΔG(exp) mean={float(aggregate_summary['aggregate_free_energy'].get('dg_exp_mean', np.nan)):.3f} kJ/mol | "
        f"SEM={float(aggregate_summary['aggregate_free_energy'].get('dg_exp_sem', np.nan)):.3f} | "
        f"selected={str(aggregate_summary['aggregate_free_energy'].get('selected_estimator', 'gated_off'))} | "
        f"ΔG(sel)={float(aggregate_summary['aggregate_free_energy'].get('dg_selected_kj', np.nan)):.3f} kJ/mol"
    )
    print(f"Summary JSON: {summary_path}")
    print(f"Report MD: {report_path}")
    return aggregate_summary


def main():
    ap = argparse.ArgumentParser(description="3PTB/BEN metal-preserving ABFE physics/biology check.")
    ap.add_argument("--pdb", default="3PTB.pdb")
    ap.add_argument("--workdir", default="test_abfe_3ptb_ben/metal_biology_check")
    ap.add_argument("--sim-ns", type=float, default=0.25)
    ap.add_argument("--sample-every-ps", type=float, default=2.0)
    ap.add_argument("--temperature-k", type=float, default=300.0)
    ap.add_argument("--pull-distance-a", type=float, default=18.0)
    ap.add_argument("--pull-k", type=float, default=2500.0)
    ap.add_argument("--minimize-iters", type=int, default=600)
    ap.add_argument("--param-min-iters", type=int, default=300)
    ap.add_argument("--eq-ps", type=float, default=10.0)
    ap.add_argument("--fixed-steering", action="store_true", help="Disable adaptive force scaling/corridor switching.")
    ap.add_argument("--steering-mode", choices=("pull_hold", "pull_hold_return"), default="pull_hold")
    ap.add_argument("--policy-mode", choices=("heuristic", "bandit"), default="heuristic")
    ap.add_argument("--policy-config-json", default="", help="Optional trained steering policy JSON.")
    ap.add_argument("--platform", default="", help="Optional OpenMM platform override; defaults to CUDA/OpenCL/Metal/CPU fallback.")
    ap.add_argument("--replicas", type=int, default=1, help="Run multiple seeded standalone replicas and aggregate.")
    ap.add_argument("--seed0", type=int, default=20260312, help="Base seed for replica batches; replica i uses seed0+i-1.")
    ap.add_argument("--random-seed", type=int, default=0, help="Single-run OpenMM seed.")
    ap.add_argument("--replica-index", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--replica-count", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    raw_pdb = Path(args.pdb).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if int(max(1, int(args.replicas))) > 1:
        run_replica_batch(args, repo_root, raw_pdb, workdir)
        return

    param_summary = parameterize_from_raw_3ptb(repo_root, raw_pdb, workdir, min_iters=int(args.param_min_iters))
    prepared_pdb = Path(param_summary["files"]["prepared_pdb"]).resolve()
    prmtop_path = Path(param_summary["files"]["prmtop"]).resolve()
    mol2_path = Path(param_summary["files"]["mol2"]).resolve()
    combined_pdb = workdir / "3PTB_BEN_CA_complex.pdb"
    ion_payload = build_complex_with_ions(prepared_pdb, raw_pdb, combined_pdb)

    pdb = PDBFile(str(combined_pdb))
    top = pdb.topology
    xyz0 = np.array(pdb.positions.value_in_unit(u.nanometer), dtype=float)
    atoms = list(top.atoms())
    n_atoms = len(atoms)

    lig_charge_map = load_mol2_charges(mol2_path)
    ligand_param_names = load_mol2_atom_names(mol2_path)

    atom_rows = []
    ligand_idx = []
    ligand_heavy_idx = []
    protein_idx = []
    protein_heavy_idx = []
    ion_idx = []
    ion_heavy_idx = []
    receptor_idx = []
    sigmas = np.zeros(n_atoms, dtype=float)
    epsilons = np.zeros(n_atoms, dtype=float)
    masses = np.zeros(n_atoms, dtype=float)
    charges = np.zeros(n_atoms, dtype=float)
    ligand_counter = 0

    for i, atom in enumerate(atoms):
        res = atom.residue
        resn = str(res.name or "").strip().upper()
        name = str(atom.name or "").strip()
        el = str(infer_element(atom) or "").strip().upper()
        is_lig = resn == "BEN"
        is_ion = (resn in ION_RESN) or (el in ION_CHARGES)
        origin = "ligand" if is_lig else ("ion" if is_ion else "protein")

        sigma, eps = nb_params(el)
        sigmas[i] = float(sigma)
        epsilons[i] = float(eps)
        masses[i] = float(mass_for_element(el))
        if origin == "ligand":
            param_name = ""
            if ligand_counter < len(ligand_param_names):
                param_name = str(ligand_param_names[ligand_counter]).strip().upper()
            ligand_counter += 1
            charges[i] = float(lig_charge_map.get(param_name or name.upper(), 0.0))
        elif origin == "ion":
            charges[i] = float(ion_charge(resn, name, el))
        else:
            charges[i] = float(estimate_protein_charge(resn, name, el))

        row = {
            "index": int(i),
            "name": str(name),
            "resn": str(resn),
            "chain": str(res.chain.id),
            "resi": str(res.id),
            "element": str(el),
            "origin": str(origin),
            "param_name": str(param_name if origin == "ligand" else ""),
        }
        atom_rows.append(row)

        if origin == "ligand":
            ligand_idx.append(int(i))
            if el != "H":
                ligand_heavy_idx.append(int(i))
        elif origin == "ion":
            ion_idx.append(int(i))
            receptor_idx.append(int(i))
            if el != "H":
                ion_heavy_idx.append(int(i))
        else:
            protein_idx.append(int(i))
            receptor_idx.append(int(i))
            if el != "H":
                protein_heavy_idx.append(int(i))

    if not ligand_idx or not receptor_idx:
        raise RuntimeError("Failed to identify ligand and receptor atoms in combined complex.")
    if ligand_counter != len(ligand_param_names):
        raise RuntimeError(
            f"Ligand atom count mismatch between complex ({ligand_counter}) and mol2 ({len(ligand_param_names)})."
        )

    system = System()
    for mass in masses:
        system.addParticle(max(1.0, float(mass)) * u.dalton)

    ff_atom_rows = [dict(row) for row in atom_rows]
    for row in ff_atom_rows:
        if str(row.get("origin", "")) == "ligand" and str(row.get("param_name", "")).strip():
            row["name"] = str(row.get("param_name", "")).strip()
    ff_counts = inject_ligand_terms_from_prmtop(system, ligand_idx, ff_atom_rows, prmtop_path)

    protein_rest = CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    protein_rest.addPerParticleParameter("k")
    protein_rest.addPerParticleParameter("x0")
    protein_rest.addPerParticleParameter("y0")
    protein_rest.addPerParticleParameter("z0")
    for idx in protein_heavy_idx:
        name = str(atom_rows[int(idx)]["name"]).strip().upper()
        k_rest = 3600.0 if name in {"N", "CA", "C", "O"} else 1300.0
        x0, y0, z0 = xyz0[int(idx)]
        protein_rest.addParticle(int(idx), [float(k_rest), float(x0), float(y0), float(z0)])
    for idx in ion_idx:
        x0, y0, z0 = xyz0[int(idx)]
        protein_rest.addParticle(int(idx), [4000.0, float(x0), float(y0), float(z0)])
    system.addForce(protein_rest)

    hold = CustomExternalForce("0.5*k_hold*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    hold.addGlobalParameter("k_hold", 0.0)
    hold.addPerParticleParameter("x0")
    hold.addPerParticleParameter("y0")
    hold.addPerParticleParameter("z0")
    for idx in ligand_heavy_idx:
        x0, y0, z0 = xyz0[int(idx)]
        hold.addParticle(int(idx), [float(x0), float(y0), float(z0)])
    system.addForce(hold)

    receptor_group = set(int(i) for i in receptor_idx)
    ligand_group = set(int(i) for i in ligand_idx)
    lj = CustomNonbondedForce("4*sqrt(epsilon1*epsilon2)*((0.5*(sigma1+sigma2)/r)^12-(0.5*(sigma1+sigma2)/r)^6)")
    lj.addPerParticleParameter("sigma")
    lj.addPerParticleParameter("epsilon")
    lj.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    lj.setCutoffDistance(1.2 * u.nanometer)
    lj.addInteractionGroup(receptor_group, ligand_group)
    lj.setForceGroup(1)

    coul = CustomNonbondedForce("138.935456*q1*q2/r")
    coul.addPerParticleParameter("q")
    coul.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
    coul.setCutoffDistance(1.2 * u.nanometer)
    coul.addInteractionGroup(receptor_group, ligand_group)
    coul.setForceGroup(2)

    for i in range(n_atoms):
        lj.addParticle([float(sigmas[i]), float(epsilons[i])])
        coul.addParticle([float(charges[i])])
    system.addForce(lj)
    system.addForce(coul)

    lig_heavy = np.asarray(ligand_heavy_idx if ligand_heavy_idx else ligand_idx, dtype=int)
    prot_anchor = np.asarray(protein_heavy_idx if protein_heavy_idx else protein_idx, dtype=int)
    if lig_heavy.size <= 0 or prot_anchor.size <= 0:
        raise RuntimeError("Failed to define heavy-atom groups for the ABFE coordinate.")

    lig_masses = np.asarray([max(1.0, masses[int(i)]) for i in lig_heavy], dtype=float)
    pull_dist_nm = max(0.4, float(args.pull_distance_a) * 0.1)
    k_total = max(200.0, float(args.pull_k))
    steering_corridors = []
    for row in list(
        rank_pull_directions(
            xyz0,
            protein_idx,
            ligand_idx,
            protein_heavy_ids=protein_heavy_idx,
            ligand_heavy_ids=ligand_heavy_idx,
        )
        or []
    )[:8]:
        direction = np.asarray(row.get("direction", [1.0, 0.0, 0.0]), dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1.0e-8:
            continue
        steering_corridors.append(
            {
                "direction": np.asarray(direction / norm, dtype=float),
                "clearance_nm": float(row.get("clearance_nm", np.nan)),
                "collision_penalty": float(row.get("collision_penalty", np.nan)),
                "alignment": float(row.get("alignment", np.nan)),
                "score": float(row.get("score", np.nan)),
            }
        )
    if not steering_corridors:
        steering_corridors = [{
            "direction": np.array([1.0, 0.0, 0.0], dtype=float),
            "clearance_nm": np.nan,
            "collision_penalty": np.nan,
            "alignment": np.nan,
            "score": np.nan,
        }]
    pull_dir = np.asarray(steering_corridors[0]["direction"], dtype=float)
    pull_clearance_nm = float(steering_corridors[0].get("clearance_nm", np.nan))
    pull_collision_penalty = float(steering_corridors[0].get("collision_penalty", np.nan))
    initial_pull_clearance_nm = float(pull_clearance_nm)
    initial_pull_collision_penalty = float(pull_collision_penalty)
    if np.isfinite(pull_clearance_nm):
        pull_dist_nm = max(pull_dist_nm, min(2.0, pull_clearance_nm + 1.0))
    pull_vec = pull_dir * pull_dist_nm
    pull_particles = [int(i) for i in lig_heavy.tolist()]
    lig_com0 = np.average(xyz0[lig_heavy], axis=0, weights=lig_masses)
    if len(pull_particles) > 12:
        d0 = np.linalg.norm(xyz0[pull_particles] - lig_com0[None, :], axis=1)
        keep = np.argsort(d0)[:12]
        pull_particles = [pull_particles[int(k)] for k in keep]

    steering_dir_index = 0
    steering_switch_count = 0
    steering_k_current = float(k_total)
    k_per_atom = float(steering_k_current / float(max(1, len(pull_particles))))
    steer = CustomExternalForce("0.5*k_pull*((x-(x0+lam*dx))^2 + (y-(y0+lam*dy))^2 + (z-(z0+lam*dz))^2)")
    steer.addGlobalParameter("k_pull", float(k_per_atom))
    steer.addGlobalParameter("lam", 0.0)
    steer.addPerParticleParameter("x0")
    steer.addPerParticleParameter("y0")
    steer.addPerParticleParameter("z0")
    steer.addPerParticleParameter("dx")
    steer.addPerParticleParameter("dy")
    steer.addPerParticleParameter("dz")
    steer_rows = []
    for idx in pull_particles:
        x0, y0, z0 = xyz0[int(idx)]
        slot = steer.addParticle(
            int(idx),
            [float(x0), float(y0), float(z0), float(pull_vec[0]), float(pull_vec[1]), float(pull_vec[2])],
        )
        steer_rows.append(
            {
                "slot": int(slot),
                "particle": int(idx),
                "x0": float(x0),
                "y0": float(y0),
                "z0": float(z0),
            }
        )
    applied_pull_vec = np.asarray(pull_vec, dtype=float)
    corridor_pull_vec = np.asarray(pull_vec, dtype=float)
    steer.setForceGroup(3)
    system.addForce(steer)

    d_all = np.linalg.norm(xyz0[prot_anchor] - lig_com0[None, :], axis=1)
    keep_idx = np.argsort(d_all)[: min(96, len(d_all))]
    site_anchor_idx = np.asarray([int(prot_anchor[int(k)]) for k in keep_idx], dtype=int)
    site_com0 = np.mean(xyz0[site_anchor_idx], axis=0)

    timestep_ps = 0.001
    try:
        integ = LangevinMiddleIntegrator(float(args.temperature_k) * u.kelvin, 5.0 / u.picosecond, timestep_ps * u.picoseconds)
    except Exception:
        integ = LangevinIntegrator(float(args.temperature_k) * u.kelvin, 5.0 / u.picosecond, timestep_ps * u.picoseconds)
    if int(args.random_seed) > 0 and hasattr(integ, "setRandomNumberSeed"):
        try:
            integ.setRandomNumberSeed(int(args.random_seed))
        except Exception:
            pass

    ctx, platform, platform_name, platform_props = build_context_with_platform_fallback(
        system,
        integ,
        platform_override=str(args.platform or ""),
    )
    ctx.setPositions(xyz0 * u.nanometer)
    try:
        if int(args.random_seed) > 0:
            ctx.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin, int(args.random_seed))
        else:
            ctx.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin)
    except Exception:
        ctx.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin)

    max_it = max(0, int(args.minimize_iters))
    if max_it > 0:
        ctx.setParameter("k_hold", 1800.0)
        LocalEnergyMinimizer.minimize(ctx, maxIterations=max_it)
        ctx.setParameter("k_hold", 0.0)
        LocalEnergyMinimizer.minimize(ctx, maxIterations=max(200, max_it // 2))

    min_positions = ctx.getState(getPositions=True).getPositions()
    try:
        integ_run = LangevinMiddleIntegrator(
            float(args.temperature_k) * u.kelvin,
            5.0 / u.picosecond,
            timestep_ps * u.picoseconds,
        )
    except Exception:
        integ_run = LangevinIntegrator(
            float(args.temperature_k) * u.kelvin,
            5.0 / u.picosecond,
            timestep_ps * u.picoseconds,
        )
    if int(args.random_seed) > 0 and hasattr(integ_run, "setRandomNumberSeed"):
        try:
            integ_run.setRandomNumberSeed(int(args.random_seed))
        except Exception:
            pass

    if platform_props:
        sim = app.Simulation(top, system, integ_run, platform, platform_props)
    else:
        sim = app.Simulation(top, system, integ_run, platform)
    sim.context.setPositions(min_positions)
    try:
        if int(args.random_seed) > 0:
            sim.context.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin, int(args.random_seed))
        else:
            sim.context.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin)
    except Exception:
        sim.context.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin)

    out_dcd = workdir / "abfe_metal_3ptb.dcd"
    out_final_pdb = workdir / "abfe_metal_3ptb_final.pdb"
    if out_dcd.exists():
        out_dcd.unlink()
    stride = max(1, int(round(float(args.sample_every_ps) / timestep_ps)))
    sim.reporters.append(DCDReporter(str(out_dcd), stride, enforcePeriodicBox=False))

    eq_ps = max(0.0, float(args.eq_ps))
    if eq_ps > 0.0:
        sim.step(max(1, int(round(eq_ps / timestep_ps))))

    sim_ns = max(0.2, float(args.sim_ns))
    prod_steps = int(round((sim_ns * 1000.0) / timestep_ps))
    n_samples = max(1, prod_steps // stride)
    lam_plan = build_abfe_steering_lambda_schedule(
        n_samples=int(n_samples),
        steering_mode=str(args.steering_mode),
        steering_enabled=True,
    )
    lam_schedule = np.asarray(lam_plan.get("lambda_schedule", []), dtype=float)
    if int(lam_schedule.size) != int(n_samples):
        lam_schedule = np.ones(int(n_samples), dtype=float)

    energies_total = []
    energies_lj = []
    energies_coul = []
    energies_bias = []
    energies_metal_total = []
    protein_min_dist = []
    rc_site_min = []
    rc_site_com = []
    xyz_series = []
    times_ps = []
    protein_contact_count_hist = []
    pull_k_hist = []
    corridor_hist = []
    steering_reason_hist = []
    steering_note_hist = []
    steering_phase_hist = []
    steering_phase_sample_hist = []
    far_hold_stable_hist = []
    policy_action_hist = []
    policy_confidence_hist = []
    policy_reward_hist = []
    policy_trace_rows = []
    lig_com_series = []

    ion_group = [int(i) for i in ion_idx]
    ligand_group_list = [int(i) for i in ligand_idx]
    adaptive_enabled = bool(not args.fixed_steering)
    ligand_ring_count = 1
    try:
        policy_pocket_desc = compute_runtime_pocket_descriptors(
            atom_rows,
            xyz0,
            protein_idx,
            ligand_idx,
            pocket_cutoff_a=6.0,
        )
    except Exception:
        policy_pocket_desc = {}
    adaptive_state = build_abfe_adaptive_steering_state_payload(
        enabled=bool(adaptive_enabled),
        steering_mode=str(args.steering_mode),
        initial_pull_k_kj_nm2=float(steering_k_current),
        n_direction_candidates=int(len(steering_corridors)),
        ligand_heavy_atoms=int(lig_heavy.size),
        formal_charge_e=float(np.sum(np.asarray(charges[np.asarray(ligand_idx, dtype=int)], dtype=float))) if len(ligand_idx) > 0 else np.nan,
        ring_count=int(ligand_ring_count),
        pocket_desc=dict(policy_pocket_desc or {}),
        direction_clearances_nm=[float(dict(c or {}).get("clearance_nm", np.nan)) for c in list(steering_corridors or [])],
        direction_collision_penalties=[float(dict(c or {}).get("collision_penalty", np.nan)) for c in list(steering_corridors or [])],
        enable_probe_phase=True,
        sample_every_ps=float(args.sample_every_ps),
    )
    policy_state = build_abfe_steering_policy_state_payload(
        enabled=bool(adaptive_enabled),
        policy_mode=str(args.policy_mode),
        ligand_heavy_atoms=int(lig_heavy.size),
        formal_charge_e=float(np.sum(np.asarray(charges[np.asarray(ligand_idx, dtype=int)], dtype=float))) if len(ligand_idx) > 0 else np.nan,
        ring_count=int(ligand_ring_count),
        target_hint=str(raw_pdb.stem),
        metal_contact_expected=bool(len(ion_idx) > 0),
        pocket_desc=dict(policy_pocket_desc or {}),
        policy_model_path=str(args.policy_config_json or ""),
    )
    steering_k_current = float((adaptive_state or {}).get("current_pull_k_kj_nm2", steering_k_current))
    try:
        sim.context.setParameter("k_pull", float(steering_k_current / float(max(1, len(pull_particles)))))
    except Exception:
        pass
    xyz_prev = np.asarray(xyz0, dtype=float)

    def apply_pull_vector(target_vec):
        nonlocal applied_pull_vec
        vv = np.asarray(target_vec, dtype=float)
        if vv.shape != (3,) or not np.all(np.isfinite(vv)):
            return False
        if np.allclose(vv, applied_pull_vec, atol=1.0e-6):
            return False
        for row in list(steer_rows or []):
            steer.setParticleParameters(
                int(row["slot"]),
                int(row["particle"]),
                [
                    float(row["x0"]),
                    float(row["y0"]),
                    float(row["z0"]),
                    float(vv[0]),
                    float(vv[1]),
                    float(vv[2]),
                ],
            )
        steer.updateParametersInContext(sim.context)
        applied_pull_vec = np.asarray(vv, dtype=float)
        return True

    for sample_idx in range(n_samples):
        lam_runtime = float(lam_schedule[int(sample_idx)])
        adaptive_state_pre = dict(adaptive_state or {})
        phase_pre = str(adaptive_state_pre.get("phase", "pull") or "pull").strip().lower()
        steering_phase_sample_hist.append(str(phase_pre or "pull"))
        try:
            pull_distance_scale_runtime = float(
                adaptive_state_pre.get(
                    "current_pull_distance_scale",
                    adaptive_state_pre.get("target_pull_distance_scale", 1.0),
                )
            )
        except Exception:
            pull_distance_scale_runtime = 1.0
        if (not np.isfinite(pull_distance_scale_runtime)) or pull_distance_scale_runtime <= 0.0:
            pull_distance_scale_runtime = 1.0
        target_pull_dist_nm = float(max(pull_dist_nm, pull_dist_nm * pull_distance_scale_runtime))
        target_pull_vec = np.asarray(pull_dir * target_pull_dist_nm, dtype=float)
        if adaptive_enabled and phase_pre == "probe":
            lam_runtime = float(adaptive_state_pre.get("probe_phase_lambda_value", lam_runtime) or lam_runtime)
        elif adaptive_enabled and phase_pre in {"far_hold", "detached_settle"}:
            if phase_pre == "detached_settle":
                lam_runtime = float(adaptive_state_pre.get("detached_settle_lambda_value", 0.05) or 0.05)
            else:
                lam_runtime = float(adaptive_state_pre.get("hold_lambda_value", 1.0) or 1.0)
            if bool(adaptive_state_pre.get("surface_detach_enabled", False)):
                detach_plan = build_abfe_surface_detach_direction_payload(
                    coords_nm=np.asarray(xyz_prev, dtype=float),
                    prot_heavy_idx=np.asarray(protein_heavy_idx, dtype=int),
                    lig_heavy_idx=np.asarray(lig_heavy, dtype=int),
                    site_anchor_idx=np.asarray(site_anchor_idx, dtype=int),
                    fallback_direction=np.asarray(pull_dir, dtype=float),
                    contact_cutoff_nm=float(adaptive_state_pre.get("surface_detach_contact_cutoff_nm", 0.45)),
                )
                detach_dir = np.asarray(detach_plan.get("direction", pull_dir), dtype=float)
                detach_scale = float(
                    max(
                        pull_distance_scale_runtime,
                        float(adaptive_state_pre.get("surface_detach_distance_scale", 1.15) or 1.15),
                    )
                )
                max_detach_nm = float(adaptive_state_pre.get("surface_detach_max_distance_nm", 2.4) or 2.4)
                detach_dist_nm = float(max(target_pull_dist_nm, min(max_detach_nm, pull_dist_nm * max(1.0, detach_scale))))
                target_pull_vec = np.asarray(detach_dir * detach_dist_nm, dtype=float)
        elif (
            adaptive_enabled
            and phase_pre == "pull"
            and bool(adaptive_state_pre.get("pull_surface_detach_active", False))
            and bool(adaptive_state_pre.get("surface_detach_enabled", False))
        ):
            detach_plan = build_abfe_surface_detach_direction_payload(
                coords_nm=np.asarray(xyz_prev, dtype=float),
                prot_heavy_idx=np.asarray(protein_heavy_idx, dtype=int),
                lig_heavy_idx=np.asarray(lig_heavy, dtype=int),
                site_anchor_idx=np.asarray(site_anchor_idx, dtype=int),
                fallback_direction=np.asarray(pull_dir, dtype=float),
                contact_cutoff_nm=float(adaptive_state_pre.get("surface_detach_contact_cutoff_nm", 0.45)),
            )
            detach_dir = np.asarray(detach_plan.get("direction", pull_dir), dtype=float)
            try:
                blend_frac = float(adaptive_state_pre.get("pull_surface_detach_blend_fraction", 0.30))
            except Exception:
                blend_frac = 0.30
            blend_frac = float(np.clip(blend_frac, 0.0, 0.75))
            detach_scale = float(
                max(
                    pull_distance_scale_runtime,
                    float(adaptive_state_pre.get("surface_detach_distance_scale", 1.15) or 1.15),
                )
            )
            max_detach_nm = float(adaptive_state_pre.get("surface_detach_max_distance_nm", 2.4) or 2.4)
            detach_dist_nm = float(max(target_pull_dist_nm, min(max_detach_nm, pull_dist_nm * max(1.0, detach_scale))))
            blend_vec = ((1.0 - blend_frac) * np.asarray(pull_dir, dtype=float)) + (blend_frac * detach_dir)
            blend_norm = float(np.linalg.norm(blend_vec))
            if blend_norm > 1.0e-8:
                target_pull_vec = np.asarray((blend_vec / blend_norm) * detach_dist_nm, dtype=float)
        elif (
            adaptive_enabled
            and phase_pre == "pull"
            and bool(adaptive_state_pre.get("keep_pull_lambda_high_until_far_hold", False))
            and bool(adaptive_state_pre.get("high_lambda_seen", False))
        ):
            lam_runtime = 1.0
        apply_pull_vector(target_pull_vec)
        sim.context.setParameter("lam", float(lam_runtime))
        sim.step(stride)

        st_pos = sim.context.getState(getPositions=True)
        xyz = np.array(st_pos.getPositions(asNumpy=True).value_in_unit(u.nanometer), dtype=float)
        xyz_prev = np.asarray(xyz, dtype=float)
        xyz_series.append(np.asarray(xyz, dtype=float))

        st_lj = sim.context.getState(getEnergy=True, groups=(1 << 1))
        st_c = sim.context.getState(getEnergy=True, groups=(1 << 2))
        e_lj = float(st_lj.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        e_c = float(st_c.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        e_bias = 0.0
        try:
            st_bias = sim.context.getState(getEnergy=True, groups=(1 << 3))
            e_bias = float(st_bias.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        except Exception:
            e_bias = np.nan
        e_tot = float(e_lj + e_c)

        lig_com = np.average(xyz[lig_heavy], axis=0, weights=lig_masses)
        dv_prot = xyz[np.asarray(protein_heavy_idx, dtype=int)][:, None, :] - xyz[lig_heavy][None, :, :]
        rij_prot = np.linalg.norm(dv_prot, axis=2)
        protein_min_dist_a = float(np.min(rij_prot) * 10.0) if rij_prot.size else np.nan
        protein_contact_count = float(np.sum(np.any(rij_prot <= 0.4, axis=1))) if rij_prot.size else np.nan
        dv_site = xyz[site_anchor_idx][:, None, :] - xyz[lig_heavy][None, :, :]
        rc_min_a = float(np.min(np.linalg.norm(dv_site, axis=2)) * 10.0)
        rc_com_a = float(np.linalg.norm(lig_com - site_com0) * 10.0)
        metal_lj, metal_c = pair_energy_components(ion_group, ligand_group_list, xyz, sigmas, epsilons, charges)

        energies_total.append(float(e_tot))
        energies_lj.append(float(e_lj))
        energies_coul.append(float(e_c))
        energies_bias.append(float(e_bias) if np.isfinite(e_bias) else np.nan)
        energies_metal_total.append(float(metal_lj + metal_c))
        protein_min_dist.append(float(protein_min_dist_a))
        rc_site_min.append(float(rc_min_a))
        rc_site_com.append(float(rc_com_a))
        times_ps.append(float((sample_idx + 1) * float(args.sample_every_ps)))
        protein_contact_count_hist.append(float(protein_contact_count))
        lig_com_series.append(np.asarray(lig_com, dtype=float))
        steer_reason = ""
        steer_note = ""
        steer_decision = {}
        if adaptive_enabled and steer_rows:
            try:
                steer_decision = build_abfe_adaptive_steering_update_decision(
                    adaptive_state=dict(adaptive_state or {}),
                    sample_index=int(sample_idx),
                    lam_value=float(lam_runtime),
                    current_pull_k_kj_nm2=float(steering_k_current),
                    current_direction_index=int(steering_dir_index),
                    direction_count=int(len(steering_corridors)),
                    switches_used=int(steering_switch_count),
                    min_dist_a=list(protein_min_dist),
                    reactive_dist_a=list(rc_site_min),
                    site_min_dist_a=list(rc_site_min),
                    site_com_dist_a=list(rc_site_com),
                    contact_count_4a=list(protein_contact_count_hist),
                    total_e_kj=list(energies_total),
                )
                heuristic_decision = dict(steer_decision or {})
                if bool((policy_state or {}).get("enabled", False)):
                    policy_obs = build_abfe_steering_policy_observation_payload(
                        heuristic_decision=dict(heuristic_decision or {}),
                        adaptive_state=dict(adaptive_state or {}),
                        policy_state=dict(policy_state or {}),
                        sample_index=int(sample_idx),
                        lam_value=float(lam_runtime),
                        current_pull_k_kj_nm2=float(steering_k_current),
                        current_direction_index=int(steering_dir_index),
                        direction_count=int(len(steering_corridors)),
                        switches_used=int(steering_switch_count),
                    )
                    policy_action = build_abfe_steering_bandit_action_payload(
                        observation=dict(policy_obs or {}),
                        policy_state=dict(policy_state or {}),
                    )
                    steer_decision = build_abfe_steering_policy_safe_action_payload(
                        heuristic_decision=dict(heuristic_decision or {}),
                        policy_action=dict(policy_action or {}),
                        adaptive_state=dict(adaptive_state or {}),
                        policy_state=dict(policy_state or {}),
                        current_pull_k_kj_nm2=float(steering_k_current),
                        current_direction_index=int(steering_dir_index),
                        direction_count=int(len(steering_corridors)),
                        switches_used=int(steering_switch_count),
                    )
                    trace_row = build_abfe_steering_policy_trace_payload(
                        observation=dict(policy_obs or {}),
                        heuristic_decision=dict(heuristic_decision or {}),
                        policy_action=dict(policy_action or {}),
                        final_decision=dict(steer_decision or {}),
                    )
                    if bool(trace_row.get("ok", False)):
                        steer_decision["policy_reward_estimate"] = float(trace_row.get("reward_estimate", np.nan))
                        policy_trace_rows.append(dict(trace_row))
                    policy_state["last_action_name"] = str(
                        steer_decision.get("policy_action_name", policy_action.get("action_name", "keep")) or "keep"
                    )
                    policy_state["last_confidence"] = float(
                        steer_decision.get("policy_confidence", policy_action.get("confidence", 0.0)) or 0.0
                    )
                    if bool(steer_decision.get("policy_applied", False)):
                        policy_state["applied_action_count"] = int(
                            max(0, int(policy_state.get("applied_action_count", 0))) + 1
                        )
                adaptive_state = dict(steer_decision.get("adaptive_state", adaptive_state))
                steering_switch_count = int(adaptive_state.get("switches_used", steering_switch_count))
                steer_reason = str(steer_decision.get("reason", "") or "")
                steer_note = ",".join(str(v) for v in list(steer_decision.get("notes", []) or []))
                if bool(steer_decision.get("update_pull_k", False)):
                    steering_k_current = float(steer_decision.get("pull_k_kj_nm2", steering_k_current))
                    k_per_atom = float(steering_k_current / float(max(1, len(pull_particles))))
                    sim.context.setParameter("k_pull", float(k_per_atom))
                if bool(steer_decision.get("switch_direction", False)):
                    next_dir = int(steer_decision.get("direction_index", steering_dir_index))
                    if 0 <= next_dir < len(steering_corridors):
                        steering_dir_index = int(next_dir)
                        pull_dir = np.asarray(steering_corridors[steering_dir_index]["direction"], dtype=float)
                        corridor_pull_vec = np.asarray(pull_dir * pull_dist_nm, dtype=float)
                        pull_vec = np.asarray(corridor_pull_vec, dtype=float)
                        pull_clearance_nm = float(steering_corridors[steering_dir_index].get("clearance_nm", np.nan))
                        pull_collision_penalty = float(steering_corridors[steering_dir_index].get("collision_penalty", np.nan))
                        apply_pull_vector(corridor_pull_vec)
            except Exception as exc:
                adaptive_enabled = False
                adaptive_state = build_abfe_adaptive_steering_state_payload(enabled=False)
                policy_state = build_abfe_steering_policy_state_payload(enabled=False)
                steer_reason = f"adaptive_error:{exc}"
        pull_k_hist.append(float(steering_k_current))
        corridor_hist.append(int(steering_dir_index))
        steering_reason_hist.append(str(steer_reason))
        steering_note_hist.append(str(steer_note))
        steering_phase_hist.append(str((adaptive_state or {}).get("phase", "pull") or "pull"))
        far_hold_stable_hist.append(bool(steer_decision.get("far_hold_stable", False)))
        policy_action_hist.append(str(steer_decision.get("policy_action_name", "keep") or "keep"))
        policy_confidence_hist.append(float(steer_decision.get("policy_confidence", 0.0) or 0.0))
        policy_reward_hist.append(float(steer_decision.get("policy_reward_estimate", np.nan)))

    state = sim.context.getState(getPositions=True)
    with out_final_pdb.open("w", encoding="utf-8") as fh:
        PDBFile.writeFile(sim.topology, state.getPositions(), fh, keepIds=True)

    arr_total = np.asarray(energies_total, dtype=float)
    arr_lj = np.asarray(energies_lj, dtype=float)
    arr_coul = np.asarray(energies_coul, dtype=float)
    arr_metal = np.asarray(energies_metal_total, dtype=float)
    arr_protein_min = np.asarray(protein_min_dist, dtype=float)
    arr_rc = np.asarray(rc_site_min, dtype=float)
    arr_rc_com = np.asarray(rc_site_com, dtype=float)
    arr_contacts = np.asarray(protein_contact_count_hist, dtype=float)
    free_energy_payload = population_free_energy_metrics(
        frames=list(range(1, len(arr_total) + 1)),
        times_ps=list(times_ps),
        lam_hist=list(lam_schedule),
        bias_energy_kj=list(energies_bias),
        lj_energy_kj=list(arr_lj),
        total_energy_kj=list(arr_total),
        coul_energy_kj=list(arr_coul),
        protein_min_dist_ang=list(arr_protein_min),
        rc_site_min_ang=list(arr_rc),
        rc_site_com_ang=list(arr_rc_com),
        protein_contact_count_4a=list(arr_contacts),
        far_hold_stable_mask=list(far_hold_stable_hist),
        detached_settle_mask=[str(v or "").strip().lower() == "detached_settle" for v in list(steering_phase_sample_hist)],
        ligand_com_nm=list(lig_com_series),
        temperature_k=float(args.temperature_k),
        bootstrap_seed=int(max(1, int(args.random_seed) or 20260312)),
        hold_config=dict(adaptive_state or {}),
    )
    post_payload = dict(free_energy_payload.get("post_payload", {}) or {})
    free_energy = dict(free_energy_payload.get("summary", {}) or {})
    bound_mask = np.asarray(post_payload.get("bound_mask", []), dtype=bool)
    unbound_mask = np.asarray(post_payload.get("unbound_mask", []), dtype=bool)
    intermediate_mask = np.asarray(post_payload.get("intermediate_mask", []), dtype=bool)

    contacts, bound_frames = contact_occupancies(atom_rows, xyz_series, ligand_heavy_idx, bound_mask)
    asp189_occ, asp189_hits, asp189_total = asp189_amidinium_occupancy(atom_rows, xyz_series, bound_mask)

    min_metal_ligand_distance = float("nan")
    if ion_group:
        min_vals = []
        for xyz in xyz_series:
            ion_xyz = np.asarray(xyz[ion_group], dtype=float)
            lig_xyz = np.asarray(xyz[lig_heavy], dtype=float)
            dv = ion_xyz[:, None, :] - lig_xyz[None, :, :]
            min_vals.append(float(np.min(np.linalg.norm(dv, axis=2)) * 10.0))
        if min_vals:
            min_metal_ligand_distance = float(min(min_vals))

    ion_info = {
        "ion_residues": list(ion_payload.get("ions", [])),
        "min_metal_ligand_distance_ang": float(min_metal_ligand_distance),
    }
    assessment = summarize_checks(
        param_summary,
        ion_info,
        arr_total,
        arr_lj,
        arr_coul,
        arr_metal,
        arr_protein_min,
        arr_rc_com,
        bound_mask,
        unbound_mask,
        asp189_occ,
        contacts,
    )

    bound_hold_mask = np.asarray(post_payload.get("bound_hold_mask", []), dtype=bool)
    unbound_hold_mask = np.asarray(post_payload.get("unbound_hold_mask", []), dtype=bool)
    estimator_bound_mask = np.asarray(post_payload.get("estimator_bound_mask", []), dtype=bool)
    estimator_unbound_mask = np.asarray(post_payload.get("estimator_unbound_mask", []), dtype=bool)
    timeseries_rows = []
    for idx in range(len(arr_total)):
        timeseries_rows.append(
            {
                "sample": int(idx + 1),
                "time_ps": float(times_ps[idx]),
                "lambda": float(lam_schedule[idx]),
                "e_total_kj_mol": float(arr_total[idx]),
                "e_lj_kj_mol": float(arr_lj[idx]),
                "e_coul_kj_mol": float(arr_coul[idx]),
                "e_bias_kj_mol": float(energies_bias[idx]) if idx < len(energies_bias) else np.nan,
                "e_metal_total_kj_mol": float(arr_metal[idx]),
                "protein_min_dist_ang": float(arr_protein_min[idx]),
                "rc_site_min_ang": float(arr_rc[idx]),
                "rc_site_com_ang": float(arr_rc_com[idx]),
                "protein_contact_count_4a": float(arr_contacts[idx]),
                "bound_like": bool(bound_mask[idx]),
                "unbound_like": bool(unbound_mask[idx]),
                "intermediate_like": bool(intermediate_mask[idx]),
                "bound_hold_like": bool(bound_hold_mask[idx]) if idx < len(bound_hold_mask) else False,
                "unbound_hold_like": bool(unbound_hold_mask[idx]) if idx < len(unbound_hold_mask) else False,
                "estimator_bound_like": bool(estimator_bound_mask[idx]) if idx < len(estimator_bound_mask) else bool(bound_mask[idx]),
                "estimator_unbound_like": bool(estimator_unbound_mask[idx]) if idx < len(estimator_unbound_mask) else bool(unbound_mask[idx]),
                "pull_k_kj_nm2": float(pull_k_hist[idx]),
                "corridor_index": int(corridor_hist[idx]),
                "steering_phase": str(steering_phase_hist[idx]),
                "far_hold_stable": bool(far_hold_stable_hist[idx]),
                "steering_reason": str(steering_reason_hist[idx]),
                "steering_notes": str(steering_note_hist[idx]),
                "policy_action": str(policy_action_hist[idx]),
                "policy_confidence": float(policy_confidence_hist[idx]),
                "policy_reward_estimate": float(policy_reward_hist[idx]),
            }
        )

    policy_summary = build_abfe_steering_policy_summary_payload(
        trace_rows=list(policy_trace_rows or []),
    )
    trace_path = workdir / "policy_trace.jsonl"
    summary = {
        "inputs": {
            "raw_pdb": str(raw_pdb),
            "sim_ns": float(sim_ns),
            "sample_every_ps": float(args.sample_every_ps),
            "temperature_k": float(args.temperature_k),
            "pull_distance_a": float(args.pull_distance_a),
            "pull_k": float(k_total),
            "steering_mode": str(args.steering_mode),
            "policy_mode": str((policy_state or {}).get("policy_mode", args.policy_mode)),
            "policy_config_json": str((policy_state or {}).get("model_source", args.policy_config_json) or ""),
            "adaptive_steering": bool(not args.fixed_steering),
            "random_seed": int(args.random_seed),
            "replica_index": int(args.replica_index),
            "replica_count": int(args.replica_count),
        },
        "system": {
            "n_atoms": int(n_atoms),
            "ligand_atoms": int(len(ligand_idx)),
            "protein_atoms": int(len(protein_idx)),
            "ion_atoms": int(len(ion_idx)),
            "ion_residues": list(ion_payload.get("ions", [])),
            "min_metal_ligand_distance_ang": float(min_metal_ligand_distance),
            "platform": str(platform_name),
            "platform_properties": dict(platform_props or {}),
            "ligand_ff_terms": dict(ff_counts),
            "steering_corridor_count": int(len(steering_corridors)),
            "steering_initial_clearance_ang": float(initial_pull_clearance_nm * 10.0) if np.isfinite(initial_pull_clearance_nm) else np.nan,
            "steering_initial_collision_penalty": float(initial_pull_collision_penalty) if np.isfinite(initial_pull_collision_penalty) else np.nan,
            "steering_switches_used": int(steering_switch_count),
            "far_hold_frames": int(np.sum(np.asarray(far_hold_stable_hist, dtype=bool))),
            "policy_applied_actions": int((policy_summary or {}).get("n_applied", 0)),
        },
        "parameterization": dict(param_summary),
        "energetics": {
            "samples": int(len(arr_total)),
            "total_min_kj_mol": float(np.min(arr_total)),
            "total_max_kj_mol": float(np.max(arr_total)),
            "total_mean_kj_mol": float(np.mean(arr_total)),
            "lj_mean_kj_mol": float(np.mean(arr_lj)),
            "coul_mean_kj_mol": float(np.mean(arr_coul)),
            "metal_mean_kj_mol": float(np.mean(arr_metal)),
            "protein_min_dist_min_ang": float(np.min(arr_protein_min)),
            "protein_min_dist_max_ang": float(np.max(arr_protein_min)),
            "rc_site_min_min_ang": float(np.min(arr_rc)),
            "rc_site_min_max_ang": float(np.max(arr_rc)),
            "rc_site_com_min_ang": float(np.min(arr_rc_com)),
            "rc_site_com_max_ang": float(np.max(arr_rc_com)),
            "protein_contact_count_min": float(np.nanmin(arr_contacts)),
            "protein_contact_count_max": float(np.nanmax(arr_contacts)),
            "pull_k_min_kj_nm2": float(np.min(np.asarray(pull_k_hist, dtype=float))),
            "pull_k_max_kj_nm2": float(np.max(np.asarray(pull_k_hist, dtype=float))),
        },
        "free_energy_proxy": dict(free_energy),
        "biology": {
            "asp189_amidinium_contact_occupancy": float(asp189_occ),
            "asp189_contact_hits": int(asp189_hits),
            "asp189_contact_bound_frames": int(asp189_total),
            "bound_frames_for_contact_analysis": int(bound_frames),
        },
        "top_bound_contacts": list(contacts[:12]),
        "policy": dict(policy_summary or {}),
        "assessment": dict(assessment),
        "files": {
            "complex_with_ions_pdb": str(combined_pdb),
            "dcd": str(out_dcd),
            "final_pdb": str(out_final_pdb),
            "policy_trace": str(trace_path),
        },
        "timeseries": list(timeseries_rows),
    }

    summary_path = workdir / "metal_biology_summary.json"
    report_path = workdir / "metal_biology_report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if policy_trace_rows:
        trace_path.write_text(
            "\n".join(json.dumps(dict(row), sort_keys=True) for row in list(policy_trace_rows or [])) + "\n",
            encoding="utf-8",
        )
    elif trace_path.exists():
        trace_path.unlink()
    write_markdown_report(report_path, summary)

    print("3PTB/BEN metal-preserving ABFE check")
    print(f"Platform: {platform_name}")
    print(f"Ions retained: {ion_payload.get('ions', [])}")
    print(
        f"Mean total interaction={float(np.mean(arr_total)):.3f} kJ/mol | "
        f"bound={assessment['mean_bound_total_kj_mol']:.3f} | "
        f"unbound={assessment['mean_unbound_total_kj_mol']:.3f}"
    )
    print(
        f"Adaptive steering={'on' if not args.fixed_steering else 'off'} | "
        f"switches={steering_switch_count} | "
        f"selected={str(free_energy.get('selected_estimator', 'gated_off'))} | "
        f"DG(sel)={float(free_energy.get('dg_selected_kj_mol', np.nan)):.3f} kJ/mol | "
        f"DG(rw)={float(free_energy.get('dg_reweighted_kj_mol', np.nan)):.3f} kJ/mol"
    )
    print(
        f"Asp189 amidinium occupancy={asp189_occ:.3f} | "
        f"metal contribution mean={float(np.mean(arr_metal)):.3f} kJ/mol | "
        f"min metal distance={min_metal_ligand_distance:.3f} A"
    )
    print(f"Overall pass: {assessment['overall_pass']}")
    print(f"Summary JSON: {summary_path}")
    print(f"Report MD: {report_path}")
    if policy_trace_rows:
        print(f"Policy trace: {trace_path}")


if __name__ == "__main__":
    main()
