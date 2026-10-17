import collections
import csv
import datetime
import os

import numpy as np


def _finite_affinity_value(pose):
    try:
        affinity = float((pose or {}).get("affinity", float("nan")))
    except Exception:
        return float("nan")
    return float(affinity) if np.isfinite(affinity) else float("nan")


def _pick_best_affinity_pose(poses):
    best_pose = None
    best_affinity = float("inf")
    for pose in list(poses or []):
        affinity = _finite_affinity_value(pose)
        if (not np.isfinite(affinity)) or affinity >= float(best_affinity):
            continue
        best_pose = pose
        best_affinity = float(affinity)
    if best_pose is None:
        return None, float("nan")
    return best_pose, float(best_affinity)


def admet_score(props):
    score = 1.0
    try:
        mw = float((props or {}).get("mw", np.nan))
        logp = float((props or {}).get("logp", np.nan))
        hba = float((props or {}).get("hba", np.nan))
        hbd = float((props or {}).get("hbd", np.nan))
        tpsa = float((props or {}).get("tpsa", np.nan))
        rotb = float((props or {}).get("rotb", np.nan))
        pains = bool((props or {}).get("pains", False))
        if np.isfinite(mw) and mw > 500.0:
            score -= min(0.30, (mw - 500.0) / 450.0)
        if np.isfinite(logp) and logp > 5.0:
            score -= min(0.20, (logp - 5.0) / 4.0)
        if np.isfinite(hba) and hba > 10.0:
            score -= min(0.12, (hba - 10.0) / 10.0)
        if np.isfinite(hbd) and hbd > 5.0:
            score -= min(0.10, (hbd - 5.0) / 6.0)
        if np.isfinite(tpsa) and tpsa > 140.0:
            score -= min(0.18, (tpsa - 140.0) / 180.0)
        if np.isfinite(rotb) and rotb > 10.0:
            score -= min(0.10, (rotb - 10.0) / 12.0)
        if pains:
            score -= 0.30
    except Exception:
        pass
    return float(np.clip(score, 0.0, 1.0))


def relax_filter_limits(base_limits, level=1):
    lim = dict(base_limits or {})
    lvl = int(max(1, level))
    if lvl == 1:
        lim["mw_max"] = float(lim.get("mw_max", 550.0)) + 80.0
        lim["logp_max"] = float(lim.get("logp_max", 5.5)) + 0.8
        lim["hba_max"] = int(lim.get("hba_max", 12)) + 2
        lim["hbd_max"] = int(lim.get("hbd_max", 7)) + 1
        lim["tpsa_max"] = float(lim.get("tpsa_max", 160.0)) + 25.0
        lim["rot_max"] = int(lim.get("rot_max", 12)) + 3
    elif lvl == 2:
        lim["mw_max"] = float(lim.get("mw_max", 550.0)) + 160.0
        lim["logp_max"] = float(lim.get("logp_max", 5.5)) + 1.5
        lim["hba_max"] = int(lim.get("hba_max", 12)) + 4
        lim["hbd_max"] = int(lim.get("hbd_max", 7)) + 2
        lim["tpsa_max"] = float(lim.get("tpsa_max", 160.0)) + 45.0
        lim["rot_max"] = int(lim.get("rot_max", 12)) + 6
    else:
        lim["mw_max"] = float(lim.get("mw_max", 550.0)) + 260.0
        lim["logp_max"] = float(lim.get("logp_max", 5.5)) + 2.5
        lim["hba_max"] = int(lim.get("hba_max", 12)) + 6
        lim["hbd_max"] = int(lim.get("hbd_max", 7)) + 3
        lim["tpsa_max"] = float(lim.get("tpsa_max", 160.0)) + 70.0
        lim["rot_max"] = int(lim.get("rot_max", 12)) + 8
        lim["block_pains"] = False
    lim["mw_max"] = float(np.clip(float(lim.get("mw_max", 550.0)), 100.0, 1200.0))
    lim["logp_max"] = float(np.clip(float(lim.get("logp_max", 5.5)), -2.0, 12.0))
    lim["hba_max"] = int(np.clip(int(lim.get("hba_max", 12)), 0, 40))
    lim["hbd_max"] = int(np.clip(int(lim.get("hbd_max", 7)), 0, 25))
    lim["tpsa_max"] = float(np.clip(float(lim.get("tpsa_max", 160.0)), 10.0, 450.0))
    lim["rot_max"] = int(np.clip(int(lim.get("rot_max", 12)), 0, 50))
    lim["block_pains"] = bool(lim.get("block_pains", True))
    return lim


def filter_fail_reason(props, limits):
    lim = dict(limits or {})
    try:
        mw = float((props or {}).get("mw", np.nan))
        logp = float((props or {}).get("logp", np.nan))
        hba = float((props or {}).get("hba", np.nan))
        hbd = float((props or {}).get("hbd", np.nan))
        tpsa = float((props or {}).get("tpsa", np.nan))
        rotb = float((props or {}).get("rotb", np.nan))
        if not np.isfinite(mw):
            return "mw_nan"
        if not np.isfinite(logp):
            return "logp_nan"
        if not np.isfinite(hba):
            return "hba_nan"
        if not np.isfinite(hbd):
            return "hbd_nan"
        if not np.isfinite(tpsa):
            return "tpsa_nan"
        if not np.isfinite(rotb):
            return "rotb_nan"
        if mw > float(lim.get("mw_max", 550.0)):
            return "mw"
        if logp > float(lim.get("logp_max", 5.5)):
            return "logp"
        if hba > float(lim.get("hba_max", 12)):
            return "hba"
        if hbd > float(lim.get("hbd_max", 7)):
            return "hbd"
        if tpsa > float(lim.get("tpsa_max", 160.0)):
            return "tpsa"
        if rotb > float(lim.get("rot_max", 12)):
            return "rotb"
        if bool(lim.get("block_pains", True)) and bool((props or {}).get("pains", False)):
            return "pains"
    except Exception:
        return "invalid_props"
    return ""


def pass_filters(props, limits):
    return filter_fail_reason(props, limits) == ""


def mpo_score(affinity_kcal, admet_value, qed_score, sim_seed, weights=None):
    wd, wa, wq, ws = (0.55, 0.25, 0.15, 0.05)
    try:
        if isinstance(weights, (list, tuple)) and len(weights) == 4:
            wd = float(weights[0])
            wa = float(weights[1])
            wq = float(weights[2])
            ws = float(weights[3])
    except Exception:
        wd, wa, wq, ws = (0.55, 0.25, 0.15, 0.05)
    wsum = max(1e-6, wd + wa + wq + ws)
    wd, wa, wq, ws = wd / wsum, wa / wsum, wq / wsum, ws / wsum
    if np.isfinite(float(affinity_kcal)):
        dock_term = float(1.0 / (1.0 + np.exp((float(affinity_kcal) + 7.5) / 1.2)))
    else:
        dock_term = 0.0
    ad = float(np.clip(float(admet_value), 0.0, 1.0))
    qd = float(np.clip(float(qed_score), 0.0, 1.0)) if np.isfinite(float(qed_score)) else 0.0
    sm = float(np.clip(float(sim_seed), 0.0, 1.0)) if np.isfinite(float(sim_seed)) else 0.0
    return float(wd * dock_term + wa * ad + wq * qd + ws * sm)


def make_leadopt_refine_result():
    return {
        "refine_status": "failed",
        "refine_method": "",
        "refine_reason": "",
        "refine_dg_kj": np.nan,
        "refine_dg_err_kj": np.nan,
        "refine_ci_low_kj": np.nan,
        "refine_ci_high_kj": np.nan,
        "refine_samples": 0,
        "refine_n_eff": 0,
        "refine_platform": "",
        "refine_mean_e_kj": np.nan,
        "refine_min_dist_a": np.nan,
        "refine_protein_atoms": 0,
        "refine_param_mol2": "",
        "refine_param_prmtop": "",
        "refine_param_source": "",
        "refine_ff_counts": {},
    }


def leadopt_sanitize_refine_parameterization_options(net_charge=0, charge_method="bcc", atom_type="gaff2"):
    try:
        q = int(net_charge)
    except Exception:
        q = 0
    cm = str(charge_method or "").strip().lower()
    at = str(atom_type or "").strip().lower()
    if not cm:
        cm = "bcc"
    if not at:
        at = "gaff2"
    return {"net_charge": int(q), "charge_method": str(cm), "atom_type": str(at)}


def leadopt_prepare_refine_param_dir(pose_file, hit_idx=0, tag="", workspace_dir="."):
    pose_path = str(pose_file or "").strip()
    pose_dir = os.path.dirname(pose_path)
    job_guess = os.path.dirname(pose_dir) if os.path.basename(pose_dir).lower() == "poses" else pose_dir
    if not job_guess:
        job_guess = str(workspace_dir or ".")
    try:
        idx = int(hit_idx)
    except Exception:
        idx = 0
    tag_txt = str(tag or "").strip() or "00000000"
    param_dir = os.path.join(str(job_guess), "refine_params", f"hit_{int(idx)+1:03d}_{tag_txt}")
    return {
        "pose_dir": str(pose_dir),
        "job_guess": str(job_guess),
        "param_dir": str(param_dir),
    }


def leadopt_prepare_parameterization_for_refinement(
    hit,
    hit_idx,
    pose_file,
    workspace_dir,
    tag,
    net_charge=0,
    charge_method="bcc",
    atom_type="gaff2",
    prepare_parameterized_fn=None,
    load_charges_and_bonds_fn=None,
    load_atom_records_fn=None,
    file_exists_fn=None,
):
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    out = {
        "ok": False,
        "reason": "",
        "param_dir": "",
        "param_mol2": "",
        "param_prmtop": "",
        "param_source": "",
        "param_charges_by_name": {},
        "param_atom_records": [],
    }
    prep = leadopt_prepare_refine_param_dir(
        pose_file=pose_file,
        hit_idx=hit_idx,
        tag=tag,
        workspace_dir=workspace_dir,
    )
    out["param_dir"] = str(prep.get("param_dir", "") or "")
    if not callable(prepare_parameterized_fn):
        out["reason"] = "prepare_parameterized_fn_missing"
        return out
    try:
        ok_param, pinfo = prepare_parameterized_fn(
            hit,
            out["param_dir"],
            net_charge=int(net_charge),
            charge_method=str(charge_method or "bcc"),
            atom_type=str(atom_type or "gaff2"),
        )
    except Exception as e:
        ok_param, pinfo = False, {"reason": str(e)}
    pinfo = dict(pinfo or {})
    if not bool(ok_param):
        out["reason"] = "parameterized_ligand_required: " + str(pinfo.get("reason", "missing_mol2_or_prmtop"))
        return out
    param_mol2 = str(pinfo.get("mol2", "") or "").strip()
    param_prmtop = str(pinfo.get("prmtop", "") or "").strip()
    out["param_mol2"] = str(param_mol2)
    out["param_prmtop"] = str(param_prmtop)
    out["param_source"] = str(pinfo.get("source", "") or "")
    if (not bool(exists_fn(param_mol2))) or (not bool(exists_fn(param_prmtop))):
        out["reason"] = "parameterization_outputs_missing_after_generation"
        return out
    param_charges_by_name = {}
    if callable(load_charges_and_bonds_fn):
        try:
            q_map, _bonds = load_charges_and_bonds_fn(param_mol2)
            param_charges_by_name = dict(q_map or {})
        except Exception:
            param_charges_by_name = {}
    param_atom_records = []
    if callable(load_atom_records_fn):
        try:
            param_atom_records = list(load_atom_records_fn(param_mol2) or [])
        except Exception:
            param_atom_records = []
    if (not param_charges_by_name) and (not param_atom_records):
        out["reason"] = "parameterized_mol2_has_no_atom_charges"
        return out
    out["ok"] = True
    out["param_charges_by_name"] = dict(param_charges_by_name)
    out["param_atom_records"] = list(param_atom_records)
    return out


def leadopt_append_charge_map_source(current_source, map_mode, n_mapped, n_total):
    src = str(current_source or "")
    try:
        mode = str(map_mode or "none")
    except Exception:
        mode = "none"
    try:
        mapped = int(n_mapped)
    except Exception:
        mapped = 0
    try:
        total = int(n_total)
    except Exception:
        total = 0
    return src + f"|charge_map={mode}:{mapped}/{total}"


def leadopt_finalize_refine_estimate(est, energies, dmins, min_samples=8):
    payload = {"ok": False, "reason": "", "update": {}}
    e_vals = [] if energies is None else list(energies)
    e_arr = np.asarray(e_vals, dtype=float)
    e_arr = e_arr[np.isfinite(e_arr)]
    n_min = int(max(1, min_samples))
    if e_arr.size < n_min:
        payload["reason"] = "too_few_valid_energy_samples"
        return payload
    e = dict(est or {})
    if not bool(e.get("ok", False)):
        payload["reason"] = str(e.get("reason", "pymbar_estimation_failed"))
        return payload
    def _finite_or_nan(v):
        try:
            x = float(v)
        except Exception:
            return np.nan
        return float(x) if np.isfinite(x) else np.nan
    d_vals = [] if dmins is None else list(dmins)
    d_arr = np.asarray(d_vals, dtype=float)
    d_arr = d_arr[np.isfinite(d_arr)]
    try:
        n_eff = int(e.get("n_eff", e_arr.size))
    except Exception:
        n_eff = int(e_arr.size)
    payload["ok"] = True
    payload["update"] = {
        "refine_status": "ok",
        "refine_method": str(e.get("method", "")),
        "refine_reason": "",
        "refine_dg_kj": _finite_or_nan(e.get("dg_kj", np.nan)),
        "refine_dg_err_kj": _finite_or_nan(e.get("dg_err_kj", np.nan)),
        "refine_ci_low_kj": _finite_or_nan(e.get("ci_low_kj", np.nan)),
        "refine_ci_high_kj": _finite_or_nan(e.get("ci_high_kj", np.nan)),
        "refine_samples": int(e_arr.size),
        "refine_n_eff": int(n_eff),
        "refine_mean_e_kj": float(np.mean(e_arr)) if e_arr.size else np.nan,
        "refine_min_dist_a": float(np.mean(d_arr)) if d_arr.size else np.nan,
    }
    return payload


def leadopt_build_complex_selection_ids(atoms, polymer_index_rows, ligand_index_rows):
    idx_to_pid = {}
    for pid, atom in enumerate(list(atoms or [])):
        try:
            pidx = int(getattr(atom, "index", 0) or 0)
        except Exception:
            pidx = 0
        if pidx > 0:
            idx_to_pid[int(pidx)] = int(pid)

    def _collect(rows):
        out = []
        for row in list(rows or []):
            try:
                _obj, pidx = row
            except Exception:
                continue
            try:
                pid = idx_to_pid.get(int(pidx), None)
            except Exception:
                pid = None
            if pid is not None:
                out.append(int(pid))
        return sorted(set(out))

    return {
        "idx_to_pid": dict(idx_to_pid),
        "protein_ids": _collect(polymer_index_rows),
        "ligand_ids": _collect(ligand_index_rows),
    }


def leadopt_infer_atom_names_elements(atoms, infer_element_fn=None):
    atom_names = []
    elements = []
    for atom in list(atoms or []):
        nm = str(getattr(atom, "name", "")).strip()
        sy = str(getattr(atom, "symbol", "")).strip()
        atom_names.append(str(nm))
        el = ""
        if callable(infer_element_fn):
            try:
                el = str(infer_element_fn(nm, sy) or "").strip()
            except Exception:
                el = ""
        if not el:
            el = sy if sy else (nm[:1] if nm else "")
        elements.append(str(el).upper())
    return {"atom_names": list(atom_names), "elements": list(elements)}


def leadopt_build_nonbonded_particle_data(
    atoms,
    elements,
    atom_names,
    ligand_ids,
    ligand_charge_map=None,
    nb_params_for_element_fn=None,
    mass_for_element_fn=None,
    estimate_charge_fn=None,
):
    atom_list = [] if atoms is None else list(atoms)
    n_atoms = int(len(atom_list))
    elems = [] if elements is None else list(elements)
    names = [] if atom_names is None else list(atom_names)
    lig_ids = [] if ligand_ids is None else list(ligand_ids)
    lig_set = set(int(i) for i in lig_ids)
    lig_charge = {int(k): float(v) for k, v in dict(ligand_charge_map or {}).items()}
    sigmas = np.zeros(n_atoms, dtype=float)
    epsilons = np.zeros(n_atoms, dtype=float)
    charges = np.zeros(n_atoms, dtype=float)
    masses = np.zeros(n_atoms, dtype=float)
    local_infos = []
    for i, atom in enumerate(atom_list):
        el = str(elems[i] if i < len(elems) else "").upper()
        nm = str(names[i] if i < len(names) else "")
        if callable(nb_params_for_element_fn):
            try:
                sig, eps = nb_params_for_element_fn(el)
            except Exception:
                sig, eps = (0.32, 0.10)
        else:
            sig, eps = (0.32, 0.10)
        sigmas[int(i)] = float(sig)
        epsilons[int(i)] = float(eps)
        if callable(mass_for_element_fn):
            try:
                mass_i = float(mass_for_element_fn(el))
            except Exception:
                mass_i = 12.0
        else:
            mass_i = 12.0
        masses[int(i)] = float(max(1.0, mass_i))
        info = {
            "name": str(nm),
            "resn": str(getattr(atom, "resn", "")).strip(),
            "chain": str(getattr(atom, "chain", "")).strip(),
            "resi": str(getattr(atom, "resi", "")).strip(),
            "element": str(el),
            "origin": "ligand" if int(i) in lig_set else "protein",
            "is_cation": False,
            "is_anion": False,
            "charge_partial": 0.0,
            "charge_formal": 0,
        }
        local_infos.append(info)
        if int(i) in lig_set and int(i) in lig_charge:
            q = float(lig_charge[int(i)])
        elif callable(estimate_charge_fn):
            try:
                q = float(estimate_charge_fn(info))
            except Exception:
                q = 0.0
        else:
            q = 0.0
        charges[int(i)] = float(np.clip(float(q), -2.0, 2.0))
    return {
        "sigmas": sigmas,
        "epsilons": epsilons,
        "charges": charges,
        "masses": masses,
        "local_infos": list(local_infos),
    }


def leadopt_heavy_atom_ids(ids, elements):
    out = []
    elems = [] if elements is None else list(elements)
    id_list = [] if ids is None else list(ids)
    for i in id_list:
        try:
            ii = int(i)
        except Exception:
            continue
        if ii < 0 or ii >= len(elems):
            continue
        if str(elems[ii]).upper() != "H":
            out.append(int(ii))
    return out


def leadopt_pick_protein_eval_ids(
    coords_nm,
    protein_ids,
    ligand_ids_for_selection,
    select_local_interacting_fn=None,
    cutoff_nm=0.90,
    min_keep=16,
):
    _ = coords_nm
    prot_list = [] if protein_ids is None else list(protein_ids)
    lig_list = [] if ligand_ids_for_selection is None else list(ligand_ids_for_selection)
    prot = [int(i) for i in prot_list]
    lig = [int(i) for i in lig_list]
    picked = []
    if callable(select_local_interacting_fn):
        try:
            picked = list(
                select_local_interacting_fn(
                    coords_nm,
                    prot,
                    lig,
                    cutoff_nm=float(cutoff_nm),
                )
                or []
            )
        except Exception:
            picked = []
    picked = [int(i) for i in picked]
    if len(picked) < int(max(0, min_keep)):
        picked = list(prot)
    return sorted(set(int(i) for i in picked))


def leadopt_build_openmm_refinement_system(
    omm_mod,
    omm_unit,
    masses,
    sigmas,
    epsilons,
    charges,
    protein_ids,
    ligand_ids,
    protein_eval,
    coords_nm,
    elements,
    local_infos,
    param_prmtop,
    inject_ligand_ff_terms_fn=None,
    cutoff_nm=1.2,
    k_prot=2600.0,
    k_lig=1800.0,
):
    out = {
        "ok": False,
        "reason": "",
        "system": None,
        "ff_info": {},
    }
    if omm_mod is None or omm_unit is None:
        out["reason"] = "openmm_missing"
        return out
    if not callable(inject_ligand_ff_terms_fn):
        out["reason"] = "inject_ligand_ff_terms_fn_missing"
        return out
    try:
        lig_ids = [] if ligand_ids is None else list(ligand_ids)
        prot_eval = [] if protein_eval is None else list(protein_eval)
        prot_ids = [] if protein_ids is None else list(protein_ids)
        masses_list = [] if masses is None else list(masses)
        local_info_list = [] if local_infos is None else list(local_infos)
        system = omm_mod.System()
        for mass_i in masses_list:
            system.addParticle(float(max(1.0, float(mass_i))) * omm_unit.dalton)
        ok_ff, ff_info = inject_ligand_ff_terms_fn(
            system,
            lig_ids,
            local_info_list,
            str(param_prmtop or ""),
            local_coords_nm=np.asarray(coords_nm, dtype=float),
        )
        if not bool(ok_ff):
            reason_ff = ff_info.get("reason", "unknown") if isinstance(ff_info, dict) else str(ff_info)
            out["reason"] = "parameterized_prmtop_mismatch: " + str(reason_ff)
            return out
        if isinstance(ff_info, dict):
            out["ff_info"] = dict(ff_info)
        cutoff = float(cutoff_nm)
        lj_force = omm_mod.CustomNonbondedForce(
            "4*epsilon*((sigma/r)^12-(sigma/r)^6);"
            "sigma=0.5*(sigma1+sigma2);"
            "epsilon=sqrt(epsilon1*epsilon2)"
        )
        lj_force.addPerParticleParameter("sigma")
        lj_force.addPerParticleParameter("epsilon")
        for i in range(int(len(sigmas))):
            lj_force.addParticle([float(sigmas[i]), float(max(1e-6, epsilons[i]))])
        lj_force.setNonbondedMethod(omm_mod.CustomNonbondedForce.CutoffNonPeriodic)
        lj_force.setCutoffDistance(cutoff * omm_unit.nanometer)
        lj_force.setUseLongRangeCorrection(False)
        lj_force.addInteractionGroup(set(int(i) for i in prot_eval), set(int(i) for i in lig_ids))
        lj_force.setForceGroup(1)
        system.addForce(lj_force)
        coul_pref = 138.935456 / 8.0
        coul_force = omm_mod.CustomNonbondedForce(f"{coul_pref:.9f}*(q1*q2)/r")
        coul_force.addPerParticleParameter("q")
        for i in range(int(len(charges))):
            coul_force.addParticle([float(charges[i])])
        coul_force.setNonbondedMethod(omm_mod.CustomNonbondedForce.CutoffNonPeriodic)
        coul_force.setCutoffDistance(cutoff * omm_unit.nanometer)
        coul_force.setUseLongRangeCorrection(False)
        coul_force.addInteractionGroup(set(int(i) for i in prot_eval), set(int(i) for i in lig_ids))
        coul_force.setForceGroup(2)
        system.addForce(coul_force)
        restr = omm_mod.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restr.addPerParticleParameter("x0")
        restr.addPerParticleParameter("y0")
        restr.addPerParticleParameter("z0")
        restr.addPerParticleParameter("k")
        xyz = np.asarray(coords_nm, dtype=float)
        elems = [] if elements is None else list(elements)
        for i in prot_ids:
            ii = int(i)
            x0, y0, z0 = xyz[ii]
            kk = float(k_prot if str(elems[ii]).upper() != "H" else (0.7 * float(k_prot)))
            restr.addParticle(ii, [float(x0), float(y0), float(z0), float(kk)])
        for i in lig_ids:
            ii = int(i)
            x0, y0, z0 = xyz[ii]
            restr.addParticle(ii, [float(x0), float(y0), float(z0), float(k_lig)])
        restr.setForceGroup(3)
        system.addForce(restr)
        out["ok"] = True
        out["system"] = system
        return out
    except Exception as e:
        out["reason"] = str(e)
        return out


def leadopt_run_openmm_refinement_sampling(
    omm_mod,
    omm_unit,
    system,
    coords_nm,
    protein_eval,
    ligand_ids,
    sigmas,
    epsilons,
    charges,
    protein_heavy,
    ligand_heavy,
    temperature_k=300.0,
    equil_steps=800,
    prod_steps=4800,
    sample_interval=24,
    minimize_max_iters=800,
    cutoff_nm=1.2,
    auto_pick_platform_fn=None,
    compute_pairwise_energy_fn=None,
):
    out = {
        "ok": False,
        "reason": "",
        "platform": "",
        "energies": [],
        "dmins": [],
    }
    if omm_mod is None or omm_unit is None:
        out["reason"] = "openmm_missing"
        return out
    if system is None:
        out["reason"] = "openmm_system_missing"
        return out
    if not callable(compute_pairwise_energy_fn):
        out["reason"] = "compute_pairwise_energy_fn_missing"
        return out
    context = None
    integrator = None
    try:
        prot_eval = [] if protein_eval is None else list(protein_eval)
        lig_ids = [] if ligand_ids is None else list(ligand_ids)
        p_heavy = np.asarray([] if protein_heavy is None else list(protein_heavy), dtype=int)
        l_heavy = np.asarray([] if ligand_heavy is None else list(ligand_heavy), dtype=int)
        platform = None
        properties = {}
        picked = "default"
        if callable(auto_pick_platform_fn):
            try:
                platform, properties, picked = auto_pick_platform_fn()
            except Exception:
                platform, properties, picked = None, {}, "default"
        integrator = omm_mod.LangevinMiddleIntegrator(
            float(max(1.0, temperature_k)) * omm_unit.kelvin,
            1.0 / omm_unit.picosecond,
            0.0015 * omm_unit.picoseconds,
        )
        context = None
        if platform is not None:
            # a GPU that rejects the requested properties (or fails to start) used to fail the whole refinement:
            # retry it without properties, then the CPU, and say so
            for props_try in ([dict(properties or {})] if properties else []) + [{}]:
                try:
                    context = omm_mod.Context(system, integrator, platform, props_try)
                    out["platform"] = str(picked)
                    break
                except Exception as exc:
                    out.setdefault("platform_notes", []).append(f"{picked} {props_try or 'default'}: {exc}")
        if context is None and platform is not None and str(picked) != "CPU":
            try:
                context = omm_mod.Context(system, integrator, omm_mod.Platform.getPlatformByName("CPU"))
                out["platform"] = "CPU"
                out.setdefault("platform_notes", []).append(f"{picked} failed; refinement ran on the CPU")
            except Exception:
                context = None
        if context is None:
            context = omm_mod.Context(system, integrator)
            out["platform"] = "default"
        xyz0 = np.asarray(coords_nm, dtype=float)
        context.setPositions(xyz0 * omm_unit.nanometer)
        try:
            omm_mod.LocalEnergyMinimizer.minimize(
                context,
                5.0 * omm_unit.kilojoule_per_mole,
                int(max(1, minimize_max_iters)),
            )
        except Exception:
            pass
        context.setVelocitiesToTemperature(float(max(1.0, temperature_k)) * omm_unit.kelvin)
        try:
            integrator.step(int(max(50, equil_steps)))
        except Exception:
            integrator.step(200)
        energies = []
        dmins = []
        n_prod = int(max(240, prod_steps))
        n_stride = int(max(8, sample_interval))
        n_blocks = int(max(1, n_prod // n_stride))
        for _ in range(n_blocks):
            integrator.step(n_stride)
            st = context.getState(getPositions=True)
            try:
                xyz_nm = np.array(st.getPositions(asNumpy=True).value_in_unit(omm_unit.nanometer), dtype=float)
            except Exception:
                xyz_nm = np.array(st.getPositions().value_in_unit(omm_unit.nanometer), dtype=float)
            e_lj, e_c = compute_pairwise_energy_fn(
                xyz_nm,
                prot_eval,
                lig_ids,
                sigmas,
                epsilons,
                charges,
                cutoff_nm=float(cutoff_nm),
            )
            e_tot = float(e_lj + e_c)
            if np.isfinite(e_tot):
                energies.append(float(e_tot))
            dmin_a = np.nan
            try:
                if p_heavy.size > 0 and l_heavy.size > 0:
                    dv = xyz_nm[p_heavy][:, None, :] - xyz_nm[l_heavy][None, :, :]
                    rij = np.linalg.norm(dv, axis=2)
                    if rij.size > 0:
                        dmin_a = float(np.min(rij) * 10.0)
            except Exception:
                dmin_a = np.nan
            dmins.append(dmin_a)
        out["ok"] = True
        out["energies"] = list(energies)
        out["dmins"] = list(dmins)
        return out
    except Exception as e:
        out["reason"] = str(e)
        return out
    finally:
        if context is not None:
            try:
                del context
            except Exception:
                pass
        if integrator is not None:
            try:
                del integrator
            except Exception:
                pass


def make_leadopt_advanced_refinement_report():
    return {
        "requested": 0,
        "attempted": 0,
        "refined": 0,
        "cancelled": False,
        "method": "",
        "message": "",
    }


def leadopt_select_advanced_refinement_rows(results, top_n=5):
    out = {"rows": [], "requested": 0, "message": ""}
    rows = []
    for i, h in enumerate(list(results or [])):
        try:
            ok = str((h or {}).get("status", "")).lower() == "ok"
            mpo = float((h or {}).get("mpo", np.nan))
            if ok and np.isfinite(mpo):
                rows.append((int(i), h))
        except Exception:
            continue
    if len(rows) <= 0:
        out["message"] = "Advanced refinement skipped: no successful docked candidates to refine."
        return out
    try:
        n_pick = int(max(1, min(int(top_n), len(rows))))
    except Exception:
        n_pick = int(len(rows))
    rows = rows[:n_pick]
    out["rows"] = list(rows)
    out["requested"] = int(n_pick)
    return out


def leadopt_needs_parameterization_build(rows, file_exists_fn=None):
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    for _idx, hh in list(rows or []):
        h = dict(hh or {})
        pm = str(h.get("param_mol2", "") or h.get("parameterized_mol2", "")).strip()
        pp = str(h.get("param_prmtop", "") or h.get("parameterized_prmtop", "")).strip()
        if not (pm and pp and bool(exists_fn(pm)) and bool(exists_fn(pp))):
            return True
    return False


def leadopt_apply_refinement_result(results, idx, ref):
    out = {
        "refined": False,
        "method": "",
        "failure_reason": "",
    }
    if not isinstance(results, list):
        out["failure_reason"] = "results_not_list"
        return out
    try:
        ii = int(idx)
    except Exception:
        out["failure_reason"] = "index_invalid"
        return out
    if ii < 0 or ii >= len(results):
        out["failure_reason"] = "index_out_of_bounds"
        return out
    row = results[ii]
    if not isinstance(row, dict):
        out["failure_reason"] = "result_row_not_dict"
        return out
    ref_payload = dict(ref or {})
    for k, v in ref_payload.items():
        try:
            row[k] = v
        except Exception:
            pass
    try:
        refined = str(ref_payload.get("refine_status", "")).lower() == "ok" and np.isfinite(
            float(ref_payload.get("refine_dg_kj", np.nan))
        )
    except Exception:
        refined = False
    out["refined"] = bool(refined)
    method = str(ref_payload.get("refine_method", "") or "").strip()
    if refined and method:
        out["method"] = str(method)
    if not refined:
        out["failure_reason"] = str(ref_payload.get("refine_reason", "") or "unknown_refine_failure")
    try:
        mpo_raw = float(row.get("mpo_dock", row.get("mpo", np.nan)))
        dg_ref = float(row.get("refine_dg_kj", np.nan))
        row["mpo_dock"] = mpo_raw
        if np.isfinite(mpo_raw) and np.isfinite(dg_ref):
            dg_term = float(1.0 / (1.0 + np.exp((dg_ref + 25.0) / 6.0)))
            mpo_new = float(0.70 * mpo_raw + 0.30 * dg_term)
            row["mpo_refine_term"] = dg_term
            row["mpo"] = mpo_new
            note0 = str(row.get("notes", "") or "").strip()
            rnote = (
                f"refine_dG={dg_ref:.2f} kJ/mol"
                f" ({str(row.get('refine_method', ''))}, n={int(row.get('refine_samples', 0))})"
            )
            row["notes"] = (note0 + "; " + rnote) if note0 else rnote
    except Exception:
        pass
    return out


def leadopt_finalize_advanced_refinement_report(report, methods=None, fail_reasons=None):
    rep = dict(report or {})
    methods_list = [str(x).strip() for x in list(methods or []) if str(x).strip()]
    if methods_list:
        rep["method"] = ",".join(sorted(set(methods_list)))
    attempted = int(rep.get("attempted", 0) or 0)
    refined = int(rep.get("refined", 0) or 0)
    msg = str(rep.get("message", "") or "").strip()
    if attempted > 0 and refined <= 0 and (not msg):
        fails = [str(x).strip() for x in list(fail_reasons or []) if str(x).strip()]
        if fails:
            rc = collections.Counter(fails)
            top = [f"{k} (n={v})" for k, v in rc.most_common(3)]
            rep["message"] = (
                "Advanced refinement ran, but no candidate produced a valid PyMBAR estimate. "
                "Top failure reasons: " + "; ".join(top)
            )
        else:
            rep["message"] = "Advanced refinement ran, but no candidate produced a valid PyMBAR estimate."
    return rep


def process_leadopt_entry(
    item,
    idx,
    *,
    smina_bin,
    receptor_input,
    center,
    size,
    exhaustiveness,
    timeout_lig,
    work_dir,
    lig_dir,
    pose_dir,
    target_ph,
    safe_name_fn,
    write_sdf,
    prepare_with_meeko,
    fallback_prepare_ligand,
    run_command,
    parse_pose_scores,
    convert_pose_for_view,
    mpo_fn,
):
    rec = dict(item or {})
    name = str(rec.get("name", f"Analog_{int(idx)+1}"))
    safe = str(safe_name_fn(name) if callable(safe_name_fn) else name).strip()[:48]
    stem = f"{int(idx+1):05d}_{safe}" if safe else f"{int(idx+1):05d}_lead"
    lig_sdf = os.path.join(str(lig_dir), f"{stem}.sdf")
    out_pose = os.path.join(str(pose_dir), f"{stem}_pose.pdbqt")
    out_log = os.path.join(str(pose_dir), f"{stem}.log")

    prep_mode = "meeko"
    prep_note = ""
    ligand_input = ""
    status = "ok"
    err_tail = ""
    affinity = float("nan")
    pose_view = ""

    ok_w, msg_w = write_sdf(rec.get("mol"), lig_sdf, name)
    if not bool(ok_w):
        status = "failed"
        prep_note = f"SDF write failed: {msg_w}"

    if status == "ok":
        meeko_out = os.path.join(str(lig_dir), f"{stem}.pdbqt")
        ok_meeko, msg_meeko = prepare_with_meeko(rec.get("mol"), meeko_out)
        if ok_meeko:
            ligand_input = str(meeko_out)
            prep_mode = "meeko"
            prep_note = str(msg_meeko or "")
        else:
            prep_mode = "fallback"
            prep_note = str(msg_meeko or "")
            fallback_job = os.path.join(str(lig_dir), f"job_{stem}")
            fallback_in = ""
            lig_info = {}
            try:
                fallback_in, lig_info = fallback_prepare_ligand(lig_sdf, fallback_job, rec, float(target_ph))
            except TypeError:
                fallback_in, lig_info = fallback_prepare_ligand(lig_sdf, fallback_job, float(target_ph))
            if fallback_in and os.path.isfile(str(fallback_in)):
                ligand_input = str(fallback_in)
                try:
                    extra = lig_info.get("notes") if isinstance(lig_info, dict) else lig_info
                except Exception:
                    extra = ""
                prep_note = f"{prep_note}; fallback={extra}"
            else:
                status = "failed"
                prep_note = f"{prep_note}; fallback ligand prep failed"

    if status == "ok":
        cmdline = [
            str(smina_bin),
            "-r",
            str(receptor_input),
            "-l",
            str(ligand_input),
            "--center_x",
            f"{float(center[0]):.4f}",
            "--center_y",
            f"{float(center[1]):.4f}",
            "--center_z",
            f"{float(center[2]):.4f}",
            "--size_x",
            f"{float(size[0]):.4f}",
            "--size_y",
            f"{float(size[1]):.4f}",
            "--size_z",
            f"{float(size[2]):.4f}",
            "--exhaustiveness",
            str(int(exhaustiveness)),
            "--num_modes",
            "1",
            "--energy_range",
            "4.0",
            "--min_rmsd_filter",
            "1.0",
            "--out",
            str(out_pose),
            "--log",
            str(out_log),
        ]
        ok, _out, err, rc = run_command(cmdline, cwd=str(work_dir), timeout=max(30, int(timeout_lig)))
        err_tail = str((err or "")[-1200:])
        if (not bool(ok)) or (not os.path.isfile(out_pose)):
            status = "failed"
            prep_note = f"{prep_note}; smina rc={int(rc)}"
        else:
            poses = list(parse_pose_scores(out_pose, out_log) or [])
            if poses:
                _best_pose, affinity = _pick_best_affinity_pose(poses)
            atom_rows = 0
            try:
                with open(out_pose, "r", encoding="utf-8", errors="ignore") as fh_pose:
                    for ln in fh_pose:
                        if ln.startswith("ATOM") or ln.startswith("HETATM"):
                            atom_rows += 1
            except Exception:
                atom_rows = 0
            if (not poses) or (not np.isfinite(float(affinity))) or int(atom_rows) <= 0:
                status = "failed"
                prep_note = f"{prep_note}; no_scored_pose_parsed"
            else:
                pose_view = str(convert_pose_for_view(out_pose, pose_dir, stem) or out_pose)

    mpo = mpo_fn(
        affinity,
        rec.get("admet", np.nan),
        rec.get("qed", np.nan),
        rec.get("sim_seed", np.nan),
    )
    return {
        "name": str(name),
        "source": str(rec.get("source", "")),
        "smiles": str(rec.get("smiles", "")),
        "affinity": float(affinity) if np.isfinite(float(affinity)) else np.nan,
        "mpo": float(mpo) if np.isfinite(float(mpo)) else np.nan,
        "admet": float(rec.get("admet", np.nan)),
        "qed": float(rec.get("qed", np.nan)),
        "sim_seed": float(rec.get("sim_seed", np.nan)),
        "mw": float(rec.get("mw", np.nan)),
        "logp": float(rec.get("logp", np.nan)),
        "hba": int(rec.get("hba", 0)) if np.isfinite(float(rec.get("hba", np.nan))) else np.nan,
        "hbd": int(rec.get("hbd", 0)) if np.isfinite(float(rec.get("hbd", np.nan))) else np.nan,
        "tpsa": float(rec.get("tpsa", np.nan)),
        "rotb": int(rec.get("rotb", 0)) if np.isfinite(float(rec.get("rotb", np.nan))) else np.nan,
        "pains": bool(rec.get("pains", False)),
        "filter_keep": bool(rec.get("filter_keep", False)),
        "filter_reason": str(rec.get("filter_reason", "pass")),
        "prep": str(prep_mode),
        "status": str(status),
        "notes": str(prep_note),
        "pose_file": str(out_pose),
        "pose_view_file": str(pose_view),
        "stderr_tail": str(err_tail),
        "mol": rec.get("mol", None),
        "param_mol2": "",
        "param_frcmod": "",
        "param_prmtop": "",
        "param_inpcrd": "",
        "refine_status": "",
        "refine_method": "",
        "refine_reason": "",
        "refine_dg_kj": np.nan,
        "refine_dg_err_kj": np.nan,
        "refine_ci_low_kj": np.nan,
        "refine_ci_high_kj": np.nan,
        "refine_samples": 0,
        "refine_n_eff": 0,
        "refine_platform": "",
        "mpo_dock": float(mpo) if np.isfinite(float(mpo)) else np.nan,
    }


def run_leadopt_entries(
    entries,
    *,
    smina_bin,
    receptor_input,
    center,
    size,
    exhaustiveness,
    timeout_lig,
    work_dir,
    lig_dir,
    pose_dir,
    target_ph,
    safe_name_fn,
    write_sdf,
    prepare_with_meeko,
    fallback_prepare_ligand,
    run_command,
    parse_pose_scores,
    convert_pose_for_view,
    mpo_fn,
    on_progress=None,
):
    entries_list = list(entries or [])
    results = []
    cancelled = False
    total = int(len(entries_list))
    for i, item in enumerate(entries_list):
        if callable(on_progress):
            try:
                keep_running = bool(on_progress(int(i), int(total), item))
            except Exception:
                keep_running = True
            if not keep_running:
                cancelled = True
                break
        row = process_leadopt_entry(
            item,
            i,
            smina_bin=smina_bin,
            receptor_input=receptor_input,
            center=center,
            size=size,
            exhaustiveness=exhaustiveness,
            timeout_lig=timeout_lig,
            work_dir=work_dir,
            lig_dir=lig_dir,
            pose_dir=pose_dir,
            target_ph=target_ph,
            safe_name_fn=safe_name_fn,
            write_sdf=write_sdf,
            prepare_with_meeko=prepare_with_meeko,
            fallback_prepare_ligand=fallback_prepare_ligand,
            run_command=run_command,
            parse_pose_scores=parse_pose_scores,
            convert_pose_for_view=convert_pose_for_view,
            mpo_fn=mpo_fn,
        )
        results.append(row)
    return results, bool(cancelled)


def leadopt_sort_key(rec):
    try:
        ok = str((rec or {}).get("status", "")).lower() == "ok"
        mpo = float((rec or {}).get("mpo", np.nan))
        aff = float((rec or {}).get("affinity", np.nan))
        dg_ref = float((rec or {}).get("refine_dg_kj", np.nan))
        good = ok and np.isfinite(mpo)
        if good and np.isfinite(dg_ref):
            return (0, 0, float(dg_ref), -float(mpo), float(aff) if np.isfinite(aff) else np.inf)
        if good:
            return (0, 1, np.inf, -float(mpo), float(aff) if np.isfinite(aff) else np.inf)
        return (1, 2, np.inf, np.inf, np.inf)
    except Exception:
        return (2, 2, np.inf, np.inf, np.inf)


def rank_leadopt_results(results):
    arr = sorted(list(results or []), key=leadopt_sort_key)
    rank = 1
    for rec in arr:
        try:
            ok = str((rec or {}).get("status", "")).lower() == "ok"
            if ok and np.isfinite(float((rec or {}).get("mpo", np.nan))):
                rec["rank"] = int(rank)
                rank += 1
            else:
                rec["rank"] = None
        except Exception:
            rec["rank"] = None
    return arr


def find_best_leadopt_result(results):
    for rec in list(results or []):
        try:
            if str((rec or {}).get("status", "")).lower() != "ok":
                continue
            mpo = float((rec or {}).get("mpo", np.nan))
            if np.isfinite(mpo):
                return rec
        except Exception:
            continue
    return None


def build_leadopt_summary_lines(
    *,
    seed_name,
    n_analogs,
    n_total,
    n_ok,
    n_fail,
    n_ref_ok,
    n_ref_attempted,
    ref_method="",
    filter_mode="strict",
    best=None,
    refine_note="",
    cancelled=False,
):
    lines = [
        "Lead optimization completed.",
        f"Seed: {str(seed_name)}",
        f"Generated analogs: {int(n_analogs)}",
        f"Docking attempts: {int(n_total)}",
        f"Docked/scored: {int(n_ok)}",
        f"Failed docking: {int(n_fail)}",
        f"Advanced refined: {int(n_ref_ok)}/{int(n_ref_attempted)}" + (f" ({str(ref_method)})" if str(ref_method).strip() else ""),
    ]
    if "diversity_rescue" in str(filter_mode):
        lines.append("Selection mode: strict + diversity rescue for docking")
    elif str(filter_mode) != "strict":
        lines.append(f"Selection mode: {str(filter_mode)}")
    if isinstance(best, dict):
        try:
            lines.append(
                f"Top lead: {best.get('name')} | MPO={float(best.get('mpo', np.nan)):.3f} | "
                f"affinity={float(best.get('affinity', np.nan)):.3f} kcal/mol"
            )
        except Exception:
            lines.append(f"Top lead: {best.get('name')}")
    if str(refine_note).strip():
        lines.append("")
        lines.append(str(refine_note).strip())
    if bool(cancelled):
        lines.append("Run was cancelled before processing all candidates.")
    return lines


def leadopt_build_candidate_rows(analogs, compute_props_fn=None, admet_score_fn=None):
    rows = []
    props_fn = compute_props_fn if callable(compute_props_fn) else None
    admet_fn = admet_score_fn if callable(admet_score_fn) else None
    for c in list(analogs or []):
        item = dict(c or {})
        props = {}
        if callable(props_fn):
            try:
                props = dict(props_fn(item.get("mol", None)) or {})
            except Exception:
                props = {}
        admet = np.nan
        if callable(admet_fn):
            try:
                admet = float(admet_fn(props))
            except Exception:
                admet = np.nan
        row = dict(item)
        row.update(props)
        row["admet"] = float(admet) if np.isfinite(float(admet)) else np.nan
        rows.append(row)
    return rows


def leadopt_optional_refine_note(wants, refine_report=None):
    if not bool(wants):
        return ""
    rep = dict(refine_report or {})
    msg = str(rep.get("message", "")).strip() if isinstance(rep, dict) else ""
    if msg:
        return msg
    tried = int(rep.get("attempted", 0) or 0) if isinstance(rep, dict) else 0
    ok = int(rep.get("refined", 0) or 0) if isinstance(rep, dict) else 0
    method = str(rep.get("method", "") or "") if isinstance(rep, dict) else ""
    if tried <= 0:
        return "Advanced refinement requested, but no eligible leads were available."
    line = f"Advanced refinement: {ok}/{tried} top leads refined with OpenMM + PyMBAR"
    if method:
        line += f" [{method}]"
    if bool(rep.get("cancelled", False)):
        line += " (cancelled early)"
    line += "."
    return line


def leadopt_make_filtered_out_row(cand):
    c = dict(cand or {})
    return {
        "name": str(c.get("name", "")),
        "source": str(c.get("source", "")),
        "smiles": str(c.get("smiles", "")),
        "affinity": np.nan,
        "mpo": np.nan,
        "admet": float(c.get("admet", np.nan)),
        "qed": float(c.get("qed", np.nan)),
        "sim_seed": float(c.get("sim_seed", np.nan)),
        "mw": float(c.get("mw", np.nan)),
        "logp": float(c.get("logp", np.nan)),
        "hba": int(c.get("hba", 0)) if np.isfinite(float(c.get("hba", np.nan))) else np.nan,
        "hbd": int(c.get("hbd", 0)) if np.isfinite(float(c.get("hbd", np.nan))) else np.nan,
        "tpsa": float(c.get("tpsa", np.nan)),
        "rotb": int(c.get("rotb", 0)) if np.isfinite(float(c.get("rotb", np.nan))) else np.nan,
        "pains": bool(c.get("pains", False)),
        "filter_keep": bool(c.get("filter_keep", False)),
        "filter_reason": str(c.get("filter_reason", "unknown")),
        "prep": "",
        "status": "filtered_out",
        "notes": "not docked (filtered)",
        "pose_file": "",
        "pose_view_file": "",
        "stderr_tail": "",
        "mol": c.get("mol", None),
        "param_mol2": "",
        "param_frcmod": "",
        "param_prmtop": "",
        "param_inpcrd": "",
        "refine_status": "",
        "refine_method": "",
        "refine_reason": "",
        "refine_dg_kj": np.nan,
        "refine_dg_err_kj": np.nan,
        "refine_ci_low_kj": np.nan,
        "refine_ci_high_kj": np.nan,
        "refine_samples": 0,
        "refine_n_eff": 0,
        "refine_platform": "",
        "mpo_dock": np.nan,
    }


def leadopt_merge_candidate_and_docked_results(candidate_rows, docked_results):
    docked = list(docked_results or [])
    cands = list(candidate_rows or [])
    result_by_name = {}
    for rr in list(docked):
        try:
            nm = str(rr.get("name", "")).strip()
        except Exception:
            nm = ""
        if nm:
            result_by_name[nm] = dict(rr)
    full_results = []
    seen_names = set()
    for cand in cands:
        try:
            nm = str(cand.get("name", "")).strip()
        except Exception:
            nm = ""
        if nm and (nm in result_by_name):
            rr = dict(result_by_name.get(nm, {}))
            rr["filter_keep"] = bool(cand.get("filter_keep", rr.get("filter_keep", False)))
            rr["filter_reason"] = str(cand.get("filter_reason", rr.get("filter_reason", "pass")))
            full_results.append(rr)
            seen_names.add(nm)
        else:
            full_results.append(leadopt_make_filtered_out_row(cand))
            if nm:
                seen_names.add(nm)
    for nm, rr in result_by_name.items():
        if nm not in seen_names:
            full_results.append(dict(rr))
    return full_results


def leadopt_compute_result_counts(docked_results, refine_report=None):
    docked = list(docked_results or [])
    n_total = int(len(docked))
    n_ok = int(
        sum(
            1
            for h in docked
            if str((h or {}).get("status", "")).lower() == "ok" and np.isfinite(float((h or {}).get("mpo", np.nan)))
        )
    )
    n_fail = int(n_total - n_ok)
    rep = dict(refine_report or {})
    n_ref_attempted = int(rep.get("attempted", 0) or 0)
    n_ref_ok = int(rep.get("refined", 0) or 0)
    ref_method = str(rep.get("method", "") or "")
    return {
        "n_total": int(n_total),
        "n_ok": int(n_ok),
        "n_fail": int(n_fail),
        "n_ref_attempted": int(n_ref_attempted),
        "n_ref_ok": int(n_ref_ok),
        "ref_method": str(ref_method),
    }


def leadopt_status_text(best, n_ok, n_total):
    try:
        if isinstance(best, dict):
            return (
                f"Status: done ({int(n_ok)}/{int(n_total)} scored). Best: {best.get('name')} | "
                f"MPO={float(best.get('mpo', np.nan)):.3f} | affinity={float(best.get('affinity', np.nan)):.3f}"
            )
    except Exception:
        pass
    return f"Status: done ({int(n_ok)}/{int(n_total)} scored)."


def leadopt_filter_rows_with_limits(rows, limits, filter_fail_reason_fn=None):
    kept = []
    rejects = {}
    reason_fn = filter_fail_reason_fn if callable(filter_fail_reason_fn) else None
    for rr in list(rows or []):
        reason = ""
        if callable(reason_fn):
            try:
                reason = str(reason_fn(rr, limits=limits) or "")
            except Exception:
                reason = "invalid_props"
        rowx = dict(rr or {})
        rowx["filter_keep"] = (reason == "")
        rowx["filter_reason"] = (reason if reason else "pass")
        if rowx["filter_keep"]:
            kept.append(rowx)
        else:
            rejects[reason] = int(rejects.get(reason, 0)) + 1
    return kept, rejects


def leadopt_rescue_top_admet_rows(candidate_rows):
    def _safe(v, default=-1e9):
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default
    rows = list(candidate_rows or [])
    rescue_n = int(np.clip(min(15, max(5, len(rows))), 1, max(1, len(rows))))
    rescue = sorted(
        rows,
        key=lambda r: (-_safe((r or {}).get("admet", np.nan)), -_safe((r or {}).get("qed", np.nan)), -_safe((r or {}).get("sim_seed", np.nan))),
    )[:rescue_n]
    filtered = []
    for rr in rescue:
        rowx = dict(rr or {})
        rowx["filter_keep"] = False
        rowx["filter_reason"] = "rescue_top_admet"
        filtered.append(rowx)
    reject_counts = {"rescue": int(max(0, len(rows) - len(filtered)))}
    return filtered, reject_counts


def leadopt_apply_diversity_rescue_rows(
    candidate_rows,
    filtered_rows,
    has_rdkit=False,
    fingerprint_fn=None,
    similarity_fn=None,
    min_dock_n=None,
):
    selected = [dict(x) for x in list(filtered_rows or [])]
    rows = list(candidate_rows or [])
    if min_dock_n is None:
        min_dock_n = int(np.clip(max(8, int(0.20 * max(1, len(rows)))), 8, 40))
    try:
        min_dock_n = int(min_dock_n)
    except Exception:
        min_dock_n = 8
    min_dock_n = int(np.clip(min_dock_n, 1, 40))
    if len(selected) >= min_dock_n or len(rows) <= len(selected):
        return selected

    chosen_names = set()
    for x in selected:
        try:
            nm = str(x.get("name", "")).strip()
        except Exception:
            nm = ""
        if nm:
            chosen_names.add(nm)
    extras = []
    for rr in rows:
        try:
            nm = str((rr or {}).get("name", "")).strip()
        except Exception:
            nm = ""
        if nm and nm in chosen_names:
            continue
        extras.append(dict(rr or {}))

    def _safe(v, default=-1e9):
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    selected_fps = []
    if bool(has_rdkit) and callable(fingerprint_fn):
        for rr in list(selected):
            m0 = rr.get("mol", None)
            if m0 is None:
                continue
            try:
                fp0 = fingerprint_fn(m0)
            except Exception:
                fp0 = None
            if fp0 is not None:
                selected_fps.append(fp0)

    while len(selected) < min_dock_n and len(extras) > 0:
        best_i = 0
        best_score = -1e18
        best_fp = None
        for i_ex, rr in enumerate(extras):
            qual = (
                0.35 * _safe((rr or {}).get("admet", np.nan), 0.0)
                + 0.20 * _safe((rr or {}).get("qed", np.nan), 0.0)
                + 0.10 * _safe((rr or {}).get("sim_seed", np.nan), 0.0)
            )
            div_term = 0.0
            fp = None
            if bool(has_rdkit) and callable(fingerprint_fn):
                try:
                    fp = fingerprint_fn(rr.get("mol", None))
                    if fp is not None:
                        if selected_fps and callable(similarity_fn):
                            max_sim = max(float(similarity_fn(fp, sfp)) for sfp in selected_fps)
                            div_term = 1.0 - max_sim
                        else:
                            div_term = 1.0
                    else:
                        div_term = 0.15
                except Exception:
                    div_term = 0.15
            score = float(0.75 * div_term + qual)
            if score > best_score:
                best_score = score
                best_i = int(i_ex)
                best_fp = fp
        pick = extras.pop(best_i)
        pick["filter_keep"] = False
        pick["filter_reason"] = "diversity_rescue_dock"
        selected.append(pick)
        try:
            nm = str(pick.get("name", "")).strip()
        except Exception:
            nm = ""
        if nm:
            chosen_names.add(nm)
        if bool(has_rdkit) and (best_fp is not None):
            selected_fps.append(best_fp)
    return selected


def leadopt_filter_and_rescue_candidates(
    candidate_rows,
    base_limits,
    filter_fail_reason_fn=None,
    relax_limits_fn=None,
    relax_levels=(1, 2, 3),
    has_rdkit=False,
    fingerprint_fn=None,
    similarity_fn=None,
):
    rows = list(candidate_rows or [])
    filtered, reject_counts = leadopt_filter_rows_with_limits(rows, base_limits, filter_fail_reason_fn=filter_fail_reason_fn)
    filter_mode = "strict"

    if len(filtered) <= 0 and callable(relax_limits_fn):
        for lvl in list(relax_levels or []):
            try:
                lim = relax_limits_fn(base_limits, level=lvl)
            except Exception:
                continue
            cand_lvl, rej_lvl = leadopt_filter_rows_with_limits(rows, lim, filter_fail_reason_fn=filter_fail_reason_fn)
            if len(cand_lvl) > 0:
                filtered = list(cand_lvl)
                reject_counts = dict(rej_lvl)
                filter_mode = f"relaxed_l{int(lvl)}"
                break

    if len(filtered) <= 0:
        filtered, reject_counts = leadopt_rescue_top_admet_rows(rows)
        filter_mode = "rescue_top_admet"

    min_dock_n = int(np.clip(max(8, int(0.20 * max(1, len(rows)))), 8, 40))
    selected = leadopt_apply_diversity_rescue_rows(
        candidate_rows=rows,
        filtered_rows=filtered,
        has_rdkit=bool(has_rdkit),
        fingerprint_fn=fingerprint_fn,
        similarity_fn=similarity_fn,
        min_dock_n=min_dock_n,
    )
    if len(selected) > len(filtered):
        filtered = list(selected)
        if "diversity_rescue" not in str(filter_mode):
            filter_mode = (str(filter_mode) + "+diversity_rescue") if str(filter_mode) else "diversity_rescue"

    return {
        "filtered": list(filtered),
        "filter_mode": str(filter_mode),
        "reject_counts": dict(reject_counts),
        "min_dock_n": int(min_dock_n),
    }


def leadopt_prepare_job_paths(workspace_dir, receptor_obj, safe_name_fn=None, tag=None, makedirs_fn=None):
    safe_fn = safe_name_fn if callable(safe_name_fn) else (lambda x: str(x))
    mk_fn = makedirs_fn if callable(makedirs_fn) else os.makedirs
    run_tag = str(tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    job_dir = os.path.join(str(workspace_dir), f"leadopt_{safe_fn(receptor_obj)}_{run_tag}")
    lig_dir = os.path.join(job_dir, "ligands")
    pose_dir = os.path.join(job_dir, "poses")
    mk_fn(job_dir, exist_ok=True)
    mk_fn(lig_dir, exist_ok=True)
    mk_fn(pose_dir, exist_ok=True)
    return {
        "tag": str(run_tag),
        "job_dir": str(job_dir),
        "lig_dir": str(lig_dir),
        "pose_dir": str(pose_dir),
    }


def leadopt_export_csv_outputs(job_dir, candidate_rows, docked_results, full_results, open_fn=None):
    open_file = open_fn if callable(open_fn) else open
    cols = [
        "rank", "name", "source", "smiles", "status", "filter_keep", "filter_reason",
        "affinity", "mpo", "admet", "qed", "sim_seed", "mw", "logp", "hba", "hbd", "tpsa", "rotb", "pains",
        "prep", "notes", "pose_file", "pose_view_file",
        "refine_status", "refine_method", "refine_reason", "refine_dg_kj", "refine_dg_err_kj",
    ]
    csv_candidates = os.path.join(str(job_dir), "leadopt_candidates_all.csv")
    csv_docked = os.path.join(str(job_dir), "leadopt_results_docked.csv")
    csv_all = os.path.join(str(job_dir), "leadopt_results_all.csv")
    ccols = [
        "name", "source", "smiles", "filter_keep", "filter_reason", "admet",
        "qed", "sim_seed", "mw", "logp", "hba", "hbd", "tpsa", "rotb", "pains",
    ]

    def _write_rows(path_out, rows_in):
        with open_file(path_out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for rr in list(rows_in or []):
                row = {}
                for cc in cols:
                    row[cc] = (rr or {}).get(cc, "")
                writer.writerow(row)

    with open_file(csv_candidates, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ccols)
        writer.writeheader()
        for rr in list(candidate_rows or []):
            writer.writerow({k: (rr or {}).get(k, "") for k in ccols})

    _write_rows(csv_docked, docked_results)
    _write_rows(csv_all, full_results)

    return {
        "csv_candidates": str(csv_candidates),
        "csv_docked": str(csv_docked),
        "csv_all": str(csv_all),
    }
