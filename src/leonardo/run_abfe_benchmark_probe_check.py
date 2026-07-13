#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from openmm import (
    CustomExternalForce,
    CustomNonbondedForce,
    LangevinIntegrator,
    LangevinMiddleIntegrator,
    LocalEnergyMinimizer,
    System,
    app,
)
from openmm import unit as u
from openmm.app import DCDReporter, Modeller, PDBFile
from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdMolDescriptors

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize as RDMS
except Exception:
    RDMS = None

from test_abfe_3ptb_ben.run_abfe_3ptb_metal_biology_check import (
    build_context_cpu_only,
    contact_occupancies,
    population_free_energy_metrics,
    rank_pull_directions,
)
from test_abfe_3ptb_ben.run_abfe_fullff_3ptb import (
    estimate_protein_charge,
    infer_element,
    inject_ligand_terms_from_prmtop,
    load_mol2_charges,
    mass_for_element,
    nb_params,
)

from .abfe_runtime import (
    build_abfe_adaptive_steering_state_payload,
    build_abfe_adaptive_steering_update_decision,
    build_abfe_free_energy_numeric_payload,
    build_abfe_gbsa_mmgbsa_rescore_payload,
    build_abfe_steering_lambda_schedule,
    build_abfe_surface_detach_direction_payload,
    compute_runtime_pocket_descriptors,
)

R_KJ_MOL_K = 0.008314462618
KCAL_PER_KJ = 1.0 / 4.184
STANDARD_PROTEIN_RESIDUES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "ASH",
    "CYS",
    "CYM",
    "CYX",
    "GLN",
    "GLU",
    "GLH",
    "GLY",
    "HIS",
    "HID",
    "HIE",
    "HIP",
    "ILE",
    "LEU",
    "LYS",
    "LYN",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ACE",
    "NME",
}
WATER_RESIDUES = {"HOH", "WAT", "SOL"}


def kj_to_kcal(value):
    try:
        vv = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(vv):
        return np.nan
    return float(vv * KCAL_PER_KJ)


def format_energy(value_kj, unit="kcal", digits=3):
    unit_txt = str(unit or "kcal").strip().lower()
    try:
        digits_i = int(max(0, int(digits)))
    except Exception:
        digits_i = 3
    if unit_txt == "kj":
        try:
            vv = float(value_kj)
        except Exception:
            vv = np.nan
        return f"{float(vv):.{digits_i}f} kJ/mol" if np.isfinite(vv) else "nan kJ/mol"
    kcal = kj_to_kcal(value_kj)
    return f"{float(kcal):.{digits_i}f} kcal/mol" if np.isfinite(kcal) else "nan kcal/mol"


def _ordered_unique_str(values):
    out = []
    seen = set()
    for value in list(values or []):
        txt = str(value or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def _pick_reasonable_protomer_variants(candidates, reference_smiles=""):
    rows = []
    for smi in _ordered_unique_str(list(candidates or []) + ([reference_smiles] if reference_smiles else [])):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        q = int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))
        n_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
        n_total = int(len(n_atoms))
        n_pos = int(sum(1 for a in n_atoms if int(a.GetFormalCharge()) > 0))
        unsupported_n_pos = 0
        for atom in n_atoms:
            if int(atom.GetFormalCharge()) <= 0:
                continue
            supported = False
            if atom.GetIsAromatic() or int(atom.GetTotalDegree()) >= 4:
                supported = True
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if bond.GetBondType() == Chem.BondType.DOUBLE and other.GetAtomicNum() == 6:
                    supported = True
                    break
            if not supported:
                unsupported_n_pos += 1
        rows.append(
            {
                "smiles": str(smi),
                "formal_charge": int(q),
                "n_pos_n": int(n_pos),
                "unsupported_n_pos": int(unsupported_n_pos),
                "all_n_pos": bool(n_total >= 2 and n_pos == n_total),
            }
        )
    if not rows:
        return []
    charges = sorted(int(r["formal_charge"]) for r in rows)
    mid = len(charges) // 2
    target_q = charges[mid] if len(charges) % 2 == 1 else int(round((charges[mid - 1] + charges[mid]) / 2.0))
    rows = sorted(
        rows,
        key=lambda r: (
            int(r["unsupported_n_pos"]),
            int(bool(r["all_n_pos"])),
            abs(int(r["formal_charge"]) - int(target_q)),
            int(r["n_pos_n"]),
            abs(int(r["formal_charge"])),
            len(str(r["smiles"])),
        ),
    )
    return [str(r["smiles"]) for r in rows]


def try_dimorphite_protonate_smiles_variants(smiles, target_ph):
    try:
        import dimorphite_dl as dmd
    except Exception:
        return []
    smi = str(smiles or "").strip()
    if not smi:
        return []
    min_ph = max(0.0, float(target_ph) - 0.25)
    max_ph = min(14.0, float(target_ph) + 0.25)
    variants = []
    try:
        fn = getattr(dmd, "protonate_smiles", None)
        if callable(fn):
            variants.extend(list(fn(smi, ph_min=min_ph, ph_max=max_ph) or []))
        fn2 = getattr(dmd, "protonate", None)
        if callable(fn2):
            variants.extend(list(fn2(smi, min_ph=min_ph, max_ph=max_ph) or []))
        if hasattr(dmd, "DimorphiteDL"):
            eng = dmd.DimorphiteDL(min_ph=min_ph, max_ph=max_ph)
            for meth in ("protonate_smiles", "protonate"):
                mm = getattr(eng, meth, None)
                if callable(mm):
                    variants.extend(list(mm(smi) or []))
    except Exception:
        return []
    return _pick_reasonable_protomer_variants(variants, reference_smiles=smi)


def _copy_charges_from_template(ref_noh, target_noh):
    out = Chem.RWMol(target_noh)
    match = target_noh.GetSubstructMatch(ref_noh)
    if match and len(match) == ref_noh.GetNumAtoms():
        for ref_idx, target_idx in enumerate(match):
            out.GetAtomWithIdx(int(target_idx)).SetFormalCharge(int(ref_noh.GetAtomWithIdx(int(ref_idx)).GetFormalCharge()))
    return Chem.Mol(out)


def enumerate_ligand_state_candidates(mol, target_ph=7.4, max_states=3, max_tautomers=8):
    try:
        max_states_i = max(1, int(max_states))
    except Exception:
        max_states_i = 3
    try:
        max_tautomers_i = max(1, int(max_tautomers))
    except Exception:
        max_tautomers_i = 8
    base = Chem.RemoveHs(Chem.Mol(mol))
    try:
        Chem.SanitizeMol(base)
    except Exception:
        pass
    base_smiles = str(Chem.MolToSmiles(base, isomericSmiles=True) or "").strip()
    candidate_rows = []

    def _append_candidate(smiles, source):
        smi = str(smiles or "").strip()
        if not smi:
            return
        candidate_rows.append({"smiles": smi, "source": str(source)})

    _append_candidate(base_smiles, "input")
    tautomer_smiles = []
    if RDMS is not None:
        try:
            enumerator = RDMS.TautomerEnumerator()
            for tmol in list(enumerator.Enumerate(base) or [])[:max_tautomers_i]:
                tautomer_smiles.append(str(Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(tmol)), isomericSmiles=True) or "").strip())
        except Exception:
            tautomer_smiles = []
    for smi in tautomer_smiles:
        _append_candidate(smi, "tautomer")

    expanded_rows = []
    for row in list(candidate_rows):
        smi = str(row.get("smiles", "") or "").strip()
        src = str(row.get("source", "") or "input")
        expanded_rows.append({"smiles": smi, "source": src})
        for prot_smi in try_dimorphite_protonate_smiles_variants(smi, target_ph=target_ph):
            expanded_rows.append({"smiles": str(prot_smi), "source": f"{src}+protomer"})

    deduped = []
    seen = set()
    for row in expanded_rows:
        smi = str(row.get("smiles", "") or "").strip()
        if not smi or smi in seen:
            continue
        seen.add(smi)
        deduped.append(dict(row))

    source_rank = {
        "input": 0,
        "input+protomer": 1,
        "tautomer": 2,
        "tautomer+protomer": 3,
    }
    base_charge = int(sum(int(a.GetFormalCharge()) for a in base.GetAtoms()))
    scored = []
    for row in deduped:
        smi = str(row.get("smiles", "") or "").strip()
        ref = Chem.MolFromSmiles(smi)
        if ref is None:
            continue
        q = int(sum(int(a.GetFormalCharge()) for a in ref.GetAtoms()))
        source = str(row.get("source", "") or "input")
        scored.append(
            {
                "smiles": smi,
                "source": source,
                "source_rank": int(source_rank.get(source, 9)),
                "formal_charge": int(q),
                "charge_penalty": abs(int(q) - int(base_charge)),
            }
        )
    scored = sorted(scored, key=lambda r: (int(r["source_rank"]), int(r["charge_penalty"]), abs(int(r["formal_charge"])), len(str(r["smiles"]))))
    input_mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(Chem.RemoveHs(Chem.Mol(input_mol)))
    except Exception:
        pass
    input_charge = int(sum(int(a.GetFormalCharge()) for a in input_mol.GetAtoms()))
    states = [
        {
            "label": f"01_input_original_q{int(input_charge):+d}",
            "source": "input_original",
            "smiles": str(base_smiles),
            "formal_charge": int(input_charge),
            "prior_weight": 1.0,
            "mol": Chem.Mol(input_mol),
        }
    ]
    next_rank = 2
    for row in scored:
        if len(states) >= max_states_i:
            break
        if str(row.get("smiles", "") or "").strip() == str(base_smiles):
            continue
        ref = Chem.MolFromSmiles(str(row["smiles"]))
        if ref is None:
            continue
        ref_noh = Chem.RemoveHs(Chem.Mol(ref))
        try:
            templated = AllChem.AssignBondOrdersFromTemplate(ref_noh, base)
        except Exception:
            templated = Chem.Mol(base)
        templated = _copy_charges_from_template(ref_noh, templated)
        try:
            Chem.SanitizeMol(templated)
        except Exception:
            pass
        state_mol = Chem.AddHs(Chem.Mol(templated), addCoords=True)
        try:
            Chem.SanitizeMol(state_mol)
        except Exception:
            pass
        label = f"{int(next_rank):02d}_{str(row['source']).replace('+', '_')}_q{int(row['formal_charge']):+d}"
        if state_mol.HasProp("_Name"):
            state_mol.SetProp("_Name", str(state_mol.GetProp("_Name") or label))
        else:
            state_mol.SetProp("_Name", label)
        prior_weight = float(1.0 / float(2 ** max(1, next_rank - 1)))
        states.append(
            {
                "label": str(label),
                "source": str(row["source"]),
                "smiles": str(row["smiles"]),
                "formal_charge": int(row["formal_charge"]),
                "prior_weight": float(prior_weight),
                "mol": Chem.Mol(state_mol),
            }
        )
        next_rank += 1
    if not states:
        states = [
            {
                "label": "01_input_q0",
                "source": "input",
                "smiles": str(base_smiles),
                "formal_charge": int(base_charge),
                "prior_weight": 1.0,
                "mol": Chem.Mol(mol),
            }
        ]
    return list(states)


def write_single_mol_sdf(mol, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(out_path))
    writer.write(Chem.Mol(mol))
    writer.close()


def build_single_state_probe_command(args, *, ligand_sdf: Path, state_label: str, state_workdir: Path):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--target",
        str(args.target),
        "--ligand",
        str(args.ligand),
        "--data-root",
        str(args.data_root),
        "--workdir",
        str(state_workdir),
        "--sim-ns",
        str(args.sim_ns),
        "--sample-every-ps",
        str(args.sample_every_ps),
        "--temperature-k",
        str(args.temperature_k),
        "--pull-distance-a",
        str(args.pull_distance_a),
        "--pull-k",
        str(args.pull_k),
        "--minimize-iters",
        str(args.minimize_iters),
        "--eq-ps",
        str(args.eq_ps),
        "--antechamber",
        str(args.antechamber),
        "--parmchk2",
        str(args.parmchk2),
        "--tleap",
        str(args.tleap),
        "--random-seed",
        str(args.random_seed),
        "--energy-unit",
        str(args.energy_unit),
        "--target-ph",
        str(args.target_ph),
        "--ensemble-max-states",
        "1",
        "--ligand-sdf",
        str(ligand_sdf),
        "--state-label",
        str(state_label),
    ]
    return list(cmd)


def build_initial_pose_state_prefilter_payload(
    *,
    protein_pdb: Path,
    ligand_mol,
    workdir: Path,
    antechamber_bin: str,
    parmchk2_bin: str,
    tleap_bin: str,
    input_formal_charge: int,
):
    try:
        param_dir = Path(workdir) / "parameterization"
        param = parameterize_ligand_ambertools(
            ligand_mol,
            param_dir,
            residue_name="LIG",
            antechamber_bin=antechamber_bin,
            parmchk2_bin=parmchk2_bin,
            tleap_bin=tleap_bin,
        )
        ligand_pdb_mol = annotate_ligand_pdb_info(
            ligand_mol,
            list(param["atom_names"]),
            residue_name="LIG",
            chain_id="L",
            residue_number=1,
        )
        ligand_pdb = Path(workdir) / "ligand_param_named.pdb"
        ligand_pdb.write_text(Chem.MolToPDBBlock(ligand_pdb_mol), encoding="utf-8")
        combined_pdb = Path(workdir) / "complex_probe_case.pdb"
        build_complex_pdb(Path(protein_pdb), ligand_pdb, combined_pdb)

        pdb = PDBFile(str(combined_pdb))
        xyz = np.array(pdb.positions.value_in_unit(u.nanometer), dtype=float)
        atoms = list(pdb.topology.atoms())
        n_atoms = len(atoms)
        lig_charge_map = load_mol2_charges(Path(param["mol2"]))
        ligand_param_names = list(param["atom_names"])

        sigmas = np.zeros(n_atoms, dtype=float)
        epsilons = np.zeros(n_atoms, dtype=float)
        charges = np.zeros(n_atoms, dtype=float)
        atom_rows = []
        ligand_idx = []
        protein_idx = []
        ligand_counter = 0
        for i, atom in enumerate(atoms):
            res = atom.residue
            resn = str(res.name or "").strip().upper()
            name = str(atom.name or "").strip()
            el = str(infer_element(atom) or "").strip().upper()
            is_lig = (resn == "LIG") and (str(res.chain.id).strip() == "L")
            origin = "ligand" if is_lig else "protein"
            sigmas[i], epsilons[i] = nb_params(el)
            if origin == "ligand":
                if ligand_counter >= len(ligand_param_names):
                    raise RuntimeError("Ligand atom count mismatch during prefilter")
                param_name = str(ligand_param_names[ligand_counter]).strip().upper()
                ligand_counter += 1
                charges[i] = float(lig_charge_map.get(param_name, 0.0))
                display_name = str(param_name)
                ligand_idx.append(int(i))
            else:
                charges[i] = float(estimate_protein_charge(resn, name, el))
                display_name = str(name)
                protein_idx.append(int(i))
            atom_rows.append(
                {
                    "index": int(i),
                    "name": str(display_name),
                    "resn": str(resn),
                    "chain": str(res.chain.id),
                    "resi": str(res.id),
                    "element": str(el),
                    "origin": str(origin),
                }
            )
        if not ligand_idx or not protein_idx:
            raise RuntimeError("Missing ligand/protein atoms in prefilter complex")

        prot_xyz = np.asarray(xyz[np.asarray(protein_idx, dtype=int)], dtype=float)
        lig_xyz = np.asarray(xyz[np.asarray(ligand_idx, dtype=int)], dtype=float)
        dv = np.asarray(prot_xyz[:, None, :] - lig_xyz[None, :, :], dtype=float)
        dmat = np.linalg.norm(dv, axis=2)
        min_dist_a = float(10.0 * np.min(dmat)) if dmat.size else np.nan
        contact_count = int(np.sum(np.any(dmat <= 0.4, axis=1))) if dmat.size else 0

        rij = np.asarray(dmat, dtype=float)
        sigma_ij = 0.5 * (
            np.asarray(sigmas[np.asarray(protein_idx, dtype=int)], dtype=float)[:, None]
            + np.asarray(sigmas[np.asarray(ligand_idx, dtype=int)], dtype=float)[None, :]
        )
        epsilon_ij = np.sqrt(
            np.asarray(epsilons[np.asarray(protein_idx, dtype=int)], dtype=float)[:, None]
            * np.asarray(epsilons[np.asarray(ligand_idx, dtype=int)], dtype=float)[None, :]
        )
        qij = (
            np.asarray(charges[np.asarray(protein_idx, dtype=int)], dtype=float)[:, None]
            * np.asarray(charges[np.asarray(ligand_idx, dtype=int)], dtype=float)[None, :]
        )
        cutoff_nm = 1.2
        pair_mask = np.asarray(np.isfinite(rij) & (rij > 1.0e-6) & (rij <= cutoff_nm), dtype=bool)
        raw_total = np.nan
        if np.any(pair_mask):
            rsel = np.asarray(rij[pair_mask], dtype=float)
            sigsel = np.asarray(sigma_ij[pair_mask], dtype=float)
            epssel = np.asarray(epsilon_ij[pair_mask], dtype=float)
            qsel = np.asarray(qij[pair_mask], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                sr = np.asarray(sigsel / rsel, dtype=float)
                lj = np.asarray(4.0 * epssel * ((sr ** 12) - (sr ** 6)), dtype=float)
                coul = np.asarray(138.935456 * qsel / rsel, dtype=float)
            raw_total = float(np.nansum(lj + coul))
        rescoring_payload = build_abfe_gbsa_mmgbsa_rescore_payload(
            coords_nm_series=np.asarray([xyz], dtype=float),
            atom_rows=list(atom_rows or []),
            charges_e=np.asarray(charges, dtype=float),
            protein_atom_idx=np.asarray(protein_idx, dtype=int),
            ligand_atom_idx=np.asarray(ligand_idx, dtype=int),
            raw_interaction_energy_kj=np.asarray([raw_total], dtype=float),
            frame_mask=np.asarray([True], dtype=bool),
        )
        rescored = np.asarray(rescoring_payload.get("rescored_total_kj", []), dtype=float)
        rescored_total = float(rescored[0]) if rescored.size else float(raw_total)
        charge_diff = int(abs(int(param["formal_charge"]) - int(input_formal_charge)))
        clash_penalty = 0.0
        if np.isfinite(min_dist_a):
            if min_dist_a < 1.6:
                clash_penalty = 220.0
            elif min_dist_a < 1.9:
                clash_penalty = 80.0
        missing_contact_penalty = 0.0 if int(contact_count) >= 4 else float(20.0 * max(0, 4 - int(contact_count)))
        score_kj = float(rescored_total + (90.0 * charge_diff) + clash_penalty + missing_contact_penalty)
        return {
            "ok": True,
            "reason": "ok",
            "score_kj": float(score_kj),
            "rescored_total_kj": float(rescored_total),
            "raw_total_kj": float(raw_total),
            "formal_charge": int(param["formal_charge"]),
            "charge_diff": int(charge_diff),
            "contact_count_4a": int(contact_count),
            "min_dist_a": float(min_dist_a),
            "parameterization": dict(param),
            "combined_pdb": str(combined_pdb),
        }
    except Exception as exc:
        return {
            "ok": False,
            "reason": str(exc),
            "score_kj": np.nan,
            "rescored_total_kj": np.nan,
            "raw_total_kj": np.nan,
            "formal_charge": np.nan,
            "charge_diff": np.nan,
            "contact_count_4a": 0,
            "min_dist_a": np.nan,
        }


def select_probe_state_candidates_with_prefilter(
    state_candidates,
    *,
    protein_pdb: Path,
    prefilter_root: Path,
    antechamber_bin: str,
    parmchk2_bin: str,
    tleap_bin: str,
    input_formal_charge: int,
    keep_count=2,
    charge_tolerance=0,
):
    rows = []
    keep_n = max(1, int(keep_count))
    charge_tol = max(0, int(charge_tolerance))
    for state in list(state_candidates or []):
        label = str(state.get("label", "state") or "state")
        state_dir = Path(prefilter_root) / label
        payload = build_initial_pose_state_prefilter_payload(
            protein_pdb=Path(protein_pdb),
            ligand_mol=Chem.Mol(state["mol"]),
            workdir=state_dir,
            antechamber_bin=str(antechamber_bin),
            parmchk2_bin=str(parmchk2_bin),
            tleap_bin=str(tleap_bin),
            input_formal_charge=int(input_formal_charge),
        )
        row = dict(state)
        row["prefilter"] = dict(payload)
        row["prefilter_keep"] = False
        rows.append(row)

    input_rows = [r for r in rows if str(r.get("source", "") or "") == "input_original"]
    selected = []
    if input_rows:
        selected.append(dict(input_rows[0]))
    eligible = []
    for row in rows:
        if str(row.get("source", "") or "") == "input_original":
            continue
        pref = dict(row.get("prefilter", {}) or {})
        if not bool(pref.get("ok", False)):
            continue
        try:
            charge_diff = int(pref.get("charge_diff", 999))
        except Exception:
            charge_diff = 999
        if charge_diff > charge_tol:
            continue
        eligible.append(row)
    eligible = sorted(
        eligible,
        key=lambda r: (
            float(dict(r.get("prefilter", {}) or {}).get("score_kj", np.inf)),
            -float(r.get("prior_weight", 0.0)),
            str(r.get("label", "")),
        ),
    )
    for row in eligible:
        if len(selected) >= keep_n:
            break
        selected.append(dict(row))
    selected_labels = {str(r.get("label", "") or "") for r in selected}
    for row in rows:
        row["prefilter_keep"] = bool(str(row.get("label", "") or "") in selected_labels)
    return {"rows": rows, "selected": selected}


def aggregate_probe_state_ensemble_payload(state_rows, temperature_k=300.0):
    rows = list(state_rows or [])
    beta = 1.0 / float(R_KJ_MOL_K * max(1.0, float(temperature_k)))
    accepted = []
    for row in rows:
        dg = float(dict(row.get("free_energy", {}) or {}).get("dg_selected_kj_mol", np.nan))
        est = str(dict(row.get("free_energy", {}) or {}).get("selected_estimator", "gated_off") or "gated_off")
        if est == "gated_off" or not np.isfinite(dg):
            continue
        accepted.append(dict(row))
    accepted = sorted(accepted, key=lambda r: (float(dict(r.get("free_energy", {}) or {}).get("dg_selected_kj_mol", np.inf)), -float(r.get("prior_weight", 0.0))))
    if not accepted:
        return {
            "ok": False,
            "selected_estimator": "gated_off",
            "accepted_state_count": 0,
            "dg_selected_kj_mol": np.nan,
            "dg_selected_kcal_mol": np.nan,
            "best_state_label": "",
            "best_state_dg_kj_mol": np.nan,
            "confidence": "low",
        }
    weights = np.asarray([max(1.0e-8, float(r.get("prior_weight", 1.0))) for r in accepted], dtype=float)
    weights = np.asarray(weights / max(np.sum(weights), 1.0e-8), dtype=float)
    dg_vals = np.asarray([float(dict(r.get("free_energy", {}) or {}).get("dg_selected_kj_mol", np.nan)) for r in accepted], dtype=float)
    finite = np.isfinite(dg_vals)
    weights = np.asarray(weights[finite], dtype=float)
    dg_vals = np.asarray(dg_vals[finite], dtype=float)
    if dg_vals.size <= 0:
        return {
            "ok": False,
            "selected_estimator": "gated_off",
            "accepted_state_count": 0,
            "dg_selected_kj_mol": np.nan,
            "dg_selected_kcal_mol": np.nan,
            "best_state_label": "",
            "best_state_dg_kj_mol": np.nan,
            "confidence": "low",
        }
    weights = np.asarray(weights / max(np.sum(weights), 1.0e-8), dtype=float)
    ref = float(np.min(beta * dg_vals))
    z = float(np.sum(weights * np.exp(-(beta * dg_vals) + ref)))
    dg_ensemble = float((-np.log(max(z, 1.0e-300)) + ref) / beta)
    best = dict(accepted[int(np.argmin(dg_vals))])
    accepted_conf = [str(dict(r.get("free_energy", {}) or {}).get("reweight_confidence_quality", "low") or "low") for r in accepted]
    if len(accepted) >= 2 and all(v == "high" for v in accepted_conf):
        confidence = "high"
    elif len(accepted) >= 1 and any(v in {"high", "medium"} for v in accepted_conf):
        confidence = "medium"
    else:
        confidence = "low"
    return {
        "ok": True,
        "selected_estimator": "weighted_state_ensemble",
        "accepted_state_count": int(len(accepted)),
        "dg_selected_kj_mol": float(dg_ensemble),
        "dg_selected_kcal_mol": float(kj_to_kcal(dg_ensemble)),
        "best_state_label": str(best.get("state_label", "") or ""),
        "best_state_dg_kj_mol": float(dict(best.get("free_energy", {}) or {}).get("dg_selected_kj_mol", np.nan)),
        "best_state_dg_kcal_mol": float(kj_to_kcal(dict(best.get("free_energy", {}) or {}).get("dg_selected_kj_mol", np.nan))),
        "confidence": str(confidence),
    }


def write_ensemble_markdown_report(out_path: Path, summary: dict, energy_unit="kcal"):
    ens = dict(summary.get("ensemble", {}) or {})
    prefilter_rows = list(dict(summary.get("prefilter", {}) or {}).get("rows", []) or [])
    lines = [
        "# Benchmark ABFE Probe State Ensemble",
        "",
        f"Target: `{summary['inputs']['target']}`",
        f"Ligand: `{summary['inputs']['ligand']}`",
        f"Experimental ΔG: `{format_energy(summary['experiment'].get('exp_dg_kj', np.nan), energy_unit)}`",
        "",
        "## Ensemble",
        "",
        f"- Candidate states: `{int(ens.get('candidate_state_count', 0) or 0)}`",
        f"- Prefilter kept: `{int(ens.get('prefilter_selected_count', 0) or 0)}`",
        f"- Accepted states: `{int(ens.get('accepted_state_count', 0) or 0)}`",
        f"- Selected estimator: `{str(ens.get('selected_estimator', 'gated_off') or 'gated_off')}`",
        f"- Ensemble ΔG: `{format_energy(ens.get('dg_selected_kj_mol', np.nan), energy_unit)}`",
        f"- Confidence: `{str(ens.get('confidence', 'low') or 'low')}`",
        f"- Best accepted state: `{str(ens.get('best_state_label', '') or 'n/a')}` "
        f"({format_energy(ens.get('best_state_dg_kj_mol', np.nan), energy_unit)})",
        "",
        "## Prefilter",
        "",
    ]
    for row in prefilter_rows:
        pref = dict(row.get("prefilter", {}) or {})
        lines.append(
            f"- `{str(row.get('state_label', '') or 'state')}` | keep `{int(bool(row.get('keep', False)))}` | "
            f"source `{str(row.get('state_source', '') or '')}` | q `{int(row.get('formal_charge', 0) or 0):+d}` | "
            f"score `{format_energy(pref.get('score_kj', np.nan), energy_unit)}` | "
            f"rescored `{format_energy(pref.get('rescored_total_kj', np.nan), energy_unit)}` | "
            f"contacts `{int(pref.get('contact_count_4a', 0) or 0)}` | "
            f"min_dist `{float(pref.get('min_dist_a', np.nan)):.2f} A`"
        )
    lines.extend(
        [
            "",
        "## States",
        "",
        ]
    )
    for row in list(summary.get("states", []) or []):
        fe = dict(row.get("free_energy", {}) or {})
        lines.extend(
            [
                f"- `{str(row.get('state_label', '') or 'state')}` | source `{str(row.get('state_source', '') or 'input')}` | "
                f"q `{int(row.get('formal_charge', 0) or 0):+d}` | prior `{float(row.get('prior_weight', 1.0)):.3f}` | "
                f"selected `{str(fe.get('selected_estimator', 'gated_off') or 'gated_off')}` | "
                f"ΔG `{format_energy(fe.get('dg_selected_kj_mol', np.nan), energy_unit)}` | "
                f"confidence `{str(fe.get('reweight_confidence_quality', 'low') or 'low')}`"
            ]
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def resolve_tool_path(name: str, explicit: str = "") -> str:
    raw = str(explicit or "").strip()
    if raw and Path(raw).is_file():
        return str(Path(raw).resolve())
    hit = shutil.which(str(name))
    if hit:
        return str(Path(hit).resolve())
    env_bin = Path(sys.executable).resolve().parent
    candidates = [
        env_bin / str(name),
        env_bin / "wrapped_progs" / str(name),
    ]
    if str(name).lower() == "tleap":
        candidates.append(env_bin / "teLeap")
    for cand in candidates:
        if cand.is_file():
            return str(cand.resolve())
    raise FileNotFoundError(f"Could not resolve required executable: {name}")


def parse_experimental_dg_kj(lig_meta: dict, temperature_k=300.0):
    meas = dict((lig_meta or {}).get("measurement", {}) or {})
    for ad in meas.get("additional", []) or []:
        if str(ad.get("type", "")).strip().lower() == "dg":
            try:
                return float(ad.get("value")), "metadata_dg"
            except Exception:
                pass

    mtype = str(meas.get("type", "")).strip().lower()
    if mtype not in {"ki", "kd"}:
        return np.nan, "none"
    try:
        val = float(meas.get("value"))
    except Exception:
        return np.nan, "none"
    unit = str(meas.get("unit", "M")).strip().lower()
    scale = 1.0
    if unit in {"nm", "nmol/l", "nmolar", "nmol"}:
        scale = 1.0e-9
    elif unit in {"um", "µm", "micromolar"}:
        scale = 1.0e-6
    elif unit in {"mm", "millimolar"}:
        scale = 1.0e-3
    elif unit in {"pm", "pmolar"}:
        scale = 1.0e-12
    elif unit in {"m", "molar"}:
        scale = 1.0
    else:
        return np.nan, "none"
    ki_m = max(val * scale, 1.0e-15)
    dg = R_KJ_MOL_K * float(temperature_k) * math.log(ki_m)
    return float(dg), "from_ki"


def load_target_ligand(target_dir: Path, ligand_name: str):
    yml_path = target_dir / "ligands.yml"
    sdf_path = target_dir / "ligands.sdf"
    ligands_meta = yaml.safe_load(yml_path.read_text(encoding="utf-8")) or {}
    if ligand_name not in ligands_meta:
        raise KeyError(f"Ligand not found in ligands.yml: {ligand_name}")
    mol = None
    for one in Chem.SDMolSupplier(str(sdf_path), removeHs=False):
        if one is None:
            continue
        nm = one.GetProp("_Name") if one.HasProp("_Name") else ""
        if nm == ligand_name:
            mol = Chem.Mol(one)
            break
    if mol is None:
        raise KeyError(f"Ligand not found in ligands.sdf: {ligand_name}")
    if mol.GetNumConformers() <= 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    elif all(at.GetAtomicNum() != 1 for at in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)
    return mol, dict(ligands_meta.get(ligand_name, {}) or {})


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


def annotate_ligand_pdb_info(mol, atom_names, residue_name="LIG", chain_id="L", residue_number=1):
    lig = Chem.Mol(mol)
    for idx, atom in enumerate(lig.GetAtoms(), start=1):
        atom_name = ""
        if (idx - 1) < len(atom_names):
            atom_name = str(atom_names[idx - 1]).strip()
        if not atom_name:
            atom_name = f"{atom.GetSymbol()}{idx}"
        atom_name = atom_name[:4]
        atom.SetMonomerInfo(
            Chem.AtomPDBResidueInfo(
                f"{atom_name:<4s}",
                serialNumber=int(idx),
                altLoc="",
                residueName=f"{str(residue_name)[:3]:<3s}",
                residueNumber=int(residue_number),
                chainId=f"{str(chain_id)[:1]:1s}",
                insertionCode="",
                occupancy=1.0,
                tempFactor=0.0,
            )
        )
    return lig


def parameterize_ligand_ambertools(
    mol,
    workdir: Path,
    residue_name="LIG",
    antechamber_bin="antechamber",
    parmchk2_bin="parmchk2",
    tleap_bin="tleap",
):
    workdir.mkdir(parents=True, exist_ok=True)
    mol = Chem.Mol(mol)
    formal_charge = int(sum(int(at.GetFormalCharge()) for at in mol.GetAtoms()))
    sdf_path = workdir / "ligand_input.sdf"
    mol2_path = workdir / "ligand_gaff2.mol2"
    frcmod_path = workdir / "ligand_gaff2.frcmod"
    prmtop_path = workdir / "ligand.prmtop"
    inpcrd_path = workdir / "ligand.inpcrd"
    tleap_in = workdir / "tleap.in"

    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()

    run_checked(
        [
            antechamber_bin,
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
    run_checked(
        [
            parmchk2_bin,
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
    run_checked([tleap_bin, "-f", str(tleap_in)], cwd=workdir)

    atom_names = load_mol2_atom_names(mol2_path)
    charge_map = load_mol2_charges(mol2_path)
    charge_sum = float(sum(float(v) for v in charge_map.values()))
    return {
        "formal_charge": int(formal_charge),
        "partial_charge_sum": float(charge_sum),
        "mol2": str(mol2_path),
        "frcmod": str(frcmod_path),
        "prmtop": str(prmtop_path),
        "inpcrd": str(inpcrd_path),
        "atom_names": list(atom_names),
    }


def build_complex_pdb(protein_pdb: Path, ligand_pdb: Path, out_pdb: Path):
    protein = PDBFile(str(protein_pdb))
    modeller = Modeller(protein.topology, protein.positions)
    dropped = []
    to_delete = []
    for residue in modeller.topology.residues():
        resname = str(residue.name or "").strip().upper()
        if resname in WATER_RESIDUES or resname not in STANDARD_PROTEIN_RESIDUES:
            to_delete.append(residue)
            dropped.append(resname or "UNK")
    if to_delete:
        modeller.delete(to_delete)
    protein_ff = app.ForceField("amber14/protein.ff14SB.xml")
    modeller.addHydrogens(protein_ff, pH=7.4)
    lig = PDBFile(str(ligand_pdb))
    modeller.add(lig.topology, lig.positions)
    with out_pdb.open("w", encoding="utf-8") as fh:
        PDBFile.writeFile(modeller.topology, modeller.positions, fh, keepIds=True)
    return {
        "combined_pdb": str(out_pdb),
        "protein_atom_count": int(sum(1 for _ in modeller.topology.atoms())),
        "dropped_residues": sorted(set(dropped)),
    }


def summarize_generic_checks(
    *,
    formal_charge,
    partial_charge_sum,
    mean_bound_total,
    mean_unbound_total,
    free_energy,
    top_contacts,
    exp_dg_kj,
):
    selected_estimator = str(free_energy.get("selected_estimator", "gated_off") or "gated_off")
    dg_selected = float(free_energy.get("dg_selected_kj_mol", np.nan))
    dg_reweighted = float(free_energy.get("dg_reweighted_kj_mol", np.nan))
    dg_pop = float(free_energy.get("dg_pop_jeff_kj_mol", np.nan))
    delta_bound_unbound = float(mean_bound_total - mean_unbound_total)
    checks = {
        "charge_sum_matches_formal_charge": bool(abs(float(partial_charge_sum) - float(formal_charge)) <= 0.1),
        "steering_reaches_unbound_region": bool(int(free_energy.get("n_unbound", 0) or 0) > 0),
        "bound_state_more_favorable_than_unbound": bool(np.isfinite(delta_bound_unbound) and delta_bound_unbound <= -25.0),
        "selected_estimator_available": bool(selected_estimator != "gated_off"),
        "top_contacts_detected": bool(len(list(top_contacts or [])) >= 3),
    }
    errors = {
        "err_selected_vs_exp_kj": float(dg_selected - exp_dg_kj) if np.isfinite(dg_selected) and np.isfinite(exp_dg_kj) else np.nan,
        "err_reweighted_vs_exp_kj": float(dg_reweighted - exp_dg_kj) if np.isfinite(dg_reweighted) and np.isfinite(exp_dg_kj) else np.nan,
        "err_pop_jeff_vs_exp_kj": float(dg_pop - exp_dg_kj) if np.isfinite(dg_pop) and np.isfinite(exp_dg_kj) else np.nan,
    }
    return {
        "overall_pass": bool(all(checks.values())),
        "generic_checks": dict(checks),
        "delta_bound_minus_unbound_kj_mol": float(delta_bound_unbound),
        "delta_bound_minus_unbound_kcal_mol": float(kj_to_kcal(delta_bound_unbound)),
        "errors": dict(errors),
        "errors_kcal": {
            "err_selected_vs_exp_kcal": float(kj_to_kcal(errors.get("err_selected_vs_exp_kj", np.nan))),
            "err_reweighted_vs_exp_kcal": float(kj_to_kcal(errors.get("err_reweighted_vs_exp_kj", np.nan))),
            "err_pop_jeff_vs_exp_kcal": float(kj_to_kcal(errors.get("err_pop_jeff_vs_exp_kj", np.nan))),
        },
    }


def write_markdown_report(out_path: Path, summary: dict, energy_unit="kcal") -> None:
    fe = dict(summary.get("free_energy", {}) or {})
    assess = dict(summary.get("assessment", {}) or {})
    exp_dg_kj = float(summary.get("experiment", {}).get("exp_dg_kj", np.nan))
    bound_total_kj = float(summary.get("energetics", {}).get("mean_bound_total_kj_mol", np.nan))
    unbound_total_kj = float(summary.get("energetics", {}).get("mean_unbound_total_kj_mol", np.nan))
    delta_bound_kj = float(assess.get("delta_bound_minus_unbound_kj_mol", np.nan))
    lines = [
        "# Benchmark ABFE Probe Check",
        "",
        f"Target: `{summary['inputs']['target']}`",
        f"Ligand: `{summary['inputs']['ligand']}`",
        f"Experimental ΔG: `{format_energy(exp_dg_kj, energy_unit)}` ({summary['experiment'].get('exp_source', 'none')})",
        "",
        "## Parameterization",
        "",
        "- Charge method: `AM1-BCC (antechamber)`",
        f"- Formal charge: `{int(summary['ligand']['formal_charge'])}`",
        f"- Partial charge sum: `{float(summary['ligand']['partial_charge_sum']):.3f}`",
        f"- Heavy atoms: `{int(summary['ligand']['heavy_atoms'])}` | MW: `{float(summary['ligand']['mw']):.2f}`",
        "",
        "## Free Energy",
        "",
        f"- Selected estimator: `{str(fe.get('selected_estimator', 'gated_off'))}` | confidence=`{str(fe.get('reweight_confidence_quality', 'low'))}`",
        f"- Selected note: `{str(fe.get('reweight_confidence_note', ''))}`",
        f"- Selected ΔG: `{format_energy(fe.get('dg_selected_kj_mol', np.nan), energy_unit)}`",
        f"- Reweighted ΔG: `{format_energy(fe.get('dg_reweighted_kj_mol', np.nan), energy_unit)}`",
        f"- Jeffreys ΔG(pop): `{format_energy(fe.get('dg_pop_jeff_kj_mol', np.nan), energy_unit)}`",
        f"- Exp-avg diagnostic ΔG: `{format_energy(fe.get('dg_exp_kj_mol', np.nan), energy_unit)}`",
        f"- Cumulant diagnostic ΔG: `{format_energy(fe.get('dg_cum_kj_mol', np.nan), energy_unit)}`",
        f"- ESS(total): `{float(fe.get('reweight_ess_total', np.nan)):.2f}` | ESS(unbound): `{float(fe.get('reweight_ess_unbound', np.nan)):.2f}`",
        f"- State counts: `bound={int(fe.get('n_bound', 0))}`, `unbound={int(fe.get('n_unbound', 0))}`",
        f"- Hold counts: `bound={int(fe.get('n_bound_hold', 0))}`, `unbound={int(fe.get('n_unbound_hold', 0))}`",
        "",
        "## Energetics",
        "",
        f"- Bound mean total interaction: `{format_energy(bound_total_kj, energy_unit)}`",
        f"- Unbound mean total interaction: `{format_energy(unbound_total_kj, energy_unit)}`",
        f"- Bound minus unbound: `{format_energy(delta_bound_kj, energy_unit)}`",
        "",
        "## Contacts",
        "",
    ]
    for row in list(summary.get("top_bound_contacts", []) or [])[:10]:
        lines.append(
            f"- {row['resn']} {row['chain']} {row['resi']}: occupancy={float(row['bound_contact_occupancy']):.3f}"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
        ]
    )
    for key, value in dict(assess.get("generic_checks", {}) or {}).items():
        lines.append(f"- {key}: {'pass' if bool(value) else 'fail'}")
    lines.append("")
    lines.append("## Errors vs Experiment")
    lines.append("")
    for key, value in dict(assess.get("errors", {}) or {}).items():
        if np.isfinite(float(value)):
            lines.append(f"- {key}: `{format_energy(value, energy_unit)}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Run the modern probe/reweighted ABFE-lite path on a benchmark target ligand.")
    ap.add_argument("--target", required=True)
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--data-root", default="benchmark_data")
    ap.add_argument("--workdir", default="benchmark_results/benchmark_probe_case")
    ap.add_argument("--ligand-sdf", default="")
    ap.add_argument("--state-label", default="")
    ap.add_argument("--sim-ns", type=float, default=0.25)
    ap.add_argument("--sample-every-ps", type=float, default=2.0)
    ap.add_argument("--temperature-k", type=float, default=300.0)
    ap.add_argument("--target-ph", type=float, default=7.4)
    ap.add_argument("--ensemble-max-states", type=int, default=1)
    ap.add_argument("--ensemble-prefilter-keep", type=int, default=2)
    ap.add_argument("--ensemble-charge-tolerance", type=int, default=0)
    ap.add_argument("--pull-distance-a", type=float, default=18.0)
    ap.add_argument("--pull-k", type=float, default=2500.0)
    ap.add_argument("--minimize-iters", type=int, default=1200)
    ap.add_argument("--eq-ps", type=float, default=20.0)
    ap.add_argument("--antechamber", default=os.environ.get("ANTECHAMBER_BIN", "antechamber"))
    ap.add_argument("--parmchk2", default=os.environ.get("PARMCHK2_BIN", "parmchk2"))
    ap.add_argument("--tleap", default=os.environ.get("TLEAP_BIN", "tleap"))
    ap.add_argument("--random-seed", type=int, default=20260312)
    ap.add_argument("--energy-unit", choices=["kcal", "kj"], default="kcal")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    target_dir = (Path(args.data_root) / str(args.target)).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    protein_pdb = target_dir / "protein.pdb"
    if not protein_pdb.is_file():
        raise FileNotFoundError(protein_pdb)

    ligand_mol, ligand_meta = load_target_ligand(target_dir, str(args.ligand))
    ligand_sdf_override = str(args.ligand_sdf or "").strip()
    if ligand_sdf_override:
        sdf_path = Path(ligand_sdf_override).resolve()
        if not sdf_path.is_file():
            raise FileNotFoundError(sdf_path)
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        ligand_override = None
        for one in suppl:
            if one is None:
                continue
            ligand_override = Chem.Mol(one)
            break
        if ligand_override is None:
            raise RuntimeError(f"Failed to read ligand state from SDF: {sdf_path}")
        ligand_mol = Chem.Mol(ligand_override)
    exp_dg_kj, exp_source = parse_experimental_dg_kj(ligand_meta, temperature_k=float(args.temperature_k))

    if (not ligand_sdf_override) and int(max(1, int(args.ensemble_max_states))) > 1:
        state_candidates = enumerate_ligand_state_candidates(
            ligand_mol,
            target_ph=float(args.target_ph),
            max_states=int(args.ensemble_max_states),
            max_tautomers=max(4, int(args.ensemble_max_states) * 3),
        )
        if len(state_candidates) > 1:
            antechamber_bin = resolve_tool_path("antechamber", args.antechamber)
            parmchk2_bin = resolve_tool_path("parmchk2", args.parmchk2)
            tleap_bin = resolve_tool_path("tleap", args.tleap)
            state_root = workdir / "state_ensemble"
            state_root.mkdir(parents=True, exist_ok=True)
            prefilter_payload = select_probe_state_candidates_with_prefilter(
                state_candidates,
                protein_pdb=protein_pdb,
                prefilter_root=state_root / "_prefilter",
                antechamber_bin=antechamber_bin,
                parmchk2_bin=parmchk2_bin,
                tleap_bin=tleap_bin,
                input_formal_charge=int(sum(int(a.GetFormalCharge()) for a in ligand_mol.GetAtoms())),
                keep_count=int(args.ensemble_prefilter_keep),
                charge_tolerance=int(args.ensemble_charge_tolerance),
            )
            state_rows = []
            for idx, state in enumerate(list(prefilter_payload.get("selected", []) or []), start=1):
                state_label = str(state.get("label", f"{idx:02d}_state") or f"{idx:02d}_state")
                state_dir = state_root / state_label
                state_sdf = state_dir / "ligand_state.sdf"
                write_single_mol_sdf(Chem.Mol(state["mol"]), state_sdf)
                cmd = build_single_state_probe_command(
                    args,
                    ligand_sdf=state_sdf,
                    state_label=state_label,
                    state_workdir=state_dir,
                )
                run_checked(cmd, cwd=repo_root)
                state_summary_path = state_dir / "benchmark_probe_summary.json"
                if not state_summary_path.is_file():
                    raise RuntimeError(f"State summary missing: {state_summary_path}")
                state_summary = json.loads(state_summary_path.read_text(encoding="utf-8"))
                state_rows.append(
                    {
                        "state_label": str(state_label),
                        "state_source": str(state.get("source", "input") or "input"),
                        "state_smiles": str(state.get("smiles", "") or ""),
                        "formal_charge": int(state.get("formal_charge", 0) or 0),
                        "prior_weight": float(state.get("prior_weight", 1.0) or 1.0),
                        "summary_path": str(state_summary_path),
                        "free_energy": dict((state_summary or {}).get("free_energy", {}) or {}),
                    }
                )
            ensemble_payload = aggregate_probe_state_ensemble_payload(
                state_rows,
                temperature_k=float(args.temperature_k),
            )
            ensemble_payload["candidate_state_count"] = int(len(state_candidates))
            ensemble_payload["prefilter_selected_count"] = int(len(list(prefilter_payload.get("selected", []) or [])))
            summary = {
                "inputs": {
                    "target": str(args.target),
                    "ligand": str(args.ligand),
                    "temperature_k": float(args.temperature_k),
                    "target_ph": float(args.target_ph),
                    "ensemble_max_states": int(args.ensemble_max_states),
                    "ensemble_prefilter_keep": int(args.ensemble_prefilter_keep),
                    "ensemble_charge_tolerance": int(args.ensemble_charge_tolerance),
                },
                "experiment": {
                    "exp_dg_kj": float(exp_dg_kj),
                    "exp_dg_kcal": float(kj_to_kcal(exp_dg_kj)),
                    "exp_source": str(exp_source),
                },
                "ensemble": dict(ensemble_payload),
                "prefilter": {
                    "rows": [
                        {
                            "state_label": str(r.get("label", "") or ""),
                            "state_source": str(r.get("source", "") or ""),
                            "formal_charge": int(r.get("formal_charge", 0) or 0),
                            "prior_weight": float(r.get("prior_weight", 1.0) or 1.0),
                            "keep": bool(r.get("prefilter_keep", False)),
                            "prefilter": dict(r.get("prefilter", {}) or {}),
                        }
                        for r in list(prefilter_payload.get("rows", []) or [])
                    ]
                },
                "states": list(state_rows),
            }
            summary_path = workdir / "benchmark_probe_ensemble_summary.json"
            report_path = workdir / "benchmark_probe_ensemble_report.md"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            write_ensemble_markdown_report(report_path, summary, energy_unit=str(args.energy_unit))
            print("Benchmark ABFE probe state ensemble")
            print(f"Target: {args.target} | Ligand: {args.ligand} | states={len(state_candidates)}")
            print(
                f"ΔG(exp)={format_energy(exp_dg_kj, args.energy_unit)} | "
                f"ΔG(ensemble)={format_energy(ensemble_payload.get('dg_selected_kj_mol', np.nan), args.energy_unit)} | "
                f"accepted_states={int(ensemble_payload.get('accepted_state_count', 0) or 0)} / "
                f"prefilter_kept={int(ensemble_payload.get('prefilter_selected_count', 0) or 0)} | "
                f"confidence={str(ensemble_payload.get('confidence', 'low') or 'low')}"
            )
            print(f"Best state: {str(ensemble_payload.get('best_state_label', '') or 'n/a')}")
            print(f"Summary JSON: {summary_path}")
            print(f"Report MD: {report_path}")
            return

    antechamber_bin = resolve_tool_path("antechamber", args.antechamber)
    parmchk2_bin = resolve_tool_path("parmchk2", args.parmchk2)
    tleap_bin = resolve_tool_path("tleap", args.tleap)

    param_dir = workdir / "parameterization"
    param = parameterize_ligand_ambertools(
        ligand_mol,
        param_dir,
        residue_name="LIG",
        antechamber_bin=antechamber_bin,
        parmchk2_bin=parmchk2_bin,
        tleap_bin=tleap_bin,
    )
    ligand_pdb_mol = annotate_ligand_pdb_info(
        ligand_mol,
        list(param["atom_names"]),
        residue_name="LIG",
        chain_id="L",
        residue_number=1,
    )
    ligand_pdb = workdir / "ligand_param_named.pdb"
    ligand_pdb.write_text(Chem.MolToPDBBlock(ligand_pdb_mol), encoding="utf-8")
    combined_pdb = workdir / "complex_probe_case.pdb"
    build_complex_pdb(protein_pdb, ligand_pdb, combined_pdb)

    pdb = PDBFile(str(combined_pdb))
    top = pdb.topology
    xyz0 = np.array(pdb.positions.value_in_unit(u.nanometer), dtype=float)
    atoms = list(top.atoms())
    n_atoms = len(atoms)

    lig_charge_map = load_mol2_charges(Path(param["mol2"]))
    ligand_param_names = list(param["atom_names"])

    atom_rows = []
    ligand_idx = []
    ligand_heavy_idx = []
    protein_idx = []
    protein_heavy_idx = []
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
        is_lig = (resn == "LIG") and (str(res.chain.id).strip() == "L")
        origin = "ligand" if is_lig else "protein"

        sigmas[i], epsilons[i] = nb_params(el)
        masses[i] = float(mass_for_element(el))
        if origin == "ligand":
            if ligand_counter >= len(ligand_param_names):
                raise RuntimeError("Ligand atom count mismatch between combined complex and parameterized MOL2.")
            param_name = str(ligand_param_names[ligand_counter]).strip().upper()
            ligand_counter += 1
            charges[i] = float(lig_charge_map.get(param_name, 0.0))
            display_name = str(param_name)
        else:
            charges[i] = float(estimate_protein_charge(resn, name, el))
            display_name = str(name)

        atom_rows.append(
            {
                "index": int(i),
                "name": str(display_name),
                "resn": str(resn),
                "chain": str(res.chain.id),
                "resi": str(res.id),
                "element": str(el),
                "origin": str(origin),
            }
        )
        if origin == "ligand":
            ligand_idx.append(int(i))
            if el != "H":
                ligand_heavy_idx.append(int(i))
        else:
            protein_idx.append(int(i))
            if el != "H":
                protein_heavy_idx.append(int(i))

    if ligand_counter != len(ligand_param_names):
        raise RuntimeError(
            f"Ligand atom count mismatch between complex ({ligand_counter}) and MOL2 ({len(ligand_param_names)})."
        )
    if not ligand_idx or not protein_idx:
        raise RuntimeError("Failed to identify ligand/protein atoms in the combined complex.")

    system = System()
    for mass in masses:
        system.addParticle(max(1.0, float(mass)) * u.dalton)

    ff_counts = inject_ligand_terms_from_prmtop(system, ligand_idx, atom_rows, Path(param["prmtop"]))
    pocket_desc = compute_runtime_pocket_descriptors(
        atom_rows,
        xyz0,
        protein_idx,
        ligand_idx,
        pocket_cutoff_a=6.0,
    )

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

    receptor_group = set(int(i) for i in protein_idx)
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
    lig_masses = np.asarray([max(1.0, masses[int(i)]) for i in lig_heavy], dtype=float)
    lig_com0 = np.average(xyz0[lig_heavy], axis=0, weights=lig_masses)

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
        steering_corridors = [
            {
                "direction": np.array([1.0, 0.0, 0.0], dtype=float),
                "clearance_nm": np.nan,
                "collision_penalty": np.nan,
                "alignment": np.nan,
                "score": np.nan,
            }
        ]

    pull_dist_nm = max(0.4, float(args.pull_distance_a) * 0.1)
    steering_dir_index = 0
    steering_switch_count = 0
    steering_k_current = float(max(200.0, float(args.pull_k)))
    pull_dir = np.asarray(steering_corridors[0]["direction"], dtype=float)
    pull_vec = np.asarray(pull_dir * pull_dist_nm, dtype=float)
    pull_particles = [int(i) for i in lig_heavy.tolist()]
    if len(pull_particles) > 12:
        d0 = np.linalg.norm(xyz0[pull_particles] - lig_com0[None, :], axis=1)
        keep = np.argsort(d0)[:12]
        pull_particles = [pull_particles[int(k)] for k in keep]
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
        steer_rows.append({"slot": int(slot), "particle": int(idx), "x0": float(x0), "y0": float(y0), "z0": float(z0)})
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
    ctx, platform, platform_name = build_context_cpu_only(system, integ)
    ctx.setPositions(xyz0 * u.nanometer)
    try:
        ctx.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin, int(args.random_seed))
    except Exception:
        ctx.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin)
    if int(args.minimize_iters) > 0:
        ctx.setParameter("k_hold", 1800.0)
        LocalEnergyMinimizer.minimize(ctx, maxIterations=int(args.minimize_iters))
        ctx.setParameter("k_hold", 0.0)
        LocalEnergyMinimizer.minimize(ctx, maxIterations=max(200, int(args.minimize_iters) // 2))

    min_positions = ctx.getState(getPositions=True).getPositions()
    try:
        integ_run = LangevinMiddleIntegrator(float(args.temperature_k) * u.kelvin, 5.0 / u.picosecond, timestep_ps * u.picoseconds)
    except Exception:
        integ_run = LangevinIntegrator(float(args.temperature_k) * u.kelvin, 5.0 / u.picosecond, timestep_ps * u.picoseconds)
    if int(args.random_seed) > 0 and hasattr(integ_run, "setRandomNumberSeed"):
        try:
            integ_run.setRandomNumberSeed(int(args.random_seed))
        except Exception:
            pass
    sim = app.Simulation(top, system, integ_run, platform)
    sim.context.setPositions(min_positions)
    try:
        sim.context.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin, int(args.random_seed))
    except Exception:
        sim.context.setVelocitiesToTemperature(float(args.temperature_k) * u.kelvin)

    out_dcd = workdir / "benchmark_probe_case.dcd"
    out_final_pdb = workdir / "benchmark_probe_case_final.pdb"
    if out_dcd.exists():
        out_dcd.unlink()
    stride = max(1, int(round(float(args.sample_every_ps) / timestep_ps)))
    sim.reporters.append(DCDReporter(str(out_dcd), stride, enforcePeriodicBox=False))
    if float(args.eq_ps) > 0.0:
        sim.step(max(1, int(round(float(args.eq_ps) / timestep_ps))))

    ligand_ring_count = int(max(0, rdMolDescriptors.CalcNumRings(Chem.RemoveHs(Chem.Mol(ligand_mol)))))
    rot_bonds = int(Lipinski.NumRotatableBonds(Chem.RemoveHs(Chem.Mol(ligand_mol))))
    adaptive_state = build_abfe_adaptive_steering_state_payload(
        enabled=True,
        steering_mode="pull_hold",
        initial_pull_k_kj_nm2=float(steering_k_current),
        n_direction_candidates=int(len(steering_corridors)),
        ligand_heavy_atoms=int(lig_heavy.size),
        formal_charge_e=float(np.sum(np.asarray(charges[np.asarray(ligand_idx, dtype=int)], dtype=float))),
        ring_count=int(ligand_ring_count),
        rotatable_bonds=int(rot_bonds),
        pocket_desc=dict(pocket_desc or {}),
        direction_clearances_nm=[float(dict(c or {}).get("clearance_nm", np.nan)) for c in list(steering_corridors or [])],
        direction_collision_penalties=[float(dict(c or {}).get("collision_penalty", np.nan)) for c in list(steering_corridors or [])],
        enable_probe_phase=True,
        sample_every_ps=float(args.sample_every_ps),
    )
    steering_k_current = float((adaptive_state or {}).get("current_pull_k_kj_nm2", steering_k_current))
    try:
        sim.context.setParameter("k_pull", float(steering_k_current / float(max(1, len(pull_particles)))))
    except Exception:
        pass

    sim_ns = max(0.2, float(args.sim_ns))
    prod_steps = int(round((sim_ns * 1000.0) / timestep_ps))
    n_samples = max(1, prod_steps // stride)
    lam_plan = build_abfe_steering_lambda_schedule(n_samples=int(n_samples), steering_mode="pull_hold", steering_enabled=True)
    lam_schedule = np.asarray(lam_plan.get("lambda_schedule", []), dtype=float)
    if int(lam_schedule.size) != int(n_samples):
        lam_schedule = np.ones(int(n_samples), dtype=float)

    energies_total = []
    energies_lj = []
    energies_coul = []
    energies_bias = []
    protein_min_dist = []
    rc_site_min = []
    rc_site_com = []
    protein_contact_count_hist = []
    xyz_series = []
    lig_com_series = []
    times_ps = []
    lam_hist = []
    pull_k_hist = []
    corridor_hist = []
    steering_phase_hist = []
    steering_phase_sample_hist = []
    steering_reason_hist = []
    steering_note_hist = []
    far_hold_stable_hist = []
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
                [float(row["x0"]), float(row["y0"]), float(row["z0"]), float(vv[0]), float(vv[1]), float(vv[2])],
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
        if (not np.isfinite(pull_distance_scale_runtime)) or (pull_distance_scale_runtime <= 0.0):
            pull_distance_scale_runtime = 1.0
        target_pull_dist_nm = float(max(pull_dist_nm, pull_dist_nm * pull_distance_scale_runtime))
        target_pull_vec = np.asarray(pull_dir * target_pull_dist_nm, dtype=float)
        if phase_pre == "bound_settle":
            lam_runtime = float(adaptive_state_pre.get("bound_settle_lambda_value", 0.1) or 0.1)
        elif phase_pre == "probe":
            lam_runtime = float(adaptive_state_pre.get("probe_phase_lambda_value", lam_runtime) or lam_runtime)
        elif phase_pre in {"far_hold", "detached_settle"}:
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
                detach_dist_nm = float(
                    max(
                        target_pull_dist_nm,
                        min(max_detach_nm, pull_dist_nm * max(1.0, detach_scale)),
                    )
                )
                target_pull_vec = np.asarray(detach_dir * detach_dist_nm, dtype=float)
        elif phase_pre == "pull" and bool(adaptive_state_pre.get("pull_surface_detach_active", False)) and bool(adaptive_state_pre.get("surface_detach_enabled", False)):
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
            pull_detach_dist_nm = float(
                max(
                    target_pull_dist_nm,
                    min(max_detach_nm, pull_dist_nm * max(1.0, detach_scale)),
                )
            )
            blend_vec = ((1.0 - blend_frac) * np.asarray(pull_dir, dtype=float)) + (blend_frac * detach_dir)
            blend_norm = float(np.linalg.norm(blend_vec))
            if blend_norm > 1.0e-8:
                target_pull_vec = np.asarray((blend_vec / blend_norm) * pull_detach_dist_nm, dtype=float)
        elif bool(adaptive_state_pre.get("keep_pull_lambda_high_until_far_hold", False)) and bool(adaptive_state_pre.get("high_lambda_seen", False)):
            lam_runtime = 1.0
        apply_pull_vector(target_pull_vec)
        sim.context.setParameter("lam", float(lam_runtime))
        sim.step(stride)

        xyz = np.array(
            sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer),
            dtype=float,
        )
        xyz_prev = np.asarray(xyz, dtype=float)
        xyz_series.append(np.asarray(xyz, dtype=float))

        e_lj = float(sim.context.getState(getEnergy=True, groups=(1 << 1)).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        e_c = float(sim.context.getState(getEnergy=True, groups=(1 << 2)).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
        try:
            e_bias = float(sim.context.getState(getEnergy=True, groups=(1 << 3)).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole))
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

        energies_total.append(float(e_tot))
        energies_lj.append(float(e_lj))
        energies_coul.append(float(e_c))
        energies_bias.append(float(e_bias) if np.isfinite(e_bias) else np.nan)
        protein_min_dist.append(float(protein_min_dist_a))
        rc_site_min.append(float(rc_min_a))
        rc_site_com.append(float(rc_com_a))
        protein_contact_count_hist.append(float(protein_contact_count))
        lig_com_series.append(np.asarray(lig_com, dtype=float))
        times_ps.append(float((sample_idx + 1) * float(args.sample_every_ps)))
        lam_hist.append(float(lam_runtime))

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
        adaptive_state = dict(steer_decision.get("adaptive_state", adaptive_state))
        steering_switch_count = int(adaptive_state.get("switches_used", steering_switch_count))
        if bool(steer_decision.get("update_pull_k", False)):
            steering_k_current = float(steer_decision.get("pull_k_kj_nm2", steering_k_current))
            sim.context.setParameter("k_pull", float(steering_k_current / float(max(1, len(pull_particles)))))
        if bool(steer_decision.get("switch_direction", False)):
            next_dir = int(steer_decision.get("direction_index", steering_dir_index))
            if 0 <= next_dir < len(steering_corridors):
                steering_dir_index = int(next_dir)
                pull_dir = np.asarray(steering_corridors[steering_dir_index]["direction"], dtype=float)
                corridor_pull_vec = np.asarray(pull_dir * pull_dist_nm, dtype=float)
                apply_pull_vector(corridor_pull_vec)

        pull_k_hist.append(float(steering_k_current))
        corridor_hist.append(int(steering_dir_index))
        steering_phase_hist.append(str((adaptive_state or {}).get("phase", "pull") or "pull"))
        steering_reason_hist.append(str(steer_decision.get("reason", "") or ""))
        steering_note_hist.append(",".join(str(v) for v in list(steer_decision.get("notes", []) or [])))
        far_hold_stable_hist.append(bool(steer_decision.get("far_hold_stable", False)))

    state = sim.context.getState(getPositions=True)
    with out_final_pdb.open("w", encoding="utf-8") as fh:
        PDBFile.writeFile(sim.topology, state.getPositions(), fh, keepIds=True)

    arr_total = np.asarray(energies_total, dtype=float)
    arr_lj = np.asarray(energies_lj, dtype=float)
    arr_coul = np.asarray(energies_coul, dtype=float)
    arr_protein_min = np.asarray(protein_min_dist, dtype=float)
    arr_rc = np.asarray(rc_site_min, dtype=float)
    arr_rc_com = np.asarray(rc_site_com, dtype=float)
    arr_contacts = np.asarray(protein_contact_count_hist, dtype=float)

    free_energy_payload = population_free_energy_metrics(
        frames=list(range(1, len(arr_total) + 1)),
        times_ps=list(times_ps),
        lam_hist=list(lam_hist),
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
        bootstrap_seed=int(args.random_seed),
        hold_config=dict(adaptive_state or {}),
    )
    post_payload = dict(free_energy_payload.get("post_payload", {}) or {})
    valid_fe_mask = np.asarray(post_payload.get("valid_mask_fe", []), dtype=bool)
    rescored_total_valid = np.asarray([], dtype=float)
    rescoring_payload = build_abfe_gbsa_mmgbsa_rescore_payload(
        coords_nm_series=np.asarray(xyz_series, dtype=float),
        atom_rows=list(atom_rows or []),
        charges_e=np.asarray(charges, dtype=float),
        protein_atom_idx=np.asarray(protein_idx, dtype=int),
        ligand_atom_idx=np.asarray(ligand_idx, dtype=int),
        raw_interaction_energy_kj=np.asarray(arr_total, dtype=float),
        frame_mask=np.asarray(valid_fe_mask, dtype=bool),
        ligand_prmtop_file=str(param.get("prmtop", "") or ""),
    )
    if bool(rescoring_payload.get("ok", False)):
        rescored_total_valid = np.asarray(rescoring_payload.get("rescored_total_kj", []), dtype=float)
    numeric_payload = build_abfe_free_energy_numeric_payload(
        frames_valid=np.asarray(post_payload.get("frames_valid", []), dtype=int),
        bias_valid=np.asarray(post_payload.get("bias_valid", []), dtype=float),
        total_valid=np.asarray(post_payload.get("total_valid", []), dtype=float),
        rescored_total_valid=np.asarray(rescored_total_valid, dtype=float),
        ligand_internal_valid=np.asarray(rescoring_payload.get("ligand_internal_kj", []), dtype=float),
        rescored_energy_label="gbsa_obc2",
        distance_valid=np.asarray(post_payload.get("distance_valid", []), dtype=float),
        coul_valid=np.asarray(post_payload.get("coul_valid", []), dtype=float),
        bound_valid_mask=np.asarray(post_payload.get("bound_valid_mask", []), dtype=bool),
        unbound_valid_mask=np.asarray(post_payload.get("unbound_valid_mask", []), dtype=bool),
        bound_cutoff_a=float(post_payload.get("bound_cutoff_a", np.nan)),
        n_bound=int(post_payload.get("n_bound", 0) or 0),
        n_dist_valid=int(post_payload.get("n_dist_valid", 0) or 0),
        n_bound_raw=int(post_payload.get("n_bound_raw", post_payload.get("n_bound", 0)) or 0),
        n_unbound_raw=int(post_payload.get("n_unbound_raw", post_payload.get("n_unbound", 0)) or 0),
        n_bound_hold=int(post_payload.get("n_bound_hold", post_payload.get("n_bound", 0)) or 0),
        n_unbound_hold=int(post_payload.get("n_unbound_hold", post_payload.get("n_unbound", 0)) or 0),
        estimator_window_mode=str(post_payload.get("estimator_window_mode", "state_fallback") or "state_fallback"),
        use_count_fraction=bool(int(post_payload.get("n_dist_valid", 0) or 0) > 0),
        temperature_k=float(args.temperature_k),
        v_eff_nm3=float(post_payload.get("v_eff_nm3", np.nan)),
        bootstrap_seed=int(args.random_seed),
        n_boot=200,
    )
    free_energy = dict(free_energy_payload.get("summary", {}) or {})
    free_energy["rescoring_ok"] = bool(rescoring_payload.get("ok", False))
    free_energy["rescoring_reason"] = str(rescoring_payload.get("reason", "") or "")
    free_energy["dg_exp_kj_mol"] = float(numeric_payload.get("dg_exp", np.nan))
    free_energy["dg_cum_kj_mol"] = float(numeric_payload.get("dg_cum", np.nan))
    free_energy["ci_low_kj_mol"] = float(numeric_payload.get("ci_low", np.nan))
    free_energy["ci_high_kj_mol"] = float(numeric_payload.get("ci_high", np.nan))
    free_energy["rescored_energy_label"] = str(numeric_payload.get("rescored_energy_label", "") or "")
    free_energy["ligand_internal_available"] = bool(rescoring_payload.get("ligand_internal_ok", False))
    free_energy["ligand_internal_reason"] = str(rescoring_payload.get("ligand_internal_reason", "") or "")
    free_energy["dg_rescored_state_expavg_kj_mol"] = float(numeric_payload.get("dg_rescored_state_expavg_kj", np.nan))
    free_energy["dg_rescored_state_blockexpavg_kj_mol"] = float(numeric_payload.get("dg_rescored_state_blockexpavg_kj", np.nan))
    free_energy["dg_rescored_state_expavg_strain_kj_mol"] = float(numeric_payload.get("dg_rescored_state_expavg_strain_kj", np.nan))
    free_energy["dg_rescored_state_blockexpavg_strain_kj_mol"] = float(numeric_payload.get("dg_rescored_state_blockexpavg_strain_kj", np.nan))
    free_energy["dg_rescored_state_mean_kj_mol"] = float(numeric_payload.get("dg_rescored_state_mean_kj", np.nan))
    free_energy["dg_ligand_strain_kj_mol"] = float(numeric_payload.get("dg_ligand_strain_kj", np.nan))
    free_energy["selected_estimator"] = str(numeric_payload.get("selected_estimator", free_energy.get("selected_estimator", "gated_off")) or "gated_off")
    free_energy["dg_selected_kj_mol"] = float(numeric_payload.get("dg_selected_kj", free_energy.get("dg_selected_kj_mol", np.nan)))
    free_energy["kd_selected_m"] = float(numeric_payload.get("kd_selected_m", free_energy.get("kd_selected_m", np.nan)))
    free_energy["p_kd_selected"] = float(numeric_payload.get("p_kd_selected", free_energy.get("p_kd_selected", np.nan)))
    free_energy["dg_reweighted_kj_mol"] = float(numeric_payload.get("dg_reweighted_kj", free_energy.get("dg_reweighted_kj_mol", np.nan)))
    free_energy["kd_app_reweighted_m"] = float(numeric_payload.get("kd_app_reweighted_m", free_energy.get("kd_app_reweighted_m", np.nan)))
    free_energy["p_kd_reweighted"] = float(numeric_payload.get("p_kd_reweighted", free_energy.get("p_kd_reweighted", np.nan)))
    free_energy["reweight_ci_low_kj_mol"] = float(numeric_payload.get("reweight_ci_low", free_energy.get("reweight_ci_low_kj_mol", np.nan)))
    free_energy["reweight_ci_high_kj_mol"] = float(numeric_payload.get("reweight_ci_high", free_energy.get("reweight_ci_high_kj_mol", np.nan)))
    free_energy["reweight_ess_total"] = float(numeric_payload.get("reweight_ess_total", free_energy.get("reweight_ess_total", np.nan)))
    free_energy["reweight_ess_bound"] = float(numeric_payload.get("reweight_ess_bound", free_energy.get("reweight_ess_bound", np.nan)))
    free_energy["reweight_ess_unbound"] = float(numeric_payload.get("reweight_ess_unbound", free_energy.get("reweight_ess_unbound", np.nan)))
    free_energy["reweight_confidence_quality"] = str(numeric_payload.get("reweight_confidence_quality", free_energy.get("reweight_confidence_quality", "low")) or "low")
    free_energy["reweight_confidence_note"] = str(numeric_payload.get("reweight_confidence_note", free_energy.get("reweight_confidence_note", "")) or "")
    free_energy["reweight_confidence_ok"] = bool(numeric_payload.get("reweight_confidence_ok", free_energy.get("reweight_confidence_ok", False)))
    for key in (
        "dg_selected_kj_mol",
        "dg_reweighted_kj_mol",
        "dg_pop_kj_mol",
        "dg_pop_jeff_kj_mol",
        "dg_exp_kj_mol",
        "dg_cum_kj_mol",
        "ci_low_kj_mol",
        "ci_high_kj_mol",
        "reweight_ci_low_kj_mol",
        "reweight_ci_high_kj_mol",
        "dg_rescored_state_expavg_kj_mol",
        "dg_rescored_state_blockexpavg_kj_mol",
        "dg_rescored_state_expavg_strain_kj_mol",
        "dg_rescored_state_blockexpavg_strain_kj_mol",
        "dg_rescored_state_mean_kj_mol",
        "dg_ligand_strain_kj_mol",
    ):
        if key in free_energy:
            free_energy[key.replace("_kj_mol", "_kcal_mol")] = float(kj_to_kcal(free_energy.get(key, np.nan)))

    bound_mask = np.asarray(post_payload.get("bound_mask", []), dtype=bool)
    unbound_mask = np.asarray(post_payload.get("unbound_mask", []), dtype=bool)
    contacts, bound_frames = contact_occupancies(atom_rows, xyz_series, ligand_heavy_idx, bound_mask)

    def _mean_if(mask, arr):
        sel = np.asarray(arr, dtype=float)[np.asarray(mask, dtype=bool)]
        if sel.size <= 0:
            return np.nan
        return float(np.mean(sel))

    mean_bound_total = _mean_if(bound_mask, arr_total)
    mean_unbound_total = _mean_if(unbound_mask, arr_total)
    assessment = summarize_generic_checks(
        formal_charge=int(param["formal_charge"]),
        partial_charge_sum=float(param["partial_charge_sum"]),
        mean_bound_total=float(mean_bound_total),
        mean_unbound_total=float(mean_unbound_total),
        free_energy=dict(free_energy),
        top_contacts=list(contacts),
        exp_dg_kj=float(exp_dg_kj),
    )

    summary = {
        "inputs": {
            "target": str(args.target),
            "ligand": str(args.ligand),
            "state_label": str(args.state_label or ""),
            "protein_pdb": str(protein_pdb),
            "sim_ns": float(sim_ns),
            "sample_every_ps": float(args.sample_every_ps),
            "temperature_k": float(args.temperature_k),
            "target_ph": float(args.target_ph),
            "pull_distance_a": float(args.pull_distance_a),
            "pull_k": float(args.pull_k),
            "random_seed": int(args.random_seed),
        },
        "experiment": {
            "exp_dg_kj": float(exp_dg_kj),
            "exp_dg_kcal": float(kj_to_kcal(exp_dg_kj)),
            "exp_source": str(exp_source),
            "measurement": dict((ligand_meta or {}).get("measurement", {}) or {}),
        },
        "ligand": {
            "state_label": str(args.state_label or ""),
            "formal_charge": int(param["formal_charge"]),
            "partial_charge_sum": float(param["partial_charge_sum"]),
            "heavy_atoms": int(Chem.RemoveHs(Chem.Mol(ligand_mol)).GetNumHeavyAtoms()),
            "mw": float(rdMolDescriptors.CalcExactMolWt(Chem.Mol(ligand_mol))),
            "ring_count": int(ligand_ring_count),
            "rotatable_bonds": int(rot_bonds),
        },
        "parameterization": {
            "mol2": str(param["mol2"]),
            "frcmod": str(param["frcmod"]),
            "prmtop": str(param["prmtop"]),
            "inpcrd": str(param["inpcrd"]),
            "ligand_ff_terms": dict(ff_counts),
        },
        "system": {
            "n_atoms": int(n_atoms),
            "ligand_atoms": int(len(ligand_idx)),
            "protein_atoms": int(len(protein_idx)),
            "platform": str(platform_name),
            "steering_corridor_count": int(len(steering_corridors)),
            "steering_switches_used": int(steering_switch_count),
            "far_hold_frames": int(np.sum(np.asarray(far_hold_stable_hist, dtype=bool))),
        },
        "energetics": {
            "samples": int(len(arr_total)),
            "mean_total_kj_mol": float(np.mean(arr_total)),
            "mean_total_kcal_mol": float(kj_to_kcal(np.mean(arr_total))),
            "mean_lj_kj_mol": float(np.mean(arr_lj)),
            "mean_lj_kcal_mol": float(kj_to_kcal(np.mean(arr_lj))),
            "mean_coul_kj_mol": float(np.mean(arr_coul)),
            "mean_coul_kcal_mol": float(kj_to_kcal(np.mean(arr_coul))),
            "mean_bound_total_kj_mol": float(mean_bound_total),
            "mean_bound_total_kcal_mol": float(kj_to_kcal(mean_bound_total)),
            "mean_unbound_total_kj_mol": float(mean_unbound_total),
            "mean_unbound_total_kcal_mol": float(kj_to_kcal(mean_unbound_total)),
            "protein_min_dist_min_ang": float(np.min(arr_protein_min)),
            "protein_min_dist_max_ang": float(np.max(arr_protein_min)),
            "rc_site_com_min_ang": float(np.min(arr_rc_com)),
            "rc_site_com_max_ang": float(np.max(arr_rc_com)),
        },
        "free_energy": dict(free_energy),
        "top_bound_contacts": list(contacts[:12]),
        "assessment": dict(assessment),
        "files": {
            "combined_pdb": str(combined_pdb),
            "dcd": str(out_dcd),
            "final_pdb": str(out_final_pdb),
        },
    }

    summary_path = workdir / "benchmark_probe_summary.json"
    report_path = workdir / "benchmark_probe_report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(report_path, summary, energy_unit=str(args.energy_unit))

    print("Benchmark ABFE probe check")
    print(f"Target: {args.target} | Ligand: {args.ligand} | Platform: {platform_name}")
    print(
        f"Charge sum={float(param['partial_charge_sum']):.3f} vs formal={int(param['formal_charge'])} | "
        f"selected={str(free_energy.get('selected_estimator', 'gated_off'))} | "
        f"confidence={str(free_energy.get('reweight_confidence_quality', 'low'))}"
    )
    print(
        f"ΔG(exp)={format_energy(exp_dg_kj, args.energy_unit)} | "
        f"ΔG(sel)={format_energy(free_energy.get('dg_selected_kj_mol', np.nan), args.energy_unit)} | "
        f"ΔG(rw)={format_energy(free_energy.get('dg_reweighted_kj_mol', np.nan), args.energy_unit)} | "
        f"ΔG(pop,Jeff)={format_energy(free_energy.get('dg_pop_jeff_kj_mol', np.nan), args.energy_unit)}"
    )
    print(
        f"bound={format_energy(mean_bound_total, args.energy_unit)} | "
        f"unbound={format_energy(mean_unbound_total, args.energy_unit)} | "
        f"top_contacts={len(list(contacts or []))}"
    )
    print(f"Overall pass: {bool(assessment.get('overall_pass', False))}")
    print(f"Summary JSON: {summary_path}")
    print(f"Report MD: {report_path}")


if __name__ == "__main__":
    main()
