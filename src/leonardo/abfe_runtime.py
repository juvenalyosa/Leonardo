import contextlib
import io
import json
import os
import re
import sys

import numpy as np

# Shown in every ABFE-lite report, the agent's result card and the GUI.  On trypsin-benzamidine
# (3 replicas x 3 ns, Apple M4 GPU, 2026-10-18) the replicas agreed with each other but the estimators did
# not agree with each other or with experiment (selected -62, reweighted standard-state -33.7, population -2.5, exp-avg +0.7 kJ/mol;
# experiment about -27).  Remove only after a multi-target validation on a dedicated GPU.
ABFE_EXPERIMENTAL_NOTE = (
    "EXPERIMENTAL: ABFE-lite is not validated against experiment. Its ΔG estimators can disagree with each "
    "other and with measured affinities by tens of kJ/mol; use it to compare poses or ligands qualitatively, "
    "not as a binding free energy."
)

# Compact ABFE-lite calibration trained with leave-one-target-out group CV.
# Model origin: benchmark_results/diverse_abfe_panel_best_all_available/calibration_compact/deploy_models.json
ABFE_COMPACT_CALIBRATION = {
    "version": "compact_ridge_v2_2026-02-15_weighted_15targets",
    "feature_cols": [
        "dg_expavg_kj",
        "dg_cumulant_kj",
        "energy_mean_kj",
        "coul_mean_kj",
        "contact_frac_mean",
    ],
    "med": [
        -103.32896243429056,
        -210.66235874932585,
        -51.80361932026551,
        -1.1516357919470126,
        0.6669706896551724,
    ],
    "mu": [
        -105.07374269630864,
        -254.00517274900093,
        -51.358020027413495,
        -1.6597458958285465,
        0.683062073309747,
    ],
    "sd": [
        12.57587394432435,
        241.02707114729597,
        12.97203698741658,
        2.1502741767931104,
        0.1640741726380525,
    ],
    "w": [
        -1.8934763955003795,
        2.7286420466782912,
        5.061550923315438,
        1.6759892480671745,
        3.0953280791780378,
    ],
    "y_mean": -41.426190519446195,
    "alpha": 0.0,
}

ABFE_POCKET_DESCRIPTOR_COLS = [
    "pocket_n_atoms",
    "pocket_n_residues",
    "pocket_volume_bbox_a3",
    "pocket_volume_sphere95_a3",
    "pocket_atom_density_per_a3",
    "pocket_polar_atom_frac",
    "pocket_frac_hydrophobic",
    "pocket_frac_polar",
    "pocket_frac_positive",
    "pocket_frac_negative",
    "pocket_frac_aromatic",
    "pocket_charge_est",
    "pocket_charge_density",
]

ABFE_POCKET_HYDROPHOBIC_RES = {
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "MET",
    "PRO",
    "PHE",
    "TRP",
    "TYR",
    "CYS",
}
ABFE_POCKET_POLAR_RES = {"SER", "THR", "ASN", "GLN", "CYS", "TYR", "HIS", "HID", "HIE", "HSP", "HIP"}
ABFE_POCKET_POSITIVE_RES = {"LYS", "ARG"}
ABFE_POCKET_NEGATIVE_RES = {"ASP", "GLU"}
ABFE_POCKET_HIS_RES = {"HIS", "HID", "HIE", "HSP", "HIP"}
ABFE_POCKET_AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS", "HID", "HIE", "HSP", "HIP"}

ABFE_RUNTIME_DEPLOY_ENV = "PROTLIGINTERACT_ABFE_DEPLOY_JSON"
ABFE_RUNTIME_DEPLOY_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "abfe_deploy_models.json"),
    os.path.join(
        os.path.dirname(__file__),
        "benchmark_results",
        "diverse_abfe_panel_best_all_available_5ns_rerun",
        "calibration_endpoint_homogeneous_with_pocket_ic50norm",
        "all",
        "deploy_models.json",
    ),
]

_ABFE_RUNTIME_DEPLOY_CACHE = None
_ABFE_RUNTIME_DEPLOY_SOURCE = ""
OPENMM_PLATFORM_OVERRIDE_ENV = "PROTLIGINTERACT_OPENMM_PLATFORM"
# the documented Leonardo name; code_v2 already honoured it, but this plan did not, so on macOS a request for
# a GPU through it ended on the CPU
OPENMM_PLATFORM_OVERRIDE_ENVS = (OPENMM_PLATFORM_OVERRIDE_ENV, "LEONARDO_OPENMM_PLATFORM")


def openmm_platform_override():
    """The platform the user asked for through the environment ('' if none)."""
    for name in OPENMM_PLATFORM_OVERRIDE_ENVS:
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            return value
    return ""
OPENMM_CPU_THREADS_ENV = "PROTLIGINTERACT_OPENMM_CPU_THREADS"
OPENMM_GPU_PLATFORM_NAMES = {"CUDA", "Metal", "HIP", "OpenCL"}
ABFE_STEERING_POLICY_MODEL_ENV = "PROTLIGINTERACT_ABFE_STEERING_POLICY_JSON"
ABFE_STEERING_ACTION_NAMES = (
    "keep",
    "increase_pull",
    "reduce_pull",
    "switch_corridor",
    "detach_boost",
    "extend_hold",
)
ABFE_STEERING_POLICY_FEATURE_NAMES = (
    "bias",
    "phase_pull",
    "phase_far_hold",
    "phase_return",
    "actionable",
    "stalled",
    "soft_clash",
    "hard_clash",
    "energy_spike",
    "far_ready",
    "far_hold_sustain",
    "can_switch",
    "k_fraction",
    "escape_gap_fraction",
    "contact_pressure",
    "size_scale",
    "charge_density",
    "formal_charge_scaled",
    "ring_count_scale",
    "rotatable_bonds_scale",
    "pocket_n_atoms_scaled",
    "pocket_volume_bbox_scaled",
    "pocket_atom_density_scaled",
    "pocket_frac_hydrophobic",
    "pocket_frac_polar",
    "pocket_charge_density_scaled",
    "metal_contact_expected",
    "latest_min_dist_scaled",
    "latest_site_com_scaled",
    "latest_contact_scaled",
    "hold_progress_fraction",
    "hold_gap_fraction",
    "recent_far_hold_break",
)

ABFE_STEERING_POLICY_OOD_FEATURE_NAMES = (
    "size_scale",
    "formal_charge_scaled",
    "ring_count_scale",
    "rotatable_bonds_scale",
    "metal_contact_expected",
    "pocket_n_atoms_scaled",
    "pocket_volume_bbox_scaled",
    "pocket_atom_density_scaled",
    "pocket_frac_hydrophobic",
    "pocket_frac_polar",
    "pocket_charge_density_scaled",
)

_ABFE_STEERING_POLICY_MODEL_CACHE = {}


def _nan_pocket_descriptor_row():
    return {k: np.nan for k in ABFE_POCKET_DESCRIPTOR_COLS}


def _normalize_abfe_policy_pocket_desc(pocket_desc=None):
    out = _nan_pocket_descriptor_row()
    src = dict(pocket_desc or {})
    for key in ABFE_POCKET_DESCRIPTOR_COLS:
        try:
            vv = float(src.get(key, np.nan))
        except Exception:
            vv = np.nan
        out[key] = float(vv) if np.isfinite(vv) else np.nan
    return dict(out)


def infer_abfe_steering_scaffold_key(smiles=""):
    smi = str(smiles or "").strip()
    if not smi:
        return ""
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
    except Exception:
        return str(smi)
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return str(smi)
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        except Exception:
            scaffold = None
        if scaffold is not None and scaffold.GetNumAtoms() > 0:
            txt = str(Chem.MolToSmiles(scaffold, isomericSmiles=True) or "").strip()
            if txt:
                return txt
        txt = str(Chem.MolToSmiles(mol, isomericSmiles=True) or "").strip()
        return txt if txt else str(smi)
    except Exception:
        return str(smi)


def _sanitize_md_temperature_k(value, default=300.0):
    note = ""
    try:
        out = float(value)
    except Exception:
        return float(default), "defaulted_invalid_temperature"
    if (not np.isfinite(out)) or out <= 0.0:
        return float(default), "defaulted_invalid_temperature"
    if out < 50.0:
        return 50.0, "clamped_low_temperature"
    if out > 1000.0:
        return 1000.0, "clamped_high_temperature"
    return float(out), note


def _sanitize_md_timestep_ps(value, default=0.001):
    note = ""
    try:
        out = float(value)
    except Exception:
        return float(default), "defaulted_invalid_timestep"
    if (not np.isfinite(out)) or out <= 0.0:
        return float(default), "defaulted_invalid_timestep"
    if out > 0.004:
        return 0.004, "clamped_timestep_to_4fs"
    return float(out), note


def _sanitize_md_friction_inv_ps(value, default=5.0):
    note = ""
    try:
        out = float(value)
    except Exception:
        return float(default), "defaulted_invalid_friction"
    if (not np.isfinite(out)) or out <= 0.0:
        return float(default), "defaulted_invalid_friction"
    if out > 100.0:
        return 100.0, "clamped_high_friction"
    return float(out), note


def _sanitize_md_duration_ns(value, default=1.0):
    note = ""
    try:
        out = float(value)
    except Exception:
        return float(default), "defaulted_invalid_duration"
    if (not np.isfinite(out)) or out <= 0.0:
        return float(default), "defaulted_invalid_duration"
    if out < 0.2:
        return 0.2, "clamped_short_duration"
    return float(out), note


_PRECISION_SUPPORT = {}


def openmm_precision_supported(platform_name, precision="mixed"):
    """Whether this machine's `platform_name` accepts `precision` (one tiny context, cached per process).
    Apple GPUs have no 64-bit floats: their OpenCL rejects 'mixed' ("No compatible OpenCL platform is available")
    and only runs 'single'. A platform that is not installed, or fails for any precision, counts as supported:
    only a definite "single works, this precision does not" downgrades."""
    key = (str(platform_name), str(precision))
    if key in _PRECISION_SUPPORT:
        return _PRECISION_SUPPORT[key]
    ok = True
    try:
        import openmm as mm

        names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
        if str(platform_name) in names:
            plat = mm.Platform.getPlatformByName(str(platform_name))

            def _try(prec):
                system = mm.System()
                system.addParticle(1.0)
                try:
                    ctx = mm.Context(system, mm.VerletIntegrator(0.001), plat, {"Precision": prec})
                    del ctx
                    return True
                except Exception:
                    return False

            ok = _try(str(precision)) or not _try("single")
    except Exception:
        ok = True
    _PRECISION_SUPPORT[key] = ok
    return ok


def openmm_gpu_properties(platform_name):
    """Context properties for a GPU platform: mixed precision where the device supports it, else single."""
    if str(platform_name or "").upper() not in {"CUDA", "OPENCL", "HIP", "METAL"}:
        return {}, []
    if openmm_precision_supported(platform_name, "mixed"):
        return {"Precision": "mixed"}, ["added_mixed_precision_for_gpu_platform"]
    return {"Precision": "single"}, ["gpu_has_no_mixed_precision_using_single"]



def openmm_cpu_thread_budget(cpu_count=None, env_value=None):
    """
    Threads for the OpenMM CPU platform.

    OpenMM uses every core by default. Inside PyMOL that leaves nothing for the
    GUI, the window server and the kernel's own daemons; long ABFE runs on a
    laptop can starve them until macOS panics (watchdog timeout). Keep two cores
    free unless the user sets PROTLIGINTERACT_OPENMM_CPU_THREADS.
    """
    raw = os.environ.get(OPENMM_CPU_THREADS_ENV, "") if env_value is None else env_value
    try:
        forced = int(str(raw or "").strip())
    except ValueError:
        forced = 0
    if forced > 0:
        return int(forced)
    try:
        n = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    except (TypeError, ValueError):
        n = 1
    n = max(1, n)
    return int(max(1, n - 2)) if n > 2 else 1


def _normalize_platform_properties(platform_name, platform_properties, cpu_count=None):
    props = dict(platform_properties or {})
    picked = str(platform_name or "").strip()
    notes = []
    if not picked:
        return props, notes
    existing_keys = {str(k).strip().lower(): k for k in props.keys()}
    if "precision" not in existing_keys:
        gpu_props, gpu_notes = openmm_gpu_properties(picked)
        props.update(gpu_props)
        notes.extend(gpu_notes)
    if picked.upper() == "CPU" and "threads" not in existing_keys:
        threads = openmm_cpu_thread_budget(cpu_count=cpu_count)
        props["Threads"] = str(int(threads))
        notes.append(f"cpu_threads_capped:{int(threads)}")
    return props, notes


def _canonical_openmm_platform_name(value):
    txt = str(value or "").strip()
    if not txt:
        return ""
    alias_map = {
        "cuda": "CUDA",
        "metal": "Metal",
        "hip": "HIP",
        "opencl": "OpenCL",
        "cpu": "CPU",
        "reference": "Reference",
        "ref": "Reference",
        "auto": "",
        "default": "",
        "none": "",
    }
    return str(alias_map.get(txt.lower(), ""))


def build_openmm_platform_selection_plan(
    available_platform_names=None,
    *,
    platform_override="",
    os_name="",
    cpu_count=None,
):
    available = {}
    for name in list(available_platform_names or []):
        canonical = _canonical_openmm_platform_name(name)
        if canonical and canonical not in available:
            available[canonical] = str(name)

    override_raw = str(platform_override or openmm_platform_override() or "").strip()
    override = _canonical_openmm_platform_name(override_raw)
    notes = []
    order = []

    if override_raw and not override:
        notes.append(f"ignored_unknown_platform_override:{override_raw}")
    if override:
        notes.append(f"explicit_platform_override:{override}")
        order.append(str(override))

    normalized_os = str(os_name or sys.platform or "").strip().lower()
    on_macos = normalized_os.startswith("darwin") or normalized_os.startswith("mac")
    default_order = ["CUDA", "Metal", "HIP", "OpenCL", "CPU", "Reference"]
    if on_macos and not override:
        # Metal can hard-crash some macOS/OpenMM combinations during Context creation.
        default_order = ["CPU", "Reference", "Metal", "OpenCL", "HIP", "CUDA"]
        notes.append("macos_auto_prefers_cpu")

    for name in default_order:
        if name not in order:
            order.append(str(name))

    picked = ""
    for name in order:
        if name in available:
            picked = str(name)
            break

    props, prop_notes = _normalize_platform_properties(picked, {}, cpu_count=cpu_count)
    notes.extend(list(prop_notes))
    if override and override not in available:
        notes.append(f"requested_platform_unavailable:{override}")

    return {
        "ok": True,
        "override_env": str(OPENMM_PLATFORM_OVERRIDE_ENV),
        "platform_override_raw": str(override_raw),
        "platform_override": str(override),
        "available_platform_names": list(available.keys()),
        "candidate_order": list(order),
        "picked_platform_name": str(picked),
        "platform_properties": dict(props),
        "notes": list(notes),
    }


def compute_runtime_pocket_descriptors(
    atom_infos,
    coords_nm,
    protein_ids,
    ligand_ids,
    pocket_cutoff_a=6.0,
    infer_element_fn=None,
):
    """
    Compute pocket descriptors aligned with augment_abfe_with_pocket_descriptors.py
    from in-memory protein/ligand coordinates.
    """
    desc = _nan_pocket_descriptor_row()
    try:
        if coords_nm is None:
            return desc
        p_idx = np.asarray(protein_ids, dtype=int)
        l_idx = np.asarray(ligand_ids, dtype=int)
        if p_idx.size == 0 or l_idx.size == 0:
            return desc
        xyz = np.asarray(coords_nm, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            return desc
        if int(np.max(p_idx)) >= xyz.shape[0] or int(np.max(l_idx)) >= xyz.shape[0]:
            return desc
        p_xyz_a = np.asarray(xyz[p_idx], dtype=float) * 10.0
        l_xyz_a = np.asarray(xyz[l_idx], dtype=float) * 10.0
        if p_xyz_a.size == 0 or l_xyz_a.size == 0:
            return desc
        diff = p_xyz_a[:, None, :] - l_xyz_a[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        min_d2 = np.min(d2, axis=1)
        cutoff2 = float(pocket_cutoff_a) ** 2
        mask = min_d2 <= cutoff2
        if not np.any(mask):
            k = int(min(len(min_d2), 64))
            if k <= 0:
                return desc
            idx = np.argpartition(min_d2, k - 1)[:k]
            mask = np.zeros(len(min_d2), dtype=bool)
            mask[idx] = True
        atom_idx = np.where(mask)[0]
        n_atoms = int(atom_idx.size)
        if n_atoms <= 0:
            return desc
        pcoords = p_xyz_a[atom_idx]
        span = np.ptp(pcoords, axis=0)
        span = np.where(np.isfinite(span), np.maximum(span, 0.1), 0.1)
        bbox_vol = float(span[0] * span[1] * span[2])
        lig_center = np.mean(l_xyz_a, axis=0)
        radial = np.linalg.norm(pcoords - lig_center[None, :], axis=1)
        r95 = float(np.percentile(radial, 95.0)) if radial.size else 0.1
        r95 = max(r95, 0.1)
        sphere95_vol = float((4.0 / 3.0) * np.pi * (r95 ** 3))
        atom_density = float(n_atoms / max(sphere95_vol, 1e-6))
        sel_global = p_idx[atom_idx]
        residue_seen = set()
        res_names = []
        polar_atoms = 0
        for gi in sel_global:
            ii = int(gi)
            if ii < 0 or ii >= len(atom_infos):
                continue
            info = atom_infos[ii] if isinstance(atom_infos[ii], dict) else {}
            el = str(info.get("element", "")).strip().upper()
            if (not el) and callable(infer_element_fn):
                try:
                    el = str(infer_element_fn(str(info.get("name", "")), "")).strip().upper()
                except Exception:
                    el = ""
            if el in ("N", "O", "S"):
                polar_atoms += 1
            resn = str(info.get("resn", "")).strip().upper()
            chain = str(info.get("chain", "")).strip()
            resi = str(info.get("resi", "")).strip()
            if not resn:
                continue
            key = (chain, resi, resn)
            if key in residue_seen:
                continue
            residue_seen.add(key)
            res_names.append(resn)
        n_res = int(len(res_names))
        polar_atom_frac = float(polar_atoms / float(max(n_atoms, 1)))
        if n_res > 0:
            frac_h = float(np.mean([rn in ABFE_POCKET_HYDROPHOBIC_RES for rn in res_names]))
            frac_p = float(np.mean([rn in ABFE_POCKET_POLAR_RES for rn in res_names]))
            frac_pos = float(np.mean([rn in ABFE_POCKET_POSITIVE_RES for rn in res_names]))
            frac_neg = float(np.mean([rn in ABFE_POCKET_NEGATIVE_RES for rn in res_names]))
            frac_ar = float(np.mean([rn in ABFE_POCKET_AROMATIC_RES for rn in res_names]))
            charge_est = 0.0
            for rn in res_names:
                if rn in ABFE_POCKET_POSITIVE_RES:
                    charge_est += 1.0
                elif rn in ABFE_POCKET_NEGATIVE_RES:
                    charge_est -= 1.0
                elif rn in ABFE_POCKET_HIS_RES:
                    charge_est += 0.1
            charge_density = float(charge_est / float(max(n_res, 1)))
        else:
            frac_h = frac_p = frac_pos = frac_neg = frac_ar = np.nan
            charge_est = np.nan
            charge_density = np.nan
        desc.update(
            {
                "pocket_n_atoms": float(n_atoms),
                "pocket_n_residues": float(n_res),
                "pocket_volume_bbox_a3": float(bbox_vol),
                "pocket_volume_sphere95_a3": float(sphere95_vol),
                "pocket_atom_density_per_a3": float(atom_density),
                "pocket_polar_atom_frac": float(polar_atom_frac),
                "pocket_frac_hydrophobic": float(frac_h),
                "pocket_frac_polar": float(frac_p),
                "pocket_frac_positive": float(frac_pos),
                "pocket_frac_negative": float(frac_neg),
                "pocket_frac_aromatic": float(frac_ar),
                "pocket_charge_est": float(charge_est),
                "pocket_charge_density": float(charge_density),
            }
        )
    except Exception:
        pass
    return desc


def build_abfe_local_pocket_restraint_plan_payload(
    *,
    atom_rows=None,
    coords_nm=None,
    protein_atom_idx=None,
    ligand_atom_idx=None,
    ligand_heavy_idx=None,
    pocket_desc=None,
    base_backbone_k_kj_nm2=3600.0,
    base_sidechain_k_kj_nm2=1300.0,
):
    rows = list(atom_rows or [])
    prot_idx = np.asarray(protein_atom_idx if protein_atom_idx is not None else [], dtype=int)
    lig_idx = np.asarray(ligand_atom_idx if ligand_atom_idx is not None else [], dtype=int)
    lig_heavy = np.asarray(ligand_heavy_idx if ligand_heavy_idx is not None else [], dtype=int)
    try:
        xyz = np.asarray(coords_nm, dtype=float)
    except Exception:
        xyz = np.asarray([], dtype=float)

    out = {
        "ok": False,
        "reason": "",
        "rows": [],
        "n_softened": 0,
        "n_backbone_softened": 0,
        "n_sidechain_softened": 0,
        "inner_shell_a": np.nan,
        "middle_shell_a": np.nan,
        "outer_shell_a": np.nan,
        "flex_strength": np.nan,
    }
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        out["reason"] = "invalid_coords"
        return out
    if prot_idx.size == 0 or (lig_idx.size == 0 and lig_heavy.size == 0):
        out["reason"] = "missing_groups"
        return out

    lig_anchor = np.asarray(lig_heavy if lig_heavy.size > 0 else lig_idx, dtype=int)
    if lig_anchor.size == 0:
        out["reason"] = "missing_ligand_anchor"
        return out

    pocket_row = _normalize_abfe_policy_pocket_desc(pocket_desc)
    try:
        pocket_density = float(pocket_row.get("pocket_atom_density_per_a3", np.nan))
    except Exception:
        pocket_density = np.nan
    try:
        pocket_hydrophobic = float(pocket_row.get("pocket_frac_hydrophobic", np.nan))
    except Exception:
        pocket_hydrophobic = np.nan
    buried_scale = float(np.clip(pocket_density / 0.04, 0.0, 2.0)) if np.isfinite(pocket_density) else 0.0
    pocket_hydrophobic_scale = float(np.clip(pocket_hydrophobic, 0.0, 1.5)) if np.isfinite(pocket_hydrophobic) else 0.0
    size_scale = float(np.clip((float(lig_anchor.size) - 10.0) / 22.0, 0.0, 1.0))
    flex_strength = float(
        np.clip(0.26 + (0.12 * size_scale) + (0.10 * buried_scale) + (0.06 * pocket_hydrophobic_scale), 0.24, 0.56)
    )
    inner_shell_a = float(np.clip(4.2 + (0.8 * size_scale) + (0.4 * buried_scale), 4.0, 5.8))
    middle_shell_a = float(np.clip(inner_shell_a + 2.0 + (0.3 * size_scale), inner_shell_a + 1.6, 7.8))
    outer_shell_a = float(np.clip(middle_shell_a + 2.0 + (0.3 * buried_scale), middle_shell_a + 1.6, 10.0))

    side_inner_scale = float(np.clip(1.0 - (0.52 * flex_strength), 0.48, 0.78))
    side_middle_scale = float(np.clip(1.0 - (0.28 * flex_strength), 0.70, 0.88))
    side_outer_scale = float(np.clip(1.0 - (0.10 * flex_strength), 0.88, 0.97))
    bb_inner_scale = float(np.clip(1.0 - (0.14 * flex_strength), 0.84, 0.95))
    bb_middle_scale = float(np.clip(1.0 - (0.06 * flex_strength), 0.92, 0.98))
    bb_outer_scale = 1.0

    lig_xyz = np.asarray(xyz[lig_anchor], dtype=float)
    residue_min_dist = {}
    residue_members = {}
    backbone_names = {"N", "CA", "C", "O"}
    for idx in prot_idx.tolist():
        i = int(idx)
        if i < 0 or i >= len(rows):
            continue
        row = dict(rows[i] or {})
        element = str(row.get("element", "")).strip().upper()
        if element == "H":
            continue
        res_key = (
            str(row.get("chain", "") or ""),
            str(row.get("resi", "") or ""),
            str(row.get("resn", "") or ""),
        )
        residue_members.setdefault(res_key, []).append(i)
    for res_key, members in residue_members.items():
        try:
            res_xyz = np.asarray([xyz[int(i)] for i in members], dtype=float)
            dmat = np.linalg.norm(res_xyz[:, None, :] - lig_xyz[None, :, :], axis=2)
            residue_min_dist[res_key] = float(10.0 * np.min(dmat))
        except Exception:
            residue_min_dist[res_key] = np.nan

    rows_out = []
    n_softened = 0
    n_backbone_softened = 0
    n_sidechain_softened = 0
    for idx in prot_idx.tolist():
        i = int(idx)
        if i < 0 or i >= len(rows):
            continue
        row = dict(rows[i] or {})
        element = str(row.get("element", "")).strip().upper()
        if element == "H":
            continue
        atom_name = str(row.get("name", "")).strip().upper()
        res_key = (
            str(row.get("chain", "") or ""),
            str(row.get("resi", "") or ""),
            str(row.get("resn", "") or ""),
        )
        dmin_a = float(residue_min_dist.get(res_key, np.nan))
        is_backbone = bool(atom_name in backbone_names)
        base_k = float(base_backbone_k_kj_nm2 if is_backbone else base_sidechain_k_kj_nm2)
        scale = 1.0
        shell = "global"
        if np.isfinite(dmin_a):
            if dmin_a <= inner_shell_a:
                scale = float(bb_inner_scale if is_backbone else side_inner_scale)
                shell = "inner"
            elif dmin_a <= middle_shell_a:
                scale = float(bb_middle_scale if is_backbone else side_middle_scale)
                shell = "middle"
            elif dmin_a <= outer_shell_a:
                scale = float(bb_outer_scale if is_backbone else side_outer_scale)
                shell = "outer"
        k_rest = float(base_k * scale)
        if is_backbone:
            k_rest = float(max(2200.0, k_rest))
        else:
            k_rest = float(max(420.0, k_rest))
        softened = bool(scale < 0.995)
        if softened:
            n_softened += 1
            if is_backbone:
                n_backbone_softened += 1
            else:
                n_sidechain_softened += 1
        rows_out.append(
            {
                "index": int(i),
                "k_rest_kj_nm2": float(k_rest),
                "scale": float(scale),
                "residue_min_dist_a": float(dmin_a) if np.isfinite(dmin_a) else np.nan,
                "shell": str(shell),
                "is_backbone": bool(is_backbone),
            }
        )

    out.update(
        {
            "ok": True,
            "reason": "ok",
            "rows": list(rows_out),
            "n_softened": int(n_softened),
            "n_backbone_softened": int(n_backbone_softened),
            "n_sidechain_softened": int(n_sidechain_softened),
            "inner_shell_a": float(inner_shell_a),
            "middle_shell_a": float(middle_shell_a),
            "outer_shell_a": float(outer_shell_a),
            "flex_strength": float(flex_strength),
        }
    )
    return out


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _abfe_standard_state_from_state_ratio(dg_state_kj=np.nan, v_eff_nm3=np.nan, temperature_k=300.0):
    """Convert a sampled bound/unbound state ratio into a 1 M standard-state estimate.

    The state-ratio DeltaG is dimensionless: it compares probabilities inside
    the sampled bound and unbound definitions.  Kd additionally depends on the
    effective unbound volume sampled by the ligand, so the standard-state
    correction must be reported explicitly instead of mixing the two quantities.
    """
    dg_state = _safe_float(dg_state_kj)
    v_eff = _safe_float(v_eff_nm3)
    temp = _safe_float(temperature_k)
    k_b = 0.008314462618
    kbt = float(k_b * temp) if np.isfinite(temp) and temp > 0.0 else np.nan
    out = {
        "ok": False,
        "reason": "invalid_input",
        "kbt": float(kbt) if np.isfinite(kbt) else np.nan,
        "v_eff_nm3": float(v_eff) if np.isfinite(v_eff) else np.nan,
        "c_eff_m": np.nan,
        "standard_state_correction_kj": np.nan,
        "dg_state_kj": float(dg_state) if np.isfinite(dg_state) else np.nan,
        "dg_standard_kj": np.nan,
        "kd_standard_m": np.nan,
        "p_kd_standard": np.nan,
    }
    if not np.isfinite(dg_state):
        out["reason"] = "invalid_state_delta_g"
        return out
    if not np.isfinite(kbt) or kbt <= 0.0:
        out["reason"] = "invalid_temperature"
        return out
    if not np.isfinite(v_eff) or v_eff <= 0.0:
        out["reason"] = "invalid_effective_volume"
        return out
    # 1 molecule / nm^3 = 1.66054 mol/L.  Dividing by the sampled unbound
    # volume gives the concentration represented by one ligand in that volume.
    c_eff_m = float(1.66054 / max(float(v_eff), 1.0e-9))
    correction = float(kbt * np.log(max(c_eff_m, 1.0e-300)))
    dg_standard = float(dg_state + correction)
    kd_m = float(np.exp(np.clip(dg_standard / kbt, -700.0, 700.0)))
    p_kd = float(-np.log10(max(kd_m, 1.0e-300)))
    out.update(
        {
            "ok": True,
            "reason": "ok",
            "c_eff_m": float(c_eff_m),
            "standard_state_correction_kj": float(correction),
            "dg_standard_kj": float(dg_standard),
            "kd_standard_m": float(kd_m),
            "p_kd_standard": float(p_kd),
        }
    )
    return out


def _abfe_float_series(values=None, n=None, fill=np.nan):
    if values is None:
        arr = np.asarray([], dtype=float)
    else:
        try:
            arr = np.asarray(list(values), dtype=float)
        except Exception:
            arr = np.asarray([], dtype=float)
    if n is None:
        return np.asarray(arr, dtype=float)
    n_i = _abfe_positive_int(n, default=0, minimum=0)
    out = np.full(int(n_i), float(fill), dtype=float)
    if n_i > 0 and arr.size > 0:
        copy_n = min(int(n_i), int(arr.size))
        out[:copy_n] = np.asarray(arr[:copy_n], dtype=float)
    return out


def _abfe_bool_series(values=None, n=None, fill=False):
    if values is None:
        arr = np.asarray([], dtype=bool)
    else:
        try:
            arr = np.asarray(list(values), dtype=bool)
        except Exception:
            arr = np.asarray([], dtype=bool)
    if n is None:
        return np.asarray(arr, dtype=bool)
    n_i = _abfe_positive_int(n, default=0, minimum=0)
    out = np.full(int(n_i), bool(fill), dtype=bool)
    if n_i > 0 and arr.size > 0:
        copy_n = min(int(n_i), int(arr.size))
        out[:copy_n] = np.asarray(arr[:copy_n], dtype=bool)
    return out


def _abfe_int_series(values=None, n=None, start=1):
    if values is None:
        arr = np.asarray([], dtype=int)
    else:
        try:
            arr = np.asarray(list(values), dtype=int)
        except Exception:
            arr = np.asarray([], dtype=int)
    if n is None:
        return np.asarray(arr, dtype=int)
    n_i = _abfe_positive_int(n, default=0, minimum=0)
    if n_i <= 0:
        return np.asarray([], dtype=int)
    out = np.arange(int(start), int(start) + int(n_i), dtype=int)
    if arr.size > 0:
        copy_n = min(int(n_i), int(arr.size))
        out[:copy_n] = np.asarray(arr[:copy_n], dtype=int)
    return out


def _abfe_positive_int(value, default=1, minimum=1):
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return int(max(int(minimum), out))


def _abfe_downsample_indices(n_total, max_points=0, include_last=True):
    n = _abfe_positive_int(n_total, default=0, minimum=0)
    try:
        max_i = int(max_points or 0)
    except Exception:
        max_i = 0
    if n <= 0:
        return np.asarray([], dtype=int)
    if max_i <= 0 or n <= max_i:
        return np.arange(n, dtype=int)
    max_i = max(1, int(max_i))
    idx = np.unique(np.linspace(0, n - 1, max_i, dtype=int))
    if bool(include_last) and idx.size and int(idx[-1]) != (n - 1):
        idx = np.unique(np.append(idx, n - 1)).astype(int)
    return np.asarray(idx, dtype=int)


def build_abfe_local_resource_plan(
    *,
    n_atoms=0,
    n_samples=0,
    sim_ns=0.0,
    sample_every_ps=1.0,
    requested_max_traj_states=1500,
    coordinate_memory_budget_mb=256.0,
    live_plot_max_points=800,
    rescore_max_frames=96,
    rescore_memory_budget_mb=192.0,
):
    """
    Memory guard for ABFE-lite on local CPU/GPU environments.

    Numeric observables are cheap and remain sampled at every interval. Full
    coordinate frames dominate RAM, so they are stored only at a stride that
    fits inside the coordinate budget and still includes the final sample.
    """
    n_atoms_i = _abfe_positive_int(n_atoms, default=0, minimum=0)
    try:
        n_samples_i = int(n_samples or 0)
    except Exception:
        n_samples_i = 0
    if n_samples_i <= 0:
        try:
            sim_ps = max(0.0, float(sim_ns) * 1000.0)
            sp = max(1.0e-6, float(sample_every_ps))
            n_samples_i = int(max(1, np.ceil(sim_ps / sp)))
        except Exception:
            n_samples_i = 1
    n_samples_i = max(1, int(n_samples_i))

    max_states_req = _abfe_positive_int(requested_max_traj_states, default=1500, minimum=1)
    plot_max = _abfe_positive_int(live_plot_max_points, default=800, minimum=25)
    rescore_max = _abfe_positive_int(rescore_max_frames, default=96, minimum=1)
    try:
        coord_budget = float(coordinate_memory_budget_mb)
    except Exception:
        coord_budget = 256.0
    coord_budget = float(max(8.0, coord_budget))
    try:
        rescore_budget = float(rescore_memory_budget_mb)
    except Exception:
        rescore_budget = 192.0
    rescore_budget = float(max(8.0, rescore_budget))

    bytes_per_state = int(max(1, n_atoms_i) * 3 * np.dtype(np.float32).itemsize)
    mb_per_state = float(bytes_per_state) / 1048576.0
    raw_all_states_mb = float(mb_per_state * n_samples_i)
    max_states_by_memory = int(max(1, np.floor(coord_budget / max(mb_per_state, 1.0e-9))))
    max_saved_states = int(min(max_states_req, max_states_by_memory, n_samples_i))
    saved_indices = _abfe_downsample_indices(
        n_samples_i,
        max_points=max_saved_states,
        include_last=True,
    )
    state_stride = int(max(1, np.ceil(float(n_samples_i) / float(max(1, saved_indices.size)))))
    plot_stride = int(max(1, np.ceil(float(n_samples_i) / float(max(1, plot_max)))))
    view_stride = int(max(1, state_stride))

    # GBSA rescoring creates OpenMM contexts and selected coordinate copies.
    # Keep its frame count lower than the visual trajectory budget.
    max_rescore_by_memory = int(max(1, np.floor(rescore_budget / max(mb_per_state, 1.0e-9))))
    max_rescore_frames_i = int(min(rescore_max, max_rescore_by_memory, max(1, saved_indices.size)))

    notes = []
    if raw_all_states_mb > coord_budget:
        notes.append("coordinate_trajectory_decimated")
    if n_samples_i > plot_max:
        notes.append("live_plot_decimated")
    if max_rescore_frames_i < max(1, saved_indices.size):
        notes.append("gbsa_rescore_decimated")

    return {
        "ok": True,
        "n_atoms": int(n_atoms_i),
        "n_samples": int(n_samples_i),
        "bytes_per_state": int(bytes_per_state),
        "mb_per_state": float(mb_per_state),
        "coordinate_memory_budget_mb": float(coord_budget),
        "estimated_all_states_mb": float(raw_all_states_mb),
        "requested_max_traj_states": int(max_states_req),
        "max_saved_states": int(max_saved_states),
        "saved_sample_indices": [int(i) for i in np.asarray(saved_indices, dtype=int)],
        "estimated_saved_state_count": int(saved_indices.size),
        "estimated_saved_states_mb": float(mb_per_state * int(saved_indices.size)),
        "state_save_stride": int(state_stride),
        "view_update_stride": int(view_stride),
        "plot_update_stride": int(plot_stride),
        "live_plot_max_points": int(plot_max),
        "rescore_max_frames": int(max_rescore_frames_i),
        "rescore_memory_budget_mb": float(rescore_budget),
        "notes": list(notes),
        "memory_guard_active": bool(raw_all_states_mb > coord_budget or n_samples_i > plot_max),
    }


def build_abfe_state_save_decision(
    *,
    sample_index=0,
    n_samples=0,
    saved_sample_indices=None,
    state_save_stride=1,
):
    s = _abfe_positive_int(sample_index, default=0, minimum=0)
    n = _abfe_positive_int(n_samples, default=0, minimum=0)
    if saved_sample_indices is not None:
        if isinstance(saved_sample_indices, set):
            save_set = saved_sample_indices
        else:
            try:
                save_set = {int(i) for i in list(saved_sample_indices or [])}
            except Exception:
                save_set = set()
        should_store = bool(s in save_set)
    else:
        stride = _abfe_positive_int(state_save_stride, default=1, minimum=1)
        should_store = bool((s % stride) == 0)
    if n > 0 and s == (n - 1):
        should_store = True
    return {
        "ok": True,
        "sample_index": int(s),
        "n_samples": int(n),
        "should_store": bool(should_store),
    }


def _abfe_effective_sample_size(weights):
    arr = np.asarray(weights if weights is not None else [], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size <= 0:
        return np.nan
    denom = float(np.sum(np.square(finite)))
    if denom <= 0.0:
        return np.nan
    numer = float(np.sum(finite))
    return float((numer * numer) / denom)


def _abfe_weighted_expavg(samples, weights, beta):
    arr = np.asarray(samples if samples is not None else [], dtype=float)
    if arr.size <= 0 or (not np.isfinite(beta)) or beta <= 0.0:
        return np.nan
    finite = np.isfinite(arr)
    if weights is not None:
        ww = np.asarray(weights, dtype=float)
        if ww.size == arr.size:
            finite = np.logical_and(finite, np.isfinite(ww))
            finite = np.logical_and(finite, ww > 0.0)
        else:
            ww = np.asarray([], dtype=float)
    else:
        ww = np.asarray([], dtype=float)
    if not np.any(finite):
        return np.nan
    arr = np.asarray(arr[finite], dtype=float)
    if ww.size == finite.size:
        ww = np.asarray(ww[finite], dtype=float)
    else:
        ww = np.ones(arr.size, dtype=float)
    sum_w = float(np.sum(ww))
    if sum_w <= 0.0:
        return np.nan
    e_shift = float(np.min(arr))
    expo = np.exp(np.clip(-beta * (arr - e_shift), -120.0, 120.0))
    weighted = float(np.sum(ww * expo) / sum_w)
    if weighted <= 0.0:
        return np.nan
    return float(e_shift - (1.0 / beta) * np.log(weighted))


def _abfe_default_block_size(n_samples, minimum=3, maximum=12):
    try:
        n_i = int(n_samples)
    except Exception:
        n_i = 0
    if n_i <= 0:
        return int(max(1, int(minimum)))
    return int(np.clip(round(np.sqrt(float(n_i))), int(max(1, int(minimum))), int(max(int(minimum), int(maximum)))))


def _abfe_weighted_mean(samples, weights=None):
    arr = np.asarray(samples if samples is not None else [], dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.nan
    arr = np.asarray(arr[finite], dtype=float)
    ww = np.asarray([], dtype=float)
    if weights is not None:
        try:
            ww = np.asarray(weights, dtype=float)
        except Exception:
            ww = np.asarray([], dtype=float)
    if ww.size == finite.size:
        ww = np.asarray(ww[finite], dtype=float)
    else:
        ww = np.asarray([], dtype=float)
    if ww.size == arr.size and np.any(np.isfinite(ww)):
        ww = np.asarray(np.clip(ww, 0.0, None), dtype=float)
        if float(np.sum(ww)) > 0.0:
            return float(np.average(arr, weights=ww))
    return float(np.mean(arr))


def _abfe_winsorize(samples, lower_q=0.08, upper_q=0.92):
    arr = np.asarray(samples if samples is not None else [], dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.asarray([], dtype=float)
    arr = np.asarray(arr[finite], dtype=float)
    try:
        low_q = float(lower_q)
    except Exception:
        low_q = 0.08
    try:
        high_q = float(upper_q)
    except Exception:
        high_q = 0.92
    low_q = float(np.clip(low_q, 0.0, 0.49))
    high_q = float(np.clip(high_q, 0.51, 1.0))
    if arr.size < 4 or high_q <= low_q:
        return np.asarray(arr, dtype=float)
    lo = float(np.nanquantile(arr, low_q))
    hi = float(np.nanquantile(arr, high_q))
    return np.asarray(np.clip(arr, lo, hi), dtype=float)


def _abfe_weighted_block_expavg(samples, weights, beta, block_size=0, lower_q=0.08, upper_q=0.92):
    arr = np.asarray(samples if samples is not None else [], dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite) or (not np.isfinite(beta)):
        return np.nan
    arr = np.asarray(arr[finite], dtype=float)
    ww = np.asarray([], dtype=float)
    if weights is not None:
        try:
            ww = np.asarray(weights, dtype=float)
        except Exception:
            ww = np.asarray([], dtype=float)
    if ww.size == finite.size:
        ww = np.asarray(ww[finite], dtype=float)
    else:
        ww = np.asarray([], dtype=float)
    if ww.size != arr.size:
        ww = np.ones(arr.size, dtype=float)
    ww = np.asarray(np.clip(ww, 0.0, None), dtype=float)
    if float(np.sum(ww)) <= 0.0:
        ww = np.ones(arr.size, dtype=float)
    try:
        block_i = int(block_size)
    except Exception:
        block_i = 0
    if block_i <= 0:
        block_i = _abfe_default_block_size(arr.size)
    block_i = max(1, int(block_i))
    if arr.size <= block_i:
        clipped = _abfe_winsorize(arr, lower_q=lower_q, upper_q=upper_q)
        return _abfe_weighted_expavg(clipped, ww, beta)
    block_means = []
    block_weights = []
    for start in range(0, arr.size, block_i):
        stop = min(arr.size, start + block_i)
        arr_blk = np.asarray(arr[start:stop], dtype=float)
        ww_blk = np.asarray(ww[start:stop], dtype=float)
        if arr_blk.size <= 0:
            continue
        block_means.append(float(_abfe_weighted_mean(arr_blk, ww_blk)))
        block_weights.append(float(np.sum(np.clip(ww_blk, 0.0, None))))
    if not block_means:
        return np.nan
    block_means = _abfe_winsorize(np.asarray(block_means, dtype=float), lower_q=lower_q, upper_q=upper_q)
    if block_means.size <= 0:
        return np.nan
    block_weights = np.asarray(block_weights[: block_means.size], dtype=float)
    if block_weights.size != block_means.size or float(np.sum(np.clip(block_weights, 0.0, None))) <= 0.0:
        block_weights = np.ones(block_means.size, dtype=float)
    return _abfe_weighted_expavg(block_means, block_weights, beta)


def _abfe_gbsa_radius_scale_for_element(element_symbol):
    elem = str(element_symbol or "").strip().upper()
    radius_a = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.50,
        "F": 1.50,
        "P": 1.85,
        "S": 1.80,
        "CL": 1.70,
        "BR": 1.85,
        "I": 1.98,
        "ZN": 1.39,
        "MG": 1.73,
        "CA": 1.94,
        "FE": 1.32,
        "CU": 1.40,
        "MN": 1.39,
        "NA": 2.27,
        "K": 2.75,
    }.get(elem, 1.70)
    scale = {
        "H": 0.85,
        "C": 0.72,
        "N": 0.79,
        "O": 0.85,
        "F": 0.88,
        "P": 0.86,
        "S": 0.96,
        "CL": 0.80,
        "BR": 0.80,
        "I": 0.80,
    }.get(elem, 0.80)
    return float(radius_a * 0.1), float(scale)


def build_abfe_gbsa_mmgbsa_rescore_payload(
    *,
    coords_nm_series=None,
    coords_scale=1.0,
    atom_rows=None,
    charges_e=None,
    protein_atom_idx=None,
    ligand_atom_idx=None,
    raw_interaction_energy_kj=None,
    frame_mask=None,
    original_frame_indices=None,
    max_rescore_frames=128,
    frame_stride=0,
    memory_budget_mb=192.0,
    solvent_dielectric=78.5,
    solute_dielectric=1.0,
    ligand_prmtop_file="",
):
    coords_input = coords_nm_series if coords_nm_series is not None else []
    atom_info = list(atom_rows or [])
    charge_arr = np.asarray(charges_e if charges_e is not None else [], dtype=float)
    prot_idx = np.asarray(protein_atom_idx if protein_atom_idx is not None else [], dtype=int)
    lig_idx = np.asarray(ligand_atom_idx if ligand_atom_idx is not None else [], dtype=int)
    out = {
        "ok": False,
        "reason": "invalid_input",
        "n_frames": 0,
        "n_input_frames": 0,
        "n_selected_before_limits": 0,
        "n_selected_after_limits": 0,
        "frame_stride_used": 1,
        "max_rescore_frames": int(max(1, int(max_rescore_frames or 128))),
        "memory_budget_mb": float(memory_budget_mb) if np.isfinite(_safe_float(memory_budget_mb)) else 192.0,
        "estimated_coords_mb": 0.0,
        "selected_frame_indices": np.asarray([], dtype=int),
        "selected_original_frame_indices": np.asarray([], dtype=int),
        "gbsa_complex_kj": np.asarray([], dtype=float),
        "gbsa_protein_kj": np.asarray([], dtype=float),
        "gbsa_ligand_kj": np.asarray([], dtype=float),
        "gbsa_delta_solv_kj": np.asarray([], dtype=float),
        "rescored_total_kj": np.asarray([], dtype=float),
        "ligand_internal_ok": False,
        "ligand_internal_reason": "not_requested",
        "ligand_internal_kj": np.asarray([], dtype=float),
        "rescored_total_strain_kj": np.asarray([], dtype=float),
    }
    coords_seq = None
    if isinstance(coords_input, np.ndarray):
        coords_shape = tuple(coords_input.shape)
    else:
        try:
            coords_seq = list(coords_input or [])
        except Exception:
            coords_seq = []
        if len(coords_seq) <= 0:
            coords_shape = (0, 0, 0)
        else:
            first_shape = tuple(np.asarray(coords_seq[0]).shape)
            coords_shape = (len(coords_seq),) + tuple(first_shape)
    if len(coords_shape) != 3 or coords_shape[0] <= 0 or coords_shape[1] <= 0 or coords_shape[2] != 3:
        out["reason"] = "coords_shape_mismatch"
        return out
    n_frames = int(coords_shape[0])
    n_atoms = int(coords_shape[1])
    out["n_input_frames"] = int(n_frames)
    if charge_arr.size != n_atoms or len(atom_info) != n_atoms or prot_idx.size <= 0 or lig_idx.size <= 0:
        out["reason"] = "atom_metadata_mismatch"
        return out
    frame_sel = np.arange(n_frames, dtype=int)
    if frame_mask is not None:
        mask = np.asarray(frame_mask, dtype=bool)
        if mask.size == n_frames:
            frame_sel = np.where(mask)[0].astype(int)
    if frame_sel.size <= 0:
        out["reason"] = "no_selected_frames"
        return out
    out["n_selected_before_limits"] = int(frame_sel.size)
    stride_used = _abfe_positive_int(frame_stride, default=1, minimum=0)
    if stride_used <= 0:
        stride_used = 1
    if stride_used > 1 and frame_sel.size > 1:
        kept = list(frame_sel[::stride_used])
        if int(kept[-1]) != int(frame_sel[-1]):
            kept.append(int(frame_sel[-1]))
        frame_sel = np.asarray(sorted(set(int(v) for v in kept)), dtype=int)

    max_frames_i = _abfe_positive_int(max_rescore_frames, default=128, minimum=1)
    try:
        mem_budget = float(memory_budget_mb)
    except Exception:
        mem_budget = 192.0
    mem_budget = float(max(8.0, mem_budget))
    per_frame_mb = float(max(1, n_atoms) * 3 * np.dtype(np.float32).itemsize) / 1048576.0
    max_frames_by_memory = int(max(1, np.floor(mem_budget / max(per_frame_mb, 1.0e-9))))
    max_frames_i = int(min(max_frames_i, max_frames_by_memory))
    if frame_sel.size > max_frames_i:
        pick = _abfe_downsample_indices(int(frame_sel.size), max_points=int(max_frames_i), include_last=True)
        frame_sel = np.asarray(frame_sel[np.asarray(pick, dtype=int)], dtype=int)
    out["frame_stride_used"] = int(stride_used)
    out["max_rescore_frames"] = int(max_frames_i)
    out["memory_budget_mb"] = float(mem_budget)
    out["n_selected_after_limits"] = int(frame_sel.size)
    out["estimated_coords_mb"] = float(per_frame_mb * int(frame_sel.size))

    try:
        scale = float(coords_scale)
    except Exception:
        scale = 1.0
    if not np.isfinite(scale):
        scale = 1.0
    if isinstance(coords_input, np.ndarray):
        coords_sel = np.asarray(coords_input[np.asarray(frame_sel, dtype=int)], dtype=np.float32)
    else:
        coords_sel = np.asarray(
            [np.asarray(coords_seq[int(i)], dtype=np.float32) for i in list(frame_sel)],
            dtype=np.float32,
        )
    if float(scale) != 1.0:
        coords_sel = np.asarray(coords_sel * float(scale), dtype=np.float32)
    orig_idx_arr = np.asarray(original_frame_indices if original_frame_indices is not None else [], dtype=int)
    if orig_idx_arr.size != n_frames:
        orig_idx_arr = np.arange(n_frames, dtype=int)
    selected_orig_idx = np.asarray(orig_idx_arr[np.asarray(frame_sel, dtype=int)], dtype=int)
    raw_sel = np.asarray([], dtype=float)
    if raw_interaction_energy_kj is not None:
        raw_arr = np.asarray(raw_interaction_energy_kj, dtype=float)
        if raw_arr.size == n_frames:
            raw_sel = np.asarray(raw_arr[frame_sel], dtype=float)
    try:
        from openmm import Context, GBSAOBCForce, Platform, System, VerletIntegrator
        from openmm import unit as omm_unit
        from openmm.app import AmberPrmtopFile, NoCutoff
    except Exception as exc:
        out["reason"] = f"openmm_import_failed:{exc}"
        return out

    def _build_gbsa_context(atom_indices):
        idx_arr = np.asarray(atom_indices, dtype=int)
        system = System()
        force = GBSAOBCForce()
        force.setSolventDielectric(float(solvent_dielectric))
        force.setSoluteDielectric(float(solute_dielectric))
        for atom_idx in list(idx_arr):
            system.addParticle(12.0)
            row = dict(atom_info[int(atom_idx)] or {})
            radius_nm, scale = _abfe_gbsa_radius_scale_for_element(row.get("element", "C"))
            force.addParticle(
                float(charge_arr[int(atom_idx)]),
                float(radius_nm),
                float(scale),
            )
        system.addForce(force)
        integrator = VerletIntegrator(0.001 * omm_unit.picoseconds)
        context = None
        for name in ("CPU", "Reference"):
            try:
                platform = Platform.getPlatformByName(str(name))
                context = Context(system, integrator, platform)
                break
            except Exception:
                context = None
        if context is None:
            context = Context(system, integrator)
        return {
            "indices": np.asarray(idx_arr, dtype=int),
            "integrator": integrator,
            "context": context,
        }

    complex_ctx = None
    protein_ctx = None
    ligand_ctx = None
    ligand_mm_ctx = None
    try:
        complex_ctx = _build_gbsa_context(np.arange(n_atoms, dtype=int))
        protein_ctx = _build_gbsa_context(prot_idx)
        ligand_ctx = _build_gbsa_context(lig_idx)
        gbsa_complex = np.full(coords_sel.shape[0], np.nan, dtype=float)
        gbsa_protein = np.full(coords_sel.shape[0], np.nan, dtype=float)
        gbsa_ligand = np.full(coords_sel.shape[0], np.nan, dtype=float)
        ligand_internal = np.full(coords_sel.shape[0], np.nan, dtype=float)
        ligand_prmtop_txt = str(ligand_prmtop_file or "").strip()
        if ligand_prmtop_txt:
            try:
                ligand_prmtop = AmberPrmtopFile(str(ligand_prmtop_txt))
                prm_atoms = list(ligand_prmtop.topology.atoms())
                prm_names = [str(getattr(atom, "name", "") or "").strip().upper() for atom in prm_atoms]
                loc_names = [str(dict(atom_info[int(idx)] or {}).get("name", "") or "").strip().upper() for idx in list(lig_idx)]
                if len(prm_names) != len(loc_names):
                    raise RuntimeError(f"ligand atom count mismatch: prmtop={len(prm_names)} coords={len(loc_names)}")
                if prm_names != loc_names:
                    raise RuntimeError("ligand atom names/order mismatch between prmtop and sampled coordinates")
                lig_sys = ligand_prmtop.createSystem(
                    nonbondedMethod=NoCutoff,
                    constraints=None,
                    rigidWater=False,
                    removeCMMotion=False,
                )
                lig_integrator = VerletIntegrator(0.001 * omm_unit.picoseconds)
                lig_context = None
                for name in ("CPU", "Reference"):
                    try:
                        platform = Platform.getPlatformByName(str(name))
                        lig_context = Context(lig_sys, lig_integrator, platform)
                        break
                    except Exception:
                        lig_context = None
                if lig_context is None:
                    lig_context = Context(lig_sys, lig_integrator)
                ligand_mm_ctx = {
                    "context": lig_context,
                    "integrator": lig_integrator,
                    "indices": np.asarray(lig_idx, dtype=int),
                }
                out["ligand_internal_reason"] = "ok"
            except Exception as exc:
                out["ligand_internal_reason"] = f"ligand_internal_failed:{exc}"
        for frame_i, xyz in enumerate(coords_sel):
            xyz_nm = np.asarray(xyz, dtype=float)
            complex_ctx["context"].setPositions(xyz_nm * omm_unit.nanometer)
            gbsa_complex[frame_i] = float(
                complex_ctx["context"]
                .getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(omm_unit.kilojoule_per_mole)
            )
            xyz_prot = np.asarray(xyz_nm[np.asarray(protein_ctx["indices"], dtype=int)], dtype=float)
            protein_ctx["context"].setPositions(xyz_prot * omm_unit.nanometer)
            gbsa_protein[frame_i] = float(
                protein_ctx["context"]
                .getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(omm_unit.kilojoule_per_mole)
            )
            xyz_lig = np.asarray(xyz_nm[np.asarray(ligand_ctx["indices"], dtype=int)], dtype=float)
            ligand_ctx["context"].setPositions(xyz_lig * omm_unit.nanometer)
            gbsa_ligand[frame_i] = float(
                ligand_ctx["context"]
                .getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(omm_unit.kilojoule_per_mole)
            )
            if isinstance(ligand_mm_ctx, dict):
                ligand_mm_ctx["context"].setPositions(xyz_lig * omm_unit.nanometer)
                ligand_internal[frame_i] = float(
                    ligand_mm_ctx["context"]
                    .getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(omm_unit.kilojoule_per_mole)
                )
        delta_solv = np.asarray(gbsa_complex - gbsa_protein - gbsa_ligand, dtype=float)
        rescored_total = np.asarray(delta_solv, dtype=float)
        if raw_sel.size == delta_solv.size:
            rescored_total = np.asarray(raw_sel + delta_solv, dtype=float)
        ligand_internal_ok = bool(np.any(np.isfinite(ligand_internal)))
        rescored_total_strain = np.asarray([], dtype=float)
        if ligand_internal_ok and ligand_internal.size == rescored_total.size:
            rescored_total_strain = np.asarray(rescored_total + ligand_internal, dtype=float)
        out.update(
            {
                "ok": True,
                "reason": "ok",
                "n_frames": int(coords_sel.shape[0]),
                "n_input_frames": int(n_frames),
                "n_selected_before_limits": int(out.get("n_selected_before_limits", coords_sel.shape[0])),
                "n_selected_after_limits": int(coords_sel.shape[0]),
                "frame_stride_used": int(out.get("frame_stride_used", 1)),
                "max_rescore_frames": int(out.get("max_rescore_frames", coords_sel.shape[0])),
                "memory_budget_mb": float(out.get("memory_budget_mb", 192.0)),
                "estimated_coords_mb": float(out.get("estimated_coords_mb", 0.0)),
                "selected_frame_indices": np.asarray(frame_sel, dtype=int),
                "selected_original_frame_indices": np.asarray(selected_orig_idx, dtype=int),
                "gbsa_complex_kj": np.asarray(gbsa_complex, dtype=float),
                "gbsa_protein_kj": np.asarray(gbsa_protein, dtype=float),
                "gbsa_ligand_kj": np.asarray(gbsa_ligand, dtype=float),
                "gbsa_delta_solv_kj": np.asarray(delta_solv, dtype=float),
                "rescored_total_kj": np.asarray(rescored_total, dtype=float),
                "ligand_internal_ok": bool(ligand_internal_ok),
                "ligand_internal_kj": np.asarray(ligand_internal, dtype=float) if ligand_internal_ok else np.asarray([], dtype=float),
                "rescored_total_strain_kj": np.asarray(rescored_total_strain, dtype=float),
            }
        )
        return out
    except Exception as exc:
        out["reason"] = f"gbsa_rescore_failed:{exc}"
        return out
    finally:
        for ctx_payload in (complex_ctx, protein_ctx, ligand_ctx, ligand_mm_ctx):
            if not isinstance(ctx_payload, dict):
                continue
            try:
                del ctx_payload["context"]
            except Exception:
                pass
            try:
                del ctx_payload["integrator"]
            except Exception:
                pass


def _abfe_min_true_run_mask(mask, min_run=1):
    arr = np.asarray(mask if mask is not None else [], dtype=bool)
    n = int(arr.size)
    if n <= 0:
        return np.asarray([], dtype=bool)
    try:
        min_run_i = int(min_run)
    except Exception:
        min_run_i = 1
    min_run_i = max(1, int(min_run_i))
    if min_run_i <= 1:
        return np.asarray(arr, dtype=bool)
    out = np.zeros(n, dtype=bool)
    start = 0
    while start < n:
        if not bool(arr[start]):
            start += 1
            continue
        stop = start + 1
        while stop < n and bool(arr[stop]):
            stop += 1
        if (stop - start) >= min_run_i:
            out[start:stop] = True
        start = stop
    return out


def _abfe_fill_short_false_gaps(mask, max_gap=0):
    arr = np.asarray(mask if mask is not None else [], dtype=bool)
    n = int(arr.size)
    if n <= 0:
        return np.asarray(arr, dtype=bool)
    try:
        max_gap_i = int(max_gap)
    except Exception:
        max_gap_i = 0
    max_gap_i = max(0, int(max_gap_i))
    if max_gap_i <= 0:
        return np.asarray(arr, dtype=bool)
    out = np.asarray(arr, dtype=bool).copy()
    start = 0
    while start < n:
        if bool(out[start]):
            start += 1
            continue
        stop = start + 1
        while stop < n and (not bool(out[stop])):
            stop += 1
        gap_len = int(stop - start)
        left_true = bool(start > 0 and out[start - 1])
        right_true = bool(stop < n and out[stop])
        if left_true and right_true and gap_len <= max_gap_i:
            out[start:stop] = True
        start = stop
    return np.asarray(out, dtype=bool)


def _abfe_gap_tolerant_true_run_mask(mask, min_run=1, max_gap=0):
    arr = np.asarray(mask if mask is not None else [], dtype=bool)
    if arr.size <= 0:
        return np.asarray(arr, dtype=bool)
    bridged = _abfe_fill_short_false_gaps(arr, max_gap=max_gap)
    run_mask = _abfe_min_true_run_mask(bridged, min_run=min_run)
    return np.logical_and(np.asarray(run_mask, dtype=bool), np.asarray(arr, dtype=bool))


def build_abfe_state_coverage_diagnostic_payload(
    *,
    site_com_dist=None,
    min_dist=None,
    contact_count_4a=None,
    bound_mask=None,
    unbound_mask=None,
    bound_hold_mask=None,
    unbound_hold_mask=None,
    far_hold_stable_mask=None,
    detached_settle_mask=None,
    unbound_site_com_threshold_a=10.0,
    unbound_min_dist_a=3.5,
    unbound_max_contact_count_4a=4,
    min_hold_state_frames=4,
):
    """Summarize whether ABFE-lite sampled both binding states.

    A large exit coordinate alone is not an unbound state.  The ligand must also
    lose close protein contacts and survive the hold-window filters; otherwise
    any DeltaG/Kd would be a single-state diagnostic rather than a binding
    estimate.
    """
    site = _abfe_float_series(site_com_dist)
    n = int(site.size)
    mins = _abfe_float_series(min_dist, n=n)
    contacts = _abfe_float_series(contact_count_4a, n=n)
    bound = _abfe_bool_series(bound_mask, n=n, fill=False)
    unbound = _abfe_bool_series(unbound_mask, n=n, fill=False)
    bound_hold = _abfe_bool_series(bound_hold_mask, n=n, fill=False)
    unbound_hold = _abfe_bool_series(unbound_hold_mask, n=n, fill=False)
    far_hold = _abfe_bool_series(far_hold_stable_mask, n=n, fill=False)
    detached = _abfe_bool_series(detached_settle_mask, n=n, fill=False)

    finite_site = np.isfinite(site)
    finite_min = np.isfinite(mins)
    finite_contacts = np.isfinite(contacts)
    max_site = float(np.nanmax(site[finite_site])) if np.any(finite_site) else np.nan
    max_min = float(np.nanmax(mins[finite_min])) if np.any(finite_min) else np.nan
    min_contact = float(np.nanmin(contacts[finite_contacts])) if np.any(finite_contacts) else np.nan
    site_reached = bool(np.isfinite(max_site) and max_site >= float(unbound_site_com_threshold_a))

    idx_max_site = int(np.nanargmax(site)) if np.any(finite_site) else -1
    min_at_max_site = float(mins[idx_max_site]) if 0 <= idx_max_site < mins.size and np.isfinite(mins[idx_max_site]) else np.nan
    contacts_at_max_site = (
        float(contacts[idx_max_site])
        if 0 <= idx_max_site < contacts.size and np.isfinite(contacts[idx_max_site])
        else np.nan
    )
    contact_clear_any = bool(
        np.any(
            np.logical_and.reduce(
                [
                    finite_site,
                    finite_min,
                    np.asarray(site >= float(unbound_site_com_threshold_a), dtype=bool),
                    np.asarray(mins >= float(unbound_min_dist_a), dtype=bool),
                    (
                        np.asarray(contacts <= float(unbound_max_contact_count_4a), dtype=bool)
                        if np.any(finite_contacts)
                        else np.ones(n, dtype=bool)
                    ),
                ]
            )
        )
    ) if n > 0 else False

    n_bound = int(np.sum(bound))
    n_unbound = int(np.sum(unbound))
    n_bound_hold = int(np.sum(bound_hold))
    n_unbound_hold = int(np.sum(unbound_hold))
    min_hold = int(max(1, int(min_hold_state_frames)))
    binding_allowed = bool(n_bound_hold >= min_hold and n_unbound_hold >= min_hold)

    if binding_allowed:
        status = "estimable_hold_windows"
        note = "Bound and unbound hold windows are populated; selected estimator may be used if ESS checks pass."
    elif n_unbound > 0:
        status = "unbound_geometry_without_stable_hold"
        note = "Unbound-like geometry was sampled, but hold-window coverage is insufficient for a selected binding estimate."
    elif not site_reached:
        status = "exit_coordinate_not_reached"
        note = "The ligand did not reach the unbound exit-coordinate threshold."
    elif not contact_clear_any:
        status = "surface_associated_not_bulk_unbound"
        note = "The ligand moved along the exit coordinate but retained close protein contacts, so it was not accepted as unbound."
    elif int(np.sum(far_hold)) <= 0 and int(np.sum(detached)) <= 0:
        status = "no_stable_far_hold"
        note = "The trajectory reached exit-like geometry but never entered a stable far-hold/detached window."
    else:
        status = "unbound_rejected_by_state_filters"
        note = "Unbound-like frames were rejected by the state or hold-window filters."

    return {
        "ok": True,
        "status": str(status),
        "note": str(note),
        "binding_estimate_allowed": bool(binding_allowed),
        "n_samples": int(n),
        "n_bound_raw": int(n_bound),
        "n_unbound_raw": int(n_unbound),
        "n_bound_hold": int(n_bound_hold),
        "n_unbound_hold": int(n_unbound_hold),
        "min_hold_state_frames": int(min_hold),
        "far_hold_frames": int(np.sum(far_hold)),
        "detached_settle_frames": int(np.sum(detached)),
        "max_exit_coordinate_a": float(max_site) if np.isfinite(max_site) else np.nan,
        "max_min_protein_ligand_distance_a": float(max_min) if np.isfinite(max_min) else np.nan,
        "min_contact_count_4a": float(min_contact) if np.isfinite(min_contact) else np.nan,
        "min_dist_at_max_exit_a": float(min_at_max_site) if np.isfinite(min_at_max_site) else np.nan,
        "contact_count_at_max_exit_4a": float(contacts_at_max_site) if np.isfinite(contacts_at_max_site) else np.nan,
        "unbound_site_com_threshold_a": float(unbound_site_com_threshold_a),
        "unbound_min_dist_a": float(unbound_min_dist_a),
        "unbound_max_contact_count_4a": int(unbound_max_contact_count_4a),
        "exit_coordinate_reached": bool(site_reached),
        "contact_clearance_reached": bool(contact_clear_any),
    }


def build_abfe_bias_reweighted_population_payload(
    *,
    bound_valid_mask=None,
    unbound_valid_mask=None,
    bias_valid=None,
    temperature_k=300.0,
    v_eff_nm3=np.nan,
    bootstrap_seed=20260212,
    n_boot=0,
):
    bound_arr = np.asarray(bound_valid_mask if bound_valid_mask is not None else [], dtype=bool)
    unbound_arr = np.asarray(unbound_valid_mask if unbound_valid_mask is not None else [], dtype=bool)
    bias_arr = np.asarray(bias_valid if bias_valid is not None else [], dtype=float)
    out = {
        "ok": False,
        "reason": "insufficient_state_frames",
        "n_state_frames": int(bound_arr.size),
        "n_bound_frames": int(np.sum(bound_arr)),
        "n_unbound_frames": int(np.sum(unbound_arr)),
        "n_bias_finite": int(np.sum(np.isfinite(bias_arr))),
        "p_bound_reweighted": np.nan,
        "p_unbound_reweighted": np.nan,
        "dg_reweighted_kj": np.nan,
        "dg_reweighted_state_kj": np.nan,
        "dg_reweighted_standard_kj": np.nan,
        "standard_state_correction_kj": np.nan,
        "c_eff_m": np.nan,
        "kd_app_reweighted_m": np.nan,
        "kd_standard_reweighted_m": np.nan,
        "p_kd_reweighted": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "ess_total": np.nan,
        "ess_bound": np.nan,
        "ess_unbound": np.nan,
        "block_size": 0,
        "weights": np.asarray([], dtype=float),
    }
    if bound_arr.size <= 0 or unbound_arr.size != bound_arr.size or bias_arr.size != bound_arr.size:
        out["reason"] = "shape_mismatch"
        return out
    state_mask = np.logical_or(bound_arr, unbound_arr)
    finite_mask = np.logical_and(state_mask, np.isfinite(bias_arr))
    if int(np.sum(finite_mask)) <= 0:
        out["reason"] = "no_finite_bias"
        return out
    bound_f = np.asarray(bound_arr[finite_mask], dtype=bool)
    unbound_f = np.asarray(unbound_arr[finite_mask], dtype=bool)
    bias_f = np.asarray(bias_arr[finite_mask], dtype=float)
    if int(np.sum(bound_f)) <= 0 or int(np.sum(unbound_f)) <= 0:
        out["reason"] = "single_state"
        return out
    k_b = 0.008314462618
    kbt = float(k_b * float(temperature_k))
    if not np.isfinite(kbt) or kbt <= 0.0:
        out["reason"] = "invalid_temperature"
        return out
    beta = float(1.0 / kbt)
    logw = beta * np.asarray(bias_f, dtype=float)
    finite_logw = logw[np.isfinite(logw)]
    if finite_logw.size <= 0:
        out["reason"] = "invalid_weights"
        return out
    def _weights_for_percentile(percentile):
        pct = float(np.clip(float(percentile), 50.0, 100.0))
        clip_hi = float(np.percentile(logw, pct))
        logw_clip = np.minimum(logw, clip_hi)
        logw_shift = float(np.max(logw_clip))
        weights_local = np.exp(np.clip(logw_clip - logw_shift, -120.0, 0.0))
        return pct, weights_local

    min_state_ess = float(max(3, min(int(np.sum(bound_f)), int(np.sum(unbound_f)), 10)))
    target_ess = float(min(max(10, int(round(0.2 * bound_f.size))), max(3, bound_f.size // 2)))
    clip_percentile_used = 100.0
    weights = np.asarray([], dtype=float)
    ess_total = np.nan
    ess_bound = np.nan
    ess_unbound = np.nan
    for pct_try in (100.0, 99.5, 99.0, 97.5, 95.0, 92.5, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0):
        pct_now, weights_now = _weights_for_percentile(float(pct_try))
        ess_total_now = _abfe_effective_sample_size(weights_now)
        ess_bound_now = _abfe_effective_sample_size(weights_now[bound_f])
        ess_unbound_now = _abfe_effective_sample_size(weights_now[unbound_f])
        weights = np.asarray(weights_now, dtype=float)
        ess_total = float(ess_total_now)
        ess_bound = float(ess_bound_now)
        ess_unbound = float(ess_unbound_now)
        clip_percentile_used = float(pct_now)
        if (
            np.isfinite(ess_total_now)
            and np.isfinite(ess_bound_now)
            and np.isfinite(ess_unbound_now)
            and float(ess_total_now) >= float(target_ess)
            and float(ess_bound_now) >= float(min_state_ess)
            and float(ess_unbound_now) >= float(min_state_ess)
        ):
            break
    w_bound = float(np.sum(weights[bound_f]))
    w_unbound = float(np.sum(weights[unbound_f]))
    w_total = float(w_bound + w_unbound)
    if w_total <= 0.0 or w_bound <= 0.0 or w_unbound <= 0.0:
        out["reason"] = "degenerate_weights"
        return out
    p_bound = float(np.clip(w_bound / w_total, 0.0, 1.0))
    p_unbound = float(np.clip(w_unbound / w_total, 0.0, 1.0))
    dg_reweighted = float(-kbt * np.log(p_bound / p_unbound))
    std_payload = _abfe_standard_state_from_state_ratio(
        dg_state_kj=float(dg_reweighted),
        v_eff_nm3=v_eff_nm3,
        temperature_k=float(temperature_k),
    )
    weights_aligned = np.zeros(bound_arr.size, dtype=float)
    weights_aligned[np.asarray(finite_mask, dtype=bool)] = np.asarray(weights, dtype=float)
    kd_app = np.nan
    p_kd = np.nan
    if bool(std_payload.get("ok", False)):
        kd_app = float(std_payload.get("kd_standard_m", np.nan))
        p_kd = float(std_payload.get("p_kd_standard", np.nan))
    out.update(
        {
            "ok": True,
            "reason": "ok",
            "n_state_frames": int(bound_f.size),
            "n_bound_frames": int(np.sum(bound_f)),
            "n_unbound_frames": int(np.sum(unbound_f)),
            "n_bias_finite": int(np.sum(np.isfinite(bias_f))),
            "p_bound_reweighted": float(p_bound),
            "p_unbound_reweighted": float(p_unbound),
            "dg_reweighted_kj": float(dg_reweighted),
            "dg_reweighted_state_kj": float(dg_reweighted),
            "dg_reweighted_standard_kj": float(std_payload.get("dg_standard_kj", np.nan)),
            "standard_state_correction_kj": float(std_payload.get("standard_state_correction_kj", np.nan)),
            "c_eff_m": float(std_payload.get("c_eff_m", np.nan)),
            "kd_app_reweighted_m": float(kd_app) if np.isfinite(kd_app) else np.nan,
            "kd_standard_reweighted_m": float(kd_app) if np.isfinite(kd_app) else np.nan,
            "p_kd_reweighted": float(p_kd) if np.isfinite(p_kd) else np.nan,
            "ess_total": float(ess_total),
            "ess_bound": float(ess_bound),
            "ess_unbound": float(ess_unbound),
            "clip_percentile_used": float(clip_percentile_used),
            "weights": np.asarray(weights_aligned, dtype=float),
            "weights_finite": np.asarray(weights, dtype=float),
        }
    )
    try:
        n_boot_i = int(n_boot)
    except Exception:
        n_boot_i = 0
    n_boot_i = max(0, int(n_boot_i))
    if n_boot_i > 0 and weights.size >= 6:
        rng = np.random.default_rng(int(bootstrap_seed))
        block_size = int(np.clip(round(np.sqrt(weights.size)), 3, 20))
        n_blocks = int(np.ceil(float(weights.size) / float(block_size)))
        boot = np.full(int(n_boot_i), np.nan, dtype=float)
        for boot_idx in range(int(n_boot_i)):
            starts = rng.integers(0, weights.size, size=int(max(1, n_blocks)))
            sample_idx = []
            for start in starts.tolist():
                sample_idx.extend(((start + np.arange(block_size)) % weights.size).tolist())
            idx = np.asarray(sample_idx[: weights.size], dtype=int)
            wb = float(np.sum(weights[idx][bound_f[idx]]))
            wu = float(np.sum(weights[idx][unbound_f[idx]]))
            if wb > 0.0 and wu > 0.0:
                boot[int(boot_idx)] = float(-kbt * np.log((wb / (wb + wu)) / (wu / (wb + wu))))
        finite_boot = boot[np.isfinite(boot)]
        out["block_size"] = int(block_size)
        if finite_boot.size >= max(8, min(32, int(n_boot_i) // 4)):
            ci_low, ci_high = np.percentile(finite_boot, [2.5, 97.5])
            out["ci_low"] = float(ci_low)
            out["ci_high"] = float(ci_high)
    return out


def _normalize_target_token(s):
    return re.sub(r"[^a-z0-9]+", "", str(s or "").strip().lower())


def _ridge_predict_single(values, model, feature_cols=None):
    if not isinstance(model, dict):
        return np.nan, []
    try:
        med = np.asarray(model.get("med", []), dtype=float)
        mu = np.asarray(model.get("mu", []), dtype=float)
        sd = np.asarray(model.get("sd", []), dtype=float)
        w = np.asarray(model.get("w", []), dtype=float)
        y_mean = float(model.get("y_mean", np.nan))
        vals = np.asarray(values, dtype=float).reshape(-1).copy()
    except (TypeError, ValueError, OverflowError):
        return np.nan, []
    arrays = (med, mu, sd, w)
    if (
        med.size == 0
        or any(arr.ndim != 1 or arr.size != med.size for arr in arrays)
        or any(not np.all(np.isfinite(arr)) for arr in arrays)
        or not np.isfinite(y_mean)
    ):
        return np.nan, []
    if vals.size < med.size:
        vals = np.concatenate([vals, med[vals.size :]], axis=0)
    elif vals.size > med.size:
        vals = vals[: med.size]
    missing_idx = np.where(~np.isfinite(vals))[0].tolist()
    if missing_idx:
        vals[missing_idx] = med[missing_idx]
    sd_safe = np.where(np.abs(sd) > 1e-12, sd, 1.0)
    z = (vals - mu) / sd_safe
    y = float(np.dot(z, w) + y_mean)
    feat_cols = list(feature_cols or [])
    missing = [feat_cols[i] for i in missing_idx if i < len(feat_cols)]
    return y, missing


def _load_abfe_runtime_deploy():
    global _ABFE_RUNTIME_DEPLOY_CACHE, _ABFE_RUNTIME_DEPLOY_SOURCE
    if _ABFE_RUNTIME_DEPLOY_CACHE is not None:
        return _ABFE_RUNTIME_DEPLOY_CACHE
    candidates = []
    env_path = str(os.environ.get(ABFE_RUNTIME_DEPLOY_ENV, "") or "").strip()
    if env_path:
        candidates.append(env_path)
    candidates.extend(ABFE_RUNTIME_DEPLOY_CANDIDATES)
    for pth in candidates:
        pp = os.path.abspath(os.path.expanduser(str(pth)))
        if not os.path.isfile(pp):
            continue
        try:
            with open(pp, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            if not isinstance(obj, dict):
                continue
            compact = obj.get("ridge_compact_weighted")
            if not isinstance(compact, dict):
                continue
            cols = compact.get("feature_cols")
            if not isinstance(cols, list) or not cols or not all(isinstance(c, str) and c.strip() for c in cols):
                continue
            med = np.asarray(compact.get("med", []), dtype=float)
            if med.ndim != 1 or med.size != len(cols):
                continue
            predicted, _ = _ridge_predict_single(med, compact, feature_cols=cols)
            if not np.isfinite(predicted):
                continue
            _ABFE_RUNTIME_DEPLOY_CACHE = obj
            _ABFE_RUNTIME_DEPLOY_SOURCE = pp
            return _ABFE_RUNTIME_DEPLOY_CACHE
        except Exception:
            continue
    _ABFE_RUNTIME_DEPLOY_CACHE = {}
    _ABFE_RUNTIME_DEPLOY_SOURCE = ""
    return _ABFE_RUNTIME_DEPLOY_CACHE


def infer_abfe_target_hint(texts, target_order):
    if not target_order:
        return ""
    tokens = [str(t) for t in target_order if str(t).strip()]
    if not tokens:
        return ""
    for raw in texts:
        s = str(raw or "")
        if not s:
            continue
        low = s.lower()
        norm = _normalize_target_token(low)
        for t in tokens:
            tl = str(t).lower()
            tnorm = _normalize_target_token(tl)
            if not tl:
                continue
            if re.search(rf"(^|[^a-z0-9]){re.escape(tl)}([^a-z0-9]|$)", low):
                return t
            if tnorm and tnorm in norm:
                return t
    return ""


def predict_abfe_runtime_calibrated_dg(feature_map, target_hint="", endpoint_group=""):
    """
    Runtime ABFE calibration with deploy model selection:
      - compact unknown-target ridge model
      - target-aware ridge model when target hint is recognized
    Falls back to legacy compact calibration if deploy model is unavailable.
    """
    fmap = dict(feature_map or {})
    legacy_cols = list(ABFE_COMPACT_CALIBRATION.get("feature_cols", []))
    legacy_vals = np.array([_safe_float(fmap.get(c, np.nan)) for c in legacy_cols], dtype=float)
    legacy_dg, legacy_missing = _ridge_predict_single(legacy_vals, ABFE_COMPACT_CALIBRATION, feature_cols=legacy_cols)
    deploy = _load_abfe_runtime_deploy()
    if not isinstance(deploy, dict) or not deploy:
        return {
            "dg_cal_kj": float(legacy_dg),
            "mode_used": "compact_legacy",
            "missing_features": list(legacy_missing),
            "target_used": "",
            "model_version": str(ABFE_COMPACT_CALIBRATION.get("version", "compact_legacy")),
            "deploy_source": "",
            "endpoint_offset_kj": 0.0,
        }
    compact_model = deploy.get("ridge_compact_weighted", {})
    compact_cols = list(compact_model.get("feature_cols", []))
    if not compact_cols:
        compact_cols = legacy_cols
        compact_model = ABFE_COMPACT_CALIBRATION
    compact_vals = np.array([_safe_float(fmap.get(c, np.nan)) for c in compact_cols], dtype=float)
    dg_compact, missing_compact = _ridge_predict_single(compact_vals, compact_model, feature_cols=compact_cols)
    dg_final = float(dg_compact)
    missing_final = list(missing_compact)
    mode_used = "compact_unknown"
    target_used = ""
    targetaware_model = deploy.get("ridge_targetaware_weighted", {})
    if not isinstance(targetaware_model, dict):
        targetaware_model = {}
    ta_cols_raw = targetaware_model.get("feature_cols", compact_cols)
    ta_cols = list(ta_cols_raw) if isinstance(ta_cols_raw, (list, tuple)) else []
    target_order_raw = targetaware_model.get("target_order", [])
    target_order = [str(t) for t in target_order_raw if str(t).strip()] if isinstance(target_order_raw, (list, tuple)) else []
    ta_weights = targetaware_model.get("w", [])
    ta_can = (
        len(ta_cols) > 0
        and all(isinstance(c, str) and c.strip() for c in ta_cols)
        and len(target_order) > 0
        and isinstance(ta_weights, (list, tuple))
        and len(ta_weights) == (len(ta_cols) + len(target_order))
    )
    hint_raw = str(target_hint or "").strip()
    hint_norm = _normalize_target_token(hint_raw)
    target_lookup = {_normalize_target_token(t): t for t in target_order}
    matched_target = str(target_lookup.get(hint_norm, ""))
    if ta_can and matched_target:
        ta_vals = np.array([_safe_float(fmap.get(c, np.nan)) for c in ta_cols], dtype=float)
        onehot = np.zeros(len(target_order), dtype=float)
        try:
            onehot[target_order.index(matched_target)] = 1.0
        except Exception:
            onehot[:] = 0.0
        ta_feat_cols = list(ta_cols) + [f"target::{t}" for t in target_order]
        ta_all = np.concatenate([ta_vals, onehot], axis=0)
        dg_ta, missing_ta = _ridge_predict_single(ta_all, targetaware_model, feature_cols=ta_feat_cols)
        if np.isfinite(dg_ta):
            dg_final = float(dg_ta)
            missing_final = [m for m in missing_ta if not str(m).startswith("target::")]
            mode_used = "targetaware_panel"
            target_used = matched_target
    endpoint_offset_kj = 0.0
    endpoint_group_txt = str(endpoint_group or "").strip().lower()
    if mode_used.startswith("compact") and endpoint_group_txt:
        offs = deploy.get("endpoint_residual_offsets_weighted", {})
        if isinstance(offs, dict) and bool(offs.get("enabled", False)):
            ridge_offs = offs.get("ridge_compact_offsets", {})
            if isinstance(ridge_offs, dict):
                for k, v in ridge_offs.items():
                    if str(k).strip().lower() == endpoint_group_txt:
                        vv = _safe_float(v)
                        if np.isfinite(vv):
                            endpoint_offset_kj = float(vv)
                            dg_final = float(dg_final + endpoint_offset_kj)
                        break
    model_version = str(
        deploy.get(
            "best_model_groupcv_abfe_only",
            deploy.get("best_model_groupcv", "runtime_deploy"),
        )
    )
    if mode_used == "targetaware_panel":
        model_version = f"{model_version}+ridge_targetaware"
    else:
        model_version = f"{model_version}+ridge_compact"
    return {
        "dg_cal_kj": float(dg_final),
        "mode_used": mode_used,
        "missing_features": list(missing_final),
        "target_used": str(target_used),
        "model_version": model_version,
        "deploy_source": str(_ABFE_RUNTIME_DEPLOY_SOURCE or ""),
        "endpoint_offset_kj": float(endpoint_offset_kj),
    }


def predict_abfe_compact_calibrated_dg(
    dg_expavg_kj,
    dg_cumulant_kj,
    energy_mean_kj,
    coul_mean_kj,
    contact_frac_mean,
):
    """Apply compact ridge calibration to map ABFE-lite outputs to corrected DeltaG."""
    model = ABFE_COMPACT_CALIBRATION
    vals = np.array(
        [
            dg_expavg_kj,
            dg_cumulant_kj,
            energy_mean_kj,
            coul_mean_kj,
            contact_frac_mean,
        ],
        dtype=float,
    )
    med = np.array(model["med"], dtype=float)
    mu = np.array(model["mu"], dtype=float)
    sd = np.array(model["sd"], dtype=float)
    w = np.array(model["w"], dtype=float)
    y_mean = float(model["y_mean"])
    missing_idx = np.where(~np.isfinite(vals))[0].tolist()
    if missing_idx:
        vals[missing_idx] = med[missing_idx]
    sd_safe = np.where(np.abs(sd) > 1e-12, sd, 1.0)
    z = (vals - mu) / sd_safe
    dg_cal = float(np.dot(z, w) + y_mean)
    missing_feats = [model["feature_cols"][i] for i in missing_idx]
    return dg_cal, missing_feats


def build_abfe_scene_restore_plan(
    *,
    known_objects=None,
    final_view_object="",
    context_object="",
    source_object="",
    restore_enabled=None,
    keep_scene=False,
):
    known = []
    for obj in list(known_objects or []):
        name = str(obj or "").strip()
        if name:
            known.append(str(name))
    known_set = set(known)
    final_obj = str(final_view_object or "").strip()
    context_obj = str(context_object or "").strip()
    source_obj = str(source_object or "").strip()
    restore_set = {str(o or "").strip() for o in list(restore_enabled or []) if str(o or "").strip()}

    actions = []
    force_enable = []
    zoom_selection = ""
    mode = "none"

    if final_obj:
        mode = "final_view"
        keep_targets = {str(final_obj)}
        if context_obj:
            keep_targets.add(str(context_obj))
        for on in list(known or []):
            if bool(keep_scene):
                enable_on = bool((on in restore_set) or (on in keep_targets))
            else:
                enable_on = bool(on in keep_targets)
            actions.append({"action": ("enable" if enable_on else "disable"), "object": str(on)})
        force_enable = [str(final_obj)]
        if context_obj:
            force_enable.append(str(context_obj))
        zoom_selection = (f"({context_obj}) or ({final_obj})" if context_obj else str(final_obj))
    elif restore_set:
        mode = "restore_enabled"
        restore_targets = set(restore_set)
        if context_obj:
            restore_targets.add(str(context_obj))
        for on in list(known or []):
            enable_on = bool(on in restore_targets)
            actions.append({"action": ("enable" if enable_on else "disable"), "object": str(on)})
    else:
        mode = "fallback_source"
        if source_obj and source_obj in known_set:
            force_enable.append(str(source_obj))
        if context_obj and context_obj in known_set and context_obj not in force_enable:
            force_enable.append(str(context_obj))
        if context_obj and source_obj:
            zoom_selection = f"({context_obj}) or ({source_obj})"
        elif source_obj:
            zoom_selection = str(source_obj)

    return {
        "ok": True,
        "mode": str(mode),
        "actions": list(actions),
        "force_enable": list(force_enable),
        "zoom_selection": str(zoom_selection),
        "final_view_object": str(final_obj),
        "context_object": str(context_obj),
        "source_object": str(source_obj),
        "restore_enabled": sorted(list(restore_set)),
        "keep_scene": bool(keep_scene),
    }


def build_abfe_scene_restore_default_payload():
    return {
        "ok": True,
        "mode": "none",
        "actions": [],
        "force_enable": [],
        "zoom_selection": "",
        "final_view_object": "",
        "context_object": "",
        "source_object": "",
        "restore_enabled": [],
        "keep_scene": False,
    }


def build_abfe_scene_apply_operations(
    *,
    actions=None,
    force_enable_actions=None,
    zoom_selection="",
    default_action="disable",
    zoom_buffer_a=8.0,
):
    def _normalize_actions(items, fallback_action):
        out = []
        for item in list(items or []):
            if isinstance(item, dict):
                obj = str(item.get("object", "") or "").strip()
                act = str(item.get("action", "") or "").strip().lower()
                if act not in ("enable", "disable"):
                    act = str(fallback_action)
            else:
                obj = str(item or "").strip()
                act = str(fallback_action)
            if not obj:
                continue
            if str(act) not in ("enable", "disable"):
                act = str(fallback_action)
            out.append({"action": str(act), "object": str(obj)})
        return out

    dflt = str(default_action or "").strip().lower()
    if dflt not in ("enable", "disable"):
        dflt = "disable"

    try:
        zoom_buf = float(zoom_buffer_a)
    except Exception:
        zoom_buf = 8.0
    if (not np.isfinite(zoom_buf)) or (zoom_buf <= 0.0):
        zoom_buf = 8.0

    scene_actions = _normalize_actions(actions, dflt)
    force_actions = _normalize_actions(force_enable_actions, "enable")
    zoom_sel = str(zoom_selection or "").strip()

    operations = []
    if scene_actions:
        operations.append(
            {
                "kind": "scene_actions",
                "actions": list(scene_actions),
                "default_action": str(dflt),
            }
        )
    if force_actions:
        operations.append(
            {
                "kind": "force_enable_actions",
                "actions": list(force_actions),
                "default_action": str(dflt),
            }
        )
    if zoom_sel:
        operations.append(
            {
                "kind": "zoom",
                "selection": str(zoom_sel),
                "buffer_a": float(zoom_buf),
            }
        )

    return {
        "ok": True,
        "default_action": str(dflt),
        "zoom_buffer_a": float(zoom_buf),
        "scene_actions": list(scene_actions),
        "force_enable_actions": list(force_actions),
        "zoom_selection": str(zoom_sel),
        "has_scene_actions": bool(len(scene_actions) > 0),
        "has_force_enable_actions": bool(len(force_actions) > 0),
        "should_zoom": bool(zoom_sel),
        "operations": list(operations),
    }


def build_abfe_object_inventory_payload(
    *,
    object_names=None,
):
    names = []
    for obj in list(object_names or []):
        name = str(obj or "").strip()
        if name:
            names.append(str(name))
    return {
        "ok": True,
        "object_names": list(names),
        "count": int(len(names)),
    }


def build_abfe_final_scene_restore_execution_payload(
    *,
    known_objects=None,
    final_view_object="",
    context_object="",
    source_object="",
    restore_enabled=None,
    keep_scene_requested=False,
):
    final_obj = str(final_view_object or "").strip()
    context_obj = str(context_object or "").strip()
    source_obj = str(source_object or "").strip()
    restore_list = [str(o or "").strip() for o in list(restore_enabled or []) if str(o or "").strip()]
    keep_scene_applied = bool(keep_scene_requested) if final_obj else False

    scene_plan = build_abfe_scene_restore_plan(
        known_objects=list(known_objects or []),
        final_view_object=str(final_obj),
        context_object=str(context_obj),
        source_object=str(source_obj),
        restore_enabled=list(restore_list),
        keep_scene=bool(keep_scene_applied),
    )

    actions = list(scene_plan.get("actions", []) or [])
    force_enable = [str(obj) for obj in list(scene_plan.get("force_enable", []) or []) if str(obj or "").strip()]
    force_enable_actions = [{"action": "enable", "object": str(obj)} for obj in force_enable]
    zoom_sel = str(scene_plan.get("zoom_selection", "") or "").strip()
    mode = str(scene_plan.get("mode", ""))

    if final_obj:
        debug_line = (
            f"ABFE final view: object={final_obj}, keep_scene={keep_scene_applied}, "
            f"context={context_obj}, restored_visible={len(restore_list)}"
        )
        should_log_debug = True
    else:
        debug_line = ""
        should_log_debug = False

    return {
        "ok": True,
        "mode": str(mode),
        "scene_plan": dict(scene_plan or {}),
        "actions": list(actions),
        "has_actions": bool(len(actions) > 0),
        "force_enable_actions": list(force_enable_actions),
        "has_force_enable_actions": bool(len(force_enable_actions) > 0),
        "zoom_selection": str(zoom_sel),
        "should_zoom": bool(zoom_sel),
        "keep_scene_applied": bool(keep_scene_applied),
        "should_log_debug": bool(should_log_debug),
        "debug_line": str(debug_line),
        "final_view_object": str(final_obj),
        "context_object": str(context_obj),
        "source_object": str(source_obj),
        "restore_enabled_count": int(len(restore_list)),
    }


def build_abfe_final_scene_restore_execution_default_payload():
    scene_plan = build_abfe_scene_restore_default_payload()
    return {
        "ok": True,
        "mode": str(scene_plan.get("mode", "none")),
        "scene_plan": dict(scene_plan or {}),
        "actions": [],
        "has_actions": False,
        "force_enable_actions": [],
        "has_force_enable_actions": False,
        "zoom_selection": "",
        "should_zoom": False,
        "keep_scene_applied": False,
        "should_log_debug": False,
        "debug_line": "",
        "final_view_object": "",
        "context_object": "",
        "source_object": "",
        "restore_enabled_count": 0,
    }


def build_abfe_local_md_cleanup_plan(
    *,
    live_view_object="",
    known_objects=None,
    context_object="",
    source_object="",
    state_index=1,
):
    live_obj = str(live_view_object or "").strip()
    context_obj = str(context_object or "").strip()
    source_obj = str(source_object or "").strip()
    known_set = {str(o or "").strip() for o in list(known_objects or []) if str(o or "").strip()}

    try:
        st_idx = int(state_index or 1)
    except Exception:
        st_idx = 1
    st_idx = max(1, int(st_idx))

    should_delete_live_view = bool(live_obj and live_obj in known_set)
    should_build_context_fallback = bool((not context_obj) and source_obj)

    return {
        "ok": True,
        "live_view_object": str(live_obj),
        "should_delete_live_view": bool(should_delete_live_view),
        "has_context_object": bool(context_obj),
        "context_object": str(context_obj),
        "source_object": str(source_obj),
        "should_build_context_fallback": bool(should_build_context_fallback),
        "context_source_object": str(source_obj),
        "context_state_index": int(st_idx),
    }


def build_abfe_live_view_tracking_default_payload():
    return {
        "ok": True,
        "local_model_indices": [],
        "idx_selection": "",
    }


def build_abfe_live_view_atom_count_payload(
    *,
    atom_count=0,
):
    try:
        n_atoms = int(atom_count or 0)
    except Exception:
        n_atoms = 0
    n_atoms = max(0, int(n_atoms))
    return {
        "ok": True,
        "live_view_atom_count": int(n_atoms),
    }


def build_abfe_local_md_teardown_plan(
    *,
    has_context=False,
    has_integrator=False,
    clear_references=True,
):
    release_order = []
    if bool(has_context):
        release_order.append("context")
    if bool(has_integrator):
        release_order.append("integrator")

    return {
        "ok": True,
        "has_context": bool(has_context),
        "has_integrator": bool(has_integrator),
        "should_release_context": bool(has_context),
        "should_release_integrator": bool(has_integrator),
        "release_order": list(release_order),
        "clear_references": bool(clear_references),
    }


def build_abfe_openmm_staged_prep_plan(
    *,
    minimize_max_iters=1500,
    has_lig_hold=False,
    sim_ns=1.0,
    timestep_ps=0.001,
):
    notes = []
    try:
        max_it = int(max(500, float(minimize_max_iters)))
    except Exception:
        max_it = 1500

    sim_ns_used, sim_note = _sanitize_md_duration_ns(sim_ns, default=1.0)
    if sim_note:
        notes.append(str(sim_note))
    dt_ps, dt_note = _sanitize_md_timestep_ps(timestep_ps, default=0.001)
    if dt_note:
        notes.append(str(dt_note))
    use_legacy_step_fallback = bool(dt_note == "defaulted_invalid_timestep")

    minimize_ops = []
    if int(max_it) > 0:
        if bool(has_lig_hold):
            minimize_ops.append({"set_k_hold": 2600.0, "minimize_max_iters": int(max_it)})
            minimize_ops.append({"set_k_hold": 1100.0, "minimize_max_iters": int(max(300, round(0.75 * int(max_it))))})
            minimize_ops.append({"set_k_hold": 320.0, "minimize_max_iters": int(max(250, round(0.50 * int(max_it))))})
            minimize_ops.append({"set_k_hold": 0.0, "minimize_max_iters": int(max(200, round(0.40 * int(max_it))))})
        else:
            minimize_ops.append({"set_k_hold": None, "minimize_max_iters": int(max_it)})

    eq_hold_ps = 40.0 if bool(has_lig_hold) else 0.0
    eq_soft_ps = 60.0 if bool(has_lig_hold) else 0.0
    eq_free_ps = max(120.0, min(320.0, 0.08 * float(sim_ns_used) * 1000.0))
    try:
        if use_legacy_step_fallback:
            raise ValueError("legacy_timestep_fallback")
        eq_hold_steps = int(max(1, round(float(eq_hold_ps) / float(dt_ps))))
        eq_soft_steps = int(max(1, round(float(eq_soft_ps) / float(dt_ps))))
        eq_free_steps = int(max(1, round(float(eq_free_ps) / float(dt_ps))))
    except Exception:
        eq_hold_steps = 25000
        eq_soft_steps = 25000
        eq_free_steps = 120000

    equilibration_ops = []
    if bool(has_lig_hold):
        equilibration_ops.append({"set_k_hold": 320.0, "steps": int(eq_hold_steps)})
        equilibration_ops.append({"set_k_hold": 120.0, "steps": int(eq_soft_steps)})
        equilibration_ops.append({"set_k_hold": 0.0, "steps": int(eq_free_steps)})
    else:
        equilibration_ops.append({"set_k_hold": None, "steps": int(eq_free_steps)})

    equilibration_total_ps = float(eq_hold_ps + eq_soft_ps + eq_free_ps)
    total_runtime_ps = float(sim_ns_used * 1000.0)
    recommended_discard_fraction = float(
        min(0.5, equilibration_total_ps / total_runtime_ps)
    ) if total_runtime_ps > 0.0 else 0.25
    rescue_relax_ps = 25.0 if bool(has_lig_hold) else 15.0
    try:
        if use_legacy_step_fallback:
            raise ValueError("legacy_timestep_fallback")
        rescue_relax_steps = int(max(1, round(float(rescue_relax_ps) / float(dt_ps))))
    except Exception:
        rescue_relax_steps = 25000 if bool(has_lig_hold) else 15000

    return {
        "ok": True,
        "max_it": int(max_it),
        "has_lig_hold": bool(has_lig_hold),
        "sim_ns_used": float(sim_ns_used),
        "timestep_ps_used": float(dt_ps),
        "minimize_ops": list(minimize_ops),
        "eq_hold_ps": float(eq_hold_ps),
        "eq_soft_ps": float(eq_soft_ps),
        "eq_free_ps": float(eq_free_ps),
        "eq_hold_steps": int(eq_hold_steps),
        "eq_soft_steps": int(eq_soft_steps),
        "eq_free_steps": int(eq_free_steps),
        "equilibration_ops": list(equilibration_ops),
        "equilibration_ps_total": float(equilibration_total_ps),
        "recommended_analysis_discard_ps": float(equilibration_total_ps),
        "recommended_analysis_discard_fraction": float(recommended_discard_fraction),
        "positive_energy_rescue_threshold_kj": 75.0,
        "clash_min_dist_a_threshold": 1.55,
        "rescue_max_rounds": 3,
        "rescue_minimize_max_iters": int(max(200, round(0.40 * int(max_it)))),
        "rescue_relax_ps": float(rescue_relax_ps),
        "rescue_relax_steps": int(rescue_relax_steps),
        "notes": list(notes),
    }


def build_abfe_openmm_context_setup_plan(
    *,
    temperature_k=300.0,
    timestep_ps=0.001,
    friction_inv_ps=5.0,
    picked_platform_name="",
    has_platform=False,
    platform_properties=None,
):
    notes = []
    t_k, temp_note = _sanitize_md_temperature_k(temperature_k, default=300.0)
    if temp_note:
        notes.append(str(temp_note))
    dt_ps, dt_note = _sanitize_md_timestep_ps(timestep_ps, default=0.001)
    if dt_note:
        notes.append(str(dt_note))
    gamma_ps, gamma_note = _sanitize_md_friction_inv_ps(friction_inv_ps, default=5.0)
    if gamma_note:
        notes.append(str(gamma_note))

    integrator_candidates = [
        {"kind": "LangevinMiddleIntegrator"},
        {"kind": "LangevinIntegrator"},
    ]
    picked = str(picked_platform_name or "").strip()
    props, prop_notes = _normalize_platform_properties(picked, platform_properties)
    notes.extend(list(prop_notes))
    picked = str(picked_platform_name or "").strip()
    has_plat = bool(has_platform) and bool(picked)

    if has_plat:
        context_attempts = [
            {
                "mode": "platform",
                "use_platform": True,
                "use_properties": bool(len(props) > 0),
                "picked_platform_name": str(picked),
                "properties": dict(props),
            },
            {
                "mode": "default",
                "use_platform": False,
                "use_properties": False,
                "picked_platform_name": "",
                "properties": {},
            },
        ]
    else:
        context_attempts = [
            {
                "mode": "default",
                "use_platform": False,
                "use_properties": False,
                "picked_platform_name": "",
                "properties": {},
            }
        ]

    report_interval_steps = int(max(1, round(1.0 / float(dt_ps))))
    checkpoint_interval_steps = int(max(report_interval_steps, round(50.0 / float(dt_ps))))

    return {
        "ok": True,
        "temperature_k": float(t_k),
        "timestep_ps": float(dt_ps),
        "friction_inv_ps": float(gamma_ps),
        "integration_regime": "hmr_candidate" if float(dt_ps) > 0.002 else "standard_hbonds",
        "requires_hydrogen_mass_repartitioning": bool(float(dt_ps) > 0.002),
        "recommended_report_interval_steps": int(report_interval_steps),
        "recommended_checkpoint_interval_steps": int(checkpoint_interval_steps),
        "integrator_candidates": list(integrator_candidates),
        "context_attempts": list(context_attempts),
        "picked_platform_name": str(picked),
        "has_platform": bool(has_plat),
        "platform_properties": dict(props),
        "notes": list(notes),
    }


def prepare_abfe_live_view_preflight(
    *,
    local_globals=None,
    atoms=None,
    n_local=0,
    loaded_object="",
    live_view_object="abfe_local_md_live",
):
    live_obj = str(live_view_object or "").strip() or "abfe_local_md_live"
    loaded_obj = str(loaded_object or "").strip()
    try:
        n_local_i = int(n_local or 0)
    except Exception:
        n_local_i = 0
    n_local_i = max(0, int(n_local_i))

    local_model_indices = []
    reason = ""
    try:
        local_model_indices = [
            int(getattr(list(atoms or [])[int(g)], "index", 0) or 0)
            for g in list(local_globals or [])
        ]
    except Exception:
        local_model_indices = []
        reason = "index_resolution_failed"
    local_model_indices = [int(i) for i in list(local_model_indices or []) if int(i) > 0]

    idx_sel = ""
    if local_model_indices:
        idx_sel = "index " + "+".join(str(i) for i in list(local_model_indices or []))
    create_sel = ""
    if loaded_obj and idx_sel:
        create_sel = f"({loaded_obj}) and ({idx_sel})"

    can_create = bool((n_local_i > 0) and (len(local_model_indices) == int(n_local_i)) and create_sel)
    if (not reason) and (not can_create):
        if n_local_i <= 0:
            reason = "invalid_n_local"
        elif not loaded_obj:
            reason = "missing_loaded_object"
        elif not local_model_indices:
            reason = "empty_local_indices"
        elif len(local_model_indices) != int(n_local_i):
            reason = "local_index_count_mismatch"
        else:
            reason = "selection_unavailable"

    return {
        "ok": True,
        "can_create": bool(can_create),
        "reason": str(reason),
        "live_view_object": str(live_obj),
        "loaded_object": str(loaded_obj),
        "n_local": int(n_local_i),
        "local_model_indices": list(local_model_indices),
        "idx_selection": str(idx_sel),
        "create_selection": str(create_sel),
    }


def build_abfe_steering_lambda_schedule(
    *,
    n_samples=0,
    steering_mode="pull_hold",
    steering_enabled=False,
):
    try:
        n_total = int(n_samples or 0)
    except Exception:
        n_total = 0
    n_total = max(0, int(n_total))
    mode_txt = str(steering_mode or "").strip().lower()
    if mode_txt != "pull_hold_return":
        mode_txt = "pull_hold"

    lam_schedule = np.zeros(n_total, dtype=float)
    n_warm = 0
    n_active = 0
    n_cycles = 0

    if bool(steering_enabled) and n_total > 0:
        if n_total == 1:
            lam_schedule[0] = 1.0
            n_active = 1
            n_cycles = 1
        else:
            n_warm = int(max(0, min(n_total // 3, round(0.06 * n_total))))
            n_active = int(max(1, n_total - n_warm))
            if mode_txt == "pull_hold_return":
                n_cycles = 1
                n_pull = max(1, int(round(0.30 * n_active)))
                n_hold = max(1, int(round(0.16 * n_active)))
                n_ret = max(1, int(round(0.30 * n_active)))
                n_settle = max(1, int(n_active - n_pull - n_hold - n_ret))
                while (n_pull + n_hold + n_ret + n_settle) > n_active:
                    if n_settle > 1:
                        n_settle -= 1
                    elif n_hold > 1:
                        n_hold -= 1
                    elif n_ret > 1:
                        n_ret -= 1
                    elif n_pull > 1:
                        n_pull -= 1
                    else:
                        break
                while (n_pull + n_hold + n_ret + n_settle) < n_active:
                    n_settle += 1
                lam_pull = 0.5 * (
                    1.0 - np.cos(np.linspace(0.0, np.pi, n_pull, endpoint=True, dtype=float))
                )
                lam_hold = np.ones(n_hold, dtype=float)
                lam_ret = 0.5 * (
                    1.0 + np.cos(np.linspace(0.0, np.pi, n_ret, endpoint=True, dtype=float))
                )
                lam_settle = np.zeros(n_settle, dtype=float)
                lam_active = np.concatenate([lam_pull, lam_hold, lam_ret, lam_settle], axis=0)[:n_active]
            else:
                n_cycles = 1
                n_pull = max(1, int(round(0.40 * n_active)))
                n_hold = max(1, int(n_active - n_pull))
                lam_pull = 0.5 * (
                    1.0 - np.cos(np.linspace(0.0, np.pi, n_pull, endpoint=True, dtype=float))
                )
                lam_hold = np.ones(n_hold, dtype=float)
                lam_active = np.concatenate([lam_pull, lam_hold], axis=0)[:n_active]
            if n_warm > 0:
                lam_schedule[:n_warm] = 0.0
                lam_schedule[n_warm:] = lam_active
            else:
                lam_schedule[:] = lam_active

    return {
        "ok": True,
        "steering_enabled": bool(steering_enabled),
        "steering_mode": str(mode_txt),
        "n_samples": int(n_total),
        "n_warm": int(n_warm),
        "n_active": int(n_active),
        "n_cycles": int(n_cycles),
        "lambda_schedule": np.asarray(lam_schedule, dtype=float),
    }


def build_abfe_surface_detach_direction_payload(
    *,
    coords_nm=None,
    prot_heavy_idx=None,
    lig_heavy_idx=None,
    site_anchor_idx=None,
    fallback_direction=None,
    contact_cutoff_nm=0.45,
):
    fallback = np.asarray(fallback_direction if fallback_direction is not None else [1.0, 0.0, 0.0], dtype=float)
    fb_norm = float(np.linalg.norm(fallback))
    if fb_norm < 1.0e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        fb_norm = 1.0
    fallback = fallback / fb_norm
    out = {
        "ok": True,
        "direction": np.asarray(fallback, dtype=float),
        "reason": "fallback_direction",
        "contact_count": 0,
        "ligand_com_nm": np.asarray([np.nan, np.nan, np.nan], dtype=float),
        "surface_centroid_nm": np.asarray([np.nan, np.nan, np.nan], dtype=float),
        "protein_com_nm": np.asarray([np.nan, np.nan, np.nan], dtype=float),
        "contact_cutoff_nm": float(contact_cutoff_nm),
    }

    xyz = np.asarray(coords_nm if coords_nm is not None else [], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] <= 0:
        out["reason"] = "invalid_coordinates"
        return out

    prot_idx = np.asarray(prot_heavy_idx if prot_heavy_idx is not None else [], dtype=int)
    lig_idx = np.asarray(lig_heavy_idx if lig_heavy_idx is not None else [], dtype=int)
    site_idx = np.asarray(site_anchor_idx if site_anchor_idx is not None else [], dtype=int)
    prot_idx = prot_idx[np.logical_and(prot_idx >= 0, prot_idx < xyz.shape[0])]
    lig_idx = lig_idx[np.logical_and(lig_idx >= 0, lig_idx < xyz.shape[0])]
    site_idx = site_idx[np.logical_and(site_idx >= 0, site_idx < xyz.shape[0])]
    if prot_idx.size == 0 or lig_idx.size == 0:
        out["reason"] = "missing_heavy_atom_groups"
        return out

    lig_xyz = np.asarray(xyz[lig_idx], dtype=float)
    prot_xyz = np.asarray(xyz[prot_idx], dtype=float)
    lig_com = np.mean(lig_xyz, axis=0)
    prot_com = np.mean(prot_xyz, axis=0)
    out["ligand_com_nm"] = np.asarray(lig_com, dtype=float)
    out["protein_com_nm"] = np.asarray(prot_com, dtype=float)

    def _norm_or_zero(vec):
        vv = np.asarray(vec, dtype=float)
        nn = float(np.linalg.norm(vv))
        if nn < 1.0e-8:
            return np.zeros(3, dtype=float)
        return vv / nn

    away_global = _norm_or_zero(lig_com - prot_com)
    dv = np.asarray(prot_xyz[:, None, :] - lig_xyz[None, :, :], dtype=float)
    dmat = np.linalg.norm(dv, axis=2)
    cutoff = float(max(0.25, contact_cutoff_nm))
    close_rows = np.where(np.any(dmat <= cutoff, axis=1))[0]
    if close_rows.size > 0:
        surface_xyz = np.asarray(prot_xyz[close_rows], dtype=float)
        surface_centroid = np.mean(surface_xyz, axis=0)
        local_surface = _norm_or_zero(lig_com - surface_centroid)
        out["reason"] = "surface_contacts"
        out["contact_count"] = int(close_rows.size)
        out["surface_centroid_nm"] = np.asarray(surface_centroid, dtype=float)
    else:
        flat_idx = int(np.argmin(dmat))
        ip = int(flat_idx // max(1, lig_xyz.shape[0]))
        surface_centroid = np.asarray(prot_xyz[ip], dtype=float)
        local_surface = _norm_or_zero(lig_com - surface_centroid)
        out["reason"] = "nearest_surface_atom"
        out["surface_centroid_nm"] = np.asarray(surface_centroid, dtype=float)

    site_vec = np.zeros(3, dtype=float)
    if site_idx.size > 0:
        site_vec = _norm_or_zero(lig_com - np.mean(xyz[site_idx], axis=0))
    blend = (
        0.65 * local_surface
        + 0.25 * away_global
        + 0.10 * (site_vec if np.linalg.norm(site_vec) > 1.0e-8 else fallback)
    )
    blend_norm = float(np.linalg.norm(blend))
    if blend_norm < 1.0e-8:
        blend = np.asarray(fallback, dtype=float)
        blend_norm = float(np.linalg.norm(blend))
    out["direction"] = np.asarray(blend / max(blend_norm, 1.0e-8), dtype=float)
    return out


def build_abfe_tunnel_restraint_payload(
    *,
    enabled=True,
    origin_nm=None,
    direction=None,
    radius_a=3.5,
    k_radial_kj_mol_nm2=650.0,
    k_axial_kj_mol_nm2=1000.0,
    pull_distance_a=12.0,
    axial_margin_a=4.0,
    shape="cylinder",
    cone_half_angle_deg=35.0,
    cone_length_a=6.0,
):
    """
    Validate a cylindrical guide tunnel for ABFE-lite steering.

    ``shape="funnel"`` turns the constant radius into a Limongelli-style funnel:
    a cone that is wide at the binding site (the bound ligand keeps its natural
    fluctuations and can re-dock in any orientation) and narrows to the tunnel
    radius ``cone_length_a`` along the exit axis, followed by the cylinder up to
    the axial cap.  Unbound sampling is confined to a small volume in front of
    the entrance, so the return pull finds the pocket instead of the surface.

    The tunnel is intentionally soft: the ligand center of mass is unpenalized
    inside the cylinder and receives a harmonic radial wall only when it leaves
    the entrance corridor.  A short negative axial wall prevents escape through
    the back of the pocket.
    """
    out = {
        "ok": False,
        "enabled": bool(enabled),
        "reason": "",
        "origin_nm": np.asarray([np.nan, np.nan, np.nan], dtype=float),
        "direction": np.asarray([1.0, 0.0, 0.0], dtype=float),
        "radius_nm": np.nan,
        "k_radial_kj_mol_nm2": np.nan,
        "k_axial_kj_mol_nm2": np.nan,
        "zmin_nm": -0.25,
        "zmax_nm": np.nan,
        "shape": "cylinder",
        "tan_alpha": 0.0,
        "zcc_nm": 0.0,
        "energy_expression": ABFE_TUNNEL_ENERGY_EXPRESSION,
        "notes": [],
    }
    if not bool(enabled):
        out["reason"] = "disabled"
        return out
    origin = np.asarray(origin_nm if origin_nm is not None else [], dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        out["reason"] = "invalid_origin"
        return out
    vec = np.asarray(direction if direction is not None else [], dtype=float)
    if vec.shape != (3,) or not np.all(np.isfinite(vec)):
        out["reason"] = "invalid_direction"
        return out
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-8:
        out["reason"] = "zero_direction"
        return out
    try:
        radius_nm = 0.1 * float(radius_a)
    except Exception:
        radius_nm = 0.35
    if (not np.isfinite(radius_nm)) or radius_nm <= 0.0:
        radius_nm = 0.35
        out["notes"].append("radius_defaulted")
    radius_nm = float(np.clip(radius_nm, 0.20, 1.20))
    try:
        k_radial = float(k_radial_kj_mol_nm2)
    except Exception:
        k_radial = 650.0
    if (not np.isfinite(k_radial)) or k_radial <= 0.0:
        k_radial = 650.0
        out["notes"].append("radial_k_defaulted")
    k_radial = float(np.clip(k_radial, 25.0, 5000.0))
    try:
        k_axial = float(k_axial_kj_mol_nm2)
    except Exception:
        k_axial = 1000.0
    if (not np.isfinite(k_axial)) or k_axial <= 0.0:
        k_axial = 1000.0
        out["notes"].append("axial_k_defaulted")
    k_axial = float(np.clip(k_axial, 25.0, 8000.0))
    try:
        pull_nm = 0.1 * float(pull_distance_a)
    except Exception:
        pull_nm = 1.2
    if (not np.isfinite(pull_nm)) or pull_nm <= 0.0:
        pull_nm = 1.2
        out["notes"].append("pull_distance_defaulted")
    try:
        margin_nm = 0.1 * float(axial_margin_a)
    except Exception:
        margin_nm = 0.4
    if (not np.isfinite(margin_nm)) or margin_nm < 0.0:
        margin_nm = 0.4
    zmax_nm = float(max(pull_nm + margin_nm, 2.0 * pull_nm, radius_nm + 0.6))
    shape_txt = str(shape or "cylinder").strip().lower()
    if shape_txt not in {"cylinder", "funnel"}:
        out["notes"].append(f"unknown_shape_defaulted:{shape_txt}")
        shape_txt = "cylinder"
    tan_alpha = 0.0
    zcc_nm = 0.0
    if shape_txt == "funnel":
        try:
            alpha = float(cone_half_angle_deg)
        except (TypeError, ValueError):
            alpha = 35.0
        if not np.isfinite(alpha):
            alpha = 35.0
        alpha = float(np.clip(alpha, 10.0, 60.0))
        tan_alpha = float(np.tan(np.deg2rad(alpha)))
        try:
            zcc_nm = 0.1 * float(cone_length_a)
        except (TypeError, ValueError):
            zcc_nm = 0.6
        if (not np.isfinite(zcc_nm)) or zcc_nm <= 0.0:
            zcc_nm = 0.6
            out["notes"].append("cone_length_defaulted")
        # the cone must end before the axial cap, otherwise there is no cylinder
        zcc_nm = float(min(zcc_nm, max(0.1, zmax_nm - 0.3)))
    out.update(
        {
            "ok": True,
            "reason": "ok",
            "origin_nm": np.asarray(origin, dtype=float),
            "direction": np.asarray(vec / norm, dtype=float),
            "radius_nm": float(radius_nm),
            "k_radial_kj_mol_nm2": float(k_radial),
            "k_axial_kj_mol_nm2": float(k_axial),
            "zmin_nm": -float(max(0.15, min(0.50, radius_nm))),
            "zmax_nm": float(zmax_nm),
            "shape": str(shape_txt),
            "tan_alpha": float(tan_alpha),
            "zcc_nm": float(zcc_nm),
            "energy_expression": ABFE_TUNNEL_ENERGY_EXPRESSION,
        }
    )
    return out


# Single-COM wall for CustomCentroidBondForce(1, ...): free inside, harmonic outside.
# The allowed radius is r0 on the cylinder and grows as (zcc - z)*tan_a toward the site,
# so tan_a = 0 recovers the plain cylinder.
ABFE_TUNNEL_ENERGY_EXPRESSION = (
    "0.5*k_cyl*step(radial-rallow)*(radial-rallow)^2"
    " + 0.5*k_axial*step(zmin-axisproj)*(zmin-axisproj)^2"
    " + 0.5*k_axial*step(axisproj-zmax)*(axisproj-zmax)^2;"
    "rallow=r0+tan_a*step(zcc-axisproj)*(zcc-axisproj);"
    "radial=sqrt(max(0.0, rx*rx+ry*ry+rz*rz-axisproj*axisproj));"
    "axisproj=rx*ux+ry*uy+rz*uz;"
    "rx=x1-ox; ry=y1-oy; rz=z1-oz"
)


def abfe_tunnel_wall_energy(com_nm, payload):
    """NumPy mirror of ABFE_TUNNEL_ENERGY_EXPRESSION (kJ/mol), for tests and diagnostics."""
    p = dict(payload or {})
    r = np.asarray(com_nm, dtype=float) - np.asarray(p.get("origin_nm"), dtype=float)
    u = np.asarray(p.get("direction"), dtype=float)
    z = float(np.dot(r, u))
    radial = float(np.sqrt(max(0.0, float(np.dot(r, r)) - z * z)))
    zcc = float(p.get("zcc_nm", 0.0) or 0.0)
    rallow = float(p["radius_nm"]) + float(p.get("tan_alpha", 0.0) or 0.0) * max(0.0, zcc - z)
    k_cyl = float(p["k_radial_kj_mol_nm2"])
    k_ax = float(p["k_axial_kj_mol_nm2"])
    e = 0.5 * k_cyl * max(0.0, radial - rallow) ** 2
    e += 0.5 * k_ax * max(0.0, float(p["zmin_nm"]) - z) ** 2
    e += 0.5 * k_ax * max(0.0, z - float(p["zmax_nm"])) ** 2
    return float(e)


def build_abfe_funnel_axis_payload(
    *,
    coords_nm=None,
    protein_heavy_idx=None,
    ligand_heavy_idx=None,
    tube_radius_a=5.0,
    pocket_skip_a=4.0,
    n_directions=500,
):
    """
    Pick the funnel/exit axis as the most open way from the ligand to bulk solvent.

    A direction is scored by casting a ligand-sized tube (``tube_radius_a``) from the
    ligand center: protein heavy atoms inside the tube past the pocket (``pocket_skip_a``)
    block it; atoms in a shell just outside the tube mean the path hugs the surface.
    Among the least blocked directions the one reaching solvent soonest wins.  The
    returned ``exit_depth_nm`` is how far along the axis the last nearby protein atom
    is, i.e. where the ligand is really out; the cone should reach about that far and
    the pull must go beyond it.
    """
    out = {"ok": False, "reason": "", "direction": np.asarray([1.0, 0.0, 0.0]), "origin_nm": None,
           "exit_depth_nm": np.nan, "blocked_atoms": -1, "surface_atoms": -1}
    try:
        xyz = np.asarray(coords_nm, dtype=float)
        p_idx = np.asarray(list(protein_heavy_idx or []), dtype=int)
        l_idx = np.asarray(list(ligand_heavy_idx or []), dtype=int)
    except (TypeError, ValueError):
        out["reason"] = "invalid_input"
        return out
    if xyz.ndim != 2 or xyz.shape[1] != 3 or p_idx.size == 0 or l_idx.size == 0:
        out["reason"] = "invalid_input"
        return out
    origin = np.mean(xyz[l_idx], axis=0)
    rel = xyz[p_idx] - origin[None, :]
    r2 = np.sum(rel * rel, axis=1)
    tube = 0.1 * float(tube_radius_a)
    shell = tube + 0.3
    skip = 0.1 * float(pocket_skip_a)
    n = int(max(50, n_directions))
    k = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * k / n
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * k
    dirs = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)
    fwd = rel @ dirs.T  # (n_atoms, n_dirs)
    lat = np.sqrt(np.maximum(0.0, r2[:, None] - fwd * fwd))
    ahead = fwd > skip
    blocked = np.sum(ahead & (lat < tube), axis=0)
    surface = np.sum(ahead & (lat < shell), axis=0)
    near = (fwd > 0.0) & (lat < shell)
    depth = np.where(np.any(near, axis=0), np.max(np.where(near, fwd, 0.0), axis=0), 0.0)
    order = np.lexsort((depth, surface, blocked))
    best = int(order[0])
    out.update(
        {
            "ok": True,
            "reason": "ok",
            "direction": np.asarray(dirs[best], dtype=float),
            "origin_nm": np.asarray(origin, dtype=float),
            "exit_depth_nm": float(depth[best]),
            "blocked_atoms": int(blocked[best]),
            "surface_atoms": int(surface[best]),
        }
    )
    return out


def build_abfe_memory_guard_decision(
    *,
    available_mb=None,
    total_mb=None,
    swap_used_mb=None,
    swap_used_start_mb=None,
    wired_mb=None,
    pressure_level=None,
    min_available_mb=1536.0,
    min_available_fraction=0.08,
    max_swap_used_mb=4096.0,
    max_swap_growth_mb=1024.0,
    max_wired_fraction=0.70,
):
    """
    Decide whether a long in-process MD run must stop to protect the machine.

    The failure it prevents is macOS running out of pageable memory until watchdogd misses its check-ins
    and the kernel panics (16 GB M4, 2026-09-29 and 2026-10-17).  Each trigger stops the run on its own:
    low system available memory, swap in use above max_swap_used_mb, swap grown during this run by more
    than max_swap_growth_mb, wired memory above max_wired_fraction of RAM, or critical kernel pressure.

    Never make the low-memory or swap triggers conditional on another reading.  On 2026-10-17 the guard
    was relaxed to "low memory AND 2 GB swap growth" because it stopped a run on a busy laptop; the
    next run panicked the machine with all 16 GB wired, swap nearly full and the kernel's pressure flag
    still False.  Wired memory (GPU buffers on Apple Silicon) cannot be paged out, so it is the reading
    that sees that failure coming.  Unknown readings never abort.
    """

    def _mb(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return np.nan
        return f if np.isfinite(f) and f >= 0.0 else np.nan

    def _num(v, default):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return float(default)
        return f if np.isfinite(f) else float(default)

    avail = _mb(available_mb)
    total = _mb(total_mb)
    swap = _mb(swap_used_mb)
    swap0 = _mb(swap_used_start_mb)
    wired = _mb(wired_mb)
    floor_mb = float(max(256.0, _num(min_available_mb, 1536.0)))
    if np.isfinite(total) and total > 0.0:
        frac = _num(min_available_fraction, 0.08)
        floor_mb = float(max(floor_mb, total * float(np.clip(frac, 0.0, 0.5))))
    swap_cap = _num(max_swap_used_mb, 4096.0)
    growth = float(swap - swap0) if (np.isfinite(swap) and np.isfinite(swap0)) else np.nan
    growth_cap = _num(max_swap_growth_mb, 1024.0)
    wired_frac = float(wired / total) if (np.isfinite(wired) and np.isfinite(total) and total > 0.0) else np.nan
    wired_cap = float(np.clip(_num(max_wired_fraction, 0.70), 0.3, 0.95))
    try:
        level = int(pressure_level) if pressure_level is not None else 0
    except (TypeError, ValueError):
        level = 0
    reasons = []
    if np.isfinite(avail) and avail < floor_mb:
        reasons.append(f"available_memory_low:{avail:.0f}MB<{floor_mb:.0f}MB")
    if np.isfinite(swap) and swap_cap > 0.0 and swap > swap_cap:
        reasons.append(f"swap_used_high:{swap:.0f}MB>{swap_cap:.0f}MB")
    if np.isfinite(growth) and growth_cap > 0.0 and growth > growth_cap:
        reasons.append(f"swap_grew_during_run:{growth:.0f}MB>{growth_cap:.0f}MB")
    if np.isfinite(wired_frac) and wired_frac > wired_cap:
        reasons.append(f"wired_memory_high:{wired:.0f}MB>{100.0 * wired_cap:.0f}%_of_{total:.0f}MB")
    if level >= 4:
        reasons.append("memory_pressure_critical")
    return {
        "ok": True,
        "should_abort": bool(reasons),
        "reasons": list(reasons),
        "available_mb": float(avail) if np.isfinite(avail) else None,
        "floor_mb": float(floor_mb),
        "swap_used_mb": float(swap) if np.isfinite(swap) else None,
        "swap_growth_mb": float(growth) if np.isfinite(growth) else None,
        "wired_mb": float(wired) if np.isfinite(wired) else None,
        "pressure_level": level,
    }


def build_abfe_pymol_state_budget(
    *,
    n_atoms=0,
    requested_max_states=1500,
    budget_mb=384.0,
    bytes_per_atom_state=160.0,
):
    """
    Cap the states of the final ABFE trajectory object loaded into PyMOL.

    PyMOL keeps a full coordinate set per state plus per-state representation
    geometry (cartoon, sticks) once a state is drawn, far more than the float32
    coordinates Leonardo keeps.  ``bytes_per_atom_state`` is a deliberately
    conservative estimate of that total.
    """
    try:
        n = int(n_atoms or 0)
    except (TypeError, ValueError):
        n = 0
    n = max(1, n)
    try:
        req = int(requested_max_states or 1)
    except (TypeError, ValueError):
        req = 1
    req = max(1, req)
    try:
        budget = float(budget_mb)
    except (TypeError, ValueError):
        budget = 384.0
    budget = float(max(16.0, budget if np.isfinite(budget) else 384.0))
    try:
        per_atom = float(bytes_per_atom_state)
    except (TypeError, ValueError):
        per_atom = 160.0
    per_atom = float(max(12.0, per_atom if np.isfinite(per_atom) else 160.0))
    mb_per_state = (n * per_atom) / 1048576.0
    by_memory = int(max(2, np.floor(budget / max(mb_per_state, 1.0e-9))))
    max_states = int(min(req, by_memory))
    return {
        "ok": True,
        "max_states": int(max_states),
        "estimated_mb": float(mb_per_state * max_states),
        "limited_by_memory": bool(by_memory < req),
    }


def build_abfe_adaptive_steering_state_payload(
    *,
    enabled=False,
    steering_mode="pull_hold",
    initial_pull_k_kj_nm2=800.0,
    n_direction_candidates=1,
    ligand_heavy_atoms=0,
    formal_charge_e=np.nan,
    ring_count=0,
    rotatable_bonds=0,
    pocket_desc=None,
    direction_clearances_nm=None,
    direction_collision_penalties=None,
    enable_probe_phase=False,
    sample_every_ps=None,
    update_interval_samples=None,
    stall_window_samples=None,
):
    try:
        k0 = float(initial_pull_k_kj_nm2)
    except Exception:
        k0 = 800.0
    if (not np.isfinite(k0)) or (k0 <= 0.0):
        k0 = 800.0
    k0 = float(max(200.0, k0))

    try:
        n_dirs = int(n_direction_candidates or 0)
    except Exception:
        n_dirs = 0
    n_dirs = max(1, int(n_dirs))

    try:
        lig_heavy = int(ligand_heavy_atoms or 0)
    except Exception:
        lig_heavy = 0
    lig_heavy = max(0, int(lig_heavy))
    size_scale = float(np.clip((float(lig_heavy) - 8.0) / 24.0, 0.0, 1.0))
    try:
        formal_q = float(formal_charge_e)
    except Exception:
        formal_q = np.nan
    if not np.isfinite(formal_q):
        formal_q = np.nan
    try:
        ring_n = int(ring_count or 0)
    except Exception:
        ring_n = 0
    ring_n = max(0, int(ring_n))
    try:
        rot_n = int(rotatable_bonds or 0)
    except Exception:
        rot_n = 0
    rot_n = max(0, int(rot_n))
    pocket_row = _normalize_abfe_policy_pocket_desc(pocket_desc)
    try:
        pocket_density = float(pocket_row.get("pocket_atom_density_per_a3", np.nan))
    except Exception:
        pocket_density = np.nan
    try:
        pocket_hydrophobic = float(pocket_row.get("pocket_frac_hydrophobic", np.nan))
    except Exception:
        pocket_hydrophobic = np.nan
    try:
        pocket_polar = float(pocket_row.get("pocket_frac_polar", np.nan))
    except Exception:
        pocket_polar = np.nan
    try:
        pocket_charge_density = float(pocket_row.get("pocket_charge_density", np.nan))
    except Exception:
        pocket_charge_density = np.nan
    charge_scale = float(np.clip(abs(formal_q) / 2.0, 0.0, 2.0)) if np.isfinite(formal_q) else 0.0
    ring_scale = float(np.clip(float(ring_n) / 4.0, 0.0, 2.0))
    flex_scale = float(np.clip(float(rot_n) / 12.0, 0.0, 2.0))
    buried_scale = float(np.clip(pocket_density / 0.04, 0.0, 2.0)) if np.isfinite(pocket_density) else 0.0
    pocket_polar_scale = float(np.clip(pocket_polar, 0.0, 1.5)) if np.isfinite(pocket_polar) else 0.0
    pocket_hydrophobic_scale = float(np.clip(pocket_hydrophobic, 0.0, 1.5)) if np.isfinite(pocket_hydrophobic) else 0.0
    pocket_charge_scale = float(np.clip(abs(pocket_charge_density) / 0.03, 0.0, 2.0)) if np.isfinite(pocket_charge_density) else 0.0
    clearances = []
    for value in list(direction_clearances_nm or []):
        try:
            vv = float(value)
        except Exception:
            vv = np.nan
        clearances.append(float(vv) if np.isfinite(vv) else np.nan)
    collision_penalties = []
    for value in list(direction_collision_penalties or []):
        try:
            vv = float(value)
        except Exception:
            vv = np.nan
        collision_penalties.append(float(vv) if np.isfinite(vv) else np.nan)
    finite_clearances = np.asarray([v for v in clearances if np.isfinite(v)], dtype=float)
    best_clearance_nm = float(np.max(finite_clearances)) if finite_clearances.size else np.nan
    clearance_penalty_scale = (
        float(np.clip((0.90 - best_clearance_nm) / 0.35, 0.0, 2.0))
        if np.isfinite(best_clearance_nm)
        else 0.0
    )
    try:
        sample_ps = float(sample_every_ps)
    except Exception:
        sample_ps = 2.0
    if (not np.isfinite(sample_ps)) or (sample_ps <= 0.0):
        sample_ps = 2.0

    if update_interval_samples is None:
        update_i = int(max(3, round(4 + (2 * size_scale))))
    else:
        try:
            update_i = int(update_interval_samples)
        except Exception:
            update_i = 6
        update_i = max(2, int(update_i))

    if stall_window_samples is None:
        stall_i = int(max(update_i + 2, round(8 + (4 * size_scale))))
    else:
        try:
            stall_i = int(stall_window_samples)
        except Exception:
            stall_i = update_i + 4
        stall_i = max(update_i + 1, int(stall_i))

    mode_txt = str(steering_mode or "").strip().lower()
    if mode_txt not in {"pull_hold", "pull_hold_return"}:
        mode_txt = "pull_hold"

    if n_dirs >= 6:
        switch_cap = 5
    elif n_dirs >= 5:
        switch_cap = 4
    elif n_dirs >= 4:
        switch_cap = 3
    elif n_dirs >= 3:
        switch_cap = 2
    else:
        switch_cap = 1
    max_switches = int(min(max(0, n_dirs - 1), switch_cap))
    if not bool(enabled):
        max_switches = 0

    force_scale = float(np.clip(1.0 + (0.12 * charge_scale) + (0.10 * buried_scale) + (0.10 * flex_scale) - (0.15 * clearance_penalty_scale), 0.75, 1.50))
    k_target = float(max(200.0, k0 * force_scale))
    k_min = float(max(200.0, 0.45 * k_target))
    k_max = float(max(k_target + (1200.0 + 2200.0 * size_scale), k_target * (3.10 + 0.80 * size_scale + 0.10 * flex_scale)))
    bulk_escape_scale = float(np.clip((0.65 * size_scale) + (0.35 * flex_scale) + (0.18 * ring_scale), 0.0, 2.5))
    sticky_escape_scale = float(
        np.clip(
            (0.55 * buried_scale)
            + (0.45 * pocket_hydrophobic_scale)
            + (0.30 * clearance_penalty_scale)
            + (0.15 * pocket_charge_scale),
            0.0,
            2.5,
        )
    )
    pull_distance_target_scale = float(
        np.clip(
            1.0
            + (0.16 * size_scale)
            + (0.08 * flex_scale)
            + (0.05 * ring_scale)
            + (0.12 * buried_scale)
            + (0.10 * pocket_hydrophobic_scale)
            + (0.12 * clearance_penalty_scale),
            1.0,
            1.65,
        )
    )
    pull_distance_probe_scale = float(
        np.clip(
            1.0 + (0.04 * size_scale) + (0.02 * buried_scale) - (0.02 * clearance_penalty_scale),
            1.0,
            min(1.15, pull_distance_target_scale),
        )
    )
    pull_distance_max_scale = float(
        np.clip(
            pull_distance_target_scale
            + 0.28
            + (0.18 * size_scale)
            + (0.10 * flex_scale)
            + (0.10 * buried_scale)
            + (0.10 * clearance_penalty_scale),
            max(1.25, pull_distance_target_scale),
            2.15,
        )
    )
    pull_distance_push_scale = float(
        np.clip(1.08 + (0.08 * size_scale) + (0.05 * flex_scale) + (0.04 * buried_scale), 1.05, 1.28)
    )
    pull_distance_relax_scale = float(np.clip(0.94 - (0.02 * clearance_penalty_scale), 0.82, 0.96))
    progress_site_thresh = float(0.26 + (0.16 * size_scale) + (0.06 * flex_scale) + (0.04 * buried_scale))
    progress_reactive_thresh = float(0.08 + (0.08 * size_scale) + (0.04 * flex_scale))
    far_hold_samples = int(max(2, round((20.0 + (10.0 * size_scale)) / (sample_ps * max(1, update_i)))))
    far_hold_fraction = 0.55
    far_hold_floor_fraction = float(np.clip(0.18 + (0.04 * size_scale) + (0.03 * buried_scale), 0.16, 0.28))
    far_hold_relaxed_fraction = float(np.clip(far_hold_floor_fraction + 0.04 + (0.03 * size_scale), 0.20, 0.34))
    far_hold_relax_after_samples = int(max(2, round(2 + (1.0 * size_scale) + (1.0 * buried_scale))))
    detached_settle_enabled = bool(enabled and (size_scale >= 0.35 or sticky_escape_scale >= 0.45))
    detached_settle_samples = int(
        max(4, round((16.0 + (10.0 * size_scale) + (8.0 * buried_scale)) / (sample_ps * max(1, update_i))))
    )
    detached_settle_lambda_value = float(np.clip(0.008 + (0.006 * size_scale) + (0.004 * buried_scale), 0.008, 0.02))
    detached_settle_max_lambda = float(np.clip(detached_settle_lambda_value + 0.05, 0.06, 0.10))
    detached_settle_floor_fraction = float(np.clip(0.004 + (0.004 * size_scale), 0.003, 0.010))
    detached_settle_k_floor = float(max(5.0, k_target * detached_settle_floor_fraction))
    far_site_com = float(9.5 + (2.0 * size_scale) + (1.0 * flex_scale) + (0.5 * buried_scale))
    far_min_dist = float(3.0 + (0.15 * size_scale) + (0.10 * buried_scale))
    far_max_contacts = int(max(1, round(3.0 + (0.12 * lig_heavy) + (1.0 * pocket_polar_scale))))
    far_release_site = float(max(8.5, far_site_com - 0.75)) if 'far_site_com' in locals() else float(max(8.5, far_site_com))
    far_release_min = float(max(2.7, far_min_dist - 0.15))
    far_release_contacts = int(max(far_max_contacts + 1, far_max_contacts))
    target_escape_site = float(max(8.0 + (4.0 * size_scale), far_site_com))
    bound_site_com = float(3.8 + (0.5 * size_scale) + (0.25 * flex_scale))
    bound_max_min_dist = 3.5
    bound_min_contacts = int(max(4, round(4.0 + (0.18 * lig_heavy) + (1.0 * pocket_hydrophobic_scale))))
    probe_enabled_flag = bool(enabled and enable_probe_phase and n_dirs > 1)
    probe_direction_limit = int(min(n_dirs, max(2, 3 + int(size_scale >= 0.65))))
    probe_window_samples = int(max(update_i, round(6 + (2 * size_scale) + (2 * buried_scale))))
    probe_lambda_value = float(np.clip(0.55 + (0.05 * charge_scale) - (0.05 * clearance_penalty_scale), 0.45, 0.70))
    probe_force_scale = float(np.clip(0.60 + (0.06 * charge_scale) + (0.05 * buried_scale) - (0.08 * clearance_penalty_scale), 0.45, 0.85))
    probe_k = float(np.clip(k_target * probe_force_scale, k_min, k_target))
    pull_surface_detach_enabled = bool(
        enabled
        and (
            (size_scale >= 0.45)
            or (buried_scale >= 0.55)
            or (pocket_hydrophobic_scale >= 0.60)
            or (clearance_penalty_scale >= 0.60)
        )
    )
    pull_surface_detach_activation_stalls = int(max(1, 2 if (size_scale >= 0.45 or buried_scale >= 0.55) else 3))
    pull_surface_detach_blend_fraction = float(
        np.clip(0.22 + (0.12 * size_scale) + (0.08 * buried_scale) + (0.06 * clearance_penalty_scale), 0.20, 0.55)
    )
    surface_detach_distance_scale = float(
        np.clip(1.15 + (0.10 * size_scale) + (0.12 * buried_scale) + (0.08 * pocket_hydrophobic_scale), 1.05, 1.85)
    )
    surface_detach_max_distance_nm = float(np.clip(2.40 + (0.35 * size_scale) + (0.20 * buried_scale), 2.40, 3.40))
    # The bound-settle pre-phase is part of the probe/corridor workflow. Plain
    # pull/hold runs should start pulling immediately, even for large ligands,
    # so existing protocols remain predictable.
    bound_settle_enabled = bool(enabled and enable_probe_phase and (size_scale >= 0.45 or sticky_escape_scale >= 0.55))
    bound_settle_lambda_value = float(np.clip(0.08 + (0.03 * charge_scale) + (0.02 * buried_scale), 0.05, 0.18))
    bound_settle_samples = int(
        max(2, round((6.0 + (4.0 * size_scale) + (3.0 * min(sticky_escape_scale, 1.5))) / max(sample_ps, 0.25)))
    )
    bound_settle_k = float(np.clip(max(k_min, probe_k * 0.80), k_min, probe_k if probe_enabled_flag else k_target))
    if bound_settle_enabled:
        current_k0 = float(bound_settle_k)
        current_pull_distance_scale0 = 1.0
        phase0 = "bound_settle"
    elif probe_enabled_flag:
        current_k0 = float(probe_k)
        current_pull_distance_scale0 = float(pull_distance_probe_scale)
        phase0 = "probe"
    else:
        current_k0 = float(k_target)
        current_pull_distance_scale0 = float(pull_distance_target_scale)
        phase0 = "pull"

    return {
        "ok": True,
        "enabled": bool(enabled),
        "steering_mode": str(mode_txt),
        "initial_pull_k_kj_nm2": float(k_target),
        "input_pull_k_kj_nm2": float(k0),
        "current_pull_k_kj_nm2": float(current_k0),
        "current_pull_distance_scale": float(current_pull_distance_scale0),
        "target_pull_distance_scale": float(pull_distance_target_scale),
        "probe_pull_distance_scale": float(pull_distance_probe_scale),
        "pull_distance_scale_min": 1.0,
        "pull_distance_scale_max": float(pull_distance_max_scale),
        "pull_distance_push_scale": float(pull_distance_push_scale),
        "pull_distance_relax_scale": float(pull_distance_relax_scale),
        "k_min_kj_nm2": float(k_min),
        "k_max_kj_nm2": float(k_max),
        "update_interval_samples": int(update_i),
        "stall_window_samples": int(stall_i),
        "pull_phase_lambda_threshold": 0.25,
        "high_lambda_threshold": 0.70,
        "progress_site_com_threshold_a": float(progress_site_thresh),
        "progress_reactive_threshold_a": float(progress_reactive_thresh),
        "target_escape_site_com_a": float(target_escape_site),
        "bound_site_com_threshold_a": float(bound_site_com),
        "bound_max_min_dist_a": float(bound_max_min_dist),
        "bound_min_contact_count_4a": int(bound_min_contacts),
        "far_unbound_site_com_a": float(far_site_com),
        "far_unbound_min_dist_a": float(far_min_dist),
        "far_unbound_max_contact_count_4a": int(far_max_contacts),
        "far_hold_release_site_com_a": float(far_release_site),
        "far_hold_release_min_dist_a": float(far_release_min),
        "far_hold_release_max_contact_count_4a": int(far_release_contacts),
        "far_hold_target_samples": int(far_hold_samples),
        "far_hold_count": 0,
        "far_hold_k_fraction": float(far_hold_fraction),
        "far_hold_k_floor_kj_nm2": float(max(200.0, k_target * far_hold_floor_fraction)),
        "far_hold_relaxed_k_fraction": float(far_hold_relaxed_fraction),
        "far_hold_relax_after_samples": int(far_hold_relax_after_samples),
        "far_hold_target_k_kj_nm2": np.nan,
        "detached_settle_enabled": bool(detached_settle_enabled),
        "detached_settle_lambda_value": float(detached_settle_lambda_value),
        "detached_settle_max_lambda": float(detached_settle_max_lambda),
        "detached_settle_target_samples": int(detached_settle_samples),
        "detached_settle_count": 0,
        "detached_settle_k_floor_kj_nm2": float(detached_settle_k_floor),
        "probe_enabled": bool(probe_enabled_flag),
        "bound_settle_enabled": bool(bound_settle_enabled),
        "bound_settle_lambda_value": float(bound_settle_lambda_value),
        "bound_settle_target_samples": int(bound_settle_samples),
        "bound_settle_count": 0,
        "bound_settle_pull_k_kj_nm2": float(bound_settle_k),
        "probe_phase_lambda_value": float(probe_lambda_value),
        "probe_window_samples": int(probe_window_samples),
        "probe_direction_limit": int(probe_direction_limit),
        "probe_pull_k_kj_nm2": float(probe_k),
        "post_probe_pull_k_kj_nm2": float(k_target),
        "post_probe_pull_distance_scale": float(pull_distance_target_scale),
        "probe_best_direction_index": 0,
        "probe_best_score": np.nan,
        "probe_scores": [np.nan for _ in range(int(n_dirs))],
        "probe_counts": [0 for _ in range(int(n_dirs))],
        "probe_early_accept_score": float(1.35 + (0.15 * buried_scale) + (0.10 * charge_scale)),
        "direction_clearances_nm": list(clearances[: int(n_dirs)]),
        "direction_collision_penalties": list(collision_penalties[: int(n_dirs)]),
        "formal_charge_e": float(formal_q) if np.isfinite(formal_q) else np.nan,
        "ring_count": int(ring_n),
        "rotatable_bonds": int(rot_n),
        "pocket_buried_scale": float(buried_scale),
        "pocket_polar_scale": float(pocket_polar_scale),
        "pocket_hydrophobic_scale": float(pocket_hydrophobic_scale),
        "pocket_charge_scale": float(pocket_charge_scale),
        "best_corridor_clearance_nm": float(best_clearance_nm) if np.isfinite(best_clearance_nm) else np.nan,
        "clearance_penalty_scale": float(clearance_penalty_scale),
        "bound_hold_min_run_samples": 4,
        "unbound_hold_min_run_samples": 4,
        "bound_hold_max_gap_samples": 0,
        "unbound_hold_max_gap_samples": 3,
        "unbound_hold_max_gap_cap_samples": 6,
        "stall_count": 0,
        "bulk_escape_scale": float(bulk_escape_scale),
        "sticky_escape_scale": float(sticky_escape_scale),
        "far_hold_break_count": 0,
        "recent_far_hold_break": False,
        "clash_soft_min_dist_a": 1.90,
        "clash_hard_min_dist_a": 1.60,
        "energy_spike_threshold_kj": float(max(120.0, 0.10 * k0)),
        "n_direction_candidates": int(n_dirs),
        "max_direction_switches": int(max_switches),
        "switches_used": 0,
        "adaptation_count": 0,
        "last_update_sample": -999999,
        "last_direction_index": 0,
        "keep_pull_lambda_high_until_far_hold": bool(enabled),
        "high_lambda_seen": False,
        "surface_detach_enabled": bool(enabled),
        "surface_detach_contact_cutoff_nm": 0.45,
        "surface_detach_distance_scale": float(surface_detach_distance_scale),
        "surface_detach_max_distance_nm": float(surface_detach_max_distance_nm),
        "pull_surface_detach_enabled": bool(pull_surface_detach_enabled),
        "pull_surface_detach_active": False,
        "pull_surface_detach_activation_stalls": int(pull_surface_detach_activation_stalls),
        "pull_surface_detach_blend_fraction": float(pull_surface_detach_blend_fraction),
        "ligand_heavy_atoms": int(lig_heavy),
        "size_scale": float(size_scale),
        "phase": str(phase0),
    }


def build_abfe_probe_direction_score_payload(
    *,
    progress_site_com_a=np.nan,
    progress_reactive_a=np.nan,
    latest_min_dist_a=np.nan,
    latest_contact_count_4a=np.nan,
    latest_total_e_kj=np.nan,
    median_total_e_kj=np.nan,
    clearance_nm=np.nan,
    collision_penalty=np.nan,
    soft_clash=False,
    hard_clash=False,
    energy_spike=False,
):
    score = 0.0
    if np.isfinite(progress_site_com_a):
        score += 1.80 * float(progress_site_com_a)
    if np.isfinite(progress_reactive_a):
        score += 0.70 * float(progress_reactive_a)
    if np.isfinite(latest_min_dist_a):
        score += 0.60 * float(np.clip(latest_min_dist_a - 2.5, 0.0, 4.0))
    if np.isfinite(latest_contact_count_4a):
        score -= 0.35 * float(np.clip(latest_contact_count_4a, 0.0, 12.0))
    if np.isfinite(clearance_nm):
        score += 2.00 * float(np.clip(clearance_nm, 0.0, 1.5))
    if np.isfinite(collision_penalty):
        score -= 0.0025 * float(np.clip(collision_penalty, 0.0, 500.0))
    if np.isfinite(latest_total_e_kj) and np.isfinite(median_total_e_kj):
        score -= 0.010 * float(np.clip(latest_total_e_kj - median_total_e_kj, 0.0, 200.0))
    if soft_clash:
        score -= 1.25
    if hard_clash:
        score -= 3.50
    if energy_spike:
        score -= 0.80
    return {
        "ok": True,
        "score": float(score),
        "soft_clash": bool(soft_clash),
        "hard_clash": bool(hard_clash),
        "energy_spike": bool(energy_spike),
    }


def _sanitize_abfe_steering_policy_mode(policy_mode):
    mode_txt = str(policy_mode or "").strip().lower()
    if mode_txt == "safe_rl":
        mode_txt = "bandit"
    if mode_txt not in {"heuristic", "bandit"}:
        mode_txt = "heuristic"
    return str(mode_txt)


def _sanitize_abfe_steering_policy_model_config(model_config, source=""):
    cfg = dict(model_config or {})
    feature_names = list(cfg.get("feature_names", []) or [])
    if not feature_names:
        feature_names = list(ABFE_STEERING_POLICY_FEATURE_NAMES)
    clean_feature_names = []
    seen = set()
    for name in feature_names:
        txt = str(name or "").strip()
        if txt and txt not in seen:
            clean_feature_names.append(txt)
            seen.add(txt)
    if not clean_feature_names:
        clean_feature_names = list(ABFE_STEERING_POLICY_FEATURE_NAMES)

    clean_actions = {}
    for action_name in ABFE_STEERING_ACTION_NAMES:
        raw_action = dict((cfg.get("actions", {}) or {}).get(action_name, {}) or {})
        raw_weights = list(raw_action.get("weights", []) or [])
        if len(raw_weights) != len(clean_feature_names):
            continue
        weights = []
        valid = True
        for value in raw_weights:
            try:
                vv = float(value)
            except Exception:
                valid = False
                break
            if not np.isfinite(vv):
                valid = False
                break
            weights.append(float(vv))
        if not valid:
            continue
        try:
            n_train = int(raw_action.get("n_train", 0) or 0)
        except Exception:
            n_train = 0
        try:
            mean_reward = float(raw_action.get("mean_reward", np.nan))
        except Exception:
            mean_reward = np.nan
        clean_actions[action_name] = {
            "weights": list(weights),
            "n_train": int(max(0, n_train)),
            "mean_reward": float(mean_reward) if np.isfinite(mean_reward) else np.nan,
        }

    if not clean_actions:
        return {
            "ok": False,
            "source": str(source or ""),
            "feature_names": list(clean_feature_names),
            "actions": {},
            "version": "",
            "score_blend_alpha": 0.0,
            "min_action_score": 0.35,
            "note": "no_valid_actions",
        }

    try:
        blend_alpha = float(cfg.get("score_blend_alpha", 0.35))
    except Exception:
        blend_alpha = 0.35
    if (not np.isfinite(blend_alpha)) or blend_alpha < 0.0:
        blend_alpha = 0.35
    blend_alpha = float(np.clip(blend_alpha, 0.0, 2.0))

    try:
        min_action_score = float(cfg.get("min_action_score", 0.35))
    except Exception:
        min_action_score = 0.35
    if not np.isfinite(min_action_score):
        min_action_score = 0.35
    min_action_score = float(np.clip(min_action_score, 0.0, 5.0))

    clean_ood_feature_names = []
    seen_ood = set()
    for name in list(cfg.get("ood_feature_names", []) or []):
        txt = str(name or "").strip()
        if txt and txt not in seen_ood:
            clean_ood_feature_names.append(txt)
            seen_ood.add(txt)
    if not clean_ood_feature_names:
        clean_ood_feature_names = [n for n in ABFE_STEERING_POLICY_OOD_FEATURE_NAMES if n in set(clean_feature_names)]
    raw_ood_center = list(cfg.get("ood_center", []) or [])
    raw_ood_scale = list(cfg.get("ood_scale", []) or [])
    clean_ood_center = []
    clean_ood_scale = []
    ood_enabled = False
    if clean_ood_feature_names and len(raw_ood_center) == len(clean_ood_feature_names) and len(raw_ood_scale) == len(clean_ood_feature_names):
        valid_ood = True
        for cc, ss in zip(raw_ood_center, raw_ood_scale):
            try:
                c_f = float(cc)
                s_f = float(ss)
            except Exception:
                valid_ood = False
                break
            if (not np.isfinite(c_f)) or (not np.isfinite(s_f)) or (s_f <= 0.0):
                valid_ood = False
                break
            clean_ood_center.append(float(c_f))
            clean_ood_scale.append(float(max(s_f, 1.0e-6)))
        ood_enabled = bool(valid_ood)
    if not ood_enabled:
        clean_ood_feature_names = []
        clean_ood_center = []
        clean_ood_scale = []
    try:
        ood_distance_threshold = float(cfg.get("ood_distance_threshold", np.nan))
    except Exception:
        ood_distance_threshold = np.nan
    if not np.isfinite(ood_distance_threshold):
        ood_distance_threshold = np.nan
    try:
        ood_max_abs_z = float(cfg.get("ood_max_abs_z", np.nan))
    except Exception:
        ood_max_abs_z = np.nan
    if not np.isfinite(ood_max_abs_z):
        ood_max_abs_z = np.nan

    return {
        "ok": True,
        "source": str(source or ""),
        "version": str(cfg.get("version", "") or "safe_bandit_linear_v1"),
        "model_type": str(cfg.get("model_type", "") or "ridge_action_value"),
        "feature_names": list(clean_feature_names),
        "actions": dict(clean_actions),
        "score_blend_alpha": float(blend_alpha),
        "min_action_score": float(min_action_score),
        "ood_enabled": bool(ood_enabled),
        "ood_feature_names": list(clean_ood_feature_names),
        "ood_center": list(clean_ood_center),
        "ood_scale": list(clean_ood_scale),
        "ood_distance_threshold": float(ood_distance_threshold) if np.isfinite(ood_distance_threshold) else np.nan,
        "ood_max_abs_z": float(ood_max_abs_z) if np.isfinite(ood_max_abs_z) else np.nan,
        "train_rows": int(max(0, int(cfg.get("train_rows", 0) or 0))),
        "n_trace_files": int(max(0, int(cfg.get("n_trace_files", 0) or 0))),
    }


def _load_abfe_steering_policy_model_config(policy_model_path=""):
    raw_path = str(policy_model_path or os.environ.get(ABFE_STEERING_POLICY_MODEL_ENV, "") or "").strip()
    if not raw_path:
        return {
            "ok": False,
            "source": "",
            "feature_names": list(ABFE_STEERING_POLICY_FEATURE_NAMES),
            "actions": {},
            "version": "",
            "score_blend_alpha": 0.0,
            "min_action_score": 0.35,
            "note": "disabled",
        }

    source = os.path.abspath(os.path.expanduser(raw_path))
    cached = _ABFE_STEERING_POLICY_MODEL_CACHE.get(source)
    if isinstance(cached, dict):
        return dict(cached)
    if not os.path.isfile(source):
        payload = {
            "ok": False,
            "source": str(source),
            "feature_names": list(ABFE_STEERING_POLICY_FEATURE_NAMES),
            "actions": {},
            "version": "",
            "score_blend_alpha": 0.0,
            "min_action_score": 0.35,
            "note": "missing_file",
        }
        _ABFE_STEERING_POLICY_MODEL_CACHE[source] = dict(payload)
        return payload

    try:
        with open(source, "r", encoding="utf-8") as fh:
            raw_cfg = json.load(fh)
    except Exception as exc:
        payload = {
            "ok": False,
            "source": str(source),
            "feature_names": list(ABFE_STEERING_POLICY_FEATURE_NAMES),
            "actions": {},
            "version": "",
            "score_blend_alpha": 0.0,
            "min_action_score": 0.35,
            "note": f"json_error:{exc.__class__.__name__}",
        }
        _ABFE_STEERING_POLICY_MODEL_CACHE[source] = dict(payload)
        return payload

    payload = _sanitize_abfe_steering_policy_model_config(raw_cfg, source=str(source))
    _ABFE_STEERING_POLICY_MODEL_CACHE[source] = dict(payload)
    return dict(payload)


def build_abfe_steering_policy_feature_payload(
    *,
    observation=None,
    policy_state=None,
):
    obs = dict(observation or {})
    pstate = dict(policy_state or {})
    phase = str(obs.get("phase", "pull") or "pull").strip().lower()
    if phase not in {"pull", "far_hold", "detached_settle", "return"}:
        phase = "pull"

    def _float_val(key, default=np.nan):
        try:
            out = float(obs.get(key, pstate.get(key, default)))
        except Exception:
            out = float(default)
        return float(out) if np.isfinite(out) else float(default)

    def _bool_val(key, default=False):
        if key in obs:
            return bool(obs.get(key, default))
        return bool(pstate.get(key, default))

    k_fraction = float(np.clip(_float_val("k_fraction", 0.0), 0.0, 1.5))
    escape_gap = float(np.clip(_float_val("escape_gap_fraction", 0.0), 0.0, 3.0))
    contact_pressure = float(np.clip(_float_val("contact_pressure", 0.0), 0.0, 4.0))
    size_scale = float(np.clip(_float_val("size_scale", 0.0), 0.0, 2.0))
    charge_density = float(np.clip(_float_val("charge_density", 0.0), 0.0, 2.0))
    formal_q = _float_val("formal_charge_e", np.nan)
    ring_count = int(max(0, int(round(_float_val("ring_count", pstate.get("ring_count", 0) or 0)))))
    rotatable_bonds = int(
        max(0, int(round(_float_val("rotatable_bonds", pstate.get("rotatable_bonds", 0) or 0))))
    )
    formal_charge_scaled = float(np.clip(formal_q / 2.0, -2.0, 2.0)) if np.isfinite(formal_q) else 0.0
    pocket_n_atoms = _float_val("pocket_n_atoms", np.nan)
    pocket_bbox_vol = _float_val("pocket_volume_bbox_a3", np.nan)
    pocket_atom_density = _float_val("pocket_atom_density_per_a3", np.nan)
    pocket_frac_h = _float_val("pocket_frac_hydrophobic", np.nan)
    pocket_frac_p = _float_val("pocket_frac_polar", np.nan)
    pocket_charge_density = _float_val("pocket_charge_density", np.nan)
    pocket_n_atoms_scaled = float(np.clip(pocket_n_atoms / 100.0, 0.0, 3.0)) if np.isfinite(pocket_n_atoms) else 0.0
    pocket_volume_bbox_scaled = float(np.clip(pocket_bbox_vol / 1500.0, 0.0, 3.0)) if np.isfinite(pocket_bbox_vol) else 0.0
    pocket_atom_density_scaled = float(np.clip(pocket_atom_density / 0.05, 0.0, 3.0)) if np.isfinite(pocket_atom_density) else 0.0
    pocket_charge_density_scaled = float(np.clip(pocket_charge_density / 0.05, -3.0, 3.0)) if np.isfinite(pocket_charge_density) else 0.0
    far_min = max(1.0, _float_val("far_unbound_min_dist_a", 3.5))
    far_site = max(1.0, _float_val("far_unbound_site_com_a", 10.0))
    far_contacts = max(1.0, _float_val("far_unbound_max_contact_count_4a", 4.0))
    latest_min = _float_val("latest_min_dist_a", np.nan)
    latest_site = _float_val("latest_site_com_dist_a", np.nan)
    latest_contacts = _float_val("latest_contact_count_4a", np.nan)
    latest_min_scaled = (
        float(np.clip(latest_min / far_min, 0.0, 3.0)) if np.isfinite(latest_min) else 0.0
    )
    latest_site_scaled = (
        float(np.clip(latest_site / far_site, 0.0, 3.0)) if np.isfinite(latest_site) else 0.0
    )
    latest_contact_scaled = (
        float(np.clip(latest_contacts / far_contacts, 0.0, 4.0)) if np.isfinite(latest_contacts) else 0.0
    )
    hold_progress_fraction = float(np.clip(_float_val("hold_progress_fraction", 0.0), 0.0, 2.0))
    hold_gap_fraction = float(np.clip(_float_val("hold_gap_fraction", 0.0), 0.0, 2.0))

    features = {
        "bias": 1.0,
        "phase_pull": 1.0 if phase == "pull" else 0.0,
        "phase_far_hold": 1.0 if phase == "far_hold" else 0.0,
        "phase_detached_settle": 1.0 if phase == "detached_settle" else 0.0,
        "phase_return": 1.0 if phase == "return" else 0.0,
        "actionable": 1.0 if _bool_val("actionable", False) else 0.0,
        "stalled": 1.0 if _bool_val("stalled", False) else 0.0,
        "soft_clash": 1.0 if _bool_val("soft_clash", False) else 0.0,
        "hard_clash": 1.0 if _bool_val("hard_clash", False) else 0.0,
        "energy_spike": 1.0 if _bool_val("energy_spike", False) else 0.0,
        "far_ready": 1.0 if _bool_val("far_ready", False) else 0.0,
        "far_hold_sustain": 1.0 if _bool_val("far_hold_sustain", False) else 0.0,
        "can_switch": 1.0 if _bool_val("can_switch", False) else 0.0,
        "k_fraction": float(k_fraction),
        "escape_gap_fraction": float(escape_gap),
        "contact_pressure": float(contact_pressure),
        "size_scale": float(size_scale),
        "charge_density": float(charge_density),
        "formal_charge_scaled": float(formal_charge_scaled),
        "ring_count_scale": float(np.clip(float(ring_count) / 4.0, 0.0, 2.0)),
        "rotatable_bonds_scale": float(np.clip(float(rotatable_bonds) / 12.0, 0.0, 2.0)),
        "pocket_n_atoms_scaled": float(pocket_n_atoms_scaled),
        "pocket_volume_bbox_scaled": float(pocket_volume_bbox_scaled),
        "pocket_atom_density_scaled": float(pocket_atom_density_scaled),
        "pocket_frac_hydrophobic": float(np.clip(pocket_frac_h, 0.0, 1.5)) if np.isfinite(pocket_frac_h) else 0.0,
        "pocket_frac_polar": float(np.clip(pocket_frac_p, 0.0, 1.5)) if np.isfinite(pocket_frac_p) else 0.0,
        "pocket_charge_density_scaled": float(pocket_charge_density_scaled),
        "metal_contact_expected": 1.0 if _bool_val("metal_contact_expected", False) else 0.0,
        "latest_min_dist_scaled": float(latest_min_scaled),
        "latest_site_com_scaled": float(latest_site_scaled),
        "latest_contact_scaled": float(latest_contact_scaled),
        "hold_progress_fraction": float(hold_progress_fraction),
        "hold_gap_fraction": float(hold_gap_fraction),
        "recent_far_hold_break": 1.0 if _bool_val("recent_far_hold_break", False) else 0.0,
    }
    vector = [float(features.get(name, 0.0)) for name in ABFE_STEERING_POLICY_FEATURE_NAMES]
    return {
        "ok": True,
        "phase": str(phase),
        "feature_names": list(ABFE_STEERING_POLICY_FEATURE_NAMES),
        "features": dict(features),
        "vector": list(vector),
    }


def build_abfe_steering_policy_state_payload(
    *,
    enabled=False,
    policy_mode="heuristic",
    ligand_heavy_atoms=0,
    formal_charge_e=np.nan,
    ring_count=0,
    rotatable_bonds=0,
    metal_contact_expected=False,
    target_hint="",
    ligand_smiles="",
    ligand_scaffold="",
    pocket_desc=None,
    policy_model_path="",
):
    mode_txt = _sanitize_abfe_steering_policy_mode(policy_mode)
    try:
        lig_heavy = int(ligand_heavy_atoms or 0)
    except Exception:
        lig_heavy = 0
    lig_heavy = max(0, int(lig_heavy))
    size_scale = float(np.clip((float(lig_heavy) - 8.0) / 24.0, 0.0, 1.0))

    try:
        formal_q = float(formal_charge_e)
    except Exception:
        formal_q = np.nan
    if not np.isfinite(formal_q):
        formal_q = np.nan

    try:
        ring_n = int(ring_count or 0)
    except Exception:
        ring_n = 0
    ring_n = max(0, int(ring_n))

    try:
        rot_n = int(rotatable_bonds or 0)
    except Exception:
        rot_n = 0
    rot_n = max(0, int(rot_n))

    enabled_flag = bool(enabled and mode_txt != "heuristic")
    max_hold_ext = int(max(1, round(2.0 + (2.0 * size_scale))))
    policy_model_cfg = _load_abfe_steering_policy_model_config(policy_model_path=policy_model_path)
    pocket_row = _normalize_abfe_policy_pocket_desc(pocket_desc)
    target_txt = str(target_hint or "").strip()
    smiles_txt = str(ligand_smiles or "").strip()
    scaffold_txt = str(ligand_scaffold or "").strip()
    if (not scaffold_txt) and smiles_txt:
        scaffold_txt = str(infer_abfe_steering_scaffold_key(smiles_txt) or "").strip()
    out = {
        "ok": True,
        "enabled": bool(enabled_flag),
        "policy_mode": str(mode_txt),
        "model_version": str(policy_model_cfg.get("version", "") or "safe_bandit_v1"),
        "model_source": str(policy_model_cfg.get("source", "") or ""),
        "policy_model_loaded": bool(policy_model_cfg.get("ok", False)),
        "policy_model_config": dict(policy_model_cfg),
        "trace_enabled": bool(enabled_flag),
        "ligand_heavy_atoms": int(lig_heavy),
        "size_scale": float(size_scale),
        "formal_charge_e": float(formal_q) if np.isfinite(formal_q) else np.nan,
        "ring_count": int(ring_n),
        "rotatable_bonds": int(rot_n),
        "metal_contact_expected": bool(metal_contact_expected),
        "target_hint": str(target_txt),
        "ligand_smiles": str(smiles_txt),
        "ligand_scaffold": str(scaffold_txt),
        "k_push_scale": float(1.12 + (0.08 * size_scale)),
        "k_relax_scale": 0.82,
        "detach_scale_multiplier": float(1.10 + (0.08 * size_scale)),
        "max_hold_extension_samples": int(max_hold_ext),
        "last_action_name": "keep",
        "last_confidence": 0.0,
        "applied_action_count": 0,
        **dict(pocket_row),
    }
    return out


def configure_abfe_steering_cycles(adaptive_state=None, *, n_cycles=1, n_samples=0, warmup_samples=0):
    """
    Unbind/rebind cycles for the "pull_hold_return" mode.

    The steering lambda is then driven by the state machine (``cycle_lambda``): it ramps
    0 -> 1 while pulling out along the corridor/tunnel, holds while the ligand is
    unbound, ramps 1 -> 0 on "return" with the pull force restored (so the ligand is
    brought back into the site), and a cycle counts once the ligand has stayed bound for
    ``rebound_target_samples``. After the last cycle the phase is "bound_hold" (lambda 0).
    Other modes are returned unchanged (single pull, no return).
    """
    state = dict(adaptive_state or {})
    mode = str(state.get("steering_mode", "pull_hold") or "pull_hold").strip().lower()
    if (not bool(state.get("enabled", False))) or mode != "pull_hold_return":
        state["state_driven_lambda"] = False
        state["cycles_target"] = 1
        state["cycles_completed"] = 0
        return state
    try:
        cycles = int(n_cycles)
    except Exception:
        cycles = 1
    cycles = int(np.clip(cycles, 1, 20))
    try:
        total = int(n_samples or 0)
    except Exception:
        total = 0
    try:
        warm = int(warmup_samples or 0)
    except Exception:
        warm = 0
    total = max(0, total)
    warm = int(np.clip(warm, 0, max(0, total - 1)))
    per_cycle = float(max(1, total - warm)) / float(cycles) if total > 0 else 60.0
    # ~22% of each cycle to pull out and ~22% to bring back; the rest is spent unbound
    # (far hold / detached settle) and bound (rebound check)
    ramp_samples = int(max(3, round(0.22 * per_cycle)))
    state.update(
        {
            "state_driven_lambda": True,
            "cycles_target": int(cycles),
            "cycles_completed": 0,
            "cycle_lambda": 0.0,
            "cycle_lambda_step": float(1.0 / float(ramp_samples)),
            "cycle_ramp_samples": int(ramp_samples),
            "warmup_samples": int(warm),
            "rebound_target_samples": int(max(4, round(0.08 * per_cycle))),
            "rebound_since_sample": -1,
            "return_timeout_samples": int(max(ramp_samples, round(0.20 * per_cycle))),
            # out of the pocket but still stuck to the protein surface (e.g. a salt bridge on the
            # rim): the far-unbound test never passes, so return after this many samples anyway
            "out_of_site_timeout_samples": int(max(4, round(0.12 * per_cycle))),
            "out_of_site_since_sample": -1,
            "return_zero_since_sample": -1,
            "return_force_boosts": 0,
        }
    )
    return state


def _abfe_advance_cycle_lambda(state, sample_index):
    """Per-sample lambda ramp of the state-driven (cycle) mode."""
    phase = str(state.get("phase", "pull") or "pull").strip().lower()
    lam = float(state.get("cycle_lambda", 0.0) or 0.0)
    step = float(state.get("cycle_lambda_step", 0.1) or 0.1)
    if phase in {"pull", "probe"} and int(sample_index) >= int(state.get("warmup_samples", 0) or 0):
        lam = min(1.0, lam + step)
    elif phase in {"far_hold", "detached_settle"}:
        lam = 1.0
    elif phase == "return":
        lam = max(0.0, lam - step)
    elif phase in {"bound_hold", "bound_settle"}:
        lam = 0.0
    state["cycle_lambda"] = float(np.clip(lam, 0.0, 1.0))
    return state


def build_abfe_adaptive_steering_update_decision(
    *,
    adaptive_state=None,
    sample_index=0,
    lam_value=0.0,
    current_pull_k_kj_nm2=800.0,
    current_direction_index=0,
    direction_count=1,
    switches_used=0,
    min_dist_a=None,
    reactive_dist_a=None,
    site_min_dist_a=None,
    site_com_dist_a=None,
    contact_count_4a=None,
    total_e_kj=None,
):
    state = dict(adaptive_state or {})
    if not bool(state.get("enabled", False)):
        return {
            "ok": True,
            "should_adapt": False,
            "reason": "disabled",
            "update_pull_k": False,
            "pull_k_kj_nm2": float(current_pull_k_kj_nm2),
            "switch_direction": False,
            "direction_index": int(current_direction_index),
            "phase": str(state.get("phase", "pull") or "pull"),
            "hold_lambda_value": np.nan,
            "far_hold_stable": False,
            "adaptive_state": dict(state),
        }

    try:
        sidx = int(sample_index or 0)
    except Exception:
        sidx = 0
    sidx = max(0, int(sidx))
    try:
        lam_f = float(lam_value)
    except Exception:
        lam_f = 0.0
    if not np.isfinite(lam_f):
        lam_f = 0.0
    if bool(state.get("state_driven_lambda", False)):
        state = _abfe_advance_cycle_lambda(state, sidx)
        if str(state.get("phase", "")).strip().lower() == "bound_hold":
            return {
                "ok": True,
                "should_adapt": False,
                "reason": "cycles_complete_bound_hold",
                "update_pull_k": False,
                "pull_k_kj_nm2": float(current_pull_k_kj_nm2),
                "update_pull_distance_scale": False,
                "pull_distance_scale": float(state.get("current_pull_distance_scale", 1.0) or 1.0),
                "switch_direction": False,
                "direction_index": int(current_direction_index or 0),
                "phase": "bound_hold",
                "hold_lambda_value": 0.0,
                "far_hold_stable": False,
                "notes": [],
                "adaptive_state": dict(state),
            }
    high_lambda_threshold = float(state.get("high_lambda_threshold", 0.70))
    state["high_lambda_seen"] = bool(state.get("high_lambda_seen", False) or (lam_f >= high_lambda_threshold))

    try:
        current_k = float(current_pull_k_kj_nm2)
    except Exception:
        current_k = float(state.get("current_pull_k_kj_nm2", state.get("initial_pull_k_kj_nm2", 800.0)))
    if (not np.isfinite(current_k)) or (current_k <= 0.0):
        current_k = float(state.get("initial_pull_k_kj_nm2", 800.0))

    try:
        direction_i = int(current_direction_index or 0)
    except Exception:
        direction_i = 0
    direction_i = max(0, int(direction_i))
    try:
        n_dirs = int(direction_count or 0)
    except Exception:
        n_dirs = 0
    n_dirs = max(1, int(n_dirs))

    try:
        last_update = int(state.get("last_update_sample", -999999))
    except Exception:
        last_update = -999999
    try:
        update_interval = int(state.get("update_interval_samples", 6))
    except Exception:
        update_interval = 6
    update_interval = max(1, int(update_interval))

    out = {
        "ok": True,
        "should_adapt": False,
        "reason": "waiting",
        "update_pull_k": False,
        "pull_k_kj_nm2": float(current_k),
        "update_pull_distance_scale": False,
        "pull_distance_scale": np.nan,
        "switch_direction": False,
        "direction_index": int(direction_i),
        "phase": str(state.get("phase", "pull") or "pull"),
        "hold_lambda_value": np.nan,
        "far_hold_stable": False,
        "progress_site_com_a": np.nan,
        "progress_reactive_a": np.nan,
        "latest_min_dist_a": np.nan,
        "latest_site_com_dist_a": np.nan,
        "latest_reactive_dist_a": np.nan,
        "latest_contact_count_4a": np.nan,
        "latest_total_e_kj": np.nan,
        "notes": [],
        "adaptive_state": dict(state),
    }

    phase = str(state.get("phase", "pull") or "pull").strip().lower()
    if phase not in {"bound_settle", "probe", "pull", "far_hold", "detached_settle", "return", "bound_hold"}:
        phase = "pull"

    try:
        current_distance_scale = float(state.get("current_pull_distance_scale", state.get("target_pull_distance_scale", 1.0)))
    except Exception:
        current_distance_scale = 1.0
    if (not np.isfinite(current_distance_scale)) or (current_distance_scale <= 0.0):
        current_distance_scale = 1.0
    try:
        distance_scale_min = float(state.get("pull_distance_scale_min", 1.0))
    except Exception:
        distance_scale_min = 1.0
    try:
        distance_scale_max = float(state.get("pull_distance_scale_max", max(1.0, current_distance_scale)))
    except Exception:
        distance_scale_max = max(1.0, current_distance_scale)
    if (not np.isfinite(distance_scale_min)) or (distance_scale_min <= 0.0):
        distance_scale_min = 1.0
    if (not np.isfinite(distance_scale_max)) or (distance_scale_max < distance_scale_min):
        distance_scale_max = max(distance_scale_min, current_distance_scale)
    try:
        distance_push_scale = float(state.get("pull_distance_push_scale", 1.12))
    except Exception:
        distance_push_scale = 1.12
    if (not np.isfinite(distance_push_scale)) or (distance_push_scale <= 1.0):
        distance_push_scale = 1.12
    try:
        distance_relax_scale = float(state.get("pull_distance_relax_scale", 0.92))
    except Exception:
        distance_relax_scale = 0.92
    if (not np.isfinite(distance_relax_scale)) or (distance_relax_scale <= 0.0) or (distance_relax_scale >= 1.0):
        distance_relax_scale = 0.92
    out["pull_distance_scale"] = float(current_distance_scale)

    if phase == "bound_settle":
        next_state = dict(state)
        next_state["last_update_sample"] = int(sidx)
        next_state["phase"] = "bound_settle"
        next_state["current_pull_k_kj_nm2"] = float(current_k)
        next_state["current_pull_distance_scale"] = float(max(1.0, current_distance_scale))
        try:
            settle_count = int(state.get("bound_settle_count", 0) or 0)
        except Exception:
            settle_count = 0
        settle_count = int(settle_count + 1)
        next_state["bound_settle_count"] = int(settle_count)
        try:
            settle_target = int(state.get("bound_settle_target_samples", 0) or 0)
        except Exception:
            settle_target = 0
        settle_target = max(0, int(settle_target))
        out["phase"] = "bound_settle"
        out["hold_lambda_value"] = float(state.get("bound_settle_lambda_value", 0.1) or 0.1)
        if settle_count < settle_target:
            out["reason"] = "bound_settle"
            out["adaptive_state"] = dict(next_state)
            return out
        next_phase = "probe" if bool(state.get("probe_enabled", False)) else "pull"
        next_state["phase"] = str(next_phase)
        new_k = float(
            state.get(
                "probe_pull_k_kj_nm2" if next_phase == "probe" else "post_probe_pull_k_kj_nm2",
                current_k,
            )
            or current_k
        )
        new_distance_scale = float(
            state.get(
                "probe_pull_distance_scale" if next_phase == "probe" else "post_probe_pull_distance_scale",
                state.get("target_pull_distance_scale", current_distance_scale),
            )
            or current_distance_scale
        )
        new_distance_scale = float(np.clip(new_distance_scale, distance_scale_min, distance_scale_max))
        out["phase"] = str(next_phase)
        out["update_pull_k"] = bool(abs(new_k - current_k) > 1.0e-6)
        out["update_pull_distance_scale"] = bool(abs(new_distance_scale - current_distance_scale) > 1.0e-6)
        out["pull_k_kj_nm2"] = float(new_k)
        out["pull_distance_scale"] = float(new_distance_scale)
        out["should_adapt"] = bool(out["update_pull_k"] or out["update_pull_distance_scale"])
        out["reason"] = "bound_settle_complete_enter_probe" if next_phase == "probe" else "bound_settle_complete_enter_pull"
        if out["should_adapt"]:
            next_state["adaptation_count"] = int(max(0, int(state.get("adaptation_count", 0))) + 1)
        next_state["current_pull_k_kj_nm2"] = float(new_k)
        next_state["current_pull_distance_scale"] = float(new_distance_scale)
        next_state["bound_settle_count"] = int(settle_count)
        out["adaptive_state"] = dict(next_state)
        return out

    if phase == "pull" and lam_f < float(state.get("pull_phase_lambda_threshold", 0.25)):
        out["reason"] = "pull_phase_not_active"
        return out
    if (sidx - last_update) < update_interval:
        out["reason"] = "update_interval_not_reached"
        return out

    site_com_arr = np.asarray(site_com_dist_a if site_com_dist_a is not None else [], dtype=float)
    reactive_arr = np.asarray(reactive_dist_a if reactive_dist_a is not None else [], dtype=float)
    min_dist_arr = np.asarray(min_dist_a if min_dist_a is not None else [], dtype=float)
    _ = np.asarray(site_min_dist_a if site_min_dist_a is not None else [], dtype=float)
    contact_arr = np.asarray(contact_count_4a if contact_count_4a is not None else [], dtype=float)
    total_e_arr = np.asarray(total_e_kj if total_e_kj is not None else [], dtype=float)

    try:
        stall_window = int(state.get("stall_window_samples", 12))
    except Exception:
        stall_window = 12
    stall_window = max(3, int(stall_window))
    probe_window = int(max(2, int(state.get("probe_window_samples", update_interval) or update_interval)))
    history_window = probe_window if phase == "probe" else stall_window
    if site_com_arr.size < history_window or reactive_arr.size < history_window:
        out["reason"] = "insufficient_history"
        return out

    recent_site = np.asarray(site_com_arr[-history_window:], dtype=float)
    recent_reactive = np.asarray(reactive_arr[-history_window:], dtype=float)
    recent_min = np.asarray(min_dist_arr[-min(history_window, max(1, min_dist_arr.size)):], dtype=float)
    recent_contacts = np.asarray(contact_arr[-min(history_window, max(1, contact_arr.size)):], dtype=float)
    recent_e = np.asarray(total_e_arr[-min(history_window, max(1, total_e_arr.size)):], dtype=float)

    progress_site = np.nan
    progress_reactive = np.nan
    if recent_site.size >= 2:
        progress_site = float(recent_site[-1] - recent_site[0])
    if recent_reactive.size >= 2:
        progress_reactive = float(recent_reactive[-1] - recent_reactive[0])
    latest_min = float(recent_min[-1]) if recent_min.size else np.nan
    latest_site = float(recent_site[-1]) if recent_site.size else np.nan
    latest_reactive = float(recent_reactive[-1]) if recent_reactive.size else np.nan
    latest_contacts = float(recent_contacts[-1]) if recent_contacts.size else np.nan
    latest_e = float(recent_e[-1]) if recent_e.size else np.nan
    median_e = float(np.nanmedian(recent_e)) if recent_e.size else np.nan

    out["progress_site_com_a"] = float(progress_site) if np.isfinite(progress_site) else np.nan
    out["progress_reactive_a"] = float(progress_reactive) if np.isfinite(progress_reactive) else np.nan
    out["latest_min_dist_a"] = float(latest_min) if np.isfinite(latest_min) else np.nan
    out["latest_site_com_dist_a"] = float(latest_site) if np.isfinite(latest_site) else np.nan
    out["latest_reactive_dist_a"] = float(latest_reactive) if np.isfinite(latest_reactive) else np.nan
    out["latest_contact_count_4a"] = float(latest_contacts) if np.isfinite(latest_contacts) else np.nan
    out["latest_total_e_kj"] = float(latest_e) if np.isfinite(latest_e) else np.nan

    progress_site_thresh = float(state.get("progress_site_com_threshold_a", 0.35))
    progress_reactive_thresh = float(state.get("progress_reactive_threshold_a", 0.12))
    target_escape = float(state.get("target_escape_site_com_a", 8.0))
    soft_clash = bool(np.isfinite(latest_min) and (latest_min < float(state.get("clash_soft_min_dist_a", 1.90))))
    hard_clash = bool(np.isfinite(latest_min) and (latest_min < float(state.get("clash_hard_min_dist_a", 1.60))))
    energy_spike = bool(
        np.isfinite(latest_e)
        and np.isfinite(median_e)
        and ((latest_e - median_e) > float(state.get("energy_spike_threshold_kj", 120.0)))
    )
    stalled = bool(
        np.isfinite(progress_site)
        and np.isfinite(progress_reactive)
        and (progress_site < progress_site_thresh)
        and (progress_reactive < progress_reactive_thresh)
        and np.isfinite(latest_site)
        and (latest_site < target_escape)
    )
    far_site = float(state.get("far_unbound_site_com_a", 10.0))
    far_min = float(state.get("far_unbound_min_dist_a", 3.5))
    far_max_contacts = int(max(0, int(state.get("far_unbound_max_contact_count_4a", 4))))
    far_ready = bool(
        np.isfinite(latest_site)
        and np.isfinite(latest_min)
        and (latest_site >= far_site)
        and (latest_min >= far_min)
        and (
            (not np.isfinite(latest_contacts))
            or (latest_contacts <= float(far_max_contacts))
        )
    )
    far_hold_release_site = float(state.get("far_hold_release_site_com_a", max(8.5, far_site - 0.75)))
    far_hold_release_min = float(state.get("far_hold_release_min_dist_a", max(2.8, far_min - 0.25)))
    far_hold_release_contacts = int(max(0, int(state.get("far_hold_release_max_contact_count_4a", far_max_contacts + 1))))
    far_hold_sustain = bool(
        np.isfinite(latest_site)
        and np.isfinite(latest_min)
        and (latest_site >= far_hold_release_site)
        and (latest_min >= far_hold_release_min)
        and (
            (not np.isfinite(latest_contacts))
            or (latest_contacts <= float(far_hold_release_contacts))
        )
    )
    surface_grazing = bool(
        np.isfinite(latest_site)
        and np.isfinite(latest_min)
        and (latest_site >= max(0.75 * target_escape, 0.80 * far_site))
        and (latest_min < far_hold_release_min)
        and (
            (not np.isfinite(latest_contacts))
            or (latest_contacts > float(far_hold_release_contacts))
        )
    )

    next_state = dict(state)
    next_state["last_update_sample"] = int(sidx)
    next_state["current_pull_k_kj_nm2"] = float(current_k)
    next_state["last_direction_index"] = int(direction_i)
    next_state["switches_used"] = int(max(0, int(switches_used or state.get("switches_used", 0))))
    next_state["phase"] = str(phase)
    try:
        prev_stall_count = int(state.get("stall_count", 0) or 0)
    except Exception:
        prev_stall_count = 0
    if stalled and (not far_ready):
        stall_count = int(prev_stall_count + 1)
    elif far_ready or (
        (np.isfinite(progress_site) and progress_site >= (1.25 * progress_site_thresh))
        or (np.isfinite(progress_reactive) and progress_reactive >= (1.25 * progress_reactive_thresh))
    ):
        stall_count = 0
    else:
        stall_count = int(max(0, prev_stall_count - 1))
    next_state["stall_count"] = int(stall_count)

    max_switches = int(max(0, int(state.get("max_direction_switches", 0))))
    can_switch = bool((n_dirs > 1) and (next_state["switches_used"] < max_switches) and ((direction_i + 1) < n_dirs))
    k_min = float(state.get("k_min_kj_nm2", 200.0))
    k_max = float(state.get("k_max_kj_nm2", max(k_min, current_k)))
    direction_clearances = list(state.get("direction_clearances_nm", []) or [])
    direction_collision_penalties = list(state.get("direction_collision_penalties", []) or [])
    current_clearance_nm = np.nan
    current_collision_penalty = np.nan
    if 0 <= direction_i < len(direction_clearances):
        try:
            current_clearance_nm = float(direction_clearances[direction_i])
        except Exception:
            current_clearance_nm = np.nan
    if 0 <= direction_i < len(direction_collision_penalties):
        try:
            current_collision_penalty = float(direction_collision_penalties[direction_i])
        except Exception:
            current_collision_penalty = np.nan
    new_k = float(current_k)
    new_distance_scale = float(current_distance_scale)
    new_dir = int(direction_i)
    reason = "no_change"
    far_hold_target = int(max(1, int(state.get("far_hold_target_samples", 8))))
    far_hold_count = int(max(0, int(state.get("far_hold_count", 0))))
    far_hold_k_fraction = float(state.get("far_hold_k_fraction", 0.55))
    far_hold_k_floor = float(state.get("far_hold_k_floor_kj_nm2", max(200.0, current_k * 0.25)))
    if (not np.isfinite(far_hold_k_floor)) or (far_hold_k_floor <= 0.0):
        far_hold_k_floor = float(max(200.0, current_k * 0.25))
    hold_target_k = float(state.get("far_hold_target_k_kj_nm2", np.nan))
    if (not np.isfinite(hold_target_k)) or (hold_target_k <= 0.0):
        hold_target_k = float(max(far_hold_k_floor, current_k * far_hold_k_fraction))
    detached_target = int(max(1, int(state.get("detached_settle_target_samples", 4) or 4)))
    detached_count = int(max(0, int(state.get("detached_settle_count", 0) or 0)))
    detached_lambda_value = float(state.get("detached_settle_lambda_value", 0.05) or 0.05)
    detached_k_floor = float(state.get("detached_settle_k_floor_kj_nm2", max(40.0, far_hold_k_floor * 0.50)))
    if (not np.isfinite(detached_k_floor)) or (detached_k_floor <= 0.0):
        detached_k_floor = float(max(40.0, far_hold_k_floor * 0.50))

    if phase == "far_hold":
        next_state["pull_surface_detach_active"] = False
        out["hold_lambda_value"] = 1.0
        if far_hold_sustain:
            far_hold_count = int(far_hold_count + 1)
            next_state["far_hold_count"] = int(far_hold_count)
            next_state["far_hold_target_k_kj_nm2"] = float(hold_target_k)
            next_state["recent_far_hold_break"] = False
            out["far_hold_stable"] = bool(far_hold_count >= far_hold_target and far_ready)
            if current_k > (hold_target_k + 1.0e-6):
                new_k = float(max(hold_target_k, far_hold_k_floor))
                out["update_pull_k"] = bool(new_k < current_k - 1.0e-6)
            relax_after = int(max(1, int(state.get("far_hold_relax_after_samples", 3) or 3)))
            relaxed_fraction = float(state.get("far_hold_relaxed_k_fraction", 0.28))
            if (not np.isfinite(relaxed_fraction)) or (relaxed_fraction <= 0.0):
                relaxed_fraction = 0.28
            relaxed_target_k = float(
                max(
                    far_hold_k_floor,
                    float(state.get("initial_pull_k_kj_nm2", current_k)) * relaxed_fraction,
                )
            )
            relax_ready = bool(
                far_hold_count >= relax_after
                and far_ready
                and np.isfinite(latest_site)
                and np.isfinite(latest_min)
                and latest_site >= (far_site + 1.0)
                and latest_min >= (far_min + 0.35)
                and ((not np.isfinite(latest_contacts)) or (latest_contacts <= max(0.0, float(far_max_contacts) - 1.0)))
            )
            if relax_ready and current_k > (relaxed_target_k + 1.0e-6):
                new_k = float(max(relaxed_target_k, far_hold_k_floor))
                out["update_pull_k"] = bool(new_k < current_k - 1.0e-6 or new_k < out["pull_k_kj_nm2"] - 1.0e-6)
                hold_target_k = float(new_k)
                next_state["far_hold_target_k_kj_nm2"] = float(hold_target_k)
                out["notes"].append("far_hold_bias_relax")
            if far_hold_count >= far_hold_target:
                if bool(state.get("detached_settle_enabled", False)):
                    next_state["phase"] = "detached_settle"
                    next_state["detached_settle_count"] = 1
                    new_k = float(detached_k_floor)
                    out["update_pull_k"] = bool(new_k < current_k - 1.0e-6 or bool(out["update_pull_k"]))
                    out["hold_lambda_value"] = float(detached_lambda_value)
                    reason = "enter_detached_settle"
                elif str(state.get("steering_mode", "pull_hold")) == "pull_hold_return":
                    next_state["phase"] = "return"
                    new_k = float(state.get("initial_pull_k_kj_nm2", current_k) or current_k)
                    out["update_pull_k"] = bool(abs(new_k - current_k) > 1.0e-6)
                    next_state["return_zero_since_sample"] = -1
                    next_state["rebound_since_sample"] = -1
                    reason = "far_hold_complete_enter_return"
                else:
                    next_state["phase"] = "far_hold"
                    reason = "far_hold_complete"
            else:
                reason = "far_hold_stable"
        else:
            next_state["phase"] = "pull"
            next_state["far_hold_count"] = 0
            next_state["far_hold_target_k_kj_nm2"] = np.nan
            next_state["detached_settle_count"] = 0
            next_state["recent_far_hold_break"] = True
            next_state["far_hold_break_count"] = int(max(0, int(next_state.get("far_hold_break_count", 0) or 0)) + 1)
            reason = "far_hold_broken_resume_pull"
        out["pull_k_kj_nm2"] = float(new_k)
        out["pull_distance_scale"] = float(new_distance_scale)
        out["direction_index"] = int(new_dir)
        out["phase"] = str(next_state.get("phase", phase))
        out["should_adapt"] = bool(out["update_pull_k"])
        if out["should_adapt"]:
            next_state["adaptation_count"] = int(max(0, int(state.get("adaptation_count", 0))) + 1)
        next_state["current_pull_k_kj_nm2"] = float(new_k)
        next_state["current_pull_distance_scale"] = float(new_distance_scale)
        next_state["last_direction_index"] = int(new_dir)
        out["reason"] = str(reason)
        out["adaptive_state"] = dict(next_state)
        return out

    if phase == "detached_settle":
        next_state["pull_surface_detach_active"] = False
        out["phase"] = "detached_settle"
        out["hold_lambda_value"] = float(detached_lambda_value)
        if far_ready:
            detached_count = int(detached_count + 1)
            next_state["detached_settle_count"] = int(detached_count)
            next_state["recent_far_hold_break"] = False
            target_detached_k = float(detached_k_floor)
            if current_k > (target_detached_k + 1.0e-6):
                new_k = float(target_detached_k)
                out["update_pull_k"] = bool(True)
            out["far_hold_stable"] = bool(detached_count >= detached_target and far_ready)
            if detached_count >= detached_target and str(state.get("steering_mode", "pull_hold")) == "pull_hold_return":
                # bring the ligand back with the full pull force (the detached settle force is
                # ~1% of it, too weak to move the ligand into the site)
                next_state["phase"] = "return"
                new_k = float(state.get("initial_pull_k_kj_nm2", current_k) or current_k)
                out["update_pull_k"] = bool(abs(new_k - current_k) > 1.0e-6)
                next_state["return_zero_since_sample"] = -1
                next_state["rebound_since_sample"] = -1
                reason = "detached_settle_complete_enter_return"
            elif detached_count >= detached_target:
                next_state["phase"] = "detached_settle"
                reason = "detached_settle_complete"
            else:
                reason = "detached_settle"
        else:
            next_state["phase"] = "pull"
            next_state["far_hold_count"] = 0
            next_state["detached_settle_count"] = 0
            next_state["far_hold_target_k_kj_nm2"] = np.nan
            next_state["recent_far_hold_break"] = True
            next_state["far_hold_break_count"] = int(max(0, int(next_state.get("far_hold_break_count", 0) or 0)) + 1)
            # the settle force is ~1% of the pull force; resuming "pull" with it leaves the
            # ligand on the surface for the rest of the run
            restored_k = float(state.get("initial_pull_k_kj_nm2", current_k) or current_k)
            if restored_k > current_k + 1.0e-6:
                new_k = float(restored_k)
                out["update_pull_k"] = True
            reason = "detached_settle_broken_resume_pull"
        out["pull_k_kj_nm2"] = float(new_k)
        out["pull_distance_scale"] = float(new_distance_scale)
        out["direction_index"] = int(new_dir)
        out["should_adapt"] = bool(out["update_pull_k"])
        if out["should_adapt"]:
            next_state["adaptation_count"] = int(max(0, int(state.get("adaptation_count", 0))) + 1)
        next_state["current_pull_k_kj_nm2"] = float(new_k)
        next_state["current_pull_distance_scale"] = float(new_distance_scale)
        next_state["last_direction_index"] = int(new_dir)
        out["reason"] = str(reason)
        out["adaptive_state"] = dict(next_state)
        return out

    if phase == "return":
        next_state["pull_surface_detach_active"] = False
        k_return = float(state.get("initial_pull_k_kj_nm2", current_k) or current_k)
        if current_k < k_return - 1.0e-6 and int(state.get("return_force_boosts", 0) or 0) == 0:
            new_k = float(k_return)
            out["update_pull_k"] = True
        lam_now = float(state.get("cycle_lambda", lam_f) or 0.0)
        bound_contacts_needed = 0.5 * float(state.get("bound_min_contact_count_4a", 4) or 4)
        bound_now = bool(
            np.isfinite(latest_site)
            and np.isfinite(latest_min)
            and latest_site <= float(state.get("bound_site_com_threshold_a", 4.0))
            and latest_min <= float(state.get("bound_max_min_dist_a", 3.5))
            and ((not np.isfinite(latest_contacts)) or latest_contacts >= bound_contacts_needed)
        )
        reason = "return_ramp"
        if lam_now <= 1.0e-6:
            zero_since = int(state.get("return_zero_since_sample", -1))
            if zero_since < 0:
                zero_since = int(sidx)
            next_state["return_zero_since_sample"] = int(zero_since)
            if bound_now:
                since = int(state.get("rebound_since_sample", -1))
                if since < 0:
                    since = int(sidx)
                next_state["rebound_since_sample"] = int(since)
                if (sidx - since) >= int(state.get("rebound_target_samples", 4) or 4):
                    done = int(state.get("cycles_completed", 0) or 0) + 1
                    next_state["cycles_completed"] = int(done)
                    next_state["rebound_since_sample"] = -1
                    next_state["return_zero_since_sample"] = -1
                    next_state["return_force_boosts"] = 0
                    if done < int(state.get("cycles_target", 1) or 1):
                        # next unbind cycle from the bound pose
                        next_state["phase"] = "pull"
                        next_state["far_hold_count"] = 0
                        next_state["detached_settle_count"] = 0
                        next_state["far_hold_target_k_kj_nm2"] = np.nan
                        next_state["stall_count"] = 0
                        next_state["high_lambda_seen"] = False
                        new_distance_scale = float(state.get("target_pull_distance_scale", new_distance_scale) or new_distance_scale)
                        out["update_pull_distance_scale"] = bool(abs(new_distance_scale - current_distance_scale) > 1.0e-6)
                        new_k = float(k_return)
                        out["update_pull_k"] = bool(abs(new_k - current_k) > 1.0e-6)
                        reason = "rebound_start_next_cycle"
                    else:
                        next_state["phase"] = "bound_hold"
                        reason = "rebound_cycles_complete"
                else:
                    reason = "rebound_settling"
            else:
                next_state["rebound_since_sample"] = -1
                timeout = int(state.get("return_timeout_samples", 30) or 30)
                boosts = int(state.get("return_force_boosts", 0) or 0)
                if (sidx - zero_since) >= timeout * (boosts + 1) and current_k < (k_max * 0.985):
                    # the ligand does not find its way in: push harder toward the bound pose
                    new_k = float(min(k_max, max(current_k + 200.0, current_k * 1.5)))
                    out["update_pull_k"] = True
                    next_state["return_force_boosts"] = int(boosts + 1)
                    reason = "return_stalled_increase_force"
                else:
                    reason = "return_waiting_for_rebind"
        next_state["phase"] = str(next_state.get("phase", "return"))
        out["phase"] = str(next_state["phase"])
        out["reason"] = str(reason)
        out["hold_lambda_value"] = float(lam_now)
        out["pull_k_kj_nm2"] = float(new_k)
        out["pull_distance_scale"] = float(new_distance_scale)
        out["direction_index"] = int(new_dir)
        out["should_adapt"] = bool(out["update_pull_k"] or out.get("update_pull_distance_scale", False))
        if out["should_adapt"]:
            next_state["adaptation_count"] = int(max(0, int(state.get("adaptation_count", 0))) + 1)
        next_state["current_pull_k_kj_nm2"] = float(new_k)
        next_state["current_pull_distance_scale"] = float(new_distance_scale)
        out["adaptive_state"] = dict(next_state)
        return out

    if phase == "probe":
        probe_scores = list(state.get("probe_scores", []) or [])
        probe_counts = list(state.get("probe_counts", []) or [])
        if len(probe_scores) < int(n_dirs):
            probe_scores.extend([np.nan] * (int(n_dirs) - len(probe_scores)))
        if len(probe_counts) < int(n_dirs):
            probe_counts.extend([0] * (int(n_dirs) - len(probe_counts)))
        probe_payload = build_abfe_probe_direction_score_payload(
            progress_site_com_a=float(progress_site) if np.isfinite(progress_site) else np.nan,
            progress_reactive_a=float(progress_reactive) if np.isfinite(progress_reactive) else np.nan,
            latest_min_dist_a=float(latest_min) if np.isfinite(latest_min) else np.nan,
            latest_contact_count_4a=float(latest_contacts) if np.isfinite(latest_contacts) else np.nan,
            latest_total_e_kj=float(latest_e) if np.isfinite(latest_e) else np.nan,
            median_total_e_kj=float(median_e) if np.isfinite(median_e) else np.nan,
            clearance_nm=float(current_clearance_nm) if np.isfinite(current_clearance_nm) else np.nan,
            collision_penalty=float(current_collision_penalty) if np.isfinite(current_collision_penalty) else np.nan,
            soft_clash=bool(soft_clash),
            hard_clash=bool(hard_clash),
            energy_spike=bool(energy_spike),
        )
        probe_score = float(probe_payload.get("score", np.nan))
        probe_scores[int(direction_i)] = float(probe_score) if np.isfinite(probe_score) else probe_scores[int(direction_i)]
        probe_counts[int(direction_i)] = int(max(0, int(probe_counts[int(direction_i)])) + 1)
        next_state["probe_scores"] = list(probe_scores)
        next_state["probe_counts"] = [int(v) for v in list(probe_counts)]
        finite_probe_scores = [(ii, float(vv)) for ii, vv in enumerate(probe_scores) if np.isfinite(vv)]
        if finite_probe_scores:
            best_dir_probe, best_score_probe = max(finite_probe_scores, key=lambda kv: float(kv[1]))
        else:
            best_dir_probe, best_score_probe = int(direction_i), np.nan
        next_state["probe_best_direction_index"] = int(best_dir_probe)
        next_state["probe_best_score"] = float(best_score_probe) if np.isfinite(best_score_probe) else np.nan
        probe_limit = int(max(1, min(int(state.get("probe_direction_limit", n_dirs) or n_dirs), int(n_dirs))))
        early_accept_score = float(state.get("probe_early_accept_score", 1.35))
        early_accept = bool(
            far_ready
            or (
                np.isfinite(probe_score)
                and (probe_score >= early_accept_score)
                and (not hard_clash)
            )
        )
        next_dir_probe = int(direction_i + 1)
        if (next_dir_probe < probe_limit) and (not early_accept):
            new_dir = int(next_dir_probe)
            new_k = float(np.clip(float(state.get("probe_pull_k_kj_nm2", current_k) or current_k), k_min, k_max))
            new_distance_scale = float(
                np.clip(
                    float(state.get("probe_pull_distance_scale", current_distance_scale) or current_distance_scale),
                    distance_scale_min,
                    distance_scale_max,
                )
            )
            out["switch_direction"] = bool(new_dir != direction_i)
            out["update_pull_k"] = bool(abs(new_k - current_k) > 1.0e-6)
            out["update_pull_distance_scale"] = bool(abs(new_distance_scale - current_distance_scale) > 1.0e-6)
            reason = "probe_switch_next_corridor"
            next_state["phase"] = "probe"
        else:
            selected_dir = int(best_dir_probe)
            new_dir = int(selected_dir)
            target_probe_k = float(state.get("post_probe_pull_k_kj_nm2", current_k) or current_k)
            if np.isfinite(best_score_probe):
                target_probe_k = float(np.clip(target_probe_k * (0.95 + (0.05 * np.clip(best_score_probe, 0.0, 4.0))), k_min, k_max))
            new_k = float(np.clip(target_probe_k, k_min, k_max))
            new_distance_scale = float(
                np.clip(
                    float(state.get("post_probe_pull_distance_scale", state.get("target_pull_distance_scale", current_distance_scale)) or current_distance_scale),
                    distance_scale_min,
                    distance_scale_max,
                )
            )
            out["switch_direction"] = bool(new_dir != direction_i)
            out["update_pull_k"] = bool(abs(new_k - current_k) > 1.0e-6)
            out["update_pull_distance_scale"] = bool(abs(new_distance_scale - current_distance_scale) > 1.0e-6)
            next_state["phase"] = "far_hold" if far_ready else "pull"
            if far_ready:
                next_state["far_hold_count"] = 1
                next_state["detached_settle_count"] = 0
                hold_target_k = float(max(k_min, new_k * far_hold_k_fraction))
                next_state["far_hold_target_k_kj_nm2"] = float(hold_target_k)
                new_k = float(min(new_k, hold_target_k))
                out["update_pull_k"] = bool(abs(new_k - current_k) > 1.0e-6)
                out["hold_lambda_value"] = 1.0
                reason = "probe_complete_enter_far_hold"
            else:
                reason = "probe_complete_select_corridor"
        out["pull_k_kj_nm2"] = float(new_k)
        out["pull_distance_scale"] = float(new_distance_scale)
        out["direction_index"] = int(new_dir)
        out["phase"] = str(next_state.get("phase", phase))
        out["should_adapt"] = bool(out["update_pull_k"] or out["switch_direction"])
        out["notes"].append("probe_phase")
        if np.isfinite(probe_score):
            out["notes"].append(f"probe_score={probe_score:.3f}")
        if out["should_adapt"]:
            next_state["adaptation_count"] = int(max(0, int(state.get("adaptation_count", 0))) + 1)
        next_state["current_pull_k_kj_nm2"] = float(new_k)
        next_state["current_pull_distance_scale"] = float(new_distance_scale)
        next_state["last_direction_index"] = int(new_dir)
        out["reason"] = str(reason)
        out["adaptive_state"] = dict(next_state)
        return out

    if (
        phase == "pull"
        and bool(state.get("state_driven_lambda", False))
        and str(state.get("steering_mode", "pull_hold")) == "pull_hold_return"
    ):
        out_of_site = bool(np.isfinite(latest_site) and latest_site >= far_site)
        if out_of_site and not far_ready:
            since = int(state.get("out_of_site_since_sample", -1) if state.get("out_of_site_since_sample") is not None else -1)
            if since < 0:
                since = int(sample_index)
            next_state["out_of_site_since_sample"] = int(since)
            timeout = int(state.get("out_of_site_timeout_samples", 0) or 0)
            if timeout > 0 and (int(sample_index) - since + 1) >= timeout:
                next_state["phase"] = "return"
                next_state["out_of_site_since_sample"] = -1
                next_state["return_zero_since_sample"] = -1
                next_state["rebound_since_sample"] = -1
                next_state["pull_surface_detach_active"] = False
                new_k = float(state.get("initial_pull_k_kj_nm2", current_k) or current_k)
                out["update_pull_k"] = bool(abs(new_k - current_k) > 1.0e-6)
                out["pull_k_kj_nm2"] = float(new_k)
                out["pull_distance_scale"] = float(new_distance_scale)
                out["direction_index"] = int(new_dir)
                out["phase"] = "return"
                out["should_adapt"] = bool(out["update_pull_k"])
                out["notes"].append("out_of_site_surface_bound")
                next_state["current_pull_k_kj_nm2"] = float(new_k)
                next_state["current_pull_distance_scale"] = float(new_distance_scale)
                next_state["last_direction_index"] = int(new_dir)
                out["reason"] = "out_of_site_timeout_enter_return"
                out["adaptive_state"] = dict(next_state)
                return out
        else:
            next_state["out_of_site_since_sample"] = -1

    if far_ready:
        next_state["phase"] = "far_hold"
        next_state["far_hold_count"] = 1
        next_state["detached_settle_count"] = 0
        next_state["pull_surface_detach_active"] = False
        out["phase"] = "far_hold"
        out["hold_lambda_value"] = 1.0
        hold_target_k = float(max(far_hold_k_floor, current_k * far_hold_k_fraction))
        next_state["far_hold_target_k_kj_nm2"] = float(hold_target_k)
        new_k = float(min(current_k, hold_target_k))
        out["update_pull_k"] = bool(new_k < current_k - 1.0e-6)
        reason = "enter_far_hold"
        out["pull_k_kj_nm2"] = float(new_k)
        out["pull_distance_scale"] = float(new_distance_scale)
        out["direction_index"] = int(new_dir)
        out["should_adapt"] = bool(out["update_pull_k"])
        if out["should_adapt"]:
            next_state["adaptation_count"] = int(max(0, int(state.get("adaptation_count", 0))) + 1)
        next_state["current_pull_k_kj_nm2"] = float(new_k)
        next_state["current_pull_distance_scale"] = float(new_distance_scale)
        next_state["last_direction_index"] = int(new_dir)
        out["reason"] = str(reason)
        out["adaptive_state"] = dict(next_state)
        return out

    pull_surface_detach_enabled = bool(state.get("pull_surface_detach_enabled", False))
    try:
        pull_surface_detach_activation_stalls = int(state.get("pull_surface_detach_activation_stalls", 2) or 2)
    except Exception:
        pull_surface_detach_activation_stalls = 2
    if pull_surface_detach_enabled and phase == "pull" and (
        stall_count >= max(1, pull_surface_detach_activation_stalls)
        or surface_grazing
    ):
        next_state["pull_surface_detach_active"] = True
    elif far_ready:
        next_state["pull_surface_detach_active"] = False

    bound_contact_target = float(max(4, int(state.get("bound_min_contact_count_4a", 4) or 4)))
    heavy_contact_trap = bool(
        (np.isfinite(latest_contacts) and latest_contacts >= max(2.0, 0.80 * bound_contact_target))
        or (np.isfinite(latest_min) and latest_min <= float(state.get("bound_max_min_dist_a", 3.5)) + 0.30)
    )
    bulky_or_sticky_escape = bool(
        pull_surface_detach_enabled
        or float(state.get("size_scale", 0.0) or 0.0) >= 0.45
        or float(state.get("bulk_escape_scale", 0.0) or 0.0) >= 0.50
        or float(state.get("sticky_escape_scale", 0.0) or 0.0) >= 0.55
    )
    should_push_distance = bool(
        (phase == "pull")
        and bulky_or_sticky_escape
        and (current_distance_scale < (distance_scale_max - 1.0e-6))
        and (
            heavy_contact_trap
            or stall_count >= max(1, pull_surface_detach_activation_stalls)
            or (np.isfinite(progress_site) and progress_site < (0.75 * progress_site_thresh))
        )
    )
    if should_push_distance:
        candidate_distance_scale = float(
            np.clip(
                max(current_distance_scale + 0.08, current_distance_scale * distance_push_scale),
                distance_scale_min,
                distance_scale_max,
            )
        )
    else:
        candidate_distance_scale = float(current_distance_scale)
    surface_graze_push_distance = bool(
        (phase == "pull")
        and bulky_or_sticky_escape
        and surface_grazing
        and (current_distance_scale < (distance_scale_max - 1.0e-6))
    )

    if hard_clash or (soft_clash and energy_spike):
        new_k = float(max(k_min, current_k * 0.75))
        if current_distance_scale > (distance_scale_min + 1.0e-6):
            new_distance_scale = float(
                np.clip(current_distance_scale * distance_relax_scale, distance_scale_min, distance_scale_max)
            )
            out["update_pull_distance_scale"] = bool(new_distance_scale < current_distance_scale - 1.0e-6)
        if can_switch:
            new_dir = int(direction_i + 1)
            next_state["switches_used"] = int(next_state["switches_used"] + 1)
            reason = "hard_clash_switch_corridor"
            out["switch_direction"] = True
        else:
            reason = "hard_clash_reduce_force"
        out["update_pull_k"] = bool(new_k < current_k - 1.0e-6)
    elif stalled:
        if current_k < (k_max * 0.985):
            new_k = float(min(k_max, max(current_k + 200.0, current_k * 1.50)))
            out["update_pull_k"] = bool(new_k > current_k + 1.0e-6)
            if should_push_distance and candidate_distance_scale > (current_distance_scale + 1.0e-6):
                new_distance_scale = float(candidate_distance_scale)
                out["update_pull_distance_scale"] = True
                reason = "stall_increase_force_and_distance"
            else:
                reason = "stall_increase_force"
        elif can_switch:
            new_dir = int(direction_i + 1)
            next_state["switches_used"] = int(next_state["switches_used"] + 1)
            out["switch_direction"] = True
            if should_push_distance and candidate_distance_scale > (current_distance_scale + 1.0e-6):
                new_distance_scale = float(candidate_distance_scale)
                out["update_pull_distance_scale"] = True
                reason = "stall_switch_corridor_and_distance"
            else:
                reason = "stall_switch_corridor"
        elif should_push_distance and candidate_distance_scale > (current_distance_scale + 1.0e-6):
            new_distance_scale = float(candidate_distance_scale)
            out["update_pull_distance_scale"] = True
            reason = "stall_increase_distance"
        else:
            reason = "stall_at_force_limit"
    elif surface_graze_push_distance:
        new_distance_scale = float(candidate_distance_scale)
        out["update_pull_distance_scale"] = bool(new_distance_scale > current_distance_scale + 1.0e-6)
        if current_k < (k_max * 0.985):
            new_k = float(min(k_max, max(current_k + 150.0, current_k * 1.15)))
            out["update_pull_k"] = bool(new_k > current_k + 1.0e-6)
        reason = "surface_graze_detach_boost"
    elif soft_clash and lam_f >= high_lambda_threshold:
        new_k = float(max(k_min, current_k * 0.90))
        out["update_pull_k"] = bool(new_k < current_k - 1.0e-6)
        if current_distance_scale > (distance_scale_min + 1.0e-6):
            new_distance_scale = float(
                np.clip(current_distance_scale * distance_relax_scale, distance_scale_min, distance_scale_max)
            )
            out["update_pull_distance_scale"] = bool(new_distance_scale < current_distance_scale - 1.0e-6)
        reason = "soft_clash_reduce_force"

    out["pull_k_kj_nm2"] = float(new_k)
    out["pull_distance_scale"] = float(new_distance_scale)
    out["direction_index"] = int(new_dir)
    out["reason"] = str(reason)
    out["phase"] = str(next_state.get("phase", phase))
    out["should_adapt"] = bool(out["update_pull_k"] or out["update_pull_distance_scale"] or out["switch_direction"])
    if soft_clash:
        out["notes"].append("soft_clash")
    if hard_clash:
        out["notes"].append("hard_clash")
    if energy_spike:
        out["notes"].append("energy_spike")
    if stalled:
        out["notes"].append("stalled_progress")
    if surface_grazing:
        out["notes"].append("surface_grazing")
    if bool(next_state.get("pull_surface_detach_active", False)):
        out["notes"].append("pull_surface_detach")

    if out["should_adapt"]:
        next_state["adaptation_count"] = int(max(0, int(state.get("adaptation_count", 0))) + 1)
    next_state["current_pull_k_kj_nm2"] = float(new_k)
    next_state["current_pull_distance_scale"] = float(new_distance_scale)
    next_state["last_direction_index"] = int(new_dir)
    out["adaptive_state"] = dict(next_state)
    return out


def build_abfe_steering_policy_observation_payload(
    *,
    heuristic_decision=None,
    adaptive_state=None,
    policy_state=None,
    sample_index=0,
    lam_value=0.0,
    current_pull_k_kj_nm2=np.nan,
    current_direction_index=0,
    direction_count=1,
    switches_used=0,
):
    decision = dict(heuristic_decision or {})
    state = dict(adaptive_state or {})
    pstate = dict(policy_state or {})
    mode_txt = _sanitize_abfe_steering_policy_mode(pstate.get("policy_mode", "heuristic"))
    enabled_flag = bool(pstate.get("enabled", False) and mode_txt == "bandit")
    reason = str(decision.get("reason", "") or "").strip()
    phase = str(decision.get("phase", state.get("phase", "pull")) or "pull").strip().lower()
    if phase not in {"pull", "far_hold", "detached_settle", "return"}:
        phase = "pull"

    try:
        sidx = int(sample_index or 0)
    except Exception:
        sidx = 0
    sidx = max(0, int(sidx))

    try:
        lam_f = float(lam_value)
    except Exception:
        lam_f = 0.0
    if not np.isfinite(lam_f):
        lam_f = 0.0

    try:
        current_k = float(current_pull_k_kj_nm2)
    except Exception:
        current_k = float(state.get("current_pull_k_kj_nm2", state.get("initial_pull_k_kj_nm2", 800.0)))
    if (not np.isfinite(current_k)) or (current_k <= 0.0):
        current_k = float(state.get("initial_pull_k_kj_nm2", 800.0))

    try:
        direction_i = int(current_direction_index or 0)
    except Exception:
        direction_i = 0
    direction_i = max(0, int(direction_i))

    try:
        n_dirs = int(direction_count or 0)
    except Exception:
        n_dirs = 0
    n_dirs = max(1, int(n_dirs))

    try:
        switch_used_i = int(switches_used or 0)
    except Exception:
        switch_used_i = int(state.get("switches_used", 0) or 0)
    switch_used_i = max(0, int(switch_used_i))

    k_min = float(state.get("k_min_kj_nm2", max(200.0, current_k * 0.5)))
    k_max = float(state.get("k_max_kj_nm2", max(k_min, current_k)))
    denom = float(max(k_max - k_min, 1.0e-6))
    k_fraction = float(np.clip((current_k - k_min) / denom, 0.0, 1.0))
    max_switches = int(max(0, int(state.get("max_direction_switches", 0))))
    can_switch = bool((n_dirs > 1) and (switch_used_i < max_switches) and ((direction_i + 1) < n_dirs))

    progress_site = float(decision.get("progress_site_com_a", np.nan))
    progress_reactive = float(decision.get("progress_reactive_a", np.nan))
    latest_min = float(decision.get("latest_min_dist_a", np.nan))
    latest_site = float(decision.get("latest_site_com_dist_a", np.nan))
    latest_reactive = float(decision.get("latest_reactive_dist_a", np.nan))
    latest_contacts = float(decision.get("latest_contact_count_4a", np.nan))
    latest_e = float(decision.get("latest_total_e_kj", np.nan))
    notes = [str(v) for v in list(decision.get("notes", []) or [])]
    note_set = set(notes)

    progress_site_thresh = float(state.get("progress_site_com_threshold_a", 0.35))
    progress_reactive_thresh = float(state.get("progress_reactive_threshold_a", 0.12))
    target_escape = float(state.get("target_escape_site_com_a", 8.0))
    far_site = float(state.get("far_unbound_site_com_a", 10.0))
    far_min = float(state.get("far_unbound_min_dist_a", 3.5))
    far_max_contacts = int(max(0, int(state.get("far_unbound_max_contact_count_4a", 4))))
    release_site = float(state.get("far_hold_release_site_com_a", max(8.5, far_site - 0.75)))
    release_min = float(state.get("far_hold_release_min_dist_a", max(2.8, far_min - 0.25)))
    release_contacts = int(max(0, int(state.get("far_hold_release_max_contact_count_4a", far_max_contacts + 1))))
    hold_target = int(max(1, int(state.get("far_hold_target_samples", 8) or 8)))
    hold_count = int(max(0, int(state.get("far_hold_count", 0) or 0)))
    if phase == "detached_settle":
        hold_target = int(max(1, int(state.get("detached_settle_target_samples", hold_target) or hold_target)))
        hold_count = int(max(0, int(state.get("detached_settle_count", hold_count) or hold_count)))
    hold_progress_fraction = float(np.clip(float(hold_count) / max(float(hold_target), 1.0), 0.0, 2.0))
    hold_gap_current = int(max(0, int(state.get("unbound_hold_max_gap_samples", 3) or 0)))
    hold_gap_cap = int(max(hold_gap_current, int(state.get("unbound_hold_max_gap_cap_samples", max(hold_gap_current, 3)) or max(hold_gap_current, 3))))
    hold_gap_fraction = float(np.clip(float(hold_gap_current) / max(float(max(1, hold_gap_cap)), 1.0), 0.0, 2.0))
    recent_far_hold_break = bool(state.get("recent_far_hold_break", False))

    soft_clash = bool(("soft_clash" in note_set) or (np.isfinite(latest_min) and latest_min < float(state.get("clash_soft_min_dist_a", 1.90))))
    hard_clash = bool(("hard_clash" in note_set) or (np.isfinite(latest_min) and latest_min < float(state.get("clash_hard_min_dist_a", 1.60))))
    energy_spike = bool("energy_spike" in note_set)
    stalled = bool(
        ("stalled_progress" in note_set)
        or (
            np.isfinite(progress_site)
            and np.isfinite(progress_reactive)
            and (progress_site < progress_site_thresh)
            and (progress_reactive < progress_reactive_thresh)
            and np.isfinite(latest_site)
            and (latest_site < target_escape)
        )
    )
    far_ready = bool(
        np.isfinite(latest_site)
        and np.isfinite(latest_min)
        and (latest_site >= far_site)
        and (latest_min >= far_min)
        and ((not np.isfinite(latest_contacts)) or (latest_contacts <= float(far_max_contacts)))
    )
    far_hold_sustain = bool(
        np.isfinite(latest_site)
        and np.isfinite(latest_min)
        and (latest_site >= release_site)
        and (latest_min >= release_min)
        and ((not np.isfinite(latest_contacts)) or (latest_contacts <= float(release_contacts)))
    )
    early_reasons = {
        "disabled",
        "pull_phase_not_active",
        "update_interval_not_reached",
        "insufficient_history",
        "return_phase",
    }
    actionable = bool(enabled_flag and phase in {"pull", "far_hold", "detached_settle"} and reason not in early_reasons)

    size_scale = float(pstate.get("size_scale", state.get("size_scale", 0.0)) or 0.0)
    lig_heavy = int(max(0, int(pstate.get("ligand_heavy_atoms", state.get("ligand_heavy_atoms", 0)) or 0)))
    formal_q = float(pstate.get("formal_charge_e", np.nan))
    if not np.isfinite(formal_q):
        formal_q = np.nan
    charge_density = float(abs(formal_q) / max(1.0, float(lig_heavy))) if np.isfinite(formal_q) else 0.0
    escape_gap = float(np.clip((target_escape - latest_site) / max(target_escape, 1.0), 0.0, 2.0)) if np.isfinite(latest_site) else 1.0
    contact_pressure = float(
        np.clip(latest_contacts / max(1.0, float(max(far_max_contacts + 2, 1))), 0.0, 3.0)
    ) if np.isfinite(latest_contacts) else 0.0

    out = {
        "ok": True,
        "enabled": bool(enabled_flag),
        "policy_mode": str(mode_txt),
        "sample_index": int(sidx),
        "phase": str(phase),
        "reason": str(reason),
        "actionable": bool(actionable),
        "lam_value": float(lam_f),
        "current_pull_k_kj_nm2": float(current_k),
        "k_min_kj_nm2": float(k_min),
        "k_max_kj_nm2": float(k_max),
        "k_fraction": float(k_fraction),
        "current_direction_index": int(direction_i),
        "direction_count": int(n_dirs),
        "switches_used": int(switch_used_i),
        "max_direction_switches": int(max_switches),
        "can_switch": bool(can_switch),
        "progress_site_com_a": float(progress_site) if np.isfinite(progress_site) else np.nan,
        "progress_reactive_a": float(progress_reactive) if np.isfinite(progress_reactive) else np.nan,
        "latest_min_dist_a": float(latest_min) if np.isfinite(latest_min) else np.nan,
        "latest_site_com_dist_a": float(latest_site) if np.isfinite(latest_site) else np.nan,
        "latest_reactive_dist_a": float(latest_reactive) if np.isfinite(latest_reactive) else np.nan,
        "latest_contact_count_4a": float(latest_contacts) if np.isfinite(latest_contacts) else np.nan,
        "latest_total_e_kj": float(latest_e) if np.isfinite(latest_e) else np.nan,
        "stalled": bool(stalled),
        "soft_clash": bool(soft_clash),
        "hard_clash": bool(hard_clash),
        "energy_spike": bool(energy_spike),
        "far_ready": bool(far_ready),
        "far_hold_sustain": bool(far_hold_sustain),
        "far_hold_stable": bool(decision.get("far_hold_stable", False)),
        "size_scale": float(size_scale),
        "ligand_heavy_atoms": int(lig_heavy),
        "formal_charge_e": float(formal_q) if np.isfinite(formal_q) else np.nan,
        "charge_density": float(charge_density),
        "ring_count": int(max(0, int(pstate.get("ring_count", 0) or 0))),
        "rotatable_bonds": int(max(0, int(pstate.get("rotatable_bonds", 0) or 0))),
        "metal_contact_expected": bool(pstate.get("metal_contact_expected", False)),
        "target_escape_site_com_a": float(target_escape),
        "far_unbound_site_com_a": float(far_site),
        "far_unbound_min_dist_a": float(far_min),
        "far_unbound_max_contact_count_4a": int(far_max_contacts),
        "far_hold_target_samples": int(hold_target),
        "far_hold_count": int(hold_count),
        "hold_progress_fraction": float(hold_progress_fraction),
        "unbound_hold_max_gap_samples": int(hold_gap_current),
        "unbound_hold_max_gap_cap_samples": int(hold_gap_cap),
        "hold_gap_fraction": float(hold_gap_fraction),
        "recent_far_hold_break": bool(recent_far_hold_break),
        "escape_gap_fraction": float(escape_gap),
        "contact_pressure": float(contact_pressure),
        "notes": list(notes),
    }
    out["target_hint"] = str(pstate.get("target_hint", state.get("target_hint", "")) or "")
    out["ligand_smiles"] = str(pstate.get("ligand_smiles", state.get("ligand_smiles", "")) or "")
    out["ligand_scaffold"] = str(pstate.get("ligand_scaffold", state.get("ligand_scaffold", "")) or "")
    for key in ABFE_POCKET_DESCRIPTOR_COLS:
        try:
            vv = float(pstate.get(key, state.get(key, np.nan)))
        except Exception:
            vv = np.nan
        out[key] = float(vv) if np.isfinite(vv) else np.nan
    return out


def build_abfe_steering_policy_ood_payload(
    *,
    observation=None,
    policy_state=None,
    policy_model_config=None,
):
    obs = dict(observation or {})
    pstate = dict(policy_state or {})
    cfg = dict(policy_model_config or {})
    enabled = bool(cfg.get("ood_enabled", False))
    feature_names = list(cfg.get("ood_feature_names", []) or [])
    center = np.asarray(list(cfg.get("ood_center", []) or []), dtype=float)
    scale = np.asarray(list(cfg.get("ood_scale", []) or []), dtype=float)
    if (not enabled) or (not feature_names):
        return {
            "ok": True,
            "enabled": False,
            "in_domain": True,
            "distance": np.nan,
            "max_abs_z": np.nan,
        }
    if center.size != len(feature_names) or scale.size != len(feature_names):
        return {
            "ok": False,
            "enabled": True,
            "in_domain": False,
            "distance": np.nan,
            "max_abs_z": np.nan,
            "reason": "invalid_ood_config",
        }
    feature_payload = build_abfe_steering_policy_feature_payload(
        observation=dict(obs),
        policy_state=dict(pstate),
    )
    feature_map = dict(feature_payload.get("features", {}) or {})
    vec = []
    for name in feature_names:
        try:
            vv = float(feature_map.get(name, 0.0))
        except Exception:
            vv = 0.0
        vec.append(float(vv) if np.isfinite(vv) else 0.0)
    vec = np.asarray(vec, dtype=float)
    safe_scale = np.where(np.logical_and(np.isfinite(scale), scale > 1.0e-6), scale, 1.0)
    z = (vec - center) / safe_scale
    max_abs_z = float(np.max(np.abs(z))) if z.size else np.nan
    distance = float(np.sqrt(np.mean(np.square(z)))) if z.size else np.nan
    try:
        threshold_dist = float(cfg.get("ood_distance_threshold", np.nan))
    except Exception:
        threshold_dist = np.nan
    try:
        threshold_abs_z = float(cfg.get("ood_max_abs_z", np.nan))
    except Exception:
        threshold_abs_z = np.nan
    in_domain = True
    if np.isfinite(threshold_dist):
        in_domain = bool(in_domain and np.isfinite(distance) and distance <= threshold_dist)
    if np.isfinite(threshold_abs_z):
        in_domain = bool(in_domain and np.isfinite(max_abs_z) and max_abs_z <= threshold_abs_z)
    return {
        "ok": True,
        "enabled": True,
        "in_domain": bool(in_domain),
        "distance": float(distance) if np.isfinite(distance) else np.nan,
        "max_abs_z": float(max_abs_z) if np.isfinite(max_abs_z) else np.nan,
        "threshold_distance": float(threshold_dist) if np.isfinite(threshold_dist) else np.nan,
        "threshold_max_abs_z": float(threshold_abs_z) if np.isfinite(threshold_abs_z) else np.nan,
        "feature_names": list(feature_names),
    }


def build_abfe_steering_bandit_action_payload(
    *,
    observation=None,
    policy_state=None,
):
    obs = dict(observation or {})
    pstate = dict(policy_state or {})
    feature_payload = build_abfe_steering_policy_feature_payload(
        observation=dict(obs),
        policy_state=dict(pstate),
    )
    mode_txt = _sanitize_abfe_steering_policy_mode(
        pstate.get("policy_mode", obs.get("policy_mode", "heuristic"))
    )
    enabled_flag = bool(pstate.get("enabled", False) and mode_txt == "bandit")
    policy_model_cfg = dict(pstate.get("policy_model_config", {}) or {})
    trained_model_loaded = bool(policy_model_cfg.get("ok", False))
    ood_payload = build_abfe_steering_policy_ood_payload(
        observation=dict(obs),
        policy_state=dict(pstate),
        policy_model_config=dict(policy_model_cfg),
    )
    model_in_domain = bool(ood_payload.get("in_domain", True))
    model_blend_allowed = bool(trained_model_loaded and model_in_domain)
    default = {
        "ok": True,
        "enabled": bool(enabled_flag),
        "policy_mode": str(mode_txt),
        "action_name": "keep",
        "confidence": 0.0,
        "reason": "disabled" if not enabled_flag else "keep",
        "requested_pull_k_scale": 1.0,
        "request_switch_direction": False,
        "requested_hold_extension_samples": 0,
        "requested_hold_gap_samples": 0,
        "requested_detach_scale_multiplier": 1.0,
        "score_margin": 0.0,
        "policy_model_version": str(pstate.get("model_version", "") or "safe_bandit_v1"),
        "policy_model_source": str(pstate.get("model_source", "") or ""),
        "policy_model_used": bool(model_blend_allowed),
        "policy_model_blend_alpha": float(policy_model_cfg.get("score_blend_alpha", 0.0) or 0.0),
        "policy_model_ood_enabled": bool(ood_payload.get("enabled", False)),
        "policy_model_ood_in_domain": bool(model_in_domain),
        "policy_model_ood_distance": float(ood_payload.get("distance", np.nan)),
        "policy_model_ood_max_abs_z": float(ood_payload.get("max_abs_z", np.nan)),
    }
    if not enabled_flag:
        return default
    if not bool(obs.get("actionable", False)):
        default["reason"] = str(obs.get("reason", "not_actionable") or "not_actionable")
        return default

    size_scale = float(obs.get("size_scale", pstate.get("size_scale", 0.0)) or 0.0)
    hard_clash = bool(obs.get("hard_clash", False))
    soft_clash = bool(obs.get("soft_clash", False))
    energy_spike = bool(obs.get("energy_spike", False))
    stalled = bool(obs.get("stalled", False))
    phase = str(obs.get("phase", "pull") or "pull")
    k_fraction = float(obs.get("k_fraction", 0.0) or 0.0)
    escape_gap = float(obs.get("escape_gap_fraction", 0.0) or 0.0)
    contact_pressure = float(obs.get("contact_pressure", 0.0) or 0.0)
    far_ready = bool(obs.get("far_ready", False))
    far_hold_sustain = bool(obs.get("far_hold_sustain", False))
    can_switch = bool(obs.get("can_switch", False))
    metal_flag = bool(obs.get("metal_contact_expected", False))
    hold_progress_fraction = float(obs.get("hold_progress_fraction", 0.0) or 0.0)
    hold_gap_fraction = float(obs.get("hold_gap_fraction", 0.0) or 0.0)
    recent_break = bool(obs.get("recent_far_hold_break", False))
    hold_gap_current = int(max(0, int(obs.get("unbound_hold_max_gap_samples", pstate.get("unbound_hold_max_gap_samples", 3) or 3))))
    hold_gap_cap = int(max(hold_gap_current, int(obs.get("unbound_hold_max_gap_cap_samples", pstate.get("unbound_hold_max_gap_cap_samples", max(hold_gap_current, 3)) or max(hold_gap_current, 3)))))
    risk = (
        (1.35 if hard_clash else 0.0)
        + (0.60 if soft_clash else 0.0)
        + (0.55 if energy_spike else 0.0)
        + (0.20 if metal_flag else 0.0)
    )
    phase_far_hold = 1.0 if phase == "far_hold" else 0.0
    phase_detached = 1.0 if phase == "detached_settle" else 0.0

    scores = {
        "keep": 0.05 - (0.20 * risk),
        "increase_pull": (
            (1.25 if stalled else 0.0)
            + (0.75 * escape_gap)
            + (0.25 * size_scale)
            - (1.10 * risk)
            - (0.50 * phase_far_hold)
            - (0.65 * phase_detached)
            - (0.35 if far_ready else 0.0)
        ),
        "reduce_pull": (
            (1.15 if hard_clash else 0.0)
            + (0.75 if energy_spike else 0.0)
            + (0.35 if phase in {"far_hold", "detached_settle"} else 0.0)
            - (0.35 if stalled else 0.0)
        ),
        "switch_corridor": (
            ((0.95 if stalled else 0.0) + (0.65 * risk) + (0.35 * k_fraction))
            if can_switch
            else -999.0
        ),
        "detach_boost": (
            (1.10 * phase_far_hold * min(2.0, contact_pressure))
            + (0.25 * phase_far_hold * hold_progress_fraction)
            + (0.45 * phase_far_hold * max(0.0, 1.0 - float(obs.get("latest_min_dist_a", np.nan)) / max(float(obs.get("far_unbound_min_dist_a", 3.5)), 1.0)))
            + (0.20 if recent_break and phase in {"far_hold", "detached_settle"} else 0.0)
            - (0.50 if hard_clash else 0.0)
        ),
        "extend_hold": (
            (0.55 if phase in {"far_hold", "detached_settle"} and (not far_ready) else -999.0)
            + (0.45 * min(1.5, contact_pressure))
            + (0.25 if far_hold_sustain and (not far_ready) else 0.0)
            + (0.30 * hold_progress_fraction)
            + (0.35 * max(0.0, 1.0 - hold_gap_fraction))
            + (0.25 if recent_break else 0.0)
            - (0.60 if far_ready else 0.0)
            - (0.40 if hard_clash else 0.0)
        ),
    }
    if model_blend_allowed and bool(feature_payload.get("ok", False)):
        model_feature_names = list(policy_model_cfg.get("feature_names", []) or [])
        feature_map = dict(feature_payload.get("features", {}) or {})
        blend_alpha = float(np.clip(float(policy_model_cfg.get("score_blend_alpha", 0.35) or 0.35), 0.0, 2.0))
        for action_name in list(scores.keys()):
            model_action = dict((policy_model_cfg.get("actions", {}) or {}).get(action_name, {}) or {})
            weights = list(model_action.get("weights", []) or [])
            if len(weights) != len(model_feature_names):
                continue
            vec = np.asarray([float(feature_map.get(name, 0.0)) for name in model_feature_names], dtype=float)
            w = np.asarray(weights, dtype=float)
            model_score = float(np.dot(vec, w))
            scores[action_name] = float(scores[action_name] + (blend_alpha * model_score))
        default["policy_model_blend_alpha"] = float(blend_alpha)

    ranked = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
    best_name, best_score = ranked[0]
    second_score = float(ranked[1][1]) if len(ranked) > 1 else -999.0
    score_margin = float(best_score - second_score)
    min_action_score = 0.35
    if model_blend_allowed:
        try:
            min_action_score = float(policy_model_cfg.get("min_action_score", 0.35))
        except Exception:
            min_action_score = 0.35
    if float(best_score) < float(np.clip(min_action_score, 0.0, 5.0)):
        if trained_model_loaded and (not model_blend_allowed) and bool(ood_payload.get("enabled", False)):
            default["reason"] = "ood_fallback"
        return default

    conf = float(np.clip(0.35 + (0.25 * score_margin), 0.35, 0.99))
    out = dict(default)
    out["action_name"] = str(best_name)
    out["confidence"] = float(conf)
    out["score_margin"] = float(score_margin)
    out["reason"] = f"bandit_{best_name}"
    if best_name == "increase_pull":
        out["requested_pull_k_scale"] = float(min(1.30, max(1.04, pstate.get("k_push_scale", 1.12))))
    elif best_name == "reduce_pull":
        out["requested_pull_k_scale"] = float(min(0.95, max(0.70, pstate.get("k_relax_scale", 0.82))))
    elif best_name == "switch_corridor":
        out["request_switch_direction"] = True
    elif best_name == "detach_boost":
        out["requested_detach_scale_multiplier"] = float(
            min(1.30, max(1.02, pstate.get("detach_scale_multiplier", 1.10)))
        )
    elif best_name == "extend_hold":
        out["requested_hold_extension_samples"] = int(
            max(1, min(int(pstate.get("max_hold_extension_samples", 2) or 2), 1 + int(round(2.0 * size_scale))))
        )
        if phase in {"far_hold", "detached_settle"} and far_hold_sustain and hold_gap_current < hold_gap_cap:
            out["requested_hold_gap_samples"] = 1
    return out


def build_abfe_steering_policy_reward_payload(
    *,
    observation=None,
    final_decision=None,
):
    obs = dict(observation or {})
    final = dict(final_decision or {})
    reward = 0.0
    progress_site = float(obs.get("progress_site_com_a", np.nan))
    progress_reactive = float(obs.get("progress_reactive_a", np.nan))
    progress_site_thresh = max(0.1, float(obs.get("target_escape_site_com_a", 8.0)) * 0.05)
    progress_reactive_thresh = 0.2
    if np.isfinite(progress_site):
        reward += float(np.clip(progress_site / progress_site_thresh, -1.5, 2.0))
    if np.isfinite(progress_reactive):
        reward += 0.5 * float(np.clip(progress_reactive / progress_reactive_thresh, -1.0, 2.0))
    if bool(final.get("far_hold_stable", False)):
        reward += 1.5
    if bool(obs.get("far_hold_sustain", False)):
        reward += 0.35
    if bool(obs.get("soft_clash", False)):
        reward -= 0.6
    if bool(obs.get("hard_clash", False)):
        reward -= 1.5
    if bool(obs.get("energy_spike", False)):
        reward -= 0.8
    if bool(obs.get("stalled", False)):
        reward -= 0.35
    if bool(obs.get("recent_far_hold_break", False)):
        reward -= 0.15
    if bool(obs.get("phase", "pull") in {"far_hold", "detached_settle"}) and np.isfinite(float(obs.get("latest_contact_count_4a", np.nan))):
        if float(obs.get("latest_contact_count_4a", np.nan)) > float(obs.get("far_unbound_max_contact_count_4a", 4)):
            reward -= 0.4
    return {
        "ok": True,
        "reward_estimate": float(np.clip(reward, -5.0, 5.0)),
    }


def build_abfe_steering_policy_safe_action_payload(
    *,
    heuristic_decision=None,
    policy_action=None,
    adaptive_state=None,
    policy_state=None,
    current_pull_k_kj_nm2=np.nan,
    current_direction_index=0,
    direction_count=1,
    switches_used=0,
):
    out = dict(heuristic_decision or {})
    base_state = dict(adaptive_state or {})
    next_state = dict(out.get("adaptive_state", base_state) or {})
    pstate = dict(policy_state or {})
    action = dict(policy_action or {})
    mode_txt = _sanitize_abfe_steering_policy_mode(
        pstate.get("policy_mode", action.get("policy_mode", "heuristic"))
    )
    out.setdefault("policy_applied", False)
    out.setdefault("policy_mode", str(mode_txt))
    out.setdefault("policy_action_name", str(action.get("action_name", "keep") or "keep"))
    out.setdefault("policy_confidence", float(action.get("confidence", 0.0) or 0.0))
    out.setdefault("policy_reward_estimate", np.nan)
    out.setdefault("policy_requested_hold_extension_samples", int(action.get("requested_hold_extension_samples", 0) or 0))
    out.setdefault("policy_requested_hold_gap_samples", int(action.get("requested_hold_gap_samples", 0) or 0))
    out.setdefault(
        "policy_requested_detach_scale_multiplier",
        float(action.get("requested_detach_scale_multiplier", 1.0) or 1.0),
    )
    if "notes" not in out or not isinstance(out.get("notes"), list):
        out["notes"] = [str(v) for v in list(out.get("notes", []) or [])]

    try:
        current_k = float(current_pull_k_kj_nm2)
    except Exception:
        current_k = float(base_state.get("current_pull_k_kj_nm2", base_state.get("initial_pull_k_kj_nm2", 800.0)))
    if (not np.isfinite(current_k)) or (current_k <= 0.0):
        current_k = float(base_state.get("initial_pull_k_kj_nm2", 800.0))

    try:
        direction_i = int(current_direction_index or 0)
    except Exception:
        direction_i = 0
    direction_i = max(0, int(direction_i))

    try:
        n_dirs = int(direction_count or 0)
    except Exception:
        n_dirs = 0
    n_dirs = max(1, int(n_dirs))

    try:
        switch_used_i = int(switches_used or 0)
    except Exception:
        switch_used_i = int(next_state.get("switches_used", 0) or 0)
    switch_used_i = max(0, int(switch_used_i))

    max_switches = int(max(0, int(next_state.get("max_direction_switches", 0))))
    can_switch = bool((n_dirs > 1) and (switch_used_i < max_switches) and ((direction_i + 1) < n_dirs))
    k_min = float(next_state.get("k_min_kj_nm2", max(200.0, current_k * 0.5)))
    k_max = float(next_state.get("k_max_kj_nm2", max(k_min, current_k)))
    action_name = str(action.get("action_name", "keep") or "keep")
    enabled_flag = bool(action.get("enabled", False) and mode_txt == "bandit")
    phase = str(out.get("phase", next_state.get("phase", "pull")) or "pull").strip().lower()
    if phase not in {"pull", "far_hold", "detached_settle", "return"}:
        phase = "pull"

    next_state["policy_mode"] = str(mode_txt)
    next_state["policy_last_action_name"] = str(action_name)
    next_state["policy_last_confidence"] = float(action.get("confidence", 0.0) or 0.0)

    if not enabled_flag or action_name == "keep":
        out["adaptive_state"] = dict(next_state)
        return out

    applied = False
    heuristic_changed = bool(out.get("update_pull_k", False) or out.get("switch_direction", False))
    target_k = float(out.get("pull_k_kj_nm2", current_k))
    if (not np.isfinite(target_k)) or (target_k <= 0.0):
        target_k = float(current_k)

    try:
        requested_k_scale = float(action.get("requested_pull_k_scale", 1.0))
    except Exception:
        requested_k_scale = 1.0
    if np.isfinite(requested_k_scale) and abs(requested_k_scale - 1.0) > 1.0e-6:
        policy_k = float(np.clip(current_k * requested_k_scale, k_min, k_max))
        if requested_k_scale > 1.0:
            if (not out.get("update_pull_k", False)) or (target_k >= current_k):
                merged_k = float(max(target_k, policy_k))
                if merged_k > (target_k + 1.0e-6):
                    target_k = merged_k
                    out["update_pull_k"] = bool(target_k > current_k + 1.0e-6)
                    applied = True
        elif requested_k_scale < 1.0:
            if (not out.get("update_pull_k", False)) or (target_k <= current_k):
                merged_k = float(min(target_k, policy_k))
                if merged_k < (target_k - 1.0e-6):
                    target_k = merged_k
                    out["update_pull_k"] = bool(target_k < current_k - 1.0e-6)
                    applied = True

    if bool(action.get("request_switch_direction", False)) and (not bool(out.get("switch_direction", False))) and can_switch and phase == "pull":
        out["switch_direction"] = True
        out["direction_index"] = int(direction_i + 1)
        next_state["switches_used"] = int(switch_used_i + 1)
        applied = True

    try:
        hold_ext = int(action.get("requested_hold_extension_samples", 0) or 0)
    except Exception:
        hold_ext = 0
    hold_ext = max(0, int(hold_ext))
    if hold_ext > 0 and phase in {"far_hold", "detached_settle"} and (not bool(out.get("far_hold_stable", False))):
        target_key = "detached_settle_target_samples" if phase == "detached_settle" else "far_hold_target_samples"
        count_key = "detached_settle_count" if phase == "detached_settle" else "far_hold_count"
        current_target = int(max(1, int(next_state.get(target_key, 8) or 8)))
        current_count = int(max(0, int(next_state.get(count_key, 0) or 0)))
        if current_count < int(max(1, round(0.60 * current_target))):
            next_state[target_key] = int(min(current_target + hold_ext, current_target + 6))
            applied = True

    try:
        hold_gap_inc = int(action.get("requested_hold_gap_samples", 0) or 0)
    except Exception:
        hold_gap_inc = 0
    hold_gap_inc = max(0, int(hold_gap_inc))
    if hold_gap_inc > 0 and phase in {"far_hold", "detached_settle"} and (not bool(out.get("far_hold_stable", False))):
        current_gap = int(max(0, int(next_state.get("unbound_hold_max_gap_samples", 3) or 0)))
        gap_cap = int(max(current_gap, int(next_state.get("unbound_hold_max_gap_cap_samples", max(current_gap, 3)) or max(current_gap, 3))))
        if current_gap < gap_cap:
            next_state["unbound_hold_max_gap_samples"] = int(min(gap_cap, current_gap + hold_gap_inc))
            applied = True

    try:
        detach_mult = float(action.get("requested_detach_scale_multiplier", 1.0))
    except Exception:
        detach_mult = 1.0
    if np.isfinite(detach_mult) and detach_mult > 1.0 and phase in {"pull", "far_hold", "detached_settle"}:
        current_detach = float(next_state.get("surface_detach_distance_scale", 1.15))
        if (not np.isfinite(current_detach)) or (current_detach <= 0.0):
            current_detach = 1.15
        new_detach = float(np.clip(current_detach * detach_mult, 1.0, 1.75))
        if new_detach > (current_detach + 1.0e-6):
            next_state["surface_detach_distance_scale"] = float(new_detach)
            applied = True

    out["pull_k_kj_nm2"] = float(np.clip(target_k, k_min, k_max))
    out["should_adapt"] = bool(out.get("update_pull_k", False) or out.get("switch_direction", False) or applied)
    if applied:
        out["policy_applied"] = True
        out["policy_action_name"] = str(action_name)
        out["policy_confidence"] = float(action.get("confidence", 0.0) or 0.0)
        out["notes"].append(f"policy_{action_name}")
        if not heuristic_changed:
            next_state["adaptation_count"] = int(max(0, int(next_state.get("adaptation_count", 0))) + 1)
        next_state["current_pull_k_kj_nm2"] = float(out["pull_k_kj_nm2"])
        next_state["last_direction_index"] = int(out.get("direction_index", direction_i))
        next_state["applied_policy_action_count"] = int(max(0, int(next_state.get("applied_policy_action_count", 0))) + 1)

    out["adaptive_state"] = dict(next_state)
    return out


def build_abfe_steering_policy_trace_payload(
    *,
    observation=None,
    heuristic_decision=None,
    policy_action=None,
    final_decision=None,
):
    obs = dict(observation or {})
    heuristic = dict(heuristic_decision or {})
    action = dict(policy_action or {})
    final = dict(final_decision or {})
    feature_payload = build_abfe_steering_policy_feature_payload(
        observation=dict(obs),
        policy_state=dict(obs),
    )
    reward_payload = build_abfe_steering_policy_reward_payload(
        observation=dict(obs),
        final_decision=dict(final),
    )
    reward_est = float(reward_payload.get("reward_estimate", np.nan))
    final_reward = float(final.get("policy_reward_estimate", np.nan))
    if np.isfinite(final_reward):
        reward_est = float(final_reward)

    row = {
        "ok": True,
        "sample_index": int(obs.get("sample_index", 0) or 0),
        "phase": str(obs.get("phase", final.get("phase", heuristic.get("phase", "pull")) or "pull")),
        "policy_mode": str(action.get("policy_mode", obs.get("policy_mode", "heuristic")) or "heuristic"),
        "actionable": bool(obs.get("actionable", False)),
        "heuristic_reason": str(heuristic.get("reason", "") or ""),
        "final_reason": str(final.get("reason", heuristic.get("reason", "")) or ""),
        "policy_action_name": str(final.get("policy_action_name", action.get("action_name", "keep")) or "keep"),
        "policy_applied": bool(final.get("policy_applied", False)),
        "policy_confidence": float(final.get("policy_confidence", action.get("confidence", 0.0)) or 0.0),
        "reward_estimate": float(reward_est) if np.isfinite(reward_est) else np.nan,
        "current_pull_k_kj_nm2": float(obs.get("current_pull_k_kj_nm2", np.nan)),
        "final_pull_k_kj_nm2": float(final.get("pull_k_kj_nm2", heuristic.get("pull_k_kj_nm2", np.nan))),
        "current_direction_index": int(obs.get("current_direction_index", 0) or 0),
        "final_direction_index": int(final.get("direction_index", heuristic.get("direction_index", 0)) or 0),
        "switch_direction": bool(final.get("switch_direction", False)),
        "far_hold_stable": bool(final.get("far_hold_stable", False)),
        "progress_site_com_a": float(obs.get("progress_site_com_a", np.nan)),
        "progress_reactive_a": float(obs.get("progress_reactive_a", np.nan)),
        "latest_min_dist_a": float(obs.get("latest_min_dist_a", np.nan)),
        "latest_site_com_dist_a": float(obs.get("latest_site_com_dist_a", np.nan)),
        "latest_contact_count_4a": float(obs.get("latest_contact_count_4a", np.nan)),
        "stalled": bool(obs.get("stalled", False)),
        "soft_clash": bool(obs.get("soft_clash", False)),
        "hard_clash": bool(obs.get("hard_clash", False)),
        "energy_spike": bool(obs.get("energy_spike", False)),
        "far_ready": bool(obs.get("far_ready", False)),
        "far_hold_sustain": bool(obs.get("far_hold_sustain", False)),
        "can_switch": bool(obs.get("can_switch", False)),
        "k_fraction": float(obs.get("k_fraction", np.nan)),
        "escape_gap_fraction": float(obs.get("escape_gap_fraction", np.nan)),
        "contact_pressure": float(obs.get("contact_pressure", np.nan)),
        "size_scale": float(obs.get("size_scale", np.nan)),
        "ligand_heavy_atoms": int(obs.get("ligand_heavy_atoms", 0) or 0),
        "formal_charge_e": float(obs.get("formal_charge_e", np.nan)),
        "charge_density": float(obs.get("charge_density", np.nan)),
        "ring_count": int(obs.get("ring_count", 0) or 0),
        "rotatable_bonds": int(obs.get("rotatable_bonds", 0) or 0),
        "metal_contact_expected": bool(obs.get("metal_contact_expected", False)),
        "target_hint": str(obs.get("target_hint", "") or ""),
        "ligand_smiles": str(obs.get("ligand_smiles", "") or ""),
        "ligand_scaffold": str(obs.get("ligand_scaffold", "") or ""),
        "latest_total_e_kj": float(obs.get("latest_total_e_kj", np.nan)),
        "policy_model_version": str(action.get("policy_model_version", "") or ""),
        "policy_model_source": str(action.get("policy_model_source", "") or ""),
        "policy_model_used": bool(action.get("policy_model_used", False)),
        "policy_model_blend_alpha": float(action.get("policy_model_blend_alpha", np.nan)),
        "policy_model_ood_enabled": bool(action.get("policy_model_ood_enabled", False)),
        "policy_model_ood_in_domain": bool(action.get("policy_model_ood_in_domain", True)),
        "policy_model_ood_distance": float(action.get("policy_model_ood_distance", np.nan)),
        "policy_model_ood_max_abs_z": float(action.get("policy_model_ood_max_abs_z", np.nan)),
        "requested_hold_extension_samples": int(
            final.get("policy_requested_hold_extension_samples", action.get("requested_hold_extension_samples", 0)) or 0
        ),
        "requested_hold_gap_samples": int(
            final.get("policy_requested_hold_gap_samples", action.get("requested_hold_gap_samples", 0)) or 0
        ),
        "requested_detach_scale_multiplier": float(
            final.get(
                "policy_requested_detach_scale_multiplier",
                action.get("requested_detach_scale_multiplier", 1.0),
            ) or 1.0
        ),
    }
    for key in ABFE_POCKET_DESCRIPTOR_COLS:
        try:
            vv = float(obs.get(key, np.nan))
        except Exception:
            vv = np.nan
        row[key] = float(vv) if np.isfinite(vv) else np.nan
    for feat_name, feat_value in dict(feature_payload.get("features", {}) or {}).items():
        try:
            row[feat_name] = float(feat_value)
        except Exception:
            row[feat_name] = np.nan
    return row


def build_abfe_steering_policy_summary_payload(
    *,
    trace_rows=None,
):
    rows = list(trace_rows or [])
    action_counts = {}
    applied_counts = {}
    confs = []
    rewards = []
    model_versions = set()
    trained_model_rows = 0
    for row in rows:
        rr = dict(row or {})
        action_name = str(rr.get("policy_action_name", "keep") or "keep")
        action_counts[action_name] = int(action_counts.get(action_name, 0)) + 1
        if bool(rr.get("policy_applied", False)):
            applied_counts[action_name] = int(applied_counts.get(action_name, 0)) + 1
        if bool(rr.get("policy_model_used", False)):
            trained_model_rows += 1
        model_version = str(rr.get("policy_model_version", "") or "").strip()
        if model_version:
            model_versions.add(model_version)
        try:
            conf = float(rr.get("policy_confidence", np.nan))
        except Exception:
            conf = np.nan
        if np.isfinite(conf):
            confs.append(float(conf))
        try:
            rew = float(rr.get("reward_estimate", np.nan))
        except Exception:
            rew = np.nan
        if np.isfinite(rew):
            rewards.append(float(rew))

    return {
        "ok": True,
        "n_rows": int(len(rows)),
        "n_applied": int(sum(int(v) for v in list(applied_counts.values()) or [])),
        "mean_confidence": float(np.mean(np.asarray(confs, dtype=float))) if confs else np.nan,
        "mean_reward_estimate": float(np.mean(np.asarray(rewards, dtype=float))) if rewards else np.nan,
        "action_counts": dict(action_counts),
        "applied_action_counts": dict(applied_counts),
        "trained_model_rows": int(trained_model_rows),
        "model_versions": list(sorted(model_versions)),
        "last_action_name": str(rows[-1].get("policy_action_name", "keep")) if rows else "keep",
    }


def build_abfe_live_plot_setup_plan(
    *,
    enabled=True,
    figsize=(8.0, 5.0),
    sharex=True,
    title="ABFE-lite Live Sampling",
    energy_label="Energy (kJ/mol)",
    distance_label="Distance (Å)",
    time_label="Time (ps)",
    energy_line_label="Total E",
    distance_line_label="Reactive dist",
    energy_line_color="#1F77B4",
    energy_point_color="#D62728",
    distance_line_color="#9467BD",
    distance_point_color="#FF7F0E",
    line_width=1.2,
    point_marker="o",
    point_size=5.0,
    grid_alpha=0.25,
    legend_loc="best",
    legend_fontsize=8,
    tight_layout=True,
    draw_idle=True,
    flush_events=True,
    interactive=True,
):
    if not bool(enabled):
        return {
            "ok": True,
            "should_create": False,
            "reason": "disabled",
            "warning_prefix": "ABFE live plot setup warning:",
        }

    fig_default = (8.0, 5.0)
    try:
        fig_pair = tuple(figsize) if isinstance(figsize, (tuple, list)) else fig_default
    except Exception:
        fig_pair = fig_default
    if len(fig_pair) != 2:
        fig_pair = fig_default
    try:
        fig_w = float(fig_pair[0])
        fig_h = float(fig_pair[1])
    except Exception:
        fig_w, fig_h = fig_default
    if (not np.isfinite(fig_w)) or (fig_w <= 0.0):
        fig_w = fig_default[0]
    if (not np.isfinite(fig_h)) or (fig_h <= 0.0):
        fig_h = fig_default[1]

    try:
        lw = float(line_width)
    except Exception:
        lw = 1.2
    if (not np.isfinite(lw)) or (lw <= 0.0):
        lw = 1.2

    try:
        msz = float(point_size)
    except Exception:
        msz = 5.0
    if (not np.isfinite(msz)) or (msz <= 0.0):
        msz = 5.0

    try:
        g_alpha = float(grid_alpha)
    except Exception:
        g_alpha = 0.25
    if not np.isfinite(g_alpha):
        g_alpha = 0.25
    g_alpha = float(np.clip(g_alpha, 0.0, 1.0))

    try:
        legend_fs = float(legend_fontsize)
    except Exception:
        legend_fs = 8.0
    if (not np.isfinite(legend_fs)) or (legend_fs <= 0.0):
        legend_fs = 8.0

    return {
        "ok": True,
        "should_create": True,
        "reason": "",
        "nrows": 2,
        "ncols": 1,
        "figsize": [float(fig_w), float(fig_h)],
        "sharex": bool(sharex),
        "title": str(title),
        "energy_label": str(energy_label),
        "distance_label": str(distance_label),
        "time_label": str(time_label),
        "energy_line_label": str(energy_line_label),
        "distance_line_label": str(distance_line_label),
        "energy_line_color": str(energy_line_color),
        "energy_point_color": str(energy_point_color),
        "distance_line_color": str(distance_line_color),
        "distance_point_color": str(distance_point_color),
        "line_width": float(lw),
        "point_marker": str(point_marker or "o"),
        "point_size": float(msz),
        "grid_alpha": float(g_alpha),
        "legend_loc": str(legend_loc or "best"),
        "legend_fontsize": float(legend_fs),
        "tight_layout": bool(tight_layout),
        "draw_idle": bool(draw_idle),
        "flush_events": bool(flush_events),
        "interactive": bool(interactive),
        "warning_prefix": "ABFE live plot setup warning:",
    }


def build_abfe_live_plot_update_default_payload():
    return build_abfe_live_plot_update_payload(
        times_ps=[],
        total_e=[],
        reactive_dist=[],
        min_dist=[],
    )


def build_abfe_live_plot_state_payload(
    *,
    enabled=False,
    fig=None,
    ax_e=None,
    ax_d=None,
    line_e=None,
    point_e=None,
    line_d=None,
    point_d=None,
    strict_handles=False,
):
    handles = {
        "fig": fig,
        "ax_e": ax_e,
        "ax_d": ax_d,
        "line_e": line_e,
        "point_e": point_e,
        "line_d": line_d,
        "point_d": point_d,
    }
    missing = [str(k) for (k, v) in handles.items() if v is None]
    enabled_flag = bool(enabled)
    reason = ""
    if not enabled_flag:
        reason = "disabled"
    elif bool(strict_handles) and len(missing):
        enabled_flag = False
        reason = "missing_handles"

    if not enabled_flag:
        for k in list(handles.keys()):
            handles[k] = None

    payload = {
        "ok": True,
        "enabled": bool(enabled_flag),
        "reason": str(reason),
        "missing_handles": list(missing),
    }
    payload.update(handles)
    return payload


def build_abfe_live_plot_update_payload(
    *,
    times_ps=None,
    total_e=None,
    reactive_dist=None,
    min_dist=None,
    max_points=800,
):
    def _float_series(values):
        if values is None:
            return np.asarray([], dtype=float)
        return np.asarray(list(values), dtype=float)

    t_arr = _float_series(times_ps)
    e_arr = _float_series(total_e)
    d_arr_reactive = _float_series(reactive_dist)
    d_arr_min = _float_series(min_dist)

    d_arr = np.asarray(d_arr_reactive, dtype=float)
    distance_source = "reactive_dist"
    if not np.any(np.isfinite(d_arr)):
        d_arr = np.asarray(d_arr_min, dtype=float)
        distance_source = "min_dist"

    n = int(min(t_arr.size, e_arr.size, d_arr.size))
    n_raw = int(n)
    max_points_i = _abfe_positive_int(max_points, default=800, minimum=25)
    plot_stride = 1
    if n > 0:
        t_arr = np.asarray(t_arr[:n], dtype=float)
        e_arr = np.asarray(e_arr[:n], dtype=float)
        d_arr = np.asarray(d_arr[:n], dtype=float)
        point_t = float(t_arr[-1])
        point_e = float(e_arr[-1])
        point_d = float(d_arr[-1])
        pick_idx = _abfe_downsample_indices(n, max_points=max_points_i, include_last=True)
        if pick_idx.size and pick_idx.size < n:
            plot_stride = int(max(1, np.ceil(float(n) / float(max(1, pick_idx.size)))))
            t_arr = np.asarray(t_arr[pick_idx], dtype=float)
            e_arr = np.asarray(e_arr[pick_idx], dtype=float)
            d_arr = np.asarray(d_arr[pick_idx], dtype=float)
        point_e_x = [float(point_t)]
        point_e_y = [float(point_e)]
        if np.isfinite(point_d):
            point_d_x = [float(point_t)]
            point_d_y = [float(point_d)]
            has_point_d = True
        else:
            point_d_x = []
            point_d_y = []
            has_point_d = False
    else:
        t_arr = np.asarray([], dtype=float)
        e_arr = np.asarray([], dtype=float)
        d_arr = np.asarray([], dtype=float)
        point_e_x = []
        point_e_y = []
        point_d_x = []
        point_d_y = []
        has_point_d = False

    return {
        "ok": True,
        "distance_source": str(distance_source),
        "n_raw": int(n_raw),
        "max_points": int(max_points_i),
        "plot_stride": int(plot_stride),
        "t_arr": np.asarray(t_arr, dtype=float),
        "e_arr": np.asarray(e_arr, dtype=float),
        "d_arr": np.asarray(d_arr, dtype=float),
        "point_e_x": list(point_e_x),
        "point_e_y": list(point_e_y),
        "point_d_x": list(point_d_x),
        "point_d_y": list(point_d_y),
        "has_point_d": bool(has_point_d),
    }


def build_abfe_live_plot_apply_plan(
    *,
    should_update_live_plot=False,
    plot_payload=None,
    relim=True,
    autoscale=True,
    draw_idle=True,
    flush_events=True,
):
    payload = dict(plot_payload or {})
    if not bool(should_update_live_plot):
        return {
            "ok": True,
            "should_apply": False,
            "reason": "update_not_requested",
            "line_e_x": np.asarray([], dtype=float),
            "line_e_y": np.asarray([], dtype=float),
            "point_e_x": [],
            "point_e_y": [],
            "line_d_x": np.asarray([], dtype=float),
            "line_d_y": np.asarray([], dtype=float),
            "point_d_x": [],
            "point_d_y": [],
            "relim": bool(relim),
            "autoscale": bool(autoscale),
            "draw_idle": bool(draw_idle),
            "flush_events": bool(flush_events),
            "disable_live_plot_on_failure": True,
            "warning_prefix": "ABFE live plot update warning:",
        }

    t_arr = np.asarray(payload.get("t_arr", []), dtype=float)
    e_arr = np.asarray(payload.get("e_arr", []), dtype=float)
    d_arr = np.asarray(payload.get("d_arr", []), dtype=float)
    n = int(min(t_arr.size, e_arr.size, d_arr.size))
    if n > 0:
        t_arr = np.asarray(t_arr[:n], dtype=float)
        e_arr = np.asarray(e_arr[:n], dtype=float)
        d_arr = np.asarray(d_arr[:n], dtype=float)
    else:
        t_arr = np.asarray([], dtype=float)
        e_arr = np.asarray([], dtype=float)
        d_arr = np.asarray([], dtype=float)

    point_e_x = list(payload.get("point_e_x", []) or [])
    point_e_y = list(payload.get("point_e_y", []) or [])
    has_point_d = bool(payload.get("has_point_d", False))
    if has_point_d:
        point_d_x = list(payload.get("point_d_x", []) or [])
        point_d_y = list(payload.get("point_d_y", []) or [])
    else:
        point_d_x = []
        point_d_y = []

    return {
        "ok": True,
        "should_apply": True,
        "reason": "",
        "line_e_x": np.asarray(t_arr, dtype=float),
        "line_e_y": np.asarray(e_arr, dtype=float),
        "point_e_x": list(point_e_x),
        "point_e_y": list(point_e_y),
        "line_d_x": np.asarray(t_arr, dtype=float),
        "line_d_y": np.asarray(d_arr, dtype=float),
        "point_d_x": list(point_d_x),
        "point_d_y": list(point_d_y),
        "relim": bool(relim),
        "autoscale": bool(autoscale),
        "draw_idle": bool(draw_idle),
        "flush_events": bool(flush_events),
        "disable_live_plot_on_failure": True,
        "warning_prefix": "ABFE live plot update warning:",
    }


def build_abfe_md_sample_payload(
    *,
    xyz_nm=None,
    sample_index=0,
    sample_every_ps=1.0,
    e_lj=np.nan,
    e_c=np.nan,
    lam_value=0.0,
    prot_heavy_idx=None,
    lig_heavy_idx=None,
    lig_masses=None,
    local_lig_idx=None,
    site_anchor_idx=None,
    site_com0_nm=None,
    contact_cutoff_a=4.0,
    include_coords=True,
):
    xyz = np.asarray(xyz_nm if xyz_nm is not None else [], dtype=float)
    prot_idx = np.asarray(prot_heavy_idx if prot_heavy_idx is not None else [], dtype=int)
    lig_idx = np.asarray(lig_heavy_idx if lig_heavy_idx is not None else [], dtype=int)
    lig_mass_arr = np.asarray(lig_masses if lig_masses is not None else [], dtype=float)
    local_lig = np.asarray(local_lig_idx if local_lig_idx is not None else [], dtype=int)
    site_idx = np.asarray(site_anchor_idx if site_anchor_idx is not None else [], dtype=int)
    site_com0 = np.asarray(site_com0_nm if site_com0_nm is not None else [np.nan, np.nan, np.nan], dtype=float)

    e_lj_f = float(e_lj)
    e_c_f = float(e_c)
    e_tot = float(e_lj_f + e_c_f)

    dmin_a = np.nan
    contact_count_4a = np.nan
    if prot_idx.size and lig_idx.size and xyz.ndim == 2 and xyz.shape[1] == 3:
        dv = xyz[prot_idx][:, None, :] - xyz[lig_idx][None, :, :]
        rij = np.linalg.norm(dv, axis=2)
        if rij.size:
            dmin_a = float(np.min(rij) * 10.0)
            contact_cutoff_nm = max(0.0, 0.1 * float(contact_cutoff_a))
            if contact_cutoff_nm > 0.0:
                near_contact = np.any(rij <= contact_cutoff_nm, axis=1)
                contact_count_4a = float(np.sum(near_contact))

    try:
        lxyz = xyz[lig_idx]
        com = np.average(lxyz, axis=0, weights=lig_mass_arr)
    except Exception:
        com = np.mean(xyz[local_lig], axis=0)

    rc_a = np.nan
    site_com_a = np.nan
    site_dmin_a = np.nan
    if lig_idx.size > 0 and site_idx.size > 0 and xyz.ndim == 2 and xyz.shape[1] == 3:
        site_com_a = float(np.linalg.norm(np.asarray(com, dtype=float) - site_com0) * 10.0)
        dv_site = xyz[site_idx][:, None, :] - xyz[lig_idx][None, :, :]
        rij_site = np.linalg.norm(dv_site, axis=2)
        if rij_site.size:
            site_dmin_a = float(np.min(rij_site) * 10.0)
        if np.isfinite(site_dmin_a):
            rc_a = float(site_dmin_a)
        elif np.isfinite(site_com_a):
            rc_a = float(site_com_a)

    frame_i = int(sample_index) + 1
    try:
        sp = float(sample_every_ps)
    except Exception:
        sp = 1.0
    time_ps = float(frame_i * sp)

    # Coordinate arrays are the dominant ABFE-lite memory cost. The sampling
    # loop can keep scalar observables at every frame and request coordinates
    # only for decimated visualization/rescoring states.
    coords_a = np.asarray([], dtype=np.float32)
    if bool(include_coords):
        coords_a = np.asarray(xyz, dtype=np.float32) * 10.0

    return {
        "ok": True,
        "frame": int(frame_i),
        "time_ps": float(time_ps),
        "e_lj": float(e_lj_f),
        "e_c": float(e_c_f),
        "e_tot": float(e_tot),
        "dmin_a": float(dmin_a) if np.isfinite(dmin_a) else np.nan,
        "protein_contact_count_4a": float(contact_count_4a) if np.isfinite(contact_count_4a) else np.nan,
        "reactive_dist_a": float(rc_a) if np.isfinite(rc_a) else np.nan,
        "site_min_dist_a": float(site_dmin_a) if np.isfinite(site_dmin_a) else np.nan,
        "site_com_dist_a": float(site_com_a) if np.isfinite(site_com_a) else np.nan,
        "lig_com_nm": np.asarray(com, dtype=float),
        "lam_value": float(lam_value),
        "coords_included": bool(include_coords),
        "coords_a": np.asarray(coords_a, dtype=np.float32),
    }


def build_abfe_live_update_decision(
    *,
    sample_index=0,
    n_samples=0,
    live_view_ready=False,
    view_update_stride=1,
    live_plot_enabled=False,
    plot_update_stride=1,
):
    try:
        s = int(sample_index or 0)
    except Exception:
        s = 0
    s = max(0, int(s))
    try:
        n = int(n_samples or 0)
    except Exception:
        n = 0
    n = max(0, int(n))
    try:
        v_stride = int(view_update_stride or 1)
    except Exception:
        v_stride = 1
    v_stride = max(1, int(v_stride))
    try:
        p_stride = int(plot_update_stride or 1)
    except Exception:
        p_stride = 1
    p_stride = max(1, int(p_stride))

    is_last = bool(n > 0 and s == (n - 1))
    update_view_tick = bool((s % v_stride) == 0)
    update_plot_tick = bool((s % p_stride) == 0)

    should_update_live_view = bool(live_view_ready) and bool(update_view_tick or is_last)
    should_update_live_plot = bool(live_plot_enabled) and bool(update_plot_tick or is_last)

    return {
        "ok": True,
        "sample_index": int(s),
        "n_samples": int(n),
        "is_last_sample": bool(is_last),
        "view_update_stride": int(v_stride),
        "plot_update_stride": int(p_stride),
        "should_update_live_view": bool(should_update_live_view),
        "should_update_live_plot": bool(should_update_live_plot),
    }


def build_abfe_live_view_update_plan(
    *,
    should_update_live_view=False,
    xyz_nm=None,
    live_view_object="",
    state=1,
    scale_to_a=10.0,
    refresh=True,
):
    if not bool(should_update_live_view):
        return {
            "ok": True,
            "should_apply": False,
            "reason": "update_not_requested",
            "coords_a": np.asarray([], dtype=np.float32),
            "object_name": str(live_view_object or ""),
            "state": int(max(1, int(state or 1))),
            "refresh": bool(refresh),
            "disable_live_view_on_failure": True,
            "warning_prefix": "ABFE live view update warning:",
        }

    obj_name = str(live_view_object or "").strip()
    if not obj_name:
        return {
            "ok": True,
            "should_apply": False,
            "reason": "missing_live_view_object",
            "coords_a": np.asarray([], dtype=np.float32),
            "object_name": "",
            "state": int(max(1, int(state or 1))),
            "refresh": bool(refresh),
            "disable_live_view_on_failure": True,
            "warning_prefix": "ABFE live view update warning:",
        }

    try:
        xyz = np.asarray(xyz_nm, dtype=float)
    except Exception:
        xyz = np.asarray([], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] <= 0:
        return {
            "ok": True,
            "should_apply": False,
            "reason": "invalid_xyz_nm",
            "coords_a": np.asarray([], dtype=np.float32),
            "object_name": str(obj_name),
            "state": int(max(1, int(state or 1))),
            "refresh": bool(refresh),
            "disable_live_view_on_failure": True,
            "warning_prefix": "ABFE live view update warning:",
        }

    try:
        state_i = int(state or 1)
    except Exception:
        state_i = 1
    state_i = max(1, int(state_i))

    try:
        scale = float(scale_to_a)
    except Exception:
        scale = 10.0
    if not np.isfinite(scale):
        scale = 10.0

    coords_a = np.asarray(xyz, dtype=np.float32) * float(scale)
    return {
        "ok": True,
        "should_apply": True,
        "reason": "",
        "coords_a": np.asarray(coords_a, dtype=np.float32),
        "object_name": str(obj_name),
        "state": int(state_i),
        "refresh": bool(refresh),
        "disable_live_view_on_failure": True,
        "warning_prefix": "ABFE live view update warning:",
    }


def build_abfe_ui_process_events_decision(
    *,
    sample_index=0,
    n_samples=0,
    pump_stride=1,
    enabled=True,
):
    try:
        s = int(sample_index or 0)
    except Exception:
        s = 0
    s = max(0, int(s))

    try:
        n = int(n_samples or 0)
    except Exception:
        n = 0
    n = max(0, int(n))

    try:
        stride = int(pump_stride or 1)
    except Exception:
        stride = 1
    stride = max(1, int(stride))

    is_last = bool(n > 0 and s == (n - 1))
    pump_tick = bool((s % stride) == 0)
    should_pump = bool(enabled) and bool(pump_tick or is_last)

    return {
        "ok": True,
        "sample_index": int(s),
        "n_samples": int(n),
        "pump_stride": int(stride),
        "enabled": bool(enabled),
        "is_last_sample": bool(is_last),
        "pump_tick": bool(pump_tick),
        "should_pump_events": bool(should_pump),
    }


def build_abfe_md_progress_dialog_message(
    *,
    bias_enabled=False,
    steering_mode="",
):
    mode_txt = str(steering_mode or "").strip().lower()
    if mode_txt not in ("pull_hold", "pull_hold_return"):
        mode_txt = "pull_hold"

    msg = "Running ABFE-lite whole-protein MD (gas phase)..."
    if bool(bias_enabled):
        if mode_txt == "pull_hold_return":
            msg = "Running ABFE-lite whole-protein MD (gas phase, pull/hold/return steering)..."
        else:
            msg = "Running ABFE-lite whole-protein MD (gas phase, pull/hold steering)..."

    return {
        "ok": True,
        "bias_enabled": bool(bias_enabled),
        "steering_mode": str(mode_txt),
        "message": str(msg),
    }


def build_abfe_md_progress_dialog_plan(
    *,
    n_samples=0,
    bias_enabled=False,
    steering_mode="",
    cancel_text="Cancel",
    minimum=0,
    modality="window_modal",
):
    msg_payload = build_abfe_md_progress_dialog_message(
        bias_enabled=bool(bias_enabled),
        steering_mode=str(steering_mode),
    )

    try:
        n_total = int(n_samples or 0)
    except Exception:
        n_total = 0
    n_total = max(0, int(n_total))

    try:
        min_value = int(minimum or 0)
    except Exception:
        min_value = 0
    min_value = max(0, int(min_value))
    max_value = max(int(min_value), int(n_total))

    cancel_label = str(cancel_text or "").strip()
    if not cancel_label:
        cancel_label = "Cancel"

    mode = str(modality or "").strip().lower()
    if mode not in ("window_modal", "non_modal"):
        mode = "window_modal"

    return {
        "ok": True,
        "message": str(msg_payload.get("message", "Running ABFE-lite whole-protein MD (gas phase)...")),
        "cancel_text": str(cancel_label),
        "minimum": int(min_value),
        "maximum": int(max_value),
        "modality": str(mode),
        "bias_enabled": bool(msg_payload.get("bias_enabled", False)),
        "steering_mode": str(msg_payload.get("steering_mode", "pull_hold")),
    }


def build_abfe_md_progress_step_decision(
    *,
    sample_index=0,
    n_samples=0,
    was_canceled=False,
):
    try:
        s = int(sample_index or 0)
    except Exception:
        s = 0
    s = max(0, int(s))

    try:
        n = int(n_samples or 0)
    except Exception:
        n = 0
    n = max(0, int(n))

    is_last = bool(n > 0 and s == (n - 1))
    clamped_current = int(min(max(s, 0), n))
    next_value = int(min(max(s + 1, 0), n))
    canceled = bool(was_canceled)

    return {
        "ok": True,
        "sample_index": int(s),
        "n_samples": int(n),
        "was_canceled": bool(canceled),
        "should_abort": bool(canceled),
        "is_last_sample": bool(is_last),
        "progress_current_value": int(clamped_current),
        "progress_next_value": int(next_value),
        "should_set_progress": bool(not canceled),
        "completed": bool((not canceled) and is_last),
    }


def build_abfe_md_progress_finalize_decision(
    *,
    n_samples=0,
    n_sampled=0,
    was_canceled=False,
    set_to_max_on_cancel=False,
):
    try:
        n = int(n_samples or 0)
    except Exception:
        n = 0
    n = max(0, int(n))

    try:
        sampled = int(n_sampled or 0)
    except Exception:
        sampled = 0
    sampled = max(0, int(sampled))
    sampled = int(min(sampled, n))

    canceled = bool(was_canceled)
    force_max = bool(set_to_max_on_cancel)
    if canceled and (not force_max):
        final_value = int(sampled)
    else:
        final_value = int(n)
    final_value = int(min(max(final_value, 0), n))

    return {
        "ok": True,
        "n_samples": int(n),
        "n_sampled": int(sampled),
        "was_canceled": bool(canceled),
        "set_to_max_on_cancel": bool(force_max),
        "final_value": int(final_value),
        "should_set_progress": True,
        "completed": bool(final_value >= n and n > 0 and not canceled),
    }


def build_abfe_traj_state_pick_plan(
    *,
    n_sampled=0,
    max_states_view=1500,
):
    try:
        n_total = int(n_sampled or 0)
    except Exception:
        n_total = 0
    n_total = max(0, int(n_total))

    try:
        max_states = int(1500 if max_states_view is None else max_states_view)
    except Exception:
        max_states = 1500
    max_states = max(1, int(max_states))

    if n_total <= 0:
        stride_view = 1
        picked = []
    else:
        stride_view = int(max(1, np.ceil(float(n_total) / float(max_states))))
        picked = list(range(0, int(n_total), int(stride_view)))
        if picked and picked[-1] != (int(n_total) - 1):
            picked.append(int(n_total) - 1)
        if not picked:
            picked = [0]

    return {
        "ok": True,
        "n_sampled": int(n_total),
        "max_states_view": int(max_states),
        "stride_view": int(stride_view),
        "picked_indices": [int(i) for i in list(picked or [])],
        "n_selected": int(len(list(picked or []))),
    }


def build_abfe_final_traj_build_plan(
    *,
    sampled_state_count=0,
    n_local_model_indices=0,
    n_local=0,
    local_model_indices=None,
    idx_selection="",
    loaded_object="",
    state_index=1,
    live_view_object="",
    live_view_atom_count=0,
    timestamp_token="",
    existing_objects=None,
    object_prefix="abfe_local_md_traj",
    max_states_view=1500,
    group_name="ABFE.LocalMD",
):
    try:
        n_sampled = int(sampled_state_count or 0)
    except Exception:
        n_sampled = 0
    n_sampled = max(0, int(n_sampled))

    try:
        n_local_idx = int(n_local_model_indices or 0)
    except Exception:
        n_local_idx = 0
    n_local_idx = max(0, int(n_local_idx))

    try:
        n_local_expected = int(n_local or 0)
    except Exception:
        n_local_expected = 0
    n_local_expected = max(0, int(n_local_expected))

    try:
        st_idx = int(state_index or 1)
    except Exception:
        st_idx = 1
    st_idx = max(1, int(st_idx))

    try:
        live_atoms = int(live_view_atom_count or 0)
    except Exception:
        live_atoms = 0
    live_atoms = max(0, int(live_atoms))

    local_model_idx = []
    try:
        local_model_idx = [int(i) for i in list(local_model_indices or [])]
    except Exception:
        local_model_idx = []

    if n_sampled <= 0:
        return {
            "ok": True,
            "should_build": False,
            "reason": "no_sampled_states",
            "final_view_object": "",
            "picked_indices": [],
            "load_steps": [],
        }

    if n_local_expected <= 0:
        return {
            "ok": True,
            "should_build": False,
            "reason": "invalid_n_local",
            "final_view_object": "",
            "picked_indices": [],
            "load_steps": [],
        }

    if n_local_idx != n_local_expected:
        return {
            "ok": True,
            "should_build": False,
            "reason": "local_index_count_mismatch",
            "final_view_object": "",
            "picked_indices": [],
            "load_steps": [],
        }

    prefix = str(object_prefix or "abfe_local_md_traj").strip()
    if not prefix:
        prefix = "abfe_local_md_traj"
    ts_token = re.sub(r"[^0-9A-Za-z_-]+", "", str(timestamp_token or "").strip())
    if not ts_token:
        ts_token = "traj"
    final_view_obj = f"{prefix}_{ts_token}"

    existing_set = {str(o) for o in list(existing_objects or [])}
    should_delete_existing = bool(final_view_obj in existing_set)

    live_obj = str(live_view_object or "").strip()
    use_live_source = bool(live_obj and live_atoms == n_local_expected)

    idx_sel = str(idx_selection or "").strip()
    if (not idx_sel) and (not use_live_source) and len(local_model_idx):
        idx_sel = "index " + "+".join(str(i) for i in local_model_idx)

    loaded_obj = str(loaded_object or "").strip()
    if use_live_source:
        create_selection = str(live_obj)
        create_source_state = 1
        create_mode = "from_object"
    else:
        if (not loaded_obj) or (not idx_sel):
            return {
                "ok": True,
                "should_build": False,
                "reason": "missing_create_selection",
                "final_view_object": str(final_view_obj),
                "picked_indices": [],
                "load_steps": [],
            }
        create_selection = f"({loaded_obj}) and ({idx_sel})"
        create_source_state = int(st_idx)
        create_mode = "from_selection"

    pick_plan = build_abfe_traj_state_pick_plan(
        n_sampled=int(n_sampled),
        max_states_view=int(max_states_view),
    )
    picked = [int(i) for i in list(pick_plan.get("picked_indices", []) or [])]
    if (not picked) and n_sampled > 0:
        picked = [0]

    load_steps = []
    out_state = 1
    for src_idx in picked:
        if src_idx < 0 or src_idx >= n_sampled:
            continue
        load_steps.append(
            {
                "src_index": int(src_idx),
                "out_state": int(out_state),
                "duplicate_state_before_load": bool(out_state > 1),
            }
        )
        out_state += 1

    return {
        "ok": True,
        "should_build": True,
        "reason": "",
        "final_view_object": str(final_view_obj),
        "should_delete_existing": bool(should_delete_existing),
        "use_live_source": bool(use_live_source),
        "create_mode": str(create_mode),
        "create_selection": str(create_selection),
        "create_source_state": int(create_source_state),
        "idx_selection": str(idx_sel),
        "n_sampled": int(n_sampled),
        "n_local_expected": int(n_local_expected),
        "picked_indices": [int(i) for i in list(picked or [])],
        "n_selected": int(len(list(picked or []))),
        "load_steps": list(load_steps),
        "apply_visual_style": True,
        "group_name": str(group_name or "ABFE.LocalMD"),
        "set_traj_topology_object": True,
        "set_loaded_object": True,
        "update_structure_combo": True,
        "frame_after_build": 1,
        "ready_log_line": (
            f"ABFE trajectory object ready: {final_view_obj} "
            f"({len(list(picked or []))}/{n_sampled} sampled states)."
        ),
    }


def build_abfe_final_traj_post_build_ui_plan(
    *,
    final_view_object="",
    update_structure_combo=True,
    frame_after_build=1,
    n_selected=0,
    n_sampled=0,
    ready_log_line="",
):
    final_obj = str(final_view_object or "").strip()
    if not final_obj:
        return {
            "ok": True,
            "should_apply": False,
            "reason": "missing_final_view_object",
            "should_refresh_structure_combo": False,
            "structure_combo_text": "",
            "should_set_frame": False,
            "frame_index": 1,
            "should_log_ready": False,
            "ready_log_line": "",
        }

    try:
        frame_idx = int(frame_after_build or 1)
    except Exception:
        frame_idx = 1
    frame_idx = max(1, int(frame_idx))

    try:
        n_sel = int(n_selected or 0)
    except Exception:
        n_sel = 0
    n_sel = max(0, int(n_sel))

    try:
        n_total = int(n_sampled or 0)
    except Exception:
        n_total = 0
    n_total = max(0, int(n_total))

    line = str(ready_log_line or "").strip()
    if not line:
        line = f"ABFE trajectory object ready: {final_obj} ({n_sel}/{n_total} sampled states)."

    return {
        "ok": True,
        "should_apply": True,
        "reason": "",
        "should_refresh_structure_combo": bool(update_structure_combo),
        "structure_combo_text": str(final_obj),
        "should_set_frame": True,
        "frame_index": int(frame_idx),
        "should_log_ready": True,
        "ready_log_line": str(line),
    }


def build_abfe_post_sampling_metrics(
    *,
    frames=None,
    times_ps=None,
    lam_hist=None,
    bias_e=None,
    lj_e=None,
    coul_e=None,
    total_e=None,
    min_dist=None,
    reactive_dist=None,
    site_min_dist=None,
    site_com_dist=None,
    protein_contact_count_4a=None,
    lig_com=None,
    far_hold_stable_mask=None,
    detached_settle_mask=None,
    bound_site_com_threshold_a=4.0,
    bound_max_min_dist_a=3.5,
    bound_min_contact_count_4a=8,
    unbound_site_com_threshold_a=10.0,
    unbound_min_dist_a=3.5,
    unbound_max_contact_count_4a=4,
    bound_hold_max_lambda=0.30,
    unbound_hold_min_lambda=0.85,
    detached_settle_max_lambda=0.25,
    bound_hold_min_run_samples=4,
    unbound_hold_min_run_samples=4,
    bound_hold_max_gap_samples=0,
    unbound_hold_max_gap_samples=3,
    min_hold_state_frames=4,
    temperature_k=300.0,
    local_cutoff_nm=1.0,
):
    total_arr = _abfe_float_series(total_e)
    n_samples = int(total_arr.size)
    frames_arr = _abfe_int_series(frames, n=n_samples, start=1)
    times_arr = _abfe_float_series(times_ps, n=n_samples)
    lam_arr = _abfe_float_series(lam_hist, n=n_samples)
    lj_arr = _abfe_float_series(lj_e, n=n_samples)
    coul_arr = _abfe_float_series(coul_e, n=n_samples)
    min_arr = _abfe_float_series(min_dist, n=n_samples)
    reactive_arr = _abfe_float_series(reactive_dist, n=n_samples)
    site_min_arr = _abfe_float_series(site_min_dist, n=n_samples)
    site_com_arr = _abfe_float_series(site_com_dist, n=n_samples)
    contact_arr = _abfe_float_series(protein_contact_count_4a, n=n_samples)
    far_hold_arr = _abfe_bool_series(far_hold_stable_mask, n=n_samples, fill=False)
    detached_arr = _abfe_bool_series(detached_settle_mask, n=n_samples, fill=False)
    bias_arr = _abfe_float_series(bias_e, n=n_samples)

    valid = np.isfinite(total_arr)
    analysis_mask = np.ones_like(valid, dtype=bool)
    if len(valid) >= 40:
        burn_n = int(max(1, round(0.05 * len(valid))))
        analysis_mask[:burn_n] = False
    valid_fe = np.logical_and(valid, analysis_mask)
    if not np.any(valid_fe):
        valid_fe = np.array(valid, dtype=bool)

    dist_metric = "site_anchor_min"
    dist_arr = np.array(site_min_arr, dtype=float)
    if not np.any(np.isfinite(dist_arr)):
        if np.any(np.isfinite(site_com_arr)):
            dist_metric = "reaction_site_com"
            dist_arr = np.array(site_com_arr, dtype=float)
        else:
            dist_metric = "min_heavy_any_protein"
            dist_arr = np.array(min_arr, dtype=float)
    if dist_metric == "reaction_site_com":
        bound_cutoff_a = 8.0
    else:
        bound_cutoff_a = 4.5

    finite_min = np.logical_and(np.isfinite(dist_arr), analysis_mask)
    if not np.any(finite_min):
        finite_min = np.isfinite(dist_arr)
    fe_base = np.logical_and(valid, analysis_mask)
    has_site = np.any(np.isfinite(site_com_arr))
    has_min = np.any(np.isfinite(min_arr))
    if has_site and has_min:
        stable_unbound_phase_mask = None
        if far_hold_arr.size == site_com_arr.size:
            stable_unbound_phase_mask = np.asarray(far_hold_arr, dtype=bool)
        if detached_arr.size == site_com_arr.size:
            detached_phase_mask = np.asarray(detached_arr, dtype=bool)
            if stable_unbound_phase_mask is None:
                stable_unbound_phase_mask = detached_phase_mask
            else:
                stable_unbound_phase_mask = np.logical_or(stable_unbound_phase_mask, detached_phase_mask)
        bound_mask = np.logical_and.reduce(
            [
                fe_base,
                np.isfinite(site_com_arr),
                np.isfinite(min_arr),
                np.asarray(site_com_arr <= float(bound_site_com_threshold_a), dtype=bool),
                np.asarray(min_arr <= float(bound_max_min_dist_a), dtype=bool),
            ]
        )
        if np.any(np.isfinite(contact_arr)):
            bound_mask = np.logical_and(
                bound_mask,
                np.asarray(contact_arr >= float(bound_min_contact_count_4a), dtype=bool),
            )
        unbound_mask = np.logical_and.reduce(
            [
                fe_base,
                np.isfinite(site_com_arr),
                np.isfinite(min_arr),
                np.asarray(site_com_arr >= float(unbound_site_com_threshold_a), dtype=bool),
                np.asarray(min_arr >= float(unbound_min_dist_a), dtype=bool),
            ]
        )
        if np.any(np.isfinite(contact_arr)):
            unbound_mask = np.logical_and(
                unbound_mask,
                np.asarray(contact_arr <= float(unbound_max_contact_count_4a), dtype=bool),
            )
        if stable_unbound_phase_mask is not None and stable_unbound_phase_mask.size == unbound_mask.size:
            unbound_mask = np.logical_and(unbound_mask, stable_unbound_phase_mask)
    else:
        distance_state_mask = np.logical_and(finite_min, fe_base)
        bound_mask = distance_state_mask & (dist_arr <= bound_cutoff_a)
        unbound_mask = np.logical_and(distance_state_mask, np.logical_not(bound_mask))
    intermediate_mask = np.logical_and(fe_base, np.logical_not(np.logical_or(bound_mask, unbound_mask)))

    lam_has = bool(lam_arr.size == bound_mask.size and np.any(np.isfinite(lam_arr)))
    bound_hold_candidate = np.logical_and(np.asarray(bound_mask, dtype=bool), np.asarray(valid, dtype=bool))
    if lam_has:
        bound_hold_candidate = np.logical_and(
            bound_hold_candidate,
            np.asarray(lam_arr <= float(bound_hold_max_lambda), dtype=bool),
        )
    bound_hold_mask = _abfe_gap_tolerant_true_run_mask(
        bound_hold_candidate,
        min_run=int(bound_hold_min_run_samples),
        max_gap=int(bound_hold_max_gap_samples),
    )
    if not np.any(bound_hold_mask):
        bound_hold_mask = _abfe_gap_tolerant_true_run_mask(
            np.logical_and(np.asarray(bound_mask, dtype=bool), np.asarray(valid, dtype=bool)),
            min_run=max(2, int(bound_hold_min_run_samples)),
            max_gap=int(bound_hold_max_gap_samples),
        )

    unbound_geom_candidate = np.logical_and(np.asarray(unbound_mask, dtype=bool), np.asarray(valid, dtype=bool))
    unbound_hold_candidate = np.asarray(unbound_geom_candidate, dtype=bool)
    if lam_has:
        unbound_hold_candidate = np.logical_and(
            unbound_hold_candidate,
            np.asarray(lam_arr >= float(unbound_hold_min_lambda), dtype=bool),
        )
    detached_settle_used_for_unbound_hold = False
    detached_settle_sparse_fallback_used = False
    unbound_hold_mask = np.zeros_like(unbound_hold_candidate, dtype=bool)
    if detached_arr.size == unbound_hold_candidate.size and np.any(detached_arr):
        detached_candidate = np.logical_and(unbound_geom_candidate, np.asarray(detached_arr, dtype=bool))
        if lam_has:
            detached_candidate = np.logical_and(
                detached_candidate,
                np.asarray(lam_arr <= float(detached_settle_max_lambda), dtype=bool),
            )
        detached_min_run = int(max(2, min(int(unbound_hold_min_run_samples), 3)))
        detached_hold_mask = _abfe_gap_tolerant_true_run_mask(
            detached_candidate,
            min_run=detached_min_run,
            max_gap=int(unbound_hold_max_gap_samples),
        )
        min_detached_n = int(max(4, min_hold_state_frames))
        detached_candidate_n = int(np.sum(detached_candidate))
        detached_phase_n = int(np.sum(np.asarray(detached_arr, dtype=bool)))
        detached_sparse_min_n = int(max(min_detached_n, round(0.50 * max(1, detached_phase_n))))
        if int(np.sum(detached_hold_mask)) >= min_detached_n:
            unbound_hold_mask = np.asarray(detached_hold_mask, dtype=bool)
            detached_settle_used_for_unbound_hold = True
        elif detached_candidate_n >= detached_sparse_min_n:
            unbound_hold_mask = np.asarray(detached_candidate, dtype=bool)
            detached_settle_used_for_unbound_hold = True
            detached_settle_sparse_fallback_used = True
    if (not np.any(unbound_hold_mask)) and far_hold_arr.size == unbound_hold_candidate.size:
        unbound_hold_candidate = np.logical_and(unbound_hold_candidate, np.asarray(far_hold_arr, dtype=bool))
        unbound_hold_mask = _abfe_gap_tolerant_true_run_mask(
            unbound_hold_candidate,
            min_run=int(unbound_hold_min_run_samples),
            max_gap=int(unbound_hold_max_gap_samples),
        )
    sparse_unbound_hold_fallback_used = False
    if (not np.any(unbound_hold_mask)) and far_hold_arr.size == unbound_hold_candidate.size:
        sparse_candidate = np.asarray(unbound_hold_candidate, dtype=bool)
        sparse_n = int(np.sum(sparse_candidate))
        far_hold_n = int(np.sum(np.asarray(far_hold_arr, dtype=bool)))
        min_sparse_n = int(max(1, min_hold_state_frames))
        min_sparse_frac_n = int(max(2, round(0.40 * max(1, far_hold_n))))
        if sparse_n >= min_sparse_n and sparse_n >= min_sparse_frac_n:
            unbound_hold_mask = np.asarray(sparse_candidate, dtype=bool)
            sparse_unbound_hold_fallback_used = True

    n_bound_raw = int(np.sum(bound_mask))
    n_unbound_raw = int(np.sum(unbound_mask))
    n_bound_hold = int(np.sum(bound_hold_mask))
    n_unbound_hold = int(np.sum(unbound_hold_mask))
    state_coverage = build_abfe_state_coverage_diagnostic_payload(
        site_com_dist=np.asarray(site_com_arr, dtype=float),
        min_dist=np.asarray(min_arr, dtype=float),
        contact_count_4a=np.asarray(contact_arr, dtype=float),
        bound_mask=np.asarray(bound_mask, dtype=bool),
        unbound_mask=np.asarray(unbound_mask, dtype=bool),
        bound_hold_mask=np.asarray(bound_hold_mask, dtype=bool),
        unbound_hold_mask=np.asarray(unbound_hold_mask, dtype=bool),
        far_hold_stable_mask=np.asarray(far_hold_arr, dtype=bool),
        detached_settle_mask=np.asarray(detached_arr, dtype=bool),
        unbound_site_com_threshold_a=float(unbound_site_com_threshold_a),
        unbound_min_dist_a=float(unbound_min_dist_a),
        unbound_max_contact_count_4a=int(unbound_max_contact_count_4a),
        min_hold_state_frames=int(min_hold_state_frames),
    )
    use_hold_windows = bool(
        n_bound_hold >= int(max(1, min_hold_state_frames))
        and n_unbound_hold >= int(max(1, min_hold_state_frames))
    )
    estimator_window_mode = "hold_only" if use_hold_windows else "state_fallback"
    estimator_bound_mask = np.asarray(bound_hold_mask if use_hold_windows else bound_mask, dtype=bool)
    estimator_unbound_mask = np.asarray(unbound_hold_mask if use_hold_windows else unbound_mask, dtype=bool)

    valid_fe = np.logical_or(estimator_bound_mask, estimator_unbound_mask)
    n_dist_valid = int(np.sum(valid_fe))
    n_bound = int(np.sum(estimator_bound_mask))
    n_unbound = int(np.sum(estimator_unbound_mask))
    if n_dist_valid > 0:
        p_bound = float(n_bound / float(n_dist_valid))
        p_unbound = float(n_unbound / float(n_dist_valid))
    else:
        p_bound = np.nan
        p_unbound = np.nan

    k_b = 0.008314462618  # kJ/mol/K
    kbt = k_b * float(temperature_k)
    dg_pop = np.nan
    if n_dist_valid > 0 and n_bound > 0 and n_unbound > 0:
        dg_pop = float(-kbt * np.log(p_bound / p_unbound))

    if n_dist_valid > 0:
        p_bound_jeff = float((n_bound + 0.5) / (n_dist_valid + 1.0))
        p_unbound_jeff = float((n_unbound + 0.5) / (n_dist_valid + 1.0))
        dg_pop_jeff = float(-kbt * np.log(p_bound_jeff / p_unbound_jeff))
    else:
        p_bound_jeff = np.nan
        p_unbound_jeff = np.nan
        dg_pop_jeff = np.nan

    try:
        com_raw = np.asarray(list(lig_com) if lig_com is not None else [], dtype=float)
    except Exception:
        com_raw = np.asarray([], dtype=float)
    com_arr = np.full((int(n_samples), 3), np.nan, dtype=float)
    if com_raw.ndim == 2 and com_raw.shape[1] == 3 and n_samples > 0:
        copy_n = min(int(n_samples), int(com_raw.shape[0]))
        if copy_n > 0:
            com_arr[:copy_n, :] = np.asarray(com_raw[:copy_n, :], dtype=float)
    com_finite = np.all(np.isfinite(com_arr), axis=1)
    # The standard-state correction needs the volume the ligand explores while
    # unbound.  A box over bound + unbound frames spans the whole exit path
    # and inflates V_eff (more favourable DeltaG standard).
    com_unbound = np.asarray(com_finite, dtype=bool)
    if com_unbound.size == estimator_unbound_mask.size:
        com_unbound = np.logical_and(com_unbound, np.asarray(estimator_unbound_mask, dtype=bool))
    if int(np.sum(com_unbound)) >= 2:
        com_finite = com_unbound
    elif com_finite.size == valid_fe.size:
        com_finite = np.logical_and(com_finite, np.asarray(valid_fe, dtype=bool))
    com_eval = np.asarray(com_arr[com_finite], dtype=float)
    if com_eval.shape[0] >= 2:
        mins = np.nanmin(com_eval, axis=0)
        maxs = np.nanmax(com_eval, axis=0)
        span = np.maximum(maxs - mins, 0.2)
        v_eff_nm3 = float(np.prod(span))
    else:
        try:
            local_cutoff_nm_f = float(local_cutoff_nm)
        except Exception:
            local_cutoff_nm_f = 1.0
        v_eff_nm3 = float((4.0 / 3.0) * np.pi * (local_cutoff_nm_f ** 3))
    c_eff_m = float(1.66054 / max(v_eff_nm3, 1e-6))
    pop_standard = _abfe_standard_state_from_state_ratio(
        dg_state_kj=float(dg_pop),
        v_eff_nm3=float(v_eff_nm3),
        temperature_k=float(temperature_k),
    )
    pop_jeff_standard = _abfe_standard_state_from_state_ratio(
        dg_state_kj=float(dg_pop_jeff),
        v_eff_nm3=float(v_eff_nm3),
        temperature_k=float(temperature_k),
    )

    kd_app_m = np.nan
    p_kd_app = np.nan
    if bool(pop_standard.get("ok", False)) and n_dist_valid > 0 and n_bound > 0 and n_unbound > 0:
        kd_app_m = float(pop_standard.get("kd_standard_m", np.nan))
        p_kd_app = float(pop_standard.get("p_kd_standard", np.nan))

    kd_app_jeff_m = np.nan
    p_kd_jeff = np.nan
    if bool(pop_jeff_standard.get("ok", False)) and np.isfinite(p_bound_jeff) and np.isfinite(p_unbound_jeff) and p_bound_jeff > 0.0:
        kd_app_jeff_m = float(pop_jeff_standard.get("kd_standard_m", np.nan))
        p_kd_jeff = float(pop_jeff_standard.get("p_kd_standard", np.nan))

    bound_valid_mask = np.asarray(estimator_bound_mask[valid_fe], dtype=bool)
    unbound_valid_mask = np.asarray(estimator_unbound_mask[valid_fe], dtype=bool)
    bias_valid = np.asarray(bias_arr[valid_fe], dtype=float)
    reweight_payload = build_abfe_bias_reweighted_population_payload(
        bound_valid_mask=np.asarray(bound_valid_mask, dtype=bool),
        unbound_valid_mask=np.asarray(unbound_valid_mask, dtype=bool),
        bias_valid=np.asarray(bias_valid, dtype=float),
        temperature_k=float(temperature_k),
        v_eff_nm3=float(v_eff_nm3),
        n_boot=0,
    )

    return {
        "ok": True,
        "frames_arr": frames_arr,
        "times_arr": times_arr,
        "lam_arr": lam_arr,
        "bias_arr": bias_arr,
        "lj_arr": lj_arr,
        "coul_arr": coul_arr,
        "total_arr": total_arr,
        "min_arr": min_arr,
        "reactive_arr": reactive_arr,
        "site_min_arr": site_min_arr,
        "site_com_arr": site_com_arr,
        "contact_count_arr": contact_arr,
        "valid_mask": valid,
        "analysis_mask": analysis_mask,
        "bound_mask": np.asarray(bound_mask, dtype=bool),
        "unbound_mask": np.asarray(unbound_mask, dtype=bool),
        "bound_hold_mask": np.asarray(bound_hold_mask, dtype=bool),
        "unbound_hold_mask": np.asarray(unbound_hold_mask, dtype=bool),
        "unbound_hold_sparse_fallback_used": bool(sparse_unbound_hold_fallback_used),
        "detached_settle_mask": np.asarray(detached_arr, dtype=bool),
        "detached_settle_used_for_unbound_hold": bool(detached_settle_used_for_unbound_hold),
        "detached_settle_sparse_fallback_used": bool(detached_settle_sparse_fallback_used),
        "estimator_bound_mask": np.asarray(estimator_bound_mask, dtype=bool),
        "estimator_unbound_mask": np.asarray(estimator_unbound_mask, dtype=bool),
        "intermediate_mask": np.asarray(intermediate_mask, dtype=bool),
        "state_coverage_diagnostic": dict(state_coverage or {}),
        "binding_estimate_allowed": bool(state_coverage.get("binding_estimate_allowed", False)),
        "binding_estimate_status": str(state_coverage.get("status", "")),
        "binding_estimate_note": str(state_coverage.get("note", "")),
        "valid_mask_fe": valid_fe,
        "frames_valid": frames_arr[valid_fe],
        "bias_valid": np.asarray(bias_valid, dtype=float),
        "total_valid": total_arr[valid_fe],
        "lj_valid": lj_arr[valid_fe],
        "coul_valid": coul_arr[valid_fe],
        "min_valid": min_arr[valid_fe],
        "distance_valid": dist_arr[valid_fe],
        "bound_valid_mask": np.asarray(bound_valid_mask, dtype=bool),
        "unbound_valid_mask": np.asarray(unbound_valid_mask, dtype=bool),
        "distance_metric": str(dist_metric),
        "dist_arr": np.array(dist_arr, dtype=float),
        "bound_cutoff_a": float(bound_cutoff_a),
        "n_bound_raw": int(n_bound_raw),
        "n_unbound_raw": int(n_unbound_raw),
        "n_bound_hold": int(n_bound_hold),
        "n_unbound_hold": int(n_unbound_hold),
        "estimator_window_mode": str(estimator_window_mode),
        "estimator_uses_hold_windows": bool(use_hold_windows),
        "n_dist_valid": int(n_dist_valid),
        "n_bound": int(n_bound),
        "n_unbound": int(n_unbound),
        "p_bound": float(p_bound) if np.isfinite(p_bound) else np.nan,
        "p_unbound": float(p_unbound) if np.isfinite(p_unbound) else np.nan,
        "dg_pop_kj": float(dg_pop) if np.isfinite(dg_pop) else np.nan,
        "dg_pop_state_kj": float(dg_pop) if np.isfinite(dg_pop) else np.nan,
        "dg_pop_standard_kj": float(pop_standard.get("dg_standard_kj", np.nan)),
        "standard_state_correction_kj": float(pop_standard.get("standard_state_correction_kj", np.nan)),
        "p_bound_jeff": float(p_bound_jeff) if np.isfinite(p_bound_jeff) else np.nan,
        "p_unbound_jeff": float(p_unbound_jeff) if np.isfinite(p_unbound_jeff) else np.nan,
        "dg_pop_jeff_kj": float(dg_pop_jeff) if np.isfinite(dg_pop_jeff) else np.nan,
        "dg_pop_jeff_standard_kj": float(pop_jeff_standard.get("dg_standard_kj", np.nan)),
        "v_eff_nm3": float(v_eff_nm3),
        "c_eff_m": float(c_eff_m),
        "kd_app_m": float(kd_app_m) if np.isfinite(kd_app_m) else np.nan,
        "p_kd_app": float(p_kd_app) if np.isfinite(p_kd_app) else np.nan,
        "kd_app_jeff_m": float(kd_app_jeff_m) if np.isfinite(kd_app_jeff_m) else np.nan,
        "p_kd_jeff": float(p_kd_jeff) if np.isfinite(p_kd_jeff) else np.nan,
        "p_bound_reweighted": float(reweight_payload.get("p_bound_reweighted", np.nan)),
        "p_unbound_reweighted": float(reweight_payload.get("p_unbound_reweighted", np.nan)),
        "dg_reweighted_kj": float(reweight_payload.get("dg_reweighted_kj", np.nan)),
        "dg_reweighted_state_kj": float(reweight_payload.get("dg_reweighted_state_kj", np.nan)),
        "dg_reweighted_standard_kj": float(reweight_payload.get("dg_reweighted_standard_kj", np.nan)),
        "standard_state_correction_reweighted_kj": float(reweight_payload.get("standard_state_correction_kj", np.nan)),
        "kd_app_reweighted_m": float(reweight_payload.get("kd_app_reweighted_m", np.nan)),
        "kd_standard_reweighted_m": float(reweight_payload.get("kd_standard_reweighted_m", np.nan)),
        "p_kd_reweighted": float(reweight_payload.get("p_kd_reweighted", np.nan)),
        "reweight_ess_total": float(reweight_payload.get("ess_total", np.nan)),
        "reweight_ess_bound": float(reweight_payload.get("ess_bound", np.nan)),
        "reweight_ess_unbound": float(reweight_payload.get("ess_unbound", np.nan)),
    }


def build_abfe_local_md_result_payload(
    *,
    frames_arr=None,
    times_arr=None,
    lam_arr=None,
    bias_arr=None,
    total_arr=None,
    lj_arr=None,
    coul_arr=None,
    min_arr=None,
    reactive_arr=None,
    site_min_arr=None,
    site_com_arr=None,
    dist_arr=None,
    dist_metric="",
    valid=None,
    analysis_mask=None,
    valid_fe=None,
    frames_valid=None,
    bias_valid=None,
    total_valid=None,
    rescored_total_valid=None,
    ligand_internal_valid=None,
    lj_valid=None,
    coul_valid=None,
    min_valid=None,
    distance_valid=None,
    bound_valid_mask=None,
    unbound_valid_mask=None,
    platform_name="",
    sampling_mode="",
    steering_mode="",
    policy_mode="heuristic",
    use_whole_requested=False,
    local_cutoff_a=np.nan,
    protein_atoms_total=0,
    protein_atoms_used=0,
    ligand_atoms_used=0,
    bias_enabled=False,
    guide_tunnel_enabled=False,
    guide_tunnel_active=False,
    guide_tunnel_radius_a=np.nan,
    guide_tunnel_k_kj_nm2=np.nan,
    guide_tunnel_axis=None,
    guide_tunnel_origin_nm=None,
    pull_distance_a=np.nan,
    pull_k_kj_nm2=np.nan,
    pull_k_effective_kj_nm2=np.nan,
    pull_clearance_nm=np.nan,
    pull_collision_penalty=np.nan,
    bound_cutoff_a=np.nan,
    n_dist_valid=0,
    n_bound=0,
    n_unbound=0,
    n_bound_raw=0,
    n_unbound_raw=0,
    n_bound_hold=0,
    n_unbound_hold=0,
    estimator_window_mode="state_fallback",
    binding_estimate_status="",
    binding_estimate_note="",
    binding_estimate_allowed=False,
    p_bound=np.nan,
    p_unbound=np.nan,
    dg_pop=np.nan,
    p_bound_jeff=np.nan,
    p_unbound_jeff=np.nan,
    dg_pop_jeff=np.nan,
    kd_app_m=np.nan,
    p_kd_app=np.nan,
    kd_app_jeff_m=np.nan,
    p_kd_jeff=np.nan,
    p_bound_reweighted=np.nan,
    p_unbound_reweighted=np.nan,
    dg_reweighted=np.nan,
    dg_reweighted_state=np.nan,
    dg_reweighted_standard=np.nan,
    standard_state_correction_kj=np.nan,
    c_eff_m=np.nan,
    kd_app_reweighted_m=np.nan,
    p_kd_reweighted=np.nan,
    reweight_ess_total=np.nan,
    reweight_ess_bound=np.nan,
    reweight_ess_unbound=np.nan,
    reweight_confidence_quality="low",
    reweight_confidence_note="",
    reweight_confidence_ok=False,
    dg_selected=np.nan,
    kd_selected_m=np.nan,
    p_kd_selected=np.nan,
    selected_estimator="gated_off",
    dg_selected_state=np.nan,
    dg_selected_standard=np.nan,
    estimate_kind="",
    production_reliability="",
    production_note="",
    rescored_energy_label="",
    dg_rescored_state_expavg_kj=np.nan,
    dg_rescored_state_blockexpavg_kj=np.nan,
    dg_rescored_state_expavg_strain_kj=np.nan,
    dg_rescored_state_blockexpavg_strain_kj=np.nan,
    dg_rescored_state_mean_kj=np.nan,
    rescored_bound_expavg_kj=np.nan,
    rescored_unbound_expavg_kj=np.nan,
    rescored_bound_blockexpavg_kj=np.nan,
    rescored_unbound_blockexpavg_kj=np.nan,
    rescored_bound_expavg_strain_kj=np.nan,
    rescored_unbound_expavg_strain_kj=np.nan,
    rescored_bound_blockexpavg_strain_kj=np.nan,
    rescored_unbound_blockexpavg_strain_kj=np.nan,
    rescored_bound_mean_kj=np.nan,
    rescored_unbound_mean_kj=np.nan,
    dg_ligand_strain_kj=np.nan,
    v_eff_nm3=np.nan,
    sim_ns=np.nan,
    sample_every_ps=np.nan,
    memory_guard_active=False,
    coordinate_memory_budget_mb=np.nan,
    coordinate_states_saved=0,
    coordinate_states_possible=0,
    estimated_all_states_mb=np.nan,
    estimated_saved_states_mb=np.nan,
    rescore_max_frames=0,
    ligand_charge_source="",
    ligand_ff_mode="",
    ligand_xml_source="",
    ligand_target_ph=np.nan,
    protein_gap_count=0,
    abfe_traj_object="",
    abfe_context_object="",
    target_hint="",
    policy_summary=None,
    policy_trace=None,
    pocket_desc=None,
    random_seed=0,
    replica_index=1,
    replica_count=1,
):
    pull_k_eff = float(pull_k_effective_kj_nm2)
    pull_clear_nm = float(pull_clearance_nm)
    pull_collision = float(pull_collision_penalty)
    result = {
        "frames": np.asarray(frames_arr if frames_arr is not None else [], dtype=int),
        "times_ps": np.asarray(times_arr if times_arr is not None else [], dtype=float),
        "steering_lambda": np.asarray(lam_arr if lam_arr is not None else [], dtype=float),
        "bias_energy_kj": np.asarray(bias_arr if bias_arr is not None else [], dtype=float),
        "total_energy_kj": np.asarray(total_arr if total_arr is not None else [], dtype=float),
        "lj_energy_kj": np.asarray(lj_arr if lj_arr is not None else [], dtype=float),
        "coul_energy_kj": np.asarray(coul_arr if coul_arr is not None else [], dtype=float),
        "min_dist_a": np.asarray(min_arr if min_arr is not None else [], dtype=float),
        "reactive_dist_a": np.asarray(reactive_arr if reactive_arr is not None else [], dtype=float),
        "site_min_dist_a": np.asarray(site_min_arr if site_min_arr is not None else [], dtype=float),
        "site_com_dist_a": np.asarray(site_com_arr if site_com_arr is not None else [], dtype=float),
        "distance_a": np.asarray(dist_arr if dist_arr is not None else [], dtype=float),
        "distance_metric": str(dist_metric),
        "valid_mask": np.asarray(valid if valid is not None else [], dtype=bool),
        "analysis_mask": np.asarray(analysis_mask if analysis_mask is not None else [], dtype=bool),
        "valid_mask_fe": np.asarray(valid_fe if valid_fe is not None else [], dtype=bool),
        "frames_valid": np.asarray(frames_valid if frames_valid is not None else [], dtype=int),
        "bias_valid": np.asarray(bias_valid if bias_valid is not None else [], dtype=float),
        "total_valid": np.asarray(total_valid if total_valid is not None else [], dtype=float),
        "rescored_total_valid": np.asarray(rescored_total_valid if rescored_total_valid is not None else [], dtype=float),
        "ligand_internal_valid": np.asarray(ligand_internal_valid if ligand_internal_valid is not None else [], dtype=float),
        "lj_valid": np.asarray(lj_valid if lj_valid is not None else [], dtype=float),
        "coul_valid": np.asarray(coul_valid if coul_valid is not None else [], dtype=float),
        "min_valid": np.asarray(min_valid if min_valid is not None else [], dtype=float),
        "distance_valid": np.asarray(distance_valid if distance_valid is not None else [], dtype=float),
        "bound_valid_mask": np.asarray(bound_valid_mask if bound_valid_mask is not None else [], dtype=bool),
        "unbound_valid_mask": np.asarray(unbound_valid_mask if unbound_valid_mask is not None else [], dtype=bool),
        "energy_mode": f"openmm ({platform_name})",
        "protocol": "abfe_lite_local_md",
        "protocol_detail": "staged_minimize_equilibrate_pull",
        "sampling_mode": str(sampling_mode),
        "steering_mode": str(steering_mode),
        "policy_mode": str(_sanitize_abfe_steering_policy_mode(policy_mode)),
        "local_mode_requested": bool(not use_whole_requested),
        "local_mode_active": False,
        "whole_protein_mode": True,
        "local_cutoff_a": float(local_cutoff_a),
        "protein_atoms_total": int(protein_atoms_total),
        "protein_atoms_used": int(protein_atoms_used),
        "ligand_atoms_used": int(ligand_atoms_used),
        "bias_enabled": bool(bias_enabled),
        "guide_tunnel_enabled": bool(guide_tunnel_enabled),
        "guide_tunnel_active": bool(guide_tunnel_active),
        "guide_tunnel_radius_a": (
            float(guide_tunnel_radius_a) if np.isfinite(float(guide_tunnel_radius_a)) else np.nan
        ),
        "guide_tunnel_k_kj_nm2": (
            float(guide_tunnel_k_kj_nm2) if np.isfinite(float(guide_tunnel_k_kj_nm2)) else np.nan
        ),
        "guide_tunnel_axis": np.asarray(
            guide_tunnel_axis if guide_tunnel_axis is not None else [np.nan, np.nan, np.nan],
            dtype=float,
        ),
        "guide_tunnel_origin_nm": np.asarray(
            guide_tunnel_origin_nm if guide_tunnel_origin_nm is not None else [np.nan, np.nan, np.nan],
            dtype=float,
        ),
        "pull_distance_a": float(pull_distance_a),
        "pull_k_kj_nm2": float(pull_k_kj_nm2),
        "pull_k_effective_kj_nm2": (float(pull_k_eff) if np.isfinite(pull_k_eff) else np.nan),
        "pull_clearance_a": (float(pull_clear_nm * 10.0) if np.isfinite(pull_clear_nm) else np.nan),
        "pull_collision_penalty": (float(pull_collision) if np.isfinite(pull_collision) else np.nan),
        "bound_cutoff_a": float(bound_cutoff_a),
        "n_dist_valid": int(n_dist_valid),
        "n_bound": int(n_bound),
        "n_unbound": int(n_unbound),
        "n_bound_raw": int(n_bound_raw),
        "n_unbound_raw": int(n_unbound_raw),
        "n_bound_hold": int(n_bound_hold),
        "n_unbound_hold": int(n_unbound_hold),
        "estimator_window_mode": str(estimator_window_mode or "state_fallback"),
        "binding_estimate_status": str(binding_estimate_status or ""),
        "binding_estimate_note": str(binding_estimate_note or ""),
        "binding_estimate_allowed": bool(binding_estimate_allowed),
        "p_bound": float(p_bound) if np.isfinite(float(p_bound)) else np.nan,
        "p_unbound": float(p_unbound) if np.isfinite(float(p_unbound)) else np.nan,
        "dg_pop_kj": float(dg_pop) if np.isfinite(float(dg_pop)) else np.nan,
        "p_bound_jeff": float(p_bound_jeff) if np.isfinite(float(p_bound_jeff)) else np.nan,
        "p_unbound_jeff": float(p_unbound_jeff) if np.isfinite(float(p_unbound_jeff)) else np.nan,
        "dg_pop_jeff_kj": float(dg_pop_jeff) if np.isfinite(float(dg_pop_jeff)) else np.nan,
        "kd_app_m": float(kd_app_m) if np.isfinite(float(kd_app_m)) else np.nan,
        "p_kd_app": float(p_kd_app) if np.isfinite(float(p_kd_app)) else np.nan,
        "kd_app_jeff_m": float(kd_app_jeff_m) if np.isfinite(float(kd_app_jeff_m)) else np.nan,
        "p_kd_jeff": float(p_kd_jeff) if np.isfinite(float(p_kd_jeff)) else np.nan,
        "p_bound_reweighted": float(p_bound_reweighted) if np.isfinite(float(p_bound_reweighted)) else np.nan,
        "p_unbound_reweighted": float(p_unbound_reweighted) if np.isfinite(float(p_unbound_reweighted)) else np.nan,
        "dg_reweighted_kj": float(dg_reweighted) if np.isfinite(float(dg_reweighted)) else np.nan,
        "dg_reweighted_state_kj": float(dg_reweighted_state) if np.isfinite(float(dg_reweighted_state)) else np.nan,
        "dg_reweighted_standard_kj": float(dg_reweighted_standard) if np.isfinite(float(dg_reweighted_standard)) else np.nan,
        "standard_state_correction_kj": float(standard_state_correction_kj) if np.isfinite(float(standard_state_correction_kj)) else np.nan,
        "c_eff_m": float(c_eff_m) if np.isfinite(float(c_eff_m)) else np.nan,
        "kd_app_reweighted_m": float(kd_app_reweighted_m) if np.isfinite(float(kd_app_reweighted_m)) else np.nan,
        "p_kd_reweighted": float(p_kd_reweighted) if np.isfinite(float(p_kd_reweighted)) else np.nan,
        "reweight_ess_total": float(reweight_ess_total) if np.isfinite(float(reweight_ess_total)) else np.nan,
        "reweight_ess_bound": float(reweight_ess_bound) if np.isfinite(float(reweight_ess_bound)) else np.nan,
        "reweight_ess_unbound": float(reweight_ess_unbound) if np.isfinite(float(reweight_ess_unbound)) else np.nan,
        "reweight_confidence_quality": str(reweight_confidence_quality or "low"),
        "reweight_confidence_note": str(reweight_confidence_note or ""),
        "reweight_confidence_ok": bool(reweight_confidence_ok),
        "dg_selected_kj": float(dg_selected) if np.isfinite(float(dg_selected)) else np.nan,
        "dg_selected_state_kj": float(dg_selected_state) if np.isfinite(float(dg_selected_state)) else np.nan,
        "dg_selected_standard_kj": float(dg_selected_standard) if np.isfinite(float(dg_selected_standard)) else np.nan,
        "kd_selected_m": float(kd_selected_m) if np.isfinite(float(kd_selected_m)) else np.nan,
        "p_kd_selected": float(p_kd_selected) if np.isfinite(float(p_kd_selected)) else np.nan,
        "selected_estimator": str(selected_estimator or "gated_off"),
        "estimate_kind": str(estimate_kind or ""),
        "production_reliability": str(production_reliability or ""),
        "production_note": str(production_note or ""),
        "rescored_energy_label": str(rescored_energy_label or ""),
        "dg_rescored_state_expavg_kj": float(dg_rescored_state_expavg_kj) if np.isfinite(float(dg_rescored_state_expavg_kj)) else np.nan,
        "dg_rescored_state_blockexpavg_kj": float(dg_rescored_state_blockexpavg_kj) if np.isfinite(float(dg_rescored_state_blockexpavg_kj)) else np.nan,
        "dg_rescored_state_expavg_strain_kj": float(dg_rescored_state_expavg_strain_kj) if np.isfinite(float(dg_rescored_state_expavg_strain_kj)) else np.nan,
        "dg_rescored_state_blockexpavg_strain_kj": float(dg_rescored_state_blockexpavg_strain_kj) if np.isfinite(float(dg_rescored_state_blockexpavg_strain_kj)) else np.nan,
        "dg_rescored_state_mean_kj": float(dg_rescored_state_mean_kj) if np.isfinite(float(dg_rescored_state_mean_kj)) else np.nan,
        "rescored_bound_expavg_kj": float(rescored_bound_expavg_kj) if np.isfinite(float(rescored_bound_expavg_kj)) else np.nan,
        "rescored_unbound_expavg_kj": float(rescored_unbound_expavg_kj) if np.isfinite(float(rescored_unbound_expavg_kj)) else np.nan,
        "rescored_bound_blockexpavg_kj": float(rescored_bound_blockexpavg_kj) if np.isfinite(float(rescored_bound_blockexpavg_kj)) else np.nan,
        "rescored_unbound_blockexpavg_kj": float(rescored_unbound_blockexpavg_kj) if np.isfinite(float(rescored_unbound_blockexpavg_kj)) else np.nan,
        "rescored_bound_expavg_strain_kj": float(rescored_bound_expavg_strain_kj) if np.isfinite(float(rescored_bound_expavg_strain_kj)) else np.nan,
        "rescored_unbound_expavg_strain_kj": float(rescored_unbound_expavg_strain_kj) if np.isfinite(float(rescored_unbound_expavg_strain_kj)) else np.nan,
        "rescored_bound_blockexpavg_strain_kj": float(rescored_bound_blockexpavg_strain_kj) if np.isfinite(float(rescored_bound_blockexpavg_strain_kj)) else np.nan,
        "rescored_unbound_blockexpavg_strain_kj": float(rescored_unbound_blockexpavg_strain_kj) if np.isfinite(float(rescored_unbound_blockexpavg_strain_kj)) else np.nan,
        "rescored_bound_mean_kj": float(rescored_bound_mean_kj) if np.isfinite(float(rescored_bound_mean_kj)) else np.nan,
        "rescored_unbound_mean_kj": float(rescored_unbound_mean_kj) if np.isfinite(float(rescored_unbound_mean_kj)) else np.nan,
        "dg_ligand_strain_kj": float(dg_ligand_strain_kj) if np.isfinite(float(dg_ligand_strain_kj)) else np.nan,
        "v_eff_nm3": float(v_eff_nm3),
        "sim_ns": float(sim_ns),
        "sample_every_ps": float(sample_every_ps),
        "memory_guard_active": bool(memory_guard_active),
        "coordinate_memory_budget_mb": (
            float(_safe_float(coordinate_memory_budget_mb)) if np.isfinite(_safe_float(coordinate_memory_budget_mb)) else np.nan
        ),
        "coordinate_states_saved": int(max(0, int(coordinate_states_saved or 0))),
        "coordinate_states_possible": int(max(0, int(coordinate_states_possible or 0))),
        "estimated_all_states_mb": (
            float(_safe_float(estimated_all_states_mb)) if np.isfinite(_safe_float(estimated_all_states_mb)) else np.nan
        ),
        "estimated_saved_states_mb": (
            float(_safe_float(estimated_saved_states_mb)) if np.isfinite(_safe_float(estimated_saved_states_mb)) else np.nan
        ),
        "rescore_max_frames": int(max(0, int(rescore_max_frames or 0))),
        "ligand_charge_source": str(ligand_charge_source),
        "ligand_ff_mode": str(ligand_ff_mode),
        "ligand_xml_source": str(ligand_xml_source),
        "ligand_target_ph": float(ligand_target_ph) if np.isfinite(float(ligand_target_ph)) else np.nan,
        "protein_gap_count": int(protein_gap_count),
        "abfe_traj_object": str(abfe_traj_object),
        "abfe_context_object": str(abfe_context_object),
        "target_hint": str(target_hint),
        "policy_summary": dict(policy_summary or {}),
        "policy_trace": list(policy_trace or []),
        "random_seed": int(random_seed or 0),
        "replica_index": int(max(1, int(replica_index or 1))),
        "replica_count": int(max(1, int(replica_count or 1))),
    }
    pdesc = dict(pocket_desc or {})
    for k in ABFE_POCKET_DESCRIPTOR_COLS:
        result[k] = float(pdesc.get(k, np.nan))
    return result


def abfe_distance_axis_label(distance_metric):
    metric = str(distance_metric or "").strip()
    dist_label_map = {
        "reaction_site_com": "Reactive Distance to Initial Pocket (Å)",
        "site_anchor_min": "Min Distance to Initial Pocket Anchor (Å)",
        "min_heavy_any_protein": "Min Heavy-Atom Distance (Å)",
    }
    return str(dist_label_map.get(metric, "Distance (Å)"))


def build_abfe_free_energy_report_payload(
    *,
    protocol="",
    whole_protein_mode=False,
    local_active=False,
    prot_used=0,
    prot_total=0,
    local_cutoff_a=np.nan,
    lig_chain="",
    lig_resi="",
    lig_resn="",
    energy_mode="",
    dg_exp=np.nan,
    ci_low=np.nan,
    ci_high=np.nan,
    dg_cum=np.nan,
    n_samples=0,
    n_dist_valid=0,
    n_bound=0,
    n_unbound=0,
    p_bound=np.nan,
    p_unbound=np.nan,
    dg_pop=np.nan,
    dg_pop_jeff=np.nan,
    p_bound_reweighted=np.nan,
    p_unbound_reweighted=np.nan,
    dg_reweighted=np.nan,
    dg_reweighted_state=np.nan,
    dg_reweighted_standard=np.nan,
    standard_state_correction_kj=np.nan,
    c_eff_m=np.nan,
    kd_app_m=np.nan,
    p_kd_app=np.nan,
    kd_app_jeff_m=np.nan,
    p_kd_jeff=np.nan,
    kd_app_reweighted_m=np.nan,
    p_kd_reweighted=np.nan,
    reweight_ci_low=np.nan,
    reweight_ci_high=np.nan,
    reweight_ess_total=np.nan,
    reweight_confidence_quality="low",
    reweight_confidence_note="",
    reweight_confidence_ok=False,
    dg_selected=np.nan,
    dg_selected_state=np.nan,
    dg_selected_standard=np.nan,
    kd_selected_m=np.nan,
    p_kd_selected=np.nan,
    selected_estimator="gated_off",
    estimate_kind="",
    production_reliability="",
    production_note="",
    rescored_energy_label="",
    dg_rescored_state_expavg_kj=np.nan,
    dg_rescored_state_mean_kj=np.nan,
    estimator_window_mode="state_fallback",
    n_bound_raw=0,
    n_unbound_raw=0,
    n_bound_hold=0,
    n_unbound_hold=0,
    binding_estimate_status="",
    binding_estimate_note="",
    sim_ns=np.nan,
    sample_every_ps=np.nan,
    bias_enabled=False,
    sampling_mode="",
    pull_distance_a=np.nan,
    pull_k_kj_nm2=np.nan,
    pull_k_effective_kj_nm2=np.nan,
    pull_clearance_a=np.nan,
    bound_cutoff_a=np.nan,
    ligand_ff_mode="",
    ligand_charge_source="",
    ligand_target_ph=np.nan,
    target_hint="",
    distance_metric="",
    replica_summary=None,
):
    protocol_txt = str(protocol or "")
    if protocol_txt == "abfe_lite_local_md":
        intro = "ABFE-lite whole-protein estimate from interaction-energy reweighting."
    else:
        intro = "Trajectory free-energy estimate from interaction-energy reweighting (fallback mode)."

    if protocol_txt == "abfe_lite_local_md" and bool(whole_protein_mode):
        proto_label = "ABFE-lite whole-protein MD"
    elif protocol_txt == "abfe_lite_local_md":
        proto_label = "ABFE-lite MD"
    else:
        proto_label = "Trajectory Rescoring"

    if bool(whole_protein_mode):
        scope_label_plot = f"ABFE-lite Whole-Protein ({int(prot_used)}/{int(prot_total)} protein atoms)"
    else:
        if bool(local_active):
            scope_label_plot = (
                f"ABFE-lite Local ({int(prot_used)}/{int(prot_total)} protein atoms, {float(local_cutoff_a):.1f} Å)"
            )
        else:
            scope_label_plot = "Global"

    suptitle = (
        f"{proto_label} Free-Energy Estimate [{str(energy_mode)}] | {scope_label_plot} | "
        f"{str(lig_resn)} {str(lig_chain)}{str(lig_resi)} | "
        f"ΔGexp={float(dg_exp):.2f} kJ/mol (95% CI {float(ci_low):.2f}, {float(ci_high):.2f})"
    )

    state_line = "Population model: no valid distance samples."
    prob_line = ""
    kd_line = ""
    reweight_line = ""
    selected_line = ""
    rescored_line = ""
    if int(n_dist_valid) > 0:
        state_line = (
            f"Bound frames: {int(n_bound)}/{int(n_dist_valid)} | Unbound frames: {int(n_unbound)}/{int(n_dist_valid)}"
        )
        if int(n_bound_raw) > 0 or int(n_unbound_raw) > 0:
            state_line += f" | raw bound/unbound={int(n_bound_raw)}/{int(n_unbound_raw)}"
        if int(n_bound_hold) > 0 or int(n_unbound_hold) > 0:
            state_line += f" | hold windows={int(n_bound_hold)}/{int(n_unbound_hold)}"
        prob_line = f"P(bound)={float(p_bound):.4f} | P(unbound)={float(p_unbound):.4f}"
        dg_state_for_report = _safe_float(dg_reweighted_state)
        if not np.isfinite(dg_state_for_report):
            dg_state_for_report = _safe_float(dg_reweighted)
        dg_standard_for_report = _safe_float(dg_reweighted_standard)
        if not np.isfinite(dg_standard_for_report) and np.isfinite(_safe_float(kd_app_reweighted_m)):
            kd_tmp = max(float(kd_app_reweighted_m), 1.0e-300)
            temp_kbt = 0.008314462618 * 300.0
            dg_standard_for_report = float(temp_kbt * np.log(kd_tmp))
        if np.isfinite(dg_state_for_report) and np.isfinite(float(kd_app_reweighted_m)):
            reweight_line = (
                f"Bias-reweighted ΔG(state)={float(dg_state_for_report):.3f} kJ/mol"
            )
            if np.isfinite(dg_standard_for_report):
                reweight_line += f" | ΔG(app,std)={float(dg_standard_for_report):.3f} kJ/mol"
            if np.isfinite(_safe_float(standard_state_correction_kj)):
                reweight_line += f" | std correction={float(standard_state_correction_kj):.3f} kJ/mol"
            if np.isfinite(_safe_float(c_eff_m)):
                reweight_line += f" | Ceff={float(c_eff_m):.3e} M"
            reweight_line += f" | Kd(app)={float(kd_app_reweighted_m):.3e} M | pKd={float(p_kd_reweighted):.2f}"
            if np.isfinite(float(reweight_ci_low)) and np.isfinite(float(reweight_ci_high)):
                # The bootstrap CI is on DeltaG(state); the standard-state value
                # is shifted by the (constant) correction, so its CI is too.
                reweight_line += f" | CI95(state)=[{float(reweight_ci_low):.3f}, {float(reweight_ci_high):.3f}]"
                corr_for_ci = _safe_float(standard_state_correction_kj)
                if np.isfinite(corr_for_ci):
                    reweight_line += (
                        f" | CI95(app,std)=[{float(reweight_ci_low) + corr_for_ci:.3f}, "
                        f"{float(reweight_ci_high) + corr_for_ci:.3f}]"
                    )
            if np.isfinite(float(reweight_ess_total)):
                reweight_line += f" | ESS={float(reweight_ess_total):.1f}"
        conf_txt = str(reweight_confidence_quality or "low").strip().lower()
        selected_line = (
            f"Selected estimator: {str(selected_estimator or 'gated_off')} | "
            f"window_mode={str(estimator_window_mode or 'state_fallback')} | confidence={conf_txt or 'low'}"
        )
        if bool(reweight_confidence_ok) and np.isfinite(float(dg_selected)) and np.isfinite(float(kd_selected_m)):
            selected_line += (
                f" | ΔG(app,std)={float(dg_selected):.3f} kJ/mol | "
                f"Kd(app)={float(kd_selected_m):.3e} M | pKd={float(p_kd_selected):.2f}"
            )
        if bool(reweight_confidence_ok) and np.isfinite(float(dg_selected_state)):
            selected_line += f" | ΔG(state)={float(dg_selected_state):.3f} kJ/mol"
            if np.isfinite(float(dg_selected_standard)):
                selected_line += f" | ΔG(standard)={float(dg_selected_standard):.3f} kJ/mol"
        est_kind_txt = str(estimate_kind or "").strip()
        prod_rel_txt = str(production_reliability or "").strip()
        if est_kind_txt:
            selected_line += f" | estimate={est_kind_txt}"
        if prod_rel_txt:
            selected_line += f" | reliability={prod_rel_txt}"
        prod_note_txt = str(production_note or "").strip()
        if prod_note_txt:
            selected_line += f" | {prod_note_txt}"
        rescored_tag = str(rescored_energy_label or "").strip()
        if rescored_tag and np.isfinite(float(dg_rescored_state_expavg_kj)):
            rescored_line = (
                f"Solvent-rescored hold estimate ({rescored_tag}): "
                f"ΔG(expavg)={float(dg_rescored_state_expavg_kj):.3f} kJ/mol"
            )
            if np.isfinite(float(dg_rescored_state_mean_kj)):
                rescored_line += f" | ΔG(mean)={float(dg_rescored_state_mean_kj):.3f} kJ/mol"
        note_txt = str(reweight_confidence_note or "").strip()
        if note_txt:
            selected_line += f" | {note_txt}"
        if np.isfinite(float(dg_pop)) and np.isfinite(float(kd_app_m)):
            kd_line = (
                f"ΔG(pop)={float(dg_pop):.3f} kJ/mol | Kd(app)={float(kd_app_m):.3e} M | "
                f"pKd(app)={float(p_kd_app):.2f}"
            )
        elif np.isfinite(float(dg_pop_jeff)) and np.isfinite(float(kd_app_jeff_m)):
            missing_state = (
                "unbound" if int(n_unbound) <= 0 else ("bound" if int(n_bound) <= 0 else "stable hold-window")
            )
            kd_line = (
                f"Single-state sampled: missing {missing_state} state. "
                f"Jeffreys pseudo-count diagnostic ΔG={float(dg_pop_jeff):.3f} kJ/mol; "
                "not reported as BFE/Kd."
            )
        state_status_txt = str(binding_estimate_status or "").strip()
        state_note_txt = str(binding_estimate_note or "").strip()
        if state_status_txt:
            selected_line += f" | state_status={state_status_txt}"
        if state_note_txt:
            selected_line += f" | {state_note_txt}"

    sampling_line = ""
    if protocol_txt == "abfe_lite_local_md" and np.isfinite(float(sim_ns)) and np.isfinite(float(sample_every_ps)):
        sampling_line = f"Sampling: {float(sim_ns):.2f} ns total | every {float(sample_every_ps):.2f} ps"
    replica_line = ""
    replica_conv_line = ""
    rep = dict(replica_summary or {})
    rep_count = max(0, int(rep.get("replica_count", 0) or 0))
    rep_req = max(rep_count, int(rep.get("requested_replica_count", rep_count) or rep_count))
    if rep_count > 1:
        sim_rep = _safe_float(rep.get("sim_ns_per_replica", np.nan))
        sim_total = _safe_float(rep.get("sim_ns_total", np.nan))
        if np.isfinite(sim_rep) and np.isfinite(sim_total) and np.isfinite(float(sample_every_ps)):
            sampling_line = (
                f"Sampling: {float(sim_rep):.2f} ns per replica | "
                f"{int(rep_count)} replicas ({float(sim_total):.2f} ns aggregate) | "
                f"every {float(sample_every_ps):.2f} ps"
            )
        replica_line = f"Replicas: {int(rep_count)}"
        if rep_req > rep_count:
            replica_line += f"/{int(rep_req)} successful"
        exp_mean = _safe_float(rep.get("dg_exp_mean", np.nan))
        exp_sem = _safe_float(rep.get("dg_exp_sem", np.nan))
        exp_span = _safe_float(rep.get("dg_exp_span", np.nan))
        if np.isfinite(exp_mean):
            replica_line += f" | ΔG(exp) mean={float(exp_mean):.3f} kJ/mol"
        if np.isfinite(exp_sem):
            replica_line += f" ± {float(exp_sem):.3f} SEM"
        if np.isfinite(exp_span):
            replica_line += f" | span={float(exp_span):.3f}"
        conv_label = str(rep.get("convergence_quality", "") or "").strip().lower()
        unbound_cov = _safe_float(rep.get("unbound_coverage_fraction", np.nan))
        gap_mean = _safe_float(rep.get("exp_cum_gap_mean", np.nan))
        replica_conv_line = "Replica convergence"
        if conv_label:
            replica_conv_line += f": {conv_label}"
        if np.isfinite(unbound_cov):
            replica_conv_line += f" | unbound coverage={float(unbound_cov):.2f}"
        if np.isfinite(gap_mean):
            replica_conv_line += f" | |ΔGexp-ΔGcum|={float(gap_mean):.3f}"
        note_txt = str(rep.get("convergence_note", "") or "").strip()
        if note_txt:
            replica_conv_line += f" | {note_txt}"

    steer_line = ""
    if bool(bias_enabled):
        steer_line = (
            f"Steering: {str(sampling_mode)}, pull={float(pull_distance_a):.2f} Å, "
            f"k_total={float(pull_k_kj_nm2):.1f} kJ/mol/nm²"
        )
        if np.isfinite(float(pull_k_effective_kj_nm2)):
            steer_line += f", k_per_atom={float(pull_k_effective_kj_nm2):.1f}"
        if np.isfinite(float(pull_clearance_a)):
            steer_line += f", clearance={float(pull_clearance_a):.2f} Å"

    scope_txt = "ABFE-lite whole-protein" if bool(whole_protein_mode) else ("ABFE-lite local" if bool(local_active) else "global")
    scope_line = f"Scope: {scope_txt} ({int(prot_used)}/{int(prot_total)} protein atoms)"
    dist_axis_label = abfe_distance_axis_label(distance_metric)
    chemistry_parts = [
        f"Ligand FF mode: {str(ligand_ff_mode or 'n/a')}",
        f"charges: {str(ligand_charge_source or 'n/a')}",
    ]
    ligand_target_ph_f = _safe_float(ligand_target_ph)
    if np.isfinite(ligand_target_ph_f):
        chemistry_parts.append(f"target pH={float(ligand_target_ph_f):.2f}")
    target_hint_txt = str(target_hint or "").strip()
    if target_hint_txt:
        chemistry_parts.append(f"target hint={target_hint_txt}")
    chemistry_line = " | ".join(str(x) for x in chemistry_parts if str(x).strip())

    def _fmt_kj(value, digits=3):
        vv = _safe_float(value)
        if not np.isfinite(vv):
            return "n/a"
        return f"{float(vv):.{int(max(0, digits))}f} kJ/mol"

    def _fmt_m(value):
        vv = _safe_float(value)
        if not np.isfinite(vv):
            return "n/a"
        return f"{float(vv):.3e} M"

    def _fmt_pkd(value):
        vv = _safe_float(value)
        if not np.isfinite(vv):
            return "n/a"
        return f"{float(vv):.2f}"

    conf_txt = str(reweight_confidence_quality or "low").strip().lower() or "low"
    conf_palette = {
        "high": {"bg": "#DFF3E8", "fg": "#135E3A", "border": "#8BC8A5"},
        "medium": {"bg": "#FFF0CC", "fg": "#8A5A00", "border": "#F0C36B"},
        "low": {"bg": "#FDE2E1", "fg": "#8A2F2C", "border": "#E39A96"},
    }
    conf_style = dict(conf_palette.get(conf_txt, conf_palette["low"]))
    proto_badge_bg = "#E8EEF7"
    proto_badge_fg = "#244B74"
    selected_value_txt = _fmt_kj(dg_selected) if bool(reweight_confidence_ok) else "gated off"
    selected_kd_txt = _fmt_m(kd_selected_m) if bool(reweight_confidence_ok) else "n/a"
    selected_pkd_txt = _fmt_pkd(p_kd_selected) if bool(reweight_confidence_ok) else "n/a"
    steering_txt = "Adaptive probe + hold steering" if bool(bias_enabled) else "Passive trajectory rescoring"
    summary_strip = str(selected_line or reweight_line or state_line)
    if rescored_line:
        summary_strip = f"{summary_strip} | {rescored_line}" if summary_strip else rescored_line
    replica_html = ""
    if replica_line or replica_conv_line:
        replica_bits = []
        if replica_line:
            replica_bits.append(
                f"<div style=\"margin-top:10px; padding:10px 12px; background:#F5F8FC; border:1px solid #D7E3F2; border-radius:10px; line-height:1.45;\">{replica_line}</div>"
            )
        if replica_conv_line:
            replica_bits.append(
                f"<div style=\"margin-top:8px; padding:10px 12px; background:#F7FAF7; border:1px solid #D7E6D7; border-radius:10px; line-height:1.45;\">{replica_conv_line}</div>"
            )
        replica_html = "".join(replica_bits)
    chemistry_html = ""
    if chemistry_line:
        chemistry_html = (
            "<div style=\"margin-top:10px; padding:10px 12px; background:#FFFDFC; "
            "border:1px solid #E8DED3; border-radius:10px; line-height:1.45;\">"
            f"{chemistry_line}</div>"
        )
    message_html = "\n".join(
        [
            "<div style=\"font-family:'Helvetica Neue',Helvetica,Arial,sans-serif; min-width:560px; color:#24313f;\">",
            "  <div style=\"margin-bottom:10px;\">",
            f"    <span style=\"display:inline-block; padding:4px 10px; border-radius:999px; background:{proto_badge_bg}; color:{proto_badge_fg}; font-weight:700; font-size:12px; margin-right:6px;\">{proto_label}</span>",
            f"    <span style=\"display:inline-block; padding:4px 10px; border-radius:999px; background:{conf_style['bg']}; color:{conf_style['fg']}; border:1px solid {conf_style['border']}; font-weight:700; font-size:12px;\">Confidence: {conf_txt}</span>",
            "  </div>",
            "  <div style=\"margin-bottom:10px; padding:8px 10px; background:#FFF4E5; border:1px solid #F0C27B; "
            f"border-radius:8px; font-size:12px; color:#7A4A00;\">{ABFE_EXPERIMENTAL_NOTE}</div>",
            f"  <div style=\"font-size:22px; font-weight:800; color:#17212B; margin-bottom:2px;\">Selected ΔG: {selected_value_txt}</div>",
            f"  <div style=\"font-size:12px; color:#5A6572; margin-bottom:10px;\">Kd(app): {selected_kd_txt} &nbsp;&nbsp; pKd: {selected_pkd_txt} &nbsp;&nbsp; Estimator: {str(selected_estimator or 'gated_off')}</div>",
            "  <table style=\"width:100%; border-collapse:separate; border-spacing:0 6px; font-size:12px;\">",
            "    <tr>",
            f"      <td style=\"padding:8px 10px; background:#F8F4EE; border-radius:8px;\"><b>ΔG exp-avg</b><br/>{_fmt_kj(dg_exp)}</td>",
            f"      <td style=\"padding:8px 10px; background:#F8F4EE; border-radius:8px;\"><b>95% CI</b><br/>[{_fmt_kj(ci_low).replace(' kJ/mol', '')}, {_fmt_kj(ci_high).replace(' kJ/mol', '')}] kJ/mol</td>",
            f"      <td style=\"padding:8px 10px; background:#F8F4EE; border-radius:8px;\"><b>ΔG cumulant</b><br/>{_fmt_kj(dg_cum)}</td>",
            "    </tr>",
            "    <tr>",
            f"      <td style=\"padding:8px 10px; background:#F8F4EE; border-radius:8px;\"><b>Frames used</b><br/>{int(n_samples)} FE frames</td>",
            f"      <td style=\"padding:8px 10px; background:#F8F4EE; border-radius:8px;\"><b>Bound / Unbound</b><br/>{int(n_bound)} / {int(n_unbound)}</td>",
            f"      <td style=\"padding:8px 10px; background:#F8F4EE; border-radius:8px;\"><b>ESS</b><br/>{(f'{float(reweight_ess_total):.1f}' if np.isfinite(_safe_float(reweight_ess_total)) else 'n/a')}</td>",
            "    </tr>",
            "  </table>",
            f"  <div style=\"margin-top:10px; padding:10px 12px; background:#FCFAF6; border:1px solid #E6DDCF; border-radius:10px; line-height:1.45;\">{state_line}<br/>{reweight_line or selected_line or kd_line}{('<br/>' + rescored_line) if rescored_line else ''}</div>",
            f"  {replica_html}",
            f"  {chemistry_html}",
            f"  <div style=\"margin-top:10px; font-size:12px; color:#54606C;\">{steering_txt} | {scope_line} | {dist_axis_label}</div>",
            f"  <div style=\"margin-top:6px; font-size:12px; color:#6C7883;\">{str(reweight_confidence_note or '')}</div>",
            "</div>",
        ]
    )

    msg_lines = [
        str(intro),
        "",
        ABFE_EXPERIMENTAL_NOTE,
        "",
        f"ΔG (exp-avg): {float(dg_exp):.3f} kJ/mol",
        f"95% CI: [{float(ci_low):.3f}, {float(ci_high):.3f}] kJ/mol",
        f"ΔG (cumulant): {float(dg_cum):.3f} kJ/mol",
        str(state_line),
    ]
    if prob_line:
        msg_lines.append(str(prob_line))
    if reweight_line:
        msg_lines.append(str(reweight_line))
    if selected_line:
        msg_lines.append(str(selected_line))
    if rescored_line:
        msg_lines.append(str(rescored_line))
    if kd_line:
        msg_lines.append(str(kd_line))
    msg_lines.append(f"Frames used: {int(n_samples)} | Engine: {str(energy_mode)}")
    if sampling_line:
        msg_lines.append(str(sampling_line))
    if replica_line:
        msg_lines.append(str(replica_line))
    if replica_conv_line:
        msg_lines.append(str(replica_conv_line))
    if steer_line:
        msg_lines.append(str(steer_line))
    msg_lines.append(str(scope_line))
    msg_lines.append(f"Distance metric: {dist_axis_label} | bound cutoff={float(bound_cutoff_a):.2f} Å")
    if chemistry_line:
        msg_lines.append(str(chemistry_line))

    return {
        "ok": True,
        "proto_label": str(proto_label),
        "scope_label_plot": str(scope_label_plot),
        "suptitle": str(suptitle),
        "dist_axis_label": str(dist_axis_label),
        "state_line": str(state_line),
        "prob_line": str(prob_line),
        "reweight_line": str(reweight_line),
        "selected_line": str(selected_line),
        "rescored_line": str(rescored_line),
        "kd_line": str(kd_line),
        "intro": str(intro),
        "sampling_line": str(sampling_line),
        "replica_line": str(replica_line),
        "replica_conv_line": str(replica_conv_line),
        "chemistry_line": str(chemistry_line),
        "steer_line": str(steer_line),
        "scope_line": str(scope_line),
        "scope_txt": str(scope_txt),
        "summary_strip": str(summary_strip),
        "message_html": str(message_html),
        "dialog_title": f"{proto_label} Summary",
        "message_lines": list(msg_lines),
        "message_text": "\n".join(msg_lines),
        "experimental_note": ABFE_EXPERIMENTAL_NOTE,
    }


def build_abfe_free_energy_debug_line(
    *,
    protocol="",
    energy_mode="",
    whole_protein_mode=False,
    local_active=False,
    prot_used=0,
    prot_total=0,
    n_samples=0,
    sampling_mode="",
    bias_enabled=False,
    n_bound=0,
    n_dist_valid=0,
    n_unbound=0,
    dg_exp=np.nan,
    dg_cum=np.nan,
    dg_pop=np.nan,
    dg_reweighted=np.nan,
    kd_app_m=np.nan,
    kd_app_reweighted_m=np.nan,
    reweight_confidence_quality="low",
    selected_estimator="gated_off",
    contact_frac_mean=np.nan,
    target_hint="",
    ci_low=np.nan,
    ci_high=np.nan,
    replica_summary=None,
):
    scope_txt = "whole" if bool(whole_protein_mode) else ("local" if bool(local_active) else "global")
    dbg = (
        f"Free-energy estimate: protocol={str(protocol)}, mode={str(energy_mode)}, scope={scope_txt} "
        f"({int(prot_used)}/{int(prot_total)}), n={int(n_samples)}, "
        f"sampling_mode={str(sampling_mode)}, bias={bool(bias_enabled)}, "
        f"bound={int(n_bound)}/{int(n_dist_valid)}, unbound={int(n_unbound)}/{int(n_dist_valid)}, "
        f"DGexp={float(dg_exp):.3f}, DGcum={float(dg_cum):.3f}, "
        f"DGpop={float(dg_pop):.3f}, DGreweight={float(dg_reweighted):.3f}, "
        f"Kd_app={float(kd_app_m):.3e} M, Kd_rw={float(kd_app_reweighted_m):.3e} M, "
        f"rw_conf={str(reweight_confidence_quality or 'low')}, "
        f"selected={str(selected_estimator or 'gated_off')}, "
        f"contact_frac={float(contact_frac_mean):.4f}, target_hint={str(target_hint) or 'none'}, "
        f"CI95=({float(ci_low):.3f},{float(ci_high):.3f})."
    )
    rep = dict(replica_summary or {})
    rep_count = max(0, int(rep.get("replica_count", 0) or 0))
    if rep_count > 1:
        dbg += (
            f" replicas={int(rep_count)}, rep_conv={str(rep.get('convergence_quality', '') or 'unknown')}, "
            f"rep_sem={float(_safe_float(rep.get('dg_exp_sem', np.nan))):.3f}, "
            f"rep_span={float(_safe_float(rep.get('dg_exp_span', np.nan))):.3f}."
        )
    return dbg


def build_abfe_replica_aggregate_payload(
    *,
    replica_results=None,
    requested_replica_count=0,
    temperature_k=300.0,
    bootstrap_seed=20260212,
    n_boot=300,
):
    rows = []
    replica_inputs = list(replica_results or [])
    accepted_replicas = []
    accepted_positions = []
    total_chunks = []
    dist_chunks = []
    coul_chunks = []
    bias_chunks = []
    rescored_chunks = []
    ligand_internal_chunks = []
    bound_chunks = []
    unbound_chunks = []
    run_frames_ref = None
    run_exp_rows = []
    run_cum_rows = []
    n_bound_total = 0
    n_unbound_total = 0
    n_dist_valid_total = 0
    v_eff_vals = []
    sim_ns_vals = []
    sample_every_vals = []
    bound_cutoff_vals = []
    unbound_rep_hits = 0
    bound_rep_hits = 0

    def _stats(values):
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        out = {
            "n": int(finite.size),
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "span": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
        if finite.size <= 0:
            return out
        out["mean"] = float(np.mean(finite))
        out["std"] = float(np.std(finite, ddof=1 if finite.size > 1 else 0))
        out["sem"] = float(out["std"] / np.sqrt(max(int(finite.size), 1)))
        out["min"] = float(np.min(finite))
        out["max"] = float(np.max(finite))
        out["span"] = float(out["max"] - out["min"])
        return out

    for rep_pos, raw in enumerate(replica_inputs, start=1):
        rep = dict(raw or {})
        total_valid = _abfe_float_series(rep.get("total_valid"))
        if total_valid.size <= 0:
            continue
        distance_valid = _abfe_float_series(rep.get("distance_valid"))
        if distance_valid.size == 0:
            distance_valid = _abfe_float_series(rep.get("min_valid"))
        # Align within each replica before concatenating. Padding only after
        # concatenation shifts the next replica's observables onto this one.
        distance_valid = _abfe_float_series(distance_valid, n=total_valid.size)
        rep["distance_valid"] = distance_valid
        for key in ("coul_valid", "bias_valid", "rescored_total_valid", "ligand_internal_valid"):
            rep[key] = _abfe_float_series(rep.get(key), n=total_valid.size)
        for key in ("bound_valid_mask", "unbound_valid_mask"):
            rep[key] = _abfe_bool_series(rep.get(key), n=total_valid.size, fill=False)
        bound_cutoff_a = _safe_float(rep.get("bound_cutoff_a", np.nan))
        rep_n_bound = int(rep.get("n_bound", 0) or 0)
        rep_n_unbound = int(rep.get("n_unbound", 0) or 0)
        rep_n_dist_valid = int(rep.get("n_dist_valid", np.sum(np.isfinite(distance_valid))) or 0)
        numeric = build_abfe_free_energy_numeric_payload(
            frames_valid=np.arange(1, total_valid.size + 1, dtype=int),
            bias_valid=np.asarray(rep.get("bias_valid", []), dtype=float),
            total_valid=np.asarray(total_valid, dtype=float),
            rescored_total_valid=np.asarray(rep.get("rescored_total_valid", []), dtype=float),
            ligand_internal_valid=np.asarray(rep.get("ligand_internal_valid", []), dtype=float),
            rescored_energy_label=str(rep.get("rescored_energy_label", "") or ""),
            distance_valid=np.asarray(distance_valid, dtype=float),
            coul_valid=np.asarray(rep.get("coul_valid", []), dtype=float),
            bound_valid_mask=np.asarray(rep.get("bound_valid_mask", []), dtype=bool),
            unbound_valid_mask=np.asarray(rep.get("unbound_valid_mask", []), dtype=bool),
            bound_cutoff_a=float(bound_cutoff_a),
            n_bound=int(rep_n_bound),
            n_dist_valid=int(rep_n_dist_valid),
            n_bound_raw=int(rep.get("n_bound_raw", rep_n_bound) or rep_n_bound),
            n_unbound_raw=int(rep.get("n_unbound_raw", rep_n_unbound) or rep_n_unbound),
            n_bound_hold=int(rep.get("n_bound_hold", rep_n_bound) or rep_n_bound),
            n_unbound_hold=int(rep.get("n_unbound_hold", rep_n_unbound) or rep_n_unbound),
            estimator_window_mode=str(rep.get("estimator_window_mode", "state_fallback") or "state_fallback"),
            use_count_fraction=bool(rep_n_dist_valid > 0),
            temperature_k=float(temperature_k),
            v_eff_nm3=float(rep.get("v_eff_nm3", np.nan)),
            bootstrap_seed=int(bootstrap_seed) + int(rep_pos),
            n_boot=int(n_boot),
        )
        if not bool(numeric.get("ok", False)):
            continue
        accepted_replicas.append(rep)
        accepted_positions.append(rep_pos - 1)
        rows.append(
            {
                "replica_index": int(rep.get("replica_index", rep_pos) or rep_pos),
                "random_seed": int(rep.get("random_seed", 0) or 0),
                "n_samples": int(numeric.get("n_samples", total_valid.size)),
                "n_dist_valid": int(rep_n_dist_valid),
                "n_bound": int(rep_n_bound),
                "n_unbound": int(rep_n_unbound),
                "dg_exp": float(numeric.get("dg_exp", np.nan)),
                "dg_cum": float(numeric.get("dg_cum", np.nan)),
                "ci_low": float(numeric.get("ci_low", np.nan)),
                "ci_high": float(numeric.get("ci_high", np.nan)),
                "coul_mean": float(numeric.get("coul_mean", np.nan)),
                "contact_frac_mean": float(numeric.get("contact_frac_mean", np.nan)),
                "dg_pop_kj": float(rep.get("dg_pop_kj", np.nan)),
                "dg_reweighted_kj": float(numeric.get("dg_reweighted_kj", np.nan)),
                "kd_app_reweighted_m": float(numeric.get("kd_app_reweighted_m", np.nan)),
                "reweight_ess_total": float(numeric.get("reweight_ess_total", np.nan)),
                "reweight_confidence_quality": str(numeric.get("reweight_confidence_quality", "low")),
                "reweight_confidence_ok": bool(numeric.get("reweight_confidence_ok", False)),
                "dg_selected_kj": float(numeric.get("dg_selected_kj", np.nan)),
                "kd_selected_m": float(
                    numeric.get(
                        "kd_selected_m",
                        numeric.get("kd_app_reweighted_m", np.nan),
                    )
                ),
                "selected_estimator": str(numeric.get("selected_estimator", "gated_off")),
                "p_bound": float(rep.get("p_bound", np.nan)),
                "p_unbound": float(rep.get("p_unbound", np.nan)),
            }
        )
        total_chunks.append(np.asarray(total_valid, dtype=float))
        dist_chunks.append(np.asarray(distance_valid, dtype=float))
        coul_chunks.append(np.asarray(rep.get("coul_valid", []), dtype=float))
        bias_chunks.append(np.asarray(rep.get("bias_valid", []), dtype=float))
        rescored_chunks.append(np.asarray(rep.get("rescored_total_valid", []), dtype=float))
        ligand_internal_chunks.append(np.asarray(rep.get("ligand_internal_valid", []), dtype=float))
        bound_chunks.append(np.asarray(rep.get("bound_valid_mask", []), dtype=bool))
        unbound_chunks.append(np.asarray(rep.get("unbound_valid_mask", []), dtype=bool))
        n_bound_total += int(rep_n_bound)
        n_unbound_total += int(rep_n_unbound)
        n_dist_valid_total += int(rep_n_dist_valid)
        if int(rep_n_unbound) > 0:
            unbound_rep_hits += 1
        if int(rep_n_bound) > 0:
            bound_rep_hits += 1
        vv = _safe_float(rep.get("v_eff_nm3", np.nan))
        if np.isfinite(vv):
            v_eff_vals.append(float(vv))
        sim_val = _safe_float(rep.get("sim_ns", np.nan))
        if np.isfinite(sim_val):
            sim_ns_vals.append(float(sim_val))
        step_val = _safe_float(rep.get("sample_every_ps", np.nan))
        if np.isfinite(step_val):
            sample_every_vals.append(float(step_val))
        if np.isfinite(bound_cutoff_a):
            bound_cutoff_vals.append(float(bound_cutoff_a))
        run_frames_i = np.asarray(numeric.get("run_frames", []), dtype=int)
        run_exp_i = np.asarray(numeric.get("run_exp", []), dtype=float)
        run_cum_i = np.asarray(numeric.get("run_cum", []), dtype=float)
        if run_frames_i.size > 0 and run_exp_i.size == run_frames_i.size and run_cum_i.size == run_frames_i.size:
            if run_frames_ref is None:
                run_frames_ref = np.asarray(run_frames_i, dtype=int)
            min_len = min(int(run_frames_ref.size), int(run_frames_i.size))
            run_frames_ref = np.asarray(run_frames_ref[:min_len], dtype=int)
            run_exp_rows = [np.asarray(rr[:min_len], dtype=float) for rr in run_exp_rows]
            run_cum_rows = [np.asarray(rr[:min_len], dtype=float) for rr in run_cum_rows]
            run_exp_rows.append(np.asarray(run_exp_i[:min_len], dtype=float))
            run_cum_rows.append(np.asarray(run_cum_i[:min_len], dtype=float))

    rep_count = int(len(rows))
    req_count = int(max(len(replica_inputs), int(requested_replica_count or 0)))
    if rep_count <= 0:
        return {
            "ok": False,
            "reason": "no_replicas",
            "replica_count": 0,
            "requested_replica_count": int(req_count),
            "replica_rows": [],
        }

    concat_total = np.concatenate(total_chunks, axis=0) if total_chunks else np.asarray([], dtype=float)
    concat_dist = np.concatenate(dist_chunks, axis=0) if dist_chunks else np.asarray([], dtype=float)
    concat_coul = np.concatenate(coul_chunks, axis=0) if coul_chunks else np.asarray([], dtype=float)
    concat_bias = np.concatenate(bias_chunks, axis=0) if bias_chunks else np.asarray([], dtype=float)
    concat_rescored = np.concatenate(rescored_chunks, axis=0) if rescored_chunks else np.asarray([], dtype=float)
    concat_ligand_internal = np.concatenate(ligand_internal_chunks, axis=0) if ligand_internal_chunks else np.asarray([], dtype=float)
    concat_bound = np.concatenate(bound_chunks, axis=0) if bound_chunks else np.asarray([], dtype=bool)
    concat_unbound = np.concatenate(unbound_chunks, axis=0) if unbound_chunks else np.asarray([], dtype=bool)
    pooled_numeric = build_abfe_free_energy_numeric_payload(
        frames_valid=np.arange(1, concat_total.size + 1, dtype=int),
        bias_valid=np.asarray(concat_bias, dtype=float),
        total_valid=np.asarray(concat_total, dtype=float),
        rescored_total_valid=np.asarray(concat_rescored, dtype=float),
        ligand_internal_valid=np.asarray(concat_ligand_internal, dtype=float),
        rescored_energy_label=str(next((str(rep.get("rescored_energy_label", "") or "") for rep in accepted_replicas if str(rep.get("rescored_energy_label", "") or "").strip()), "")),
        distance_valid=np.asarray(concat_dist, dtype=float),
        coul_valid=np.asarray(concat_coul, dtype=float),
        bound_valid_mask=np.asarray(concat_bound, dtype=bool),
        unbound_valid_mask=np.asarray(concat_unbound, dtype=bool),
        bound_cutoff_a=float(np.nanmean(bound_cutoff_vals)) if bound_cutoff_vals else np.nan,
        n_bound=int(n_bound_total),
        n_dist_valid=int(n_dist_valid_total),
        n_bound_raw=int(sum(int(rep.get("n_bound_raw", rep.get("n_bound", 0) or 0)) for rep in accepted_replicas)),
        n_unbound_raw=int(sum(int(rep.get("n_unbound_raw", rep.get("n_unbound", 0) or 0)) for rep in accepted_replicas)),
        n_bound_hold=int(sum(int(rep.get("n_bound_hold", rep.get("n_bound", 0) or 0)) for rep in accepted_replicas)),
        n_unbound_hold=int(sum(int(rep.get("n_unbound_hold", rep.get("n_unbound", 0) or 0)) for rep in accepted_replicas)),
        estimator_window_mode=("hold_only" if all(str(rep.get("estimator_window_mode", "state_fallback") or "state_fallback") == "hold_only" for rep in accepted_replicas) else "state_fallback"),
        use_count_fraction=bool(n_dist_valid_total > 0),
        temperature_k=float(temperature_k),
        v_eff_nm3=float(np.nanmean(v_eff_vals)) if v_eff_vals else np.nan,
        bootstrap_seed=int(bootstrap_seed),
        n_boot=int(n_boot),
    )

    exp_stats = _stats([r.get("dg_exp", np.nan) for r in rows])
    cum_stats = _stats([r.get("dg_cum", np.nan) for r in rows])
    gap_stats = _stats(
        [
            abs(float(r.get("dg_exp", np.nan)) - float(r.get("dg_cum", np.nan)))
            if np.isfinite(_safe_float(r.get("dg_exp", np.nan))) and np.isfinite(_safe_float(r.get("dg_cum", np.nan)))
            else np.nan
            for r in rows
        ]
    )
    contact_stats = _stats([r.get("contact_frac_mean", np.nan) for r in rows])
    coul_stats = _stats([r.get("coul_mean", np.nan) for r in rows])
    dg_pop_stats = _stats([r.get("dg_pop_kj", np.nan) for r in rows])
    dg_reweight_stats = _stats([r.get("dg_reweighted_kj", np.nan) for r in rows])
    reweight_ess_stats = _stats([r.get("reweight_ess_total", np.nan) for r in rows])
    selected_rows = [
        dict(r)
        for r in rows
        if bool(r.get("reweight_confidence_ok", False)) and np.isfinite(_safe_float(r.get("dg_selected_kj", np.nan)))
    ]
    dg_selected_stats = _stats([r.get("dg_selected_kj", np.nan) for r in selected_rows])
    exp_arr = np.asarray([r.get("dg_exp", np.nan) for r in rows], dtype=float)
    exp_finite = exp_arr[np.isfinite(exp_arr)]
    if exp_finite.size >= 2:
        n_boot_i = max(32, int(n_boot or 128))
        rng = np.random.default_rng(int(bootstrap_seed) + 7919)
        boot_mean = np.empty(int(n_boot_i), dtype=float)
        for b in range(int(n_boot_i)):
            idx = rng.integers(0, exp_finite.size, size=exp_finite.size)
            boot_mean[b] = float(np.mean(exp_finite[idx]))
        rep_ci_low, rep_ci_high = np.percentile(boot_mean, [2.5, 97.5])
    elif exp_finite.size == 1:
        rep_ci_low = float(rows[0].get("ci_low", np.nan))
        rep_ci_high = float(rows[0].get("ci_high", np.nan))
    else:
        rep_ci_low = np.nan
        rep_ci_high = np.nan

    if run_exp_rows and run_frames_ref is not None:
        run_exp_mat = np.vstack(run_exp_rows)
        run_cum_mat = np.vstack(run_cum_rows)
        run_exp_mean = np.nanmean(run_exp_mat, axis=0)
        run_cum_mean = np.nanmean(run_cum_mat, axis=0)
        exp_counts = np.sum(np.isfinite(run_exp_mat), axis=0)
        cum_counts = np.sum(np.isfinite(run_cum_mat), axis=0)
        run_exp_sem = np.nanstd(run_exp_mat, axis=0, ddof=1 if rep_count > 1 else 0) / np.sqrt(np.maximum(exp_counts, 1))
        run_cum_sem = np.nanstd(run_cum_mat, axis=0, ddof=1 if rep_count > 1 else 0) / np.sqrt(np.maximum(cum_counts, 1))
    else:
        run_frames_ref = np.asarray([], dtype=int)
        run_exp_mean = np.asarray([], dtype=float)
        run_cum_mean = np.asarray([], dtype=float)
        run_exp_sem = np.asarray([], dtype=float)
        run_cum_sem = np.asarray([], dtype=float)

    k_b = 0.008314462618
    kbt = float(k_b * float(temperature_k))
    p_bound = np.nan
    p_unbound = np.nan
    dg_pop = np.nan
    p_bound_jeff = np.nan
    p_unbound_jeff = np.nan
    dg_pop_jeff = np.nan
    kd_app_m = np.nan
    p_kd_app = np.nan
    kd_app_jeff_m = np.nan
    p_kd_jeff = np.nan
    if n_dist_valid_total > 0:
        p_bound = float(np.clip(float(n_bound_total) / float(max(n_dist_valid_total, 1)), 0.0, 1.0))
        p_unbound = float(np.clip(float(n_unbound_total) / float(max(n_dist_valid_total, 1)), 0.0, 1.0))
        if np.isfinite(kbt) and kbt > 0.0 and p_bound > 0.0 and p_unbound > 0.0:
            dg_pop = float(-kbt * np.log(p_bound / p_unbound))
        p_bound_jeff = float((n_bound_total + 0.5) / (n_dist_valid_total + 1.0))
        p_unbound_jeff = float((n_unbound_total + 0.5) / (n_dist_valid_total + 1.0))
        if np.isfinite(kbt) and kbt > 0.0 and p_bound_jeff > 0.0 and p_unbound_jeff > 0.0:
            dg_pop_jeff = float(-kbt * np.log(p_bound_jeff / p_unbound_jeff))
    v_eff_nm3 = float(np.nanmean(v_eff_vals)) if v_eff_vals else np.nan
    if np.isfinite(v_eff_nm3) and v_eff_nm3 > 0.0:
        c_eff_m = float(1.66054 / max(v_eff_nm3, 1.0e-6))
        if np.isfinite(p_bound) and np.isfinite(p_unbound) and p_bound > 0.0:
            kd_app_m = float(c_eff_m * (p_unbound / p_bound))
            p_kd_app = float(-np.log10(max(kd_app_m, 1.0e-300)))
        if np.isfinite(p_bound_jeff) and np.isfinite(p_unbound_jeff) and p_bound_jeff > 0.0:
            kd_app_jeff_m = float(c_eff_m * (p_unbound_jeff / p_bound_jeff))
            p_kd_jeff = float(-np.log10(max(kd_app_jeff_m, 1.0e-300)))

    unbound_cov = float(unbound_rep_hits / float(max(rep_count, 1)))
    score = 0
    if np.isfinite(exp_stats["sem"]) and exp_stats["sem"] <= 2.0:
        score += 1
    if np.isfinite(exp_stats["span"]) and exp_stats["span"] <= 5.0:
        score += 1
    if np.isfinite(gap_stats["mean"]) and gap_stats["mean"] <= 3.0:
        score += 1
    if unbound_cov >= 0.67:
        score += 1
    if score >= 4:
        conv_quality = "high"
        conv_note = "replica spread is small and unbound coverage is consistent"
    elif score >= 2:
        conv_quality = "medium"
        conv_note = "sampling is usable but still benefits from more replicas or longer hold time"
    else:
        conv_quality = "low"
        conv_note = "replica spread or unbound coverage is still weak"

    primary_idx = 0
    if np.isfinite(exp_stats["mean"]):
        best_delta = None
        for idx, row in enumerate(rows):
            dv = _safe_float(row.get("dg_exp", np.nan))
            if not np.isfinite(dv):
                continue
            delta = abs(float(dv) - float(exp_stats["mean"]))
            if best_delta is None or delta < best_delta:
                best_delta = float(delta)
                primary_idx = int(idx)

    selected_replica_count = int(len(selected_rows))
    selected_replica_fraction = float(selected_replica_count / float(max(rep_count, 1)))
    min_selected_reps = 1 if rep_count <= 1 else max(2, int(np.ceil(0.5 * float(rep_count))))
    selected_confidence_quality = "low"
    selected_confidence_note = "too few replicas produced accepted hold-window estimates"
    selected_confidence_ok = False
    selected_estimator = "gated_off"
    dg_selected_summary = np.nan
    kd_selected_summary = np.nan
    p_kd_selected_summary = np.nan
    selected_estimators = sorted(set(str(r.get("selected_estimator", "gated_off") or "gated_off") for r in selected_rows))
    if selected_replica_count >= int(min_selected_reps):
        selected_confidence_ok = True
        if len(selected_estimators) == 1 and selected_estimators[0] != "gated_off":
            selected_estimator = f"replica_aggregate_{selected_estimators[0]}"
        else:
            selected_estimator = "replica_aggregate_mixed_selected_estimates"
        dg_selected_summary = float(dg_selected_stats["mean"]) if np.isfinite(dg_selected_stats["mean"]) else np.nan
        # Report the Kd corresponding to the aggregated DeltaG. Averaging Kd
        # separately breaks DeltaG = RT ln(Kd / 1 M) for nonidentical replicas.
        if np.isfinite(dg_selected_summary) and np.isfinite(kbt) and kbt > 0.0:
            kd_selected_summary = float(np.exp(np.clip(dg_selected_summary / kbt, -700.0, 700.0)))
        if np.isfinite(kd_selected_summary) and kd_selected_summary > 0.0:
            p_kd_selected_summary = float(-np.log10(max(kd_selected_summary, 1.0e-300)))
        if (
            selected_replica_fraction >= 0.9
            and np.isfinite(dg_selected_stats["sem"])
            and dg_selected_stats["sem"] <= 3.0
        ):
            selected_confidence_quality = "high"
            selected_confidence_note = "accepted hold-window estimates were reproduced across nearly all replicas with low spread"
        else:
            selected_confidence_quality = "medium"
            selected_confidence_note = "accepted hold-window estimates were reproduced across multiple replicas, but additional replicas would tighten confidence"
    elif selected_replica_count > 0:
        selected_confidence_note = (
            f"only {selected_replica_count}/{rep_count} replicas produced accepted hold-window estimates"
        )

    pooled_reweight_confidence_quality = str(pooled_numeric.get("reweight_confidence_quality", "low"))
    pooled_reweight_confidence_note = str(pooled_numeric.get("reweight_confidence_note", ""))
    pooled_reweight_confidence_ok = bool(pooled_numeric.get("reweight_confidence_ok", False))

    summary = {
        "replica_count": int(rep_count),
        "requested_replica_count": int(req_count),
        "failed_replica_count": int(max(req_count - rep_count, 0)),
        "primary_replica_index": int(accepted_positions[primary_idx]),
        "dg_exp_mean": float(exp_stats["mean"]) if np.isfinite(exp_stats["mean"]) else np.nan,
        "dg_exp_std": float(exp_stats["std"]) if np.isfinite(exp_stats["std"]) else np.nan,
        "dg_exp_sem": float(exp_stats["sem"]) if np.isfinite(exp_stats["sem"]) else np.nan,
        "dg_exp_span": float(exp_stats["span"]) if np.isfinite(exp_stats["span"]) else np.nan,
        "dg_exp_ci_low": float(rep_ci_low) if np.isfinite(rep_ci_low) else np.nan,
        "dg_exp_ci_high": float(rep_ci_high) if np.isfinite(rep_ci_high) else np.nan,
        "dg_cum_mean": float(cum_stats["mean"]) if np.isfinite(cum_stats["mean"]) else np.nan,
        "dg_cum_std": float(cum_stats["std"]) if np.isfinite(cum_stats["std"]) else np.nan,
        "dg_cum_sem": float(cum_stats["sem"]) if np.isfinite(cum_stats["sem"]) else np.nan,
        "coul_mean": float(coul_stats["mean"]) if np.isfinite(coul_stats["mean"]) else np.nan,
        "contact_frac_mean": float(contact_stats["mean"]) if np.isfinite(contact_stats["mean"]) else np.nan,
        "dg_pop_mean": float(dg_pop_stats["mean"]) if np.isfinite(dg_pop_stats["mean"]) else np.nan,
        "dg_pop_sem": float(dg_pop_stats["sem"]) if np.isfinite(dg_pop_stats["sem"]) else np.nan,
        "dg_reweighted_mean": float(dg_reweight_stats["mean"]) if np.isfinite(dg_reweight_stats["mean"]) else np.nan,
        "dg_reweighted_sem": float(dg_reweight_stats["sem"]) if np.isfinite(dg_reweight_stats["sem"]) else np.nan,
        "dg_selected_mean": float(dg_selected_stats["mean"]) if np.isfinite(dg_selected_stats["mean"]) else np.nan,
        "dg_selected_sem": float(dg_selected_stats["sem"]) if np.isfinite(dg_selected_stats["sem"]) else np.nan,
        "selected_replica_count": int(selected_replica_count),
        "selected_replica_fraction": float(selected_replica_fraction),
        "n_samples_total": int(sum(int(r.get("n_samples", 0) or 0) for r in rows)),
        "n_dist_valid_total": int(n_dist_valid_total),
        "n_bound_total": int(n_bound_total),
        "n_unbound_total": int(n_unbound_total),
        "p_bound": float(p_bound) if np.isfinite(p_bound) else np.nan,
        "p_unbound": float(p_unbound) if np.isfinite(p_unbound) else np.nan,
        "dg_pop_kj": float(dg_pop) if np.isfinite(dg_pop) else np.nan,
        "p_bound_jeff": float(p_bound_jeff) if np.isfinite(p_bound_jeff) else np.nan,
        "p_unbound_jeff": float(p_unbound_jeff) if np.isfinite(p_unbound_jeff) else np.nan,
        "dg_pop_jeff_kj": float(dg_pop_jeff) if np.isfinite(dg_pop_jeff) else np.nan,
        "p_bound_reweighted": float(pooled_numeric.get("p_bound_reweighted", np.nan)),
        "p_unbound_reweighted": float(pooled_numeric.get("p_unbound_reweighted", np.nan)),
        "dg_reweighted_kj": float(pooled_numeric.get("dg_reweighted_kj", np.nan)),
        "kd_app_reweighted_m": float(pooled_numeric.get("kd_app_reweighted_m", np.nan)),
        "p_kd_reweighted": float(pooled_numeric.get("p_kd_reweighted", np.nan)),
        "reweight_ci_low": float(pooled_numeric.get("reweight_ci_low", np.nan)),
        "reweight_ci_high": float(pooled_numeric.get("reweight_ci_high", np.nan)),
        "reweight_ess_total": float(pooled_numeric.get("reweight_ess_total", np.nan)),
        "reweight_ess_bound": float(pooled_numeric.get("reweight_ess_bound", np.nan)),
        "reweight_ess_unbound": float(pooled_numeric.get("reweight_ess_unbound", np.nan)),
        "reweight_ess_mean": float(reweight_ess_stats["mean"]) if np.isfinite(reweight_ess_stats["mean"]) else np.nan,
        "pooled_reweight_confidence_quality": str(pooled_reweight_confidence_quality),
        "pooled_reweight_confidence_note": str(pooled_reweight_confidence_note),
        "pooled_reweight_confidence_ok": bool(pooled_reweight_confidence_ok),
        "reweight_confidence_quality": str(selected_confidence_quality),
        "reweight_confidence_note": str(selected_confidence_note),
        "reweight_confidence_ok": bool(selected_confidence_ok),
        "dg_selected_kj": float(dg_selected_summary) if np.isfinite(dg_selected_summary) else np.nan,
        "kd_selected_m": float(kd_selected_summary) if np.isfinite(kd_selected_summary) else np.nan,
        "p_kd_selected": float(p_kd_selected_summary) if np.isfinite(p_kd_selected_summary) else np.nan,
        "selected_estimator": str(selected_estimator),
        "kd_app_m": float(kd_app_m) if np.isfinite(kd_app_m) else np.nan,
        "p_kd_app": float(p_kd_app) if np.isfinite(p_kd_app) else np.nan,
        "kd_app_jeff_m": float(kd_app_jeff_m) if np.isfinite(kd_app_jeff_m) else np.nan,
        "p_kd_jeff": float(p_kd_jeff) if np.isfinite(p_kd_jeff) else np.nan,
        "v_eff_nm3": float(v_eff_nm3) if np.isfinite(v_eff_nm3) else np.nan,
        "sim_ns_per_replica": float(np.nanmean(sim_ns_vals)) if sim_ns_vals else np.nan,
        "sim_ns_total": float(np.nansum(sim_ns_vals)) if sim_ns_vals else np.nan,
        "sample_every_ps": float(np.nanmean(sample_every_vals)) if sample_every_vals else np.nan,
        "unbound_coverage_fraction": float(unbound_cov),
        "bound_coverage_fraction": float(bound_rep_hits / float(max(rep_count, 1))),
        "exp_cum_gap_mean": float(gap_stats["mean"]) if np.isfinite(gap_stats["mean"]) else np.nan,
        "convergence_quality": str(conv_quality),
        "convergence_ok": bool(conv_quality in {"high", "medium"}),
        "convergence_note": str(conv_note),
    }
    return {
        "ok": True,
        "reason": "ready",
        "replica_count": int(rep_count),
        "requested_replica_count": int(req_count),
        "primary_replica_index": int(accepted_positions[primary_idx]),
        "replica_rows": list(rows),
        "summary": dict(summary),
        "pooled_numeric": dict(pooled_numeric or {}),
        "run_frames_common": np.asarray(run_frames_ref if run_frames_ref is not None else [], dtype=int),
        "run_exp_mean": np.asarray(run_exp_mean, dtype=float),
        "run_exp_sem": np.asarray(run_exp_sem, dtype=float),
        "run_cum_mean": np.asarray(run_cum_mean, dtype=float),
        "run_cum_sem": np.asarray(run_cum_sem, dtype=float),
    }


def build_abfe_free_energy_numeric_payload(
    *,
    frames_valid=None,
    bias_valid=None,
    total_valid=None,
    rescored_total_valid=None,
    ligand_internal_valid=None,
    rescored_energy_label="",
    distance_valid=None,
    coul_valid=None,
    bound_valid_mask=None,
    unbound_valid_mask=None,
    bound_cutoff_a=np.nan,
    n_bound=0,
    n_dist_valid=0,
    n_bound_raw=0,
    n_unbound_raw=0,
    n_bound_hold=0,
    n_unbound_hold=0,
    estimator_window_mode="state_fallback",
    use_count_fraction=False,
    temperature_k=300.0,
    v_eff_nm3=np.nan,
    bootstrap_seed=20260212,
    n_boot=300,
):
    total_raw = _abfe_float_series(total_valid)
    n_raw = int(total_raw.size)
    if n_raw <= 0:
        return {
            "ok": False,
            "reason": "no_samples",
            "n_samples": 0,
        }
    frames_raw = _abfe_int_series(frames_valid, n=n_raw, start=1)
    bias_raw = _abfe_float_series(bias_valid, n=n_raw)
    dist_raw = _abfe_float_series(distance_valid, n=n_raw)
    coul_raw = _abfe_float_series(coul_valid, n=n_raw)
    bound_raw = _abfe_bool_series(bound_valid_mask, n=n_raw, fill=False)
    unbound_raw = _abfe_bool_series(unbound_valid_mask, n=n_raw, fill=False)
    rescored_raw = (
        _abfe_float_series(rescored_total_valid, n=n_raw)
        if np.asarray(rescored_total_valid if rescored_total_valid is not None else []).size > 0
        else np.asarray([], dtype=float)
    )
    ligand_internal_raw = (
        _abfe_float_series(ligand_internal_valid, n=n_raw)
        if np.asarray(ligand_internal_valid if ligand_internal_valid is not None else []).size > 0
        else np.asarray([], dtype=float)
    )

    finite_total = np.isfinite(total_raw)
    if not np.any(finite_total):
        return {
            "ok": False,
            "reason": "no_finite_samples",
            "n_samples": 0,
        }
    frames_arr = np.asarray(frames_raw[finite_total], dtype=int)
    bias_arr = np.asarray(bias_raw[finite_total], dtype=float)
    total_arr = np.asarray(total_raw[finite_total], dtype=float)
    dist_arr = np.asarray(dist_raw[finite_total], dtype=float)
    coul_arr = np.asarray(coul_raw[finite_total], dtype=float)
    bound_arr = np.asarray(bound_raw[finite_total], dtype=bool)
    unbound_arr = np.asarray(unbound_raw[finite_total], dtype=bool)
    rescored_arr = (
        np.asarray(rescored_raw[finite_total], dtype=float)
        if rescored_raw.size == n_raw
        else np.asarray([], dtype=float)
    )
    ligand_internal_arr = (
        np.asarray(ligand_internal_raw[finite_total], dtype=float)
        if ligand_internal_raw.size == n_raw
        else np.asarray([], dtype=float)
    )

    n_samples = int(total_arr.size)

    k_b = 0.008314462618
    kbt = float(k_b * float(temperature_k))
    beta = float(1.0 / kbt) if kbt > 0.0 else np.nan

    def _dg_expavg(samples):
        arr = np.array(samples, dtype=float)
        if arr.size == 0 or (not np.isfinite(beta)):
            return np.nan
        e_shift = float(np.min(arr))
        w = np.exp(-beta * (arr - e_shift))
        return float(e_shift - (1.0 / beta) * np.log(np.mean(w)))

    dg_exp = _dg_expavg(total_arr)
    mean_e = float(np.mean(total_arr))
    var_e = float(np.var(total_arr, ddof=1 if total_arr.size > 1 else 0))
    dg_cum = float(mean_e - 0.5 * beta * var_e) if np.isfinite(beta) else np.nan
    coul_mean = float(np.mean(coul_arr)) if coul_arr.size else np.nan

    contact_frac_mean = np.nan
    if bool(use_count_fraction) and int(n_dist_valid) > 0:
        contact_frac_mean = float(np.clip(float(n_bound) / float(max(int(n_dist_valid), 1)), 0.0, 1.0))
    if not np.isfinite(contact_frac_mean):
        finite_min_for_contact = np.isfinite(dist_arr)
        if np.any(finite_min_for_contact):
            contact_cut_a = float(bound_cutoff_a) if np.isfinite(float(bound_cutoff_a)) and float(bound_cutoff_a) > 0.0 else 4.5
            contact_frac_mean = float(np.mean(dist_arr[finite_min_for_contact] <= contact_cut_a))

    try:
        n_boot_i = int(n_boot)
    except Exception:
        n_boot_i = 300
    n_boot_i = max(1, int(n_boot_i))

    reweight_payload = build_abfe_bias_reweighted_population_payload(
        bound_valid_mask=np.asarray(bound_arr, dtype=bool),
        unbound_valid_mask=np.asarray(unbound_arr, dtype=bool),
        bias_valid=np.asarray(bias_arr, dtype=float),
        temperature_k=float(temperature_k),
        v_eff_nm3=float(v_eff_nm3),
        bootstrap_seed=int(bootstrap_seed) + 17,
        n_boot=int(n_boot_i),
    )

    rng = np.random.default_rng(int(bootstrap_seed))
    boot = np.empty(n_boot_i, dtype=float)
    for b in range(n_boot_i):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot[b] = _dg_expavg(total_arr[idx])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    run_frames = []
    run_exp = []
    run_cum = []
    for i in range(3, n_samples + 1):
        seg = total_arr[:i]
        run_frames.append(int(frames_arr[i - 1]) if (i - 1) < frames_arr.size else int(i))
        run_exp.append(_dg_expavg(seg))
        m = float(np.mean(seg))
        v = float(np.var(seg, ddof=1 if seg.size > 1 else 0))
        run_cum.append(float(m - 0.5 * beta * v) if np.isfinite(beta) else np.nan)
    run_frames = np.array(run_frames, dtype=int)
    run_exp = np.array(run_exp, dtype=float)
    run_cum = np.array(run_cum, dtype=float)

    e_bins = int(np.clip(np.sqrt(total_arr.size), 12, 60))
    hist_weights = None
    if bool(reweight_payload.get("ok", False)):
        reweight_weights = np.asarray(reweight_payload.get("weights", []), dtype=float)
        if reweight_weights.size == total_arr.size and np.any(np.isfinite(reweight_weights)):
            hist_weights = np.asarray(np.clip(reweight_weights, 0.0, None), dtype=float)
    e_hist, e_edges = np.histogram(total_arr, bins=e_bins, density=False, weights=hist_weights)
    if np.any(np.isfinite(e_hist)) and float(np.nansum(e_hist)) > 0.0:
        e_hist = np.asarray(e_hist, dtype=float) / float(np.nansum(e_hist))
    e_cent = 0.5 * (e_edges[:-1] + e_edges[1:])
    e_pmf = -kbt * np.log(np.clip(e_hist, 1e-12, None))
    e_pmf = e_pmf - float(np.nanmin(e_pmf))

    finite_min = np.isfinite(dist_arr)
    r_cent = np.array([])
    r_pmf = np.array([])
    if np.any(finite_min):
        r_vals = dist_arr[finite_min]
        r_bins = int(np.clip(np.sqrt(r_vals.size), 10, 50))
        r_weights = None
        if hist_weights is not None and hist_weights.size == dist_arr.size:
            r_weights = np.asarray(hist_weights[finite_min], dtype=float)
        r_hist, r_edges = np.histogram(r_vals, bins=r_bins, density=False, weights=r_weights)
        if np.any(np.isfinite(r_hist)) and float(np.nansum(r_hist)) > 0.0:
            r_hist = np.asarray(r_hist, dtype=float) / float(np.nansum(r_hist))
        r_cent = 0.5 * (r_edges[:-1] + r_edges[1:])
        r_pmf = -kbt * np.log(np.clip(r_hist, 1e-12, None))
        r_pmf = r_pmf - float(np.nanmin(r_pmf))

    def _weighted_state_center(mask):
        arr_mask = np.asarray(mask if mask is not None else [], dtype=bool)
        if arr_mask.size != dist_arr.size:
            return None
        arr_mask = np.logical_and(arr_mask, np.isfinite(dist_arr))
        arr_mask = np.logical_and(arr_mask, np.isfinite(total_arr))
        if not np.any(arr_mask):
            return None
        weights_sel = None
        if hist_weights is not None and hist_weights.size == dist_arr.size:
            weights_sel = np.asarray(hist_weights[arr_mask], dtype=float)
            if (not np.any(np.isfinite(weights_sel))) or float(np.nansum(np.clip(weights_sel, 0.0, None))) <= 0.0:
                weights_sel = None
        x_vals = np.asarray(dist_arr[arr_mask], dtype=float)
        y_vals = np.asarray(total_arr[arr_mask], dtype=float)
        if weights_sel is None:
            x_mid = float(np.mean(x_vals))
            y_mid = float(np.mean(y_vals))
        else:
            ww = np.asarray(np.clip(weights_sel, 0.0, None), dtype=float)
            x_mid = float(np.average(x_vals, weights=ww))
            y_mid = float(np.average(y_vals, weights=ww))
        return {
            "x": float(x_mid),
            "y": float(y_mid),
            "n": int(np.sum(arr_mask)),
        }

    has_surface = False
    Xc = None
    Yc = None
    Fsurf = None
    surface_clip_max_kj = np.nan
    bound_center = _weighted_state_center(bound_arr)
    unbound_center = _weighted_state_center(unbound_arr)
    if np.sum(finite_min) >= 20:
        rr = dist_arr[finite_min]
        ee = total_arr[finite_min]
        xbins = int(np.clip(np.sqrt(rr.size), 12, 45))
        ybins = int(np.clip(np.sqrt(ee.size), 12, 45))
        surf_weights = None
        if hist_weights is not None and hist_weights.size == dist_arr.size:
            surf_weights = np.asarray(hist_weights[finite_min], dtype=float)
        H_mass, xedges, yedges = np.histogram2d(rr, ee, bins=(xbins, ybins), density=False, weights=surf_weights)
        H_count, _xedges_raw, _yedges_raw = np.histogram2d(rr, ee, bins=(xedges, yedges), density=False)
        total_mass = float(np.nansum(H_mass))
        if total_mass > 0.0:
            P = np.asarray(H_mass, dtype=float) / total_mass
            F = np.full_like(P, np.nan, dtype=float)
            occupied_mask = np.asarray(H_count >= 2.0, dtype=bool)
            if np.any(occupied_mask):
                F[occupied_mask] = -kbt * np.log(np.clip(P[occupied_mask], 1.0e-12, None))
                finite_F = np.asarray(F[np.isfinite(F)], dtype=float)
                if finite_F.size > 0:
                    F = F - float(np.nanmin(finite_F))
                    clip_seed = float(np.nanpercentile(finite_F, 96.0)) if finite_F.size >= 6 else float(np.nanmax(finite_F))
                    surface_clip_max_kj = float(np.clip(clip_seed, 3.0, max(3.0, float(np.nanmax(finite_F)))))
                    Xc = 0.5 * (xedges[:-1] + xedges[1:])
                    Yc = 0.5 * (yedges[:-1] + yedges[1:])
                    Fsurf = F.T
                    has_surface = True

    def _safe_int(value, default=0):
        try:
            return int(value)
        except Exception:
            return int(default)

    mode_txt = str(estimator_window_mode or "").strip().lower()
    hold_only = bool(mode_txt == "hold_only")
    n_bound_eff = int(np.sum(bound_arr)) if bound_arr.size else int(max(0, _safe_int(n_bound, 0)))
    n_unbound_eff = int(np.sum(unbound_arr)) if unbound_arr.size else int(max(0, _safe_int(n_dist_valid, 0) - _safe_int(n_bound, 0)))
    hold_bound_n = int(max(0, _safe_int(n_bound_hold, n_bound_eff)))
    hold_unbound_n = int(max(0, _safe_int(n_unbound_hold, n_unbound_eff)))
    ess_total = _safe_float(reweight_payload.get("ess_total", np.nan))
    ess_bound = _safe_float(reweight_payload.get("ess_bound", np.nan))
    ess_unbound = _safe_float(reweight_payload.get("ess_unbound", np.nan))
    rescored_label_txt = str(rescored_energy_label or "").strip().lower()
    if not rescored_label_txt:
        rescored_label_txt = "gbsa_obc2"
    rescored_ok = bool(rescored_arr.size == total_arr.size and np.any(np.isfinite(rescored_arr)))
    rescored_bound_expavg = np.nan
    rescored_unbound_expavg = np.nan
    rescored_bound_mean = np.nan
    rescored_unbound_mean = np.nan
    rescored_state_expavg = np.nan
    rescored_state_mean = np.nan
    rescored_bound_blockexpavg = np.nan
    rescored_unbound_blockexpavg = np.nan
    rescored_state_blockexpavg = np.nan
    rescored_bound_expavg_strain = np.nan
    rescored_unbound_expavg_strain = np.nan
    rescored_state_expavg_strain = np.nan
    rescored_bound_blockexpavg_strain = np.nan
    rescored_unbound_blockexpavg_strain = np.nan
    rescored_state_blockexpavg_strain = np.nan
    ligand_internal_ok = bool(ligand_internal_arr.size == rescored_arr.size and np.any(np.isfinite(ligand_internal_arr)))
    ligand_internal_bound_mean = np.nan
    ligand_internal_unbound_mean = np.nan
    strain_penalty_bound_mean = np.nan
    strain_penalty_unbound_mean = np.nan
    dg_ligand_strain = np.nan
    block_size_bound = 0
    block_size_unbound = 0
    if rescored_ok:
        state_weights = None
        if hist_weights is not None and hist_weights.size == rescored_arr.size:
            state_weights = np.asarray(hist_weights, dtype=float)
        bound_rescore_mask = np.logical_and(np.asarray(bound_arr, dtype=bool), np.isfinite(rescored_arr))
        unbound_rescore_mask = np.logical_and(np.asarray(unbound_arr, dtype=bool), np.isfinite(rescored_arr))
        strain_penalty_arr = np.asarray([], dtype=float)
        if ligand_internal_ok:
            ligand_internal_finite = np.asarray(np.isfinite(ligand_internal_arr), dtype=bool)
            ligand_internal_ref_pool = np.asarray(ligand_internal_arr[ligand_internal_finite], dtype=float)
            if ligand_internal_ref_pool.size > 0:
                internal_ref = float(np.nanquantile(ligand_internal_ref_pool, 0.10))
                strain_penalty_arr = np.asarray(ligand_internal_arr - internal_ref, dtype=float)
                strain_penalty_arr = np.where(np.isfinite(strain_penalty_arr), np.clip(strain_penalty_arr, 0.0, None), np.nan)
                positive_penalty = np.asarray(strain_penalty_arr[np.isfinite(strain_penalty_arr) & (strain_penalty_arr > 0.0)], dtype=float)
                if positive_penalty.size >= 4:
                    penalty_cap = float(np.nanquantile(positive_penalty, 0.90))
                    if np.isfinite(penalty_cap) and penalty_cap > 0.0:
                        strain_penalty_arr = np.where(np.isfinite(strain_penalty_arr), np.clip(strain_penalty_arr, 0.0, penalty_cap), np.nan)
        if np.any(bound_rescore_mask):
            bound_vals = np.asarray(rescored_arr[bound_rescore_mask], dtype=float)
            bound_w = (
                np.asarray(state_weights[bound_rescore_mask], dtype=float)
                if state_weights is not None and state_weights.size == rescored_arr.size
                else None
            )
            rescored_bound_expavg = _abfe_weighted_expavg(bound_vals, bound_w, beta)
            block_size_bound = _abfe_default_block_size(bound_vals.size)
            rescored_bound_blockexpavg = _abfe_weighted_block_expavg(bound_vals, bound_w, beta, block_size=block_size_bound)
            rescored_bound_mean = (
                float(np.average(bound_vals, weights=np.asarray(bound_w, dtype=float)))
                if bound_w is not None and np.any(np.isfinite(bound_w)) and float(np.sum(bound_w)) > 0.0
                else float(np.mean(bound_vals))
            )
            if ligand_internal_ok and strain_penalty_arr.size == rescored_arr.size:
                bound_strain_vals = np.asarray(rescored_arr[bound_rescore_mask] + strain_penalty_arr[bound_rescore_mask], dtype=float)
                rescored_bound_expavg_strain = _abfe_weighted_expavg(bound_strain_vals, bound_w, beta)
                rescored_bound_blockexpavg_strain = _abfe_weighted_block_expavg(bound_strain_vals, bound_w, beta, block_size=block_size_bound)
                ligand_internal_bound_mean = _abfe_weighted_mean(ligand_internal_arr[bound_rescore_mask], bound_w)
                if strain_penalty_arr.size == rescored_arr.size:
                    strain_penalty_bound_mean = _abfe_weighted_mean(strain_penalty_arr[bound_rescore_mask], bound_w)
        if np.any(unbound_rescore_mask):
            unbound_vals = np.asarray(rescored_arr[unbound_rescore_mask], dtype=float)
            unbound_w = (
                np.asarray(state_weights[unbound_rescore_mask], dtype=float)
                if state_weights is not None and state_weights.size == rescored_arr.size
                else None
            )
            rescored_unbound_expavg = _abfe_weighted_expavg(unbound_vals, unbound_w, beta)
            block_size_unbound = _abfe_default_block_size(unbound_vals.size)
            rescored_unbound_blockexpavg = _abfe_weighted_block_expavg(unbound_vals, unbound_w, beta, block_size=block_size_unbound)
            rescored_unbound_mean = (
                float(np.average(unbound_vals, weights=np.asarray(unbound_w, dtype=float)))
                if unbound_w is not None and np.any(np.isfinite(unbound_w)) and float(np.sum(unbound_w)) > 0.0
                else float(np.mean(unbound_vals))
            )
            if ligand_internal_ok:
                rescored_unbound_expavg_strain = float(rescored_unbound_expavg) if np.isfinite(rescored_unbound_expavg) else np.nan
                rescored_unbound_blockexpavg_strain = float(rescored_unbound_blockexpavg) if np.isfinite(rescored_unbound_blockexpavg) else np.nan
                ligand_internal_unbound_mean = _abfe_weighted_mean(ligand_internal_arr[unbound_rescore_mask], unbound_w)
                strain_penalty_unbound_mean = 0.0
        if np.isfinite(rescored_bound_expavg) and np.isfinite(rescored_unbound_expavg):
            rescored_state_expavg = float(rescored_bound_expavg - rescored_unbound_expavg)
        if np.isfinite(rescored_bound_blockexpavg) and np.isfinite(rescored_unbound_blockexpavg):
            rescored_state_blockexpavg = float(rescored_bound_blockexpavg - rescored_unbound_blockexpavg)
        if np.isfinite(rescored_bound_mean) and np.isfinite(rescored_unbound_mean):
            rescored_state_mean = float(rescored_bound_mean - rescored_unbound_mean)
        if np.isfinite(rescored_bound_expavg_strain) and np.isfinite(rescored_unbound_expavg_strain):
            rescored_state_expavg_strain = float(rescored_bound_expavg_strain - rescored_unbound_expavg_strain)
        if np.isfinite(rescored_bound_blockexpavg_strain) and np.isfinite(rescored_unbound_blockexpavg_strain):
            rescored_state_blockexpavg_strain = float(rescored_bound_blockexpavg_strain - rescored_unbound_blockexpavg_strain)
        if np.isfinite(strain_penalty_bound_mean) and np.isfinite(strain_penalty_unbound_mean):
            dg_ligand_strain = float(strain_penalty_bound_mean - strain_penalty_unbound_mean)
        elif np.isfinite(ligand_internal_bound_mean) and np.isfinite(ligand_internal_unbound_mean):
            dg_ligand_strain = float(ligand_internal_bound_mean - ligand_internal_unbound_mean)
    ci_width = np.nan
    if np.isfinite(_safe_float(reweight_payload.get("ci_low", np.nan))) and np.isfinite(_safe_float(reweight_payload.get("ci_high", np.nan))):
        ci_width = float(reweight_payload.get("ci_high", np.nan)) - float(reweight_payload.get("ci_low", np.nan))
    confidence_quality = "low"
    confidence_note = "hold-window coverage or effective sample size is weak"
    confidence_ok = False
    if (
        hold_only
        and np.isfinite(ess_total)
        and np.isfinite(ess_bound)
        and np.isfinite(ess_unbound)
        and ess_total >= 20.0
        and ess_bound >= 8.0
        and ess_unbound >= 8.0
        and hold_bound_n >= 8
        and hold_unbound_n >= 8
        and ((not np.isfinite(ci_width)) or (ci_width <= 12.0))
    ):
        confidence_quality = "high"
        confidence_note = "stable hold windows are present for both states and reweighting ESS is strong"
        confidence_ok = True
    elif (
        hold_only
        and np.isfinite(ess_total)
        and np.isfinite(ess_bound)
        and np.isfinite(ess_unbound)
        and ess_total >= 10.0
        and ess_bound >= 4.0
        and ess_unbound >= 4.0
        and hold_bound_n >= 4
        and hold_unbound_n >= 4
    ):
        confidence_quality = "medium"
        confidence_note = "hold-window estimate is usable, but more stable windows would tighten confidence"
        confidence_ok = True
    elif not hold_only:
        confidence_note = "estimator fell back to broad state windows because hold windows were insufficient"

    dg_selected_state = float(reweight_payload.get("dg_reweighted_state_kj", reweight_payload.get("dg_reweighted_kj", np.nan))) if confidence_ok else np.nan
    dg_selected_standard = float(reweight_payload.get("dg_reweighted_standard_kj", np.nan)) if confidence_ok else np.nan
    dg_selected = (
        float(dg_selected_standard)
        if confidence_ok and np.isfinite(float(dg_selected_standard))
        else (float(dg_selected_state) if confidence_ok else np.nan)
    )
    kd_selected = float(reweight_payload.get("kd_standard_reweighted_m", reweight_payload.get("kd_app_reweighted_m", np.nan))) if confidence_ok else np.nan
    p_kd_selected = float(reweight_payload.get("p_kd_reweighted", np.nan)) if confidence_ok else np.nan
    selected_estimator = "bias_reweighted_hold_windows" if confidence_ok else "gated_off"
    estimate_kind = "standard_state_apparent" if confidence_ok else "gated_off"
    if confidence_ok and hold_only and np.isfinite(rescored_state_blockexpavg):
        dg_selected = float(rescored_state_blockexpavg)
        dg_selected_state = np.nan
        dg_selected_standard = float(dg_selected)
        if np.isfinite(kbt) and kbt > 0.0:
            kd_selected = float(np.exp(np.clip(float(dg_selected) / float(kbt), -120.0, 120.0)))
            p_kd_selected = float(-np.log10(max(kd_selected, 1.0e-300)))
        else:
            kd_selected = np.nan
            p_kd_selected = np.nan
        selected_estimator = f"{rescored_label_txt}_hold_window_blockexpavg"
        estimate_kind = "rescored_hold_window_apparent"
    elif confidence_ok and hold_only and np.isfinite(rescored_state_expavg):
        dg_selected = float(rescored_state_expavg)
        dg_selected_state = np.nan
        dg_selected_standard = float(dg_selected)
        if np.isfinite(kbt) and kbt > 0.0:
            kd_selected = float(np.exp(np.clip(float(dg_selected) / float(kbt), -120.0, 120.0)))
            p_kd_selected = float(-np.log10(max(kd_selected, 1.0e-300)))
        else:
            kd_selected = np.nan
            p_kd_selected = np.nan
        selected_estimator = f"{rescored_label_txt}_hold_window_expavg"
        estimate_kind = "rescored_hold_window_apparent"
    elif confidence_ok and hold_only and np.isfinite(rescored_state_expavg_strain):
        dg_selected = float(rescored_state_expavg_strain)
        dg_selected_state = np.nan
        dg_selected_standard = float(dg_selected)
        if np.isfinite(kbt) and kbt > 0.0:
            kd_selected = float(np.exp(np.clip(float(dg_selected) / float(kbt), -120.0, 120.0)))
            p_kd_selected = float(-np.log10(max(kd_selected, 1.0e-300)))
        else:
            kd_selected = np.nan
            p_kd_selected = np.nan
        selected_estimator = f"{rescored_label_txt}_hold_window_expavg_strain"
        estimate_kind = "rescored_hold_window_apparent"

    production_reliability = "diagnostic"
    production_note = "Single short ABFE-lite runs are workflow diagnostics; quantitative ABFE requires replicas and convergence checks."
    if confidence_ok:
        if hold_bound_n >= 50 and hold_unbound_n >= 50 and np.isfinite(ess_unbound) and ess_unbound >= 25.0:
            production_reliability = "apparent"
            production_note = "Bound/unbound hold windows are populated; report as an apparent short-run estimate, not production ABFE."
        else:
            production_reliability = "qualitative"
            production_note = "State separation is present, but unbound hold-window coverage is too small for a production affinity claim."

    return {
        "ok": True,
        "n_samples": int(n_samples),
        "kbt": float(kbt),
        "beta": float(beta) if np.isfinite(beta) else np.nan,
        "dg_exp": float(dg_exp) if np.isfinite(dg_exp) else np.nan,
        "dg_cum": float(dg_cum) if np.isfinite(dg_cum) else np.nan,
        "coul_mean": float(coul_mean) if np.isfinite(coul_mean) else np.nan,
        "contact_frac_mean": float(contact_frac_mean) if np.isfinite(contact_frac_mean) else np.nan,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_bound_reweighted": float(reweight_payload.get("p_bound_reweighted", np.nan)),
        "p_unbound_reweighted": float(reweight_payload.get("p_unbound_reweighted", np.nan)),
        "dg_reweighted_kj": float(reweight_payload.get("dg_reweighted_kj", np.nan)),
        "dg_reweighted_state_kj": float(reweight_payload.get("dg_reweighted_state_kj", np.nan)),
        "dg_reweighted_standard_kj": float(reweight_payload.get("dg_reweighted_standard_kj", np.nan)),
        "standard_state_correction_kj": float(reweight_payload.get("standard_state_correction_kj", np.nan)),
        "c_eff_m": float(reweight_payload.get("c_eff_m", np.nan)),
        "kd_app_reweighted_m": float(reweight_payload.get("kd_app_reweighted_m", np.nan)),
        "kd_standard_reweighted_m": float(reweight_payload.get("kd_standard_reweighted_m", np.nan)),
        "p_kd_reweighted": float(reweight_payload.get("p_kd_reweighted", np.nan)),
        "rescored_energy_label": str(rescored_label_txt),
        "rescored_energy_available": bool(rescored_ok),
        "ligand_internal_available": bool(ligand_internal_ok),
        "rescored_total_valid": np.asarray(rescored_arr, dtype=float) if rescored_ok else np.asarray([], dtype=float),
        "dg_rescored_state_expavg_kj": float(rescored_state_expavg) if np.isfinite(rescored_state_expavg) else np.nan,
        "dg_rescored_state_blockexpavg_kj": float(rescored_state_blockexpavg) if np.isfinite(rescored_state_blockexpavg) else np.nan,
        "dg_rescored_state_expavg_strain_kj": float(rescored_state_expavg_strain) if np.isfinite(rescored_state_expavg_strain) else np.nan,
        "dg_rescored_state_blockexpavg_strain_kj": float(rescored_state_blockexpavg_strain) if np.isfinite(rescored_state_blockexpavg_strain) else np.nan,
        "dg_rescored_state_mean_kj": float(rescored_state_mean) if np.isfinite(rescored_state_mean) else np.nan,
        "rescored_bound_expavg_kj": float(rescored_bound_expavg) if np.isfinite(rescored_bound_expavg) else np.nan,
        "rescored_unbound_expavg_kj": float(rescored_unbound_expavg) if np.isfinite(rescored_unbound_expavg) else np.nan,
        "rescored_bound_blockexpavg_kj": float(rescored_bound_blockexpavg) if np.isfinite(rescored_bound_blockexpavg) else np.nan,
        "rescored_unbound_blockexpavg_kj": float(rescored_unbound_blockexpavg) if np.isfinite(rescored_unbound_blockexpavg) else np.nan,
        "rescored_bound_expavg_strain_kj": float(rescored_bound_expavg_strain) if np.isfinite(rescored_bound_expavg_strain) else np.nan,
        "rescored_unbound_expavg_strain_kj": float(rescored_unbound_expavg_strain) if np.isfinite(rescored_unbound_expavg_strain) else np.nan,
        "rescored_bound_blockexpavg_strain_kj": float(rescored_bound_blockexpavg_strain) if np.isfinite(rescored_bound_blockexpavg_strain) else np.nan,
        "rescored_unbound_blockexpavg_strain_kj": float(rescored_unbound_blockexpavg_strain) if np.isfinite(rescored_unbound_blockexpavg_strain) else np.nan,
        "rescored_bound_mean_kj": float(rescored_bound_mean) if np.isfinite(rescored_bound_mean) else np.nan,
        "rescored_unbound_mean_kj": float(rescored_unbound_mean) if np.isfinite(rescored_unbound_mean) else np.nan,
        "ligand_internal_bound_mean_kj": float(ligand_internal_bound_mean) if np.isfinite(ligand_internal_bound_mean) else np.nan,
        "ligand_internal_unbound_mean_kj": float(ligand_internal_unbound_mean) if np.isfinite(ligand_internal_unbound_mean) else np.nan,
        "dg_ligand_strain_kj": float(dg_ligand_strain) if np.isfinite(dg_ligand_strain) else np.nan,
        "rescored_block_size_bound": int(block_size_bound),
        "rescored_block_size_unbound": int(block_size_unbound),
        "reweight_ci_low": float(reweight_payload.get("ci_low", np.nan)),
        "reweight_ci_high": float(reweight_payload.get("ci_high", np.nan)),
        "reweight_ess_total": float(reweight_payload.get("ess_total", np.nan)),
        "reweight_ess_bound": float(reweight_payload.get("ess_bound", np.nan)),
        "reweight_ess_unbound": float(reweight_payload.get("ess_unbound", np.nan)),
        "reweight_clip_percentile_used": float(reweight_payload.get("clip_percentile_used", np.nan)),
        "n_bound_raw": int(max(0, _safe_int(n_bound_raw, 0))),
        "n_unbound_raw": int(max(0, _safe_int(n_unbound_raw, 0))),
        "n_bound_hold": int(hold_bound_n),
        "n_unbound_hold": int(hold_unbound_n),
        "estimator_window_mode": str(mode_txt or "state_fallback"),
        "reweight_confidence_quality": str(confidence_quality),
        "reweight_confidence_note": str(confidence_note),
        "reweight_confidence_ok": bool(confidence_ok),
        "dg_selected_kj": float(dg_selected) if np.isfinite(dg_selected) else np.nan,
        "dg_selected_state_kj": float(dg_selected_state) if np.isfinite(dg_selected_state) else np.nan,
        "dg_selected_standard_kj": float(dg_selected_standard) if np.isfinite(dg_selected_standard) else np.nan,
        "kd_selected_m": float(kd_selected) if np.isfinite(kd_selected) else np.nan,
        "kd_selected_standard_m": float(kd_selected) if np.isfinite(kd_selected) else np.nan,
        "p_kd_selected": float(p_kd_selected) if np.isfinite(p_kd_selected) else np.nan,
        "selected_estimator": str(selected_estimator),
        "estimate_kind": str(estimate_kind),
        "production_reliability": str(production_reliability),
        "production_note": str(production_note),
        "run_frames": np.asarray(run_frames, dtype=int),
        "run_exp": np.asarray(run_exp, dtype=float),
        "run_cum": np.asarray(run_cum, dtype=float),
        "e_cent": np.asarray(e_cent, dtype=float),
        "e_pmf": np.asarray(e_pmf, dtype=float),
        "finite_min": np.asarray(finite_min, dtype=bool),
        "r_cent": np.asarray(r_cent, dtype=float),
        "r_pmf": np.asarray(r_pmf, dtype=float),
        "has_surface": bool(has_surface),
        "Xc": (np.asarray(Xc, dtype=float) if Xc is not None else None),
        "Yc": (np.asarray(Yc, dtype=float) if Yc is not None else None),
        "Fsurf": (np.asarray(Fsurf, dtype=float) if Fsurf is not None else None),
        "surface_clip_max_kj": float(surface_clip_max_kj) if np.isfinite(surface_clip_max_kj) else np.nan,
        "bound_center": dict(bound_center or {}),
        "unbound_center": dict(unbound_center or {}),
    }


def build_abfe_free_energy_plot_payload(
    *,
    run_frames=None,
    run_exp=None,
    run_cum=None,
    dg_exp=np.nan,
    ci_low=np.nan,
    ci_high=np.nan,
    e_cent=None,
    e_pmf=None,
    r_cent=None,
    r_pmf=None,
    dist_axis_label="Distance (Å)",
    has_surface=False,
    Xc=None,
    Yc=None,
    Fsurf=None,
    surface_clip_max_kj=np.nan,
    bound_cutoff_a=np.nan,
    bound_center=None,
    unbound_center=None,
    replica_count=1,
    replica_run_exp_sem=None,
    replica_run_cum_sem=None,
    max_convergence_points=1200,
):
    run_frames_arr = np.asarray(run_frames if run_frames is not None else [], dtype=int)
    run_exp_arr = np.asarray(run_exp if run_exp is not None else [], dtype=float)
    run_cum_arr = np.asarray(run_cum if run_cum is not None else [], dtype=float)
    e_cent_arr = np.asarray(e_cent if e_cent is not None else [], dtype=float)
    e_pmf_arr = np.asarray(e_pmf if e_pmf is not None else [], dtype=float)
    r_cent_arr = np.asarray(r_cent if r_cent is not None else [], dtype=float)
    r_pmf_arr = np.asarray(r_pmf if r_pmf is not None else [], dtype=float)
    bound_center_row = dict(bound_center or {})
    unbound_center_row = dict(unbound_center or {})
    run_exp_sem_arr = np.asarray(replica_run_exp_sem if replica_run_exp_sem is not None else [], dtype=float)
    run_cum_sem_arr = np.asarray(replica_run_cum_sem if replica_run_cum_sem is not None else [], dtype=float)
    n_conv_raw = int(min(run_frames_arr.size, run_exp_arr.size, run_cum_arr.size))
    conv_stride = 1
    if n_conv_raw > 0:
        run_frames_arr = np.asarray(run_frames_arr[:n_conv_raw], dtype=int)
        run_exp_arr = np.asarray(run_exp_arr[:n_conv_raw], dtype=float)
        run_cum_arr = np.asarray(run_cum_arr[:n_conv_raw], dtype=float)
        if run_exp_sem_arr.size >= n_conv_raw:
            run_exp_sem_arr = np.asarray(run_exp_sem_arr[:n_conv_raw], dtype=float)
        if run_cum_sem_arr.size >= n_conv_raw:
            run_cum_sem_arr = np.asarray(run_cum_sem_arr[:n_conv_raw], dtype=float)
        max_conv = _abfe_positive_int(max_convergence_points, default=1200, minimum=50)
        pick_conv = _abfe_downsample_indices(n_conv_raw, max_points=max_conv, include_last=True)
        if pick_conv.size and pick_conv.size < n_conv_raw:
            conv_stride = int(max(1, np.ceil(float(n_conv_raw) / float(max(1, pick_conv.size)))))
            run_frames_arr = np.asarray(run_frames_arr[pick_conv], dtype=int)
            run_exp_arr = np.asarray(run_exp_arr[pick_conv], dtype=float)
            run_cum_arr = np.asarray(run_cum_arr[pick_conv], dtype=float)
            if run_exp_sem_arr.size == n_conv_raw:
                run_exp_sem_arr = np.asarray(run_exp_sem_arr[pick_conv], dtype=float)
            if run_cum_sem_arr.size == n_conv_raw:
                run_cum_sem_arr = np.asarray(run_cum_sem_arr[pick_conv], dtype=float)

    has_rpmf = bool(r_cent_arr.size > 0 and r_pmf_arr.size > 0)
    has_surface_i = bool(has_surface) and (Xc is not None) and (Yc is not None) and (Fsurf is not None)
    has_rep_overlay = bool(
        int(replica_count or 1) > 1
        and run_frames_arr.size > 0
        and run_exp_arr.size == run_frames_arr.size
        and run_exp_sem_arr.size == run_frames_arr.size
    )
    surf_vmax = _safe_float(surface_clip_max_kj)

    def _marker_payload(center_row, label, color, marker):
        row = dict(center_row or {})
        x = _safe_float(row.get("x", np.nan))
        y = _safe_float(row.get("y", np.nan))
        return {
            "enabled": bool(np.isfinite(x) and np.isfinite(y)),
            "x": float(x) if np.isfinite(x) else np.nan,
            "y": float(y) if np.isfinite(y) else np.nan,
            "n": int(max(0, int(row.get("n", 0) or 0))),
            "label": str(label),
            "color": str(color),
            "marker": str(marker),
        }

    return {
        "ok": True,
        "theme": {
            "figure_facecolor": "#F7F8FA",
            "axes_facecolor": "#FFFFFF",
            "spine_color": "#D1D5DB",
            "tick_color": "#374151",
            "title_color": "#111827",
            "subtitle_color": "#6B7280",
            "grid_color": "#E5E7EB",
        },
        "convergence": {
            "n_raw": int(n_conv_raw),
            "plot_stride": int(conv_stride),
            "x_exp": np.asarray(run_frames_arr, dtype=int),
            "y_exp": np.asarray(run_exp_arr, dtype=float),
            "x_cum": np.asarray(run_frames_arr, dtype=int),
            "y_cum": np.asarray(run_cum_arr, dtype=float),
            "line_exp": {"color": "#111827", "linewidth": 1.9, "label": "ΔG exp-avg"},
            "line_cum": {"color": "#6B7280", "linewidth": 1.5, "label": "ΔG cumulant"},
            "hline_exp": {"y": float(dg_exp), "color": "#111827", "linestyle": "--", "linewidth": 1.0, "alpha": 0.8},
            "ci_band": {
                "ymin": float(ci_low),
                "ymax": float(ci_high),
                "color": "#6B7280",
                "alpha": 0.12,
                "label": "95% CI (bootstrap)",
            },
            "xlabel": "Frame",
            "ylabel": "ΔG estimate (kJ/mol)",
            "title": "Free-Energy Convergence",
            "grid_alpha": 0.18,
            "legend": {"loc": "best", "fontsize": 8},
            "replica_overlay": {
                "enabled": bool(has_rep_overlay),
                "replica_count": int(max(1, int(replica_count or 1))),
                "x": np.asarray(run_frames_arr, dtype=int),
                "exp_mean": np.asarray(run_exp_arr, dtype=float),
                "exp_sem": np.asarray(run_exp_sem_arr, dtype=float),
                "cum_mean": np.asarray(run_cum_arr, dtype=float),
                "cum_sem": np.asarray(run_cum_sem_arr, dtype=float),
                "exp_band": {"color": "#111827", "alpha": 0.10, "label": "Replica exp ± SEM"},
                "cum_band": {"color": "#6B7280", "alpha": 0.09, "label": "Replica cum ± SEM"},
            },
        },
        "energy_pmf": {
            "x": np.asarray(e_cent_arr, dtype=float),
            "y": np.asarray(e_pmf_arr, dtype=float),
            "line": {"color": "#374151", "linewidth": 1.8},
            "fill_color": "#D1D5DB",
            "fill_alpha": 0.22,
            "xlabel": "Interaction Energy (kJ/mol)",
            "ylabel": "PMF (kJ/mol)",
            "title": "PMF from Energy Distribution",
            "grid_alpha": 0.18,
        },
        "distance_pmf": {
            "has_data": bool(has_rpmf),
            "x": np.asarray(r_cent_arr, dtype=float),
            "y": np.asarray(r_pmf_arr, dtype=float),
            "line": {"color": "#4B5563", "linewidth": 1.8},
            "fill_color": "#E5E7EB",
            "fill_alpha": 0.22,
            "xlabel": str(dist_axis_label),
            "ylabel": "PMF (kJ/mol)",
            "title": "PMF along ABFE Distance Coordinate",
            "grid_alpha": 0.18,
            "bound_cutoff_line": {
                "enabled": bool(np.isfinite(_safe_float(bound_cutoff_a))),
                "x": float(bound_cutoff_a) if np.isfinite(_safe_float(bound_cutoff_a)) else np.nan,
                "color": "#111827",
                "linestyle": "--",
                "linewidth": 1.0,
                "label": "Bound cutoff",
            },
            "no_data_text": "No valid distance samples",
        },
        "surface": {
            "has_data": bool(has_surface_i),
            "Xc": (np.asarray(Xc, dtype=float) if Xc is not None else None),
            "Yc": (np.asarray(Yc, dtype=float) if Yc is not None else None),
            "Fsurf": (np.asarray(Fsurf, dtype=float) if Fsurf is not None else None),
            "levels": 28,
            "contour_levels": 9,
            "cmap": "Greys_r",
            "contour_color": "#111827",
            "vmax": float(surf_vmax) if np.isfinite(surf_vmax) else np.nan,
            "cbar_label": "Free Energy (kJ/mol)",
            "xlabel": str(dist_axis_label),
            "ylabel": "Interaction Energy (kJ/mol)",
            "title": "2D Free-Energy Surface",
            "bound_marker": _marker_payload(bound_center_row, "bound basin", "#111827", "o"),
            "unbound_marker": _marker_payload(unbound_center_row, "unbound basin", "#6B7280", "^"),
            "no_data_text": "Insufficient samples for 2D surface",
        },
    }


def build_abfe_post_interaction_followup_plan(
    *,
    protocol="",
    abfe_traj_object="",
    object_exists=False,
    n_states=0,
    max_target_states=300,
    top_n=12,
):
    protocol_ok = bool(str(protocol or "").strip() == "abfe_lite_local_md")
    abfe_obj = str(abfe_traj_object or "").strip()
    exists = bool(object_exists)
    try:
        n_states_i = int(n_states or 0)
    except Exception:
        n_states_i = 0
    n_states_i = max(0, int(n_states_i))
    try:
        max_target_i = int(max_target_states or 300)
    except Exception:
        max_target_i = 300
    max_target_i = max(1, int(max_target_i))
    try:
        top_n_i = int(top_n or 12)
    except Exception:
        top_n_i = 12
    top_n_i = max(1, int(top_n_i))

    step = int(max(1, np.ceil(float(max(1, n_states_i)) / float(max_target_i))))
    should_run = bool(protocol_ok and abfe_obj and exists)

    reason = ""
    if not protocol_ok:
        reason = "protocol_not_local_md"
    elif not abfe_obj:
        reason = "missing_abfe_object"
    elif not exists:
        reason = "abfe_object_not_loaded"
    else:
        reason = "ready"

    return {
        "ok": True,
        "should_run": bool(should_run),
        "reason": str(reason),
        "protocol_ok": bool(protocol_ok),
        "abfe_object": str(abfe_obj),
        "object_exists": bool(exists),
        "n_states": int(n_states_i),
        "max_target_states": int(max_target_i),
        "step": int(step),
        "top_n": int(top_n_i),
    }


def prepare_abfe_post_interaction_followup_runtime_plan(
    *,
    protocol="",
    result_payload=None,
    get_names_fn=None,
    count_states_fn=None,
    max_target_states=300,
    top_n=12,
    followup_plan_builder_fn=None,
):
    data = dict(result_payload or {})
    abfe_obj = str(data.get("abfe_traj_object", "") or "").strip()

    object_names = []
    if callable(get_names_fn):
        object_names = list(get_names_fn("objects") or [])
    object_exists = bool(abfe_obj and any(str(name) == abfe_obj for name in object_names))

    n_states = 0
    if object_exists and callable(count_states_fn):
        try:
            n_states = int(count_states_fn(abfe_obj))
        except Exception:
            n_states = 0

    builder = (
        followup_plan_builder_fn
        if callable(followup_plan_builder_fn)
        else build_abfe_post_interaction_followup_plan
    )
    return builder(
        protocol=str(protocol or ""),
        abfe_traj_object=str(abfe_obj),
        object_exists=bool(object_exists),
        n_states=int(n_states),
        max_target_states=max_target_states,
        top_n=top_n,
    )


def resolve_abfe_post_interaction_dialog_payload(
    post_result=None,
    *,
    title="ABFE Key Interactions",
    message_prefix="Top binding residues (occupancy): ",
):
    data = dict(post_result or {}) if isinstance(post_result, dict) else {}
    summary = str(data.get("summary", "") or "").strip()
    if summary:
        return {
            "ok": True,
            "show_dialog": True,
            "reason": "ready",
            "title": str(title or "ABFE Key Interactions"),
            "message": f"{str(message_prefix or '')}{summary}",
            "summary": str(summary),
        }
    return {
        "ok": True,
        "show_dialog": False,
        "reason": (
            "missing_post_result"
            if not isinstance(post_result, dict)
            else "missing_summary"
        ),
        "title": str(title or "ABFE Key Interactions"),
        "message": "",
        "summary": "",
    }


def collect_abfe_interaction_presence_sets(interactions=None):
    frame_res = set()
    frame_types = set()
    for inter in list(interactions or []):
        if not isinstance(inter, dict):
            continue
        pres = str(inter.get("prot_res", "") or "").strip()
        if pres:
            frame_res.add(str(pres))
        itype = str(inter.get("type", "") or "").strip()
        if itype:
            frame_types.add(str(itype))
    return {
        "ok": True,
        "residues": sorted(list(frame_res)),
        "types": sorted(list(frame_types)),
    }


def build_abfe_post_interaction_report_payload(
    *,
    residue_counts=None,
    type_counts=None,
    frames_analyzed=0,
    top_n=12,
    top_types_n=8,
    preview_n=6,
    type_color_map=None,
):
    try:
        frames_i = int(frames_analyzed or 0)
    except Exception:
        frames_i = 0
    frames_i = max(0, int(frames_i))
    try:
        top_n_i = int(top_n or 12)
    except Exception:
        top_n_i = 12
    top_n_i = max(1, int(top_n_i))
    try:
        top_types_i = int(top_types_n or 8)
    except Exception:
        top_types_i = 8
    top_types_i = max(1, int(top_types_i))
    try:
        preview_i = int(preview_n or 6)
    except Exception:
        preview_i = 6
    preview_i = max(1, int(preview_i))

    res_counts = {}
    for k, v in dict(residue_counts or {}).items():
        key = str(k or "").strip()
        if not key:
            continue
        try:
            vv = int(v or 0)
        except Exception:
            vv = 0
        if vv > 0:
            res_counts[str(key)] = int(vv)

    typ_counts = {}
    for k, v in dict(type_counts or {}).items():
        key = str(k or "").strip()
        if not key:
            continue
        try:
            vv = int(v or 0)
        except Exception:
            vv = 0
        if vv > 0:
            typ_counts[str(key)] = int(vv)

    if frames_i <= 0:
        return {
            "ok": False,
            "reason": "no_frames_analyzed",
            "frames_analyzed": int(frames_i),
            "top_residues": [],
            "summary": "",
        }
    if not res_counts:
        return {
            "ok": False,
            "reason": "no_residue_counts",
            "frames_analyzed": int(frames_i),
            "top_residues": [],
            "summary": "",
        }

    sorted_res = sorted(list(res_counts.items()), key=lambda x: int(x[1]), reverse=True)
    sorted_types = sorted(list(typ_counts.items()), key=lambda x: int(x[1]), reverse=True)
    top_res = list(sorted_res[:top_n_i])
    top_types = list(sorted_types[:top_types_i])

    res_labels = [str(k) for k, _v in list(top_res or [])]
    res_occ = [100.0 * float(v) / float(frames_i) for _k, v in list(top_res or [])]
    type_labels = [str(k) for k, _v in list(top_types or [])]
    type_occ = [100.0 * float(v) / float(frames_i) for _k, v in list(top_types or [])]

    color_map = dict(type_color_map or {})
    type_colors = [str(color_map.get(str(tn), "#7f7f7f")) for tn in list(type_labels or [])]

    top_preview = ", ".join(
        f"{res_labels[i]} ({res_occ[i]:.1f}%)" for i in range(min(int(preview_i), len(res_labels)))
    )
    top_residues = [
        {"residue": str(res_labels[i]), "occupancy_pct": float(res_occ[i])}
        for i in range(len(res_labels))
    ]

    return {
        "ok": True,
        "reason": "ready",
        "frames_analyzed": int(frames_i),
        "top_residues": list(top_residues),
        "summary": str(top_preview),
        "residue_counts_sorted": list(sorted_res),
        "type_counts_sorted": list(sorted_types),
        "res_labels": list(res_labels),
        "res_occ": list(res_occ),
        "type_labels": list(type_labels),
        "type_occ": list(type_occ),
        "type_colors": list(type_colors),
        "plot_title_residues": f"ABFE Key Binding Residues (top {len(res_labels)})",
        "plot_title_types": "ABFE Interaction Type Persistence",
        "plot_suptitle": f"ABFE Post-Processing ({frames_i} sampled frames)",
        "plot_ylabel_residues": "Occupancy (%)",
        "plot_ylabel_types": "Frame Presence (%)",
        "plot_no_types_text": "No interaction-type persistence data",
    }


def dg_to_kd_pkd(dg_kj, temperature_k=300.0):
    """Convert DeltaG (kJ/mol) to Kd (M) and pKd using 1 M standard state."""
    if not np.isfinite(float(dg_kj)):
        return np.nan, np.nan
    k_b = 0.008314462618
    kbt = k_b * float(temperature_k)
    if kbt <= 0.0:
        return np.nan, np.nan
    expo = float(np.clip(float(dg_kj) / kbt, -700.0, 700.0))
    kd_m = float(np.exp(expo))
    p_kd = float(-np.log10(max(kd_m, 1e-300)))
    return kd_m, p_kd
