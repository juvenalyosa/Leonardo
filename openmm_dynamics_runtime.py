"""
Runtime helpers for Leonardo's OpenMM Dynamics module.

The GUI owns PyMOL object/selection capture, while this module owns the
workflow-neutral rules: predefined collective variables, numeric validation,
and environment capability summaries.  Keeping those rules here makes advanced
sampling templates testable without launching PyMOL.
"""

import json
import os
from math import isfinite


def _ensure_writable_mpl_config_dir():
    """Point Matplotlib at a writable cache before any optional dependency imports it."""
    import tempfile

    configured = str(os.environ.get("MPLCONFIGDIR", "") or "").strip()
    if configured and os.path.isdir(configured) and os.access(configured, os.W_OK):
        return
    cache_dir = os.path.join(tempfile.gettempdir(), "leonardo_mpl")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = cache_dir


_ensure_writable_mpl_config_dir()


_CV_TYPE_REQUIREMENTS = {
    "distance": ("selection_a", "selection_b"),
    "angle": ("selection_a", "selection_b", "selection_c"),
    "dihedral": ("selection_a", "selection_b", "selection_c", "selection_d"),
    "rmsd": ("selection_a",),
}


def _clip_float(value, default, lower, upper):
    try:
        val = float(value)
    except Exception:
        val = float(default)
    if not isfinite(val):
        val = float(default)
    return float(min(max(val, float(lower)), float(upper)))


def _clip_int(value, default, lower, upper):
    try:
        val = int(value)
    except Exception:
        val = int(default)
    return int(min(max(val, int(lower)), int(upper)))


def openmm_cv_type_requirements(cv_type):
    """Return required selection keys for a supported collective-variable type."""
    key = str(cv_type or "distance").strip().lower()
    return tuple(_CV_TYPE_REQUIREMENTS.get(key, _CV_TYPE_REQUIREMENTS["distance"]))


def openmm_cv_expression(cv_type, selection_a="", selection_b="", selection_c="", selection_d=""):
    """Build the human-readable CV expression saved in advanced templates."""
    key = str(cv_type or "distance").strip().lower()
    g1 = str(selection_a or "group_a").strip()
    g2 = str(selection_b or "group_b").strip()
    g3 = str(selection_c or "group_c").strip()
    g4 = str(selection_d or "group_d").strip()
    if key == "angle":
        return f"angle({g1},{g2},{g3})"
    if key == "dihedral":
        return f"dihedral({g1},{g2},{g3},{g4})"
    if key == "rmsd":
        return f"rmsd({g1})"
    return f"distance({g1},{g2})"


def openmm_cv_presets():
    """
    Return the built-in CV presets.

    Presets intentionally use PyMOL selection templates.  The UI resolves
    placeholders such as {ligand} and {binding_site} from the active Leonardo
    context, then still lets advanced users edit the selections manually.
    """
    return [
        {
            "id": "protein_ligand_distance",
            "label": "Protein-ligand distance",
            "category": "binding",
            "cv_type": "distance",
            "selection_a": "{ligand}",
            "selection_b": "{binding_site}",
            "range_nm": [0.25, 2.00],
            "metadynamics": {"min_value": 0.15, "max_value": 2.50, "bias_width": 0.03, "periodic": False},
            "umbrella": {"n_windows": 24, "spring_k_kj_mol_nm2": 1500.0},
            "summary": "Distance between ligand heavy atoms and nearby binding-site atoms.",
        },
        {
            "id": "ligand_unbinding_distance",
            "label": "Ligand unbinding distance",
            "category": "binding",
            "cv_type": "distance",
            "selection_a": "{ligand}",
            "selection_b": "{binding_site}",
            "range_nm": [0.30, 4.00],
            "metadynamics": {"min_value": 0.20, "max_value": 4.50, "bias_width": 0.05, "periodic": False},
            "umbrella": {"n_windows": 32, "spring_k_kj_mol_nm2": 1000.0},
            "summary": "Wider distance CV for ligand exit-path or residence-time exploration.",
        },
        {
            "id": "backbone_rmsd",
            "label": "Backbone RMSD",
            "category": "protein",
            "cv_type": "rmsd",
            "selection_a": "(polymer and name CA)",
            "selection_b": "",
            "range_nm": [0.00, 0.60],
            "metadynamics": {"min_value": 0.0, "max_value": 0.8, "bias_width": 0.02, "periodic": False},
            "umbrella": {"n_windows": 18, "spring_k_kj_mol_nm2": 2500.0},
            "summary": "Global C-alpha RMSD from the prepared starting structure.",
        },
        {
            "id": "loop_gate_distance",
            "label": "Loop/gate distance",
            "category": "protein",
            "cv_type": "distance",
            "selection_a": "{selection_or_gate_a}",
            "selection_b": "{selection_or_gate_b}",
            "range_nm": [0.30, 1.80],
            "metadynamics": {"min_value": 0.15, "max_value": 2.20, "bias_width": 0.03, "periodic": False},
            "umbrella": {"n_windows": 20, "spring_k_kj_mol_nm2": 1200.0},
            "summary": "Distance between two user-selected protein gates or loop anchors.",
        },
        {
            "id": "reaction_bond_distance",
            "label": "Reaction bond distance",
            "category": "reaction",
            "cv_type": "distance",
            "selection_a": "{selection_or_atom_a}",
            "selection_b": "{selection_or_atom_b}",
            "range_nm": [0.10, 0.35],
            "metadynamics": {"min_value": 0.08, "max_value": 0.45, "bias_width": 0.005, "periodic": False},
            "umbrella": {"n_windows": 18, "spring_k_kj_mol_nm2": 6000.0},
            "summary": "Short atom-pair distance for bond forming/breaking reaction coordinates.",
        },
        {
            "id": "reaction_angle",
            "label": "Reaction angle",
            "category": "reaction",
            "cv_type": "angle",
            "selection_a": "{selection_or_atom_a}",
            "selection_b": "{selection_or_atom_b}",
            "selection_c": "{selection_or_atom_c}",
            "range_nm": [0.50, 2.80],
            "metadynamics": {"min_value": 0.0, "max_value": 3.1416, "bias_width": 0.05, "periodic": False},
            "umbrella": {"n_windows": 24, "spring_k_kj_mol_nm2": 800.0},
            "summary": "Three-atom angular coordinate for reaction geometry control.",
        },
        {
            "id": "torsion_dihedral",
            "label": "Torsion/dihedral",
            "category": "conformation",
            "cv_type": "dihedral",
            "selection_a": "{selection_or_atom_a}",
            "selection_b": "{selection_or_atom_b}",
            "selection_c": "{selection_or_atom_c}",
            "selection_d": "{selection_or_atom_d}",
            "range_nm": [-3.1416, 3.1416],
            "metadynamics": {"min_value": -3.1416, "max_value": 3.1416, "bias_width": 0.08, "periodic": True},
            "umbrella": {"n_windows": 36, "spring_k_kj_mol_nm2": 300.0},
            "summary": "Periodic four-atom torsion coordinate.",
        },
        {
            "id": "metal_coordination_distance",
            "label": "Metal coordination distance",
            "category": "coordination",
            "cv_type": "distance",
            "selection_a": "{selection_or_metal}",
            "selection_b": "{selection_or_ligand_atom}",
            "range_nm": [0.15, 0.35],
            "metadynamics": {"min_value": 0.10, "max_value": 0.50, "bias_width": 0.005, "periodic": False},
            "umbrella": {"n_windows": 18, "spring_k_kj_mol_nm2": 7000.0},
            "summary": "Metal-ligand coordination distance with tight sampling width.",
        },
    ]


def openmm_cv_preset_by_id(preset_id):
    """Fetch one preset by id, returning the default binding preset on miss."""
    pid = str(preset_id or "").strip()
    presets = openmm_cv_presets()
    for row in presets:
        if str(row.get("id", "")) == pid:
            return dict(row)
    return dict(presets[0])


def _replace_placeholders(text, context):
    out = str(text or "").strip()
    ctx = dict(context or {})
    for key, value in ctx.items():
        out = out.replace("{" + str(key) + "}", str(value or "").strip())
    return out


def openmm_default_cv_context(
    object_name="",
    ligand_selection="",
    binding_site_selection="",
    active_selection="",
):
    """Build placeholder values used when a preset is applied in the UI."""
    obj = str(object_name or "").strip()
    lig = str(ligand_selection or "").strip() or "(organic)"
    site = str(binding_site_selection or "").strip() or f"((polymer) within 6 of {lig})"
    active = str(active_selection or "").strip()
    fallback_atom = active or lig
    return {
        "object": obj,
        "ligand": lig,
        "binding_site": site,
        "selection_or_gate_a": active or "(polymer and name CA)",
        "selection_or_gate_b": site,
        "selection_or_atom_a": fallback_atom,
        "selection_or_atom_b": active or site,
        "selection_or_atom_c": active or "(polymer and name CA)",
        "selection_or_atom_d": active or "(polymer and name CA)",
        "selection_or_metal": active or "(elem Zn+Mg+Mn+Fe+Cu+Ca)",
        "selection_or_ligand_atom": lig,
    }


def apply_openmm_cv_preset(preset_id, context=None, workflow="umbrella"):
    """Resolve a preset into one normalized CV definition for a workflow."""
    preset = openmm_cv_preset_by_id(preset_id)
    ctx = dict(context or {})
    cv_type = str(preset.get("cv_type", "distance") or "distance").strip().lower()
    cv = {
        "preset_id": str(preset.get("id", "")),
        "name": str(preset.get("id", "cv")).strip(),
        "label": str(preset.get("label", "")),
        "type": cv_type,
        "selection_a": _replace_placeholders(preset.get("selection_a", ""), ctx),
        "selection_b": _replace_placeholders(preset.get("selection_b", ""), ctx),
        "selection_c": _replace_placeholders(preset.get("selection_c", ""), ctx),
        "selection_d": _replace_placeholders(preset.get("selection_d", ""), ctx),
        "periodic": bool((preset.get("metadynamics", {}) or {}).get("periodic", False)),
        "range_nm": list(preset.get("range_nm", [0.0, 1.0]) or [0.0, 1.0])[:2],
        "summary": str(preset.get("summary", "")),
    }
    cv["expression"] = openmm_cv_expression(
        cv_type,
        cv.get("selection_a", ""),
        cv.get("selection_b", ""),
        cv.get("selection_c", ""),
        cv.get("selection_d", ""),
    )
    meta = dict(preset.get("metadynamics", {}) or {})
    umb = dict(preset.get("umbrella", {}) or {})
    cv["metadynamics"] = {
        "min_value": _clip_float(meta.get("min_value", cv["range_nm"][0]), 0.0, -1000.0, 1000.0),
        "max_value": _clip_float(meta.get("max_value", cv["range_nm"][1]), 1.0, -1000.0, 1000.0),
        "bias_width": _clip_float(meta.get("bias_width", 0.02), 0.02, 1.0e-5, 1000.0),
        "periodic": bool(meta.get("periodic", cv["periodic"])),
        "grid_width": meta.get("grid_width", None),
    }
    cv["umbrella"] = {
        "n_windows": _clip_int(umb.get("n_windows", 24), 24, 2, 200),
        "spring_k_kj_mol_nm2": _clip_float(umb.get("spring_k_kj_mol_nm2", 1500.0), 1500.0, 1.0, 200000.0),
    }
    return normalize_openmm_cv_definition(cv, workflow=workflow)


def normalize_openmm_cv_definition(cv, workflow="umbrella"):
    """Validate one CV dictionary and clip numeric fields for templates."""
    src = dict(cv or {})
    cv_type = str(src.get("type", src.get("cv_type", "distance")) or "distance").strip().lower()
    if cv_type not in _CV_TYPE_REQUIREMENTS:
        cv_type = "distance"
    out = {
        "name": str(src.get("name", "cv") or "cv").strip() or "cv",
        "label": str(src.get("label", "") or ""),
        "type": cv_type,
        "preset_id": str(src.get("preset_id", "") or ""),
        "selection_a": str(src.get("selection_a", src.get("group_a", "")) or "").strip(),
        "selection_b": str(src.get("selection_b", src.get("group_b", "")) or "").strip(),
        "selection_c": str(src.get("selection_c", src.get("group_c", "")) or "").strip(),
        "selection_d": str(src.get("selection_d", src.get("group_d", "")) or "").strip(),
        "periodic": bool(src.get("periodic", False)),
        "summary": str(src.get("summary", "") or ""),
    }
    rng = list(src.get("range_nm", []) or [])
    if len(rng) < 2:
        rng = [0.0, 1.0]
    r0 = _clip_float(rng[0], 0.0, -1000.0, 1000.0)
    r1 = _clip_float(rng[1], 1.0, -1000.0, 1000.0)
    if r1 <= r0:
        r1 = r0 + 0.1
    out["range_nm"] = [float(r0), float(r1)]
    meta = dict(src.get("metadynamics", {}) or {})
    min_v = _clip_float(src.get("min_value", meta.get("min_value", out["range_nm"][0])), out["range_nm"][0], -1000.0, 1000.0)
    max_v = _clip_float(src.get("max_value", meta.get("max_value", out["range_nm"][1])), out["range_nm"][1], -1000.0, 1000.0)
    if max_v <= min_v:
        max_v = min_v + 0.1
    out["metadynamics"] = {
        "min_value": float(min_v),
        "max_value": float(max_v),
        "bias_width": _clip_float(src.get("bias_width", meta.get("bias_width", 0.02)), 0.02, 1.0e-5, 1000.0),
        "periodic": bool(src.get("periodic", meta.get("periodic", out["periodic"]))),
        "grid_width": src.get("grid_width", meta.get("grid_width", None)),
    }
    umb = dict(src.get("umbrella", {}) or {})
    out["umbrella"] = {
        "n_windows": _clip_int(src.get("n_windows", umb.get("n_windows", 24)), 24, 2, 200),
        "spring_k_kj_mol_nm2": _clip_float(
            src.get("spring_k_kj_mol_nm2", umb.get("spring_k_kj_mol_nm2", 1500.0)),
            1500.0,
            1.0,
            200000.0,
        ),
    }
    out["expression"] = openmm_cv_expression(
        cv_type,
        out.get("selection_a", ""),
        out.get("selection_b", ""),
        out.get("selection_c", ""),
        out.get("selection_d", ""),
    )
    # Backward-compatible aliases for older JSON templates produced by the UI.
    out["force_expression"] = str(out["expression"])
    out["group_a"] = str(out.get("selection_a", ""))
    out["group_b"] = str(out.get("selection_b", ""))
    out["group_c"] = str(out.get("selection_c", ""))
    out["group_d"] = str(out.get("selection_d", ""))
    out["min_value"] = float(out["metadynamics"]["min_value"])
    out["max_value"] = float(out["metadynamics"]["max_value"])
    out["bias_width"] = float(out["metadynamics"]["bias_width"])
    out["grid_width"] = out["metadynamics"]["grid_width"]
    valid = validate_openmm_cv_definitions([out], workflow=workflow)
    out["valid"] = bool(valid.get("ok", False))
    out["warnings"] = list(valid.get("warnings", []) or [])
    out["errors"] = list(valid.get("errors", []) or [])
    return out


def validate_openmm_cv_definitions(cvs, workflow="umbrella"):
    """Validate CV count, required selections, and workflow limits."""
    wf = str(workflow or "umbrella").strip().lower()
    arr = [dict(c) for c in list(cvs or []) if isinstance(c, dict)]
    errors = []
    warnings = []
    if not arr:
        errors.append("At least one collective variable is required.")
    if wf == "umbrella" and len(arr) != 1:
        errors.append("Umbrella sampling requires exactly one active CV.")
    if wf == "metadynamics" and len(arr) > 3:
        errors.append("OpenMM metadynamics supports at most three BiasVariable CVs.")
    for i, cv in enumerate(arr, start=1):
        cv_type = str(cv.get("type", "distance") or "distance").strip().lower()
        required = openmm_cv_type_requirements(cv_type)
        for key in required:
            if not str(cv.get(key, "") or "").strip():
                errors.append(f"CV{i} ({cv_type}) missing {key}.")
        meta = dict(cv.get("metadynamics", {}) or {})
        if wf == "metadynamics":
            min_v = _clip_float(meta.get("min_value", cv.get("min_value", 0.0)), 0.0, -1000.0, 1000.0)
            max_v = _clip_float(meta.get("max_value", cv.get("max_value", 1.0)), 1.0, -1000.0, 1000.0)
            width = _clip_float(meta.get("bias_width", cv.get("bias_width", 0.02)), 0.02, 1.0e-5, 1000.0)
            if max_v <= min_v:
                errors.append(f"CV{i} maxValue must be greater than minValue.")
            if width >= abs(max_v - min_v):
                warnings.append(f"CV{i} biasWidth is large relative to the CV range.")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings, "count": len(arr)}


def summarize_openmm_cv_definitions(cvs):
    """Build a compact one-line summary for labels/logging."""
    rows = []
    for cv in list(cvs or []):
        if not isinstance(cv, dict):
            continue
        label = str(cv.get("label", "") or cv.get("name", "cv") or "cv")
        cv_type = str(cv.get("type", "distance") or "distance")
        expr = str(cv.get("expression", "") or "").strip()
        rows.append(f"{label} [{cv_type}]" + (f": {expr}" if expr else ""))
    return "CVs: " + (" | ".join(rows) if rows else "none")


def normalize_openmm_run_controls(raw=None):
    """Clip core OpenMM run controls before worker execution."""
    data = dict(raw or {})
    mode = str(data.get("mode", "explicit") or "explicit").strip().lower()
    if mode not in ("gas", "implicit", "explicit"):
        mode = "explicit"
    ensemble = str(data.get("ensemble", "npt") or "npt").strip().lower()
    if ensemble not in ("nvt", "npt", "nve"):
        ensemble = "npt"
    if mode != "explicit" and ensemble == "npt":
        ensemble = "nvt"
    return {
        "mode": mode,
        "ensemble": ensemble,
        "ph": _clip_float(data.get("ph", 7.4), 7.4, 0.0, 14.0),
        "ionic_strength": _clip_float(data.get("ionic_strength", 0.15), 0.15, 0.0, 2.0),
        "padding_a": _clip_float(data.get("padding_a", 10.0), 10.0, 4.0, 30.0),
        "cutoff_a": _clip_float(data.get("cutoff_a", 10.0), 10.0, 6.0, 20.0),
        "temperature_k": _clip_float(data.get("temperature_k", 300.0), 300.0, 50.0, 600.0),
        "pressure_bar": _clip_float(data.get("pressure_bar", 1.0), 1.0, 0.1, 50.0),
        "timestep_fs": _clip_float(data.get("timestep_fs", 2.0), 2.0, 0.2, 5.0),
        "friction": _clip_float(data.get("friction", 1.0), 1.0, 0.1, 20.0),
        "min_iters": _clip_int(data.get("min_iters", 2000), 2000, 0, 200000),
        "heat_steps": _clip_int(data.get("heat_steps", 5000), 5000, 0, 50000000),
        "equil_steps": _clip_int(data.get("equil_steps", 20000), 20000, 0, 50000000),
        "prod_steps": _clip_int(data.get("prod_steps", 100000), 100000, 0, 200000000),
        "traj_stride": _clip_int(data.get("traj_stride", 100), 100, 1, 1000000),
        "report_stride": _clip_int(data.get("report_stride", 100), 100, 1, 1000000),
        "min_tolerance_kj_mol_nm": _clip_float(data.get("min_tolerance_kj_mol_nm", 10.0), 10.0, 0.001, 5000.0),
        "min_report_stride": _clip_int(data.get("min_report_stride", 25), 25, 1, 100000),
    }


def build_openmm_capability_report(openmm_module=None, app_module=None):
    """Return a small capability report for UI status labels and tests."""
    if openmm_module is None:
        return {"ok": False, "version": "", "platforms": [], "has_metadynamics": False, "message": "OpenMM missing"}
    version = ""
    try:
        version = str(getattr(getattr(openmm_module, "version", None), "version", "") or "")
    except Exception:
        version = ""
    platforms = []
    try:
        for i in range(int(openmm_module.Platform.getNumPlatforms())):
            platforms.append(str(openmm_module.Platform.getPlatform(i).getName()))
    except Exception:
        platforms = []
    has_meta = bool(
        app_module is not None
        and hasattr(app_module, "Metadynamics")
        and hasattr(app_module, "BiasVariable")
    )
    return {
        "ok": True,
        "version": version,
        "platforms": platforms,
        "has_metadynamics": bool(has_meta),
        "message": (
            "OpenMM OK"
            + (f" {version}" if version else "")
            + (f" | platforms={', '.join(platforms)}" if platforms else "")
            + f" | metadynamics={'OK' if has_meta else 'missing'}"
        ),
    }


def _load_openmm():
    import openmm as omm
    import openmm.app as app
    from openmm import unit

    return omm, app, unit


def _pick_openmm_platform(omm, preferred="CPU"):
    """Choose an available OpenMM platform name for backend validation tests."""
    names = []
    try:
        for i in range(int(omm.Platform.getNumPlatforms())):
            names.append(str(omm.Platform.getPlatform(i).getName()))
    except Exception:
        names = []
    requested = str(preferred or "").strip()
    order = [requested]
    if requested and requested not in {"CPU", "Reference"}:
        order.extend(["CPU", "Reference"])
    else:
        order.extend(["CPU", "Reference"])
    for name in order:
        if not name or name not in names:
            continue
        try:
            return omm.Platform.getPlatformByName(name), name
        except Exception:
            continue
    return None, "default"


def _make_openmm_simulation_with_platform_fallback(omm, app, topology, system, integrator, preferred="CPU"):
    """Create a Simulation, falling back when an available platform rejects the system.

    Some installations list OpenCL or CUDA but cannot instantiate a Context on
    the current hardware.  The OpenMM dynamics tools should continue on CPU in
    that case instead of failing after the user selected an accelerated backend.
    """
    names = []
    try:
        for i in range(int(omm.Platform.getNumPlatforms())):
            names.append(str(omm.Platform.getPlatform(i).getName()))
    except Exception:
        names = []
    requested = str(preferred or "").strip()
    order = []
    if requested:
        order.append(requested)
    order.extend(["CPU", "Reference"])

    errors = []
    seen = set()
    for name in order:
        if not name or name in seen or name not in names:
            continue
        seen.add(name)
        try:
            platform = omm.Platform.getPlatformByName(name)
            return app.Simulation(topology, system, integrator, platform), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
    try:
        return app.Simulation(topology, system, integrator), "default"
    except Exception as exc:
        if errors:
            raise RuntimeError("No compatible OpenMM platform: " + "; ".join(errors[-4:])) from exc
        raise


def _build_two_particle_topology(app):
    top = app.Topology()
    chain = top.addChain("A")
    residue = top.addResidue("DUM", chain)
    atom_a = top.addAtom("C1", app.element.carbon, residue)
    atom_b = top.addAtom("C2", app.element.carbon, residue)
    top.addBond(atom_a, atom_b)
    return top


def _build_two_particle_system(omm, unit, restraint_center_nm=None, restraint_k=200.0):
    """
    Build a tiny bonded system used only for smoke validation.

    The physical bond keeps the two particles stable.  When restraint_center_nm
    is provided, a second CustomBondForce represents an umbrella restraint on
    the same distance CV.
    """
    system = omm.System()
    system.addParticle(12.0 * unit.dalton)
    system.addParticle(12.0 * unit.dalton)
    bond = omm.HarmonicBondForce()
    bond.addBond(
        0,
        1,
        0.15 * unit.nanometer,
        1000.0 * unit.kilojoule_per_mole / unit.nanometer**2,
    )
    system.addForce(bond)
    if restraint_center_nm is not None:
        restr = omm.CustomBondForce("0.5*k*(r-r0)^2")
        restr.addGlobalParameter("k", float(restraint_k))
        restr.addGlobalParameter("r0", float(restraint_center_nm))
        restr.addBond(0, 1, [])
        try:
            restr.setName("Leonardo_UmbrellaSmokeRestraint")
        except Exception:
            pass
        system.addForce(restr)
    return system


def _make_two_particle_simulation(omm, app, unit, system, temperature_k=300.0, timestep_fs=1.0, platform_name="CPU"):
    top = _build_two_particle_topology(app)
    integrator = omm.LangevinMiddleIntegrator(
        _clip_float(temperature_k, 300.0, 50.0, 600.0) * unit.kelvin,
        1.0 / unit.picosecond,
        _clip_float(timestep_fs, 1.0, 0.1, 5.0) * 0.001 * unit.picoseconds,
    )
    simulation, picked = _make_openmm_simulation_with_platform_fallback(
        omm,
        app,
        top,
        system,
        integrator,
        preferred=platform_name,
    )
    positions = [(0.0, 0.0, 0.0), (0.20, 0.0, 0.0)] * unit.nanometer
    simulation.context.setPositions(positions)
    try:
        simulation.context.setVelocitiesToTemperature(_clip_float(temperature_k, 300.0, 50.0, 600.0) * unit.kelvin)
    except Exception:
        pass
    return simulation, picked


def _distance_nm_from_simulation(simulation, unit):
    state = simulation.context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    dx = float(pos[1][0] - pos[0][0])
    dy = float(pos[1][1] - pos[0][1])
    dz = float(pos[1][2] - pos[0][2])
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _potential_kj_mol(simulation, unit):
    state = simulation.context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def _load_matplotlib_pyplot():
    """
    Load matplotlib in headless mode with a writable cache directory.

    Leonardo is often launched from PyMOL or conda environments where the user
    home cache is not writable.  Setting MPLCONFIGDIR here keeps plot generation
    deterministic for backend tests and plugin installs.
    """
    _ensure_writable_mpl_config_dir()
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _linspace(start, stop, count):
    n = int(max(1, int(count)))
    if n == 1:
        return [float(start)]
    return [float(start) + (float(stop) - float(start)) * (float(i) / float(n - 1)) for i in range(n)]


def run_openmm_umbrella_smoke(
    output_dir,
    cv_definition=None,
    *,
    n_windows=3,
    steps_per_window=10,
    spring_k_kj_mol_nm2=200.0,
    temperature_k=300.0,
    timestep_fs=1.0,
    platform_name="CPU",
):
    """
    Execute a tiny real OpenMM umbrella-sampling smoke test.

    This validates the mechanics of harmonic window restraints and reporting on
    a two-particle distance CV.  It is deliberately not a production PMF/WHAM
    calculation.
    """
    out_dir = str(output_dir or "").strip()
    if not out_dir:
        return {"ok": False, "error": "output_dir is required"}
    os.makedirs(out_dir, exist_ok=True)
    cv = normalize_openmm_cv_definition(cv_definition or apply_openmm_cv_preset("protein_ligand_distance"), workflow="umbrella")
    valid = validate_openmm_cv_definitions([cv], workflow="umbrella")
    if not bool(valid.get("ok", False)):
        return {"ok": False, "error": "; ".join(str(x) for x in valid.get("errors", []) or []), "validation": valid}
    try:
        omm, app, unit = _load_openmm()
    except Exception as exc:
        return {"ok": False, "error": f"OpenMM import failed: {exc}"}
    rng = list(cv.get("range_nm", [0.12, 0.30]) or [0.12, 0.30])
    start = _clip_float(rng[0] if len(rng) > 0 else 0.12, 0.12, 0.05, 1.0)
    stop = _clip_float(rng[1] if len(rng) > 1 else 0.30, 0.30, 0.05, 1.0)
    if stop <= start:
        stop = start + 0.05
    win_count = _clip_int(n_windows, 3, 2, 12)
    steps = _clip_int(steps_per_window, 10, 1, 1000)
    spring = _clip_float(spring_k_kj_mol_nm2, 200.0, 1.0, 200000.0)
    centers = _linspace(start, stop, win_count)
    rows = []
    platform_used = "default"
    for idx, center in enumerate(centers, start=1):
        system = _build_two_particle_system(omm, unit, restraint_center_nm=float(center), restraint_k=float(spring))
        simulation, picked = _make_two_particle_simulation(
            omm,
            app,
            unit,
            system,
            temperature_k=temperature_k,
            timestep_fs=timestep_fs,
            platform_name=platform_name,
        )
        platform_used = str(picked or platform_used)
        e0 = _potential_kj_mol(simulation, unit)
        simulation.minimizeEnergy(maxIterations=20)
        simulation.step(int(steps))
        e1 = _potential_kj_mol(simulation, unit)
        dist = _distance_nm_from_simulation(simulation, unit)
        rows.append(
            {
                "window": int(idx),
                "center_nm": float(center),
                "final_distance_nm": float(dist),
                "initial_energy_kj_mol": float(e0),
                "final_energy_kj_mol": float(e1),
                "steps": int(steps),
            }
        )
    csv_path = os.path.join(out_dir, "umbrella_smoke_windows.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("window,center_nm,final_distance_nm,initial_energy_kj_mol,final_energy_kj_mol,steps\n")
        for row in rows:
            fh.write(
                f"{row['window']},{row['center_nm']:.8f},{row['final_distance_nm']:.8f},"
                f"{row['initial_energy_kj_mol']:.10f},{row['final_energy_kj_mol']:.10f},{row['steps']}\n"
            )
    summary = {
        "ok": True,
        "workflow": "umbrella",
        "platform": str(platform_used),
        "cv": dict(cv),
        "n_windows": int(len(rows)),
        "steps_per_window": int(steps),
        "spring_k_kj_mol_nm2": float(spring),
        "csv_path": str(csv_path),
        "rows": list(rows),
        "notes": ["two_particle_distance_cv_smoke", "not_a_production_pmf"],
    }
    summary_path = os.path.join(out_dir, "umbrella_smoke_summary.json")
    _write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary


def run_openmm_metadynamics_smoke(
    output_dir,
    cv_definitions=None,
    *,
    steps=20,
    temperature_k=300.0,
    bias_factor=3.0,
    hill_height_kj_mol=0.05,
    frequency_steps=2,
    save_frequency_steps=10,
    platform_name="CPU",
):
    """
    Execute a tiny real OpenMM metadynamics smoke test.

    The template CVs are validated, then a two-particle distance BiasVariable is
    used to verify OpenMM's Metadynamics machinery writes bias files and steps.
    """
    out_dir = str(output_dir or "").strip()
    if not out_dir:
        return {"ok": False, "error": "output_dir is required"}
    os.makedirs(out_dir, exist_ok=True)
    bias_dir = os.path.join(out_dir, "bias")
    os.makedirs(bias_dir, exist_ok=True)
    cvs = [normalize_openmm_cv_definition(c, workflow="metadynamics") for c in list(cv_definitions or []) if isinstance(c, dict)]
    if not cvs:
        cvs = [apply_openmm_cv_preset("protein_ligand_distance", context={}, workflow="metadynamics")]
    valid = validate_openmm_cv_definitions(cvs, workflow="metadynamics")
    if not bool(valid.get("ok", False)):
        return {"ok": False, "error": "; ".join(str(x) for x in valid.get("errors", []) or []), "validation": valid}
    try:
        omm, app, unit = _load_openmm()
    except Exception as exc:
        return {"ok": False, "error": f"OpenMM import failed: {exc}"}
    if not hasattr(app, "Metadynamics") or not hasattr(app, "BiasVariable"):
        return {"ok": False, "error": "OpenMM app.Metadynamics/BiasVariable is unavailable."}
    cv0 = dict(cvs[0])
    meta = dict(cv0.get("metadynamics", {}) or {})
    min_v = _clip_float(meta.get("min_value", 0.08), 0.08, 0.01, 2.0)
    max_v = _clip_float(meta.get("max_value", 0.40), 0.40, 0.02, 2.5)
    if max_v <= min_v:
        max_v = min_v + 0.05
    width = _clip_float(meta.get("bias_width", 0.01), 0.01, 1.0e-5, max(1.0e-4, max_v - min_v))
    system = _build_two_particle_system(omm, unit, restraint_center_nm=None)
    cv_force = omm.CustomBondForce("r")
    cv_force.addBond(0, 1, [])
    bias = app.BiasVariable(cv_force, float(min_v), float(max_v), float(width), False)
    meta_driver = app.Metadynamics(
        system,
        [bias],
        _clip_float(temperature_k, 300.0, 50.0, 600.0) * unit.kelvin,
        _clip_float(bias_factor, 3.0, 1.01, 200.0),
        _clip_float(hill_height_kj_mol, 0.05, 0.001, 200.0) * unit.kilojoule_per_mole,
        _clip_int(frequency_steps, 2, 1, 1000000),
        _clip_int(save_frequency_steps, 10, 1, 1000000),
        str(bias_dir),
    )
    simulation, platform_used = _make_two_particle_simulation(
        omm,
        app,
        unit,
        system,
        temperature_k=temperature_k,
        timestep_fs=1.0,
        platform_name=platform_name,
    )
    e0 = _potential_kj_mol(simulation, unit)
    meta_driver.step(simulation, _clip_int(steps, 20, 1, 10000))
    e1 = _potential_kj_mol(simulation, unit)
    dist = _distance_nm_from_simulation(simulation, unit)
    bias_files = sorted(str(x) for x in os.listdir(bias_dir) if str(x).endswith(".npy"))
    summary = {
        "ok": True,
        "workflow": "metadynamics",
        "platform": str(platform_used),
        "cvs": list(cvs),
        "steps": _clip_int(steps, 20, 1, 10000),
        "initial_energy_kj_mol": float(e0),
        "final_energy_kj_mol": float(e1),
        "final_distance_nm": float(dist),
        "bias_dir": str(bias_dir),
        "bias_files": list(bias_files),
        "notes": ["two_particle_distance_cv_smoke", "not_a_production_free_energy"],
    }
    summary_path = os.path.join(out_dir, "metadynamics_smoke_summary.json")
    _write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary


def run_openmm_advanced_template_smoke(template_payload, output_dir, workflow_override=""):
    """Run a tiny backend smoke test from a Leonardo advanced-template payload."""
    payload = dict(template_payload or {})
    workflow = str(workflow_override or payload.get("workflow", "") or "").strip().lower()
    params = dict(payload.get("parameters", payload) or {})
    if not workflow:
        if "collective_variables" in params:
            workflow = "metadynamics"
        elif "cv" in params:
            workflow = "umbrella"
    if workflow == "umbrella":
        cv = dict(params.get("cv", {}) or {})
        windowing = dict(params.get("windowing", {}) or {})
        thermostat = dict(params.get("thermostat", {}) or {})
        return run_openmm_umbrella_smoke(
            output_dir,
            cv_definition=cv,
            n_windows=min(4, _clip_int(windowing.get("n_windows", 3), 3, 2, 200)),
            steps_per_window=10,
            spring_k_kj_mol_nm2=float(windowing.get("spring_k_kj_mol_nm2", (cv.get("umbrella", {}) or {}).get("spring_k_kj_mol_nm2", 200.0)) or 200.0),
            temperature_k=float(thermostat.get("temperature_k", 300.0) or 300.0),
            timestep_fs=float(thermostat.get("timestep_fs", 1.0) or 1.0),
        )
    if workflow == "metadynamics":
        cvs = list(params.get("collective_variables", []) or [])
        return run_openmm_metadynamics_smoke(
            output_dir,
            cv_definitions=cvs,
            steps=min(30, _clip_int(params.get("total_steps", 20), 20, 1, 500000000)),
            temperature_k=float(params.get("temperature_k", 300.0) or 300.0),
            bias_factor=float(params.get("bias_factor", 3.0) or 3.0),
            hill_height_kj_mol=float(params.get("hill_height_kj_mol", 0.05) or 0.05),
            frequency_steps=min(5, _clip_int(params.get("frequency_steps", 2), 2, 1, 10000000)),
            save_frequency_steps=min(10, _clip_int(params.get("save_frequency_steps", 10), 10, 1, 100000000)),
        )
    return {"ok": False, "error": f"Simple OpenMM advanced smoke is not implemented for workflow '{workflow}'."}


def _alanine_dipeptide_openff_system(forcefield_name="openff-2.2.1.offxml"):
    """
    Build capped alanine (ACE-ALA-NME) with OpenFF/SMIRNOFF.

    This avoids relying on package sample files.  Atom ordering follows the
    heavy-atom order in the SMILES and is used for the standard phi/psi torsions:
    phi = C(ACE)-N-CA-C, psi = N-CA-C-N(NME).
    """
    from openff.toolkit import ForceField, Molecule

    smiles = "CC(=O)N[C@@H](C)C(=O)NC"
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    mol.generate_conformers(n_conformers=1)
    ff_errors = []
    ff = None
    picked = ""
    for ff_name in [forcefield_name, "openff-2.2.1.offxml", "openff-2.1.0.offxml", "openff-2.0.0.offxml"]:
        if not ff_name:
            continue
        try:
            ff = ForceField(str(ff_name))
            picked = str(ff_name)
            break
        except Exception as exc:
            ff_errors.append(f"{ff_name}: {exc}")
    if ff is None:
        raise RuntimeError("Could not load OpenFF force field: " + "; ".join(ff_errors[-4:]))
    interchange = ff.create_interchange(mol.to_topology())
    return {
        "molecule": mol,
        "forcefield": picked,
        "system": interchange.to_openmm(),
        "topology": interchange.to_openmm_topology(),
        "positions": interchange.positions.to_openmm(),
        "torsions": {"phi": (1, 3, 4, 6), "psi": (3, 4, 6, 8)},
    }


def _add_alanine_phi_psi_metadynamics(omm, app, unit, system, torsions, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    phi_force = omm.CustomTorsionForce("theta")
    phi_force.addTorsion(*[int(x) for x in torsions["phi"]], [])
    try:
        phi_force.setName("Leonardo_AlaninePhiCV")
    except Exception:
        pass
    psi_force = omm.CustomTorsionForce("theta")
    psi_force.addTorsion(*[int(x) for x in torsions["psi"]], [])
    try:
        psi_force.setName("Leonardo_AlaninePsiCV")
    except Exception:
        pass
    width = _clip_float(config.get("bias_width_rad", 0.35), 0.35, 0.01, 2.0)
    grid = _clip_int(config.get("grid_width", 48), 48, 12, 180)
    phi = app.BiasVariable(phi_force, -3.141592653589793, 3.141592653589793, width, True, grid)
    psi = app.BiasVariable(psi_force, -3.141592653589793, 3.141592653589793, width, True, grid)
    meta = app.Metadynamics(
        system,
        [phi, psi],
        _clip_float(config.get("temperature_k", 300.0), 300.0, 50.0, 600.0) * unit.kelvin,
        _clip_float(config.get("bias_factor", 8.0), 8.0, 1.01, 200.0),
        _clip_float(config.get("hill_height_kj_mol", 1.0), 1.0, 0.001, 200.0) * unit.kilojoule_per_mole,
        _clip_int(config.get("frequency_steps", 25), 25, 1, 1000000),
        _clip_int(config.get("save_frequency_steps", 250), 250, 1, 1000000),
        str(output_dir),
    )
    return meta


def _shift_fes_minimum(fes):
    import numpy as np

    arr = np.asarray(fes, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return arr
    return arr - float(np.nanmin(arr[finite]))


def _fes_rmsd(a, b):
    import numpy as np

    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((aa[mask] - bb[mask]) ** 2)))


def _write_alanine_fes_csv(path, fes):
    import numpy as np

    arr = np.asarray(fes, dtype=float)
    n_phi, n_psi = arr.shape
    phi_vals = np.linspace(-180.0, 180.0, int(n_phi), endpoint=False)
    psi_vals = np.linspace(-180.0, 180.0, int(n_psi), endpoint=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("phi_deg,psi_deg,diagnostic_grid_kj_mol\n")
        for i, phi in enumerate(phi_vals):
            for j, psi in enumerate(psi_vals):
                fh.write(f"{float(phi):.6f},{float(psi):.6f},{float(arr[i, j]):.10f}\n")


def _plot_alanine_fes(path, fes, title="Alanine dipeptide diagnostic grid"):
    import numpy as np

    plt = _load_matplotlib_pyplot()

    arr = np.asarray(fes, dtype=float)
    n_phi, n_psi = arr.shape
    phi_vals = np.linspace(-180.0, 180.0, int(n_phi), endpoint=False)
    psi_vals = np.linspace(-180.0, 180.0, int(n_psi), endpoint=False)
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    im = ax.imshow(
        arr.T,
        origin="lower",
        extent=[float(phi_vals[0]), 180.0, float(psi_vals[0]), 180.0],
        aspect="auto",
        cmap="viridis",
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Shifted diagnostic grid (kJ/mol)")
    ax.set_xlabel("phi (deg)")
    ax.set_ylabel("psi (deg)")
    ax.set_title(str(title))
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-180.0, 180.0)
    ax.grid(color="white", alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _alanine_atom_labels():
    """
    Human-readable labels for ACE-ALA-NME heavy atoms.

    The OpenFF molecule is built from the SMILES
    CC(=O)N[C@@H](C)C(=O)NC.  OpenFF preserves this heavy-atom order, so these
    labels are stable and are shared by the FES and CV-suite smoke tests.
    """
    return {
        0: "ACE methyl C",
        1: "ACE carbonyl C",
        2: "ACE carbonyl O",
        3: "Ala amide N",
        4: "Ala CA",
        5: "Ala CB",
        6: "Ala carbonyl C",
        7: "Ala carbonyl O",
        8: "NME amide N",
        9: "NME methyl C",
    }


def _alanine_cv_suite_definitions():
    """
    Map every Leonardo CV preset to a concrete ACE-ALA-NME measurement.

    Some UI presets are protein/ligand specific.  For this backend test we keep
    the preset identity and CV type, but bind them to chemically sensible
    alanine atom groups so all eight presets can be evaluated on one trajectory.
    """
    labels = _alanine_atom_labels()
    mapping = {
        "protein_ligand_distance": {
            "atom_indices": (4, 8),
            "note": "CA to NME amide N distance used as a compact binding-distance analog.",
        },
        "ligand_unbinding_distance": {
            "atom_indices": (0, 9),
            "note": "Terminal cap distance used as a wider separation-coordinate analog.",
        },
        "backbone_rmsd": {
            "atom_indices": (1, 3, 4, 6, 8),
            "note": "Backbone-like heavy atoms aligned to the minimized reference.",
        },
        "loop_gate_distance": {
            "atom_indices": (5, 9),
            "note": "Ala side-chain carbon to NME methyl distance used as a gate-distance analog.",
        },
        "reaction_bond_distance": {
            "atom_indices": (1, 3),
            "note": "ACE carbonyl C to amide N bond-distance coordinate.",
        },
        "reaction_angle": {
            "atom_indices": (3, 4, 6),
            "note": "N-CA-C backbone angle coordinate.",
        },
        "torsion_dihedral": {
            "atom_indices": (1, 3, 4, 6),
            "note": "Alanine phi torsion: ACE C - N - CA - C.",
        },
        "metal_coordination_distance": {
            "atom_indices": (2, 8),
            "note": "Carbonyl O to NME N distance used only to exercise the coordination-distance preset.",
        },
    }
    definitions = []
    for preset in openmm_cv_presets():
        pid = str(preset.get("id", "") or "")
        spec = dict(mapping.get(pid, {}) or {})
        atom_indices = tuple(int(x) for x in spec.get("atom_indices", ()) or ())
        atom_names = [labels.get(i, f"atom {i}") for i in atom_indices]
        cv_type = str(preset.get("cv_type", "distance") or "distance").strip().lower()
        cv = normalize_openmm_cv_definition(
            {
                "name": pid,
                "label": str(preset.get("label", "") or pid),
                "type": cv_type,
                "preset_id": pid,
                "selection_a": atom_names[0] if len(atom_names) > 0 else "",
                "selection_b": atom_names[1] if len(atom_names) > 1 else "",
                "selection_c": atom_names[2] if len(atom_names) > 2 else "",
                "selection_d": atom_names[3] if len(atom_names) > 3 else "",
                "periodic": bool((preset.get("metadynamics", {}) or {}).get("periodic", False)),
                "range_nm": list(preset.get("range_nm", [0.0, 1.0]) or [0.0, 1.0]),
                "metadynamics": dict(preset.get("metadynamics", {}) or {}),
                "umbrella": dict(preset.get("umbrella", {}) or {}),
                "summary": str(preset.get("summary", "") or ""),
            },
            workflow="metadynamics",
        )
        cv.update(
            {
                "atom_indices": list(atom_indices),
                "atom_labels": list(atom_names),
                "value_unit": "deg" if cv_type in ("angle", "dihedral") else "nm",
                "alanine_mapping_note": str(spec.get("note", "")),
            }
        )
        definitions.append(cv)
    return definitions


def _positions_to_numpy_nm(positions, unit):
    import numpy as np

    raw = positions.value_in_unit(unit.nanometer) if hasattr(positions, "value_in_unit") else positions
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 1:
        arr = np.asarray([[float(v[0]), float(v[1]), float(v[2])] for v in raw], dtype=float)
    return arr


def _angle_rad(points):
    import numpy as np

    a, b, c = [np.asarray(p, dtype=float) for p in points]
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 0.0:
        return float("nan")
    cos_val = float(np.dot(v1, v2) / denom)
    cos_val = max(-1.0, min(1.0, cos_val))
    return float(np.arccos(cos_val))


def _dihedral_rad(points):
    import numpy as np

    p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in points]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = float(np.linalg.norm(b1))
    if norm <= 0.0:
        return float("nan")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.arctan2(y, x))


def _rmsd_nm(mobile, reference):
    import numpy as np

    mob = np.asarray(mobile, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if mob.shape != ref.shape or mob.size == 0:
        return float("nan")
    mob_c = mob - mob.mean(axis=0)
    ref_c = ref - ref.mean(axis=0)
    cov = mob_c.T @ ref_c
    u, _s, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    aligned = mob_c @ rot
    diff = aligned - ref_c
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _evaluate_alanine_cv(cv, positions_nm, reference_nm):
    import numpy as np

    idx = [int(x) for x in cv.get("atom_indices", []) or []]
    cv_type = str(cv.get("type", "distance") or "distance").strip().lower()
    if cv_type == "distance" and len(idx) >= 2:
        return float(np.linalg.norm(positions_nm[idx[0]] - positions_nm[idx[1]]))
    if cv_type == "angle" and len(idx) >= 3:
        return float(np.degrees(_angle_rad([positions_nm[idx[0]], positions_nm[idx[1]], positions_nm[idx[2]]])))
    if cv_type == "dihedral" and len(idx) >= 4:
        return float(np.degrees(_dihedral_rad([positions_nm[idx[0]], positions_nm[idx[1]], positions_nm[idx[2]], positions_nm[idx[3]]])))
    if cv_type == "rmsd" and idx:
        return _rmsd_nm(positions_nm[idx], reference_nm[idx])
    return float("nan")


def _plot_alanine_cv_grid(path, definitions, rows, title):
    import math

    import numpy as np

    plt = _load_matplotlib_pyplot()
    defs = [dict(x) for x in list(definitions or [])]
    by_id = {}
    for row in list(rows or []):
        by_id.setdefault(str(row.get("preset_id", "")), []).append(dict(row))
    n = max(1, len(defs))
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(float(n) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 2.7 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, cv in zip(axes.ravel(), defs):
        ax.set_visible(True)
        data = by_id.get(str(cv.get("preset_id", "")), [])
        xs = np.asarray([float(r.get("time_ps", 0.0)) for r in data], dtype=float)
        ys = np.asarray([float(r.get("value", float("nan"))) for r in data], dtype=float)
        ax.plot(xs, ys, color="#2f3437", linewidth=1.6)
        ax.set_title(str(cv.get("label", cv.get("name", "CV"))), fontsize=10)
        ax.set_xlabel("time (ps)")
        ax.set_ylabel(f"{cv.get('type', 'cv')} ({cv.get('value_unit', '')})")
        ax.grid(color="#d8d8d8", alpha=0.65, linewidth=0.7)
        finite = ys[np.isfinite(ys)]
        if finite.size:
            ax.axhline(float(np.mean(finite)), color="#8b8f93", linestyle="--", linewidth=0.9)
    fig.suptitle(str(title), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_alanine_cv_histograms(path, definitions, rows, title):
    import math

    import numpy as np

    plt = _load_matplotlib_pyplot()
    defs = [dict(x) for x in list(definitions or [])]
    by_id = {}
    for row in list(rows or []):
        by_id.setdefault(str(row.get("preset_id", "")), []).append(dict(row))
    n = max(1, len(defs))
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(float(n) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 2.7 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, cv in zip(axes.ravel(), defs):
        ax.set_visible(True)
        values = np.asarray(
            [float(r.get("value", float("nan"))) for r in by_id.get(str(cv.get("preset_id", "")), [])],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size:
            ax.hist(values, bins=min(24, max(4, int(values.size // 2))), color="#4f565b", alpha=0.85)
        ax.set_title(str(cv.get("label", cv.get("name", "CV"))), fontsize=10)
        ax.set_xlabel(f"{cv.get('type', 'cv')} ({cv.get('value_unit', '')})")
        ax.set_ylabel("count")
        ax.grid(color="#d8d8d8", alpha=0.65, linewidth=0.7)
    fig.suptitle(str(title), fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_alanine_cv_timeseries_csv(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("sample,step,time_ps,preset_id,label,type,value,unit,atom_indices\n")
        for row in list(rows or []):
            indices = " ".join(str(x) for x in row.get("atom_indices", []) or [])
            fh.write(
                f"{int(row['sample'])},{int(row['step'])},{float(row['time_ps']):.8f},"
                f"{row['preset_id']},{row['label']},{row['type']},{float(row['value']):.10f},"
                f"{row['unit']},{indices}\n"
            )


def run_openmm_alanine_cv_suite(
    output_dir,
    *,
    steps=1000,
    sample_interval=25,
    forcefield_name="openff-2.2.1.offxml",
    platform_name="CPU",
    temperature_k=300.0,
    timestep_fs=1.0,
    random_seed=20260701,
):
    """
    Test and plot every Leonardo CV preset on one alanine-dipeptide trajectory.

    This is a backend validation/diagnostic run.  It verifies that each preset
    can be normalized, mapped to concrete atoms, evaluated over a real OpenMM
    trajectory, and plotted without launching PyMOL.
    """
    out_dir = str(output_dir or "").strip()
    if not out_dir:
        return {"ok": False, "error": "output_dir is required"}
    os.makedirs(out_dir, exist_ok=True)
    try:
        omm, app, unit = _load_openmm()
    except Exception as exc:
        return {"ok": False, "error": f"OpenMM import failed: {exc}"}
    try:
        build = _alanine_dipeptide_openff_system(forcefield_name=forcefield_name)
    except Exception as exc:
        return {"ok": False, "error": f"Alanine/OpenFF system build failed: {exc}"}

    total_steps = _clip_int(steps, 1000, 1, 10000000)
    stride = _clip_int(sample_interval, 25, 1, max(1, int(total_steps)))
    temp = _clip_float(temperature_k, 300.0, 50.0, 600.0)
    timestep = _clip_float(timestep_fs, 1.0, 0.1, 5.0)
    integrator = omm.LangevinMiddleIntegrator(
        temp * unit.kelvin,
        1.0 / unit.picosecond,
        timestep * 0.001 * unit.picoseconds,
    )
    try:
        integrator.setRandomNumberSeed(int(random_seed))
    except Exception:
        pass
    simulation, picked = _make_openmm_simulation_with_platform_fallback(
        omm,
        app,
        build["topology"],
        build["system"],
        integrator,
        preferred=platform_name,
    )
    simulation.context.setPositions(build["positions"])
    try:
        simulation.context.setVelocitiesToTemperature(temp * unit.kelvin, int(random_seed))
    except Exception:
        simulation.context.setVelocitiesToTemperature(temp * unit.kelvin)
    e0 = _potential_kj_mol(simulation, unit)
    simulation.minimizeEnergy(maxIterations=200)
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    reference_nm = _positions_to_numpy_nm(state.getPositions(asNumpy=True), unit)
    minimized_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    definitions = _alanine_cv_suite_definitions()
    rows = []
    sample_index = 0

    def sample(step):
        nonlocal sample_index
        state_now = simulation.context.getState(getPositions=True)
        positions_nm = _positions_to_numpy_nm(state_now.getPositions(asNumpy=True), unit)
        for cv in definitions:
            value = _evaluate_alanine_cv(cv, positions_nm, reference_nm)
            rows.append(
                {
                    "sample": int(sample_index),
                    "step": int(step),
                    "time_ps": float(step) * timestep * 0.001,
                    "preset_id": str(cv.get("preset_id", "")),
                    "label": str(cv.get("label", cv.get("name", "CV"))),
                    "type": str(cv.get("type", "")),
                    "value": float(value),
                    "unit": str(cv.get("value_unit", "")),
                    "atom_indices": list(cv.get("atom_indices", []) or []),
                }
            )
        sample_index += 1

    sample(0)
    current_step = 0
    while current_step < int(total_steps):
        chunk = min(int(stride), int(total_steps) - int(current_step))
        simulation.step(int(chunk))
        current_step += int(chunk)
        sample(current_step)
    final_energy = _potential_kj_mol(simulation, unit)

    csv_path = os.path.join(out_dir, "alanine_cv_timeseries.csv")
    _write_alanine_cv_timeseries_csv(csv_path, rows)
    all_plot = os.path.join(out_dir, "alanine_cv_all_presets_timeseries.png")
    hist_plot = os.path.join(out_dir, "alanine_cv_all_presets_histograms.png")
    _plot_alanine_cv_grid(all_plot, definitions, rows, "Alanine dipeptide CV preset time series")
    _plot_alanine_cv_histograms(hist_plot, definitions, rows, "Alanine dipeptide CV preset distributions")
    type_plots = {}
    for cv_type in sorted({str(cv.get("type", "")) for cv in definitions}):
        subset = [cv for cv in definitions if str(cv.get("type", "")) == cv_type]
        path = os.path.join(out_dir, f"alanine_cv_{cv_type}_timeseries.png")
        _plot_alanine_cv_grid(path, subset, rows, f"Alanine {cv_type} CVs")
        type_plots[cv_type] = str(path)

    stats = []
    for cv in definitions:
        values = [float(r["value"]) for r in rows if str(r.get("preset_id", "")) == str(cv.get("preset_id", "")) and isfinite(float(r["value"]))]
        if values:
            mean = sum(values) / float(len(values))
            variance = sum((x - mean) ** 2 for x in values) / float(len(values))
            vmin = min(values)
            vmax = max(values)
        else:
            mean = variance = vmin = vmax = float("nan")
        stats.append(
            {
                "preset_id": str(cv.get("preset_id", "")),
                "label": str(cv.get("label", "")),
                "type": str(cv.get("type", "")),
                "unit": str(cv.get("value_unit", "")),
                "atom_indices": list(cv.get("atom_indices", []) or []),
                "atom_labels": list(cv.get("atom_labels", []) or []),
                "min": float(vmin) if isfinite(float(vmin)) else None,
                "max": float(vmax) if isfinite(float(vmax)) else None,
                "mean": float(mean) if isfinite(float(mean)) else None,
                "std": float(variance**0.5) if isfinite(float(variance)) else None,
                "mapping_note": str(cv.get("alanine_mapping_note", "")),
            }
        )
    summary = {
        "ok": True,
        "workflow": "alanine_dipeptide_cv_suite",
        "platform": str(picked),
        "forcefield": str(build.get("forcefield", "")),
        "steps": int(total_steps),
        "sample_interval": int(stride),
        "samples": int(sample_index),
        "temperature_k": float(temp),
        "timestep_fs": float(timestep),
        "initial_energy_kj_mol": float(e0),
        "minimized_energy_kj_mol": float(minimized_energy),
        "final_energy_kj_mol": float(final_energy),
        "cv_count": int(len(definitions)),
        "cvs": definitions,
        "stats": stats,
        "timeseries_csv": str(csv_path),
        "plot_paths": {
            "all_timeseries": str(all_plot),
            "all_histograms": str(hist_plot),
            "by_type_timeseries": dict(type_plots),
        },
        "notes": [
            "All Leonardo CV presets were evaluated on the same ACE-ALA-NME trajectory.",
            "Protein/ligand-specific presets are mapped to alanine atom groups for backend validation.",
        ],
    }
    summary_path = os.path.join(out_dir, "alanine_cv_suite_summary.json")
    _write_json(summary_path, summary)
    summary["summary_json"] = str(summary_path)
    return summary


def run_openmm_alanine_dipeptide_fes(
    output_dir,
    *,
    max_blocks=40,
    block_steps=250,
    convergence_rmsd_kj_mol=2.0,
    stable_blocks=2,
    forcefield_name="openff-2.2.1.offxml",
    platform_name="CPU",
    random_seed=20260701,
):
    """
    Run a compact alanine-dipeptide phi/psi metadynamics diagnostic-grid check.

    Plotting stability is operationally defined as the RMSD between
    consecutive shifted grids staying below `convergence_rmsd_kj_mol` for
    `stable_blocks` blocks. This exercises Leonardo's OpenMM/CV backend and
    plotting path; it is not a production PMF or a thermodynamic convergence
    claim.
    """
    out_dir = str(output_dir or "").strip()
    if not out_dir:
        return {"ok": False, "error": "output_dir is required"}
    os.makedirs(out_dir, exist_ok=True)
    try:
        omm, app, unit = _load_openmm()
    except Exception as exc:
        return {"ok": False, "error": f"OpenMM import failed: {exc}"}
    try:
        build = _alanine_dipeptide_openff_system(forcefield_name=forcefield_name)
    except Exception as exc:
        return {"ok": False, "error": f"Alanine/OpenFF system build failed: {exc}"}
    system = build["system"]
    meta_dir = os.path.join(out_dir, "metadynamics_bias")
    meta = _add_alanine_phi_psi_metadynamics(
        omm,
        app,
        unit,
        system,
        torsions=build["torsions"],
        output_dir=meta_dir,
        config={
            "temperature_k": 300.0,
            "bias_factor": 8.0,
            "hill_height_kj_mol": 1.0,
            "frequency_steps": max(1, int(block_steps) // 10),
            "save_frequency_steps": max(1, int(block_steps)),
            "bias_width_rad": 0.35,
            "grid_width": 48,
        },
    )
    integrator = omm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )
    try:
        integrator.setRandomNumberSeed(int(random_seed))
    except Exception:
        pass
    simulation, picked = _make_openmm_simulation_with_platform_fallback(
        omm,
        app,
        build["topology"],
        system,
        integrator,
        preferred=platform_name,
    )
    simulation.context.setPositions(build["positions"])
    try:
        simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin, int(random_seed))
    except Exception:
        simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)
    e0 = _potential_kj_mol(simulation, unit)
    simulation.minimizeEnergy(maxIterations=200)
    previous = None
    stable_count = 0
    history = []
    final_fes = None
    plotting_stability_passed = False
    max_b = _clip_int(max_blocks, 40, 1, 10000)
    steps_b = _clip_int(block_steps, 250, 1, 1000000)
    threshold = _clip_float(convergence_rmsd_kj_mol, 2.0, 0.001, 100000.0)
    stable_needed = _clip_int(stable_blocks, 2, 1, 100)
    for block in range(1, int(max_b) + 1):
        meta.step(simulation, int(steps_b))
        fes = _shift_fes_minimum(meta.getFreeEnergy())
        rmsd = _fes_rmsd(fes, previous) if previous is not None else float("nan")
        if previous is not None and isfinite(rmsd) and float(rmsd) <= float(threshold):
            stable_count += 1
        else:
            stable_count = 0
        final_fes = fes
        previous = fes.copy()
        history.append(
            {
                "block": int(block),
                "steps_total": int(block * steps_b),
                "fes_rmsd_kj_mol": float(rmsd) if isfinite(rmsd) else None,
                "stable_count": int(stable_count),
            }
        )
        if stable_count >= stable_needed:
            plotting_stability_passed = True
            break
    e1 = _potential_kj_mol(simulation, unit)
    if final_fes is None:
        return {"ok": False, "error": "Metadynamics produced no FES grid."}
    fes_csv = os.path.join(out_dir, "alanine_phi_psi_fes.csv")
    conv_csv = os.path.join(out_dir, "alanine_fes_convergence.csv")
    fes_png = os.path.join(out_dir, "alanine_phi_psi_fes.png")
    _write_alanine_fes_csv(fes_csv, final_fes)
    with open(conv_csv, "w", encoding="utf-8") as fh:
        fh.write("block,steps_total,fes_rmsd_kj_mol,stable_count\n")
        for row in history:
            rmsd = row.get("fes_rmsd_kj_mol", None)
            fh.write(
                f"{int(row['block'])},{int(row['steps_total'])},"
                f"{'' if rmsd is None else f'{float(rmsd):.10f}'},{int(row['stable_count'])}\n"
            )
    title = (
        "Alanine dipeptide phi/psi diagnostic grid"
        + (
            f" | stability criterion passed at {int(history[-1]['steps_total'])} steps"
            if plotting_stability_passed
            else " | max blocks reached"
        )
    )
    _plot_alanine_fes(fes_png, final_fes, title=title)
    bias_files = sorted(str(x) for x in os.listdir(meta_dir) if str(x).endswith(".npy"))
    summary = {
        "ok": True,
        "workflow": "alanine_dipeptide_phi_psi_metadynamics",
        "diagnostic_only": True,
        "plotting_stability_passed": bool(plotting_stability_passed),
        "criterion": {
            "rmsd_kj_mol": float(threshold),
            "stable_blocks": int(stable_needed),
        },
        "blocks_run": int(len(history)),
        "steps_total": int(history[-1]["steps_total"]) if history else 0,
        "block_steps": int(steps_b),
        "platform": str(picked),
        "forcefield": str(build.get("forcefield", "")),
        "initial_energy_kj_mol": float(e0),
        "final_energy_kj_mol": float(e1),
        "torsions": dict(build.get("torsions", {})),
        "history": list(history),
        "fes_csv": str(fes_csv),
        "convergence_csv": str(conv_csv),
        "fes_png": str(fes_png),
        "bias_dir": str(meta_dir),
        "bias_files": list(bias_files),
        "notes": [
            "ACE-ALA-NME built from SMILES and parametrized with OpenFF",
            "The stability check is block-to-block shifted-grid RMSD; not a publication-grade production free-energy run.",
        ],
    }
    summary_json = os.path.join(out_dir, "alanine_fes_summary.json")
    _write_json(summary_json, summary)
    summary["summary_json"] = str(summary_json)
    return summary


__all__ = [
    "apply_openmm_cv_preset",
    "build_openmm_capability_report",
    "normalize_openmm_cv_definition",
    "normalize_openmm_run_controls",
    "openmm_cv_expression",
    "openmm_cv_preset_by_id",
    "openmm_cv_presets",
    "openmm_cv_type_requirements",
    "openmm_default_cv_context",
    "run_openmm_alanine_cv_suite",
    "run_openmm_alanine_dipeptide_fes",
    "run_openmm_advanced_template_smoke",
    "run_openmm_metadynamics_smoke",
    "run_openmm_umbrella_smoke",
    "summarize_openmm_cv_definitions",
    "validate_openmm_cv_definitions",
]
