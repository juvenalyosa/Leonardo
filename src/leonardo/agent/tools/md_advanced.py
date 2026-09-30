"""
Advanced OpenMM set-up: enhanced-sampling templates (umbrella sampling, metadynamics, OpenMM-ML)
and the choice of crystal ligands / HETATM groups and ligand XMLs that go into the MD system.

Both tools drive Leonardo's own OpenMM dialog handlers. The parameter pop-ups of the advanced
workflow are QDialogs that cannot be answered headlessly, so the tool builds the same payload
the pop-up returns (through the same CV preset / normalize / validate functions) and hands it
to ``_on_openmm_run_advanced_workflow``, which writes the template exactly as the GUI does.

Leonardo's advanced workflow prepares and validates templates; it has no production executor
for umbrella sampling or metadynamics on the real system yet (the "simple test" runs the OpenMM
machinery on a two-particle toy CV). The tool results say so, and never report a PMF.
"""

import glob
import math
import os
import re

from ..registry import ToolError
from . import _headless

KT_KJ_PER_K = 0.0083144626          # Boltzmann constant, kJ/mol/K

CV_PRESETS = ["protein_ligand_distance", "ligand_unbinding_distance", "backbone_rmsd", "loop_gate_distance",
              "reaction_bond_distance", "reaction_angle", "torsion_dihedral", "metal_coordination_distance"]
CV_TYPES = ["distance", "angle", "dihedral", "rmsd"]
_REQUIRED = {"distance": ("selection_a", "selection_b"), "angle": ("selection_a", "selection_b", "selection_c"),
             "dihedral": ("selection_a", "selection_b", "selection_c", "selection_d"), "rmsd": ("selection_a",)}
_SEL_PROPS = {k: {"type": "string", "description": "PyMOL selection (evaluated within the object)"}
              for k in ("selection_a", "selection_b", "selection_c", "selection_d")}
_PROMPTS = {"umbrella": "_openmm_prompt_umbrella_params", "metadynamics": "_openmm_prompt_metadynamics_params",
            "openmm_ml": "_openmm_prompt_openmm_ml_params"}


def _runtime():
    from ... import openmm_dynamics_runtime as rt

    return rt


def _require_object(dlg, obj):
    from pymol import cmd

    if dlg is None:
        raise ToolError("Leonardo's dialog is not open")
    if obj not in cmd.get_names("objects"):
        raise ToolError(f"no object named {obj!r}; load the structure first")
    if cmd.count_atoms(f"({obj})") <= 0:
        raise ToolError(f"object {obj!r} has no atoms")


def _select_openmm_object(dlg, obj):
    """Make ``obj`` the OpenMM module's current object (what the GUI handlers read)."""
    _require_object(dlg, obj)
    try:
        dlg._refresh_openmm_object_combo()
    except Exception:
        pass
    combo = getattr(dlg, "cmb_openmm_object", None)
    if combo is not None:
        if combo.findText(obj) < 0:
            raise ToolError(f"{obj!r} is not available to the OpenMM module")
        combo.setCurrentText(obj)


# ------------------------------------------------------------------------------------------- #
# enhanced sampling: parameter mapping (pure functions, unit-tested)
# ------------------------------------------------------------------------------------------- #
def _check_cv_range(cv_type, lo, hi, periodic, label="CV"):
    """Range sanity in the CV's own unit (nm for distance/RMSD, rad for angle/dihedral)."""
    if hi <= lo:
        raise ToolError(f"{label}: the end value ({hi}) must be larger than the start value ({lo})")
    if cv_type == "angle" and (lo < 0.0 or hi > math.pi + 1e-3):
        raise ToolError(f"{label}: an angle CV is in radians within 0-3.1416 (got {lo}-{hi})")
    if cv_type == "dihedral" and (lo < -math.pi - 1e-3 or hi > math.pi + 1e-3):
        raise ToolError(f"{label}: a dihedral CV is in radians within -3.1416-3.1416 (got {lo}-{hi})")
    if cv_type in ("distance", "rmsd") and lo < 0.0:
        raise ToolError(f"{label}: a {cv_type} CV cannot be negative (nm)")
    if periodic and cv_type != "dihedral":
        raise ToolError(f"{label}: only a dihedral CV can be periodic")


def _overlay_cv(base, workflow, cv_type=None, name=None, selections=None, start=None, end=None, periodic=None,
                windows=None, force_constant=None, bias_width=None, grid_width=None):
    """Apply the tool's arguments on top of a resolved preset CV (as the pop-up's fields would)."""
    rt = _runtime()
    cv = dict(base or {})
    cv_type = str(cv_type or cv.get("type") or "distance").lower()
    if cv_type not in CV_TYPES:
        raise ToolError(f"cv_type must be one of {CV_TYPES}")
    if cv_type != cv.get("type"):
        # a different CV type than the preset: the preset's selections/ranges no longer apply
        cv = {"name": f"cv_{cv_type}", "type": cv_type, "preset_id": cv.get("preset_id", ""),
              "range_nm": {"angle": [0.5, 2.8], "dihedral": [-math.pi, math.pi]}.get(cv_type, [0.2, 1.2])}
        for key in ("selection_a", "selection_b", "selection_c", "selection_d"):
            if not (selections or {}).get(key):
                cv[key] = ""
    cv["type"] = cv_type
    if name:
        cv["name"] = str(name)
    for key, val in (selections or {}).items():
        if val:
            cv[key] = str(val).strip()
    for key in ("selection_a", "selection_b", "selection_c", "selection_d"):
        if key not in _REQUIRED[cv_type]:
            cv[key] = ""
    meta = dict(cv.get("metadynamics") or {})
    umb = dict(cv.get("umbrella") or {})
    rng = list(cv.get("range_nm") or [0.2, 1.2])
    if workflow == "metadynamics":
        rng = [meta.get("min_value", rng[0]), meta.get("max_value", rng[1])]
    lo = float(start) if start is not None else float(rng[0])
    hi = float(end) if end is not None else float(rng[1])
    per = bool(meta.get("periodic", cv.get("periodic", False))) if periodic is None else bool(periodic)
    if periodic is None and cv_type != "dihedral":
        per = False
    _check_cv_range(cv_type, lo, hi, per, label=str(cv.get("name") or "CV"))
    cv["periodic"] = per
    cv["range_nm"] = [lo, hi]
    meta.update(min_value=lo, max_value=hi, periodic=per)
    if bias_width is not None:
        meta["bias_width"] = float(bias_width)
    if grid_width is not None:
        meta["grid_width"] = int(grid_width) or None
    if workflow == "metadynamics" and float(meta.get("bias_width", 0.02)) >= (hi - lo):
        raise ToolError(f"bias_width {meta.get('bias_width')} is not smaller than the CV range {hi - lo:.4g}; "
                        "use a Gaussian width of roughly 1/10-1/50 of the range")
    cv["metadynamics"] = meta
    for k in ("min_value", "max_value", "bias_width", "grid_width"):
        cv.pop(k, None)                  # normalize() prefers top-level aliases over the nested dict
    if windows is not None:
        umb["n_windows"] = int(windows)
    if force_constant is not None:
        umb["spring_k_kj_mol_nm2"] = float(force_constant)
    cv["umbrella"] = umb
    for k in ("n_windows", "spring_k_kj_mol_nm2"):
        cv.pop(k, None)
    out = rt.normalize_openmm_cv_definition(cv, workflow=workflow)
    if out.get("errors"):
        raise ToolError("invalid collective variable: " + "; ".join(out["errors"]))
    return out


def _umbrella_overlap_warning(start, end, windows, k, temperature_k, cv_type):
    """Neighbouring windows must overlap: spacing well under ~2 standard deviations of each window."""
    spacing = abs(float(end) - float(start)) / max(1, int(windows) - 1)
    sigma = math.sqrt(KT_KJ_PER_K * float(temperature_k) / float(k))
    unit = "rad" if cv_type in ("angle", "dihedral") else "nm"
    info = {"window_spacing": round(spacing, 5), "window_sigma": round(sigma, 5), "unit": unit}
    if spacing > 2.0 * sigma:
        need_k = KT_KJ_PER_K * float(temperature_k) / (spacing / 2.0) ** 2
        need_windows = int(math.ceil(abs(float(end) - float(start)) / (2.0 * sigma))) + 1
        info["warning"] = (f"window spacing {spacing:.4g} {unit} exceeds 2 sigma ({2 * sigma:.4g} {unit}) of each "
                           f"restraint (k={float(k):g}): neighbouring histograms may not overlap and a PMF from "
                           f"WHAM/MBAR would be unreliable. Use at least {need_windows} windows at this k, or lower "
                           f"k to about {need_k:.0f} kJ/mol/{unit}^2.")
    return info


def _steps(ns, dt_fs):
    return int(round(float(ns) * 1.0e6 / float(dt_fs)))


def build_umbrella_payload(base_cv, object, args):
    """The dict ``_openmm_prompt_umbrella_params`` returns, from the tool's arguments."""
    dt = float(args.get("timestep_fs") or 2.0)
    temp = float(args.get("temperature_K") or 300.0)
    cv = _overlay_cv(base_cv, "umbrella", cv_type=args.get("cv_type"), name=args.get("cv_name"),
                     selections={k: args.get(k) for k in _SEL_PROPS}, start=args.get("cv_start"),
                     end=args.get("cv_end"), periodic=args.get("periodic"), windows=args.get("windows"),
                     force_constant=args.get("force_constant_kj_mol_nm2"))
    umb = cv["umbrella"]
    eq_ps = args.get("equilibration_ps_per_window")
    prod_ns = args.get("production_ns_per_window")
    eq = _steps((100.0 if eq_ps is None else float(eq_ps)) / 1000.0, dt)
    prod = max(100, _steps(0.4 if prod_ns is None else float(prod_ns), dt))
    stride = int(args.get("sample_stride_steps") or 200)
    if stride > prod:
        raise ToolError(f"sample_stride_steps {stride} is longer than the production run of each window ({prod} steps)")
    payload = {
        "docs": "OpenMM user guide 3.7.16 + CustomCVForce API",
        "object": object,
        "cv": dict(cv),
        "windowing": {"n_windows": int(umb["n_windows"]), "spring_k_kj_mol_nm2": float(umb["spring_k_kj_mol_nm2"]),
                      "equil_steps": int(eq), "prod_steps": int(prod), "sample_stride": stride},
        "thermostat": {"temperature_k": temp, "timestep_fs": dt},
    }
    lo, hi = cv["range_nm"]
    overlap = _umbrella_overlap_warning(lo, hi, umb["n_windows"], umb["spring_k_kj_mol_nm2"], temp, cv["type"])
    total = int(umb["n_windows"]) * (eq + prod)
    summary = {"windows": int(umb["n_windows"]), "force_constant": float(umb["spring_k_kj_mol_nm2"]),
               "cv_range": [lo, hi], "samples_per_window": prod // stride, "overlap": overlap,
               "total_md_steps": total, "total_simulated_ns": round(total * dt / 1.0e6, 4)}
    return payload, summary


def build_metadynamics_payload(base_cv, object, args, workspace_dir):
    rt = _runtime()
    dt = float(args.get("timestep_fs") or 2.0)
    temp = float(args.get("temperature_K") or 300.0)
    cv1 = _overlay_cv(base_cv, "metadynamics", cv_type=args.get("cv_type"), name=args.get("cv_name") or "cv1",
                      selections={k: args.get(k) for k in _SEL_PROPS}, start=args.get("cv_start"),
                      end=args.get("cv_end"), periodic=args.get("periodic"), bias_width=args.get("bias_width"),
                      grid_width=args.get("grid_width"))
    cvs = [cv1]
    for i, extra in enumerate(list(args.get("extra_cvs") or []), start=2):
        e = dict(extra or {})
        typ = str(e.get("cv_type") or "distance").lower()
        if e.get("cv_start") is None or e.get("cv_end") is None or e.get("bias_width") is None:
            raise ToolError(f"extra CV {i} needs cv_start, cv_end and bias_width")
        base = {"name": e.get("cv_name") or f"cv{i}", "type": typ}
        cvs.append(_overlay_cv(base, "metadynamics", cv_type=typ, name=e.get("cv_name") or f"cv{i}",
                               selections={k: e.get(k) for k in _SEL_PROPS}, start=e["cv_start"], end=e["cv_end"],
                               periodic=e.get("periodic"), bias_width=e["bias_width"], grid_width=e.get("grid_width")))
    check = rt.validate_openmm_cv_definitions(cvs, workflow="metadynamics")
    if not check.get("ok"):
        raise ToolError("invalid collective variables: " + "; ".join(check.get("errors") or []))
    total = max(100, _steps(args.get("production_ns") or 1.0, dt))
    freq = int(args.get("deposit_frequency_steps") or 500)
    if freq >= total:
        raise ToolError(f"deposit_frequency_steps {freq} must be shorter than the run ({total} steps)")
    save = int(args.get("save_frequency_steps") or max(freq, 5000))
    if save % freq:
        save = (save // freq + 1) * freq           # OpenMM requires saveFrequency to be a multiple of frequency
    payload = {
        "docs": "Metadynamics/BiasVariable API",
        "object": object,
        "temperature_k": temp,
        "bias_factor": float(args.get("bias_factor") or 10.0),
        "hill_height_kj_mol": float(args.get("hill_height_kj_mol") or 1.2),
        "frequency_steps": freq,
        "save_frequency_steps": save,
        "total_steps": int(total),
        "timestep_fs": dt,
        "bias_dir": os.path.join(workspace_dir, "metadynamics_bias"),
        "collective_variables": cvs,
    }
    gamma = payload["bias_factor"]
    summary = {"n_cvs": len(cvs), "hills": total // freq, "total_md_steps": int(total),
               "total_simulated_ns": round(total * dt / 1.0e6, 4),
               "delta_T_K": round(temp * (gamma - 1.0), 1),
               "note": ("well-tempered: the bias converges to (1 - 1/bias_factor) of -F; free energy is "
                        "F = -(bias_factor/(bias_factor-1)) * V_bias")}
    if cv1.get("warnings"):
        summary["warnings"] = list(cv1["warnings"])
    return payload, summary


def build_openmm_ml_payload(object, args):
    try:
        import openmmml  # noqa: F401
        has_ml = True
    except Exception:
        has_ml = False
    dt = float(args.get("timestep_fs") or 1.0)
    region = str(args.get("ml_region_selection") or "(resn LIG)").strip()
    mode = str(args.get("ml_mode") or "ml_only")
    payload = {
        "docs": "OpenMM-ML add-on package",
        "openmm_ml_detected": bool(has_ml),
        "object": object,
        "ml_model": str(args.get("ml_model") or "ani2x"),
        "mode": mode,
        "ml_region_selection": region,
        "mm_forcefield_file": str(args.get("mm_forcefield_file") or "amber14-all.xml"),
        "precision": str(args.get("precision") or "mixed"),
        "minimize_iters": int(args.get("minimize_iterations") if args.get("minimize_iterations") is not None else 2000),
        "temperature_k": float(args.get("temperature_K") or 300.0),
        "timestep_fs": dt,
        "steps": max(100, _steps(args.get("production_ns") or 0.1, dt)),
        "use_pbc": True if args.get("use_pbc") is None else bool(args.get("use_pbc")),
    }
    summary = {"openmm_ml_installed": has_ml, "total_md_steps": payload["steps"],
               "total_simulated_ns": round(payload["steps"] * dt / 1.0e6, 4)}
    if not has_ml:
        summary["warning"] = "openmm-ml is not installed (conda install -c conda-forge openmm-ml)"
    return payload, summary


def _check_selections_have_atoms(obj, cvs, extra=()):
    from pymol import cmd

    empty = []
    for i, cv in enumerate(cvs, start=1):
        for key in _REQUIRED.get(cv.get("type", "distance"), ()):
            sel = str(cv.get(key) or "").strip()
            try:
                n = cmd.count_atoms(f"({obj}) and ({sel})")
            except Exception as exc:
                raise ToolError(f"CV{i} {key} {sel!r} is not a valid PyMOL selection: {exc}") from exc
            if n <= 0:
                empty.append(f"CV{i} {key} {sel!r}")
    for label, sel in extra:
        try:
            n = cmd.count_atoms(f"({obj}) and ({sel})")
        except Exception as exc:
            raise ToolError(f"{label} {sel!r} is not a valid PyMOL selection: {exc}") from exc
        if n <= 0:
            empty.append(f"{label} {sel!r}")
    if empty:
        raise ToolError(f"these selections match no atoms in {obj!r}: " + "; ".join(empty)
                        + ". Use e.g. 'resn LIG and name C1', 'chain A and resi 57 and name CA' "
                        "(list_md_components shows the HETATM groups).")


def _template_path(messages, dlg, mode, before):
    for m in messages or []:
        found = re.search(r"Template:\s*(\S.*?\.json)", str(m.get("text", "")))
        if found and os.path.isfile(found.group(1).strip()):
            return found.group(1).strip()
    try:
        new = set(glob.glob(os.path.join(dlg._openmm_workspace_dir(), f"advanced_{mode}_*.json"))) - set(before)
        if new:
            return max(new, key=os.path.getmtime)
    except Exception:
        pass
    return ""


def _write_template(dlg, mode, method_payload):
    """Drive ``_on_openmm_run_advanced_workflow`` with the pop-up answered by ``method_payload``."""
    attr = _PROMPTS[mode]
    if not hasattr(dlg, attr) or not hasattr(dlg, "_on_openmm_run_advanced_workflow"):
        raise ToolError("this Leonardo build has no advanced OpenMM workflow")
    try:
        before = glob.glob(os.path.join(dlg._openmm_workspace_dir(), f"advanced_{mode}_*.json"))
    except Exception:
        before = []
    had_instance_attr = attr in vars(dlg)
    original = vars(dlg).get(attr)
    setattr(dlg, attr, lambda: dict(method_payload))
    try:
        with _headless.quiet() as q:
            dlg._on_openmm_run_advanced_workflow(mode_override=mode)
    finally:
        if had_instance_attr:
            setattr(dlg, attr, original)
        else:
            try:
                delattr(dlg, attr)
            except AttributeError:
                pass
    msgs = list(q.messages)
    problems = [m["text"] for m in msgs if m.get("kind") in ("warning", "critical")]
    path = _template_path(msgs, dlg, mode, before)
    if not path:
        raise ToolError("Leonardo did not write the advanced template" + (": " + " / ".join(problems) if problems else ""))
    return path, problems


# ------------------------------------------------------------------------------------------- #
# MD components
# ------------------------------------------------------------------------------------------- #
_CATEGORY_WORDS = {"all": None, "ions": "ion", "ion": "ion", "waters": "water", "water": "water",
                   "cofactors": "cofactor", "cofactor": "cofactor", "ligands": "ligand", "ligand": "ligand"}


def parse_component_spec(spec):
    """'ions' / 'BEN' / 'BEN A:301' / 'A:301' / 'ZN 401' -> matcher description."""
    s = str(spec or "").strip()
    if not s:
        raise ToolError("empty component specifier")
    low = s.lower()
    if low in _CATEGORY_WORDS:
        return {"category": _CATEGORY_WORDS[low], "all": low == "all"}
    resn = chain = resi = None
    for part in s.replace(",", " ").split():
        if ":" in part:
            chain, resi = part.split(":", 1)
            chain = chain.strip().upper()
            resi = resi.strip() or None
        elif re.fullmatch(r"-?\d+[A-Za-z]?", part):
            resi = part
        else:
            resn = part.upper()
    if resn is None and resi is None:
        raise ToolError(f"cannot read component {spec!r}; use a residue name ('BEN'), 'RESN CHAIN:RESI' "
                        "('BEN A:301') or a category (ions, waters, cofactors, ligands, all)")
    return {"resn": resn, "chain": chain, "resi": resi}


def component_matches(parsed, comp, ff_class=""):
    if parsed.get("all"):
        return True
    cat = str(comp.get("category", "")).lower()
    if "category" in parsed:
        if parsed["category"] == "ligand":        # a non-standard organic group (needs parameters)
            return cat == "cofactor" and ff_class == "needs_parameterization"
        return cat == parsed["category"]
    if parsed.get("resn") and str(comp.get("resn", "")).upper() != parsed["resn"]:
        return False
    if parsed.get("chain") is not None and parsed["chain"] not in ("", "-"):
        if str(comp.get("chain", "")).upper() != parsed["chain"]:
            return False
    if parsed.get("resi") and str(comp.get("resi", "")).strip() != parsed["resi"]:
        return False
    return True


def _label(comp):
    return f"{comp.get('resn', '?')} {comp.get('chain', '') or '-'}:{comp.get('resi', '?')}"


def _components_report(dlg, obj):
    comps = list(dlg._collect_receptor_hetatm_components(obj) or [])
    selected = {dlg._dock_hetatm_component_cache_key(c) for c in (dlg._openmm_selected_components(obj) or [])}
    xml_map = dict(getattr(dlg, "_openmm_ligand_xml_map", {}) or {})
    cache = dict(getattr(dlg, "_openmm_hetatm_selection_cache", {}) or {})
    rows = []
    for c in comps:
        key = dlg._dock_hetatm_component_cache_key(c)
        xml = str(xml_map.get(dlg._openmm_component_uid(c), "") or "")
        rows.append({"component": _label(c), "resn": c.get("resn"), "chain": c.get("chain") or "",
                     "resi": c.get("resi"), "atoms": int(c.get("atom_count", 0) or 0),
                     "category": c.get("category"), "ff_class": dlg._openmm_component_ff_class(c),
                     "in_md_system": key in selected, "xml": xml if xml and os.path.isfile(xml) else ""})
    keep = {name: bool(getattr(dlg, attr).isChecked()) for name, attr in
            (("waters", "cb_openmm_keep_waters"), ("ions", "cb_openmm_keep_ions"),
             ("cofactors", "cb_openmm_keep_cofactors")) if hasattr(dlg, attr)}
    settings = dict(getattr(dlg, "_settings", {}) or {})
    try:
        xml_paths = list(dlg._resolve_openmm_ligand_xml_paths(obj) or [])
    except Exception:
        xml_paths = []
    return {"object": obj, "components": rows,
            "selection_source": "explicit (configure_md_components / Customize Components)"
            if cache.get(obj) else "keep_* checkboxes",
            "keep_checkboxes": keep, "ligand_xml_paths": xml_paths,
            "auto_parameterization": {"charge_method": settings.get("openmm_crystal_charge_method", "bcc"),
                                      "atom_type": settings.get("openmm_crystal_atom_type", "gaff2"),
                                      "net_charge": settings.get("openmm_crystal_net_charge", 0)}}


def _components_card(rep):
    inc = [r["component"] for r in rep["components"] if r["in_md_system"]]
    out = [r["component"] for r in rep["components"] if not r["in_md_system"] and r["category"] != "water"]
    waters = [r for r in rep["components"] if r["category"] == "water"]
    w_in = sum(1 for r in waters if r["in_md_system"])
    return (f"MD components of **{rep['object']}**: in system: {', '.join(inc[:12]) or 'none'}"
            + (f"; left out: {', '.join(out[:12])}" if out else "")
            + (f"; crystal waters {w_in}/{len(waters)}" if waters else "")
            + (f"; ligand XML: {', '.join(os.path.basename(p) for p in rep['ligand_xml_paths'][:4])}"
               if rep["ligand_xml_paths"] else "") + ".")


def _apply_component_selection(dlg, obj, include, exclude):
    comps = list(dlg._collect_receptor_hetatm_components(obj) or [])
    if not comps:
        raise ToolError(f"{obj!r} has no HETATM components (nothing to include or exclude)")
    ff = {id(c): dlg._openmm_component_ff_class(c) for c in comps}
    chosen, notes = [], []
    for spec in include or []:
        parsed = parse_component_spec(spec)
        if str(spec).strip().lower() == "none":
            continue
        hits = [c for c in comps if component_matches(parsed, c, ff[id(c)])]
        if not hits and not parsed.get("category"):
            raise ToolError(f"no component matches {spec!r}; available: "
                            + ", ".join(_label(c) + f" ({c.get('category')})" for c in comps[:30]))
        if not hits:
            notes.append(f"{obj} has no {spec} (nothing added for it)")
        chosen.extend(h for h in hits if h not in chosen)
    for spec in exclude or []:
        parsed = parse_component_spec(spec)
        hits = [c for c in comps if component_matches(parsed, c, ff[id(c)])]
        if not hits and not parsed.get("category"):
            raise ToolError(f"no component matches {spec!r}; available: "
                            + ", ".join(_label(c) for c in comps[:30]))
        chosen = [c for c in chosen if c not in hits]
    keys = sorted({dlg._dock_hetatm_component_cache_key(c) for c in chosen})

    # answer Leonardo's HETATM chooser for this call only (the "remember" branch writes the cache)
    def prompt(receptor_obj, components, flow_label="docking", cache_dict=None, *args, **kwargs):
        cache = cache_dict if isinstance(cache_dict, dict) else dlg._openmm_hetatm_selection_cache
        if keys:
            cache[str(receptor_obj)] = list(keys)
        else:
            cache.pop(str(receptor_obj), None)
        sel = [c for c in components if dlg._dock_hetatm_component_cache_key(c) in keys]
        counts = {k: sum(1 for c in sel if str(c.get("category", "")).lower() == k) for k in ("cofactor", "ion", "water")}
        return {"cancelled": False, "selected": sel, "all": list(components), "selected_counts": counts}

    had = "_prompt_dock_hetatm_selection" in vars(dlg)
    previous = vars(dlg).get("_prompt_dock_hetatm_selection")
    dlg._prompt_dock_hetatm_selection = prompt
    try:
        if not keys:
            # Leonardo falls back to the keep_* checkboxes when no explicit selection exists
            for attr in ("cb_openmm_keep_waters", "cb_openmm_keep_ions", "cb_openmm_keep_cofactors"):
                if hasattr(dlg, attr):
                    getattr(dlg, attr).setChecked(False)
        with _headless.quiet() as q:
            dlg._on_openmm_customize_components()
    finally:
        if had:
            dlg._prompt_dock_hetatm_selection = previous
        else:
            try:
                del dlg._prompt_dock_hetatm_selection
            except AttributeError:
                pass
    # the crystal-ligand XML selection (Crystal Ligands dialog) follows the same choice
    cand = {dlg._openmm_component_uid(c) for c in dlg._openmm_detect_ligand_candidates(obj) or []}
    try:
        dlg._openmm_ligand_xml_selected_keys[obj] = sorted(
            dlg._openmm_component_uid(c) for c in chosen if dlg._openmm_component_uid(c) in cand)
    except Exception:
        pass
    return notes + [m["text"] for m in q.messages if m.get("kind") in ("warning", "critical")]


def _add_ligand_xml(dlg, paths):
    from pymol.Qt import QtWidgets

    clean = []
    for p in paths:
        p = os.path.abspath(os.path.expanduser(str(p).strip()))
        if not os.path.isfile(p):
            raise ToolError(f"ligand XML not found: {p}")
        kind = dlg._openmm_xml_kind(p)
        if kind not in ("forcefield", "system"):
            raise ToolError(f"{p} is not an OpenMM ForceField (<ForceField>) or System (<System>) XML")
        clean.append(p)
    QFD = QtWidgets.QFileDialog
    had = "getOpenFileNames" in vars(QFD)
    original = vars(QFD).get("getOpenFileNames")
    QFD.getOpenFileNames = staticmethod(lambda *a, **k: (list(clean), "OpenMM XML (*.xml)"))
    try:
        with _headless.quiet():
            dlg._on_openmm_pick_ligand_xml()
    finally:
        if had:
            QFD.getOpenFileNames = original
        else:
            try:
                del QFD.getOpenFileNames
            except AttributeError:
                pass
    return clean


# ------------------------------------------------------------------------------------------- #
def register(reg):
    cv_props = {
        "cv_preset": {"type": "string", "enum": CV_PRESETS,
                      "description": "Leonardo CV preset used as the starting point (selections resolved for the "
                                     "object; the ligand is the active/organic one). Default protein_ligand_distance."},
        "cv_type": {"type": "string", "enum": CV_TYPES,
                    "description": "distance (A-B centroids, nm), angle (A-B-C, rad), dihedral (A-B-C-D, rad), "
                                   "rmsd (of A vs the start, nm). Overrides the preset's type."},
        "cv_name": {"type": "string"},
        **_SEL_PROPS,
        "cv_start": {"type": "number", "minimum": -3.1416, "maximum": 20,
                     "description": "Umbrella: first window centre; metadynamics: minValue. nm or rad."},
        "cv_end": {"type": "number", "minimum": -3.1416, "maximum": 20,
                   "description": "Umbrella: last window centre; metadynamics: maxValue. nm or rad."},
        "periodic": {"type": "boolean", "description": "dihedral CVs only"},
    }
    extra_cv = {"type": "object", "additionalProperties": False, "properties": {
        **{k: v for k, v in cv_props.items() if k != "cv_preset"},
        "bias_width": {"type": "number", "minimum": 0.0001, "maximum": 10},
        "grid_width": {"type": "integer", "minimum": 0, "maximum": 5000}},
        "required": ["cv_type", "selection_a", "cv_start", "cv_end", "bias_width"]}

    @reg.tool(
        "run_enhanced_sampling",
        "Enhanced-sampling set-up with Leonardo's advanced OpenMM workflow: umbrella sampling (one CV, harmonic "
        "windows), well-tempered metadynamics (1-3 CVs, Gaussian hills) or an OpenMM-ML (ANI) potential. The CV "
        "(distance/angle/dihedral/RMSD between PyMOL selections, or a preset such as protein_ligand_distance or "
        "ligand_unbinding_distance) and all parameters are validated (selections must match atoms, umbrella "
        "window overlap is checked) and saved as Leonardo's advanced template JSON, exactly as the GUI does. "
        "action='smoke_test' also runs Leonardo's quick OpenMM backend test (a two-particle toy system with "
        "these settings, seconds). IMPORTANT: Leonardo does not yet execute production umbrella/metadynamics on "
        "the real complex, so no PMF or free energy is produced: report the template and its checks only. For a "
        "ligand unbinding estimate use run_abfe (guided exit); for plain MD use run_md.",
        {"properties": {
            "object": {"type": "string"},
            "mode": {"type": "string", "enum": ["umbrella", "metadynamics", "openmm_ml"]},
            "action": {"type": "string", "enum": ["template", "smoke_test"],
                       "description": "template (default): validate and save; smoke_test: also run the backend "
                                      "check (umbrella/metadynamics only)"},
            **cv_props,
            "windows": {"type": "integer", "minimum": 2, "maximum": 200, "description": "umbrella windows"},
            "force_constant_kj_mol_nm2": {"type": "number", "minimum": 1, "maximum": 200000,
                                          "description": "umbrella spring constant (kJ/mol/nm^2, or /rad^2)"},
            "equilibration_ps_per_window": {"type": "number", "minimum": 0, "maximum": 10000},
            "production_ns_per_window": {"type": "number", "minimum": 0.0002, "maximum": 100},
            "sample_stride_steps": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "bias_factor": {"type": "number", "minimum": 1.01, "maximum": 200,
                            "description": "well-tempered bias factor (gamma); 5-15 for most systems"},
            "hill_height_kj_mol": {"type": "number", "minimum": 0.001, "maximum": 200,
                                   "description": "initial Gaussian height; ~0.5-2 kJ/mol"},
            "bias_width": {"type": "number", "minimum": 0.0001, "maximum": 10,
                           "description": "Gaussian width of CV1 (CV unit); ~1/20 of the range"},
            "grid_width": {"type": "integer", "minimum": 0, "maximum": 5000, "description": "0 = automatic"},
            "deposit_frequency_steps": {"type": "integer", "minimum": 1, "maximum": 1000000},
            "save_frequency_steps": {"type": "integer", "minimum": 1, "maximum": 100000000},
            "extra_cvs": {"type": "array", "maxItems": 2, "items": extra_cv,
                          "description": "metadynamics CV2/CV3 (each needs cv_start, cv_end, bias_width)"},
            "production_ns": {"type": "number", "minimum": 0.0002, "maximum": 1000,
                              "description": "metadynamics / OpenMM-ML run length"},
            "temperature_K": {"type": "number", "minimum": 200, "maximum": 500},
            "timestep_fs": {"type": "number", "minimum": 0.5, "maximum": 4.0},
            "ml_model": {"type": "string", "enum": ["ani2x", "ani1ccx", "ani2xt"]},
            "ml_mode": {"type": "string", "enum": ["ml_only", "mixed"]},
            "ml_region_selection": {"type": "string", "description": "mixed ML/MM: atoms treated with the ML potential"},
            "mm_forcefield_file": {"type": "string"},
            "precision": {"type": "string", "enum": ["mixed", "single", "double"]},
            "minimize_iterations": {"type": "integer", "minimum": 0, "maximum": 200000},
            "use_pbc": {"type": "boolean"},
        }, "required": ["object", "mode"]},
        permission="compute_heavy", title="Enhanced sampling (OpenMM)")
    def run_enhanced_sampling(ctx, object, mode, action="template", cv_preset=None, **args):
        dlg = ctx.dlg
        _select_openmm_object(dlg, object)
        if action == "smoke_test" and mode not in ("umbrella", "metadynamics"):
            raise ToolError("smoke_test is available for umbrella and metadynamics only")
        rt = _runtime()
        workspace = dlg._openmm_workspace_dir()
        if mode == "openmm_ml":
            payload, summary = build_openmm_ml_payload(object, args)
            if payload["mode"] == "mixed":
                _check_selections_have_atoms(object, [], extra=[("ml_region_selection", payload["ml_region_selection"])])
            cvs = []
        else:
            base = dlg._openmm_cv_from_selected_preset(workflow=mode, obj_name=object,
                                                       preset_id=cv_preset or "protein_ligand_distance")
            if mode == "umbrella":
                payload, summary = build_umbrella_payload(base, object, args)
                cvs = [payload["cv"]]
            else:
                payload, summary = build_metadynamics_payload(base, object, args, workspace)
                cvs = list(payload["collective_variables"])
            _check_selections_have_atoms(object, cvs)
        template, problems = _write_template(dlg, mode, payload)
        info = {"workflow": mode, "object": object, "template": template, "output_dir": workspace,
                "executed_production": False,
                "cvs": [{"name": c.get("name"), "type": c.get("type"), "expression": c.get("expression"),
                         "range": c.get("range_nm") if mode == "umbrella" else
                         [c["metadynamics"]["min_value"], c["metadynamics"]["max_value"]],
                         "periodic": c.get("periodic")} for c in cvs],
                "summary": summary,
                "note": "Template only: Leonardo has no production umbrella/metadynamics executor for the real "
                        "system yet; no PMF/free energy was computed."}
        try:
            info["ligand_xml_paths"] = list(dlg._resolve_openmm_ligand_xml_paths(object) or [])
        except Exception:
            pass
        if problems:
            info["leonardo_warnings"] = problems[:4]
        smoke = None
        if action == "smoke_test":
            try:
                import openmm  # noqa: F401
            except Exception as exc:
                raise ToolError(f"OpenMM is not installed in Leonardo's Python ({exc}); the template was saved "
                                f"to {template}") from exc
            import datetime

            out_dir = os.path.join(workspace, f"advanced_smoke_{mode}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
            res = dict(rt.run_openmm_advanced_template_smoke({"workflow": mode, "parameters": payload}, out_dir,
                                                             workflow_override=mode) or {})
            if not res.get("ok"):
                raise ToolError(f"OpenMM backend test failed: {res.get('error', 'unknown error')} "
                                f"(template saved to {template})")
            smoke = {"ok": True, "platform": res.get("platform"), "output_dir": out_dir,
                     "summary_json": res.get("summary_json"), "csv": res.get("csv_path"),
                     "bias_dir": res.get("bias_dir"), "notes": res.get("notes")}
            if mode == "umbrella":
                smoke["windows"] = [{"center_nm": round(r["center_nm"], 4),
                                     "final_nm": round(r["final_distance_nm"], 4)} for r in res.get("rows", [])]
            else:
                smoke["steps"] = res.get("steps")
                smoke["bias_files"] = len(res.get("bias_files") or [])
            info["smoke_test"] = smoke
            try:
                dlg.lbl_openmm_advanced.setText(f"Simple {mode} test OK | output={out_dir}")
            except Exception:
                pass
        s = summary
        if mode == "umbrella":
            ov = s["overlap"]
            detail = (f"{s['windows']} windows over {s['cv_range'][0]:.3g}-{s['cv_range'][1]:.3g} {ov['unit']}, "
                      f"k={s['force_constant']:g}, {s['total_simulated_ns']} ns in total"
                      + (f"; WARNING: {ov['warning']}" if ov.get("warning") else "; windows overlap"))
        elif mode == "metadynamics":
            detail = (f"{s['n_cvs']} CV(s), {payload['bias_factor']:g} bias factor, "
                      f"{payload['hill_height_kj_mol']:g} kJ/mol hills every {payload['frequency_steps']} steps, "
                      f"{s['total_simulated_ns']} ns")
        else:
            detail = (f"{payload['ml_model']} ({payload['mode']}), {s['total_simulated_ns']} ns"
                      + (f"; WARNING: {s['warning']}" if s.get("warning") else ""))
        cv_txt = "; ".join(f"{c['name']}: {c['expression']}" for c in info["cvs"][:3])
        info["_card"] = (f"**{mode}** template for **{object}** — {detail}."
                         + (f" CV {cv_txt}." if cv_txt else "")
                         + f" Saved: `{os.path.basename(template)}`."
                         + (" Backend smoke test OK (toy two-particle CV, not a PMF)." if smoke else "")
                         + " No production run/PMF: Leonardo only prepares this workflow.")
        return info

    @reg.tool(
        "list_md_components",
        "The HETATM groups of a structure as Leonardo's OpenMM module sees them: crystal ligands, cofactors, "
        "ions and waters, their force-field class (standard / needs_parameterization), whether each goes into "
        "the MD system, their ligand XML, and the ligand XMLs the next MD run will load.",
        {"properties": {"object": {"type": "string"}}, "required": ["object"]},
        permission="read", title="List MD components")
    def list_md_components(ctx, object):
        dlg = ctx.dlg
        _require_object(dlg, object)
        rep = _components_report(dlg, object)
        rep["_card"] = _components_card(rep)
        return rep

    @reg.tool(
        "configure_md_components",
        "Choose what goes into the next OpenMM system (run_md, minimize_structure): which crystal ligands, "
        "cofactors, ions and waters are kept (Leonardo's 'Customize Components' and 'Crystal Ligands' choices), "
        "the charge model used when a kept ligand must be parameterized (antechamber, run at MD time), and "
        "external ligand OpenMM XML files (ForceField or System XML) or the current Leonardo ligand's XML. "
        "include/exclude take residue names ('BEN'), 'RESN CHAIN:RESI' ('HEM A:501') or categories "
        "(ions, waters, cofactors, ligands, all, none). An explicit choice here replaces run_md's keep_* "
        "switches until run_md is called with keep_* again.",
        {"properties": {
            "object": {"type": "string"},
            "include": {"type": "array", "items": {"type": "string"}, "maxItems": 100,
                        "description": "components to keep; ['none'] = protein only"},
            "exclude": {"type": "array", "items": {"type": "string"}, "maxItems": 100,
                        "description": "removed from 'include' (e.g. include ['all'], exclude ['waters'])"},
            "ligand_xml": {"type": "array", "items": {"type": "string"}, "maxItems": 20,
                           "description": "paths of ligand OpenMM XML files to add"},
            "use_current_ligand_xml": {"type": "boolean",
                                       "description": "add the XML of the ligand parameterized in Leonardo"},
            "clear_ligand_xml": {"type": "boolean", "description": "forget the external XML files first (the ligand "
                                                                      "parametrized in Leonardo stays listed)"},
            "charge_method": {"type": "string", "enum": ["bcc", "gas"],
                              "description": "AM1-BCC (bcc) or Gasteiger for auto-parameterized crystal ligands"},
            "atom_type": {"type": "string", "enum": ["gaff2", "gaff"]},
            "net_charge": {"type": "integer", "minimum": -10, "maximum": 10,
                           "description": "net charge used for every auto-parameterized ligand"},
        }, "required": ["object"]},
        permission="compute_light", title="Configure MD components")
    def configure_md_components(ctx, object, include=None, exclude=None, ligand_xml=None,
                                use_current_ligand_xml=False, clear_ligand_xml=False, charge_method=None,
                                atom_type=None, net_charge=None):
        dlg = ctx.dlg
        _select_openmm_object(dlg, object)
        if exclude and include is None:
            include = ["all"]
        warnings = []
        if include is not None:
            warnings += _apply_component_selection(dlg, object, include, exclude)
        settings = getattr(dlg, "_settings", None)
        if settings is None:
            dlg._settings = settings = {}
        for key, val in (("openmm_crystal_charge_method", charge_method), ("openmm_crystal_atom_type", atom_type),
                         ("openmm_crystal_net_charge", net_charge)):
            if val is not None:
                settings[key] = val
        if clear_ligand_xml:
            dlg._openmm_set_ligand_xml_paths([])
        added = _add_ligand_xml(dlg, ligand_xml) if ligand_xml else []
        if use_current_ligand_xml:
            with _headless.quiet() as q:
                dlg._on_openmm_use_current_ligand_xml()
            bad = [m["text"] for m in q.messages if m.get("kind") in ("warning", "critical")]
            if bad:
                raise ToolError(bad[0] + " (parametrize_ligand first, or pass ligand_xml paths)")
        try:
            dlg._openmm_update_component_summary()
            dlg._openmm_update_ligand_status()
        except Exception:
            pass
        rep = _components_report(dlg, object)
        if added:
            rep["added_xml"] = added
        needs = [r["component"] for r in rep["components"]
                 if r["in_md_system"] and r["ff_class"] == "needs_parameterization" and not r["xml"]]
        if needs:
            rep["will_parameterize_at_md_time"] = needs
        if warnings:
            rep["leonardo_warnings"] = warnings[:4]
        rep["_card"] = _components_card(rep) + (f" To parameterize at MD start: {', '.join(needs)}." if needs else "")
        return rep
