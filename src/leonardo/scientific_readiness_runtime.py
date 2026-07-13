import copy
import importlib.util

from .dock_runtime import resolve_tool_path

SCIENTIFIC_SKILLS_REPO_ROOT = "https://github.com/K-Dense-AI/claude-scientific-skills/tree/main/scientific-skills"


def _skill(name, purpose):
    skill_name = str(name or "").strip()
    return {
        "name": skill_name,
        "purpose": str(purpose or "").strip(),
        "url": f"{SCIENTIFIC_SKILLS_REPO_ROOT}/{skill_name}",
    }


def _python_dep(module, package, reason):
    return {
        "module": str(module or "").strip(),
        "package": str(package or module or "").strip(),
        "reason": str(reason or "").strip(),
    }


def _executable(name, reason):
    return {
        "name": str(name or "").strip(),
        "reason": str(reason or "").strip(),
    }


SCIENTIFIC_WORKFLOW_CATALOG = [
    {
        "id": "bioseq",
        "title": "Sequence bioinformatics",
        "summary": "Protein-sequence cleanup, MSA conservation, and simulation-planning heuristics.",
        "runtime_modules": ["bioseq_runtime"],
        "skills": [
            _skill("biopython", "Sequence I/O, Bio.PDB structure parsing, and robust bioinformatics workflows."),
        ],
        "required_python": [
            _python_dep("Bio", "biopython", "Core sequence and structure parsing support."),
        ],
        "recommended_python": [],
        "required_executables": [],
        "recommended_executables": [
            _executable("blastp", "Local homolog search."),
            _executable("hmmscan", "Local profile/domain scans."),
            _executable("clustalo", "Local MSA generation."),
            _executable("mafft", "Alternative local MSA generation."),
        ],
        "notes": [
            "Sequence workflows still provide heuristic reports and outbound URLs when local CLIs are absent.",
        ],
    },
    {
        "id": "alphafold",
        "title": "AlphaFold/ColabFold",
        "summary": "Model discovery, confidence parsing, and AlphaFold/ColabFold result inspection.",
        "runtime_modules": ["alphafold_runtime"],
        "skills": [
            _skill("alphafold-database", "Working with AlphaFold predictions, pLDDT, and PAE assets."),
            _skill("biopython", "mmCIF/PDB handling for structural follow-up workflows."),
            _skill("scientific-visualization", "Publication-grade confidence and error plots."),
        ],
        "required_python": [],
        "recommended_python": [
            _python_dep("Bio", "biopython", "Structure parsing for downloaded AlphaFold models."),
            _python_dep("matplotlib", "matplotlib", "Confidence and PAE plotting."),
        ],
        "required_executables": [],
        "recommended_executables": [
            _executable("colabfold_batch", "Local ColabFold prediction runs."),
            _executable("nvidia-smi", "GPU availability checks for local prediction jobs."),
        ],
        "notes": [
            "The runtime can inspect existing AlphaFold outputs even if local prediction binaries are missing.",
        ],
    },
    {
        "id": "docking",
        "title": "Docking",
        "summary": "Single-system docking and structure-preparation orchestration.",
        "runtime_modules": ["dock_runtime", "smina_runtime"],
        "skills": [
            _skill("rdkit", "Ligand sanitization and molecular manipulation."),
            _skill("datamol", "Higher-level molecule handling for drug-discovery workflows."),
        ],
        "required_python": [],
        "recommended_python": [
            _python_dep("rdkit", "rdkit", "Ligand normalization and chemistry-aware preprocessing."),
        ],
        "required_executables": [
            _executable("smina", "Docking engine."),
            _executable("obabel", "Structure conversion and protonation fallback."),
        ],
        "recommended_executables": [
            _executable("prepare_receptor4.py", "Legacy AutoDockTools receptor preparation."),
            _executable("prepare_ligand4.py", "Legacy AutoDockTools ligand preparation."),
        ],
        "notes": [
            "AutoDockTools helpers remain optional because the suite can operate with obabel + smina alone.",
        ],
    },
    {
        "id": "redocking",
        "title": "Redocking",
        "summary": "Repeat redocking, RMSD analysis, and WT vs mutant comparison workflows.",
        "runtime_modules": ["redock_runtime"],
        "skills": [
            _skill("rdkit", "RMSD-ready ligand matching and atom-map handling."),
            _skill("datamol", "Cheminformatics workflow convenience."),
            _skill("medchem", "Compound quality heuristics for result triage."),
        ],
        "required_python": [],
        "recommended_python": [
            _python_dep("rdkit", "rdkit", "Best-pose comparisons and molecule normalization."),
        ],
        "required_executables": [
            _executable("smina", "Docking engine."),
            _executable("obabel", "Ligand/receptor file conversion."),
        ],
        "recommended_executables": [
            _executable("prepare_receptor4.py", "Legacy receptor preparation."),
            _executable("prepare_ligand4.py", "Legacy ligand preparation."),
            _executable("prepare_flexreceptor4.py", "Legacy flexible-residue preparation."),
        ],
        "notes": [
            "RDKit-backed diagnostics are optional but materially improve RMSD and atom-map robustness.",
        ],
    },
    {
        "id": "virtual_screening",
        "title": "Virtual screening",
        "summary": "Batch entry preparation, docking loops, ranking, and summary generation.",
        "runtime_modules": ["vs_runtime"],
        "skills": [
            _skill("rdkit", "Batch molecule parsing and descriptor-aware screening."),
            _skill("datamol", "Library-oriented molecule handling."),
            _skill("medchem", "Triaging compounds with drug-likeness and alert filters."),
        ],
        "required_python": [],
        "recommended_python": [
            _python_dep("rdkit", "rdkit", "Screening-library normalization and descriptor support."),
        ],
        "required_executables": [
            _executable("smina", "Docking engine."),
            _executable("obabel", "Format conversion and protonation fallback."),
        ],
        "recommended_executables": [
            _executable("prepare_receptor4.py", "Legacy receptor preparation."),
            _executable("prepare_ligand4.py", "Legacy ligand preparation."),
        ],
        "notes": [
            "The screening runtime benefits from medchem-style filtering even though the base loop can run without it.",
        ],
    },
    {
        "id": "lead_optimization",
        "title": "Lead optimization",
        "summary": "Ranking, ADMET/MPO-style triage, and optional refinement workflows.",
        "runtime_modules": ["leadopt_runtime"],
        "skills": [
            _skill("rdkit", "Descriptor calculation and chemistry-aware triage."),
            _skill("datamol", "Molecule standardization in medicinal chemistry workflows."),
            _skill("medchem", "Drug-likeness and structural-alert heuristics."),
            _skill("molecular-dynamics", "OpenMM-backed refinement and sampling."),
        ],
        "required_python": [],
        "recommended_python": [
            _python_dep("rdkit", "rdkit", "Ligand properties and chemistry-aware filters."),
            _python_dep("openmm", "openmm", "Optional refinement sampling."),
        ],
        "required_executables": [
            _executable("smina", "Docking engine."),
            _executable("obabel", "Format conversion and protonation fallback."),
        ],
        "recommended_executables": [
            _executable("antechamber", "Ligand parameterization for refinement."),
            _executable("parmchk2", "GAFF/GAFF2 parameter completion."),
            _executable("tleap", "AMBER topology assembly."),
        ],
        "notes": [
            "Refinement is optional; the scoring and ranking helpers remain usable without OpenMM/AmberTools.",
        ],
    },
    {
        "id": "abfe",
        "title": "ABFE-lite",
        "summary": "ABFE calibration, pocket descriptors, and OpenMM-based free-energy workflows.",
        "runtime_modules": ["abfe_runtime"],
        "skills": [
            _skill("molecular-dynamics", "OpenMM-based sampling and trajectory-aware analysis."),
            _skill("rdkit", "Ligand feature extraction and chemistry-aware preprocessing."),
            _skill("scikit-learn", "Calibration and model-selection helpers."),
            _skill("scientific-visualization", "Free-energy diagnostics and reporting."),
        ],
        "required_python": [
            _python_dep("openmm", "openmm", "Core MD/ABFE simulation engine."),
            _python_dep("pymbar", "pymbar", "Free-energy estimation and uncertainty analysis."),
        ],
        "recommended_python": [
            _python_dep("rdkit", "rdkit", "Pocket/ligand descriptor support."),
            _python_dep("sklearn", "scikit-learn", "Calibration helpers."),
            _python_dep("matplotlib", "matplotlib", "Diagnostic free-energy plots."),
        ],
        "required_executables": [
            _executable("antechamber", "Ligand charge/atom typing."),
            _executable("parmchk2", "Missing GAFF parameter completion."),
            _executable("tleap", "Topology generation."),
        ],
        "recommended_executables": [
            _executable("reduce", "Hydrogen placement cleanup."),
            _executable("fpocket", "Pocket descriptor generation."),
        ],
        "notes": [
            "ABFE core readiness should be treated as strict because calibration results are only meaningful with the full simulation stack.",
        ],
    },
    {
        "id": "openmm_qmmm",
        "title": "OpenMM QM/MM",
        "summary": "QM/MM region selection and OpenMM-coupled MOPAC probes.",
        "runtime_modules": ["openmm_qmmm_runtime", "qmmm_mopac_runtime"],
        "skills": [
            _skill("molecular-dynamics", "OpenMM simulation setup and QM/MM-adjacent workflows."),
        ],
        "required_python": [
            _python_dep("openmm", "openmm", "Simulation backend for QM/MM coupling."),
        ],
        "recommended_python": [],
        "required_executables": [
            _executable("mopac", "Semiempirical QM backend."),
        ],
        "recommended_executables": [],
        "notes": [
            "The OpenMM helpers are Python-side, but production QM/MM execution depends on a working MOPAC binary.",
        ],
    },
    {
        "id": "psi4_qm",
        "title": "Psi4/QM utilities",
        "summary": "Psi4 cube handling, charge inference, and ligand-rebuild helper workflows.",
        "runtime_modules": ["psi4_runtime", "psi4_cluster_runtime"],
        "skills": [
            _skill("rdkit", "Ligand rebuilding and chemistry-aware file handling."),
            _skill("scientific-visualization", "ESP/cube-derived scientific figures."),
        ],
        "required_python": [
            _python_dep("psi4", "psi4", "Quantum-chemistry engine."),
            _python_dep("qcelemental", "qcelemental", "Molecular/QC schema helpers."),
        ],
        "recommended_python": [
            _python_dep("rdkit", "rdkit", "Ligand rebuild and normalization support."),
            _python_dep("matplotlib", "matplotlib", "Cube/ESP visualization."),
        ],
        "required_executables": [
            _executable("psi4", "Quantum-chemistry executable."),
        ],
        "recommended_executables": [
            _executable("antechamber", "Ligand charge/atom typing for rebuild paths."),
            _executable("parmchk2", "Parameter completion for rebuild paths."),
            _executable("tleap", "AMBER topology assembly for rebuild paths."),
        ],
        "notes": [
            "Cube parsing helpers are lightweight, but end-to-end QM workflows require the full Psi4 stack.",
        ],
    },
]


def list_scientific_workflow_ids():
    return [str(item.get("id", "")) for item in SCIENTIFIC_WORKFLOW_CATALOG]


def build_scientific_workflow_catalog():
    return copy.deepcopy(SCIENTIFIC_WORKFLOW_CATALOG)


def _workflow_lookup(workflow_id):
    target = str(workflow_id or "").strip().lower()
    for item in SCIENTIFIC_WORKFLOW_CATALOG:
        if str(item.get("id", "")).lower() == target:
            return copy.deepcopy(item)
    raise ValueError(f"Unknown scientific workflow: {workflow_id}")


def _python_module_available(module_name, python_available_fn=None):
    name = str(module_name or "").strip()
    if not name:
        return False
    finder = python_available_fn if callable(python_available_fn) else importlib.util.find_spec
    try:
        return bool(finder(name))
    except Exception:
        return False


def _check_python_deps(items, python_available_fn=None):
    available = []
    missing = []
    for item in list(items or []):
        payload = dict(item or {})
        ok = _python_module_available(payload.get("module", ""), python_available_fn=python_available_fn)
        payload["available"] = bool(ok)
        if ok:
            available.append(payload)
        else:
            missing.append(payload)
    return available, missing


def _check_executables(items, executable_resolver=None, sys_prefix=None, sys_executable=None):
    available = []
    missing = []
    for item in list(items or []):
        payload = dict(item or {})
        name = str(payload.get("name", "") or "").strip()
        if not name:
            continue
        if callable(executable_resolver):
            try:
                resolved = executable_resolver(name)
            except Exception:
                resolved = None
        else:
            resolved = resolve_tool_path(name, sys_prefix=sys_prefix, sys_executable=sys_executable)
        ok = bool(resolved)
        payload["available"] = bool(ok)
        payload["path"] = str(resolved or "")
        if ok:
            available.append(payload)
        else:
            missing.append(payload)
    return available, missing


def _build_install_hints(report):
    hints = []
    missing_py_required = list(report.get("missing_python_required_details", []) or [])
    missing_py_recommended = list(report.get("missing_python_recommended_details", []) or [])
    missing_exe_required = list(report.get("missing_executables_required_details", []) or [])
    missing_exe_recommended = list(report.get("missing_executables_recommended_details", []) or [])
    if missing_py_required:
        pkgs = sorted({str(item.get("package", "") or "").strip() for item in missing_py_required if item.get("package")})
        if pkgs:
            hints.append("Install required Python packages: " + ", ".join(pkgs))
    if missing_exe_required:
        names = [str(item.get("name", "") or "").strip() for item in missing_exe_required if item.get("name")]
        if names:
            hints.append("Install or configure required executables: " + ", ".join(names))
    if missing_py_recommended:
        pkgs = sorted({str(item.get("package", "") or "").strip() for item in missing_py_recommended if item.get("package")})
        if pkgs:
            hints.append("Recommended scientific packages for fuller workflow coverage: " + ", ".join(pkgs))
    if missing_exe_recommended:
        names = [str(item.get("name", "") or "").strip() for item in missing_exe_recommended if item.get("name")]
        if names:
            hints.append("Recommended executables for full local coverage: " + ", ".join(names))
    return hints


def _resolve_status(missing_required_python, missing_required_executables, missing_recommended_python, missing_recommended_executables):
    if missing_required_python or missing_required_executables:
        return "missing_required"
    if missing_recommended_python or missing_recommended_executables:
        return "ready_with_gaps"
    return "ready"


def build_workflow_readiness_report(
    workflow_id,
    python_available_fn=None,
    executable_resolver=None,
    sys_prefix=None,
    sys_executable=None,
):
    workflow = _workflow_lookup(workflow_id)
    required_python_available, required_python_missing = _check_python_deps(
        workflow.get("required_python", []),
        python_available_fn=python_available_fn,
    )
    recommended_python_available, recommended_python_missing = _check_python_deps(
        workflow.get("recommended_python", []),
        python_available_fn=python_available_fn,
    )
    required_exec_available, required_exec_missing = _check_executables(
        workflow.get("required_executables", []),
        executable_resolver=executable_resolver,
        sys_prefix=sys_prefix,
        sys_executable=sys_executable,
    )
    recommended_exec_available, recommended_exec_missing = _check_executables(
        workflow.get("recommended_executables", []),
        executable_resolver=executable_resolver,
        sys_prefix=sys_prefix,
        sys_executable=sys_executable,
    )
    status = _resolve_status(
        required_python_missing,
        required_exec_missing,
        recommended_python_missing,
        recommended_exec_missing,
    )
    report = {
        "workflow_id": str(workflow.get("id", "") or ""),
        "title": str(workflow.get("title", "") or ""),
        "summary": str(workflow.get("summary", "") or ""),
        "status": str(status),
        "ok": bool(status != "missing_required"),
        "runtime_modules": list(workflow.get("runtime_modules", []) or []),
        "skills": list(workflow.get("skills", []) or []),
        "skill_names": [str(item.get("name", "") or "") for item in list(workflow.get("skills", []) or [])],
        "skills_repo_root": SCIENTIFIC_SKILLS_REPO_ROOT,
        "notes": list(workflow.get("notes", []) or []),
        "required_python_available_details": required_python_available,
        "required_python_missing_details": required_python_missing,
        "recommended_python_available_details": recommended_python_available,
        "recommended_python_missing_details": recommended_python_missing,
        "required_executables_available_details": required_exec_available,
        "required_executables_missing_details": required_exec_missing,
        "recommended_executables_available_details": recommended_exec_available,
        "recommended_executables_missing_details": recommended_exec_missing,
    }
    report["available_python_required"] = [str(item.get("module", "") or "") for item in required_python_available]
    report["missing_python_required"] = [str(item.get("module", "") or "") for item in required_python_missing]
    report["available_python_recommended"] = [str(item.get("module", "") or "") for item in recommended_python_available]
    report["missing_python_recommended"] = [str(item.get("module", "") or "") for item in recommended_python_missing]
    report["available_executables_required"] = [str(item.get("name", "") or "") for item in required_exec_available]
    report["missing_executables_required"] = [str(item.get("name", "") or "") for item in required_exec_missing]
    report["available_executables_recommended"] = [str(item.get("name", "") or "") for item in recommended_exec_available]
    report["missing_executables_recommended"] = [str(item.get("name", "") or "") for item in recommended_exec_missing]
    report["install_hints"] = _build_install_hints(
        {
            "missing_python_required_details": required_python_missing,
            "missing_python_recommended_details": recommended_python_missing,
            "missing_executables_required_details": required_exec_missing,
            "missing_executables_recommended_details": recommended_exec_missing,
        }
    )
    return report


def build_suite_readiness_report(
    workflow_ids=None,
    python_available_fn=None,
    executable_resolver=None,
    sys_prefix=None,
    sys_executable=None,
):
    selected = list(workflow_ids or []) or list_scientific_workflow_ids()
    reports = [
        build_workflow_readiness_report(
            workflow_id=item,
            python_available_fn=python_available_fn,
            executable_resolver=executable_resolver,
            sys_prefix=sys_prefix,
            sys_executable=sys_executable,
        )
        for item in selected
    ]
    status_counts = {"ready": 0, "ready_with_gaps": 0, "missing_required": 0}
    for item in reports:
        key = str(item.get("status", "missing_required") or "missing_required")
        status_counts[key] = int(status_counts.get(key, 0)) + 1
    return {
        "ok": all(bool(item.get("ok", False)) for item in reports),
        "workflow_ids": [str(item) for item in selected],
        "n_workflows": int(len(reports)),
        "status_counts": dict(status_counts),
        "workflow_reports": reports,
    }


def format_workflow_readiness_lines(report):
    payload = dict(report or {})
    lines = [
        f"{payload.get('workflow_id', '')}: {payload.get('status', '')}",
        "  modules: " + ", ".join(list(payload.get("runtime_modules", []) or []) or ["none"]),
        "  skills: " + ", ".join(list(payload.get("skill_names", []) or []) or ["none mapped"]),
    ]
    missing_py_required = list(payload.get("missing_python_required", []) or [])
    missing_exe_required = list(payload.get("missing_executables_required", []) or [])
    missing_py_recommended = list(payload.get("missing_python_recommended", []) or [])
    missing_exe_recommended = list(payload.get("missing_executables_recommended", []) or [])
    if missing_py_required:
        lines.append("  missing required Python: " + ", ".join(missing_py_required))
    if missing_exe_required:
        lines.append("  missing required executables: " + ", ".join(missing_exe_required))
    if missing_py_recommended:
        lines.append("  missing recommended Python: " + ", ".join(missing_py_recommended))
    if missing_exe_recommended:
        lines.append("  missing recommended executables: " + ", ".join(missing_exe_recommended))
    for note in list(payload.get("notes", []) or []):
        lines.append("  note: " + str(note))
    return lines


def format_suite_readiness_lines(report):
    payload = dict(report or {})
    counts = dict(payload.get("status_counts", {}) or {})
    lines = [
        "Scientific suite readiness:",
        "  workflows: "
        + str(payload.get("n_workflows", 0))
        + " | ready="
        + str(counts.get("ready", 0))
        + " ready_with_gaps="
        + str(counts.get("ready_with_gaps", 0))
        + " missing_required="
        + str(counts.get("missing_required", 0)),
    ]
    for item in list(payload.get("workflow_reports", []) or []):
        for ln in format_workflow_readiness_lines(item):
            lines.append("  " + str(ln))
    return lines
