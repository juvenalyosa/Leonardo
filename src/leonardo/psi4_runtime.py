import collections
import glob
import os
import re

import numpy as np

_PSI4_PCM_SOLVENT_SPECS = (
    ("Water", ("H2O",)),
    ("Propylene Carbonate", ("PC", "C4H6O3")),
    ("DMSO", ("DimethylSulfoxide", "Dimethylsulfoxide", "DimethylSulphoxide", "Dimethylsulphoxide")),
    ("Nitromethane", ("CH3NO2",)),
    ("Acetonitrile", ("MeCN", "CH3CN")),
    ("Methanol", ("MeOH", "CH3OH")),
    ("Ethanol", ("EtOH", "CH3CH2OH")),
    ("Acetone", ("C2H6CO",)),
    ("1,2-Dichloroethane", ("1,2 Dichloroethane", "C2H4Cl2")),
    ("Methylenechloride", ("Methylene Chloride", "Dichloromethane", "CH2Cl2")),
    ("THF", ("Tetrahydrofurane", "Tetrahydrofuran")),
    ("Aniline", ("C6H5NH2",)),
    ("Chlorobenzene", ("C6H5Cl",)),
    ("Chloroform", ("CHCl3",)),
    ("Toluene", ("C6H5CH3", "Methylbenzene")),
    ("1,4-Dioxane", ("1,4 Dioxane", "Dioxane", "C4H8O2")),
    ("Benzene", ("C6H6",)),
    ("Carbon Tetrachloride", ("CarbonTetrachloride", "CCl4")),
    ("Cyclohexane", ("C6H12",)),
    ("N-heptane", ("N Heptane", "Heptane", "C7H16")),
)

_PSI4_OPERATION_ALIASES = {
    "sp": "energy",
    "single": "energy",
    "single_point": "energy",
    "single point": "energy",
    "energy": "energy",
    "opt": "optimize",
    "optimize": "optimize",
    "optimization": "optimize",
    "freq": "frequencies",
    "frequency": "frequencies",
    "frequencies": "frequencies",
    "stationary": "stationary",
    "minimum": "stationary",
    "ts": "ts",
    "transition_state": "ts",
    "transition state": "ts",
    "irc": "irc",
    "esp": "esp_fit",
    "esp_fit": "esp_fit",
    "esp fit": "esp_fit",
}

_PSI4_SUPPORTED_SCRIPT_OPERATIONS = {
    "energy",
    "optimize",
    "frequencies",
    "stationary",
    "ts",
    "irc",
    "esp_fit",
}


def _psi4_int_in_range(value, default, lo, hi):
    try:
        out = int(value)
    except Exception:
        out = int(default)
    return int(max(int(lo), min(int(hi), out)))


def _psi4_float_at_least(value, default, lo):
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return float(max(float(lo), out))


def _psi4_clean_driver_token(value):
    """
    Keep user-editable method/basis strings safe for generated Python input.

    The generated Psi4 script still uses repr() for quoting, but normalizing
    whitespace here prevents silent mixed settings such as "hf / sto-3g".
    """
    txt = str(value or "").strip()
    txt = re.sub(r"[\r\n\t]+", " ", txt)
    txt = re.sub(r"\s*/\s*", "/", txt)
    txt = re.sub(r"\s+", "", txt)
    return str(txt)


def psi4_normalize_operation(operation, default="optimize"):
    op_raw = str(operation or "").strip().lower()
    op = _PSI4_OPERATION_ALIASES.get(op_raw, op_raw)
    if op not in _PSI4_SUPPORTED_SCRIPT_OPERATIONS:
        op = _PSI4_OPERATION_ALIASES.get(str(default or "").strip().lower(), str(default or "optimize"))
    if op not in _PSI4_SUPPORTED_SCRIPT_OPERATIONS:
        op = "optimize"
    return str(op)


def psi4_normalize_method_basis(method, basis="", default_method="wb97x-d", default_basis="def2-SVP"):
    """
    Normalize Psi4 method/base user input into the form used by this module.

    Leonardo sets the basis through psi4.set_options({"basis": ...}) and then
    calls energy/optimize/frequency with only the method name. If a user enters
    "hf/sto-3g" in the method field, the embedded basis must become the active
    basis to avoid contradictory Psi4 inputs.
    """
    m = _psi4_clean_driver_token(method)
    b = _psi4_clean_driver_token(basis)
    embedded_basis = ""
    if "/" in m:
        left, right = m.split("/", 1)
        m = str(left or "").strip()
        embedded_basis = str(right or "").strip()
        if embedded_basis:
            b = embedded_basis
    if not m:
        m = _psi4_clean_driver_token(default_method) or "wb97x-d"
    if not b:
        b = _psi4_clean_driver_token(default_basis) or "def2-SVP"
    return {
        "method": str(m),
        "basis": str(b),
        "method_call": str(m),
        "embedded_basis": str(embedded_basis),
    }


def psi4_parse_charge_multiplicity_from_block(molecule_block):
    for raw in str(molecule_block or "").splitlines():
        line = str(raw or "").strip()
        if (not line) or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            charge = int(parts[0])
            multiplicity = int(parts[1])
        except Exception:
            continue
        return int(charge), max(1, int(multiplicity))
    return None, None


def psi4_method_is_likely_dft(method):
    m = str(method or "").strip().lower()
    if not m:
        return False
    if m in {"hf", "scf", "mp2", "ccsd", "ccsd(t)", "ci", "fci"}:
        return False
    dft_tokens = (
        "b3lyp",
        "blyp",
        "bp86",
        "pbe",
        "pbe0",
        "svwn",
        "m06",
        "wb97",
        "b97",
        "tpss",
        "scan",
        "m11",
        "cam-",
        "lc-",
        "hse",
    )
    return any(tok in m for tok in dft_tokens)


def psi4_reference_for_multiplicity(method, multiplicity=1, explicit_reference=""):
    """
    Pick the Psi4 SCF reference needed for non-singlet jobs.

    Psi4's default RHF/RKS choices are correct for ordinary singlets, but an
    open-shell molecule must not reach the driver without UHF/UKS/ROHF. Keeping
    the choice here protects energy, TS, IRC, and ESP workflows uniformly.
    """
    ref = str(explicit_reference or "").strip().lower()
    valid = {"rhf", "uhf", "rohf", "rks", "uks"}
    if ref in valid:
        return str(ref)
    try:
        mult = int(multiplicity)
    except Exception:
        mult = 1
    if mult <= 1:
        return ""
    return "uks" if psi4_method_is_likely_dft(method) else "uhf"


def psi4_normalize_memory_gb(value, default_gb=8, min_gb=1, max_gb=256):
    txt = str(value or "").strip().lower()
    if txt.endswith("mb"):
        try:
            gb = int(round(float(txt[:-2].strip()) / 1024.0))
        except Exception:
            gb = int(default_gb)
    elif txt.endswith("gb"):
        try:
            gb = int(round(float(txt[:-2].strip())))
        except Exception:
            gb = int(default_gb)
    else:
        gb = _psi4_int_in_range(value, default_gb, min_gb, max_gb)
    return _psi4_int_in_range(gb, default_gb, min_gb, max_gb)


def psi4_normalize_threads(value, cpu_count=None, default=1):
    try:
        cap = int(cpu_count)
    except Exception:
        cap = int(os.cpu_count() or 1)
    cap = max(1, cap)
    return _psi4_int_in_range(value, default, 1, cap)


def psi4_normalize_scf_type(value, default="df"):
    scf = str(value or default or "df").strip().lower()
    if scf not in ("df", "pk", "direct", "out_of_core"):
        scf = str(default or "df").strip().lower()
    if scf not in ("df", "pk", "direct", "out_of_core"):
        scf = "df"
    return str(scf)


def psi4_normalize_pcm_solver(value, default="IEFPCM"):
    solver = str(value or default or "IEFPCM").strip().upper()
    if solver not in ("IEFPCM", "CPCM"):
        solver = str(default or "IEFPCM").strip().upper()
    if solver not in ("IEFPCM", "CPCM"):
        solver = "IEFPCM"
    return str(solver)


def psi4_normalize_irc_settings(direction="BOTH", points=20, step=0.20):
    idir = str(direction or "BOTH").strip().upper()
    if idir not in ("FORWARD", "BACKWARD", "BOTH"):
        idir = "BOTH"
    return {
        "irc_direction": str(idir),
        "irc_points": _psi4_int_in_range(points, 20, 4, 200),
        "irc_step": _psi4_float_at_least(step, 0.20, 0.01),
    }


def psi4_sanitize_freeze_atom_indices(indices):
    out = []
    for item in list(indices or []):
        try:
            idx = int(item)
        except Exception:
            continue
        if idx > 0 and idx not in out:
            out.append(int(idx))
    return sorted(out)


def psi4_build_driver_options(
    operation,
    basis,
    scf_type="df",
    geom_maxiter=100,
    irc_direction="BOTH",
    irc_points=20,
    irc_step=0.20,
    freeze_atom_indices=None,
    reference="",
):
    op = psi4_normalize_operation(operation)
    irc = psi4_normalize_irc_settings(irc_direction, irc_points, irc_step)
    freeze_atoms = psi4_sanitize_freeze_atom_indices(freeze_atom_indices)
    freeze_cart = []
    for ai in freeze_atoms:
        base = int(3 * (int(ai) - 1))
        freeze_cart.extend([int(base + 1), int(base + 2), int(base + 3)])
    opts = {
        "basis": str(basis or "def2-SVP"),
        "scf_type": psi4_normalize_scf_type(scf_type),
        "maxiter": 200,
        "geom_maxiter": _psi4_int_in_range(geom_maxiter, 100, 20, 1000),
        "e_convergence": 1e-8,
        "d_convergence": 1e-8,
        "write_trajectory": True,
    }
    if op == "ts":
        opts["opt_type"] = "TS"
    elif op == "irc":
        opts["opt_type"] = "IRC"
        # Psi4/OptKing accepts FORWARD or BACKWARD as a single run. Leonardo's
        # BOTH mode is orchestrated as two separate runs and combined later.
        opts["irc_direction"] = "FORWARD" if str(irc["irc_direction"]).upper() == "BOTH" else irc["irc_direction"]
        opts["irc_points"] = int(irc["irc_points"])
        opts["irc_step_size"] = float(irc["irc_step"])
    if freeze_cart and op in ("optimize", "stationary", "ts", "irc"):
        # OptKing takes a string of "atom xyz" pairs (1-based); a list makes psi4.set_options fail
        opts["frozen_cartesian"] = " ".join(f"{int(ai)} xyz" for ai in freeze_atoms)
    ref = str(reference or "").strip().lower()
    if ref:
        opts["reference"] = ref
    return {
        "options": dict(opts),
        "freeze_atom_indices": list(freeze_atoms),
        "freeze_cartesian": list(freeze_cart),
        **irc,
    }


def psi4_build_pcm_input_block(solvent="Water", solver="IEFPCM"):
    solvent_name = str(solvent or "Water").strip() or "Water"
    pcm_solver = psi4_normalize_pcm_solver(solver)
    # Psi4 expects this block through the pcm__input option for modern Python
    # input files; keeping it centralized prevents each workflow from drifting.
    return "\n".join(
        [
            "Units = Angstrom",
            "Medium {",
            f"  SolverType = {pcm_solver}",
            f"  Solvent = {solvent_name}",
            "}",
            "Cavity {",
            "  RadiiSet = UFF",
            "  Type = GePol",
            "  Area = 0.3",
            "  Scaling = False",
            "  Mode = Implicit",
            "}",
        ]
    )


def psi4_prepare_driver_job(job, solvent_candidates_fn=None, cpu_count=None):
    """
    Build a validated execution plan from the UI job dictionary.

    This function is intentionally free of Qt/PyMOL/Psi4 imports so unit tests
    can cover the driver choices before a costly quantum calculation starts.
    """
    data = dict(job or {})
    op = psi4_normalize_operation(data.get("operation", "optimize"))
    mb = psi4_normalize_method_basis(data.get("method", "wb97x-d"), data.get("basis", "def2-SVP"))
    block_charge, block_mult = psi4_parse_charge_multiplicity_from_block(data.get("molecule_block", ""))
    try:
        charge = int(data.get("charge", block_charge if block_charge is not None else 0) or 0)
    except Exception:
        charge = int(block_charge if block_charge is not None else 0)
    try:
        multiplicity = int(data.get("multiplicity", block_mult if block_mult is not None else 1) or 1)
    except Exception:
        multiplicity = int(block_mult if block_mult is not None else 1)
    multiplicity = max(1, int(multiplicity))
    reference = psi4_reference_for_multiplicity(
        mb["method"],
        multiplicity=multiplicity,
        explicit_reference=data.get("reference", ""),
    )
    use_pcm = bool(data.get("implicit", False))
    solvent_req = str(data.get("solvent", "Water") or "Water").strip() or "Water"
    if callable(solvent_candidates_fn):
        try:
            candidates = list(solvent_candidates_fn(solvent_req) or [])
        except Exception:
            candidates = []
    else:
        candidates = list(psi4_pcm_solvent_candidates(solvent_req) or [])
    solvent = str((candidates[0] if candidates else solvent_req) or "Water").strip() or "Water"
    scf_type = psi4_normalize_scf_type(data.get("scf_type", "df"))
    pcm_solver = psi4_normalize_pcm_solver(data.get("pcm_solver", "IEFPCM"))
    geom_maxiter = _psi4_int_in_range(data.get("geom_maxiter", 100), 100, 20, 1000)
    irc = psi4_normalize_irc_settings(
        data.get("irc_direction", "BOTH"),
        data.get("irc_points", 20),
        data.get("irc_step", 0.20),
    )
    driver = psi4_build_driver_options(
        op,
        mb["basis"],
        scf_type=scf_type,
        geom_maxiter=geom_maxiter,
        irc_direction=irc["irc_direction"],
        irc_points=irc["irc_points"],
        irc_step=irc["irc_step"],
        freeze_atom_indices=data.get("freeze_atom_indices", []),
        reference=reference,
    )
    return {
        "operation": str(op),
        "method": str(mb["method"]),
        "basis": str(mb["basis"]),
        "method_call": str(mb["method_call"]),
        "embedded_basis": str(mb["embedded_basis"]),
        "molecule_block": str(data.get("molecule_block", "") or "").strip(),
        "memory_gb": psi4_normalize_memory_gb(data.get("memory_gb", 8)),
        "threads": psi4_normalize_threads(data.get("threads", 1), cpu_count=cpu_count),
        "geom_maxiter": int(geom_maxiter),
        "implicit": bool(use_pcm),
        "solvent": str(solvent),
        "solvent_requested": str(solvent_req),
        "scf_type": str(scf_type),
        "pcm_solver": str(pcm_solver),
        "irc_direction": str(irc["irc_direction"]),
        "irc_points": int(irc["irc_points"]),
        "irc_step": float(irc["irc_step"]),
        "irc_require_valid_ts": bool(data.get("irc_require_valid_ts", True)),
        "esp_grid_file": str(data.get("esp_grid_file", "grid.dat") or "grid.dat").strip(),
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "reference": str(reference),
        "options": dict(driver["options"]),
        "freeze_atom_indices": list(driver["freeze_atom_indices"]),
        "freeze_cartesian": list(driver["freeze_cartesian"]),
        "pcm_input_block": psi4_build_pcm_input_block(solvent=solvent, solver=pcm_solver),
        "qst": dict(data.get("qst", {}) or {}),
    }


def _psi4_pcm_solvent_key(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def psi4_pcm_supported_solvents(include_explicit=False):
    out = [str(spec[0]) for spec in _PSI4_PCM_SOLVENT_SPECS]
    if bool(include_explicit):
        out.append("Explicit")
    return out


def psi4_resolve_density_cube(result, job_dir):
    res = dict(result or {})
    jdir = str(job_dir or "").strip()
    density_name = str(res.get("density_cube_file", "") or "").strip()
    density_cube = ""
    if density_name:
        if jdir and (not os.path.isabs(density_name)):
            density_cube = os.path.join(jdir, density_name)
        else:
            density_cube = density_name
    if density_cube and os.path.isfile(density_cube):
        return str(density_cube)
    search = []
    if jdir and os.path.isdir(jdir):
        for pat in ("*Dt*.cube", "*density*.cube", "*.cube"):
            search.extend(sorted(glob.glob(os.path.join(jdir, pat))))
    seen = set()
    filtered = []
    orb_files = {str(x).strip() for x in list(res.get("cube_files", []) or []) if str(x).strip()}
    for fp in search:
        if fp in seen:
            continue
        seen.add(fp)
        bn = os.path.basename(fp)
        if bn in orb_files:
            continue
        filtered.append(fp)
    if filtered:
        return str(filtered[0])
    return ""


def psi4_resolve_esp_cube(result, job_dir, density_cube, has_charges):
    res = dict(result or {})
    jdir = str(job_dir or "").strip()
    dens_cube = str(density_cube or "").strip()
    esp_name = str(res.get("esp_cube_file", "") or "").strip()
    esp_cube = os.path.join(jdir, esp_name) if (jdir and esp_name and (not os.path.isabs(esp_name))) else esp_name
    if esp_cube and os.path.isfile(esp_cube):
        return {
            "esp_cube": str(esp_cube),
            "esp_source": "qm_esp_cube",
            "needs_build": False,
            "error": "",
        }
    base_dir = jdir if jdir else os.path.dirname(dens_cube)
    fallback_cube = os.path.join(base_dir, "mulliken_esp_from_charges.cube")
    if not bool(has_charges):
        return {
            "esp_cube": str(fallback_cube),
            "esp_source": "mulliken_grid_fit",
            "needs_build": False,
            "error": "missing_charges",
        }
    return {
        "esp_cube": str(fallback_cube),
        "esp_source": "mulliken_grid_fit",
        "needs_build": True,
        "error": "",
    }


def psi4_resolve_homo_lumo_cubes(result, job_dir):
    res = dict(result or {})
    jdir = str(job_dir or "").strip()
    homo_name = str(res.get("homo_cube_file", "") or "").strip()
    lumo_name = str(res.get("lumo_cube_file", "") or "").strip()
    cfiles = [str(x) for x in list(res.get("cube_files", []) or []) if str(x).strip()]
    if (not homo_name) and (len(cfiles) >= 1):
        homo_name = str(cfiles[0])
    if (not lumo_name) and (len(cfiles) >= 2):
        lumo_name = str(cfiles[1])
    homo_cube = os.path.join(jdir, homo_name) if (jdir and homo_name and (not os.path.isabs(homo_name))) else homo_name
    lumo_cube = os.path.join(jdir, lumo_name) if (jdir and lumo_name and (not os.path.isabs(lumo_name))) else lumo_name
    return {
        "homo_name": str(homo_name),
        "lumo_name": str(lumo_name),
        "homo_cube": str(homo_cube),
        "lumo_cube": str(lumo_cube),
    }


def psi4_pcm_solvent_candidates(solvent_name):
    raw = str(solvent_name or "Water").strip()
    if not raw:
        raw = "Water"
    key = _psi4_pcm_solvent_key(raw)
    out = []

    def _push(v):
        vv = str(v or "").strip()
        if (not vv) or (vv in out):
            return
        out.append(vv)

    _push(raw)
    for canonical, aliases in _PSI4_PCM_SOLVENT_SPECS:
        match_keys = {_psi4_pcm_solvent_key(canonical)}
        for alias in aliases:
            match_keys.add(_psi4_pcm_solvent_key(alias))
        if key in match_keys:
            _push(canonical)
            for alias in aliases:
                _push(alias)
            break
    # Keep Water as a final conservative fallback for callers that try a
    # preferred solvent first and then retry with the most universally supported
    # PCM solvent.
    _push("Water")
    return out


def psi4_tail_file(path, max_lines=80):
    p = str(path or "").strip()
    if (not p) or (not os.path.isfile(p)):
        return ""
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        return "".join(lines[-int(max_lines) :]).strip()
    except Exception:
        return ""


def psi4_load_history_data(result, job_dir):
    out = {"coords_ang": np.zeros((0, 0, 0), dtype=float), "energy_hartree": np.array([], dtype=float), "symbols": []}
    res = dict(result or {})
    jdir = str(job_dir or "").strip()
    npz_name = str(res.get("history_npz", "") or "").strip()
    npz_path = os.path.join(jdir, npz_name) if (jdir and npz_name) else ""
    if npz_path and os.path.isfile(npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True)
            c = np.asarray(data.get("coords_ang"), dtype=float)
            e = np.asarray(data.get("energy_hartree"), dtype=float).reshape(-1)
            sy = [str(x) for x in np.asarray(data.get("atom_symbols")).reshape(-1).tolist()]
            if c.ndim == 3 and c.shape[2] == 3:
                out["coords_ang"] = c
                out["energy_hartree"] = e[: c.shape[0]] if e.size else np.array([], dtype=float)
                out["symbols"] = sy
                return out
        except Exception:
            pass
    e_list = res.get("history_energy_hartree")
    if isinstance(e_list, (list, tuple)):
        try:
            out["energy_hartree"] = np.asarray(e_list, dtype=float).reshape(-1)
        except Exception:
            out["energy_hartree"] = np.array([], dtype=float)
    sy_list = res.get("atom_symbols")
    if isinstance(sy_list, (list, tuple)):
        out["symbols"] = [str(x) for x in sy_list]
    return out


def psi4_load_vib_data(result, job_dir):
    out = {
        "mode_cart_ang": np.zeros((0, 0, 0), dtype=float),
        "eq_geom_ang": np.zeros((0, 0), dtype=float),
        "symbols": [],
        "freq_real": np.array([], dtype=float),
        "freq_imag": np.array([], dtype=float),
        "ir": np.array([], dtype=float),
        "trv": np.array([], dtype=object),
    }
    res = dict(result or {})
    jdir = str(job_dir or "").strip()
    npz_name = str(res.get("vib_modes_npz", "") or "").strip()
    npz_path = os.path.join(jdir, npz_name) if (jdir and npz_name) else ""
    if npz_path and os.path.isfile(npz_path):
        try:
            data = np.load(npz_path, allow_pickle=True)
            out["mode_cart_ang"] = np.asarray(data.get("mode_cart_ang"), dtype=float)
            out["eq_geom_ang"] = np.asarray(data.get("eq_geom_ang"), dtype=float)
            out["symbols"] = [str(x) for x in np.asarray(data.get("atom_symbols")).reshape(-1).tolist()]
            out["freq_real"] = np.asarray(data.get("freq_cm_real"), dtype=float).reshape(-1)
            out["freq_imag"] = np.asarray(data.get("freq_cm_imag"), dtype=float).reshape(-1)
            out["ir"] = np.asarray(data.get("ir_intensity_km_mol"), dtype=float).reshape(-1)
            out["trv"] = np.asarray(data.get("trv")).reshape(-1)
            return out
        except Exception:
            pass
    try:
        out["freq_real"] = np.asarray(res.get("freq_cm_real", []), dtype=float).reshape(-1)
        out["freq_imag"] = np.asarray(res.get("freq_cm_imag", []), dtype=float).reshape(-1)
        out["ir"] = np.asarray(res.get("ir_intensity_km_mol", []), dtype=float).reshape(-1)
        out["trv"] = np.asarray(res.get("freq_trv", []), dtype=object).reshape(-1)
    except Exception:
        pass
    sy_list = res.get("atom_symbols")
    if isinstance(sy_list, (list, tuple)):
        out["symbols"] = [str(x) for x in sy_list]
    return out


def psi4_prepare_vibrational_animation(vib_data, mode_index=None, amplitude_a=0.22, n_frames=28):
    vib = dict(vib_data or {})
    mode_cart = np.asarray(vib.get("mode_cart_ang", []), dtype=float)
    eq = np.asarray(vib.get("eq_geom_ang", []), dtype=float)
    syms = [str(x) for x in list(vib.get("symbols", []) or [])]
    if mode_cart.ndim != 3 or eq.ndim != 2 or eq.shape[1] != 3:
        return {"ok": False, "error": "invalid_shapes"}
    nm = int(mode_cart.shape[0])
    nat = int(mode_cart.shape[1])
    if (nm <= 0) or (eq.shape[0] != nat):
        return {"ok": False, "error": "mismatched_atom_count"}
    picked = mode_index
    if picked is None:
        try:
            freq_real = np.asarray(vib.get("freq_real", []), dtype=float).reshape(-1)
            freq_imag = np.asarray(vib.get("freq_imag", []), dtype=float).reshape(-1)
            ir = np.asarray(vib.get("ir", []), dtype=float).reshape(-1)
            trv = np.asarray(vib.get("trv", []), dtype=object).reshape(-1)
            valid = np.ones(nm, dtype=bool)
            if trv.size >= nm:
                valid &= np.array([str(trv[i]).upper().startswith("V") for i in range(nm)], dtype=bool)
            if freq_imag.size >= nm:
                valid &= np.abs(freq_imag[:nm]) < 1.0e-3
            if freq_real.size >= nm:
                valid &= np.abs(freq_real[:nm]) > 5.0
            if ir.size >= nm and np.any(valid):
                idx_pool = np.where(valid)[0]
                picked = int(idx_pool[int(np.argmax(ir[idx_pool]))]) if idx_pool.size else int(np.argmax(ir[:nm]))
            else:
                picked = int(np.where(valid)[0][0]) if np.any(valid) else 0
        except Exception:
            picked = 0
    try:
        mi = int(picked)
    except Exception:
        mi = 0
    mi = max(0, min(nm - 1, mi))
    vec = np.asarray(mode_cart[mi], dtype=float)
    amp = float(max(0.05, min(1.20, amplitude_a)))
    mx = float(np.max(np.linalg.norm(vec, axis=1))) if vec.size else 0.0
    if (not np.isfinite(mx)) or (mx < 1.0e-8):
        return {"ok": False, "error": "zero_mode_vector", "mode_index": int(mi)}
    vec = (vec / mx) * amp
    nfr = int(max(8, min(200, int(n_frames))))
    phase = np.sin((2.0 * np.pi * np.arange(nfr, dtype=float)) / float(nfr)).reshape((nfr, 1, 1))
    coords = eq[None, :, :] + (phase * vec[None, :, :])
    return {
        "ok": True,
        "error": "",
        "mode_index": int(mi),
        "coords_ang": np.asarray(coords, dtype=float),
        "symbols": syms,
        "obj_prefix": f"psi4_vib_m{int(mi) + 1}",
        "n_frames": int(nfr),
    }


def psi4_prepare_ir_spectrum(vib_data, include_broadened=True, fwhm_cm=18.0):
    vib = dict(vib_data or {})
    freq = np.asarray(vib.get("freq_real", []), dtype=float).reshape(-1)
    imag = np.asarray(vib.get("freq_imag", []), dtype=float).reshape(-1)
    ir = np.asarray(vib.get("ir", []), dtype=float).reshape(-1)
    trv = np.asarray(vib.get("trv", []), dtype=object).reshape(-1)
    n = int(min(freq.size, ir.size))
    if n <= 0:
        return {"ok": False, "error": "no_modes"}
    idx = np.arange(n, dtype=int)
    valid = np.isfinite(freq[:n]) & np.isfinite(ir[:n])
    if imag.size >= n:
        valid &= np.abs(imag[:n]) < 1.0e-3
    if trv.size >= n:
        valid &= np.array([str(trv[i]).upper().startswith("V") for i in range(n)], dtype=bool)
    valid &= np.abs(freq[:n]) > 5.0
    if not np.any(valid):
        return {"ok": False, "error": "no_valid_modes"}
    ii = idx[valid]
    xf = np.asarray(freq[ii], dtype=float)
    yf = np.asarray(ir[ii], dtype=float)
    x_min = float(max(200.0, float(np.min(xf) - 180.0)))
    x_max = float(min(4200.0, float(np.max(xf) + 180.0)))
    first_mode = int(ii[int(np.argmax(yf))]) if int(yf.size) > 0 else int(ii[0])
    out = {
        "ok": True,
        "error": "",
        "mode_indices": np.asarray(ii, dtype=int),
        "freq_cm": np.asarray(xf, dtype=float),
        "ir_km_mol": np.asarray(yf, dtype=float),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "first_mode_index": int(first_mode),
        "include_broadened": bool(include_broadened),
        "x_grid": np.array([], dtype=float),
        "y_norm": np.array([], dtype=float),
    }
    if not bool(include_broadened):
        return out
    sigma = float(max(1.0, fwhm_cm)) / float(2.0 * np.sqrt(2.0 * np.log(2.0)))
    x_grid = np.linspace(x_min, x_max, 2400)
    y_grid = np.zeros_like(x_grid, dtype=float)
    for f, inten in zip(xf, yf):
        y_grid += float(inten) * np.exp(-0.5 * ((x_grid - float(f)) / sigma) ** 2)
    y_norm = y_grid
    if np.max(y_norm) > 1.0e-12:
        y_norm = 100.0 * (y_norm / np.max(y_norm))
    out["x_grid"] = np.asarray(x_grid, dtype=float)
    out["y_norm"] = np.asarray(y_norm, dtype=float)
    return out


def psi4_element_token(element):
    e = str(element or "").strip().upper()
    if not e:
        return "C"
    if len(e) == 1:
        return e
    return e[0] + e[1:].lower()


def psi4_atomic_number(element, rdkit_lookup_fn=None):
    tok = psi4_element_token(element)
    if callable(rdkit_lookup_fn):
        try:
            return int(rdkit_lookup_fn(tok))
        except Exception:
            pass
    table = {
        "H": 1,
        "HE": 2,
        "LI": 3,
        "BE": 4,
        "B": 5,
        "C": 6,
        "N": 7,
        "O": 8,
        "F": 9,
        "NE": 10,
        "NA": 11,
        "MG": 12,
        "AL": 13,
        "SI": 14,
        "P": 15,
        "S": 16,
        "CL": 17,
        "AR": 18,
        "K": 19,
        "CA": 20,
        "SC": 21,
        "TI": 22,
        "V": 23,
        "CR": 24,
        "MN": 25,
        "FE": 26,
        "CO": 27,
        "NI": 28,
        "CU": 29,
        "ZN": 30,
        "BR": 35,
        "I": 53,
    }
    return int(table.get(str(tok).upper(), 6))


def psi4_atomic_mass_amu(element):
    tok = psi4_element_token(element).upper()
    table = {
        "H": 1.007825,
        "HE": 4.002603,
        "LI": 7.016004,
        "BE": 9.012182,
        "B": 11.009305,
        "C": 12.000000,
        "N": 14.003074,
        "O": 15.994915,
        "F": 18.998403,
        "NE": 19.992440,
        "NA": 22.989770,
        "MG": 23.985042,
        "AL": 26.981538,
        "SI": 27.976927,
        "P": 30.973762,
        "S": 31.972071,
        "CL": 34.968853,
        "AR": 39.962383,
        "K": 38.963707,
        "CA": 39.962591,
        "FE": 55.934936,
        "CU": 62.929598,
        "ZN": 63.929142,
        "BR": 78.918338,
        "I": 126.904468,
    }
    return float(table.get(tok, 12.0))


def psi4_validate_irc_ts_frequencies(freq_real, freq_imag, threshold_cm=10.0):
    """
    Validate the harmonic frequency signature expected before an IRC.

    IRC theory starts from a first-order saddle point, so a usable TS should
    have exactly one imaginary normal mode. Psi4 stores imaginary modes in the
    imaginary component of omega; some parsers represent them as negative real
    values, so both conventions are accepted here.
    """
    try:
        fr = np.asarray(freq_real, dtype=float).reshape(-1)
    except Exception:
        fr = np.array([], dtype=float)
    try:
        fi = np.asarray(freq_imag, dtype=float).reshape(-1)
    except Exception:
        fi = np.array([], dtype=float)
    n = int(max(fr.size, fi.size))
    if fr.size < n:
        fr = np.pad(fr, (0, n - fr.size), mode="constant")
    if fi.size < n:
        fi = np.pad(fi, (0, n - fi.size), mode="constant")
    try:
        thresh = float(abs(float(threshold_cm)))
    except Exception:
        thresh = 10.0
    imag = []
    for idx in range(n):
        val_im = float(fi[idx])
        val_re = float(fr[idx])
        if abs(val_im) >= thresh:
            imag.append({"mode_index": int(idx), "imag_cm1": float(abs(val_im)), "source": "imag"})
        elif val_re <= -thresh:
            imag.append({"mode_index": int(idx), "imag_cm1": float(abs(val_re)), "source": "negative_real"})
    primary = max(imag, key=lambda item: float(item["imag_cm1"])) if imag else {}
    return {
        "ok": bool(len(imag) == 1),
        "imaginary_mode_count": int(len(imag)),
        "imaginary_modes": list(imag),
        "primary_mode_index": int(primary.get("mode_index", -1)) if primary else -1,
        "primary_imag_cm1": float(primary.get("imag_cm1", 0.0)) if primary else 0.0,
        "threshold_cm": float(thresh),
    }


def psi4_build_irc_profile_payload(coords_ang, energy_hartree, symbols):
    """
    Build an IRC plotting/export payload from a Psi4 trajectory history.

    The intrinsic coordinate is approximated as cumulative arc length in
    mass-weighted Cartesian coordinates, sqrt(sum_i m_i dR_i^2), with Angstrom
    converted to bohr. The highest-energy frame is treated as the TS frame and
    assigned coordinate 0, giving negative/positive branches for plotting.
    """
    c = np.asarray(coords_ang, dtype=float)
    e = np.asarray(energy_hartree, dtype=float).reshape(-1)
    if c.ndim != 3 or c.shape[2] != 3 or c.shape[0] < 2 or e.size < 2:
        return {"ok": False, "error": "invalid_history"}
    n = int(min(c.shape[0], e.size))
    c = c[:n, :, :]
    e = e[:n]
    sy = [str(x) for x in list(symbols or [])]
    if len(sy) != int(c.shape[1]):
        sy = ["C"] * int(c.shape[1])
    masses = np.asarray([psi4_atomic_mass_amu(s) for s in sy], dtype=float).reshape(1, -1, 1)
    ang_to_bohr = 1.0 / 0.529177210903
    cb = c * float(ang_to_bohr)

    def _mw_dist(i, j):
        d = cb[int(j), :, :] - cb[int(i), :, :]
        return float(np.sqrt(np.sum(masses.reshape(-1, 1) * (d * d))))

    ts_idx = int(np.argmax(e))
    rc = np.zeros(n, dtype=float)
    for i in range(ts_idx + 1, n):
        rc[i] = rc[i - 1] + _mw_dist(i - 1, i)
    for i in range(ts_idx - 1, -1, -1):
        rc[i] = rc[i + 1] - _mw_dist(i, i + 1)
    rel_kcal = (e - float(np.min(e))) * 627.5094740631
    branches = []
    for i in range(n):
        if i < ts_idx:
            branches.append("backward")
        elif i > ts_idx:
            branches.append("forward")
        else:
            branches.append("ts")
    return {
        "ok": True,
        "point": [int(i) for i in range(n)],
        "coordinate_amu05_bohr": [float(v) for v in rc.tolist()],
        "energy_hartree": [float(v) for v in e.tolist()],
        "relative_energy_kcal_mol": [float(v) for v in rel_kcal.tolist()],
        "branch": branches,
        "ts_frame_index": int(ts_idx),
        "n_points": int(n),
        "coordinate_unit": "amu^0.5 bohr",
    }


def psi4_qst_reorder_atoms_to_template(template_atoms, probe_atoms):
    """
    Reorder a QST endpoint/guess so every geometry shares one atom mapping.

    QST2/QST3 only make chemical sense when reactant, product, and optional TS
    guess describe the same atoms in the same order. We first accept exact
    element order; if the source came from PDB-like atom names, unique matching
    names are used as a second path. No graph matching is attempted here because
    changing atom identity during a TS build is worse than failing loudly.
    """
    ta = [dict(a or {}) for a in list(template_atoms or [])]
    pa = [dict(a or {}) for a in list(probe_atoms or [])]
    if len(ta) != len(pa):
        return None, f"atom_count_mismatch:{len(ta)}!={len(pa)}"

    def _elem_list(items):
        return [psi4_element_token(a.get("element", a.get("symbol", "C"))) for a in list(items or [])]

    telem = _elem_list(ta)
    pelem = _elem_list(pa)
    if telem == pelem:
        return [dict(a) for a in pa], "element_order"
    tnames = [str(a.get("name", "")).strip().upper() for a in ta]
    pnames = [str(a.get("name", "")).strip().upper() for a in pa]
    if all(n for n in tnames) and all(n for n in pnames):
        if (len(set(tnames)) == len(tnames)) and (len(set(pnames)) == len(pnames)) and (set(tnames) == set(pnames)):
            pmap = {str(a.get("name", "")).strip().upper(): dict(a) for a in pa}
            reordered = [dict(pmap[n]) for n in tnames]
            if _elem_list(reordered) == telem:
                return reordered, "name_map"
    return None, (
        "Atom order/topology mismatch between QST sources. "
        "Use structures with identical atom ordering or matching unique atom names."
    )


def psi4_qst_align_coords_kabsch(ref_xyz, mob_xyz):
    ref = np.asarray(ref_xyz, dtype=float)
    mob = np.asarray(mob_xyz, dtype=float)
    if ref.shape != mob.shape or ref.ndim != 2 or ref.shape[1] != 3:
        raise RuntimeError("Kabsch alignment shape mismatch")
    if ref.shape[0] <= 0:
        return np.asarray(mob, dtype=float)
    cref = np.mean(ref, axis=0)
    cmob = np.mean(mob, axis=0)
    refc = ref - cref
    mobc = mob - cmob
    h = np.dot(mobc.T, refc)
    u, _s, vt = np.linalg.svd(h)
    r = np.dot(vt.T, u.T)
    if float(np.linalg.det(r)) < 0.0:
        vt[-1, :] *= -1.0
        r = np.dot(vt.T, u.T)
    return np.asarray(np.dot(mobc, r) + cref, dtype=float)


def psi4_qst_min_pair_distance(xyz):
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return float("inf")
    best = float("inf")
    for i in range(int(arr.shape[0])):
        for j in range(i + 1, int(arr.shape[0])):
            d = float(np.linalg.norm(arr[i, :] - arr[j, :]))
            if d < best:
                best = d
    return float(best)


def psi4_qst_build_guess_atoms(react_atoms, prod_atoms, ts_guess_atoms=None, ts_weight=0.35, n_images=9):
    """
    Build a synchronous-transit QST2/QST3 starting geometry and scan images.

    The builder follows the practical ST/QST split used by Gaussian-style
    QST2/QST3 workflows: construct a reactant/product synchronous path, use an
    optional TS guess to make that path quadratic, then hand the best geometry
    to a true TS optimizer. It does not evaluate gradients. Gradients and the
    Hessian-based TS refinement belong to Psi4/OptKing when Leonardo later runs
    ``psi4.optimize(..., opt_type="TS")`` and validates by frequency/IRC.
    """
    ra = [dict(a or {}) for a in list(react_atoms or [])]
    pa = [dict(a or {}) for a in list(prod_atoms or [])]
    if len(ra) <= 0 or len(pa) <= 0:
        return {"guess_atoms": [], "note": "Reactant and product must both contain atoms.", "payload": {}}
    p_ord, p_mode = psi4_qst_reorder_atoms_to_template(ra, pa)
    if p_ord is None:
        return {"guess_atoms": [], "note": f"Product mapping failed: {p_mode}", "payload": {}}

    t_ord = None
    t_mode = ""
    raw_ts_atoms = list(ts_guess_atoms or []) if ts_guess_atoms is not None else []
    if raw_ts_atoms:
        t_ord, t_mode = psi4_qst_reorder_atoms_to_template(ra, raw_ts_atoms)
        if t_ord is None:
            return {"guess_atoms": [], "note": f"TS-guess mapping failed: {t_mode}", "payload": {}}

    try:
        rxyz = np.asarray([[float(a.get("x", 0.0)), float(a.get("y", 0.0)), float(a.get("z", 0.0))] for a in ra], dtype=float)
        pxyz = np.asarray([[float(a.get("x", 0.0)), float(a.get("y", 0.0)), float(a.get("z", 0.0))] for a in p_ord], dtype=float)
    except Exception as e:
        return {"guess_atoms": [], "note": f"Failed parsing coordinates for QST build: {e}", "payload": {}}

    try:
        p_aln = psi4_qst_align_coords_kabsch(rxyz, pxyz)
    except Exception as e:
        return {"guess_atoms": [], "note": f"Failed aligning product to reactant: {e}", "payload": {}}

    aligned_mid = 0.5 * (rxyz + p_aln)
    raw_mid = 0.5 * (rxyz + pxyz)
    aligned_min = psi4_qst_min_pair_distance(aligned_mid)
    raw_min = psi4_qst_min_pair_distance(raw_mid)
    path_product_xyz = p_aln
    qst_geometry_mode = "aligned_midpoint"
    # Symmetric endpoint pairs can make the Kabsch-aligned average collapse
    # equivalent atoms. Prefer the direct path only when it clearly removes a
    # severe nonphysical clash; otherwise alignment avoids rigid-body artifacts.
    if aligned_min < 0.45 and raw_min > (aligned_min + 0.20):
        path_product_xyz = pxyz
        qst_geometry_mode = "direct_midpoint"

    linear_mid = 0.5 * (rxyz + path_product_xyz)
    qst_mode = "qst2"
    center_xyz = np.asarray(linear_mid, dtype=float)
    w = 0.0
    if t_ord is not None:
        try:
            txyz = np.asarray([[float(a.get("x", 0.0)), float(a.get("y", 0.0)), float(a.get("z", 0.0))] for a in t_ord], dtype=float)
            t_aln = psi4_qst_align_coords_kabsch(linear_mid, txyz)
            w = float(max(0.0, min(1.0, float(ts_weight))))
            center_xyz = ((1.0 - w) * linear_mid) + (w * t_aln)
            qst_mode = "qst3"
        except Exception as e:
            return {"guess_atoms": [], "note": f"Failed applying TS guess alignment: {e}", "payload": {}}

    if not np.all(np.isfinite(center_xyz)):
        return {"guess_atoms": [], "note": "QST guess coordinates became non-finite.", "payload": {}}

    try:
        n_img = int(n_images)
    except Exception:
        n_img = 9
    n_img = int(max(3, min(21, n_img)))
    if n_img % 2 == 0:
        n_img += 1
    lambdas = [(i + 1.0) / float(n_img + 1.0) for i in range(n_img)]
    if not any(abs(float(x) - 0.5) < 1.0e-9 for x in lambdas):
        lambdas.append(0.5)
        lambdas = sorted(set(round(float(x), 8) for x in lambdas))

    images = []
    curvature = center_xyz - linear_mid
    for lam in lambdas:
        lam = float(lam)
        if qst_mode == "qst3":
            # Quadratic interpolation through reactant, product, and the
            # effective TS guess: Q(0)=R, Q(1)=P, Q(0.5)=TS_guess.
            xyz = ((1.0 - lam) * rxyz) + (lam * path_product_xyz) + (4.0 * lam * (1.0 - lam) * curvature)
        else:
            xyz = ((1.0 - lam) * rxyz) + (lam * path_product_xyz)
        min_d = psi4_qst_min_pair_distance(xyz)
        image_atoms = []
        for i, a in enumerate(ra):
            aa = dict(a)
            aa["x"] = float(xyz[i, 0])
            aa["y"] = float(xyz[i, 1])
            aa["z"] = float(xyz[i, 2])
            image_atoms.append(aa)
        images.append(
            {
                "lambda": float(lam),
                "atoms": image_atoms,
                "min_pair_distance_ang": float(min_d),
                "rms_from_react_ang": float(np.sqrt(np.mean(np.sum((xyz - rxyz) ** 2, axis=1)))),
                "rms_to_product_ang": float(np.sqrt(np.mean(np.sum((xyz - path_product_xyz) ** 2, axis=1)))),
            }
        )

    selected = min(images, key=lambda item: abs(float(item.get("lambda", 0.0)) - 0.5))
    if float(selected.get("min_pair_distance_ang", 0.0)) < 0.35:
        viable = [im for im in images if float(im.get("min_pair_distance_ang", 0.0)) >= 0.35]
        if viable:
            selected = min(viable, key=lambda item: abs(float(item.get("lambda", 0.0)) - 0.5))
    guess_atoms = [dict(a) for a in list(selected.get("atoms", []) or [])]

    notes = [
        f"prod_map={p_mode}",
        f"guess={qst_geometry_mode}",
        f"path={'quadratic_synchronous' if qst_mode == 'qst3' else 'linear_synchronous'}",
        f"scan_images={len(images)}",
    ]
    if t_ord is not None:
        notes.append(f"ts_map={t_mode}")
        notes.append(f"ts_weight={w:.2f}")
    notes.append("gradients=psi4_optking_ts")

    payload = {
        "mode": qst_mode,
        "product_mapping": p_mode,
        "ts_mapping": t_mode,
        "geometry_mode": qst_geometry_mode,
        "ts_weight": float(w),
        "selected_lambda": float(selected.get("lambda", 0.5)),
        "selected_min_pair_distance_ang": float(selected.get("min_pair_distance_ang", 0.0)),
        "react_atoms": [dict(a) for a in ra],
        "product_atoms": [dict(a) for a in p_ord],
        "ts_guess_atoms": [dict(a) for a in list(t_ord or [])],
        "path_product_atoms": [
            {**dict(ra[i]), "x": float(path_product_xyz[i, 0]), "y": float(path_product_xyz[i, 1]), "z": float(path_product_xyz[i, 2])}
            for i in range(len(ra))
        ],
        "images": images,
        "gradient_owner": "Psi4/OptKing optimize(opt_type=TS)",
    }
    return {"guess_atoms": guess_atoms, "note": "; ".join(notes), "payload": payload}


def psi4_guess_multiplicity_from_atoms(atoms, charge=0, atomic_number_fn=None):
    atom_list = list(atoms or [])
    anum_fn = atomic_number_fn if callable(atomic_number_fn) else psi4_atomic_number
    try:
        zsum = int(sum(int(anum_fn(a.get("element", a.get("symbol", "C")))) for a in atom_list))
        nelec = int(zsum - int(charge))
        return 1 if (nelec % 2 == 0) else 2
    except Exception:
        return 1


def psi4_collect_atoms_from_xyz(xyz_file, element_token_fn=None):
    atoms = []
    p = str(xyz_file or "").strip()
    if (not p) or (not os.path.isfile(p)):
        return atoms
    to_symbol = element_token_fn if callable(element_token_fn) else psi4_element_token
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            raw_lines = [ln.strip() for ln in fh if str(ln).strip()]
    except Exception:
        return atoms
    if not raw_lines:
        return atoms
    start = 0
    try:
        int(str(raw_lines[0]).split()[0])
        start = 2 if len(raw_lines) > 2 else 1
    except Exception:
        start = 0
    idx = 1
    for ln in raw_lines[start:]:
        parts = str(ln).split()
        if len(parts) < 4:
            continue
        sym = str(to_symbol(parts[0]) or "C")
        try:
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
        except Exception:
            continue
        atoms.append(
            {
                "index": int(idx),
                "name": f"{sym}{idx}",
                "element": str(sym).upper(),
                "symbol": str(sym),
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
        )
        idx += 1
    return atoms


def psi4_mol2_atom_type_to_element(atom_type, atom_name="", infer_element_fn=None):
    txt = str(atom_type or "").strip()
    base = txt.split(".", 1)[0] if txt else ""
    letters = re.sub(r"[^A-Za-z]", "", base).upper()
    two_letter = {
        "CL",
        "BR",
        "NA",
        "MG",
        "CA",
        "FE",
        "ZN",
        "CU",
        "MN",
        "CO",
        "NI",
        "SI",
        "AL",
        "LI",
        "SE",
        "CD",
        "AG",
        "PT",
        "HG",
        "SN",
        "PB",
        "IR",
        "OS",
        "RU",
        "RH",
        "PD",
        "GA",
        "GE",
        "AS",
        "KR",
        "XE",
        "NE",
        "HE",
        "AR",
        "CS",
        "BA",
    }
    if (len(letters) >= 2) and (letters[:2] in two_letter):
        return letters[:2]
    if letters:
        return letters[:1]
    if callable(infer_element_fn):
        try:
            fallback = str(infer_element_fn(atom_name) or "").strip().upper()
            if fallback:
                return fallback
        except Exception:
            pass
    return "C"


def psi4_collect_atoms_from_mol2(mol2_file, atom_type_to_element_fn=None, element_token_fn=None):
    atoms = []
    p = str(mol2_file or "").strip()
    if (not p) or (not os.path.isfile(p)):
        return atoms
    type_to_el = atom_type_to_element_fn if callable(atom_type_to_element_fn) else psi4_mol2_atom_type_to_element
    to_symbol = element_token_fn if callable(element_token_fn) else psi4_element_token
    section = None
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = str(raw).rstrip("\n")
                if not line.strip():
                    continue
                if line.startswith("@<TRIPOS>"):
                    section = line.strip().upper()
                    continue
                if section != "@<TRIPOS>ATOM":
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    idx = int(parts[0])
                    name = str(parts[1]).strip() or f"A{idx}"
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    atom_type = str(parts[5]).strip()
                except Exception:
                    continue
                el = str(type_to_el(atom_type, name) or "C").strip().upper() or "C"
                atoms.append(
                    {
                        "index": int(idx),
                        "name": str(name),
                        "element": str(el),
                        "symbol": str(to_symbol(el) or "C"),
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                    }
                )
    except Exception:
        return []
    atoms.sort(key=lambda d: int(d.get("index", 0)))
    return atoms


def psi4_load_mol2_atom_records(mol2_file, atom_type_to_element_fn=None):
    records = []
    p = str(mol2_file or "").strip()
    if (not p) or (not os.path.isfile(p)):
        return records
    type_to_el = atom_type_to_element_fn if callable(atom_type_to_element_fn) else psi4_mol2_atom_type_to_element
    section = None
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = str(raw).strip()
                if not line:
                    continue
                if line.startswith("@<TRIPOS>"):
                    section = line.upper()
                    continue
                if section != "@<TRIPOS>ATOM":
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    atom_id = int(parts[0])
                except Exception:
                    continue
                atom_name = str(parts[1]).strip().upper()
                atom_type = str(parts[5]).strip()
                charge = 0.0
                if len(parts) >= 9:
                    try:
                        charge = float(parts[8])
                    except Exception:
                        charge = 0.0
                elem = str(type_to_el(atom_type, atom_name) or "C").strip().upper() or "C"
                records.append(
                    {
                        "id": int(atom_id),
                        "name": str(atom_name),
                        "element": str(elem),
                        "charge": float(charge),
                    }
                )
    except Exception:
        return []
    records.sort(key=lambda r: int(r.get("id", 0)))
    return records


def psi4_load_mol2_charges_and_bonds(mol2_file):
    charges_by_name = {}
    bond_name_pairs = set()
    p = str(mol2_file or "").strip()
    if (not p) or (not os.path.isfile(p)):
        return charges_by_name, bond_name_pairs
    atom_id_to_name = {}
    section = None
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = str(raw).strip()
                if not line:
                    continue
                if line.startswith("@<TRIPOS>"):
                    section = line.upper()
                    continue
                if section == "@<TRIPOS>ATOM":
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    try:
                        atom_id = int(parts[0])
                    except Exception:
                        continue
                    atom_name = str(parts[1]).strip()
                    if not atom_name:
                        continue
                    atom_name_u = atom_name.upper()
                    atom_id_to_name[int(atom_id)] = str(atom_name_u)
                    charge = 0.0
                    if len(parts) >= 9:
                        try:
                            charge = float(parts[8])
                        except Exception:
                            charge = 0.0
                    charges_by_name[str(atom_name_u)] = float(charge)
                elif section == "@<TRIPOS>BOND":
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    try:
                        a1 = int(parts[1])
                        a2 = int(parts[2])
                    except Exception:
                        continue
                    n1 = atom_id_to_name.get(int(a1))
                    n2 = atom_id_to_name.get(int(a2))
                    if (not n1) or (not n2) or (n1 == n2):
                        continue
                    pair = tuple(sorted((str(n1), str(n2))))
                    bond_name_pairs.add(pair)
    except Exception:
        return {}, set()
    return charges_by_name, bond_name_pairs


def psi4_collect_ligand_charges_from_file(ligand_file):
    path = str(ligand_file or "").strip()
    if (not path) or (not os.path.isfile(path)):
        return []
    ext = os.path.splitext(path)[1].lower()
    charges = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            if ext == ".mol2":
                in_atoms = False
                for raw in fh:
                    line = str(raw).rstrip("\n")
                    if line.startswith("@<TRIPOS>ATOM"):
                        in_atoms = True
                        continue
                    if line.startswith("@<TRIPOS>") and in_atoms:
                        break
                    if (not in_atoms) or (not line.strip()):
                        continue
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    try:
                        charges.append(float(parts[8]))
                    except Exception:
                        continue
            elif ext == ".pdbqt":
                for raw in fh:
                    if not (raw.startswith("ATOM") or raw.startswith("HETATM")):
                        continue
                    line = str(raw).rstrip("\n")
                    q = None
                    try:
                        q = float(str(line[70:76]).strip())
                    except Exception:
                        q = None
                    if q is None:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                q = float(parts[-2])
                            except Exception:
                                q = None
                    if q is not None:
                        charges.append(float(q))
    except Exception:
        return []
    return charges


def psi4_estimate_net_charge_from_charges(charges):
    try:
        arr = np.asarray(list(charges or []), dtype=float).reshape(-1)
        if arr.size <= 0:
            return 0
        return int(round(float(np.sum(arr))))
    except Exception:
        return 0


def psi4_prepare_tleap_ligand_build(mol2_file, frcmod_file, atom_type="gaff2", out_dir=None, tleap_bin="tleap"):
    mol2 = str(mol2_file or "").strip()
    frc = str(frcmod_file or "").strip()
    if (not mol2) or (not os.path.isfile(mol2)):
        return {"ok": False, "error": "Missing MOL2 file for tleap."}
    if (not frc) or (not os.path.isfile(frc)):
        return {"ok": False, "error": "Missing FRCMOD file for tleap."}
    work = str(out_dir or os.path.dirname(mol2) or ".")
    prmtop = os.path.join(work, "ligand.prmtop")
    inpcrd = os.path.join(work, "ligand.inpcrd")
    leapin = os.path.join(work, "tleap_ligand.in")
    atom_t = str(atom_type or "gaff2").strip().lower()
    leaprc = "leaprc.gaff2" if "gaff2" in atom_t else "leaprc.gaff"
    script = (
        f"source {leaprc}\n"
        f"LIG = loadmol2 {mol2}\n"
        f"loadamberparams {frc}\n"
        f"check LIG\n"
        f"saveamberparm LIG {prmtop} {inpcrd}\n"
        "quit\n"
    )
    cmdline = [str(tleap_bin or "tleap"), "-f", str(leapin)]
    return {
        "ok": True,
        "error": "",
        "mol2": str(mol2),
        "frcmod": str(frc),
        "work": str(work),
        "prmtop": str(prmtop),
        "inpcrd": str(inpcrd),
        "leapin": str(leapin),
        "script": str(script),
        "cmdline": list(cmdline),
        "atom_type": str(atom_t),
        "leaprc": str(leaprc),
    }


def psi4_pick_charged_mol2_for_rebuild(parameterized_mol2, docking_ligand_file=""):
    mol2 = str(parameterized_mol2 or "").strip()
    if mol2 and os.path.isfile(mol2):
        return str(mol2)
    src = str(docking_ligand_file or "").strip()
    if src.lower().endswith(".mol2") and os.path.isfile(src):
        return str(src)
    return ""


def psi4_prepare_rebuild_output_paths(mol2_file, work_dir=None, fallback_work_dir="."):
    mol2 = str(mol2_file or "").strip()
    work = str(work_dir or "").strip()
    if not work:
        work = os.path.dirname(mol2) or str(fallback_work_dir or ".")
    return {
        "work": str(work),
        "out_mol2": str(os.path.join(work, "ligand_rc_preserved.mol2")),
        "out_frcmod": str(os.path.join(work, "ligand_rc_preserved.frcmod")),
        "typed_tmp": str(os.path.join(work, "ligand_typed_tmp.mol2")),
    }


def psi4_build_antechamber_cmd(
    antechamber_bin,
    source_file,
    source_format,
    out_mol2,
    charge_method="bcc",
    net_charge=0,
    atom_type="gaff2",
    dr=None,
):
    cmd = [
        str(antechamber_bin or "antechamber"),
        "-i",
        str(source_file or ""),
        "-fi",
        str(source_format or "mol2"),
        "-o",
        str(out_mol2 or ""),
        "-fo",
        "mol2",
        "-c",
        str(charge_method or "bcc"),
        "-nc",
        str(int(net_charge or 0)),
        "-at",
        str(atom_type or "gaff2"),
        "-s",
        "2",
    ]
    if dr is not None:
        cmd.extend(["-dr", str(dr)])
    return list(cmd)


def psi4_build_parmchk2_cmd(parmchk2_bin, in_mol2, out_frcmod):
    return [
        str(parmchk2_bin or "parmchk2"),
        "-i",
        str(in_mol2 or ""),
        "-f",
        "mol2",
        "-o",
        str(out_frcmod or ""),
    ]


def psi4_build_rebuild_failure_payload(stage, rc=1, stderr="", stdout="", fallback_errors=None):
    tag = str(stage or "").strip().lower()
    payload = {
        "ok": False,
        "error": str(stage or "rebuild_failed"),
        "stderr": str((stderr or "")[-1200:]),
        "stdout": str((stdout or "")[-1200:]),
    }
    if tag == "antechamber_rc":
        payload["error"] = f"antechamber(rc) failed rc={int(rc)}"
        fb = [str(x).strip() for x in list(fallback_errors or []) if str(x).strip()]
        if fb:
            payload["fallback"] = "; ".join(fb[-4:])
    elif tag == "parmchk2":
        payload["error"] = f"parmchk2 failed rc={int(rc)}"
    else:
        payload["error"] = f"{str(stage or 'rebuild_failed')} rc={int(rc)}"
    return payload


def psi4_build_rebuild_success_payload(
    out_mol2,
    out_frcmod,
    out_prmtop,
    out_inpcrd,
    xml_path,
    net_charge,
    atom_type="gaff2",
    rebuild_mode="",
    rebuild_notes=None,
):
    payload = {
        "ok": True,
        "mol2": str(out_mol2 or ""),
        "frcmod": str(out_frcmod or ""),
        "prmtop": str(out_prmtop or ""),
        "inpcrd": str(out_inpcrd or ""),
        "xml": str(xml_path or ""),
        "charge": int(net_charge or 0),
        "atom_type": str(atom_type or "gaff2"),
    }
    mode = str(rebuild_mode or "").strip()
    if mode:
        payload["rebuild_mode"] = mode
    notes = [str(x).strip() for x in list(rebuild_notes or []) if str(x).strip()]
    if notes:
        payload["rebuild_notes"] = notes
    return payload


def psi4_build_net_charge_candidates(requested_charge=0, formal_charge=None, atomic_z_sum=None, clip_min=-8, clip_max=8):
    out = []

    def _push(v):
        try:
            vv = int(v)
        except Exception:
            return
        vv = int(np.clip(vv, int(clip_min), int(clip_max)))
        if vv not in out:
            out.append(vv)

    _push(requested_charge)
    _push(formal_charge)
    parity = None
    try:
        if atomic_z_sum is not None:
            parity = int(atomic_z_sum) % 2
    except Exception:
        parity = None
    if parity is not None:
        try:
            base = int(out[0]) if out else int(requested_charge)
        except Exception:
            base = 0
        for d in (-2, -1, 1, 2):
            c = int(base + d)
            if int(c % 2) == int(parity):
                _push(c)
        if formal_charge is not None:
            try:
                fc = int(formal_charge)
                for d in (-2, -1, 1, 2):
                    c = int(fc + d)
                    if int(c % 2) == int(parity):
                        _push(c)
            except Exception:
                pass
    if not out:
        try:
            out = [int(requested_charge)]
        except Exception:
            out = [0]
    return [int(v) for v in out]


def psi4_detect_sqm_charge_mismatch_hint(sqm_text):
    txt = str(sqm_text or "")
    low = txt.lower()
    for key in ("odd number of electrons", "qmcharge", "spin", "incorrectly"):
        if str(key).lower() in low:
            return "sqm_hint=odd_electron_or_charge_mismatch"
    return ""


def psi4_format_antechamber_attempt_brief(net_charge, rc, sqm_hint=""):
    note = str(sqm_hint or "").strip()
    base = f"nc={int(net_charge)} rc={int(rc)}"
    return f"{base} {note}".strip()


def psi4_format_antechamber_attempts_failure(attempt_brief, stderr="", stdout="", stderr_tail=400, stdout_tail=300):
    attempts = [str(x).strip() for x in list(attempt_brief or []) if str(x).strip()]
    se = str(stderr or "")
    so = str(stdout or "")
    return (
        f"antechamber failed after nc attempts [{'; '.join(attempts)}] "
        f"| stderr={se[-int(stderr_tail):]} | stdout={so[-int(stdout_tail):]}"
    )


def psi4_format_tool_rc_failure(tool_name, rc, stderr="", stdout="", stderr_tail=400, stdout_tail=300):
    nm = str(tool_name or "tool").strip() or "tool"
    se = str(stderr or "")
    so = str(stdout or "")
    return (
        f"{nm} failed (rc={int(rc)}) "
        f"| stderr={se[-int(stderr_tail):]} | stdout={so[-int(stdout_tail):]}"
    )


def psi4_prepare_leadopt_parameterization_paths(work_dir):
    work = str(work_dir or "").strip() or "."
    return {
        "work": str(work),
        "rdkit_sdf": str(os.path.join(work, "ligand_for_param.sdf")),
        "pose_pdb": str(os.path.join(work, "ligand_for_param.pdb")),
        "out_mol2": str(os.path.join(work, "ligand_gaff2.mol2")),
        "out_frcmod": str(os.path.join(work, "ligand_gaff2.frcmod")),
    }


def _psi4_path_is_nonempty(path, exists_fn=None):
    """
    Return True only when a path exists and, for real local files, contains data.

    Several tests inject a virtual `exists_fn`, so real filesystem size checks are
    used when possible and otherwise the injected existence contract is trusted.
    """
    p = str(path or "").strip()
    if not p:
        return False
    custom_exists = callable(exists_fn)
    exists = exists_fn if custom_exists else os.path.isfile
    try:
        if not bool(exists(p)):
            return False
    except Exception:
        return False
    try:
        if os.path.isfile(p):
            return int(os.path.getsize(p)) > 0
    except Exception:
        return False
    return bool(custom_exists)


def _psi4_normalize_parameterization_source_format(source_format, source_file=""):
    fmt = str(source_format or "").strip().lower().lstrip(".")
    if fmt == "mol":
        fmt = "mdl"
    if not fmt:
        ext = os.path.splitext(str(source_file or "").strip())[1].lower()
        fmt = {".sdf": "sdf", ".mol2": "mol2", ".mol": "mdl", ".pdb": "pdb"}.get(ext, "")
    return str(fmt)


def _psi4_remove_stale_file(path, protected_paths=None):
    p = str(path or "").strip()
    if not p:
        return
    protected = {os.path.abspath(str(x)) for x in list(protected_paths or []) if str(x or "").strip()}
    try:
        if os.path.abspath(p) in protected:
            return
    except Exception:
        return
    try:
        if os.path.isfile(p):
            os.remove(p)
    except Exception:
        pass


def psi4_prepare_pose_source_for_parameterization(
    hit,
    work_dir,
    dock_input_format_fn=None,
    convert_pdbqt_to_pdb_fn=None,
    pose_pdb_path=None,
):
    out = {"ok": False, "source_file": "", "source_fi": "", "notes": [], "error": ""}
    h = dict(hit or {})
    pose_src = str(h.get("pose_view_file", "") or h.get("pose_file", "")).strip()
    if (not pose_src) or (not os.path.isfile(pose_src)):
        out["error"] = "pose_source_missing"
        return out
    if not _psi4_path_is_nonempty(pose_src):
        out["error"] = "pose_source_empty"
        return out
    ext = os.path.splitext(pose_src)[1].lower()
    if ext == ".pdbqt":
        pose_pdb = str(pose_pdb_path or os.path.join(str(work_dir or "."), "ligand_for_param.pdb"))
        ok_c = False
        if callable(convert_pdbqt_to_pdb_fn):
            try:
                ok_c = bool(convert_pdbqt_to_pdb_fn(pose_src, pose_pdb))
            except Exception:
                ok_c = False
        if ok_c and _psi4_path_is_nonempty(pose_pdb):
            out.update(
                {
                    "ok": True,
                    "source_file": str(pose_pdb),
                    "source_fi": "pdb",
                    "notes": ["source=pose_pdbqt_to_pdb"],
                    "error": "",
                }
            )
            return out
        out["error"] = "pdbqt_conversion_failed"
        return out
    if ext in (".pdb", ".mol2", ".sdf", ".mol"):
        fi = ""
        if callable(dock_input_format_fn):
            try:
                fi = str(dock_input_format_fn(pose_src) or "").strip()
            except Exception:
                fi = ""
        if not fi:
            ext_map = {".pdb": "pdb", ".mol2": "mol2", ".sdf": "sdf", ".mol": "mdl"}
            fi = str(ext_map.get(ext, "") or "")
        if fi:
            out.update(
                {
                    "ok": True,
                    "source_file": str(pose_src),
                    "source_fi": str(fi),
                    "notes": [f"source=pose_file:{ext}"],
                    "error": "",
                }
            )
            return out
        out["error"] = "pose_format_unrecognized"
        return out
    out["error"] = "pose_extension_unsupported"
    return out


def psi4_choose_parameterization_source(rdkit_source_file="", pose_payload=None):
    out = {
        "ok": False,
        "source_file": "",
        "source_fi": "",
        "notes": [],
        "error": "could not build ligand source for per-hit parameterization",
    }
    rd = str(rdkit_source_file or "").strip()
    if rd and _psi4_path_is_nonempty(rd):
        out.update(
            {
                "ok": True,
                "source_file": str(rd),
                "source_fi": "sdf",
                "notes": ["source=rdkit_mol"],
                "error": "",
            }
        )
        return out
    pose = dict(pose_payload or {})
    if bool(pose.get("ok", False)):
        out.update(
            {
                "ok": True,
                "source_file": str(pose.get("source_file", "") or ""),
                "source_fi": str(pose.get("source_fi", "") or ""),
                "notes": [str(x) for x in list(pose.get("notes", []) or []) if str(x).strip()],
                "error": "",
            }
        )
        return out
    if str(pose.get("error", "") or "").strip():
        out["error_detail"] = str(pose.get("error", "") or "").strip()
    return out


def psi4_prepare_rdkit_source_for_parameterization(
    mol_src,
    out_sdf,
    clone_mol_fn=None,
    formal_charge_fn=None,
    atomic_z_sum_fn=None,
    has_conformer_fn=None,
    add_hs_fn=None,
    embed_fn=None,
    optimize_fn=None,
    write_sdf_fn=None,
):
    out = {
        "ok": False,
        "source_file": "",
        "formal_charge": None,
        "atomic_z_sum": None,
        "error": "",
        "notes": [],
    }
    if (not callable(clone_mol_fn)) or (not callable(write_sdf_fn)):
        out["error"] = "rdkit_callbacks_missing"
        return out
    out_path = str(out_sdf or "").strip()
    if not out_path:
        out["error"] = "rdkit_output_sdf_missing"
        return out
    try:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    except Exception:
        pass
    try:
        mol = clone_mol_fn(mol_src)
    except Exception as e:
        out["error"] = f"mol_clone_failed:{e}"
        return out
    if mol is None:
        out["error"] = "mol_clone_failed"
        return out
    if callable(formal_charge_fn):
        try:
            out["formal_charge"] = int(formal_charge_fn(mol))
        except Exception:
            out["formal_charge"] = None
    if callable(atomic_z_sum_fn):
        try:
            out["atomic_z_sum"] = int(atomic_z_sum_fn(mol))
        except Exception:
            out["atomic_z_sum"] = None
    need_conf = False
    if callable(has_conformer_fn):
        try:
            need_conf = not bool(has_conformer_fn(mol))
        except Exception:
            need_conf = False
    if need_conf:
        if callable(add_hs_fn):
            try:
                mol_h = add_hs_fn(mol)
                if mol_h is not None:
                    mol = mol_h
            except Exception:
                pass
        if callable(embed_fn):
            try:
                embed_fn(mol)
            except Exception as e:
                out["notes"].append(f"rdkit_embed_failed:{e}")
        if callable(optimize_fn):
            try:
                optimize_fn(mol)
            except Exception:
                pass
    try:
        wrote = write_sdf_fn(mol, out_path)
    except Exception as e:
        out["error"] = f"rdkit_sdf_write_failed:{e}"
        return out
    if isinstance(wrote, str) and str(wrote).strip():
        out_path = str(wrote).strip()
    if not _psi4_path_is_nonempty(out_path):
        out["error"] = "rdkit_sdf_not_written"
        return out
    out.update({"ok": True, "source_file": str(out_path), "error": ""})
    return out


def psi4_make_leadopt_parameterization_info():
    return {
        "ok": False,
        "mol2": "",
        "frcmod": "",
        "prmtop": "",
        "inpcrd": "",
        "source": "",
        "reason": "",
        "notes": [],
    }


def psi4_make_leadopt_cached_info(mol2, frcmod, prmtop, inpcrd):
    return {
        "ok": True,
        "mol2": str(mol2 or ""),
        "frcmod": str(frcmod or ""),
        "prmtop": str(prmtop or ""),
        "inpcrd": str(inpcrd or ""),
        "source": "cached",
    }


def psi4_make_leadopt_generated_info(mol2, frcmod, prmtop, inpcrd):
    return {
        "ok": True,
        "mol2": str(mol2 or ""),
        "frcmod": str(frcmod or ""),
        "prmtop": str(prmtop or ""),
        "inpcrd": str(inpcrd or ""),
        "source": "generated",
    }


def psi4_reason_missing_parameterization_tools(missing_tools):
    miss = [str(x).strip() for x in list(missing_tools or []) if str(x).strip()]
    if not miss:
        return ""
    return "ligand parameterization required for advanced refinement, but missing tools: " + ", ".join(miss)


def psi4_reason_source_for_parameterization_unavailable():
    return "could not build ligand source for per-hit parameterization"


def psi4_reason_tleap_parameterization_failed(message):
    return f"tleap failed for ligand parameterization: {message}"


def psi4_merge_notes(existing_notes, extra_notes):
    out = [str(x).strip() for x in list(existing_notes or []) if str(x).strip()]
    for note in list(extra_notes or []):
        txt = str(note).strip()
        if txt:
            out.append(txt)
    return out


def psi4_build_leadopt_param_hit_updates(mol2, frcmod, prmtop, inpcrd):
    return {
        "param_mol2": str(mol2 or ""),
        "param_frcmod": str(frcmod or ""),
        "param_prmtop": str(prmtop or ""),
        "param_inpcrd": str(inpcrd or ""),
    }


def psi4_apply_mapping_to_dict(target, mapping):
    if not isinstance(target, dict):
        return False
    data = dict(mapping or {})
    try:
        for k, v in data.items():
            target[k] = v
        return True
    except Exception:
        return False


def psi4_finalize_leadopt_parameterization_success(mol2, frcmod, prmtop, inpcrd):
    return {
        "info_update": psi4_make_leadopt_generated_info(mol2=mol2, frcmod=frcmod, prmtop=prmtop, inpcrd=inpcrd),
        "hit_updates": psi4_build_leadopt_param_hit_updates(mol2=mol2, frcmod=frcmod, prmtop=prmtop, inpcrd=inpcrd),
    }


def psi4_resolve_leadopt_cached_parameterization(
    hit,
    file_exists_fn=None,
    make_cached_info_fn=None,
):
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    make_cached_fn = make_cached_info_fn if callable(make_cached_info_fn) else psi4_make_leadopt_cached_info
    h = dict(hit or {})
    mol2_existing = str(h.get("param_mol2", "") or h.get("parameterized_mol2", "")).strip()
    frc_existing = str(h.get("param_frcmod", "") or h.get("parameterized_frcmod", "")).strip()
    prm_existing = str(h.get("param_prmtop", "") or h.get("parameterized_prmtop", "")).strip()
    crd_existing = str(h.get("param_inpcrd", "") or h.get("parameterized_inpcrd", "")).strip()
    out = {
        "ok": False,
        "mol2": "",
        "frcmod": "",
        "prmtop": "",
        "inpcrd": "",
        "info_update": {},
    }
    if not (mol2_existing and prm_existing):
        return out
    if (not _psi4_path_is_nonempty(mol2_existing, exists_fn=exists_fn)) or (
        not _psi4_path_is_nonempty(prm_existing, exists_fn=exists_fn)
    ):
        return out
    out.update(
        {
            "ok": True,
            "mol2": str(mol2_existing),
            "frcmod": str(
                frc_existing if (frc_existing and _psi4_path_is_nonempty(frc_existing, exists_fn=exists_fn)) else ""
            ),
            "prmtop": str(prm_existing),
            "inpcrd": str(
                crd_existing if (crd_existing and _psi4_path_is_nonempty(crd_existing, exists_fn=exists_fn)) else ""
            ),
        }
    )
    try:
        out["info_update"] = dict(
            make_cached_fn(
                mol2=out["mol2"],
                frcmod=out["frcmod"],
                prmtop=out["prmtop"],
                inpcrd=out["inpcrd"],
            )
            or {}
        )
    except Exception:
        out["info_update"] = psi4_make_leadopt_cached_info(
            mol2=out["mol2"],
            frcmod=out["frcmod"],
            prmtop=out["prmtop"],
            inpcrd=out["inpcrd"],
        )
    return out


def psi4_validate_leadopt_parameterization_tools(
    antechamber_bin="",
    parmchk2_bin="",
    tleap_bin="",
    reason_missing_tools_fn=None,
):
    reason_fn = reason_missing_tools_fn if callable(reason_missing_tools_fn) else psi4_reason_missing_parameterization_tools
    tools = {
        "antechamber": str(antechamber_bin or "").strip(),
        "parmchk2": str(parmchk2_bin or "").strip(),
        "tleap": str(tleap_bin or "").strip(),
    }
    missing = [nm for nm in ("antechamber", "parmchk2", "tleap") if not str(tools.get(nm, "")).strip()]
    out = {
        "ok": len(missing) == 0,
        "tools": dict(tools),
        "missing": list(missing),
        "reason": "",
    }
    if missing:
        try:
            out["reason"] = str(reason_fn(missing) or "")
        except Exception:
            out["reason"] = str(psi4_reason_missing_parameterization_tools(missing) or "")
    return out


def psi4_orchestrate_leadopt_parameterization(
    hit,
    work_dir,
    net_charge=0,
    charge_method="bcc",
    atom_type="gaff2",
    antechamber_bin="",
    parmchk2_bin="",
    tleap_bin="",
    prepare_paths_fn=None,
    resolve_cached_fn=None,
    validate_tools_fn=None,
    resolve_rdkit_state_fn=None,
    resolve_source_fn=None,
    build_charge_candidates_fn=None,
    run_pipeline_fn=None,
    finalize_success_fn=None,
    merge_notes_fn=None,
    make_info_fn=None,
    reason_source_unavailable_fn=None,
    apply_mapping_fn=None,
):
    make_info = make_info_fn if callable(make_info_fn) else psi4_make_leadopt_parameterization_info
    merge_fn = merge_notes_fn if callable(merge_notes_fn) else psi4_merge_notes
    reason_source_fn = (
        reason_source_unavailable_fn
        if callable(reason_source_unavailable_fn)
        else psi4_reason_source_for_parameterization_unavailable
    )
    prepare_paths = (
        prepare_paths_fn if callable(prepare_paths_fn) else psi4_prepare_leadopt_parameterization_paths
    )
    resolve_cached = (
        resolve_cached_fn if callable(resolve_cached_fn) else psi4_resolve_leadopt_cached_parameterization
    )
    validate_tools = (
        validate_tools_fn
        if callable(validate_tools_fn)
        else psi4_validate_leadopt_parameterization_tools
    )
    resolve_rdkit = (
        resolve_rdkit_state_fn if callable(resolve_rdkit_state_fn) else psi4_resolve_leadopt_rdkit_state
    )
    resolve_source = (
        resolve_source_fn
        if callable(resolve_source_fn)
        else psi4_resolve_leadopt_parameterization_source
    )
    build_candidates = (
        build_charge_candidates_fn
        if callable(build_charge_candidates_fn)
        else psi4_build_net_charge_candidates
    )
    run_pipeline = (
        run_pipeline_fn if callable(run_pipeline_fn) else psi4_run_leadopt_parameterization_pipeline
    )
    finalize_success = (
        finalize_success_fn
        if callable(finalize_success_fn)
        else psi4_finalize_leadopt_parameterization_success
    )
    apply_mapping = (
        apply_mapping_fn if callable(apply_mapping_fn) else psi4_apply_mapping_to_dict
    )
    out = {"ok": False, "info": {}}
    try:
        info = dict(make_info() or {})
    except Exception:
        info = psi4_make_leadopt_parameterization_info()
    if not isinstance(info.get("notes", None), list):
        info["notes"] = list(info.get("notes", []) or [])
    try:
        os.makedirs(str(work_dir), exist_ok=True)
    except Exception:
        pass
    cached_payload = dict(resolve_cached(hit=hit) or {})
    if bool(cached_payload.get("ok", False)):
        info.update(dict(cached_payload.get("info_update", {}) or {}))
        out["ok"] = True
        out["info"] = info
        return out
    tools_payload = validate_tools(
        antechamber_bin=antechamber_bin,
        parmchk2_bin=parmchk2_bin,
        tleap_bin=tleap_bin,
    )
    tools_payload = dict(tools_payload or {})
    if not bool(tools_payload.get("ok", False)):
        info["reason"] = str(tools_payload.get("reason", "") or "")
        out["info"] = info
        return out
    paths = dict(prepare_paths(work_dir) or {})
    rd_state = dict(resolve_rdkit(hit=hit, paths=paths) or {})
    rdkit_source_file = str(rd_state.get("source_file", "") or "")
    rdkit_formal_charge = rd_state.get("formal_charge", None)
    rdkit_atomic_z_sum = rd_state.get("atomic_z_sum", None)
    info["notes"] = merge_fn(
        info.get("notes", []),
        list(rd_state.get("notes", []) or []),
    )
    source_payload = dict(
        resolve_source(
            hit=hit,
            work_dir=work_dir,
            paths=paths,
            rdkit_source_file=rdkit_source_file,
        )
        or {}
    )
    info["notes"] = merge_fn(
        info.get("notes", []),
        list(source_payload.get("notes", []) or []),
    )
    source_file = str(source_payload.get("source_file", "") or "")
    source_fi = str(source_payload.get("source_fi", "") or "")
    if (not bool(source_payload.get("ok", False))) or (not source_file) or (not source_fi):
        info["reason"] = str(source_payload.get("reason", "") or reason_source_fn() or "")
        out["info"] = info
        return out
    out_mol2 = str(paths.get("out_mol2", "") or os.path.join(str(work_dir), "ligand_gaff2.mol2"))
    out_frcmod = str(paths.get("out_frcmod", "") or os.path.join(str(work_dir), "ligand_gaff2.frcmod"))
    try:
        charge_candidates = list(
            build_candidates(
                requested_charge=net_charge,
                formal_charge=rdkit_formal_charge,
                atomic_z_sum=rdkit_atomic_z_sum,
                clip_min=-8,
                clip_max=8,
            )
            or []
        )
    except Exception:
        charge_candidates = [int(net_charge or 0)]
    run_payload = dict(
        run_pipeline(
            antechamber_bin=antechamber_bin,
            parmchk2_bin=parmchk2_bin,
            source_file=str(source_file),
            source_format=str(source_fi),
            out_mol2=str(out_mol2),
            out_frcmod=str(out_frcmod),
            work_dir=str(work_dir),
            charge_candidates=charge_candidates,
            charge_method=str(charge_method or "bcc"),
            atom_type=str(atom_type or "gaff2"),
        )
        or {}
    )
    info["notes"] = merge_fn(
        info.get("notes", []),
        list(run_payload.get("notes", []) or []),
    )
    if not bool(run_payload.get("ok", False)):
        info["reason"] = str(run_payload.get("reason", "") or "ligand parameterization pipeline failed")
        out["info"] = info
        return out
    out_mol2 = str(run_payload.get("out_mol2", "") or out_mol2)
    out_frcmod = str(run_payload.get("out_frcmod", "") or out_frcmod)
    prm_out = str(run_payload.get("prmtop", "") or "")
    crd_out = str(run_payload.get("inpcrd", "") or "")
    final_payload = dict(
        finalize_success(
            mol2=str(out_mol2),
            frcmod=str(out_frcmod),
            prmtop=str(prm_out),
            inpcrd=str(crd_out),
        )
        or {}
    )
    info.update(dict(final_payload.get("info_update", {}) or {}))
    apply_mapping(hit, final_payload.get("hit_updates", {}))
    out["ok"] = True
    out["info"] = info
    return out


def psi4_normalize_rdkit_parameterization_payload(rd_payload):
    p = dict(rd_payload or {})
    out = {
        "source_file": "",
        "formal_charge": None,
        "atomic_z_sum": None,
        "notes": [],
    }
    src = str(p.get("source_file", "") or "").strip()
    if bool(p.get("ok", False)) and src and _psi4_path_is_nonempty(src):
        out["source_file"] = str(src)
    fc = p.get("formal_charge", None)
    if fc is not None:
        try:
            out["formal_charge"] = int(fc)
            out["notes"].append(f"rdkit_formal_charge={int(fc)}")
        except Exception:
            out["formal_charge"] = None
    az = p.get("atomic_z_sum", None)
    if az is not None:
        try:
            out["atomic_z_sum"] = int(az)
        except Exception:
            out["atomic_z_sum"] = None
    err = str(p.get("error", "") or "").strip()
    if err:
        out["notes"].append(f"rdkit_source_failed:{err}")
    out["notes"] = psi4_merge_notes(out.get("notes", []), list(p.get("notes", []) or []))
    return out


def psi4_resolve_leadopt_rdkit_state(
    hit,
    paths,
    has_rdkit=False,
    prepare_rdkit_source_fn=None,
    normalize_rdkit_payload_fn=None,
):
    out = {
        "source_file": "",
        "formal_charge": None,
        "atomic_z_sum": None,
        "notes": [],
    }
    if not bool(has_rdkit):
        return out
    h = dict(hit or {})
    mol_src = h.get("mol", None)
    if mol_src is None:
        return out
    p = dict(paths or {})
    src_sdf = str(p.get("rdkit_sdf", "") or "ligand_for_param.sdf")
    prep_fn = prepare_rdkit_source_fn if callable(prepare_rdkit_source_fn) else None
    if prep_fn is None:
        return out
    try:
        rd_payload = prep_fn(mol_src, src_sdf)
    except Exception as e:
        rd_payload = {"ok": False, "error": str(e)}
    norm_fn = normalize_rdkit_payload_fn if callable(normalize_rdkit_payload_fn) else psi4_normalize_rdkit_parameterization_payload
    try:
        rd_state = dict(norm_fn(rd_payload) or {})
    except Exception:
        rd_state = {}
    out["source_file"] = str(rd_state.get("source_file", "") or "")
    fc = rd_state.get("formal_charge", None)
    if fc is not None:
        try:
            out["formal_charge"] = int(fc)
        except Exception:
            out["formal_charge"] = None
    az = rd_state.get("atomic_z_sum", None)
    if az is not None:
        try:
            out["atomic_z_sum"] = int(az)
        except Exception:
            out["atomic_z_sum"] = None
    out["notes"] = psi4_merge_notes(out.get("notes", []), list(rd_state.get("notes", []) or []))
    return out


def psi4_resolve_leadopt_parameterization_source(
    hit,
    work_dir,
    paths,
    rdkit_source_file="",
    prepare_pose_source_fn=None,
    choose_source_fn=None,
    reason_source_unavailable_fn=None,
    merge_notes_fn=None,
):
    out = {
        "ok": False,
        "source_file": "",
        "source_fi": "",
        "notes": [],
        "reason": "",
    }
    p = dict(paths or {})
    pose_pdb_path = str(p.get("pose_pdb", "") or os.path.join(str(work_dir or "."), "ligand_for_param.pdb"))
    prep_fn = prepare_pose_source_fn if callable(prepare_pose_source_fn) else psi4_prepare_pose_source_for_parameterization
    choose_fn = choose_source_fn if callable(choose_source_fn) else psi4_choose_parameterization_source
    reason_fn = (
        reason_source_unavailable_fn
        if callable(reason_source_unavailable_fn)
        else psi4_reason_source_for_parameterization_unavailable
    )
    merge_fn = merge_notes_fn if callable(merge_notes_fn) else psi4_merge_notes
    try:
        pose_payload = prep_fn(hit=hit, work_dir=work_dir, pose_pdb_path=pose_pdb_path)
    except Exception:
        pose_payload = {"ok": False, "error": "pose_source_prepare_failed"}
    try:
        source_payload = choose_fn(rdkit_source_file=rdkit_source_file, pose_payload=pose_payload)
    except Exception:
        source_payload = {"ok": False}
    out["notes"] = merge_fn(out.get("notes", []), list(source_payload.get("notes", []) or []))
    out["source_file"] = str(source_payload.get("source_file", "") or "")
    out["source_fi"] = str(source_payload.get("source_fi", "") or "")
    out["ok"] = bool(source_payload.get("ok", False)) and bool(out["source_file"]) and bool(out["source_fi"])
    if not out["ok"]:
        try:
            out["reason"] = str(reason_fn() or "")
        except Exception:
            out["reason"] = str(psi4_reason_source_for_parameterization_unavailable() or "")
    return out


def psi4_run_leadopt_parameterization_pipeline(
    antechamber_bin,
    parmchk2_bin,
    source_file,
    source_format,
    out_mol2,
    out_frcmod,
    work_dir,
    charge_candidates,
    charge_method="bcc",
    atom_type="gaff2",
    run_command_fn=None,
    build_ligand_amber_fn=None,
    read_text_fn=None,
    file_exists_fn=None,
    detect_sqm_hint_fn=None,
    format_attempt_brief_fn=None,
    format_attempts_failure_fn=None,
    format_tool_failure_fn=None,
    reason_tleap_failure_fn=None,
    require_amber_files=True,
):
    src = str(source_file or "").strip()
    work = str(work_dir or "").strip() or "."
    out_mol2 = str(out_mol2 or "").strip()
    out_frcmod = str(out_frcmod or "").strip()
    source_format = _psi4_normalize_parameterization_source_format(source_format, source_file=src)
    charge_method = str(charge_method or "bcc").strip().lower() or "bcc"
    atom_type = str(atom_type or "gaff2").strip().lower() or "gaff2"
    payload = {
        "ok": False,
        "stage": "",
        "reason": "",
        "notes": [],
        "chosen_nc": None,
        "out_mol2": str(out_mol2),
        "out_frcmod": str(out_frcmod),
        "prmtop": "",
        "inpcrd": "",
        "attempt_brief": [],
    }
    if not callable(run_command_fn):
        payload["stage"] = "setup"
        payload["reason"] = "run_command_fn_missing"
        return payload
    if not callable(build_ligand_amber_fn):
        payload["stage"] = "setup"
        payload["reason"] = "build_ligand_amber_fn_missing"
        return payload
    exists_fn = file_exists_fn if callable(file_exists_fn) else os.path.isfile
    if not _psi4_path_is_nonempty(src, exists_fn=exists_fn):
        payload["stage"] = "setup"
        payload["reason"] = "source_file_missing_or_empty"
        return payload
    if source_format not in ("sdf", "mol2", "mdl", "pdb"):
        payload["stage"] = "setup"
        payload["reason"] = f"unsupported_source_format:{source_format or 'unknown'}"
        return payload
    if (not out_mol2) or (not out_frcmod):
        payload["stage"] = "setup"
        payload["reason"] = "output_paths_missing"
        return payload
    try:
        os.makedirs(work, exist_ok=True)
        os.makedirs(os.path.dirname(out_mol2) or work, exist_ok=True)
        os.makedirs(os.path.dirname(out_frcmod) or work, exist_ok=True)
    except Exception:
        pass
    detect_fn = detect_sqm_hint_fn if callable(detect_sqm_hint_fn) else psi4_detect_sqm_charge_mismatch_hint
    brief_fn = format_attempt_brief_fn if callable(format_attempt_brief_fn) else psi4_format_antechamber_attempt_brief
    attempts_fail_fn = (
        format_attempts_failure_fn if callable(format_attempts_failure_fn) else psi4_format_antechamber_attempts_failure
    )
    tool_fail_fn = format_tool_failure_fn if callable(format_tool_failure_fn) else psi4_format_tool_rc_failure
    tleap_reason_fn = (
        reason_tleap_failure_fn if callable(reason_tleap_failure_fn) else psi4_reason_tleap_parameterization_failed
    )

    def _read_text(path):
        p = str(path or "").strip()
        if not p:
            return ""
        if callable(read_text_fn):
            try:
                return str(read_text_fn(p) or "")
            except Exception:
                return ""
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                return str(fh.read() or "")
        except Exception:
            return ""

    cands = []
    for v in list(charge_candidates or []):
        try:
            vv = int(np.clip(int(v), -8, 8))
        except Exception:
            continue
        if vv not in cands:
            cands.append(vv)
    if not cands:
        cands = [0]

    ok1 = False
    out1 = ""
    err1 = ""
    rc1 = 1
    chosen_nc = None
    attempt_brief = []
    for nc_try in cands:
        # Each charge hypothesis must prove itself; stale output from a previous
        # antechamber run would otherwise be indistinguishable from a fresh MOL2.
        _psi4_remove_stale_file(out_mol2, protected_paths=[src])
        _psi4_remove_stale_file(os.path.join(work, "sqm.out"))
        cmd1 = psi4_build_antechamber_cmd(
            antechamber_bin=antechamber_bin,
            source_file=src,
            source_format=source_format,
            out_mol2=out_mol2,
            charge_method=charge_method,
            net_charge=int(nc_try),
            atom_type=atom_type,
            dr="n",
        )
        ok1, out1, err1, rc1 = run_command_fn(cmd1, cwd=str(work), timeout=2400)
        if ok1 and _psi4_path_is_nonempty(out_mol2, exists_fn=exists_fn):
            chosen_nc = int(nc_try)
            break
        sqm_out = os.path.join(str(work or "."), "sqm.out")
        sqm_hint = ""
        if _psi4_path_is_nonempty(sqm_out, exists_fn=exists_fn):
            sqm_hint = str(detect_fn(_read_text(sqm_out)) or "")
        attempt_brief.append(brief_fn(net_charge=nc_try, rc=rc1, sqm_hint=sqm_hint))
    payload["attempt_brief"] = [str(x) for x in list(attempt_brief or [])]
    if (not ok1) or (not _psi4_path_is_nonempty(out_mol2, exists_fn=exists_fn)):
        payload["stage"] = "antechamber"
        payload["reason"] = attempts_fail_fn(
            attempt_brief=attempt_brief,
            stderr=err1,
            stdout=out1,
            stderr_tail=400,
            stdout_tail=300,
        )
        return payload
    if chosen_nc is not None:
        payload["chosen_nc"] = int(chosen_nc)
        payload["notes"] = [f"antechamber_nc={int(chosen_nc)}"]

    _psi4_remove_stale_file(out_frcmod, protected_paths=[src, out_mol2])
    cmd2 = psi4_build_parmchk2_cmd(parmchk2_bin=parmchk2_bin, in_mol2=out_mol2, out_frcmod=out_frcmod)
    ok2, out2, err2, rc2 = run_command_fn(cmd2, cwd=str(work), timeout=1200)
    if (not ok2) or (not _psi4_path_is_nonempty(out_frcmod, exists_fn=exists_fn)):
        payload["stage"] = "parmchk2"
        payload["reason"] = tool_fail_fn(
            tool_name="parmchk2",
            rc=rc2,
            stderr=err2,
            stdout=out2,
            stderr_tail=400,
            stdout_tail=300,
        )
        return payload

    ok_ff, msg_ff, prm_out, crd_out = build_ligand_amber_fn(
        str(out_mol2),
        str(out_frcmod),
        atom_type=str(atom_type or "gaff2"),
        out_dir=str(work),
    )
    if not bool(ok_ff):
        if not bool(require_amber_files):
            note = str(tleap_reason_fn(msg_ff) or "").strip()
            if note:
                payload["notes"].append(f"tleap_warning:{note[:220]}")
            payload.update({"ok": True, "stage": "", "reason": ""})
            return payload
        payload["stage"] = "tleap"
        payload["reason"] = str(tleap_reason_fn(msg_ff) or "")
        return payload
    if (
        bool(require_amber_files)
        and (
            not _psi4_path_is_nonempty(prm_out, exists_fn=exists_fn)
            or not _psi4_path_is_nonempty(crd_out, exists_fn=exists_fn)
        )
    ):
        payload["stage"] = "tleap"
        payload["reason"] = str(tleap_reason_fn("Amber prmtop/inpcrd were not written.") or "")
        return payload
    payload.update(
        {
            "ok": True,
            "stage": "",
            "reason": "",
            "prmtop": str(prm_out or ""),
            "inpcrd": str(crd_out or ""),
        }
    )
    return payload


def psi4_map_ligand_charges_from_param_mol2(ligand_ids, atom_names, elems, records, charges_by_name=None):
    lig_idx = [int(i) for i in list(ligand_ids or [])]
    info = {
        "mode": "none",
        "mapped": 0,
        "missing": int(len(lig_idx)),
        "total": int(len(lig_idx)),
    }
    if len(lig_idx) <= 0:
        return {}, info
    recs = list(records or [])
    by_name = dict(charges_by_name or {})
    if recs:
        buckets = collections.defaultdict(list)
        for r in recs:
            nm = str(r.get("name", "")).strip().upper()
            if nm:
                buckets[nm].append(float(r.get("charge", 0.0)))
        m = {}
        missing_names = []
        for i in lig_idx:
            nm_u = str(atom_names[int(i)]).strip().upper()
            vals = buckets.get(nm_u, [])
            if vals:
                m[int(i)] = float(vals.pop(0))
            else:
                missing_names.append(nm_u)
        if len(m) == len(lig_idx):
            info.update({"mode": "name_bucket", "mapped": int(len(m)), "missing": 0})
            return m, info
    if recs and (len(recs) == len(lig_idx)):
        lig_elems = [str(elems[int(i)]).strip().upper() for i in lig_idx]
        rec_elems = [str(r.get("element", "")).strip().upper() for r in recs]
        if lig_elems == rec_elems:
            m = {int(lig_idx[k]): float(recs[k].get("charge", 0.0)) for k in range(len(lig_idx))}
            info.update({"mode": "index_element_exact", "mapped": int(len(m)), "missing": 0})
            return m, info
        if collections.Counter(lig_elems) == collections.Counter(rec_elems):
            pools = collections.defaultdict(collections.deque)
            for r in recs:
                el = str(r.get("element", "")).strip().upper()
                pools[el].append(float(r.get("charge", 0.0)))
            m = {}
            for li, el in zip(lig_idx, lig_elems):
                if pools[el]:
                    m[int(li)] = float(pools[el].popleft())
            if len(m) == len(lig_idx):
                info.update({"mode": "element_bucket", "mapped": int(len(m)), "missing": 0})
                return m, info
    if by_name:
        m = {}
        missing_names = []
        for i in lig_idx:
            nm_u = str(atom_names[int(i)]).strip().upper()
            if nm_u in by_name:
                m[int(i)] = float(by_name.get(nm_u, 0.0))
            else:
                missing_names.append(nm_u)
        if m:
            info.update(
                {
                    "mode": "name_dict_partial" if len(m) < len(lig_idx) else "name_dict_exact",
                    "mapped": int(len(m)),
                    "missing": int(len(lig_idx) - len(m)),
                    "missing_names": missing_names[:12],
                }
            )
            return m, info
    info.update({"mode": "no_mapping", "mapped": 0, "missing": int(len(lig_idx))})
    if recs:
        info["mol2_atom_count"] = int(len(recs))
    return {}, info


def psi4_map_charge_transfer_between_mol2_records(src_records, dst_records):
    src_recs = list(src_records or [])
    dst_recs = list(dst_records or [])
    if (not src_recs) or (not dst_recs):
        return {
            "ok": False,
            "error": "mol2_atom_parse_failed",
            "src_atoms": int(len(src_recs)),
            "dst_atoms": int(len(dst_recs)),
        }
    n_src = int(len(src_recs))
    n_dst = int(len(dst_recs))
    src_charges = [float(r.get("charge", 0.0)) for r in src_recs]
    src_names = [str(r.get("name", "") or "").strip().upper() for r in src_recs]
    dst_names = [str(r.get("name", "") or "").strip().upper() for r in dst_recs]
    src_elems = [str(r.get("element", "") or "").strip().upper() for r in src_recs]
    dst_elems = [str(r.get("element", "") or "").strip().upper() for r in dst_recs]
    mapped = None
    mode = ""
    buckets = collections.defaultdict(collections.deque)
    for rec in src_recs:
        nm = str(rec.get("name", "") or "").strip().upper()
        if nm:
            buckets[nm].append(float(rec.get("charge", 0.0)))
    tmp = []
    ok_name = True
    for nm in dst_names:
        if nm and buckets.get(nm):
            tmp.append(float(buckets[nm].popleft()))
        else:
            ok_name = False
            break
    if ok_name and (len(tmp) == n_dst):
        mapped = list(tmp)
        mode = "name_bucket"
    if (mapped is None) and (n_src == n_dst):
        if src_names == dst_names:
            mapped = list(src_charges)
            mode = "name_order"
        elif src_elems == dst_elems:
            mapped = list(src_charges)
            mode = "element_order"
        elif collections.Counter(src_elems) == collections.Counter(dst_elems):
            pools = collections.defaultdict(collections.deque)
            for rec in src_recs:
                el = str(rec.get("element", "") or "").strip().upper() or "C"
                pools[el].append(float(rec.get("charge", 0.0)))
            tmp = []
            ok_el = True
            for el in dst_elems:
                ee = str(el or "").strip().upper() or "C"
                if pools.get(ee):
                    tmp.append(float(pools[ee].popleft()))
                else:
                    ok_el = False
                    break
            if ok_el and (len(tmp) == n_dst):
                mapped = list(tmp)
                mode = "element_bucket"
    if (mapped is None) or (len(mapped) != n_dst):
        return {
            "ok": False,
            "error": "charge_mapping_failed",
            "src_atoms": int(n_src),
            "dst_atoms": int(n_dst),
            "src_name_sample": ",".join(src_names[:8]),
            "dst_name_sample": ",".join(dst_names[:8]),
            "src_elem_sample": ",".join(src_elems[:8]),
            "dst_elem_sample": ",".join(dst_elems[:8]),
        }
    return {
        "ok": True,
        "error": "",
        "mode": str(mode or ""),
        "mapped": [float(v) for v in mapped],
        "src_atoms": int(n_src),
        "dst_atoms": int(n_dst),
        "q_src": float(np.sum(np.asarray(src_charges, dtype=float))),
    }


def psi4_write_charged_mol2(in_mol2, charges, out_mol2):
    q = [float(v) for v in (list(charges) if charges is not None else [])]
    if (not in_mol2) or (not os.path.isfile(in_mol2)):
        return False, "Input MOL2 not found."
    if len(q) <= 0:
        return False, "No fitted charges were provided."
    section = None
    out_lines = []
    q_idx = 0
    try:
        with open(in_mol2, "r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = str(raw).rstrip("\n")
                if line.startswith("@<TRIPOS>"):
                    section = line.strip().upper()
                    out_lines.append(line)
                    continue
                if section == "@<TRIPOS>ATOM" and line.strip():
                    parts = line.split()
                    if len(parts) >= 9 and q_idx < len(q):
                        try:
                            atom_id = int(parts[0])
                            atom_name = str(parts[1])
                            x = float(parts[2])
                            y = float(parts[3])
                            z = float(parts[4])
                            atom_type = str(parts[5])
                            sub_id = int(float(parts[6]))
                            sub_name = str(parts[7])
                            qv = float(q[q_idx])
                            tail = ""
                            if len(parts) > 9:
                                tail = " " + " ".join(parts[9:])
                            newline = (
                                f"{atom_id:7d} {atom_name:<8} {x:10.4f} {y:10.4f} {z:10.4f} "
                                f"{atom_type:<8} {sub_id:4d} {sub_name:<8} {qv:10.6f}{tail}"
                            )
                            out_lines.append(newline)
                            q_idx += 1
                            continue
                        except Exception:
                            pass
                out_lines.append(line)
    except Exception as e:
        return False, f"Failed reading MOL2: {e}"
    if q_idx != len(q):
        return False, f"Charge count mismatch: fitted={len(q)} mapped={q_idx}"
    try:
        with open(out_mol2, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out_lines) + "\n")
    except Exception as e:
        return False, f"Failed writing charged MOL2: {e}"
    return True, ""


def psi4_transfer_mol2_charges_to_typed(
    src_charged_mol2,
    typed_mol2,
    out_mol2,
    load_records_fn=None,
    write_charged_mol2_fn=None,
    collect_charges_fn=None,
):
    src = str(src_charged_mol2 or "").strip()
    typed = str(typed_mol2 or "").strip()
    outp = str(out_mol2 or "").strip()
    if (not src) or (not os.path.isfile(src)):
        return False, {"error": "source_charged_mol2_missing"}
    if (not typed) or (not os.path.isfile(typed)):
        return False, {"error": "typed_mol2_missing"}
    if not outp:
        return False, {"error": "output_mol2_missing"}
    load_fn = load_records_fn if callable(load_records_fn) else psi4_load_mol2_atom_records
    write_fn = write_charged_mol2_fn if callable(write_charged_mol2_fn) else psi4_write_charged_mol2
    collect_fn = collect_charges_fn if callable(collect_charges_fn) else psi4_collect_ligand_charges_from_file
    src_records = list(load_fn(src) or [])
    dst_records = list(load_fn(typed) or [])
    map_payload = psi4_map_charge_transfer_between_mol2_records(src_records, dst_records)
    if not bool(map_payload.get("ok", False)):
        err = dict(map_payload or {})
        err.pop("ok", None)
        return False, err
    mapped = [float(v) for v in list(map_payload.get("mapped", []) or [])]
    mode = str(map_payload.get("mode", "") or "")
    n_src = int(map_payload.get("src_atoms", len(src_records)))
    n_dst = int(map_payload.get("dst_atoms", len(dst_records)))
    ok_w, msg_w = write_fn(typed, mapped, outp)
    if not ok_w:
        return False, {"error": "write_charged_mol2_failed", "detail": str(msg_w or "")}
    q_src = float(map_payload.get("q_src", np.sum(np.asarray(mapped, dtype=float))))
    try:
        q_out = float(np.sum(np.asarray(list(collect_fn(outp) or []), dtype=float)))
    except Exception:
        q_out = np.nan
    return True, {
        "mode": str(mode or ""),
        "src_atoms": int(n_src),
        "dst_atoms": int(n_dst),
        "q_src": float(q_src),
        "q_out": float(q_out) if np.isfinite(q_out) else np.nan,
    }


def psi4_infer_charge_multiplicity(meta, atoms, current_charge=0, multiplicity_fn=None):
    m = dict(meta or {})
    reasons = []
    charge_guess = None
    fc = m.get("formal_charge")
    if fc is not None:
        try:
            charge_guess = int(fc)
            reasons.append("formal_charge")
        except Exception:
            charge_guess = None
    if charge_guess is None:
        charge_guess = int(current_charge or 0)
        reasons.append("current_charge")
    radical_e = int(m.get("radical_electrons", 0) or 0)
    if radical_e > 0:
        mult_guess = int(max(2, min(10, radical_e + 1)))
        reasons.append("radical_electrons")
    else:
        mfn = multiplicity_fn if callable(multiplicity_fn) else psi4_guess_multiplicity_from_atoms
        try:
            mult_guess = int(max(1, min(10, int(mfn(list(atoms or []), charge=int(charge_guess))))))
        except Exception:
            mult_guess = 1
        reasons.append("electron_parity")
    return {
        "charge": int(charge_guess),
        "multiplicity": int(mult_guess),
        "reason": ",".join(reasons),
    }


def psi4_read_cube(cube_path):
    p = str(cube_path or "").strip()
    if (not p) or (not os.path.isfile(p)):
        raise FileNotFoundError(f"Cube file not found: {p}")
    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    if len(lines) < 8:
        raise RuntimeError("Cube file is too short.")
    c1 = lines[0]
    c2 = lines[1]

    def _parse_row(idx):
        toks = str(lines[idx]).split()
        if len(toks) < 4:
            raise RuntimeError(f"Invalid cube header line {idx+1}")
        return toks

    t2 = _parse_row(2)
    nat_raw = int(float(t2[0]))
    nat = abs(int(nat_raw))
    origin = np.array([float(t2[1]), float(t2[2]), float(t2[3])], dtype=float)
    tx = _parse_row(3)
    ty = _parse_row(4)
    tz = _parse_row(5)
    nx_raw = int(float(tx[0]))
    ny_raw = int(float(ty[0]))
    nz_raw = int(float(tz[0]))
    nx = abs(int(nx_raw))
    ny = abs(int(ny_raw))
    nz = abs(int(nz_raw))
    xvec = np.array([float(tx[1]), float(tx[2]), float(tx[3])], dtype=float)
    yvec = np.array([float(ty[1]), float(ty[2]), float(ty[3])], dtype=float)
    zvec = np.array([float(tz[1]), float(tz[2]), float(tz[3])], dtype=float)
    scale_to_bohr = 1.0
    if (nx_raw < 0) or (ny_raw < 0) or (nz_raw < 0):
        scale_to_bohr = 1.8897259886
    atoms = []
    idx = 6
    for _ in range(nat):
        toks = str(lines[idx]).split()
        if len(toks) < 5:
            raise RuntimeError(f"Invalid atom line in cube at {idx+1}")
        anum = int(float(toks[0]))
        achg = float(toks[1])
        axyz = np.array([float(toks[2]), float(toks[3]), float(toks[4])], dtype=float)
        atoms.append(
            {
                "atomic_number": anum,
                "nuclear_charge": achg,
                "coord": axyz,
                "coord_bohr": axyz * scale_to_bohr,
            }
        )
        idx += 1
    payload = np.fromstring(" ".join(lines[idx:]), sep=" ")
    ngrid = int(nx * ny * nz)
    if payload.size < ngrid:
        raise RuntimeError(f"Cube grid truncated: got={payload.size}, expected={ngrid}")
    data = np.asarray(payload[:ngrid], dtype=float).reshape((nx, ny, nz), order="C")
    return {
        "comments": [c1, c2],
        "nat_raw": nat_raw,
        "nat": nat,
        "nx_raw": nx_raw,
        "ny_raw": ny_raw,
        "nz_raw": nz_raw,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "origin": origin,
        "origin_bohr": origin * scale_to_bohr,
        "xvec": xvec,
        "yvec": yvec,
        "zvec": zvec,
        "xvec_bohr": xvec * scale_to_bohr,
        "yvec_bohr": yvec * scale_to_bohr,
        "zvec_bohr": zvec * scale_to_bohr,
        "atoms": atoms,
        "data": data,
        "header_lines": lines[:idx],
    }


def psi4_write_cube_data(cube_meta, data, out_cube):
    arr = np.asarray(data, dtype=float)
    nx = int(cube_meta.get("nx", 0))
    ny = int(cube_meta.get("ny", 0))
    nz = int(cube_meta.get("nz", 0))
    if arr.shape != (nx, ny, nz):
        raise RuntimeError(f"Cube data shape mismatch: got={arr.shape}, expected={(nx, ny, nz)}")
    hdr = list(cube_meta.get("header_lines", []))
    if len(hdr) < (6 + int(cube_meta.get("nat", 0))):
        raise RuntimeError("Cube header is incomplete.")
    with open(out_cube, "w", encoding="utf-8") as fh:
        for ln in hdr:
            fh.write(str(ln).rstrip("\n") + "\n")
        flat = arr.reshape(-1, order="C")
        for i in range(0, int(flat.size), 6):
            chunk = flat[i : i + 6]
            fh.write(" ".join(f"{float(v):13.5e}" for v in chunk) + "\n")


def psi4_build_mulliken_esp_cube_from_density(density_cube, charges, out_cube):
    try:
        meta = psi4_read_cube(density_cube)
    except Exception as e:
        return False, f"Failed reading density cube: {e}"
    q = np.asarray(list(charges or []), dtype=float).reshape(-1)
    atoms = list(meta.get("atoms", []) or [])
    nat = int(meta.get("nat", 0))
    if nat <= 0:
        return False, "Density cube has zero atoms."
    if q.size != nat:
        return False, f"Charge count mismatch for ESP cube: charges={int(q.size)} atoms={nat}"
    origin = np.asarray(meta.get("origin_bohr"), dtype=float)
    xv = np.asarray(meta.get("xvec_bohr"), dtype=float)
    yv = np.asarray(meta.get("yvec_bohr"), dtype=float)
    zv = np.asarray(meta.get("zvec_bohr"), dtype=float)
    nx = int(meta.get("nx", 0))
    ny = int(meta.get("ny", 0))
    nz = int(meta.get("nz", 0))
    if min(nx, ny, nz) <= 0:
        return False, "Invalid density grid dimensions."
    atom_xyz = np.asarray([np.asarray(a.get("coord_bohr"), dtype=float) for a in atoms], dtype=float)
    if atom_xyz.shape != (nat, 3):
        return False, "Invalid atom coordinates in cube header."
    esp = np.zeros((nx, ny, nz), dtype=float)
    iy = np.arange(ny, dtype=float)
    iz = np.arange(nz, dtype=float)
    chunk = 24000
    for ix in range(nx):
        base = origin + float(ix) * xv
        pts = (
            base[None, None, :]
            + iy[:, None, None] * yv[None, None, :]
            + iz[None, :, None] * zv[None, None, :]
        ).reshape(-1, 3)
        vals = np.zeros((pts.shape[0],), dtype=float)
        for st in range(0, int(pts.shape[0]), int(chunk)):
            en = min(int(pts.shape[0]), st + int(chunk))
            sub = pts[st:en, :]
            diff = sub[:, None, :] - atom_xyz[None, :, :]
            rij = np.linalg.norm(diff, axis=2)
            rij = np.clip(rij, 1.0e-8, None)
            vals[st:en] = np.sum(q[None, :] / rij, axis=1)
        esp[ix, :, :] = vals.reshape((ny, nz), order="C")
    try:
        psi4_write_cube_data(meta, esp, out_cube)
    except Exception as e:
        return False, f"Failed writing ESP cube: {e}"
    return True, ""
