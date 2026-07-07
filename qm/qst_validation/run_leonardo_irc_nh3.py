import json
import os
import subprocess
import sys
import tempfile


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import code_v2  # noqa: E402
from psi4_runtime import psi4_collect_atoms_from_xyz  # noqa: E402
from psi4_runtime import psi4_element_token  # noqa: E402
from psi4_runtime import psi4_qst_build_guess_atoms  # noqa: E402


class _DialogShim:
    pass


for _name in (
    "_psi4_pcm_force_c1",
    "_psi4_pcm_solvent_candidates",
    "_psi4_molecule_block",
    "_psi4_qst_script_payload",
    "_psi4_make_script",
):
    setattr(_DialogShim, _name, getattr(code_v2.ProtLigInteractDialog, _name))


def _make_dialog():
    dlg = _DialogShim()
    dlg._psi4_element_token = lambda element: psi4_element_token(element)
    return dlg


def _nh3_endpoints():
    react = [
        {"index": 1, "name": "N", "element": "N", "x": 0.0, "y": 0.0, "z": 0.10},
        {"index": 2, "name": "H1", "element": "H", "x": 0.0, "y": 0.94, "z": -0.27},
        {"index": 3, "name": "H2", "element": "H", "x": 0.8138, "y": -0.47, "z": -0.27},
        {"index": 4, "name": "H3", "element": "H", "x": -0.8138, "y": -0.47, "z": -0.27},
    ]
    prod = [
        {"index": 1, "name": "N", "element": "N", "x": 0.0, "y": 0.0, "z": -0.10},
        {"index": 2, "name": "H1", "element": "H", "x": 0.0, "y": 0.94, "z": 0.27},
        {"index": 3, "name": "H2", "element": "H", "x": 0.8138, "y": -0.47, "z": 0.27},
        {"index": 4, "name": "H3", "element": "H", "x": -0.8138, "y": -0.47, "z": 0.27},
    ]
    ts_guess = [
        {"index": 1, "name": "N", "element": "N", "x": 0.0, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "H1", "element": "H", "x": 0.0, "y": 0.94, "z": 0.0},
        {"index": 3, "name": "H2", "element": "H", "x": 0.8138, "y": -0.47, "z": 0.0},
        {"index": 4, "name": "H3", "element": "H", "x": -0.8138, "y": -0.47, "z": 0.0},
    ]
    return react, prod, ts_guess


def _run_script(job_dir, script):
    os.makedirs(job_dir, exist_ok=True)
    script_path = os.path.join(job_dir, "psi4_input.py")
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(script)
    proc = subprocess.run([sys.executable, script_path], cwd=job_dir, text=True, capture_output=True, timeout=420)
    result_path = os.path.join(job_dir, "psi4_result.json")
    result = {}
    if os.path.isfile(result_path):
        with open(result_path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
    return proc, result


def _ts_job(dlg, qst):
    return {
        "operation": "ts",
        "method": "hf",
        "basis": "sto-3g",
        "molecule_block": dlg._psi4_molecule_block(qst["guess_atoms"], charge=0, multiplicity=1),
        "memory_gb": 2,
        "threads": 1,
        "geom_maxiter": 40,
        "implicit": False,
        "scf_type": "pk",
        "charge": 0,
        "multiplicity": 1,
        "qst": dlg._psi4_qst_script_payload(qst["payload"], charge=0, multiplicity=1),
    }


def _irc_job(dlg, ts_atoms):
    return {
        "operation": "irc",
        "method": "hf",
        "basis": "sto-3g",
        "molecule_block": dlg._psi4_molecule_block(ts_atoms, charge=0, multiplicity=1),
        "memory_gb": 2,
        "threads": 1,
        "geom_maxiter": 80,
        "implicit": False,
        "scf_type": "pk",
        "charge": 0,
        "multiplicity": 1,
        "irc_direction": "BOTH",
        "irc_points": 6,
        "irc_step": 0.10,
        "irc_require_valid_ts": True,
    }


def main():
    dlg = _make_dialog()
    react, prod, ts_guess = _nh3_endpoints()
    root = tempfile.mkdtemp(prefix="leonardo_irc_real_", dir="/private/tmp")
    qst3 = psi4_qst_build_guess_atoms(react, prod, ts_guess_atoms=ts_guess, ts_weight=1.0, n_images=7)

    ts_dir = os.path.join(root, "01_qst3_ts")
    ts_proc, ts_result = _run_script(ts_dir, dlg._psi4_make_script(_ts_job(dlg, qst3)))
    ts_xyz = os.path.join(ts_dir, str(ts_result.get("optimized_xyz", "") or "optimized.xyz"))
    ts_atoms = psi4_collect_atoms_from_xyz(ts_xyz) if os.path.isfile(ts_xyz) else []

    irc_dir = os.path.join(root, "02_irc")
    if ts_atoms:
        irc_proc, irc_result = _run_script(irc_dir, dlg._psi4_make_script(_irc_job(dlg, ts_atoms)))
    else:
        irc_proc = None
        irc_result = {"ok": False, "error": "missing TS optimized geometry"}

    out = {
        "validation_dir": root,
        "ts": {
            "ok": bool(ts_result.get("ok", False)),
            "returncode": int(ts_proc.returncode),
            "energy_ts_hartree": ts_result.get("energy_ts_hartree"),
            "optimized_xyz": ts_xyz,
            "error": str(ts_result.get("error", "") or ""),
        },
        "irc": {
            "ok": bool(irc_result.get("ok", False)),
            "returncode": int(irc_proc.returncode) if irc_proc is not None else -1,
            "error": str(irc_result.get("error", "") or ""),
            "ts_validation": irc_result.get("irc_ts_validation", {}),
            "profile_csv": os.path.join(irc_dir, str(irc_result.get("irc_profile_csv", "") or "")),
            "points_completed": irc_result.get("irc_points_completed", 0),
            "primary_imag_cm1": irc_result.get("irc_primary_imag_cm1", 0.0),
            "coordinate_unit": irc_result.get("irc_coordinate_unit", ""),
        },
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    if not out["ts"]["ok"]:
        return 1
    if not (out["irc"]["ok"] and out["irc"]["ts_validation"].get("ok") and int(out["irc"]["points_completed"] or 0) >= 2):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
