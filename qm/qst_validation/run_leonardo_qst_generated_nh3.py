import json
import os
import subprocess
import sys
import tempfile


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import code_v2  # noqa: E402
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


def _run_generated_job(dlg, root, label, qst):
    job_dir = os.path.join(root, label)
    os.makedirs(job_dir, exist_ok=True)
    job = {
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
    script_path = os.path.join(job_dir, "psi4_input.py")
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(dlg._psi4_make_script(job))
    proc = subprocess.run(
        [sys.executable, script_path],
        cwd=job_dir,
        text=True,
        capture_output=True,
        timeout=300,
    )
    result_path = os.path.join(job_dir, "psi4_result.json")
    result = {}
    if os.path.isfile(result_path):
        with open(result_path, "r", encoding="utf-8") as fh:
            result = json.load(fh)
    imag = [float(v) for v in result.get("freq_cm_imag", []) if abs(float(v)) > 1.0e-4]
    return {
        "label": label,
        "job_dir": job_dir,
        "returncode": int(proc.returncode),
        "ok": bool(result.get("ok", False)),
        "error": str(result.get("error", "") or ""),
        "qst_scan_selected": result.get("qst_scan_selected", {}),
        "qst_scan_n": len(result.get("qst_scan", []) or []),
        "energy_ts_hartree": result.get("energy_ts_hartree"),
        "imaginary_frequency_count": len(imag),
        "imaginary_frequencies_cm1": imag,
        "stderr_tail": str(proc.stderr or "")[-1200:],
    }


def main():
    dlg = _make_dialog()
    react, prod, ts_guess = _nh3_endpoints()
    root = tempfile.mkdtemp(prefix="leonardo_qst_real_", dir="/private/tmp")
    qst2 = psi4_qst_build_guess_atoms(react, prod, n_images=7)
    qst3 = psi4_qst_build_guess_atoms(react, prod, ts_guess_atoms=ts_guess, ts_weight=1.0, n_images=7)
    out = {
        "validation_dir": root,
        "qst2_note": qst2["note"],
        "qst3_note": qst3["note"],
        "qst2": _run_generated_job(dlg, root, "qst2_nh3", qst2),
        "qst3": _run_generated_job(dlg, root, "qst3_nh3", qst3),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    if not (out["qst2"]["ok"] and out["qst3"]["ok"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
