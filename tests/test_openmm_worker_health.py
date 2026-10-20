"""The OpenMM MD worker checks the system before simulating it and watches the run (ubiquitin in vacuum)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("openmm")
QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets

UBIQUITIN = os.path.join(os.path.dirname(__file__), "data", "components", "1UBQ.pdb")


@pytest.fixture(scope="module")
def dlg():
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert qt_app is not None
    import code_v2

    d = code_v2.ProtLigInteractDialog()
    yield d
    d.close()


def _payload(tmp_path, lines=None, **over):
    pdb = tmp_path / "ubq_protein.pdb"
    if lines is None:
        with open(UBIQUITIN) as src:
            lines = [ln for ln in src if ln.startswith(("ATOM", "TER"))]
    pdb.write_text("".join(lines) + "END\n")
    data = {"input_pdb": str(pdb), "safe_obj": "ubq", "job_dir": str(tmp_path), "mode": "gas",
            "water_model": "tip3p", "ff_family": "amber14", "ensemble": "nvt", "ph": 7.0,
            "temperature_k": 300.0, "pressure_bar": 1.0, "timestep_fs": 2.0, "friction": 1.0,
            "cutoff_a": 10.0, "run_min": True, "min_iters": 300, "min_tolerance_kj_mol_nm": 10.0,
            "min_report_stride": 50, "run_heat": True, "heat_steps": 200, "run_equil": False, "equil_steps": 0,
            "run_prod": True, "prod_steps": 600, "traj_stride": 100, "report_stride": 50, "lig_xml": "",
            "lig_xml_paths": [], "include_waters": False, "prep_notes": [], "metal_restraints": False}
    data.update(over)
    return data


def test_healthy_run_reports_confidence(dlg, tmp_path):
    out = dlg._run_openmm_worker_job(_payload(tmp_path))
    assert out["ok"], out.get("error")
    health = out["health"]
    assert health["checks"] >= 2 and health["summary"]["confidence"] in ("high", "medium")
    assert {s["stage"] for s in health["samples"]} >= {"heating", "production"}
    assert 150 < health["stats"]["production_T_K"] < 450


def test_too_long_timestep_is_refused_and_4fs_gets_heavier_hydrogens(dlg, tmp_path):
    out = dlg._run_openmm_worker_job(_payload(tmp_path, timestep_fs=5.0))
    assert not out["ok"] and "Time step too long" in out["error"]
    assert not out.get("dcd_path")
    out = dlg._run_openmm_worker_job(_payload(tmp_path, timestep_fs=4.0, prod_steps=300))
    assert out["ok"], out.get("error")
    assert any("repartitioning" in n for n in out["notes"])


def test_unminimized_overlap_is_stopped_before_dynamics(dlg, tmp_path):
    with open(UBIQUITIN) as src:
        lines = [ln for ln in src if ln.startswith(("ATOM", "TER"))]
    ca = [i for i, ln in enumerate(lines) if ln[12:16] == " CA "]
    # put residue 30's CA 0.1 A from residue 60's CA
    target = lines[ca[59]]
    x, y, z = (float(target[30:38]) + 0.1, float(target[38:46]), float(target[46:54]))
    ln = lines[ca[29]]
    lines[ca[29]] = f"{ln[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{ln[54:]}"
    out = dlg._run_openmm_worker_job(_payload(tmp_path, lines=lines, run_min=False, min_iters=0))
    assert not out["ok"] and "health monitor" in out["error"]
    assert out["health"]["stopped"]["code"] in ("huge_force", "nan_start")
    assert out["health"]["summary"]["confidence"] == "low"
    assert any("Atoms on top of each other" in n for n in out["notes"])


def test_gap_bonded_across_is_flagged(dlg, tmp_path):
    with open(UBIQUITIN) as src:
        lines = [ln for ln in src if ln.startswith(("ATOM", "TER"))
                 and not (ln.startswith("ATOM") and 30 <= int(ln[22:26]) <= 35)]
    out = dlg._run_openmm_worker_job(_payload(tmp_path, lines=lines, run_heat=False, prod_steps=200))
    assert any("bonded across a gap" in n for n in out["notes"]), out["notes"][:4]
    assert out["health"]["summary"]["confidence"] == "low", (out.get("error"), list(out["health"]))
    if not out["ok"]:
        # the stretched bond makes the minimization blow up; the error names the cause, not just "NaN"
        assert out["health"]["stopped"]["code"] in ("minimization_nan", "nan", "huge_force")
        assert "chain bonded across a gap" in out["error"] or out["health"]["stopped"]["where"]
