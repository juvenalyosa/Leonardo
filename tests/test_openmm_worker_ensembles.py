"""Leonardo's OpenMM MD worker honours the requested ensemble and temperature (ubiquitin in vacuum, a few hundred steps)."""
import csv
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


def _payload(tmp_path, ensemble, temperature_k):
    pdb = tmp_path / "ubq_protein.pdb"
    with open(UBIQUITIN) as src:
        pdb.write_text("".join(ln for ln in src if ln.startswith(("ATOM", "TER"))) + "END\n")
    return {"input_pdb": str(pdb), "safe_obj": "ubq", "job_dir": str(tmp_path), "mode": "gas",
            "water_model": "tip3p", "ff_family": "amber14", "ensemble": ensemble, "ph": 7.0,
            "temperature_k": float(temperature_k), "pressure_bar": 1.0, "timestep_fs": 1.0, "friction": 1.0,
            "cutoff_a": 10.0, "run_min": True, "min_iters": 400, "min_tolerance_kj_mol_nm": 10.0,
            "min_report_stride": 50, "run_heat": False, "heat_steps": 0, "run_equil": False, "equil_steps": 0,
            "run_prod": True, "prod_steps": 400, "traj_stride": 100, "report_stride": 20, "lig_xml": "",
            "lig_xml_paths": [], "include_waters": False, "prep_notes": [], "metal_restraints": False}


def _mean_temperature(csv_path):
    with open(csv_path) as fh:
        rows = list(csv.reader(fh))
    col = next(i for i, name in enumerate(rows[0]) if "Temperature" in name)
    values = [float(r[col]) for r in rows[1:]]
    return sum(values) / len(values)


def test_nve_run_starts_from_velocities_at_the_requested_temperature(dlg, tmp_path):
    out = dlg._run_openmm_worker_job(_payload(tmp_path, "nve", 300.0))
    assert out["ok"], out.get("error")
    assert out["ensemble"] == "nve"
    # without initial velocities a Verlet run sits at the energy minimum (a few K); with them it holds
    # roughly half the requested temperature after equipartition with the potential energy
    assert _mean_temperature(out["csv_path"]) > 80.0
