from pathlib import Path

from task_monitor_runtime import task_monitor_probe_job
from task_monitor_runtime import task_monitor_probe_openmm
from task_monitor_runtime import task_monitor_probe_psi4


def test_task_monitor_probe_psi4_reads_live_log_tail(tmp_path):
    job_dir = Path(tmp_path)
    log_path = job_dir / "psi4_output.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-04-01:17:14:24,795 INFO     [psi4.driver.driver:526] Return energy(): -227.891754748038",
                "2026-04-01:17:14:24,814 INFO     [psi4.driver.task_planner:285] PLANNING Atomic: keywords={...}",
            ]
        ),
        encoding="utf-8",
    )

    probe = task_monitor_probe_psi4(job_dir)

    assert probe["source"] == "psi4_output.log"
    assert "PLANNING Atomic" in probe["message"]
    assert "psi4_output.log" in probe["signature"]


def test_task_monitor_probe_openmm_reads_state_csv(tmp_path):
    job_dir = Path(tmp_path)
    csv_path = job_dir / "protein_openmm_state.csv"
    csv_path.write_text(
        "\n".join(
            [
                '#"Step","Potential Energy (kJ/mole)","Kinetic Energy (kJ/mole)","Total Energy (kJ/mole)","Temperature (K)","Speed (ns/day)"',
                "1000,-1234.5,456.7,-777.8,299.4,12.34",
                "2000,-1200.1,455.5,-744.6,300.1,13.21",
            ]
        ),
        encoding="utf-8",
    )

    probe = task_monitor_probe_openmm(job_dir)

    assert probe["source"] == "protein_openmm_state.csv"
    assert "Step 2000" in probe["message"]
    assert "T=300.1 K" in probe["message"]
    assert "Epot=-1200.1 kJ/mol" in probe["message"]


def test_task_monitor_probe_openmm_reads_minimization_csv(tmp_path):
    job_dir = Path(tmp_path)
    min_csv = job_dir / "protein_openmm_minimization.csv"
    min_csv.write_text(
        "\n".join(
            [
                "iteration,potential_energy_kj_mol",
                "50,-5400.0",
                "100,-5501.25",
            ]
        ),
        encoding="utf-8",
    )

    probe = task_monitor_probe_openmm(job_dir)

    assert probe["source"] == "protein_openmm_minimization.csv"
    assert "Minimization iter 100" in probe["message"]
    assert "E=-5501.2 kJ/mol" in probe["message"]


def test_task_monitor_probe_dispatches_by_tool_label(tmp_path):
    job_dir = Path(tmp_path)
    csv_path = job_dir / "protein_openmm_state.csv"
    csv_path.write_text(
        "\n".join(
            [
                '#"Step","Potential Energy (kJ/mole)","Temperature (K)"',
                "500,-100.0,298.0",
            ]
        ),
        encoding="utf-8",
    )

    probe = task_monitor_probe_job("QM/MM", job_dir)

    assert "Step 500" in probe["message"]
