from scientific_readiness_runtime import build_scientific_workflow_catalog
from scientific_readiness_runtime import build_suite_readiness_report
from scientific_readiness_runtime import build_workflow_readiness_report
from scientific_readiness_runtime import format_suite_readiness_lines
from scientific_readiness_runtime import list_scientific_workflow_ids

import report_scientific_readiness


def test_workflow_ids_cover_suite_surface():
    ids = set(list_scientific_workflow_ids())
    assert {
        "bioseq",
        "alphafold",
        "docking",
        "redocking",
        "virtual_screening",
        "lead_optimization",
        "abfe",
        "openmm_qmmm",
        "psi4_qm",
    }.issubset(ids)


def test_build_scientific_workflow_catalog_returns_copy():
    catalog = build_scientific_workflow_catalog()
    catalog[0]["title"] = "mutated"
    fresh = build_scientific_workflow_catalog()
    assert fresh[0]["title"] != "mutated"


def test_bioseq_readiness_degrades_gracefully_without_local_cli_tools():
    report = build_workflow_readiness_report(
        "bioseq",
        python_available_fn=lambda name: name == "Bio",
        executable_resolver=lambda name: None,
    )
    assert report["ok"]
    assert report["status"] == "ready_with_gaps"
    assert report["missing_executables_recommended"] == ["blastp", "hmmscan", "clustalo", "mafft"]
    assert report["missing_python_required"] == []


def test_abfe_readiness_flags_missing_required_stack():
    report = build_workflow_readiness_report(
        "abfe",
        python_available_fn=lambda name: name in {"pymbar", "rdkit"},
        executable_resolver=lambda name: "/usr/bin/" + name if name in {"antechamber", "parmchk2"} else None,
    )
    assert not report["ok"]
    assert report["status"] == "missing_required"
    assert report["missing_python_required"] == ["openmm"]
    assert report["missing_executables_required"] == ["tleap"]
    assert any("openmm" in hint for hint in report["install_hints"])


def test_suite_readiness_aggregates_status_counts():
    report = build_suite_readiness_report(
        workflow_ids=["bioseq", "abfe"],
        python_available_fn=lambda name: name in {"Bio", "pymbar"},
        executable_resolver=lambda name: "/usr/bin/" + name if name in {"antechamber", "parmchk2"} else None,
    )
    assert not report["ok"]
    assert report["status_counts"]["ready_with_gaps"] == 1
    assert report["status_counts"]["missing_required"] == 1
    lines = format_suite_readiness_lines(report)
    assert any("bioseq: ready_with_gaps" in line for line in lines)
    assert any("abfe: missing_required" in line for line in lines)


def test_report_cli_list_outputs_known_workflow_ids(capsys):
    rc = report_scientific_readiness.main(["--list"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "bioseq" in captured.out
    assert "abfe" in captured.out


def test_report_cli_strict_uses_suite_status(monkeypatch, capsys):
    monkeypatch.setattr(
        report_scientific_readiness,
        "build_suite_readiness_report",
        lambda workflow_ids=None: {
            "ok": False,
            "n_workflows": 1,
            "status_counts": {"ready": 0, "ready_with_gaps": 0, "missing_required": 1},
            "workflow_reports": [],
        },
    )
    rc = report_scientific_readiness.main(["--strict"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "Scientific suite readiness:" in captured.out
