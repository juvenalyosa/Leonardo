from openmm_dynamics_runtime import apply_openmm_cv_preset
from openmm_dynamics_runtime import build_openmm_capability_report
from openmm_dynamics_runtime import normalize_openmm_run_controls
from openmm_dynamics_runtime import openmm_cv_expression
from openmm_dynamics_runtime import openmm_cv_presets
from openmm_dynamics_runtime import openmm_default_cv_context
from openmm_dynamics_runtime import run_openmm_alanine_cv_suite
from openmm_dynamics_runtime import run_openmm_alanine_dipeptide_fes
from openmm_dynamics_runtime import run_openmm_advanced_template_smoke
from openmm_dynamics_runtime import run_openmm_metadynamics_smoke
from openmm_dynamics_runtime import run_openmm_umbrella_smoke
from openmm_dynamics_runtime import summarize_openmm_cv_definitions
from openmm_dynamics_runtime import validate_openmm_cv_definitions


def test_openmm_cv_presets_resolve_context_and_validate_for_umbrella():
    ctx = openmm_default_cv_context(
        object_name="complex",
        ligand_selection="(complex and resn BEN)",
        binding_site_selection="(complex and polymer within 6 of (complex and resn BEN))",
    )
    cv = apply_openmm_cv_preset("protein_ligand_distance", context=ctx, workflow="umbrella")

    assert cv["valid"] is True
    assert cv["type"] == "distance"
    assert "BEN" in cv["selection_a"]
    assert "polymer" in cv["selection_b"]
    assert cv["expression"].startswith("distance(")
    assert cv["umbrella"]["n_windows"] >= 2
    assert cv["force_expression"] == cv["expression"]
    assert validate_openmm_cv_definitions([cv], workflow="umbrella")["ok"] is True


def test_openmm_cv_validation_catches_missing_angle_group_and_metadynamics_limits():
    bad_angle = {
        "name": "angle_bad",
        "type": "angle",
        "selection_a": "(index 1)",
        "selection_b": "(index 2)",
        "selection_c": "",
    }
    report = validate_openmm_cv_definitions([bad_angle], workflow="umbrella")
    assert report["ok"] is False
    assert any("selection_c" in err for err in report["errors"])

    four_cvs = [
        apply_openmm_cv_preset("protein_ligand_distance", context={}, workflow="metadynamics")
        for _ in range(4)
    ]
    meta_report = validate_openmm_cv_definitions(four_cvs, workflow="metadynamics")
    assert meta_report["ok"] is False
    assert any("at most three" in err for err in meta_report["errors"])


def test_openmm_cv_expression_and_summary_are_stable():
    expr = openmm_cv_expression("dihedral", "a", "b", "c", "d")
    assert expr == "dihedral(a,b,c,d)"

    cv = apply_openmm_cv_preset("torsion_dihedral", context={}, workflow="metadynamics")
    txt = summarize_openmm_cv_definitions([cv])
    assert "Torsion/dihedral" in txt
    assert "dihedral" in txt

    preset_ids = {row["id"] for row in openmm_cv_presets()}
    assert {"protein_ligand_distance", "reaction_bond_distance", "torsion_dihedral"} <= preset_ids


def test_openmm_run_controls_are_clipped_and_npt_requires_explicit_solvent():
    controls = normalize_openmm_run_controls(
        {
            "mode": "gas",
            "ensemble": "npt",
            "ph": 99,
            "temperature_k": -5,
            "timestep_fs": 99,
            "prod_steps": -1,
            "traj_stride": 0,
        }
    )

    assert controls["mode"] == "gas"
    assert controls["ensemble"] == "nvt"
    assert controls["ph"] == 14.0
    assert controls["temperature_k"] == 50.0
    assert controls["timestep_fs"] == 5.0
    assert controls["prod_steps"] == 0
    assert controls["traj_stride"] == 1


def test_openmm_capability_report_handles_missing_and_fake_runtime():
    missing = build_openmm_capability_report(None, None)
    assert missing["ok"] is False
    assert missing["message"] == "OpenMM missing"

    class _FakeVersion:
        version = "8.test"

    class _FakePlatform:
        @staticmethod
        def getNumPlatforms():
            return 2

        @staticmethod
        def getPlatform(i):
            class _One:
                def getName(self):
                    return ["CPU", "Reference"][i]

            return _One()

    class _FakeOpenMM:
        version = _FakeVersion()
        Platform = _FakePlatform()

    class _FakeApp:
        class Metadynamics:
            pass

        class BiasVariable:
            pass

    report = build_openmm_capability_report(_FakeOpenMM, _FakeApp)
    assert report["ok"] is True
    assert report["version"] == "8.test"
    assert report["platforms"] == ["CPU", "Reference"]
    assert report["has_metadynamics"] is True


def test_openmm_umbrella_smoke_runs_real_openmm_windows(tmp_path):
    import pytest

    pytest.importorskip("openmm")
    cv = apply_openmm_cv_preset("protein_ligand_distance", context={}, workflow="umbrella")
    out = run_openmm_umbrella_smoke(
        tmp_path / "umbrella",
        cv_definition=cv,
        n_windows=3,
        steps_per_window=3,
        spring_k_kj_mol_nm2=100.0,
    )

    assert out["ok"] is True
    assert out["workflow"] == "umbrella"
    assert out["n_windows"] == 3
    assert len(out["rows"]) == 3
    assert (tmp_path / "umbrella" / "umbrella_smoke_windows.csv").is_file()
    assert (tmp_path / "umbrella" / "umbrella_smoke_summary.json").is_file()


def test_openmm_metadynamics_smoke_runs_real_biasvariable(tmp_path):
    import pytest

    pytest.importorskip("openmm")
    cv = apply_openmm_cv_preset("protein_ligand_distance", context={}, workflow="metadynamics")
    out = run_openmm_metadynamics_smoke(
        tmp_path / "meta",
        cv_definitions=[cv],
        steps=4,
        frequency_steps=2,
        save_frequency_steps=2,
    )

    assert out["ok"] is True
    assert out["workflow"] == "metadynamics"
    assert out["steps"] == 4
    assert len(out["bias_files"]) >= 1
    assert (tmp_path / "meta" / "metadynamics_smoke_summary.json").is_file()


def test_openmm_advanced_template_smoke_dispatches_umbrella_and_meta(tmp_path):
    import pytest

    pytest.importorskip("openmm")
    cv_u = apply_openmm_cv_preset("protein_ligand_distance", context={}, workflow="umbrella")
    umbrella = run_openmm_advanced_template_smoke(
        {
            "workflow": "umbrella",
            "parameters": {
                "cv": cv_u,
                "windowing": {"n_windows": 3, "spring_k_kj_mol_nm2": 100.0},
                "thermostat": {"temperature_k": 300.0, "timestep_fs": 1.0},
            },
        },
        tmp_path / "template_umbrella",
    )
    assert umbrella["ok"] is True
    assert umbrella["workflow"] == "umbrella"

    cv_m = apply_openmm_cv_preset("protein_ligand_distance", context={}, workflow="metadynamics")
    meta = run_openmm_advanced_template_smoke(
        {
            "workflow": "metadynamics",
            "parameters": {
                "collective_variables": [cv_m],
                "total_steps": 4,
                "frequency_steps": 2,
                "save_frequency_steps": 2,
            },
        },
        tmp_path / "template_meta",
    )
    assert meta["ok"] is True
    assert meta["workflow"] == "metadynamics"


def test_openmm_alanine_dipeptide_fes_smoke_writes_plot(tmp_path):
    import pytest

    pytest.importorskip("openmm")
    pytest.importorskip("openff.toolkit")
    out = run_openmm_alanine_dipeptide_fes(
        tmp_path / "alanine",
        max_blocks=2,
        block_steps=2,
        convergence_rmsd_kj_mol=1.0e9,
        stable_blocks=1,
    )

    assert out["ok"] is True
    assert out["workflow"] == "alanine_dipeptide_phi_psi_metadynamics"
    assert out["diagnostic_only"] is True
    assert out["plotting_stability_passed"] is True
    assert (tmp_path / "alanine" / "alanine_phi_psi_fes.csv").is_file()
    assert (tmp_path / "alanine" / "alanine_phi_psi_fes.png").is_file()
    assert (tmp_path / "alanine" / "alanine_fes_convergence.csv").is_file()


def test_openmm_alanine_cv_suite_smoke_plots_all_presets(tmp_path):
    import os

    import pytest

    pytest.importorskip("openmm")
    pytest.importorskip("openff.toolkit")
    out = run_openmm_alanine_cv_suite(
        tmp_path / "alanine_cv",
        steps=4,
        sample_interval=2,
    )

    assert out["ok"] is True
    assert out["workflow"] == "alanine_dipeptide_cv_suite"
    assert out["cv_count"] == len(openmm_cv_presets())
    assert out["samples"] == 3
    assert {row["type"] for row in out["stats"]} == {"distance", "angle", "dihedral", "rmsd"}
    assert (tmp_path / "alanine_cv" / "alanine_cv_timeseries.csv").is_file()
    assert (tmp_path / "alanine_cv" / "alanine_cv_all_presets_timeseries.png").is_file()
    assert (tmp_path / "alanine_cv" / "alanine_cv_all_presets_histograms.png").is_file()
    for path in out["plot_paths"]["by_type_timeseries"].values():
        assert os.path.isfile(path)
