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


def test_rmsd_cv_is_zero_for_a_rigidly_rotated_copy():
    # the backbone_rmsd CV superposes before measuring: a rotated + translated copy has RMSD 0, and a
    # distorted copy keeps exactly its own RMSD
    import math

    import numpy as np

    from openmm_dynamics_runtime import _rmsd_nm

    rng = np.random.default_rng(7)
    ref = rng.normal(size=(12, 3))
    a, b = math.radians(50.0), math.radians(-35.0)
    rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(b), -math.sin(b)], [0.0, math.sin(b), math.cos(b)]])
    rot = rz @ rx
    moved = ref @ rot.T + np.array([0.4, -1.1, 2.0])
    assert _rmsd_nm(moved, ref) < 1e-9
    assert _rmsd_nm(ref, moved) < 1e-9
    noise = rng.normal(scale=0.05, size=ref.shape)
    noise -= noise.mean(axis=0)
    expected = float(np.sqrt(np.mean(np.sum(noise * noise, axis=1))))
    assert _rmsd_nm((ref + noise) @ rot.T, ref) <= expected + 1e-9
    # a mirror image cannot be superposed by a proper rotation
    assert _rmsd_nm(ref * np.array([1.0, 1.0, -1.0]), ref) > 0.1


def test_metadynamics_smoke_accepts_a_deposit_frequency_that_does_not_divide_the_save_interval(tmp_path):
    # run_enhanced_sampling saves the template with frequency 4 / save 5000; the smoke test clips them to 4 / 10,
    # which OpenMM rejects unless the save interval is rounded to a multiple of the deposit frequency
    payload = {"workflow": "metadynamics",
               "parameters": {"collective_variables": [], "total_steps": 1000, "frequency_steps": 4,
                              "save_frequency_steps": 5000}}
    out = run_openmm_advanced_template_smoke(payload, str(tmp_path / "meta"))
    assert out["ok"] is True, out
    assert out["steps"] == 30


def _place(a, b, c, bond, angle_deg, dihedral_deg):
    """NeRF: the atom bonded to c with the given c bond length, b-c-d angle and a-b-c-d dihedral."""
    import math

    import numpy as np

    bc = (c - b) / np.linalg.norm(c - b)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    th, ph = math.radians(angle_deg), math.radians(dihedral_deg)
    d2 = np.array([-bond * math.cos(th), bond * math.sin(th) * math.cos(ph), bond * math.sin(th) * math.sin(ph)])
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def test_phi_psi_free_energy_map_puts_the_hills_at_their_phi_psi(tmp_path):
    # hills deposited at a fixed (phi, psi) must give the free-energy minimum at that (phi, psi) in the CSV:
    # OpenMM's grid is indexed [psi, phi] and spans -180..+180 inclusive
    import csv
    import math

    import numpy as np
    import openmm as mm
    import openmm.app as app
    from openmm import unit

    from openmm_dynamics_runtime import _phi_psi_free_energy_grid
    from openmm_dynamics_runtime import _shift_fes_minimum
    from openmm_dynamics_runtime import _write_alanine_fes_csv

    p = [np.array([0.0, 0.14, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.15, 0.0, 0.0])]
    p.append(_place(p[0], p[1], p[2], 0.15, 110.0, -100.0))      # phi = -100
    p.append(_place(p[1], p[2], p[3], 0.15, 110.0, 140.0))       # psi = +140
    system = mm.System()
    for _ in p:
        system.addParticle(0.0)                                    # frozen: every hill lands on the same point
    system.addForce(mm.HarmonicBondForce())
    cvs = []
    for idx in ((0, 1, 2, 3), (1, 2, 3, 4)):
        f = mm.CustomTorsionForce("theta")
        f.addTorsion(*idx, [])
        cvs.append(f)
    phi = app.BiasVariable(cvs[0], -math.pi, math.pi, 0.3, True, 24)
    psi = app.BiasVariable(cvs[1], -math.pi, math.pi, 0.3, True, 36)
    meta = app.Metadynamics(system, [phi, psi], 300.0 * unit.kelvin, 5.0, 1.0 * unit.kilojoule_per_mole, 1)
    top = app.Topology()
    res = top.addResidue("DUM", top.addChain())
    for k in range(len(p)):
        top.addAtom(f"C{k}", app.element.carbon, res)
    sim = app.Simulation(top, system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions([mm.Vec3(*x) for x in p])
    meta.step(sim, 5)
    phi_deg, psi_deg = (math.degrees(v) for v in meta.getCollectiveVariables(sim))
    assert abs(phi_deg + 100.0) < 0.5 and abs(psi_deg - 140.0) < 0.5

    grid = _phi_psi_free_energy_grid(meta, unit)
    assert grid.shape == (24, 36)
    path = tmp_path / "fes.csv"
    _write_alanine_fes_csv(str(path), _shift_fes_minimum(grid))
    with open(path) as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    best = min(rows, key=lambda r: r["diagnostic_grid_kj_mol"])
    assert abs(best["phi_deg"] - phi_deg) <= 0.5 * 360.0 / 23 + 1e-6
    assert abs(best["psi_deg"] - psi_deg) <= 0.5 * 360.0 / 35 + 1e-6
