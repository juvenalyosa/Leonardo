"""After an MD: the solvated result shows the protein (water box hidden), and the CPU estimate is realistic."""
import os

import pytest

from leonardo import hardware_runtime as hw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_explicit_md_estimate_on_cpu_matches_a_measured_run(tmp_path, monkeypatch):
    monkeypatch.setattr(hw, "CACHE_PATH", str(tmp_path / "hw.json"))     # no calibration: the defaults
    # measured: 1AKI in TIP3P (~21k atoms), 0.1 ns production + 30 ps heating/equilibration, 2 fs, on a
    # 6-core i7-8850H: 38 min in total (5 min solvation + minimization, 5.6 ns/day dynamics)
    ns = 0.1 * 1.1 + 0.014                      # what the assistant's estimate asks for (estimates.py)
    est = hw.estimate_md(20524, ns, "explicit", platform_name="CPU", prof={})
    assert 25 <= est["minutes"] <= 55, est
    assert not est["calibrated"]


def test_explicit_box_size_follows_the_protein():
    """1AKI solvated by Leonardo (10 A padding, cubic box): 20524 atoms; the old rule (8 x protein) gave 15680."""
    pytest.importorskip("pymol")
    from pymol import cmd

    cmd.delete("all")
    cmd.load(os.path.join(ROOT, "examples", "gpu_validation", "1AKI.pdb"), "lyz")
    try:
        n = hw.md_system_atoms(cmd, "lyz", "explicit", 10.0)
        assert abs(n - 20524) / 20524 < 0.15, n
        assert hw.md_system_atoms(cmd, "lyz", "implicit") < 2500          # solute with hydrogens only
        assert hw.md_system_atoms(cmd, "lyz", "explicit", 15.0) > n        # more padding, more water
    finally:
        cmd.delete("all")


def test_calibrated_estimate_keeps_the_measured_speed(tmp_path, monkeypatch):
    monkeypatch.setattr(hw, "CACHE_PATH", str(tmp_path / "hw.json"))
    hw.record_md_speed("CPU", "explicit", 20700, 5.55)
    est = hw.estimate_md(20700, 0.13, "explicit", platform_name="CPU", prof=hw._load_cache())
    assert est["calibrated"] and abs(est["ns_per_day"] - 5.55) < 0.05
    assert 30 <= est["minutes"] <= 45, est      # dynamics (~34 min) + the minimization of the box


def test_md_result_hides_the_water_box_and_frames_the_protein():
    pymol = pytest.importorskip("pymol")
    from pymol import cmd

    import code_v2

    cmd.delete("all")
    cmd.load(os.path.join(ROOT, "examples", "3ptb_ben", "3PTB.pdb"), "md_result")
    assert cmd.count_atoms("md_result and resn HOH") > 0

    class Dlg:
        def _apply_ball_and_stick_style(self, selection, sphere_scale=0.22, stick_radius=0.16):
            cmd.show("sticks", selection)

    try:
        code_v2.ProtLigInteractDialog._style_md_result(Dlg(), "md_result")
        assert cmd.count_atoms("md_result and resn HOH and (rep sticks or rep spheres or rep lines "
                               "or rep nonbonded or rep nb_spheres)") == 0
        assert cmd.count_atoms("md_result and polymer and rep cartoon") > 0
        assert cmd.count_atoms("md_result and resn BEN and rep sticks") > 0     # the ligand stays visible
        assert cmd.count_atoms("md_result and resn CA and rep sticks") > 0       # structural Ca2+ kept
        cmd.alter("md_result and resn CA", "resn='NA'")                          # a counter-ion instead
        code_v2.ProtLigInteractDialog._style_md_result(Dlg(), "md_result")
        assert cmd.count_atoms("md_result and resn NA and (rep sticks or rep spheres)") == 0
    finally:
        cmd.delete("all")


def test_md_result_frames_the_current_frame_not_the_whole_drift():
    pytest.importorskip("pymol")
    from pymol import cmd

    import code_v2

    class Dlg:
        def _apply_ball_and_stick_style(self, selection, sphere_scale=0.22, stick_radius=0.16):
            pass

    cmd.delete("all")
    cmd.load(os.path.join(ROOT, "examples", "3ptb_ben", "3PTB.pdb"), "md_result")
    try:
        cmd.create("md_result", "md_result", 1, 2)
        cmd.translate([150.0, 0.0, 0.0], "md_result", state=2, camera=0)    # the protein drifted away
        cmd.frame(1)
        code_v2.ProtLigInteractDialog._style_md_result(Dlg(), "md_result")
        framed = abs(cmd.get_view()[11])
        cmd.zoom("md_result and polymer", 4, state=0)
        assert framed < 0.6 * abs(cmd.get_view()[11])
    finally:
        cmd.delete("all")


def test_assistant_md_focus_frames_the_protein_not_the_water_box():
    pytest.importorskip("pymol")
    from pymol import cmd

    from leonardo.agent import presentation

    cmd.delete("all")
    cmd.load(os.path.join(ROOT, "examples", "3ptb_ben", "3PTB.pdb"), "md_out")
    try:
        cmd.pseudoatom("md_out", pos=[60.0, 60.0, 60.0], resn="HOH", name="O")      # a corner of the box
        cmd.hide("everything", "md_out and resn HOH")
        cmd.orient("md_out", animate=0)
        cmd.zoom("md_out", buffer=2.0, complete=1)
        whole = abs(cmd.get_view()[11])
        presentation.focus("(md_out) and polymer", buffer=2.0)
        assert abs(cmd.get_view()[11]) < 0.8 * whole
        presentation.focus("md_out")                     # e.g. trajectory_rmsd_rmsf: the hidden water is ignored
        assert abs(cmd.get_view()[11]) < 0.8 * whole
    finally:
        cmd.delete("all")
