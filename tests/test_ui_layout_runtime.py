import re
from pathlib import Path

from ui_layout_runtime import (
    LAUNCHER_TOP,
    LAUNCHER_SECTIONS,
    build_responsive_window_plan,
    dialog_identity,
    launcher_plan,
    responsive_grid_columns,
)


def test_responsive_window_plan_preserves_preferred_size_on_large_screen():
    plan = build_responsive_window_plan(1920, 1080, preferred_width=1080, preferred_height=820)

    assert plan["width"] == 1080
    assert plan["height"] == 820
    assert plan["compact"] is False


def test_responsive_window_plan_never_exceeds_small_screen():
    plan = build_responsive_window_plan(
        1024,
        600,
        preferred_width=1120,
        preferred_height=900,
        minimum_width=960,
        minimum_height=740,
    )

    assert plan["width"] <= plan["maximum_width"] < 1024
    assert plan["height"] <= plan["maximum_height"] < 600
    assert plan["minimum_width"] <= plan["width"]
    assert plan["minimum_height"] <= plan["height"]
    assert plan["compact"] is True


def test_responsive_grid_columns_reflows_cards_for_available_width():
    assert responsive_grid_columns(620) == 2
    assert responsive_grid_columns(500) == 1
    assert responsive_grid_columns(1400) == 3


def test_dialog_identity_assigns_scientific_context():
    openmm = dialog_identity("OpenMM Molecular Dynamics")
    docking = dialog_identity("Pocket Discovery & Docking")
    modeler = dialog_identity("Protein Modeling & Mutation (Pre-Docking/ABFE)")

    assert openmm["category"] == "MOLECULAR SIMULATION"
    assert "simulations" in openmm["subtitle"]
    assert docking["category"] == "MOLECULAR DOCKING"
    assert modeler["category"] == "PROTEIN MODELING"


def test_launcher_plan_keeps_section_order_and_drops_missing_buttons():
    plan = launcher_plan({"btn_open_qmmm", "btn_open_protein_prep", "btn_open_psi4_cluster", "btn_unrelated"})
    assert plan == [
        ("Structures & Sequences", ["btn_open_protein_prep"]),
        ("Quantum Chemistry & QM/MM", ["btn_open_psi4_cluster", "btn_open_qmmm"]),
    ]
    assert launcher_plan(()) == []


def test_every_workflow_window_button_has_one_launcher_card():
    # a new "btn_open_*" window button must be placed in LAUNCHER_SECTIONS, or it never reaches the main window
    source = (Path(__file__).resolve().parents[1] / "legacy" / "code_v2.py").read_text(encoding="utf-8")
    created = set(re.findall(r"self\.(btn_open_[a-z0-9_]+) = QtWidgets\.QPushButton\(", source))
    # step buttons inside the Protein Modeling and Pocket & Docking windows, and the button inside the main
    # window's Open Molecules card, are not launcher entries
    inner = {n for n in created if n.endswith("_popup")} | {"btn_open_interaction_workspace"}
    placed = [n for _, names in LAUNCHER_SECTIONS for n in names]
    assert len(placed) == len(set(placed))
    assert set(placed) | set(LAUNCHER_TOP) == created - inner
