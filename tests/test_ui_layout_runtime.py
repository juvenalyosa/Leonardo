from ui_layout_runtime import build_responsive_window_plan, dialog_identity, responsive_grid_columns


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
