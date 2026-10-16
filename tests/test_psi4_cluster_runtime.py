import math

from psi4_cluster_runtime import (
    add_peptide_link_h_caps,
    adjust_atoms_for_bond_changes,
    build_cluster_freeze_indices,
    build_pymol_cluster_selection,
    cluster_workflow_stage_plan,
    parse_atom_index_list,
    parse_bond_change_pairs,
    prepare_cluster_model,
    select_cluster_atoms_around_seeds,
)


def _dist(a, b):
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    dz = float(a["z"]) - float(b["z"])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def test_parse_bond_change_pairs():
    pairs = parse_bond_change_pairs("12-45, 17:39; 45-12, bad, 0-3")
    assert (12, 45) in pairs
    assert (17, 39) in pairs
    assert len(pairs) == 2


def test_parse_atom_index_list_supports_ranges_and_dedupes():
    assert parse_atom_index_list("12, 45 80-82;45") == [12, 45, 80, 81, 82]


def test_build_pymol_cluster_selection_around_indices():
    sel = build_pymol_cluster_selection(
        source_object="prot",
        mode="around_indices",
        seed_indices="12,45",
        radius=6.5,
        expand_residues=True,
        include_waters=False,
    )

    assert sel["final_sel"].startswith("byres")
    assert "within 6.500" in sel["final_sel"]
    assert "index 12+45" in sel["final_sel"]
    assert "not (solvent and resn HOH" in sel["final_sel"]       # real water only (not lone XYZ oxygens)


def test_adjust_atoms_for_form_and_break_changes():
    atoms = [
        {"index": 1, "element": "C", "x": 0.0, "y": 0.0, "z": 0.0},
        {"index": 2, "element": "O", "x": 3.0, "y": 0.0, "z": 0.0},
        {"index": 3, "element": "C", "x": 0.0, "y": 0.0, "z": 0.0},
        {"index": 4, "element": "C", "x": 1.2, "y": 0.0, "z": 0.0},
    ]
    d_form_before = _dist(atoms[0], atoms[1])
    d_break_before = _dist(atoms[2], atoms[3])
    out, _note = adjust_atoms_for_bond_changes(
        atoms,
        form_pairs=[(1, 2)],
        break_pairs=[(3, 4)],
    )
    d_form_after = _dist(out[0], out[1])
    d_break_after = _dist(out[2], out[3])
    assert d_form_after < d_form_before
    assert d_break_after > d_break_before


def _toy_context_for_caps():
    return [
        {"index": 1, "name": "N", "element": "N", "resn": "GLY", "chain": "A", "resi": "9", "x": -3.0, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "CA", "element": "C", "resn": "GLY", "chain": "A", "resi": "9", "x": -2.0, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "C", "element": "C", "resn": "GLY", "chain": "A", "resi": "9", "x": -1.2, "y": 0.0, "z": 0.0},
        {"index": 4, "name": "N", "element": "N", "resn": "SER", "chain": "A", "resi": "10", "x": 0.0, "y": 0.0, "z": 0.0},
        {"index": 5, "name": "CA", "element": "C", "resn": "SER", "chain": "A", "resi": "10", "x": 1.2, "y": 0.0, "z": 0.0},
        {"index": 6, "name": "C", "element": "C", "resn": "SER", "chain": "A", "resi": "10", "x": 2.4, "y": 0.0, "z": 0.0},
        {"index": 7, "name": "OG", "element": "O", "resn": "SER", "chain": "A", "resi": "10", "x": 1.2, "y": 1.3, "z": 0.0},
        {"index": 8, "name": "N", "element": "N", "resn": "ALA", "chain": "A", "resi": "11", "x": 3.6, "y": 0.0, "z": 0.0},
        {"index": 9, "name": "CA", "element": "C", "resn": "ALA", "chain": "A", "resi": "11", "x": 4.8, "y": 0.0, "z": 0.0},
        {"index": 10, "name": "C", "element": "C", "resn": "ALA", "chain": "A", "resi": "11", "x": 6.0, "y": 0.0, "z": 0.0},
    ]


def test_select_cluster_atoms_around_seeds_expands_full_residue():
    atoms = _toy_context_for_caps()
    selected, meta = select_cluster_atoms_around_seeds(
        atoms,
        seed_indices=[7],
        radius=0.2,
        expand_residues=True,
    )

    assert int(meta["seed_count"]) == 1
    assert {a["index"] for a in selected} == {4, 5, 6, 7}


def test_add_peptide_link_h_caps_uses_full_context_boundaries():
    context = _toy_context_for_caps()
    cluster = [a for a in context if str(a["resi"]) == "10"]

    capped, meta = add_peptide_link_h_caps(cluster, context_atoms=context)

    caps = [a for a in capped if a.get("is_cap")]
    assert int(meta["cap_count"]) == 2
    assert {a["cap_type"] for a in caps} == {"N-link-H", "C-link-H"}
    assert len(capped) == len(cluster) + 2


def test_prepare_cluster_model_adds_caps_and_freezes_boundary():
    context = _toy_context_for_caps()
    cluster = [a for a in context if str(a["resi"]) == "10"]

    prepared, freeze, meta = prepare_cluster_model(
        cluster,
        context_atoms=context,
        add_caps=True,
        freeze_backbone=True,
        freeze_caps=True,
    )

    cap_indices = set(meta["caps"]["cap_atom_indices"])
    assert int(meta["caps"]["cap_count"]) == 2
    assert cap_indices.issubset(set(freeze))
    assert len(prepared) == len(cluster) + 2


def test_cluster_workflow_stage_plan_includes_qst3_before_irc():
    assert cluster_workflow_stage_plan(
        do_classical=False,
        do_preopt=True,
        run_qst3=True,
        run_irc=True,
    ) == ["react_preopt", "prod_preopt", "qst2_ts", "qst3_ts", "irc"]


def test_build_cluster_freeze_indices_backbone_and_caps():
    atoms = [
        {"index": 1, "name": "N", "resn": "ALA", "chain": "A", "resi": "10"},
        {"index": 2, "name": "CA", "resn": "ALA", "chain": "A", "resi": "10"},
        {"index": 3, "name": "CB", "resn": "ALA", "chain": "A", "resi": "10"},
        {"index": 4, "name": "N", "resn": "GLY", "chain": "A", "resi": "11"},
        {"index": 5, "name": "CA", "resn": "GLY", "chain": "A", "resi": "11"},
        {"index": 6, "name": "O", "resn": "GLY", "chain": "A", "resi": "11"},
    ]
    freeze, meta = build_cluster_freeze_indices(atoms, freeze_backbone=True, freeze_caps=True)
    assert isinstance(freeze, list)
    assert 1 in freeze and 2 in freeze and 4 in freeze and 5 in freeze and 6 in freeze
    assert int(meta.get("freeze_count", 0)) >= 5


def test_cluster_toy_pipeline_smoke():
    # Toy 2-residue active-site cluster with one substrate-like atom pair.
    cluster_atoms = [
        {"index": 1, "name": "N", "element": "N", "resn": "SER", "chain": "A", "resi": "10", "x": -1.2, "y": 0.0, "z": 0.0},
        {"index": 2, "name": "CA", "element": "C", "resn": "SER", "chain": "A", "resi": "10", "x": -0.1, "y": 0.0, "z": 0.0},
        {"index": 3, "name": "CB", "element": "C", "resn": "SER", "chain": "A", "resi": "10", "x": 1.2, "y": 0.0, "z": 0.0},
        {"index": 4, "name": "OG", "element": "O", "resn": "SER", "chain": "A", "resi": "10", "x": 2.2, "y": 0.0, "z": 0.0},
        {"index": 5, "name": "N", "element": "N", "resn": "HIS", "chain": "A", "resi": "57", "x": 0.0, "y": 2.0, "z": 0.0},
        {"index": 6, "name": "CA", "element": "C", "resn": "HIS", "chain": "A", "resi": "57", "x": 1.1, "y": 2.0, "z": 0.0},
        {"index": 7, "name": "NE2", "element": "N", "resn": "HIS", "chain": "A", "resi": "57", "x": 2.2, "y": 2.0, "z": 0.0},
        {"index": 8, "name": "C1", "element": "C", "resn": "LIG", "chain": "L", "resi": "1", "x": 4.6, "y": 0.0, "z": 0.0},
        {"index": 9, "name": "O1", "element": "O", "resn": "LIG", "chain": "L", "resi": "1", "x": 5.8, "y": 0.0, "z": 0.0},
    ]
    freeze, meta = build_cluster_freeze_indices(cluster_atoms, freeze_backbone=True, freeze_caps=True)
    assert len(freeze) >= 4
    assert int(meta.get("boundary_residue_count", 0)) >= 1
    form_pairs = parse_bond_change_pairs("4-8")
    break_pairs = parse_bond_change_pairs("8-9")
    assert form_pairs == [(4, 8)]
    assert break_pairs == [(8, 9)]
    d_form_before = _dist(cluster_atoms[3], cluster_atoms[7])  # 4-8
    d_break_before = _dist(cluster_atoms[7], cluster_atoms[8])  # 8-9
    prod_atoms, note = adjust_atoms_for_bond_changes(
        cluster_atoms,
        form_pairs=form_pairs,
        break_pairs=break_pairs,
    )
    assert isinstance(note, str) and "form=1" in note and "break=1" in note
    assert len(prod_atoms) == len(cluster_atoms)
    d_form_after = _dist(prod_atoms[3], prod_atoms[7])
    d_break_after = _dist(prod_atoms[7], prod_atoms[8])
    # Toy expected behavior: forming pair gets closer, breaking pair moves apart.
    assert d_form_after < d_form_before
    assert d_break_after > d_break_before
