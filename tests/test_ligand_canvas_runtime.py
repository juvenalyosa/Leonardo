import pytest

from ligand_canvas_runtime import atom_display_label
from ligand_canvas_runtime import bond_canvas_segments
from ligand_canvas_runtime import bond_line_offsets
from ligand_canvas_runtime import build_hydrogen_marker_layout
from ligand_canvas_runtime import canvas_to_model_point
from ligand_canvas_runtime import find_atom_at_canvas_point
from ligand_canvas_runtime import find_bond_at_canvas_point
from ligand_canvas_runtime import find_hydrogen_marker_at_canvas_point
from ligand_canvas_runtime import layout_atom_positions


def test_layout_atom_positions_fits_atoms_into_widget_bounds():
    layout = layout_atom_positions({0: (-2.0, -1.0), 1: (2.0, 3.0)}, 400, 300, padding=20.0)

    p0 = layout["points"][0]
    p1 = layout["points"][1]
    for px, py in (p0, p1):
        assert 0.0 <= px <= 400.0
        assert 0.0 <= py <= 300.0
    assert p0[0] < p1[0]
    assert p0[1] > p1[1]


def test_canvas_to_model_point_roundtrips_layout_centered_coordinates():
    layout = layout_atom_positions({0: (-1.0, 0.0), 1: (1.0, 0.0)}, 500, 300, padding=30.0)
    px, py = layout["points"][0]

    x, y = canvas_to_model_point(layout, px, py)

    assert abs(x + 1.0) < 1e-6
    assert abs(y - 0.0) < 1e-6


def test_find_atom_at_canvas_point_picks_nearest_atom_within_radius():
    mapped = {
        0: (100.0, 100.0),
        1: (160.0, 100.0),
    }

    assert find_atom_at_canvas_point(mapped, 107.0, 103.0, hit_radius=18.0) == 0
    assert find_atom_at_canvas_point(mapped, 154.0, 101.0, hit_radius=18.0) == 1
    assert find_atom_at_canvas_point(mapped, 220.0, 220.0, hit_radius=18.0) is None


def test_atom_display_label_includes_hydrogen_counts_and_index():
    rdkit = pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    c0 = mol.GetAtomWithIdx(0)
    o2 = mol.GetAtomWithIdx(2)

    assert atom_display_label(c0, show_indices=False) == "CH3"
    assert atom_display_label(o2, show_indices=False) == "OH"
    assert atom_display_label(c0, show_indices=True) == "CH3\n0"


def test_atom_display_label_formats_charged_hydrogenated_atoms():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.MolFromSmiles("[NH4+]")
    atom = mol.GetAtomWithIdx(0)

    assert atom_display_label(atom, show_indices=False) == "NH4+"


def test_atom_display_label_can_hide_aggregated_hydrogens():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CO")
    atom = mol.GetAtomWithIdx(0)

    assert atom_display_label(atom, show_indices=False, include_hydrogens=False) == "C"


def test_hydrogen_marker_layout_places_visible_hydrogen_badges():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CO")
    markers = build_hydrogen_marker_layout(
        mol,
        {
            0: (100.0, 100.0),
            1: (160.0, 100.0),
        },
        marker_offset=20.0,
    )

    assert 0 in markers
    assert markers[0]["label"] == "H3"
    hx, hy = markers[0]["center"]
    assert hx < 100.0
    assert abs(hy - 100.0) < 20.0


def test_find_hydrogen_marker_at_canvas_point_hits_marker_center():
    markers = {
        3: {"center": (122.0, 88.0), "count": 2, "label": "H2"},
    }

    assert find_hydrogen_marker_at_canvas_point(markers, 123.0, 87.0, hit_radius=10.0) == 3
    assert find_hydrogen_marker_at_canvas_point(markers, 150.0, 120.0, hit_radius=10.0) is None


def test_bond_line_offsets_reports_parallel_lines_for_double_and_triple():
    offsets2, dashed2 = bond_line_offsets("DOUBLE")
    offsets3, dashed3 = bond_line_offsets("TRIPLE")

    assert offsets2 == [-0.8, 0.8]
    assert dashed2 is False
    assert offsets3 == [-1.0, 0.0, 1.0]
    assert dashed3 is False


def test_bond_canvas_segments_shortens_and_splits_multiple_bonds():
    segs2, dashed2 = bond_canvas_segments((10.0, 20.0), (90.0, 20.0), "DOUBLE")
    segs3, dashed3 = bond_canvas_segments((10.0, 20.0), (90.0, 20.0), "TRIPLE")

    assert dashed2 is False
    assert len(segs2) == 2
    assert all(seg[0] > 10.0 and seg[2] < 90.0 for seg in segs2)
    assert dashed3 is False
    assert len(segs3) == 3


def test_find_bond_at_canvas_point_hits_double_bond():
    rdkit = pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.MolFromSmiles("C=C")
    points = {
        0: (100.0, 100.0),
        1: (180.0, 100.0),
    }

    hit = find_bond_at_canvas_point(mol, points, 140.0, 104.0, hit_radius=8.0)

    assert hit == (0, 1)
