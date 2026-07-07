import math

from dock_runtime import format_docking_grid_summary
from dock_runtime import normalize_docking_grid
from dock_runtime import parse_fpocket_info_text
from dock_runtime import parse_pdb_atom_coordinates_from_lines
from dock_runtime import pocket_geometry_from_coordinates


def test_parse_fpocket_info_text_accepts_common_separators():
    text = """
Pocket 1 :
    Score : 31.25
    Druggability Score = 0.72
    Number of Alpha Spheres      42
Pocket 2
    Pocket Score: -3.5
    Comment = shallow
"""

    pockets = parse_fpocket_info_text(text)

    assert [p["id"] for p in pockets] == [1, 2]
    assert math.isclose(pockets[0]["score"], 31.25)
    assert math.isclose(pockets[0]["druggability_score"], 0.72)
    assert math.isclose(pockets[0]["number_of_alpha_spheres"], 42.0)
    assert math.isclose(pockets[1]["pocket_score"], -3.5)
    assert pockets[1]["comment"] == "shallow"


def test_pocket_geometry_from_pdb_lines_supports_fixed_and_split_columns():
    lines = [
        "ATOM      1  CA  ALA A   1       0.000   1.000   2.000  1.00 10.00           C",
        "HETATM    2  O   HOH A   2       4.000   5.000   6.000  1.00 10.00           O",
        "ATOM 3 N GLY A 3 8.000 9.000 10.000 1.00 10.00 N",
    ]

    coords = parse_pdb_atom_coordinates_from_lines(lines)
    center, size, natoms = pocket_geometry_from_coordinates(coords)

    assert natoms == 3
    assert center == [4.0, 5.0, 6.0]
    assert size == [8.0, 8.0, 8.0]


def test_normalize_docking_grid_adds_padding_and_minimum_edge():
    payload = normalize_docking_grid(
        center=[1.0, 2.0, 3.0],
        size=[1.0, 4.0, 30.0],
        padding=2.0,
        min_edge=8.0,
        max_edge=40.0,
    )

    assert payload["ok"] is True
    assert payload["center"] == [1.0, 2.0, 3.0]
    assert payload["size"] == [8.0, 8.0, 34.0]


def test_format_docking_grid_summary_is_compact_and_numeric():
    text = format_docking_grid_summary([1, 2, 3], [10, 11, 12], source="pocket", pocket_id=5)

    assert "center=(1.00, 2.00, 3.00)" in text
    assert "size=(10.00, 11.00, 12.00)" in text
    assert "source=pocket" in text
    assert "pocket=5" in text
