import math

from vs_runtime import (
    format_virtual_screening_library_status,
    normalize_virtual_screening_controls,
    normalize_virtual_screening_grid,
)


def test_vs_grid_from_pocket_applies_padding_min_edge_and_max_edge():
    payload = normalize_virtual_screening_grid(
        center=(1.0, 2.0, 3.0),
        size=(1.0, 5.0, 200.0),
        padding=2.0,
        min_edge=8.0,
        max_edge=120.0,
    )

    assert payload["ok"] is True
    assert payload["center"] == (1.0, 2.0, 3.0)
    assert payload["size"] == (8.0, 9.0, 120.0)
    assert payload["padding"] == 2.0
    assert payload["min_edge"] == 8.0


def test_vs_controls_clip_numeric_values_and_fallback_bad_grid():
    payload = normalize_virtual_screening_controls(
        {
            "target_ph": 99.0,
            "max_ligs": 0,
            "center": (float("nan"), 8.0, 9.0),
            "size": (float("inf"), 0.1, 2.0),
            "exhaustiveness": 999,
            "timeout_lig": 1,
            "grid_min_edge": 8.0,
        },
        pocket={"center": (4.0, 5.0, 6.0), "size": (2.0, 3.0, 4.0)},
    )

    assert payload["ok"] is True
    assert payload["target_ph"] == 14.0
    assert payload["max_ligs"] == 1
    assert payload["center"] == (4.0, 5.0, 6.0)
    assert payload["size"] == (8.0, 8.0, 8.0)
    assert payload["exhaustiveness"] == 256
    assert payload["timeout_lig"] == 30
    assert set(payload["warnings"]) == {"center_fallback", "size_fallback"}
    assert all(math.isfinite(v) for v in payload["center"] + payload["size"])


def test_vs_library_status_has_preview_and_truncation_context():
    txt = format_virtual_screening_library_status(
        "/tmp/library.sdf",
        {"loaded": 10, "skipped": 2, "total": 13, "limit": 10, "truncated": True, "reason": "ok"},
        preview=True,
    )

    assert txt.startswith("Library preview: library.sdf")
    assert "valid=10" in txt
    assert "skipped=2" in txt
    assert "more ligands not parsed" in txt

    empty = format_virtual_screening_library_status("", {"loaded": 0, "reason": "missing_sdf"})
    assert empty == "Library: not selected | no valid ligands (missing_sdf)"
