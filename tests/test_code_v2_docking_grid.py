import math

import code_v2


class _Spin:
    def __init__(self, value=0.0):
        self._value = float(value)
        self._blocked = False

    def value(self):
        return float(self._value)

    def setValue(self, value):
        self._value = float(value)

    def blockSignals(self, blocked):
        self._blocked = bool(blocked)


def _build_dummy_dialog():
    class DummyDialog:
        pass

    for name in (
        "_dock_grid_min_edge_value",
        "_dock_grid_padding_value",
        "_dock_grid_payload_from_controls",
        "_dock_resolve_run_grid",
        "_sync_docking_grid_from_pocket",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return DummyDialog()


def _attach_grid_spins(dlg, center, size):
    dlg.spin_cx = _Spin(center[0])
    dlg.spin_cy = _Spin(center[1])
    dlg.spin_cz = _Spin(center[2])
    dlg.spin_sx = _Spin(size[0])
    dlg.spin_sy = _Spin(size[1])
    dlg.spin_sz = _Spin(size[2])


def test_dock_resolve_run_grid_prefers_current_edited_controls_over_pocket_defaults():
    dlg = _build_dummy_dialog()
    _attach_grid_spins(dlg, center=(1.0, 2.0, 3.0), size=(11.0, 12.0, 13.0))
    pocket = {
        "center": [10.0, 20.0, 30.0],
        "size": [21.0, 22.0, 23.0],
    }

    payload = dlg._dock_resolve_run_grid(pocket)

    assert payload["ok"] is True
    assert payload["source"] == "controls"
    assert payload["center"] == [1.0, 2.0, 3.0]
    assert payload["size"] == [11.0, 12.0, 13.0]


def test_dock_resolve_run_grid_falls_back_to_pocket_when_controls_are_invalid():
    dlg = _build_dummy_dialog()
    _attach_grid_spins(dlg, center=(math.nan, 2.0, 3.0), size=(11.0, 12.0, 13.0))
    pocket = {
        "center": [10.0, 20.0, 30.0],
        "size": [21.0, 22.0, 23.0],
    }

    payload = dlg._dock_resolve_run_grid(pocket)

    assert payload["ok"] is True
    assert payload["source"] == "pocket"
    assert payload["center"] == [10.0, 20.0, 30.0]
    assert payload["size"] == [21.0, 22.0, 23.0]
    assert dlg.spin_cx.value() == 10.0
    assert dlg.spin_sx.value() == 21.0


def test_sync_docking_grid_applies_user_visible_padding_and_min_edge():
    dlg = _build_dummy_dialog()
    _attach_grid_spins(dlg, center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
    dlg.spin_dock_grid_padding = _Spin(2.0)
    dlg.spin_dock_grid_min_edge = _Spin(8.0)
    pocket = {
        "id": 7,
        "center": [1.0, 2.0, 3.0],
        "size": [1.5, 10.0, 0.2],
    }

    ok, center, size = dlg._sync_docking_grid_from_pocket(pocket)

    assert ok is True
    assert center == [1.0, 2.0, 3.0]
    assert size == [8.0, 14.0, 8.0]
    assert dlg.spin_sx.value() == 8.0
    assert dlg.spin_sy.value() == 14.0
