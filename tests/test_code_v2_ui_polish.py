import code_v2


def _build_dummy_dialog():
    class DummyDialog:
        pass

    for name in (
        "_compact_ui_text",
        "_neutral_dialog_stylesheet",
        "_dock_ui_path_label",
        "_format_docking_table_float",
        "_normalize_task_monitor_storage",
        "_on_task_monitor_tick",
    ):
        setattr(DummyDialog, name, getattr(code_v2.ProtLigInteractDialog, name))
    return DummyDialog()


def test_ui_polish_rules_keep_dense_dialog_text_compact_and_neutral():
    dlg = _build_dummy_dialog()

    assert dlg._compact_ui_text("Open Ligand Source & Prep...") == "Ligand Prep..."
    assert dlg._compact_ui_text("Run minimization-only in main workflow", widget_kind="checkbox") == "Min-only in workflow"

    qss = dlg._neutral_dialog_stylesheet()
    assert "LEONARDO_NEUTRAL_POLISH_V2" in qss
    assert "background-color: #F4F5F7" in qss
    assert "margin-top: 30px" in qss
    assert "padding: 22px 12px 12px 12px" in qss
    assert "QCheckBox::indicator" in qss
    assert "spacing: 12px" in qss


def test_docking_ui_formatters_hide_invalid_rmsd_and_compact_paths():
    dlg = _build_dummy_dialog()

    assert dlg._format_docking_table_float(-7.12345) == "-7.123"
    assert dlg._format_docking_table_float(-1.0, nonnegative=True) == "n/a"
    assert dlg._format_docking_table_float(float("nan"), nonnegative=True) == "n/a"

    compact = dlg._dock_ui_path_label("/very/long/path/to/a/project/with/results/docked_poses.pdbqt", max_chars=34)
    assert compact.startswith("...")
    assert compact.endswith("docked_poses.pdbqt")


def test_task_monitor_storage_normalizes_legacy_list_state():
    dlg = _build_dummy_dialog()
    dlg._task_monitor_table = None
    dlg._task_monitor_rows = []
    dlg._task_monitor_state = [
        {
            "job_id": "J0007",
            "row": 3,
            "tool_label": "OpenMM",
            "status": "Running",
            "job_dir": "/tmp/job",
        }
    ]
    dlg._task_monitor_counter = 0

    dlg._normalize_task_monitor_storage()

    assert isinstance(dlg._task_monitor_state, dict)
    assert isinstance(dlg._task_monitor_rows, dict)
    assert "J0007" in dlg._task_monitor_state
    assert dlg._task_monitor_rows["J0007"] == 3
    assert dlg._task_monitor_state["J0007"]["task_label"] == ""
    assert dlg._task_monitor_counter == 7


def test_task_monitor_tick_accepts_empty_dict_state():
    class DummyViewport:
        def __init__(self):
            self.updated = False

        def update(self):
            self.updated = True

    class DummyTable:
        def __init__(self):
            self._viewport = DummyViewport()

        def viewport(self):
            return self._viewport

    dlg = _build_dummy_dialog()
    table = DummyTable()
    dlg._task_monitor_table = table
    dlg._task_monitor_rows = {}
    dlg._task_monitor_state = {}
    dlg._task_monitor_timer = None

    dlg._on_task_monitor_tick()

    assert table._viewport.updated is True
