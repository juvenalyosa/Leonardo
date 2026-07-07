"""Source-tree import shim for the Leonardo PyMOL plugin.

The clean working tree is named ``Leonardo_clean`` during development, while
the installable plugin package is historically imported as ``ProtLigInteract``.
This module keeps backend tests and direct source-tree imports independent of
the outer directory name without importing PyMOL at module import time.
"""

import os
import sys
from importlib import import_module


__version__ = "2.x"
PLUGIN_MENU_NAME = "Leonardo"
_dialog = None
_plugin_module = None


def _load_plugin_module():
    global __version__, _plugin_module
    if _plugin_module is None:
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
        _plugin_module = import_module("code_v2")
        __version__ = getattr(_plugin_module, "PLUGIN_BUILD", __version__)
    return _plugin_module


def __init_plugin__(app=None):
    """PyMOL plugin entry point, loaded lazily so backend imports stay light."""
    global _dialog
    try:
        from pymol.plugins import addmenuitemqt
    except Exception as exc:
        raise RuntimeError("PyMOL with plugin support is required to initialize Leonardo.") from exc

    plugin_module = _load_plugin_module()
    ProtLigInteractDialog = plugin_module.ProtLigInteractDialog

    def run_plugin_gui():
        global _dialog
        if _dialog is not None:
            try:
                _dialog.close()
                _dialog.deleteLater()
            except Exception:
                pass
        _dialog = ProtLigInteractDialog()
        _dialog.show()

    addmenuitemqt(PLUGIN_MENU_NAME, run_plugin_gui)
