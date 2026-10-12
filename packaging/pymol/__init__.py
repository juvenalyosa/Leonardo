"""Lazy package entrypoint for the PyMOL plugin."""

import os
import sys
from importlib import import_module


__version__ = "0.2.0a8"
__build__ = "2026-10-12-r1"
PLUGIN_MENU_NAME = "Leonardo"
_dialog = None
_plugin_module = None


def _load_plugin_module():
    global __version__, __build__, _plugin_module
    if _plugin_module is None:
        # Ensure the plugin directory is on sys.path so that bare-name
        # fallback imports in code_v2.py (the except branch) can locate
        # sibling modules regardless of how PyMOL sets up the package context.
        _pkg_dir = os.path.dirname(os.path.abspath(__file__))
        if _pkg_dir not in sys.path:
            sys.path.insert(0, _pkg_dir)
        _plugin_module = import_module(".code_v2", __name__)
        __version__ = getattr(_plugin_module, "PLUGIN_VERSION", __version__)
        __build__ = getattr(_plugin_module, "PLUGIN_BUILD", __build__)
        print(f"{PLUGIN_MENU_NAME}: package loaded from {__file__}")
        print(f"{PLUGIN_MENU_NAME}: code loaded from {getattr(_plugin_module, '__file__', 'unknown')}")
        print(f"{PLUGIN_MENU_NAME}: package version/build {__version__} ({__build__})")
    return _plugin_module


def __init_plugin__(app=None):
    """
    PyMOL entry point: adds the Leonardo menu item
    that launches our dialog.
    """
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
