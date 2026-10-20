"""Source-tree import shim for the Leonardo PyMOL plugin.

The clean working tree is named ``Leonardo_clean`` during development, while
the installable plugin package is historically imported as ``ProtLigInteract``.
This module keeps backend tests and direct source-tree imports independent of
the outer directory name without importing PyMOL at module import time.
"""

import os
import sys
from importlib import import_module

__version__ = "0.2.0a14"
__build__ = "2026-10-20-r1"
PLUGIN_MENU_NAME = "Leonardo"
_dialog = None
_plugin_module = None


def _load_plugin_module():
    global __version__, __build__, _plugin_module
    if _plugin_module is None:
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)
        _plugin_module = import_module("code_v2")
        __version__ = getattr(_plugin_module, "PLUGIN_VERSION", __version__)
        __build__ = getattr(_plugin_module, "PLUGIN_BUILD", __build__)
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
            _stop_assistant(_dialog)
            try:
                _dialog.close()
                _dialog.deleteLater()
            except Exception:
                pass
        _dialog = ProtLigInteractDialog()
        _dialog.show()

    addmenuitemqt(PLUGIN_MENU_NAME, run_plugin_gui)
    _install_quit_teardown()


def _stop_assistant(dlg):
    """The AI assistant (worker thread, MCP server, chat window) must stop before its dialog goes."""
    try:
        controller = getattr(dlg, "_ai_controller", None)
        if controller is not None:
            controller.shutdown()
            window = getattr(controller, "dock", None)
            if window is not None:
                window.close()
                _delete_now(window)
    except Exception:
        pass


def _delete_now(widget):
    """Destroy a Qt widget (and its children) right away. PyMOL quits with exit() from inside the close
    event: PySide6 then walks every Python-owned widget still alive while the process ends, and a Leonardo
    window still open there crashed the quit (SIGSEGV in Shiboken::BindingManager)."""
    try:
        import shiboken6

        if shiboken6.isValid(widget):
            shiboken6.delete(widget)
        return
    except ImportError:
        pass
    except Exception:
        return
    try:
        widget.deleteLater()
    except Exception:
        pass


def teardown_for_quit():
    """Close Leonardo before PyMOL exits (window close button, 'quit' command, cmd.quit())."""
    global _dialog
    dlg, _dialog = _dialog, None
    if os.environ.get("LEONARDO_DEBUG_QUIT"):
        import threading
        print("LEONARDO_QUIT teardown", threading.current_thread().name, dlg is not None, flush=True)
    if dlg is None:
        return
    _stop_assistant(dlg)
    try:
        dlg.close()
    except Exception:
        pass
    _delete_now(dlg)


def _install_quit_teardown():
    """Wrap PyMOL's cmd._quit (every way of quitting goes through it) so Leonardo is closed first."""
    try:
        from pymol import cmd
    except Exception:
        return
    original = getattr(cmd, "_quit", None)
    if original is None or getattr(original, "_leonardo_teardown", False):
        return

    def _quit(*args, **kwargs):
        try:
            teardown_for_quit()
        except Exception:
            pass
        return original(*args, **kwargs)

    _quit._leonardo_teardown = True
    cmd._quit = _quit
