"""The main window's scroll area reaches the bottom of the window, with Qt 5 and Qt 6.

Under Qt 6 the theme's polish wrapped the main window, which already scrolls, in a second scroll area: the
inner one stayed at its size hint and an empty grey band filled the lower half of the window."""
import importlib.util
import os
import subprocess
import sys

import pytest

pytest.importorskip("pymol.Qt")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROBE = r"""
from leonardo import ui_audit
QtCore, QtWidgets = ui_audit._qt()
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
ui_audit.install_non_blocking_dialogs(QtWidgets)
import code_v2
d = code_v2.ProtLigInteractDialog(); d.resize(1100, 900); d.show()
QtCore.QTimer.singleShot(400, app.quit); app.exec() if hasattr(app, "exec") else app.exec_()
sa = d._main_scroll_area
bottom = sa.mapTo(d, QtCore.QPoint(0, sa.height())).y()
nested = [w.objectName() for w in d.findChildren(QtWidgets.QScrollArea)
          if w.objectName() == "pliResponsiveDialogScroll"]
print("RESULT", d.height(), bottom, len(nested))
"""


@pytest.mark.parametrize("api", ["pyqt5", "pyside6"])
def test_main_scroll_area_reaches_the_bottom_of_the_window(api):
    if importlib.util.find_spec({"pyqt5": "PyQt5", "pyside6": "PySide6"}[api]) is None:
        pytest.skip(f"{api} is not installed")
    env = dict(os.environ, QT_API=api, QT_QPA_PLATFORM="offscreen", LEONARDO_NO_HARDWARE_DETECTION="1",
               PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "legacy"), os.path.join(ROOT, "src")]))
    proc = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=300)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    height, bottom, nested = (int(v) for v in lines[-1].split()[1:])
    assert nested == 0                          # no second scroll area around the main one
    assert height - bottom <= 40                # only the window margin below it
