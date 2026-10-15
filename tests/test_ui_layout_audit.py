"""Every Leonardo window passes the layout audit: no clipped or elided text, no squeezed fields, no
overlapping widgets, no horizontal scrolling while the screen has room (leonardo.ui_audit)."""
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("pymol.Qt")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREEN = {"screens": [{"name": "virtual", "x": 0, "y": 0, "width": 1920, "height": 1200, "logicalDpi": 96,
                       "logicalBaseDpi": 96, "dpr": 1}]}


def test_all_windows_pass_the_layout_audit(tmp_path):
    from pymol.Qt import PYQT_NAME

    if not PYQT_NAME.endswith("6"):
        pytest.skip("the offscreen screen size is configurable from Qt 6 on (Qt 5 is fixed at 800x600)")
    cfg = tmp_path / "screen.json"
    cfg.write_text(json.dumps(SCREEN))
    env = dict(os.environ, QT_QPA_PLATFORM=f"offscreen:configfile={cfg}", LEONARDO_NO_HARDWARE_DETECTION="1",
               PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "legacy"), os.path.join(ROOT, "src")]))
    out = tmp_path / "audit"
    proc = subprocess.run([sys.executable, "-m", "leonardo.ui_audit", str(out)], env=env, capture_output=True,
                          text=True, timeout=1500)
    report = json.load(open(out / "issues.json"))
    assert len(report) >= 50, proc.stdout[-2000:] + proc.stderr[-2000:]     # every workflow window was opened
    problems = {page: items for page, items in report.items() if items}
    assert not problems, json.dumps(problems, indent=1)[:4000]
