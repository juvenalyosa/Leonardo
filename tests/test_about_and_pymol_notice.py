"""Open-Source PyMOL's copyright and trademark notices travel with Leonardo (its license requires them)."""
import glob
import os

import pytest

from leonardo.about import PYMOL_COPYRIGHT_NOTICE, PYMOL_TRADEMARK_NOTE, about_html

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm(text):
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_notice_is_verbatim_from_the_installed_pymol_license():
    import sys

    candidates = glob.glob(os.path.join(sys.prefix, "..", "..", "pkgs", "pymol-open-source-*", "info", "licenses",
                                        "LICENSE.txt"))
    candidates += glob.glob(os.path.join(sys.prefix, "info", "licenses", "LICENSE.txt"))
    texts = [open(p, encoding="utf-8", errors="ignore").read() for p in candidates]
    texts = [t for t in texts if "Open-Source PyMOL Copyright Notice" in t]
    if not texts:
        pytest.skip("Open-Source PyMOL license file not found in this environment")
    upstream = _norm(texts[0])
    ours = _norm(PYMOL_COPYRIGHT_NOTICE)
    start = upstream.index(ours[0])
    assert upstream[start:start + len(ours)] == ours


def test_third_party_file_readme_and_about_carry_the_notices():
    third = open(os.path.join(REPO, "THIRD_PARTY_LICENSES.md"), encoding="utf-8").read()
    assert all(line in third for line in _norm(PYMOL_COPYRIGHT_NOTICE))
    readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    assert "Built using PyMOL" in readme.replace("built using", "Built using") and PYMOL_TRADEMARK_NOTE in readme
    html = about_html()
    assert "Copyright (C) Schrodinger, LLC" in html and PYMOL_TRADEMARK_NOTE in html and "MIT" in html


def test_header_link_opens_the_about_window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("pymol.Qt").QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import code_v2

    dlg = code_v2.ProtLigInteractDialog()
    labels = [w for w in dlg.findChildren(QtWidgets.QLabel) if "Built using PyMOL" in w.text()]
    assert labels, "no PyMOL credit in the header"
    labels[0].linkActivated.emit("about")
    app.processEvents()
    about = dlg._about_dialog
    text = about.findChild(QtWidgets.QTextBrowser, "leonardo_about_text").toPlainText()
    assert "Open-Source PyMOL is Copyright (C) Schrodinger, LLC." in text
    assert "PyMOL is a trademark of Schrodinger, LLC." in text
    about.close()
    dlg.close()
