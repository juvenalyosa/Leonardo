"""
"About & licenses" window: Leonardo's identity and license, and the Open-Source PyMOL copyright and
trademark notices. PyMOL's license asks for its copyright notice in all copies and in the supporting
documentation, and for derivative software to be plainly distinguished from Schrodinger's PyMOL products;
its trademark notice allows the slogan "Built using PyMOL(TM)" with the footnote below.
"""

from .config import PLUGIN_BUILD, PLUGIN_VERSION, SUITE_CONTACT_EMAIL, SUITE_NAME, SUITE_SUBTITLE

PYMOL_SLOGAN = "Built using PyMOL™"
PYMOL_TRADEMARK_NOTE = "PyMOL is a trademark of Schrodinger, LLC."
INDEPENDENCE_NOTE = (
    "Leonardo is an independent open-source project (MIT license). It is a plugin that runs on "
    "Open-Source PyMOL; it is not a PyMOL product distributed by Schrodinger, LLC."
)
# verbatim from the Open-Source PyMOL distribution (conda-forge pymol-open-source, LICENSE.txt)
PYMOL_COPYRIGHT_NOTICE = """\
Open-Source PyMOL Copyright Notice
==================================

The Open-Source PyMOL source code is copyrighted, but you can freely
use and copy it as long as you don't change or remove any of the
Copyright notices. The Open-Source PyMOL product is made available
under the following open-source license terms:

----------------------------------------------------------------------
Open-Source PyMOL is Copyright (C) Schrodinger, LLC.

All Rights Reserved

Permission to use, copy, modify, distribute, and distribute modified
versions of this software and its built-in documentation for any
purpose and without fee is hereby granted, provided that the above
copyright notice appears in all copies and that both the copyright
notice and this permission notice appear in supporting documentation,
and that the name of Schrodinger, LLC not be used in advertising or
publicity pertaining to distribution of the software without specific,
written prior permission.

SCHRODINGER, LLC DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN
NO EVENT SHALL SCHRODINGER, LLC BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE
USE OR PERFORMANCE OF THIS SOFTWARE.
----------------------------------------------------------------------

PyMOL Trademark Notice
======================

PyMOL(TM) is a trademark of Schrodinger, LLC. Derivative
software which contains PyMOL source code must be plainly
distinguished from any and all PyMOL products distributed by Schrodinger,
LLC in all publicity, advertising, and documentation.

The slogans, "Includes PyMOL(TM).", "Based on PyMOL(TM) technology.",
"Contains PyMOL(TM) source code.", and "Built using PyMOL(TM).", may
be used in advertising, publicity, and documentation of derivative
software provided that the notice, "PyMOL is a trademark of Schrodinger,
LLC.", is included in a footnote or at the end of the
document.

All other endorsements employing the PyMOL trademark require specific,
written prior permission.
"""


def about_html():
    """Rich text shown in the About window."""
    import html

    return (
        f"<h2 style='margin-bottom:0'>{SUITE_NAME}</h2>"
        f"<p style='margin-top:2px'>{SUITE_SUBTITLE} &middot; version {PLUGIN_VERSION} ({PLUGIN_BUILD})<br>"
        f"{PYMOL_SLOGAN}.<sup>*</sup></p>"
        f"<p>{html.escape(INDEPENDENCE_NOTE)}</p>"
        "<p>Leonardo is distributed under the MIT license (file <code>LICENSE</code>). The scientific programs it "
        "uses keep their own licenses: see <code>THIRD_PARTY_LICENSES.md</code>.</p>"
        f"<p>Bugs &amp; feedback: <a href='mailto:{SUITE_CONTACT_EMAIL}'>{SUITE_CONTACT_EMAIL}</a></p>"
        "<h3>Open-Source PyMOL</h3>"
        f"<pre style='font-size:11px'>{html.escape(PYMOL_COPYRIGHT_NOTICE)}</pre>"
        f"<p style='font-size:11px'><sup>*</sup> {PYMOL_TRADEMARK_NOTE}</p>"
    )


def show_about(parent=None):
    """Open the About & licenses window (non-modal); returns it."""
    from pymol.Qt import QtWidgets

    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle(f"About {SUITE_NAME}")
    dlg.resize(760, 680)
    dlg.setMinimumSize(640, 560)       # the notices are the point of this window: keep them readable
    dlg.setProperty("leonardoKeepSize", True)
    lay = QtWidgets.QVBoxLayout(dlg)
    text = QtWidgets.QTextBrowser(dlg)
    text.setObjectName("leonardo_about_text")
    text.setOpenExternalLinks(True)
    text.setMinimumHeight(440)         # the themed body puts spare height below its rows, not in them
    text.setHtml(about_html())
    lay.addWidget(text, 1)
    row = QtWidgets.QHBoxLayout()
    row.addStretch(1)
    btn = QtWidgets.QPushButton("Close", dlg)
    btn.clicked.connect(dlg.close)
    row.addWidget(btn)
    lay.addLayout(row)
    dlg.show()
    return dlg
