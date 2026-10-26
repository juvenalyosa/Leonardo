"""
Layout audit of Leonardo's windows: clipped text, elided tabs, squeezed controls and overlapping widgets.

Used by the test suite (every workflow window must pass) and by ``python -m leonardo.ui_audit OUTDIR`` to
capture every window and tab for a visual review. Works with PyQt5 and PySide6.
"""
from __future__ import annotations

import json
import os
import sys

TEXT_PAD = {"QPushButton": 22, "QToolButton": 14, "QCheckBox": 24, "QRadioButton": 24}


def _qt():
    from pymol.Qt import QtCore, QtWidgets

    return QtCore, QtWidgets


def _text(w):
    for attr in ("text", "title"):
        fn = getattr(w, attr, None)
        if callable(fn):
            try:
                t = fn()
            except Exception:
                continue
            if isinstance(t, str) and t.strip():
                return t.replace("&&", "\0").replace("&", "").replace("\0", "&")
    return ""


def _visible_on_page(w, root):
    return w.isVisibleTo(root) and w.width() > 2 and w.height() > 2


def audit_widget(root, size_checks=True):
    """Problems inside ``root`` (a window or a tab page) as it is laid out now."""
    QtCore, QtWidgets = _qt()
    issues = []
    for w in root.findChildren(QtWidgets.QWidget):
        if not _visible_on_page(w, root):
            continue
        cls = type(w).__name__
        fm = w.fontMetrics()
        where = w.objectName() or cls
        if isinstance(w, (QtWidgets.QPushButton, QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
            t = _text(w)
            base = next((v for k, v in TEXT_PAD.items() if isinstance(w, getattr(QtWidgets, k))), 20)
            icon = (w.iconSize().width() + 6) if not w.icon().isNull() else 0
            need = fm.horizontalAdvance(t) + base + icon
            if t and "\n" not in t and need > w.width() + 2:
                issues.append({"kind": "clipped_text", "widget": where, "text": t[:70], "need": need,
                               "have": w.width()})
        elif isinstance(w, QtWidgets.QLabel) and not w.wordWrap() and w.pixmap() is None \
                and w.textFormat() != QtCore.Qt.RichText:
            t = _text(w)
            if t and "<" not in t:
                widest = max(fm.horizontalAdvance(line) for line in t.split("\n"))
                if widest > w.width() + 3:
                    issues.append({"kind": "clipped_label", "widget": where, "text": t[:70], "need": widest,
                                   "have": w.width()})
        elif isinstance(w, QtWidgets.QGroupBox):
            t = w.title()
            if t and fm.horizontalAdvance(t) + 24 > w.width():
                issues.append({"kind": "clipped_title", "widget": where, "text": t[:70]})
        if isinstance(w, QtWidgets.QTabBar):
            for i in range(w.count()):
                t = w.tabText(i).replace("&", "")
                rect = w.tabRect(i)
                if t and rect.width() > 0 and fm.horizontalAdvance(t) + 12 > rect.width():
                    issues.append({"kind": "elided_tab", "widget": where, "text": t, "need":
                                   fm.horizontalAdvance(t) + 12, "have": rect.width()})
        if isinstance(w, (QtWidgets.QPushButton, QtWidgets.QComboBox, QtWidgets.QLineEdit, QtWidgets.QSpinBox,
                          QtWidgets.QDoubleSpinBox)):
            msh = w.minimumSizeHint()
            if isinstance(w, QtWidgets.QAbstractSpinBox):
                msh = w.sizeHint()          # Qt 6 styled spin boxes report a minimum 5 px above their size hint
            if msh.isValid() and (w.height() + 2 < msh.height()):
                issues.append({"kind": "squeezed", "widget": where, "text": _text(w)[:50],
                               "size": [w.width(), w.height()], "min": [msh.width(), msh.height()]})
    # a horizontal scroll bar while the screen still has room: the window should have been wider
    try:
        screen_w = root.window().screen().availableGeometry().width()
    except Exception:
        screen_w = 0
    for area in root.findChildren(QtWidgets.QScrollArea):
        bar = area.horizontalScrollBar()
        if area.isVisibleTo(root) and bar.isVisible() and bar.maximum() > 0 and screen_w and \
                root.window().width() < 0.9 * screen_w:
            issues.append({"kind": "needs_width", "widget": area.objectName() or "QScrollArea",
                           "overflow_px": bar.maximum(), "window_w": root.window().width()})
    try:
        screen_h = root.window().screen().availableGeometry().height()
    except Exception:
        screen_h = 0
    for area in root.findChildren(QtWidgets.QScrollArea):
        bar = area.verticalScrollBar()
        if size_checks and area.height() >= 0.3 * max(1, root.window().height()) and area.isVisibleTo(root) and \
                bar.isVisible() and bar.maximum() > 8 and screen_h and root.window().height() < 0.85 * screen_h:
            issues.append({"kind": "needs_height", "widget": area.objectName(), "overflow_px": bar.maximum(),
                           "window_h": root.window().height()})
    # widgets managed by the same layout must not overlap
    for host in [root] + root.findChildren(QtWidgets.QWidget):
        lay = host.layout()
        if lay is None or not _visible_on_page(host, root) and host is not root:
            continue
        kids = []
        for i in range(lay.count()):
            item = lay.itemAt(i)
            wid = item.widget() if item is not None else None
            if wid is not None and _visible_on_page(wid, root):
                kids.append((wid, item))
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                # the layout item's rect, not the widget's: macOS styles draw check boxes and buttons a few pixels
                # past the cell the layout gave them (layout-item margins), which is not an overlap
                inter = kids[i][1].geometry().intersected(kids[j][1].geometry())
                if inter.width() > 2 and inter.height() > 2:
                    issues.append({"kind": "overlap", "widget": host.objectName() or type(host).__name__,
                                   "a": f"{type(kids[i][0]).__name__}:{_text(kids[i][0])[:30]}",
                                   "b": f"{type(kids[j][0]).__name__}:{_text(kids[j][0])[:30]}"})
    return issues


def walk_tabs(window, app, visit, prefix=""):
    """Call visit(name, page) for the window and for every tab page, nested tabs included."""
    QtCore, QtWidgets = _qt()
    visit(prefix or window.windowTitle() or type(window).__name__, window)
    seen = set()

    def recurse(widget, path):
        for tabs in widget.findChildren(QtWidgets.QTabWidget):
            if id(tabs) in seen or not tabs.isVisibleTo(window):
                continue
            seen.add(id(tabs))
            for i in range(tabs.count()):
                tabs.setCurrentIndex(i)
                app.processEvents()
                name = f"{path} › {tabs.tabText(i).replace('&', '').strip() or i}"
                visit(name, window)
                recurse(tabs.widget(i), name)

    recurse(window, prefix or window.windowTitle() or "window")


def open_workflow_windows(dlg, app, rounds=3):
    """Open every window reachable from Leonardo's main dialog through "...". buttons (windows that create
    their buttons when they open are scanned again, up to ``rounds`` levels)."""
    QtCore, QtWidgets = _qt()
    before = {id(w) for w in app.topLevelWidgets() if w.isVisible()}
    clicked, opened = set(), []
    skip = ("Open Structure...", "Fetch Structure...", "AI Assistant...")
    for _ in range(int(rounds)):
        pending = [b for b in dlg.findChildren(QtWidgets.QPushButton)
                   if id(b) not in clicked and b.isVisible() and b.isEnabled()
                   and b.text().strip().endswith("...") and b.text().strip() not in skip]
        if not pending:
            break
        for btn in pending:
            clicked.add(id(btn))
            if not btn.isVisible():
                continue
            print("opening:", btn.text().strip(), flush=True)
            try:
                btn.click()
            except Exception as exc:
                opened.append((btn.text(), None, f"error: {exc}"))
                continue
            app.processEvents()
            new = [w for w in app.topLevelWidgets() if w.isVisible() and id(w) not in before and w is not dlg]
            for w in new:
                before.add(id(w))
                opened.append((btn.text().strip().rstrip("."), w, ""))
    return opened


def install_non_blocking_dialogs(QtWidgets):
    """Audit mode: windows that would block (modal exec, message boxes, file/input pickers) return at once."""
    def show_instead(self, *a, **k):
        if isinstance(self, QtWidgets.QMessageBox):
            return 0
        self.show()
        return 0

    for cls in (QtWidgets.QDialog, QtWidgets.QMessageBox, QtWidgets.QFileDialog, QtWidgets.QInputDialog):
        for meth in ("exec_", "exec"):
            if hasattr(cls, meth):
                setattr(cls, meth, show_instead)
    QtWidgets.QMenu.exec_ = QtWidgets.QMenu.exec = lambda self, *a, **k: None
    for name in ("information", "warning", "critical", "question", "about"):
        setattr(QtWidgets.QMessageBox, name, staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok))
    for name in ("getOpenFileName", "getSaveFileName", "getOpenFileNames"):
        setattr(QtWidgets.QFileDialog, name, staticmethod(lambda *a, **k: ([] if "Names" in name else "", "")))
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
    for name in ("getText", "getItem", "getInt", "getDouble", "getMultiLineText"):
        setattr(QtWidgets.QInputDialog, name, staticmethod(lambda *a, **k: ("", False)))


def main(out_dir, width=1440, height=900):
    QtCore, QtWidgets = _qt()
    os.makedirs(out_dir, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # no blocking: modal dialogs open like windows, message boxes answer at once
    install_non_blocking_dialogs(QtWidgets)
    import code_v2

    dlg = code_v2.ProtLigInteractDialog()
    dlg.resize(width, height)
    dlg.show()
    app.processEvents()
    report = {}
    count = [0]

    def visit_for(window):
        def visit(name, page):
            import time as _t

            end = _t.time() + 0.25
            while _t.time() < end:
                app.processEvents()
            count[0] += 1
            safe = "".join(c if c.isalnum() else "_" for c in name)[:70]
            window.grab().save(os.path.join(out_dir, f"{count[0]:03d}_{safe}.png"))
            # the main window keeps the size the user gives it: no "should be taller" check there
            report[name] = audit_widget(window, size_checks=window is not dlg)
        return visit

    walk_tabs(dlg, app, visit_for(dlg), "Leonardo")
    for title, win, err in open_workflow_windows(dlg, app):
        if win is None:
            report[title] = [{"kind": "error", "text": err}]
            continue
        walk_tabs(win, app, visit_for(win), title)
    json.dump(report, open(os.path.join(out_dir, "issues.json"), "w"), indent=1)
    total = 0
    for name, items in report.items():
        kinds = {}
        for it in items:
            kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
        total += len(items)
        if items:
            print(f"{name[:60]:60s} {kinds}")
    print("windows/pages:", len(report), "issues:", total)
    return report


if __name__ == "__main__":
    main(sys.argv[1], *(int(v) for v in sys.argv[2:4]))
