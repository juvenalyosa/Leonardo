"""
One look for every Leonardo window, and layout rules that never clip text.

``theme_qss()`` is the single stylesheet of Leonardo's windows (not PyMOL's): light canvas, white cards, one
blue accent, flat tabs with an underline, 30 px fields with drawn spin/combo arrows. ``polish_window(win)``
applies the layout rules: buttons and check boxes get the width their full text needs (no "…"), long labels
wrap without being squeezed, tab bars scroll instead of eliding, the dialog body sits in a scroll area that
respects minimum sizes (rows are never compressed into each other), and the window is sized to its content up
to the screen. Works with PyQt5 and PySide6.
"""
from __future__ import annotations

import os
import re

PALETTE = {
    "canvas": "#F2F4F7",
    "panel": "#FFFFFF",
    "panel_alt": "#F7F8FA",
    "border": "#D8DDE4",
    "border_strong": "#BCC4CF",
    "text": "#1B2430",
    "muted": "#5A6474",
    "faint": "#8A93A1",
    "accent": "#1F5FCC",
    "accent_hover": "#1A52B0",
    "accent_pressed": "#164594",
    "accent_soft": "#E7EFFC",
    "header": "#14223A",
    "header_text": "#FFFFFF",
    "header_muted": "#9EB2CE",
    "danger": "#B42318",
    "selection": "#D4E3FB",
}

FIELD_HEIGHT = 34          # line edits, combo and spin boxes (Qt 6 spin boxes need 35 with their buttons)

# roles a button can declare with setProperty("leonardoRole", ...)
PRIMARY_HINTS = ("run", "start", "launch", "compute", "calculate", "submit", "dock")


def _asset(name):
    try:
        from .config import resource_string

        path = resource_string(os.path.join("ui", name))
    except Exception:
        path = ""
    return path.replace("\\", "/") if path and os.path.isfile(path) else ""


def theme_qss():
    """Stylesheet of Leonardo's windows (applied per window, so PyMOL's own GUI keeps its look)."""
    p = PALETTE
    up, down, check, dot = (_asset(n) for n in ("arrow_up.svg", "arrow_down.svg", "check.svg", "radio_dot.svg"))
    arrows = ""
    if up and down:
        arrows = (
            f"QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({up}); width: 10px; height: 6px; }}"
            f"QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({down}); width: 10px; height: 6px; }}"
            f"QComboBox::down-arrow {{ image: url({down}); width: 10px; height: 6px; }}"
        )
    checks = (f"QCheckBox::indicator:checked {{ image: url({check}); }}" if check else "") + (
        f"QRadioButton::indicator:checked {{ image: url({dot}); }}" if dot else "")
    return (
        "/* LEONARDO_THEME_V3 */"
        f"QDialog, QMainWindow {{ background-color: {p['canvas']}; color: {p['text']};"
        "  font-size: 13px; }"
        f"QWidget#pliResponsiveDialogContent, QScrollArea#pliResponsiveDialogScroll,"
        f"QScrollArea#pliResponsiveDialogScroll > QWidget > QWidget {{ background-color: {p['canvas']}; }}"
        f"QLabel {{ color: {p['text']}; background: transparent; }}"
        # cards
        f"QGroupBox {{ background-color: {p['panel']}; border: 1px solid {p['border']}; border-radius: 8px;"
        "  margin-top: 22px; padding: 10px 10px 8px 10px; font-weight: 600; }"
        "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; top: 2px;"
        f"  padding: 0 4px; color: {p['muted']}; background: transparent; font-size: 12px; font-weight: 600; }}"
        # buttons
        f"QPushButton {{ background-color: {p['panel']}; color: {p['text']}; border: 1px solid {p['border_strong']};"
        "  border-radius: 6px; padding: 5px 14px; min-height: 20px; font-weight: 500; }"
        f"QPushButton:hover {{ background-color: {p['panel_alt']}; border-color: {p['accent']}; }}"
        f"QPushButton:pressed {{ background-color: {p['accent_soft']}; }}"
        f"QPushButton:disabled {{ color: {p['faint']}; background-color: {p['panel_alt']}; border-color: {p['border']}; }}"
        f"QPushButton[leonardoRole=\"primary\"] {{ background-color: {p['accent']}; color: #FFFFFF;"
        f"  border: 1px solid {p['accent']}; font-weight: 600; }}"
        f"QPushButton[leonardoRole=\"primary\"]:hover {{ background-color: {p['accent_hover']}; }}"
        f"QPushButton[leonardoRole=\"primary\"]:pressed {{ background-color: {p['accent_pressed']}; }}"
        f"QPushButton[leonardoRole=\"primary\"]:disabled {{ background-color: {p['border']}; color: {p['faint']};"
        f"  border-color: {p['border']}; }}"
        f"QToolButton {{ border: 1px solid transparent; border-radius: 5px; padding: 3px; }}"
        f"QToolButton:hover {{ background-color: {p['accent_soft']}; border-color: {p['border']}; }}"
        # fields
        f"QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateTimeEdit {{ background-color: {p['panel']};"
        f"  color: {p['text']}; border: 1px solid {p['border_strong']}; border-radius: 5px; padding: 4px 8px;"
        f"  min-height: 24px; selection-background-color: {p['selection']}; selection-color: {p['text']}; }}"
        f"QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {p['accent']}; }}"
        f"QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{"
        f"  background-color: {p['panel_alt']}; color: {p['faint']}; }}"
        f"QLineEdit[readOnly=\"true\"] {{ background-color: {p['panel_alt']}; }}"
        "QSpinBox, QDoubleSpinBox { padding: 2px 22px 2px 8px; }"
        "QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right;"
        f"  width: 20px; border-left: 1px solid {p['border']}; border-top-right-radius: 5px; background: {p['panel_alt']}; }}"
        "QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border;"
        "  subcontrol-position: bottom right; width: 20px;"
        f"  border-left: 1px solid {p['border']}; border-bottom-right-radius: 5px; background: {p['panel_alt']}; }}"
        f"QSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::up-button:hover,"
        f"QDoubleSpinBox::down-button:hover {{ background: {p['accent_soft']}; }}"
        "QComboBox { padding-right: 26px; }"
        f"QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right; width: 24px;"
        f"  border-left: 1px solid {p['border']}; }}"
        f"QComboBox QAbstractItemView {{ background-color: {p['panel']}; border: 1px solid {p['border_strong']};"
        f"  selection-background-color: {p['selection']}; selection-color: {p['text']}; outline: 0; }}"
        f"{arrows}"
        f"QPlainTextEdit, QTextEdit, QTextBrowser {{ background-color: {p['panel']}; border: 1px solid {p['border']};"
        f"  border-radius: 6px; padding: 6px; selection-background-color: {p['selection']}; selection-color: {p['text']}; }}"
        # check boxes / radios
        "QCheckBox, QRadioButton { spacing: 8px; padding: 2px 0; background: transparent; }"
        f"QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {p['border_strong']}; border-radius: 4px;"
        f"  background: {p['panel']}; }}"
        f"QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {p['accent']}; }}"
        f"QCheckBox::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; }}"
        f"QRadioButton::indicator {{ width: 16px; height: 16px; border: 1px solid {p['border_strong']};"
        f"  border-radius: 8px; background: {p['panel']}; }}"
        f"QRadioButton::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; }}"
        f"{checks}"
        # tabs
        f"QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 8px; background: {p['panel']};"
        "  top: -1px; }"
        "QTabBar { qproperty-drawBase: 0; }"
        f"QTabBar::tab {{ background: transparent; color: {p['muted']}; padding: 8px 16px; margin-right: 2px;"
        "  border: none; border-bottom: 2px solid transparent; font-weight: 500; }"
        f"QTabBar::tab:selected {{ color: {p['accent']}; border-bottom: 2px solid {p['accent']}; font-weight: 600; }}"
        f"QTabBar::tab:hover:!selected {{ color: {p['text']}; border-bottom: 2px solid {p['border_strong']}; }}"
        f"QTabBar::scroller {{ width: 28px; }}"
        # tables / lists
        f"QTableView, QTableWidget, QTreeView, QTreeWidget, QListView, QListWidget {{ background-color: {p['panel']};"
        f"  alternate-background-color: {p['panel_alt']}; border: 1px solid {p['border']}; border-radius: 6px;"
        f"  gridline-color: {p['border']}; selection-background-color: {p['selection']}; selection-color: {p['text']}; }}"
        f"QHeaderView::section {{ background-color: {p['panel_alt']}; color: {p['muted']}; padding: 6px 8px;"
        f"  border: none; border-bottom: 1px solid {p['border']}; border-right: 1px solid {p['border']}; font-weight: 600; }}"
        # scroll bars, progress, splitters, tooltips, menus
        "QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }"
        "QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }"
        f"QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background: {p['border_strong']};"
        "  border-radius: 4px; min-height: 28px; min-width: 28px; }"
        f"QScrollBar::handle:hover {{ background: {p['faint']}; }}"
        "QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }"
        "QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }"
        f"QProgressBar {{ background-color: {p['panel_alt']}; border: 1px solid {p['border']}; border-radius: 5px;"
        f"  color: {p['text']}; text-align: center; min-height: 16px; }}"
        f"QProgressBar::chunk {{ background-color: {p['accent']}; border-radius: 4px; }}"
        f"QSplitter::handle {{ background-color: {p['border']}; }}"
        f"QToolTip {{ background-color: {p['header']}; color: #FFFFFF; border: none; border-radius: 4px; padding: 6px 8px; }}"
        f"QMenu {{ background-color: {p['panel']}; border: 1px solid {p['border']}; padding: 4px; }}"
        f"QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}"
        f"QMenu::item:selected {{ background-color: {p['accent_soft']}; color: {p['text']}; }}"
        # dialog header
        f"QFrame#pliDialogHeader, QFrame#leonardo_brand_header {{ background-color: {p['header']}; border: none;"
        "  border-radius: 8px; }"
        f"QLabel#pliDialogKicker {{ color: {p['header_muted']}; font-size: 11px; font-weight: 600;"
        "  letter-spacing: 1px; background: transparent; }"
        f"QLabel#pliDialogTitle {{ color: {p['header_text']}; font-size: 17px; font-weight: 600; background: transparent; }}"
        f"QLabel#leonardo_brand_name {{ color: {p['header_text']}; font-size: 30px; font-weight: 700;"
        "  letter-spacing: 0.5px; background: transparent; }"
        f"QLabel#leonardo_brand_tagline {{ color: {p['header_muted']}; font-size: 12px; font-weight: 600;"
        "  letter-spacing: 2px; background: transparent; }"
        f"QLabel#leonardo_brand_version, QLabel#leonardo_brand_contact {{ color: {p['header_muted']}; font-size: 12px;"
        "  background: transparent; }"
        f"QLabel#pliDialogMark {{ color: {p['header_text']}; background: {p['accent']}; border-radius: 6px;"
        "  font-weight: 700; min-width: 26px; max-width: 26px; min-height: 26px; max-height: 26px; }"
        # status / note labels used across Leonardo
        f"QLabel[leonardoRole=\"note\"] {{ color: {p['muted']}; }}"
    )


def _qt():
    from pymol.Qt import QtCore, QtGui, QtWidgets

    return QtCore, QtGui, QtWidgets


def _full_text(w):
    stored = str(w.property("leonardoFullText") or "").strip() if hasattr(w, "property") else ""
    shown = str(w.property("leonardoShownText") or "").strip() if hasattr(w, "property") else ""
    current = str(w.text() or "")
    return stored if stored and current.strip() in (shown, stored) else current


def relax_text(root):
    """Undo eliding: every button / check box shows its full text and asks for the width it needs."""
    QtCore, QtGui, QtWidgets = _qt()
    controls = root.findChildren(QtWidgets.QPushButton) + root.findChildren(QtWidgets.QCheckBox) + \
        root.findChildren(QtWidgets.QRadioButton)
    for c in controls:
        text = _full_text(c)
        current = str(c.text() or "")
        # a literal "&" is written "&&" in Qt (a single one is a keyboard mnemonic): keep the escaped form
        if text and text.replace("&&", "&") == current.replace("&&", "&"):
            text = current
        if text:
            # "Ligand Source & Prep": an "&" before a space is text, never a keyboard mnemonic
            text = re.sub(r"(?<!&)&(?=\s)", "&&", text)
        if text and text != current:
            c.setText(text)
        c.setProperty("leonardoShownText", text)
        if isinstance(c, QtWidgets.QPushButton):
            pad = 34 + ((c.iconSize().width() + 6) if not c.icon().isNull() else 0)
        else:
            pad = 30
        need = c.fontMetrics().horizontalAdvance(text.replace("&", "")) + pad if text else 0
        if need and c.minimumWidth() < need and "\n" not in text:
            c.setMinimumWidth(need)
        pol = c.sizePolicy()
        if pol.horizontalPolicy() == QtWidgets.QSizePolicy.Ignored:
            c.setSizePolicy(QtWidgets.QSizePolicy.Minimum, pol.verticalPolicy())


def _role_buttons(root):
    QtCore, QtGui, QtWidgets = _qt()
    for b in root.findChildren(QtWidgets.QPushButton):
        if b.property("leonardoRole"):
            continue
        t = _full_text(b).strip().lower().replace("&", "")
        name = str(b.objectName() or "").lower()
        if (t.startswith(PRIMARY_HINTS) and not t.endswith("...")) or name.startswith(("btn_run", "run_btn")):
            b.setProperty("leonardoRole", "primary")
            b.style().unpolish(b)
            b.style().polish(b)


def polish_layouts(root):
    """Wrap long labels without letting them collapse; keep fields at their natural height; tabs never elide."""
    QtCore, QtGui, QtWidgets = _qt()
    for label in root.findChildren(QtWidgets.QLabel):
        if label.pixmap() is not None and not label.pixmap().isNull():
            continue
        text = str(label.text() or "")
        if "<" in text:
            text = re.sub(r"<[^>]*>", "", text)    # rich text: judge the visible text, not the markup
        pol = label.sizePolicy()
        if label.wordWrap() or len(text) > 60 or "\n" in text:
            label.setWordWrap(True)
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
            label.setMinimumWidth(0)
            label.setMinimumHeight(0)
        elif pol.horizontalPolicy() == QtWidgets.QSizePolicy.Ignored:
            label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
    for w in root.findChildren(QtWidgets.QWidget):
        if isinstance(w, (QtWidgets.QLineEdit, QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)):
            pol = w.sizePolicy()
            if pol.verticalPolicy() in (QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Maximum):
                w.setSizePolicy(pol.horizontalPolicy(), QtWidgets.QSizePolicy.Fixed)
            w.setMinimumHeight(max(w.minimumHeight(), FIELD_HEIGHT))
        elif isinstance(w, QtWidgets.QPushButton):
            w.setMinimumHeight(max(w.minimumHeight(), 30))
        if isinstance(w, (QtWidgets.QLineEdit, QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox, QtWidgets.QPushButton)):
            need = w.minimumSizeHint().height()
            if w.maximumHeight() < need:
                w.setMaximumHeight(need)          # old fixed heights (30 px) clipped the themed fields
            if need > w.minimumHeight():
                w.setMinimumHeight(need)
    for bar in root.findChildren(QtWidgets.QTabBar):
        bar.setElideMode(QtCore.Qt.ElideNone)
        bar.setUsesScrollButtons(True)
        bar.setExpanding(False)
        for i in range(bar.count()):
            t = bar.tabText(i)
            fixed = re.sub(r"(?<!&)&(?=\s)", "&&", t)
            if fixed != t:
                bar.setTabText(i, fixed)
    for box in root.findChildren(QtWidgets.QGroupBox):
        lay = box.layout()
        if lay is not None:
            lay.setContentsMargins(12, 16, 12, 12)
            if lay.spacing() < 6:
                lay.setSpacing(8)
    _role_buttons(root)


def _is_button_row(item):
    """A layout made only of buttons and stretch (the dialog's Close/Run row): it stays pinned below."""
    QtCore, QtGui, QtWidgets = _qt()
    lay = item.layout() if item is not None else None
    wid = item.widget() if item is not None else None
    if isinstance(wid, QtWidgets.QDialogButtonBox):
        return True
    if lay is None or not isinstance(lay, QtWidgets.QHBoxLayout) or lay.count() == 0:
        return False
    buttons = 0
    for i in range(lay.count()):
        sub = lay.itemAt(i)
        w = sub.widget()
        if w is None:
            if sub.spacerItem() is not None:
                continue
            return False
        if isinstance(w, (QtWidgets.QPushButton, QtWidgets.QDialogButtonBox)):
            buttons += 1
        else:
            return False
    return buttons > 0


def ensure_scroll_body(dlg):
    """Put the dialog body (between its header and its bottom button row) in a scroll area that keeps every
    child at least at its minimum size: rows scroll instead of being compressed into each other."""
    QtCore, QtGui, QtWidgets = _qt()
    if dlg is None or getattr(dlg, "_leonardo_scroll_body", False) or getattr(dlg, "_leonardo_embedded_form", False):
        return False
    layout = dlg.layout()
    if layout is None or layout.count() == 0:
        return False                        # not built yet: decide when it is shown
    if not isinstance(layout, QtWidgets.QVBoxLayout):
        dlg._leonardo_scroll_body = True
        return False
    items = [layout.itemAt(i) for i in range(layout.count())]
    if any(isinstance(it.widget(), QtWidgets.QScrollArea) for it in items if it is not None) and len(items) <= 3:
        for it in items:
            if isinstance(it.widget(), QtWidgets.QScrollArea):
                it.widget().setWidgetResizable(True)
                dlg._leonardo_responsive_scroll = it.widget()     # sized like our own scroll body
        dlg._leonardo_scroll_body = True
        return False
    head, tail, body = [], [], []
    for it in items:
        w = it.widget()
        if w is not None and str(w.objectName() or "") in ("pliDialogHeader", "leonardo_brand_header") and not body:
            head.append(it)
        else:
            body.append(it)
    while body and _is_button_row(body[-1]):
        tail.insert(0, body.pop())
    if not body:
        dlg._leonardo_scroll_body = True
        return False
    for _ in range(layout.count()):
        layout.takeAt(0)
    content = QtWidgets.QWidget()
    content.setObjectName("pliResponsiveDialogContent")
    inner = QtWidgets.QVBoxLayout(content)
    inner.setContentsMargins(2, 2, 6, 2)
    inner.setSpacing(max(8, layout.spacing()))
    for it in body:
        if it.widget() is not None:
            inner.addWidget(it.widget())
        elif it.layout() is not None:
            inner.addLayout(it.layout())
        elif it.spacerItem() is not None:
            inner.addItem(it.spacerItem())
    last = inner.itemAt(inner.count() - 1) if inner.count() else None
    if last is None or last.spacerItem() is None:
        inner.addStretch(1)                 # spare height goes below the content, not between its rows
    scroll = QtWidgets.QScrollArea(dlg)
    scroll.setObjectName("pliResponsiveDialogScroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    scroll.setWidget(content)
    for it in head:
        layout.addWidget(it.widget())
    layout.addWidget(scroll, 1)
    for it in tail:
        if it.widget() is not None:
            layout.addWidget(it.widget())
        else:
            layout.addLayout(it.layout())
    dlg._leonardo_scroll_body = True
    dlg._leonardo_responsive_scroll = scroll
    return True


def fit_to_screen(dlg, fill=0.92, allow_shrink=True):
    """Size the window to its content (no scrolling when it fits), within ``fill`` of the screen."""
    QtCore, QtGui, QtWidgets = _qt()
    try:
        screen = (dlg.screen() if hasattr(dlg, "screen") else None) or QtWidgets.QApplication.primaryScreen()
        g = screen.availableGeometry()
    except Exception:
        return
    max_w, max_h = int(g.width() * fill), int(g.height() * fill)
    lay = dlg.layout()
    if lay is None:
        return
    lay.activate()
    hint = lay.sizeHint()
    scroll = getattr(dlg, "_leonardo_responsive_scroll", None)
    extra_h = 0
    if scroll is not None and scroll.widget() is not None:
        inner = scroll.widget()
        inner_hint = inner.sizeHint()
        extra_h = max(0, inner_hint.height() - scroll.viewport().height())
        need_w = max(hint.width(), inner_hint.width() + 24)
    else:
        need_w = hint.width()
    if scroll is not None and scroll.widget() is not None:
        # the body never needs a horizontal scroll bar when the screen is wide enough for it
        need_w = max(need_w, scroll.widget().minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
                     + 24)
    want_w = min(max_w, max(dlg.width(), need_w + 8))
    content_h = hint.height() + extra_h
    if scroll is not None and scroll.widget() is not None:
        # the whole body at its natural height (a scroll area's own size hint says nothing about it)
        body = scroll.widget()
        body_h = max(body.sizeHint().height(), body.minimumSizeHint().height())
        chrome = max(0, hint.height() - scroll.sizeHint().height())
        content_h = max(content_h, body_h + chrome + 12)
    # pages that scroll on their own inside the body (tabbed dialogs): grow by what they hide
    for area in dlg.findChildren(QtWidgets.QScrollArea):
        if area is scroll or not area.isVisibleTo(dlg) or area.widget() is None or \
                area.height() < 0.3 * max(1, dlg.height()):
            continue
        hidden = area.widget().sizeHint().height() - area.viewport().height()
        if hidden > 0:
            content_h = max(content_h, dlg.height() + hidden + 8)
    # tall default sizes left empty space below the content: fit the content height (never under 360 px)
    want_h = min(max_h, max(360, content_h))
    if not allow_shrink:
        want_w, want_h = max(want_w, dlg.width()), max(want_h, dlg.height())
    if dlg.minimumHeight() > want_h:
        dlg.setMinimumHeight(want_h)
    # later resizes by the dialog's own code cannot make it narrower than its content again
    dlg.setMinimumWidth(min(max_w, max(dlg.minimumWidth(), need_w + 8)))
    if dlg.minimumHeight() > max_h:
        dlg.setMinimumHeight(max_h)
    if want_w != dlg.width() or want_h != dlg.height():
        dlg.resize(want_w, want_h)


def polish_window(win, *, fit=True, scroll=True):
    """Theme + layout rules for one Leonardo window (idempotent)."""
    if win is None or getattr(win, "_leonardo_embedded_form", False):
        return
    marker = "LEONARDO_THEME_V3"
    try:
        current = str(win.styleSheet() or "")
        if marker not in current:
            win.setStyleSheet(theme_qss())
    except Exception:
        pass
    # sizes computed before the stylesheet changed are stale: re-polish children and recompute the layout
    try:
        QtCore, QtGui, QtWidgets = _qt()
        for w in win.findChildren(QtWidgets.QWidget):
            w.ensurePolished()
            w.updateGeometry()
    except Exception:
        pass
    polish_layouts(win)
    relax_text(win)
    if win.layout() is not None:
        win.layout().invalidate()
        win.layout().activate()
    if scroll:
        ensure_scroll_body(win)
    if fit:
        fit_to_screen(win)
        # a taller tab opened later grows the window (never shrinks it while the user switches tabs)
        try:
            QtCore, QtGui, QtWidgets = _qt()
            for tabs in win.findChildren(QtWidgets.QTabWidget):
                if not tabs.property("leonardoFitOnTab"):
                    tabs.setProperty("leonardoFitOnTab", True)
                    tabs.currentChanged.connect(
                        lambda _i, target=win: QtCore.QTimer.singleShot(
                            0, lambda: fit_to_screen(target, allow_shrink=False) if target.isVisible() else None))
        except Exception:
            pass
    # Qt applies a new stylesheet on the next event-loop pass and wrapped labels settle a little later: the
    # size rules run again then (a pass scheduled at construction must not stand in for the one at show time)
    try:
        QtCore, QtGui, QtWidgets = _qt()

        def again(target=win, do_fit=fit):
            try:
                polish_layouts(target)
                relax_text(target)
                if do_fit and target.isVisible():
                    fit_to_screen(target)
            except Exception:
                pass

        QtCore.QTimer.singleShot(0, again)
        if fit:
            QtCore.QTimer.singleShot(150, again)
    except Exception:
        pass
