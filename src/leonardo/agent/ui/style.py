"""Visual style of the assistant windows: white background, dark blue accents and grays."""

ACCENT = "#1F3A6E"        # dark blue: primary actions, focus, links
ACCENT_2 = "#3E5C99"      # lighter navy: plans
OK = "#1F7A4D"
WARN = "#A65F00"
ERR = "#B42318"
BG = "#FFFFFF"
PANEL = "#F2F4F7"         # light gray: header and composer bars
CARD = "#F7F8FA"
CARD_2 = "#E6EAF0"
BORDER = "#D0D5DD"
TEXT = "#1D2939"
MUTED = "#667085"

PERMISSION_BADGE = {
    "read": ("Read", MUTED),
    "view": ("View", ACCENT),
    "compute_light": ("Compute", OK),
    "compute_heavy": ("Heavy compute", WARN),
    "destructive": ("Modifies/Deletes", ERR),
}

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QLabel {{ background: transparent; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#cardUser {{ background: #E8EEF8; border: 1px solid #B3C3E0; border-radius: 12px; }}
QFrame#cardPlan {{ background: {CARD}; border: 1px solid {ACCENT_2}; border-radius: 12px; }}
QFrame#cardAsk {{ background: {CARD}; border: 1px solid {WARN}; border-radius: 12px; }}
QFrame#cardError {{ background: #FDEDEC; border: 1px solid {ERR}; border-radius: 12px; }}
QFrame#header {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QFrame#composer {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QLabel#title {{ font-size: 16px; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#thinking {{ color: {MUTED}; font-style: italic; }}
QLabel#pill {{ border-radius: 9px; padding: 2px 10px; font-size: 11px; font-weight: 600; }}
QPlainTextEdit, QLineEdit, QComboBox, QSpinBox {{
    background: {BG}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px; color: {TEXT};
    selection-background-color: {ACCENT}; selection-color: #FFFFFF;
}}
QPlainTextEdit:focus, QLineEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QPushButton {{
    background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 14px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#primary {{ background: {ACCENT}; color: #FFFFFF; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: #2B4C8C; }}
QPushButton#primary:disabled {{ background: {CARD_2}; color: {MUTED}; }}
QPushButton#danger {{ background: transparent; border: 1px solid {ERR}; color: {ERR}; }}
QPushButton#ghost {{ background: transparent; border: none; color: {MUTED}; }}
QPushButton#ghost:hover {{ color: {TEXT}; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: #B8C0CC; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 10px; margin-top: 14px; padding-top: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {MUTED}; }}
QCheckBox {{ background: transparent; }}
QToolTip {{ background: {ACCENT}; color: #FFFFFF; border: none; }}
QComboBox QAbstractItemView {{ background: {BG}; color: {TEXT}; selection-background-color: {ACCENT};
    selection-color: #FFFFFF; }}
QMenu {{ background: {BG}; color: {TEXT}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: {ACCENT}; color: #FFFFFF; }}
"""


def _rgba(color, alpha):
    c = color.lstrip("#")
    return f"rgba({int(c[0:2], 16)}, {int(c[2:4], 16)}, {int(c[4:6], 16)}, {alpha})"


def pill_style(color):
    # Qt reads 8-digit hex as #AARRGGBB, so translucency goes through rgba()
    return f"background: {_rgba(color, 0.12)}; color: {color}; border: 1px solid {_rgba(color, 0.45)};"
