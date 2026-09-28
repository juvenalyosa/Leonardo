"""Visual style of the assistant windows (dark, matching the viewer's studio look)."""

ACCENT = "#4FB3FF"
ACCENT_2 = "#8C7BFF"
OK = "#3DDC97"
WARN = "#FFB547"
ERR = "#FF6B6B"
BG = "#161A21"
PANEL = "#1E232C"
CARD = "#252B36"
CARD_2 = "#2C3340"
BORDER = "#343C4A"
TEXT = "#E6EAF0"
MUTED = "#8E98A8"

PERMISSION_BADGE = {
    "read": ("Consulta", MUTED),
    "view": ("Vista", ACCENT),
    "compute_light": ("Cálculo", OK),
    "compute_heavy": ("Cálculo pesado", WARN),
    "destructive": ("Modifica/Borra", ERR),
}

STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QLabel {{ background: transparent; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}
QFrame#cardUser {{ background: #243449; border: 1px solid #2F4A6B; border-radius: 12px; }}
QFrame#cardPlan {{ background: {CARD}; border: 1px solid {ACCENT_2}; border-radius: 12px; }}
QFrame#cardAsk {{ background: {CARD}; border: 1px solid {WARN}; border-radius: 12px; }}
QFrame#cardError {{ background: #3A2226; border: 1px solid {ERR}; border-radius: 12px; }}
QFrame#header {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QFrame#composer {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QLabel#title {{ font-size: 16px; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#thinking {{ color: {MUTED}; font-style: italic; }}
QLabel#pill {{ border-radius: 9px; padding: 2px 10px; font-size: 11px; font-weight: 600; }}
QPlainTextEdit, QLineEdit, QComboBox, QSpinBox {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px; color: {TEXT};
    selection-background-color: {ACCENT};
}}
QPlainTextEdit:focus, QLineEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QPushButton {{
    background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 14px; color: {TEXT};
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#primary {{ background: {ACCENT}; color: #0B1220; border: none; font-weight: 600; }}
QPushButton#primary:hover {{ background: #6CC2FF; }}
QPushButton#danger {{ background: transparent; border: 1px solid {ERR}; color: {ERR}; }}
QPushButton#ghost {{ background: transparent; border: none; color: {MUTED}; }}
QPushButton#ghost:hover {{ color: {TEXT}; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 10px; margin-top: 14px; padding-top: 10px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 6px; color: {MUTED}; }}
QCheckBox {{ background: transparent; }}
QToolTip {{ background: {CARD_2}; color: {TEXT}; border: 1px solid {BORDER}; }}
"""


def pill_style(color):
    return f"background: {color}22; color: {color}; border: 1px solid {color}66;"
