"""
Chat panel of the Leonardo AI assistant (a dock widget in PyMOL's main window).

The panel shows the conversation as cards: user messages, streamed assistant text,
collapsible reasoning summaries, the plan (approve / request changes / reject), one
card per tool call (status, duration, result, "Show in viewer" to recall that step's
scene), confirmation and question cards, and a usage footer.
"""

import json

from pymol.Qt import QtCore, QtGui, QtWidgets

from .style import ACCENT, ACCENT_2, ERR, MUTED, OK, PERMISSION_BADGE, STYLESHEET, WARN, pill_style

STATUS_TEXT = {
    "idle": ("Listo", OK), "thinking": ("Pensando", ACCENT), "executing": ("Ejecutando", ACCENT_2),
    "cancelled": ("Detenido", WARN), "error": ("Error", ERR), "waiting": ("Esperando tu respuesta", WARN),
}
TOOL_STATUS = {"running": ("⏳", ACCENT), "done": ("✓", OK), "failed": ("✕", ERR), "declined": ("⤫", WARN)}


def _label(text="", obj="", wrap=True, rich=False):
    lab = QtWidgets.QLabel(text)
    if obj:
        lab.setObjectName(obj)
    lab.setWordWrap(wrap)
    lab.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse | QtCore.Qt.LinksAccessibleByMouse)
    if rich:
        lab.setTextFormat(QtCore.Qt.MarkdownText)
    return lab


def _card(obj="card"):
    frame = QtWidgets.QFrame()
    frame.setObjectName(obj)
    lay = QtWidgets.QVBoxLayout(frame)
    lay.setContentsMargins(14, 10, 14, 10)
    lay.setSpacing(6)
    return frame, lay


def _pill(text, color):
    p = QtWidgets.QLabel(text)
    p.setObjectName("pill")
    p.setStyleSheet(pill_style(color))
    return p


def _short_args(args, limit=160):
    text = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in (args or {}).items())
    return text if len(text) <= limit else text[:limit] + "…"


def _result_summary(result, limit=420):
    if not result:
        return ""
    parts = []
    for k, v in result.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
            if len(v) > 90:
                v = v[:90] + "…"
        parts.append(f"**{k}**: {v}")
    text = "  \n".join(parts)
    return text if len(text) <= limit else text[:limit] + "…"


class ChatPanel(QtWidgets.QWidget):
    """UI only: the controller wires send/stop/settings and feeds events/questions."""

    send_requested = QtCore.Signal(str)
    stop_requested = QtCore.Signal()
    settings_requested = QtCore.Signal()
    new_chat_requested = QtCore.Signal()
    answer_given = QtCore.Signal(object)
    scene_requested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(STYLESHEET)
        self.setMinimumWidth(380)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # header
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(14, 10, 10, 10)
        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(0)
        title_box.addWidget(_label("Leonardo AI", "title", wrap=False))
        self.lbl_model = _label("", "muted", wrap=False)
        title_box.addWidget(self.lbl_model)
        hl.addLayout(title_box, 1)
        self.pill_status = _pill("Listo", OK)
        hl.addWidget(self.pill_status)
        btn_new = QtWidgets.QPushButton("Nuevo")
        btn_new.setObjectName("ghost")
        btn_new.setToolTip("Nueva conversación")
        btn_new.clicked.connect(self.new_chat_requested.emit)
        hl.addWidget(btn_new)
        btn_settings = QtWidgets.QPushButton("⚙")
        btn_settings.setObjectName("ghost")
        btn_settings.setToolTip("Configuración del asistente")
        btn_settings.clicked.connect(self.settings_requested.emit)
        hl.addWidget(btn_settings)
        root.addWidget(header)

        # conversation
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        holder = QtWidgets.QWidget()
        self.feed = QtWidgets.QVBoxLayout(holder)
        self.feed.setContentsMargins(12, 12, 12, 12)
        self.feed.setSpacing(10)
        self.feed.addStretch(1)
        self.scroll.setWidget(holder)
        root.addWidget(self.scroll, 1)

        # composer
        composer = QtWidgets.QFrame()
        composer.setObjectName("composer")
        cl = QtWidgets.QVBoxLayout(composer)
        cl.setContentsMargins(12, 10, 12, 10)
        self.input = QtWidgets.QPlainTextEdit()
        self.input.setPlaceholderText(
            "Describe lo que quieres hacer, p. ej. «Carga 3PTB, encuentra el bolsillo de la benzamidina y "
            "haz redocking del ligando cristalino a pH 7.4». Ctrl+Enter para enviar.")
        self.input.setFixedHeight(84)
        cl.addWidget(self.input)
        row = QtWidgets.QHBoxLayout()
        self.lbl_usage = _label("", "muted", wrap=False)
        row.addWidget(self.lbl_usage, 1)
        self.btn_stop = QtWidgets.QPushButton("Detener")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        row.addWidget(self.btn_stop)
        self.btn_send = QtWidgets.QPushButton("Enviar")
        self.btn_send.setObjectName("primary")
        self.btn_send.clicked.connect(self._send)
        row.addWidget(self.btn_send)
        cl.addLayout(row)
        root.addWidget(composer)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Return"), self.input, activated=self._send)

        self._text_label = None
        self._text_buffer = ""
        self._thinking_label = None
        self._thinking_buffer = ""
        self._tool_cards = []
        self._step_labels = {}
        self._busy = False

    # -- helpers -------------------------------------------------------------- #
    def _add(self, widget):
        self.feed.insertWidget(self.feed.count() - 1, widget)
        QtCore.QTimer.singleShot(30, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))
        return widget

    def set_model_label(self, text):
        self.lbl_model.setText(str(text))

    def set_status(self, state):
        text, color = STATUS_TEXT.get(state, (state, MUTED))
        self.pill_status.setText(text)
        self.pill_status.setStyleSheet(pill_style(color))

    def set_busy(self, busy):
        self._busy = bool(busy)
        self.btn_send.setEnabled(not busy)
        self.btn_stop.setVisible(bool(busy))
        self.input.setReadOnly(bool(busy))

    def set_usage(self, usage, cost_text=""):
        tin = int(usage.get("input_tokens", 0))
        tout = int(usage.get("output_tokens", 0))
        self.lbl_usage.setText(f"Tokens: {tin:,} entrada · {tout:,} salida" + (f" · {cost_text}" if cost_text else ""))

    def clear(self):
        while self.feed.count() > 1:
            item = self.feed.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._end_text()
        self._tool_cards = []
        self._step_labels = {}

    def _send(self):
        text = self.input.toPlainText().strip()
        if not text or self._busy:
            return
        self.input.clear()
        self.add_user(text)
        self.send_requested.emit(text)

    # -- messages ------------------------------------------------------------ #
    def add_user(self, text):
        card, lay = _card("cardUser")
        lay.addWidget(_label(text))
        self._add(card)

    def add_notice(self, text, color=MUTED):
        lab = _label(text, "muted")
        lab.setStyleSheet(f"color: {color};")
        self._add(lab)

    def add_error(self, text):
        card, lay = _card("cardError")
        lay.addWidget(_label(f"⚠ {text}"))
        self._add(card)

    def append_text(self, delta):
        if self._text_label is None:
            card, lay = _card()
            self._text_label = _label("", rich=True)
            lay.addWidget(self._text_label)
            self._add(card)
            self._text_buffer = ""
        self._text_buffer += str(delta)
        self._text_label.setText(self._text_buffer)

    def append_thinking(self, delta):
        if self._thinking_label is None:
            box = QtWidgets.QWidget()
            bl = QtWidgets.QVBoxLayout(box)
            bl.setContentsMargins(4, 0, 4, 0)
            toggle = QtWidgets.QPushButton("▸ Razonamiento")
            toggle.setObjectName("ghost")
            toggle.setCheckable(True)
            self._thinking_label = _label("", "thinking")
            self._thinking_label.setVisible(False)
            toggle.toggled.connect(lambda on, lab=self._thinking_label, b=toggle: (
                lab.setVisible(on), b.setText(("▾" if on else "▸") + " Razonamiento")))
            bl.addWidget(toggle, 0, QtCore.Qt.AlignLeft)
            bl.addWidget(self._thinking_label)
            self._add(box)
            self._thinking_buffer = ""
        self._thinking_buffer += str(delta)
        self._thinking_label.setText(self._thinking_buffer)

    def _end_text(self):
        self._text_label = None
        self._thinking_label = None

    def end_text(self):
        self._end_text()

    # -- plan ---------------------------------------------------------------------- #
    def show_plan(self, plan, interactive=True):
        self._end_text()
        card, lay = _card("cardPlan")
        top = QtWidgets.QHBoxLayout()
        top.addWidget(_label("Plan propuesto", "title", wrap=False))
        top.addStretch(1)
        if plan.get("estimated_time"):
            top.addWidget(_pill(f"⏱ {plan['estimated_time']}", ACCENT))
        lay.addLayout(top)
        lay.addWidget(_label(str(plan.get("goal", ""))))
        for i, st in enumerate(plan.get("steps", []) or [], start=1):
            row = QtWidgets.QHBoxLayout()
            num = _pill(str(i), ACCENT_2)
            row.addWidget(num, 0, QtCore.Qt.AlignTop)
            col = QtWidgets.QVBoxLayout()
            head = QtWidgets.QHBoxLayout()
            head.addWidget(_label(f"**{st.get('title', '')}**", rich=True), 1)
            if st.get("heavy"):
                head.addWidget(_pill("cálculo pesado", WARN))
            status = _label("", "muted", wrap=False)
            head.addWidget(status)
            self._step_labels[i] = status
            col.addLayout(head)
            col.addWidget(_label(str(st.get("detail", "")), "muted"))
            row.addLayout(col, 1)
            lay.addLayout(row)
        if plan.get("assumptions"):
            lay.addWidget(_label("Supuestos: " + "; ".join(map(str, plan["assumptions"])), "muted"))
        if interactive:
            feedback = QtWidgets.QLineEdit()
            feedback.setPlaceholderText("Cambios o comentarios (opcional)")
            lay.addWidget(feedback)
            buttons = QtWidgets.QHBoxLayout()
            b_ok = QtWidgets.QPushButton("Aprobar y ejecutar")
            b_ok.setObjectName("primary")
            b_rev = QtWidgets.QPushButton("Pedir cambios")
            b_no = QtWidgets.QPushButton("Rechazar")
            b_no.setObjectName("danger")
            buttons.addWidget(b_ok)
            buttons.addWidget(b_rev)
            buttons.addStretch(1)
            buttons.addWidget(b_no)
            lay.addLayout(buttons)

            def done(decision):
                for b in (b_ok, b_rev, b_no):
                    b.setEnabled(False)
                feedback.setReadOnly(True)
                verdict = {"approve": ("Plan aprobado", OK), "revise": ("Cambios pedidos", WARN),
                           "reject": ("Plan rechazado", ERR)}[decision]
                lay.addWidget(_pill(verdict[0], verdict[1]), 0, QtCore.Qt.AlignLeft)
                self.answer_given.emit({"decision": decision, "feedback": feedback.text().strip()})
            b_ok.clicked.connect(lambda: done("approve"))
            b_rev.clicked.connect(lambda: done("revise") if feedback.text().strip() else feedback.setFocus())
            b_no.clicked.connect(lambda: done("reject"))
        self._add(card)

    def mark_step(self, step, status, summary=""):
        lab = self._step_labels.get(int(step))
        if lab is None:
            return
        icon = {"running": "⏳ en curso", "done": "✓ hecho", "failed": "✕ falló", "skipped": "↷ omitido"}.get(status, status)
        color = {"running": ACCENT, "done": OK, "failed": ERR, "skipped": MUTED}.get(status, MUTED)
        lab.setText(icon)
        lab.setStyleSheet(f"color: {color};")
        if summary:
            lab.setToolTip(summary)

    # -- tools -------------------------------------------------------------------- #
    def tool_event(self, payload):
        self._end_text()
        status = payload.get("status", "running")
        if status in ("running", "declined") or not self._tool_cards or self._tool_cards[-1][0] != payload.get("tool"):
            card, lay = _card()
            head = QtWidgets.QHBoxLayout()
            icon = _label("", wrap=False)
            head.addWidget(icon)
            head.addWidget(_label(f"**{payload.get('title', payload.get('tool'))}**", rich=True), 1)
            badge, color = PERMISSION_BADGE.get(payload.get("permission", "read"), ("", MUTED))
            head.addWidget(_pill(badge, color))
            lay.addLayout(head)
            args_lab = _label(_short_args(payload.get("args")), "muted")
            lay.addWidget(args_lab)
            body = _label("", rich=True)
            lay.addWidget(body)
            details = _label("", "muted", rich=True)
            details.setVisible(False)
            footer = QtWidgets.QHBoxLayout()
            btn_details = QtWidgets.QPushButton("▸ Detalles")
            btn_details.setObjectName("ghost")
            btn_details.setCheckable(True)
            btn_details.setVisible(False)
            btn_details.toggled.connect(lambda on, d=details, b=btn_details: (
                d.setVisible(on), b.setText(("▾" if on else "▸") + " Detalles")))
            footer.addWidget(btn_details)
            timing = _label("", "muted", wrap=False)
            footer.addWidget(timing, 1)
            btn_scene = QtWidgets.QPushButton("Ver en el visor")
            btn_scene.setObjectName("ghost")
            btn_scene.setVisible(False)
            footer.addWidget(btn_scene)
            lay.addLayout(footer)
            lay.addWidget(details)
            self._add(card)
            self._tool_cards.append((payload.get("tool"), icon, body, timing, btn_scene, details, btn_details))
        _tool, icon, body, timing, btn_scene, details, btn_details = self._tool_cards[-1]
        glyph, color = TOOL_STATUS.get(status, ("•", MUTED))
        icon.setText(glyph)
        icon.setStyleSheet(f"color: {color}; font-size: 15px;")
        if status == "done":
            result = dict(payload.get("result") or {})
            scene = str(result.get("scene", "") or "")
            rest = {k: v for k, v in result.items() if k != "scene"}
            if payload.get("card"):
                body.setText(str(payload["card"]))
                details.setText(_result_summary(rest, limit=1600))
                btn_details.setVisible(bool(rest))
            else:
                body.setText(_result_summary(rest))
            timing.setText(f"{payload.get('seconds', 0)} s")
            if scene:
                btn_scene.setVisible(True)
                btn_scene.clicked.connect(lambda _=False, s=scene: self.scene_requested.emit(s))
        elif status == "failed":
            body.setText(f"<span style='color:{ERR}'>{payload.get('error', '')}</span>")
            body.setTextFormat(QtCore.Qt.RichText)
        elif status == "declined":
            body.setText("Omitido por el usuario.")

    def progress(self, text):
        if self._tool_cards:
            self._tool_cards[-1][3].setText(str(text))

    # -- questions ------------------------------------------------------------------ #
    def ask_confirm(self, payload):
        self._end_text()
        card, lay = _card("cardAsk")
        badge, color = PERMISSION_BADGE.get(payload.get("permission", ""), ("", WARN))
        head = QtWidgets.QHBoxLayout()
        head.addWidget(_label(f"¿Ejecutar **{payload.get('title', payload.get('tool'))}**?", rich=True), 1)
        head.addWidget(_pill(badge, color))
        lay.addLayout(head)
        lay.addWidget(_label(_short_args(payload.get("args"), 400), "muted"))
        row = QtWidgets.QHBoxLayout()
        b_yes = QtWidgets.QPushButton("Ejecutar")
        b_yes.setObjectName("primary")
        b_no = QtWidgets.QPushButton("Omitir")
        row.addWidget(b_yes)
        row.addWidget(b_no)
        row.addStretch(1)
        lay.addLayout(row)

        def done(ok):
            b_yes.setEnabled(False)
            b_no.setEnabled(False)
            self.answer_given.emit({"approved": bool(ok)})
        b_yes.clicked.connect(lambda: done(True))
        b_no.clicked.connect(lambda: done(False))
        self._add(card)

    def ask_question(self, payload):
        self._end_text()
        card, lay = _card("cardAsk")
        lay.addWidget(_label(str(payload.get("question", "")), rich=True))
        buttons = []
        free = QtWidgets.QLineEdit()
        free.setPlaceholderText("Escribe tu respuesta")

        def done(text):
            for b in buttons:
                b.setEnabled(False)
            free.setReadOnly(True)
            lay.addWidget(_pill(f"→ {text}", OK), 0, QtCore.Qt.AlignLeft)
            self.answer_given.emit({"text": str(text)})
        opts = list(payload.get("options") or [])
        if opts:
            flow = QtWidgets.QHBoxLayout()
            for opt in opts:
                b = QtWidgets.QPushButton(str(opt))
                b.clicked.connect(lambda _=False, o=opt: done(o))
                buttons.append(b)
                flow.addWidget(b)
            flow.addStretch(1)
            lay.addLayout(flow)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(free, 1)
        b_send = QtWidgets.QPushButton("Responder")
        b_send.setObjectName("primary")
        b_send.clicked.connect(lambda: done(free.text().strip()) if free.text().strip() else None)
        free.returnPressed.connect(b_send.click)
        buttons.append(b_send)
        row.addWidget(b_send)
        lay.addLayout(row)
        self._add(card)
