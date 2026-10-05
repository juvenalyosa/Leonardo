"""Settings window of the AI assistant: provider, API keys, model, approvals, MCP server."""

import json

from pymol.Qt import QtCore, QtWidgets

from .. import config
from ..registry import APPROVAL_POLICIES
from .style import OK, STYLESHEET, WARN

POLICY_LABELS = [
    ("heavy", "Ask before heavy calculations and anything that deletes (recommended)"),
    ("every_step", "Ask before every step"),
    ("plan_only", "Only the plan approval (still asks before deleting)"),
]
EFFORT_LABELS = [("low", "Low (fast, cheaper)"), ("medium", "Medium"), ("high", "High (recommended)"),
                 ("xhigh", "Extra high"), ("max", "Max")]


class AssistantSettingsDialog(QtWidgets.QDialog):
    """Returns the updated settings in ``self.settings`` when accepted."""

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Leonardo AI Assistant: Settings")
        self.setStyleSheet(STYLESHEET)
        self.setMinimumWidth(560)
        self.settings = dict(settings or config.load_settings())
        # scrollable content + Save/Cancel pinned at the bottom, sized to the screen
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 10)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(14, 14, 14, 6)
        root.setSpacing(12)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        intro = QtWidgets.QLabel(
            "Choose the model that will drive Leonardo. API keys are stored in the system keychain, "
            "never in Leonardo's files. The assistant can only use Leonardo's own tools.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        root.addWidget(intro)

        # provider
        grp = QtWidgets.QGroupBox("Provider")
        g = QtWidgets.QHBoxLayout(grp)
        self.rb_anthropic = QtWidgets.QRadioButton("Claude (Anthropic)")
        self.rb_openai = QtWidgets.QRadioButton("OpenAI")
        g.addWidget(self.rb_anthropic)
        g.addWidget(self.rb_openai)
        g.addStretch(1)
        root.addWidget(grp)

        # keys
        grp_keys = QtWidgets.QGroupBox("API keys")
        form = QtWidgets.QFormLayout(grp_keys)
        self.key_edits = {}
        self.key_status = {}
        for provider, label in (("anthropic", "Anthropic key"), ("openai", "OpenAI key")):
            row = QtWidgets.QHBoxLayout()
            edit = QtWidgets.QLineEdit()
            edit.setEchoMode(QtWidgets.QLineEdit.Password)
            edit.setPlaceholderText("stored in keychain" if config.key_source(provider) == "keychain"
                                    else ("from environment variable" if config.key_source(provider)
                                          else "paste the key"))
            show = QtWidgets.QPushButton("Show")
            show.setObjectName("ghost")
            show.setCheckable(True)
            show.toggled.connect(lambda on, e=edit: e.setEchoMode(
                QtWidgets.QLineEdit.Normal if on else QtWidgets.QLineEdit.Password))
            test = QtWidgets.QPushButton("Test")
            test.clicked.connect(lambda _=False, p=provider: self._test_key(p))
            row.addWidget(edit, 1)
            row.addWidget(show)
            row.addWidget(test)
            status = QtWidgets.QLabel(self._source_text(provider))
            status.setObjectName("muted")
            box = QtWidgets.QVBoxLayout()
            box.addLayout(row)
            box.addWidget(status)
            form.addRow(label, box)
            self.key_edits[provider] = edit
            self.key_status[provider] = status
        root.addWidget(grp_keys)

        # model
        grp_model = QtWidgets.QGroupBox("Model")
        fm = QtWidgets.QFormLayout(grp_model)
        mrow = QtWidgets.QHBoxLayout()
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.setEditable(True)
        self.cmb_model.setMinimumWidth(300)
        self.btn_refresh = QtWidgets.QPushButton("Refresh from provider")
        self.btn_refresh.clicked.connect(self._refresh_models)
        mrow.addWidget(self.cmb_model, 1)
        mrow.addWidget(self.btn_refresh)
        fm.addRow("Model", mrow)
        self.cmb_effort = QtWidgets.QComboBox()
        for key, label in EFFORT_LABELS:
            self.cmb_effort.addItem(label, key)
        fm.addRow("Reasoning effort", self.cmb_effort)
        root.addWidget(grp_model)

        # approvals
        grp_ap = QtWidgets.QGroupBox("Approvals")
        fa = QtWidgets.QFormLayout(grp_ap)
        self.cmb_policy = QtWidgets.QComboBox()
        for key, label in POLICY_LABELS:
            self.cmb_policy.addItem(label, key)
        fa.addRow("During a plan", self.cmb_policy)
        self.spin_calls = QtWidgets.QSpinBox()
        self.spin_calls.setRange(5, 400)
        fa.addRow("Max tool calls per request", self.spin_calls)
        root.addWidget(grp_ap)

        # MCP server
        grp_mcp = QtWidgets.QGroupBox("External MCP server (Claude Desktop, Claude Code, ...)")
        fx = QtWidgets.QFormLayout(grp_mcp)
        self.cb_mcp = QtWidgets.QCheckBox("Serve Leonardo's tools on this computer (127.0.0.1 only)")
        self.spin_port = QtWidgets.QSpinBox()
        self.spin_port.setRange(1024, 65535)
        fx.addRow(self.cb_mcp)
        fx.addRow("Port", self.spin_port)
        self.lbl_snippet = QtWidgets.QLabel()
        self.lbl_snippet.setObjectName("muted")
        self.lbl_snippet.setWordWrap(True)
        self.lbl_snippet.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        fx.addRow(self.lbl_snippet)
        self.spin_port.valueChanged.connect(self._update_snippet)
        root.addWidget(grp_mcp)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.Save).setObjectName("primary")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(14, 0, 14, 0)
        bar.addWidget(buttons)
        outer.addLayout(bar)
        root.addStretch(1)
        try:
            avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
            hint = content.sizeHint()
            self.resize(min(hint.width() + 40, int(avail.width() * 0.9)),
                        min(hint.height() + 70, int(avail.height() * 0.85)))
        except Exception:
            self.resize(620, 640)

        self._load_into_widgets()
        self.rb_anthropic.toggled.connect(lambda _on: self._fill_models())

    # -- state <-> widgets ------------------------------------------------------ #
    def _provider(self):
        return "openai" if self.rb_openai.isChecked() else "anthropic"

    def _load_into_widgets(self):
        s = self.settings
        (self.rb_openai if s.get("provider") == "openai" else self.rb_anthropic).setChecked(True)
        self._fill_models()
        idx = self.cmb_effort.findData(s.get("effort", "high"))
        self.cmb_effort.setCurrentIndex(max(0, idx))
        idx = self.cmb_policy.findData(s.get("approval_policy", "heavy"))
        self.cmb_policy.setCurrentIndex(max(0, idx))
        self.spin_calls.setValue(int(s.get("max_tool_calls", 300)))
        self.cb_mcp.setChecked(bool(s.get("mcp_server_enabled", False)))
        self.spin_port.setValue(int(s.get("mcp_server_port", 8765)))
        self._update_snippet()

    def _fill_models(self, live=None):
        provider = self._provider()
        current = self.settings.get("model_anthropic" if provider == "anthropic" else "model_openai", "")
        self.cmb_model.clear()
        items = list(live) if live else (
            [m for m, _label in config.ANTHROPIC_MODELS] if provider == "anthropic" else list(config.OPENAI_MODELS))
        labels = dict(config.ANTHROPIC_MODELS)
        for mid in items:
            self.cmb_model.addItem(f"{labels.get(mid, mid)}  ({mid})" if mid in labels else mid, mid)
        if current:
            idx = self.cmb_model.findData(current)
            if idx < 0:
                self.cmb_model.addItem(current, current)
                idx = self.cmb_model.count() - 1
            self.cmb_model.setCurrentIndex(idx)
        if provider == "openai" and not items:
            self.cmb_model.setEditText(current or "")
            self.cmb_model.lineEdit().setPlaceholderText("press Refresh to list your OpenAI models")

    def _selected_model(self):
        data = self.cmb_model.currentData()
        text = self.cmb_model.currentText().strip()
        if data and (text == self.cmb_model.itemText(self.cmb_model.currentIndex())):
            return str(data)
        # typed model id (strip a "Label  (id)" form)
        if text.endswith(")") and "(" in text:
            return text[text.rfind("(") + 1:-1].strip()
        return text

    def _source_text(self, provider):
        src = config.key_source(provider)
        return {"keychain": "Key stored in the system keychain.",
                "environment": f"Using the {config.ENV_KEYS[provider]} environment variable."}.get(src, "No key yet.")

    def _update_snippet(self):
        url = f"http://127.0.0.1:{self.spin_port.value()}/mcp"
        snippet = json.dumps({"mcpServers": {"leonardo": {"type": "http", "url": url}}}, indent=1)
        self.lbl_snippet.setText(
            f"Endpoint: {url}\nClaude Code: claude mcp add --transport http leonardo {url}\n"
            f"Other clients:\n{snippet}\nHeavy or destructive calls from outside still ask for confirmation "
            "in Leonardo.")

    # -- actions --------------------------------------------------------------- #
    def _key_for(self, provider):
        typed = self.key_edits[provider].text().strip()
        return typed or config.get_api_key(provider)

    def _in_background(self, fn, done):
        """Run a network call off the GUI thread; ``done(result, error)`` runs back on it."""
        dlg = self

        class _Call(QtCore.QThread):
            finished_call = QtCore.Signal(object, object)

            def run(self):
                try:
                    self.finished_call.emit(fn(), None)
                except Exception as exc:
                    self.finished_call.emit(None, exc)

        worker = _Call(self)
        calls = self.__dict__.setdefault("_bg_calls", set())
        calls.add(worker)

        def finish(result, error):
            calls.discard(worker)
            QtWidgets.QApplication.restoreOverrideCursor()
            if dlg.isVisible():
                done(result, error)
        worker.finished_call.connect(finish)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        worker.start()

    def _test_key(self, provider):
        key = self._key_for(provider)
        label = self.key_status[provider]
        if not key:
            label.setText("No key to test.")
            return
        label.setText("Testing the key...")

        def done(models, error):
            if error is not None:
                label.setText(f"Key test failed: {error}")
                label.setStyleSheet(f"color: {WARN};")
            else:
                label.setText(f"Key works: {len(models)} models available.")
                label.setStyleSheet(f"color: {OK};")
        self._in_background(lambda: self._list_models(provider, key), done)

    @staticmethod
    def _list_models(provider, key):
        if provider == "openai":
            from ..providers.openai_provider import list_openai_models

            return list_openai_models(key)
        from ..providers.anthropic_provider import list_anthropic_models

        return list_anthropic_models(key)

    def _refresh_models(self):
        provider = self._provider()
        key = self._key_for(provider)
        if not key:
            QtWidgets.QMessageBox.information(self, "Models", "Add an API key first.")
            return

        def done(models, error):
            if error is not None:
                QtWidgets.QMessageBox.warning(self, "Models", f"Could not list models: {error}")
                return
            if provider == "anthropic":
                pinned = [m for m, _ in config.ANTHROPIC_MODELS]
                models = pinned + [m for m in models if m not in pinned]
            self._fill_models(models)
        self._in_background(lambda: self._list_models(provider, key), done)

    def _save(self):
        for provider, edit in self.key_edits.items():
            typed = edit.text().strip()
            if typed and not config.set_api_key(provider, typed):
                QtWidgets.QMessageBox.warning(
                    self, "Keychain", "The system keychain is not available; set the "
                    f"{config.ENV_KEYS[provider]} environment variable instead.")
        provider = self._provider()
        self.settings["provider"] = provider
        self.settings["model_anthropic" if provider == "anthropic" else "model_openai"] = self._selected_model()
        self.settings["effort"] = str(self.cmb_effort.currentData())
        self.settings["approval_policy"] = str(self.cmb_policy.currentData())
        if self.settings["approval_policy"] not in APPROVAL_POLICIES:
            self.settings["approval_policy"] = "heavy"
        self.settings["max_tool_calls"] = int(self.spin_calls.value())
        self.settings["mcp_server_enabled"] = bool(self.cb_mcp.isChecked())
        self.settings["mcp_server_port"] = int(self.spin_port.value())
        config.save_settings(self.settings)
        self.accept()

