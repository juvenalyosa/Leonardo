"""
Wires the assistant together inside Leonardo: tool registry + context, in-app MCP
server/client, optional external MCP server, provider sessions, worker threads and the
chat window (a separate window, so PyMOL's viewer keeps its size).
"""

from pymol.Qt import QtCore, QtWidgets

from .. import config, presentation
from ..mcp_server import ExternalMcpServer, McpToolClient, build_server
from ..orchestrator import SYSTEM_PROMPT, AgentSession
from ..providers.base import ProviderError
from ..registry import APPROVAL_POLICIES
from ..tools import AgentContext, _headless, build_registry
from .bridge import AgentWorker, MainThreadRunner, UiAdapter
from .chat_panel import ChatPanel
from .settings_dialog import AssistantSettingsDialog

# $ per million tokens (input, output), first-party Claude API list prices
CLAUDE_PRICES = {
    "claude-opus-5-5": (4.0, 20.0), "claude-opus-5": (5.0, 25.0), "claude-fable-5-1": (10.0, 50.0),
    "claude-sonnet-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0),
}


class _ChatWindow(QtWidgets.QWidget):
    """Top-level window holding the chat panel; remembers its geometry."""

    def __init__(self, panel):
        super().__init__(None, QtCore.Qt.Window)
        self.setWindowTitle("Leonardo AI")
        self.setMinimumSize(380, 480)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(panel)

    def _save_geometry(self):
        QtCore.QSettings("Leonardo", "AIAssistant").setValue("geometry", self.saveGeometry())

    def closeEvent(self, event):
        self._save_geometry()
        super().closeEvent(event)

    def moveEvent(self, event):
        super().moveEvent(event)
        if self.isVisible():
            self._save_geometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.isVisible():
            self._save_geometry()


class AssistantController(QtCore.QObject):
    def __init__(self, leonardo_dialog, parent=None):
        super().__init__(parent)
        self.dlg = leonardo_dialog
        self.settings = config.load_settings()
        self.runner = MainThreadRunner(self)
        self.context = AgentContext(self.dlg, run_on_main=self.runner.run)
        self.registry = build_registry(self.context)
        self.mcp_client = McpToolClient(build_server(self.registry, run_on_main=self.runner.run,
                                                     context=self.context))
        self.external = None
        self.adapter = UiAdapter(self)
        self.panel = ChatPanel()
        self.dock = None
        self.session = None
        self.worker = None
        self._pending_kind = ""

        self.adapter.event.connect(self._on_event)
        self.adapter.question.connect(self._on_question)
        self.panel.answer_given.connect(self._on_answer)
        self.panel.send_requested.connect(self.send)
        self.panel.stop_requested.connect(self.stop)
        self.panel.settings_requested.connect(self.open_settings)
        self.panel.new_chat_requested.connect(self.new_chat)
        self.panel.scene_requested.connect(lambda key: presentation.recall_scene(key))
        self._refresh_model_label()
        self._sync_external_server()

    # -- window ------------------------------------------------------------------ #
    def show(self):
        """The chat is its own window (not docked), so PyMOL's viewer keeps its full size."""
        if self.dock is None:
            self.dock = _ChatWindow(self.panel)
            self._place_window()
        self.dock.show()
        self.dock.raise_()
        self.dock.activateWindow()
        if not config.get_api_key(self.settings.get("provider", "anthropic")):
            QtCore.QTimer.singleShot(200, self.open_settings)

    def _place_window(self):
        """Last position/size if saved, else next to PyMOL's window (or at the right edge of the screen)."""
        win = self.dock
        saved = QtCore.QSettings("Leonardo", "AIAssistant").value("geometry")
        if saved is not None and win.restoreGeometry(saved):
            return
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        width = min(460, int(screen.width() * 0.35))
        height = min(860, int(screen.height() * 0.9))
        win.resize(width, height)
        x = screen.right() - width - 12
        try:
            from pymol import gui

            host = gui.get_qtwindow()
            if host is not None:
                right = host.frameGeometry().right()
                if right + width + 12 <= screen.right():
                    x = right + 8
        except Exception:
            pass
        win.move(max(screen.left(), x), screen.top() + 40)

    # -- settings ------------------------------------------------------------------ #
    def _refresh_model_label(self):
        provider = "Claude" if self.settings.get("provider") == "anthropic" else "OpenAI"
        model = config.model_for(self.settings) or "sin modelo"
        policy = {"heavy": "confirma cálculos pesados", "every_step": "confirma cada paso",
                  "plan_only": "solo aprueba el plan"}.get(self.settings.get("approval_policy"), "")
        self.panel.set_model_label(f"{provider} · {model} · {policy}")

    def open_settings(self):
        dlg = AssistantSettingsDialog(self.panel, self.settings)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            changed_model = (config.model_for(dlg.settings) != config.model_for(self.settings)
                             or dlg.settings.get("provider") != self.settings.get("provider"))
            self.settings = dlg.settings
            self._refresh_model_label()
            self._sync_external_server()
            if changed_model and self.session is not None:
                self.panel.add_notice("Modelo cambiado: la próxima petición empieza una conversación nueva.")
                self.session = None

    def _sync_external_server(self):
        want = bool(self.settings.get("mcp_server_enabled"))
        port = int(self.settings.get("mcp_server_port", 8765))
        if self.external is not None and (not want or self.external.port != port):
            self.external.stop()
            self.external = None
        if want and self.external is None:
            try:
                server = build_server(self.registry, run_on_main=self.runner.run, context=self.context,
                                      confirm=self._confirm_external,
                                      approval_policy=self.settings.get("approval_policy", "heavy"))
                self.external = ExternalMcpServer(server, port=port)
                url = self.external.start()
                self.panel.add_notice(f"Servidor MCP activo en {url}")
            except Exception as exc:
                self.external = None
                self.panel.add_error(f"No se pudo iniciar el servidor MCP: {exc}")

    def _confirm_external(self, spec, args):
        def ask():
            box = QtWidgets.QMessageBox(self.panel)
            box.setWindowTitle("Leonardo AI: external request")
            box.setText(f"An external MCP client wants to run «{spec.title}».")
            box.setInformativeText(", ".join(f"{k}={v}" for k, v in args.items()) or "(no arguments)")
            box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            return box.exec_() == QtWidgets.QMessageBox.Yes
        return bool(self.runner.run(ask))

    # -- conversation ------------------------------------------------------------ #
    def _new_session(self):
        provider = config.make_provider(self.settings, SYSTEM_PROMPT, tools=[])
        policy = self.settings.get("approval_policy", "heavy")
        if policy not in APPROVAL_POLICIES:
            policy = "heavy"
        return AgentSession(provider, self.registry, self.adapter, execute=self.mcp_client.call_tool,
                            approval_policy=policy, max_tool_calls=int(self.settings.get("max_tool_calls", 60)))

    def new_chat(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.session = None
        _headless.reset()
        self.panel.clear()
        self.panel.set_usage({})

    def send(self, text):
        if self.worker is not None and self.worker.isRunning():
            return
        if self.session is None:
            try:
                self.session = self._new_session()
            except ProviderError as exc:
                self.panel.add_error(str(exc))
                self.open_settings()
                return
        self.panel.set_busy(True)
        self.panel.set_status("thinking")
        self.worker = AgentWorker(self.session, text)
        self.worker.finished_request.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def stop(self):
        if self.session is not None:
            self.session.cancel()
        self.adapter.release()
        # a job left running must not keep Leonardo's message boxes captured
        QtCore.QTimer.singleShot(3000, lambda: _headless.reset() if not self._busy() else None)

    def _busy(self):
        return self.worker is not None and self.worker.isRunning()

    def _on_finished(self, outcome):
        self.panel.set_busy(False)
        self.panel.end_text()
        if outcome == "cancelled":
            self.panel.set_status("cancelled")
        elif outcome in ("done", "stopped"):
            self.panel.set_status("idle")

    def _on_failed(self, message):
        self.panel.set_busy(False)
        self.panel.set_status("error")
        self.panel.add_error(message)

    # -- orchestrator events / questions -------------------------------------------- #
    def _cost_text(self, usage):
        prices = CLAUDE_PRICES.get(config.model_for(self.settings))
        if self.settings.get("provider") != "anthropic" or not prices:
            return ""
        cost = usage.get("input_tokens", 0) / 1e6 * prices[0] + usage.get("output_tokens", 0) / 1e6 * prices[1]
        return f"≈ ${cost:.3f}"

    def _on_event(self, kind, payload):
        p = self.panel
        if kind == "text":
            p.append_text(payload.get("text", ""))
        elif kind == "thinking":
            p.append_thinking(payload.get("text", ""))
        elif kind == "text_end":
            p.end_text()
        elif kind == "status":
            p.set_status(payload.get("state", "idle"))
        elif kind == "usage":
            p.set_usage(payload, self._cost_text(payload))
        elif kind == "tool":
            p.tool_event(payload)
        elif kind == "progress":
            p.progress(payload.get("text", ""))
        elif kind == "step":
            p.mark_step(payload.get("step", 0), payload.get("status", ""), payload.get("summary", ""))
        elif kind == "plan":
            pass            # shown by the approval question
        elif kind == "error":
            p.add_error(payload.get("message", "error"))

    def _on_question(self, kind, payload):
        self._pending_kind = kind
        self.panel.set_status("waiting")
        if kind == "plan":
            self.panel.show_plan(payload.get("plan", {}), interactive=True)
        elif kind == "confirm_tool":
            self.panel.ask_confirm(payload)
        else:
            self.panel.ask_question(payload)

    def _on_answer(self, data):
        self.panel.set_status("executing" if self._pending_kind != "question" else "thinking")
        self.adapter.answer(data)

    def shutdown(self):
        self.stop()
        _headless.reset()
        if self.external is not None:
            self.external.stop()
        self.mcp_client.close()
