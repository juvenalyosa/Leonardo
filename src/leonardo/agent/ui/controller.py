"""
Wires the assistant together inside Leonardo: tool registry + context, in-app MCP
server/client, optional external MCP server, provider sessions, worker threads and the
chat window (a separate window, so PyMOL's viewer keeps its size).
"""

from pymol.Qt import QtCore, QtWidgets

from .. import config, pipelines, presentation
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
        self.panel.pipelines_requested.connect(self.open_pipelines)
        self.panel.scene_requested.connect(self._recall_scene)
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
        model = config.model_for(self.settings) or "no model"
        policy = {"heavy": "confirms heavy calculations", "every_step": "confirms every step",
                  "plan_only": "plan approval only"}.get(self.settings.get("approval_policy"), "")
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
                self.panel.add_notice("Model changed: the next request starts a new conversation.")
                self.session = None
            elif self.session is not None:
                # approval policy / limits apply from now on, also to the conversation in progress
                policy = self.settings.get("approval_policy", "heavy")
                self.session.approval_policy = policy if policy in APPROVAL_POLICIES else "heavy"
                self.session.needs_confirmation = APPROVAL_POLICIES[self.session.approval_policy]
                self.session.max_tool_calls = int(self.settings.get("max_tool_calls", 300))

    def open_pipelines(self):
        """Pipeline library: the chosen pipeline and molecules become a request for the assistant."""
        from .pipeline_dialog import ask_pipeline_run

        text = ask_pipeline_run(self.panel)
        if not text:
            return
        if self._busy():
            self.panel.add_notice("The assistant is busy: run the pipeline when the current request finishes.")
            return
        self.panel.add_user(text)
        self.send(text)

    def _sync_external_server(self):
        want = bool(self.settings.get("mcp_server_enabled"))
        port = int(self.settings.get("mcp_server_port", 8765))
        policy = self.settings.get("approval_policy", "heavy")
        if self.external is not None and (not want or self.external.port != port
                                          or getattr(self, "_external_policy", policy) != policy):
            self.external.stop()
            self.external = None
        if want and self.external is None:
            try:
                server = build_server(self.registry, run_on_main=self.runner.run, context=self.context,
                                      confirm=self._confirm_external,
                                      approval_policy=self.settings.get("approval_policy", "heavy"))
                self.external = ExternalMcpServer(server, port=port)
                self._external_policy = policy
                url = self.external.start()
                self.panel.add_notice(f"MCP server running at {url}")
            except Exception as exc:
                self.external = None
                self.panel.add_error(f"Could not start the MCP server: {exc}")

    EXTERNAL_CONFIRM_TIMEOUT_S = 900.0

    def _confirm_external(self, spec, args, timeout=None):
        """
        Ask the user to allow a tool call from an external MCP client. Called on the MCP server's thread: the
        question is shown non-blocking (open() + a signal) and this thread waits for the answer. A blocking
        exec_() here ran a nested event loop inside a queued call on the main thread; Qt 5 on macOS 26 crashed
        in that loop (SIGBUS in -[BSSimpleAssertion dealloc]). Unanswered requests are declined after
        ``timeout`` seconds or when the assistant shuts down.
        """
        import threading

        done = threading.Event()
        verdict = {"ok": False}

        def show():
            box = QtWidgets.QMessageBox(self.panel)
            box.setWindowTitle("Leonardo AI: external request")
            box.setText(f"An external MCP client wants to run «{spec.title}».")
            box.setInformativeText(", ".join(f"{k}={v}" for k, v in args.items()) or "(no arguments)")
            box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            box.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

            def finished(_code):
                verdict["ok"] = box.clickedButton() is box.button(QtWidgets.QMessageBox.Yes)
                done.set()
                self._external_boxes.discard(box)

            box.finished.connect(finished)
            self._external_boxes.add(box)
            self._external_waits.add(done)
            box.open()                      # returns at once: no nested event loop

        if not hasattr(self, "_external_boxes"):
            self._external_boxes, self._external_waits = set(), set()
        self.runner.run(show)
        answered = done.wait(self.EXTERNAL_CONFIRM_TIMEOUT_S if timeout is None else float(timeout))
        self._external_waits.discard(done)
        return bool(answered and verdict["ok"])

    # -- conversation ------------------------------------------------------------ #
    def _new_session(self):
        provider = config.make_provider(self.settings, SYSTEM_PROMPT, tools=[])
        policy = self.settings.get("approval_policy", "heavy")
        if policy not in APPROVAL_POLICIES:
            policy = "heavy"
        return AgentSession(provider, self.registry, self.adapter, execute=self.mcp_client.call_tool,
                            approval_policy=policy, max_tool_calls=int(self.settings.get("max_tool_calls", 300)))

    def new_chat(self):
        if self.worker is not None and self.worker.isRunning():
            return
        self.session = None
        _headless.reset()
        pipelines.clear_history()          # save_pipeline records only this chat's steps
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
        self.adapter.reset()
        self.worker = AgentWorker(self.session, text)
        self.worker.finished_request.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def stop(self):
        running = self.worker.session if self.worker is not None and self.worker.isRunning() else None
        for session in {id(s): s for s in (running, self.session) if s is not None}.values():
            session.cancel()
        self.adapter.release()
        # a job left running must not keep Leonardo's message boxes captured
        QtCore.QTimer.singleShot(3000, lambda: _headless.reset() if not self._busy() else None)

    def _recall_scene(self, key):
        try:
            presentation.recall_scene(key)
        except Exception:
            self.panel.add_notice(f"The view {key} no longer exists (the scene was cleared or reset).")

    def _busy(self):
        return self.worker is not None and self.worker.isRunning()

    def _watch_jobs(self):
        """Jobs the model left running (it ended its turn) are polled here, so their end releases
        Leonardo's message boxes (captured while an assistant job runs) without waiting for a new request."""
        if getattr(self, "_job_timer", None) is None:
            self._job_timer = QtCore.QTimer(self.panel)
            self._job_timer.setInterval(2000)

            def tick():
                if self._busy():
                    return
                try:
                    running = self._poll_and_report_jobs()
                except Exception:
                    running = 0
                if not running:
                    self._job_timer.stop()
            self._job_timer.timeout.connect(tick)
        self._job_timer.start()

    JOB_REPORT_EVERY_S = 1800.0

    def _poll_and_report_jobs(self, now=None):
        """
        Poll the jobs left running and post a progress note in the chat every JOB_REPORT_EVERY_S (state, elapsed
        time, the job's own progress line), and one when a job ends. Returns how many still run.
        """
        import time as _time

        now = _time.time() if now is None else float(now)
        reports = self.__dict__.setdefault("_job_reports", {})
        running = 0
        for job in self.context.jobs.list():
            job_id = job["job_id"]
            if job_id in reports and reports[job_id] is None:
                continue                                    # end already reported
            if job.get("status") != "running" and job_id not in reports:
                continue                                    # ended before it was ever followed here
            st = self.context.jobs.status(job_id)
            state = st.get("status")
            name = job.get("description") or job.get("kind") or job_id
            elapsed = float(st.get("elapsed_s", job.get("elapsed_s", 0)) or 0)
            started = now - elapsed
            if state in ("running", "cancelling"):
                running += 1
                due = reports.setdefault(job_id, started + self.JOB_REPORT_EVERY_S)
                if now >= due:
                    hours, minutes = int(elapsed // 3600), int(elapsed % 3600 // 60)
                    progress = str(st.get("progress") or "").strip()
                    self.panel.add_notice(f"Background job {job_id} ({name}) is still running after "
                                          f"{hours} h {minutes:02d} min" + (f": {progress}" if progress else "."))
                    reports[job_id] = due + self.JOB_REPORT_EVERY_S
                continue
            if job_id in reports or elapsed >= self.JOB_REPORT_EVERY_S:
                if state == "done":
                    self.panel.add_notice(f"Background job {job_id} ({name}) finished. Ask the assistant for the "
                                          "results (job_status).")
                elif state == "failed":
                    self.panel.add_error(f"Background job {job_id} ({name}) failed: {str(st.get('error', ''))[:300]}")
                elif state == "cancelled":
                    self.panel.add_notice(f"Background job {job_id} ({name}) was cancelled.")
            reports[job_id] = None
        return running

    def _on_finished(self, outcome):
        self._watch_jobs()
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
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(5000)          # never destroy a running QThread
        _headless.reset()
        for waiting in list(getattr(self, "_external_waits", ())):
            waiting.set()                   # pending external requests: declined
        if self.external is not None:
            self.external.stop()
        self.mcp_client.close()
