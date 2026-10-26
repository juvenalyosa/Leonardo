"""
Advanced Tools window: every tool of the AI assistant as a form, without the assistant.

Tools that have no workflow window of their own (membranes, materials, disulfide design, interfaces,
validated systems with cofactors, enhanced sampling, QM analysis, pipelines) are listed first. Each form is
built from the tool's JSON schema (``leonardo.agent.tool_forms``) and runs through the same registry, context
and job store as the assistant: long calculations start a background job that this window follows, and its
result is loaded into PyMOL exactly as when the assistant runs it. Nothing here needs an API key.
"""

import json
import threading

from pymol.Qt import QtCore, QtWidgets

from .. import tool_forms as TF
from ..registry import ToolError

JOB_POLL_MS = 2000


class _Relay(QtCore.QObject):
    """Brings the result of a tool run on a worker thread back to the GUI thread."""

    finished = QtCore.Signal(object, object)       # result dict, error text


class AdvancedToolsDialog(QtWidgets.QDialog):
    def __init__(self, registry, context, parent=None):
        super().__init__(parent)
        self.registry = registry
        self.context = context
        self.setWindowTitle("Advanced Tools")
        self.resize(1180, 780)
        self._spec = None
        self._fields = []
        self._widgets = {}
        self._job_id = ""
        self._running = False
        self._relay = _Relay(self)
        self._relay.finished.connect(self._on_finished)
        self._job_timer = QtCore.QTimer(self)
        self._job_timer.setInterval(JOB_POLL_MS)
        self._job_timer.timeout.connect(self._poll_job)

        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Every tool of the AI assistant, run directly from a form. Tools without a workflow window of their "
            "own are listed first. Empty fields keep the tool's default (shown in grey)."
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        lay.addWidget(split, 1)

        left = QtWidgets.QWidget()
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search tools (e.g. membrane, disulfide, orbitals)")
        self.search.textChanged.connect(self._fill_tree)
        left_lay.addWidget(self.search)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.currentItemChanged.connect(self._on_tool_selected)
        left_lay.addWidget(self.tree, 1)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(8, 0, 0, 0)
        self.lbl_title = QtWidgets.QLabel("Choose a tool")
        self.lbl_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.lbl_title.setWordWrap(True)
        right_lay.addWidget(self.lbl_title)
        self.lbl_meta = QtWidgets.QLabel("")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setStyleSheet("color: #5B626B;")
        right_lay.addWidget(self.lbl_meta)
        # long tool descriptions scroll here instead of pushing the form out of the window
        self.lbl_desc = QtWidgets.QTextBrowser()
        self.lbl_desc.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lbl_desc.setMaximumHeight(96)
        self.lbl_desc.setMinimumHeight(48)
        right_lay.addWidget(self.lbl_desc)
        self.form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_host)
        self.form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.form_host)
        scroll.setMinimumHeight(240)
        right_lay.addWidget(scroll, 3)
        row = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        row.addWidget(self.lbl_status, 1)
        self.btn_cancel_job = QtWidgets.QPushButton("Cancel Job")
        self.btn_cancel_job.clicked.connect(self._cancel_job)
        self.btn_cancel_job.setVisible(False)
        row.addWidget(self.btn_cancel_job)
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.setProperty("leonardoRole", "primary")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run)
        right_lay.addLayout(row)
        right_lay.addWidget(QtWidgets.QLabel("Result"))
        self.result = QtWidgets.QTextBrowser()
        self.result.setOpenExternalLinks(True)
        self.result.setMinimumHeight(150)
        right_lay.addWidget(self.result, 2)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setSizes([340, 840])

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        lay.addLayout(close_row)
        self._fill_tree()

    # -- tool list ------------------------------------------------------------------------------ #
    def _fill_tree(self, *_):
        query = self.search.text()
        current = self._spec.name if self._spec is not None else ""
        self.tree.blockSignals(True)
        self.tree.clear()
        select = None
        for section, groups in TF.group_tools(self.registry.specs()):
            sec_item = None
            for title, specs in groups:
                shown = [s for s in specs if TF.matches(s, query)]
                if not shown:
                    continue
                if sec_item is None:
                    sec_item = QtWidgets.QTreeWidgetItem([section])
                    sec_item.setFlags(QtCore.Qt.ItemIsEnabled)
                    font = sec_item.font(0)
                    font.setBold(True)
                    sec_item.setFont(0, font)
                    self.tree.addTopLevelItem(sec_item)
                group_item = QtWidgets.QTreeWidgetItem([f"{title} ({len(shown)})"])
                group_item.setFlags(QtCore.Qt.ItemIsEnabled)
                sec_item.addChild(group_item)
                for spec in shown:
                    item = QtWidgets.QTreeWidgetItem([spec.title[:1].upper() + spec.title[1:]])
                    item.setData(0, QtCore.Qt.UserRole, spec.name)
                    item.setToolTip(0, spec.name)
                    group_item.addChild(item)
                    if spec.name == current:
                        select = item
                sec_item.setExpanded(True)
                group_item.setExpanded(bool(query) or section == "Without a window")
        self.tree.blockSignals(False)
        if select is not None:
            self.tree.setCurrentItem(select)

    def _on_tool_selected(self, item, _previous=None):
        name = item.data(0, QtCore.Qt.UserRole) if item is not None else None
        if not name:
            return
        spec = self.registry.get(name)
        if spec is None or (self._spec is not None and spec.name == self._spec.name):
            return
        self._spec = spec
        self._build_form(spec)

    # -- form ----------------------------------------------------------------------------------- #
    def _object_names(self):
        try:
            from pymol import cmd

            return list(cmd.get_names("objects"))
        except Exception:
            return []

    def _build_form(self, spec):
        while self.form.rowCount():
            self.form.removeRow(0)
        self._widgets = {}
        self._fields = TF.field_specs(spec)
        self.lbl_title.setText(spec.title[:1].upper() + spec.title[1:])
        self.lbl_meta.setText(f"{spec.name} · {TF.PERMISSION_LABELS.get(spec.permission, spec.permission)}")
        self.lbl_desc.setPlainText(spec.description)
        objects = self._object_names()
        for field in self._fields:
            widget = self._field_widget(field, objects)
            label = QtWidgets.QLabel(field["name"] + (" *" if field["required"] else ""))
            tip = field["description"] or ""
            label.setToolTip(tip)
            widget.setToolTip(tip)
            if not isinstance(widget, QtWidgets.QCheckBox):
                widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            self.form.addRow(label, widget)
            self._widgets[field["name"]] = widget
        if not self._fields:
            self.form.addRow(QtWidgets.QLabel("This tool has no parameters."))
        self.btn_run.setEnabled(not self._running)
        self.result.clear()
        self.lbl_status.setText("")

    def _field_widget(self, field, objects):
        kind, default = field["kind"], TF.default_text(field)
        if kind == "bool":
            box = QtWidgets.QCheckBox(field["description"][:90] if field["description"] else "")
            box.setChecked(bool(field["default"]) if field["has_default"] else False)
            return box
        if kind == "enum":
            combo = QtWidgets.QComboBox()
            if not field["required"]:
                combo.addItem(f"(default: {default})" if default else "(default)", "")
            for value in field["enum"]:
                combo.addItem(str(value), str(value))
            return combo
        if field["is_object"]:
            combo = QtWidgets.QComboBox()
            combo.setEditable(True)
            combo.addItems(objects)
            combo.setCurrentIndex(-1 if not field["required"] or not objects else 0)
            combo.lineEdit().setPlaceholderText(default or ("PyMOL object" if field["required"] else ""))
            return combo
        edit = QtWidgets.QLineEdit()
        hint = {"list_str": "comma-separated", "list_int": "comma-separated integers",
                "list_number": "comma-separated numbers", "json": "JSON"}.get(kind, "")
        edit.setPlaceholderText(" · ".join(x for x in (default, hint) if x))
        return edit

    def _raw_values(self):
        raw = {}
        for field in self._fields:
            w = self._widgets.get(field["name"])
            if isinstance(w, QtWidgets.QCheckBox):
                raw[field["name"]] = w.isChecked()
            elif isinstance(w, QtWidgets.QComboBox) and not w.isEditable():
                raw[field["name"]] = w.currentData()
            elif isinstance(w, QtWidgets.QComboBox):
                raw[field["name"]] = w.currentText()
            elif isinstance(w, QtWidgets.QLineEdit):
                raw[field["name"]] = w.text()
        return raw

    # -- running -------------------------------------------------------------------------------- #
    def _run(self):
        spec = self._spec
        if spec is None or self._running:
            return
        try:
            args = TF.collect_args(self._fields, self._raw_values())
        except ValueError as exc:
            self._show_error(str(exc))
            return
        if spec.permission == "destructive":
            answer = QtWidgets.QMessageBox.question(
                self, "Advanced Tools", f"«{spec.title}» deletes or overwrites objects or files. Run it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes:
                return
        self._set_running(True, f"Running {spec.title}…")
        self.result.setPlainText(f"{spec.name}({json.dumps(args, ensure_ascii=False)})")
        if spec.main_thread:
            # PyMOL and the Leonardo dialog are not thread-safe; long work starts a job and returns at once
            QtCore.QTimer.singleShot(0, lambda: self._call(spec.name, args))
        else:
            threading.Thread(target=self._call, args=(spec.name, args), daemon=True).start()

    def _call(self, name, args):
        try:
            result = self.registry.call(name, args, context=self.context)
            self._relay.finished.emit(result, None)
        except ToolError as exc:
            self._relay.finished.emit(None, str(exc))
        except Exception as exc:          # never take the window down with a tool bug
            self._relay.finished.emit(None, f"{type(exc).__name__}: {exc}")

    def _on_finished(self, result, error):
        if error:
            self._set_running(False, "")
            self._show_error(error)
            return
        job = result.get("job_id") if isinstance(result, dict) else ""
        if job and result.get("status") in ("running", "cancelling"):
            self._job_id = str(job)
            self.btn_cancel_job.setVisible(True)
            self.lbl_status.setText(f"Job {job} running… (the AI assistant sees it too)")
            self._show_result(result)
            self._job_timer.start()
            return
        self._set_running(False, "Done.")
        self._show_result(result)

    def _poll_job(self):
        if not self._job_id:
            self._job_timer.stop()
            return
        try:
            st = self.context.jobs.status(self._job_id)
        except Exception as exc:
            st = {"status": "failed", "error": str(exc)}
        state = st.get("status")
        if state in ("running", "cancelling"):
            progress = str(st.get("progress") or "").strip()
            elapsed = int(float(st.get("elapsed_s", 0) or 0))
            self.lbl_status.setText(f"Job {self._job_id} {state} · {elapsed // 60} min {elapsed % 60:02d} s"
                                    + (f" · {progress}" if progress else ""))
            return
        self._job_timer.stop()
        job, self._job_id = self._job_id, ""
        self.btn_cancel_job.setVisible(False)
        if state == "done":
            self._set_running(False, f"Job {job} finished.")
            self._show_result(st.get("result") if isinstance(st.get("result"), dict) else st)
        elif state == "cancelled":
            self._set_running(False, f"Job {job} was cancelled.")
        else:
            self._set_running(False, "")
            self._show_error(f"Job {job} {state}: {st.get('error', '')}")

    def _cancel_job(self):
        if self._job_id:
            try:
                self.context.jobs.cancel(self._job_id)
            except Exception as exc:
                self._show_error(f"Could not cancel job {self._job_id}: {exc}")

    def _set_running(self, running, status):
        self._running = bool(running)
        self.btn_run.setEnabled(not running and self._spec is not None)
        self.lbl_status.setText(status)

    def _show_result(self, result):
        summary = TF.result_summary(result)
        details = json.dumps({k: v for k, v in (result or {}).items() if k != "_card"}, indent=1, default=str,
                             ensure_ascii=False)
        if len(details) > 20000:
            details = details[:20000] + "\n… (truncated)"
        self.result.setMarkdown(summary + "\n\n---\n\n```\n" + details + "\n```")

    def _show_error(self, message):
        self.lbl_status.setText("Error")
        self.result.setMarkdown(f"**Error:** {message}")

    def closeEvent(self, event):
        # a job keeps running when the window closes (the Task Monitor and the assistant still see it)
        self._job_timer.stop()
        super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if self._job_id:
            self._job_timer.start()
