"""Pipeline library window: pick a saved pipeline, change its molecules (parameters) and run it."""

import json

from pymol.Qt import QtCore, QtWidgets

from .. import pipelines as P
from .style import STYLESHEET


class PipelineDialog(QtWidgets.QDialog):
    """``run_text`` holds the request for the assistant after Run (the dialog then closes)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Leonardo pipelines")
        self.setStyleSheet(STYLESHEET)
        self.resize(720, 520)
        self.run_text = ""
        self._fields = {}
        lay = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("Saved pipelines"))
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self._show)
        left.addWidget(self.list, 1)
        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(self._delete)
        left.addWidget(self.btn_delete)
        lay.addLayout(left, 2)
        right = QtWidgets.QVBoxLayout()
        self.lbl_title = QtWidgets.QLabel("")
        self.lbl_title.setObjectName("title")
        self.lbl_title.setWordWrap(True)
        right.addWidget(self.lbl_title)
        self.lbl_steps = QtWidgets.QLabel("")
        self.lbl_steps.setWordWrap(True)
        self.lbl_steps.setObjectName("muted")
        right.addWidget(self.lbl_steps)
        right.addWidget(QtWidgets.QLabel("Molecules and settings for this run:"))
        self.form_host = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.form_host)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.form_host)
        right.addWidget(scroll, 1)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        self.btn_run = QtWidgets.QPushButton("Run with these molecules")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run)
        right.addLayout(row)
        lay.addLayout(right, 5)
        self._items = []
        self._reload()

    def _reload(self):
        self._items = P.list_all()
        self.list.clear()
        for it in self._items:
            self.list.addItem(it["name"])
        empty = not self._items
        self.btn_run.setEnabled(not empty)
        self.btn_delete.setEnabled(not empty)
        if empty:
            self.lbl_title.setText("No saved pipelines yet")
            self.lbl_steps.setText("Run a workflow with the assistant and ask it to save it as a pipeline "
                                   "(e.g. \"save this as a pipeline called docking+MD\").")
        else:
            self.list.setCurrentRow(0)

    def _show(self, row):
        while self.form.rowCount():
            self.form.removeRow(0)
        self._fields = {}
        if not 0 <= row < len(self._items):
            return
        it = self._items[row]
        self.lbl_title.setText(f"{it['name']}" + (f" — {it['description']}" if it.get("description") else ""))
        self.lbl_steps.setText("Steps: " + " → ".join(f"{i + 1}. {t}" for i, t in enumerate(it["steps"])))
        for name, default in it["parameters"].items():
            if isinstance(default, (list, dict)):
                w = QtWidgets.QPlainTextEdit(json.dumps(default, indent=1))
                w.setMaximumHeight(110)
            else:
                w = QtWidgets.QLineEdit("" if default is None else str(default))
            self.form.addRow(name, w)
            self._fields[name] = (w, default)

    def values(self):
        out = {}
        for name, (w, default) in self._fields.items():
            text = w.toPlainText() if isinstance(w, QtWidgets.QPlainTextEdit) else w.text()
            if isinstance(default, (list, dict)):
                try:
                    out[name] = json.loads(text)
                except Exception:
                    out[name] = text
            elif isinstance(default, bool):
                out[name] = text.strip().lower() in ("1", "true", "yes", "si", "sí")
            elif isinstance(default, int):
                try:
                    out[name] = int(text)
                except ValueError:
                    out[name] = text
            elif isinstance(default, float):
                try:
                    out[name] = float(text)
                except ValueError:
                    out[name] = text
            else:
                try:                              # no saved type: numbers/lists/JSON as such, else text
                    out[name] = json.loads(text) if default is None else text.strip()
                except ValueError:
                    out[name] = text.strip()
        return out

    def _run(self):
        row = self.list.currentRow()
        if not 0 <= row < len(self._items):
            return
        name = self._items[row]["name"]
        vals = self.values()
        self.run_text = (f"Run the saved pipeline \"{name}\" with these parameter values: "
                         + json.dumps(vals, ensure_ascii=False))
        self.accept()

    def _delete(self):
        row = self.list.currentRow()
        if not 0 <= row < len(self._items):
            return
        name = self._items[row]["name"]
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Delete pipeline")
        box.setText(f"Delete the pipeline “{name}”?")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if box.exec_() == QtWidgets.QMessageBox.Yes:
            P.delete(name)
            self._reload()


def ask_pipeline_run(parent=None):
    dlg = PipelineDialog(parent)
    return dlg.run_text if dlg.exec_() == QtWidgets.QDialog.Accepted else ""


__all__ = ["PipelineDialog", "ask_pipeline_run", "QtCore"]
