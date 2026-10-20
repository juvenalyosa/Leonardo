"""
Ramachandran window: phi/psi of a PyMOL object (one structure, or every state of a trajectory / NMR
ensemble) against the favoured and allowed regions of high-resolution PDB chains, with the outliers listed
(click one to see it) and the figure saved as a PNG.  The analysis is ``ramachandran_runtime``.
"""
from __future__ import annotations

import os
import shutil
import time

from pymol.Qt import QtCore, QtGui, QtWidgets

from . import ramachandran_runtime as rama
from .protein_prep_dialog import _polymer_objects

TITLE = "Ramachandran Plot"


class RamachandranDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, workspace=None, submit_task=None):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.resize(1080, 900)
        self._workspace = workspace or (lambda: os.path.join(os.path.expanduser("~"), "leonardo_ramachandran"))
        self._submit_task = submit_task
        self._png = ""
        self._pixmap = None
        self._object = ""
        self._build_ui()
        self.refresh_objects()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Backbone φ/ψ of every residue against the favoured (98%) and allowed (99.95%) regions of "
            "high-resolution PDB chains, in the six MolProbity classes. A good structure has more than 98% of "
            "residues favoured and fewer than 0.05% outliers.")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Object:"))
        self.cmb_object = QtWidgets.QComboBox()
        self.cmb_object.setMinimumWidth(200)
        row.addWidget(self.cmb_object, 1)
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_objects)
        row.addWidget(self.btn_refresh)
        row.addWidget(QtWidgets.QLabel("Chains:"))
        self.edit_chains = QtWidgets.QLineEdit()
        self.edit_chains.setPlaceholderText("all (or A,B)")
        self.edit_chains.setMaximumWidth(110)
        row.addWidget(self.edit_chains)
        self.cb_states = QtWidgets.QCheckBox("All states (trajectory / ensemble)")
        row.addWidget(self.cb_states)
        self.btn_plot = QtWidgets.QPushButton("Plot")
        self.btn_plot.setDefault(True)
        self.btn_plot.clicked.connect(self._on_plot)
        row.addWidget(self.btn_plot)
        outer.addLayout(row)

        self.lbl_summary = QtWidgets.QLabel("")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        outer.addWidget(self.lbl_summary)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.lbl_image = QtWidgets.QLabel("Choose an object and press Plot.")
        self.lbl_image.setAlignment(QtCore.Qt.AlignCenter)
        self.scroll.setWidget(self.lbl_image)
        split.addWidget(self.scroll)
        side = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.addWidget(QtWidgets.QLabel("Outliers (click to show):"))
        self.list_out = QtWidgets.QListWidget()
        self.list_out.itemClicked.connect(self._show_residue)
        sv.addWidget(self.list_out, 1)
        split.addWidget(side)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)

        bottom = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save image...")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_image)
        bottom.addWidget(self.btn_save)
        bottom.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        bottom.addWidget(btn_close)
        outer.addLayout(bottom)

    def refresh_objects(self):
        current = self.cmb_object.currentText()
        self.cmb_object.clear()
        objs = _polymer_objects()
        self.cmb_object.addItems(objs)
        if current in objs:
            self.cmb_object.setCurrentText(current)

    def _on_plot(self):
        from pymol import cmd

        obj = self.cmb_object.currentText().strip()
        if not obj:
            QtWidgets.QMessageBox.information(self, TITLE, "Load a protein first.")
            return
        all_states = self.cb_states.isChecked() and cmd.count_states(obj) > 1
        folder = self._workspace()
        os.makedirs(folder, exist_ok=True)
        pdb = os.path.join(folder, time.strftime(f"{obj}_%Y%m%d_%H%M%S.pdb"))
        cmd.save(pdb, f"({obj}) and polymer and not alt B+C+D", state=0 if all_states else 1)
        png = os.path.splitext(pdb)[0] + "_rama.png"
        chains = [c.strip() for c in self.edit_chains.text().split(",") if c.strip()] or None
        self._object = obj
        self.btn_plot.setEnabled(False)
        self.lbl_summary.setText("Computing…")

        def work():
            return rama.analyze(pdb, png, chains=chains, models="all" if all_states else "first",
                                title=f"Ramachandran plot: {obj}")

        if self._submit_task is None:
            try:
                self._done(work(), png)
            except Exception as exc:
                self._failed(str(exc))
            return
        self._submit_task(work, lambda res: self._done(res, png), self._failed,
                          lambda: self.btn_plot.setEnabled(True))

    def _failed(self, msg):
        self.btn_plot.setEnabled(True)
        self.lbl_summary.setText(f"Failed: {msg}")

    def _done(self, result, png):
        self.btn_plot.setEnabled(True)
        rows, stats = result
        if not rows:
            self.lbl_summary.setText("No residue has both φ and ψ.")
            return
        self._png = png
        self._pixmap = QtGui.QPixmap(png)
        self._fit_image()
        self.btn_save.setEnabled(True)
        frames = f" × {stats['models']} frames" if stats.get("models", 1) > 1 else ""
        self.lbl_summary.setText(
            f"<b>{self._object}</b>: {stats['residues']} residues{frames} · favoured "
            f"<b>{stats['favoured_pct']:.1f}%</b> · allowed {stats['allowed_pct']:.1f}% · outliers "
            f"<b>{stats['outliers_pct']:.2f}%</b> — {stats['verdict']}")
        self.list_out.clear()
        for o in stats["outliers"]:
            extra = f"  ({o['fraction_of_models']:.0%} of frames)" if "fraction_of_models" in o else ""
            item = QtWidgets.QListWidgetItem(f"{o['residue']}  φ {o['phi']:.0f}°  ψ {o['psi']:.0f}°{extra}")
            item.setData(QtCore.Qt.UserRole, o["residue"])
            self.list_out.addItem(item)
        if not stats["outliers"]:
            self.list_out.addItem("none")

    def _fit_image(self):
        if self._pixmap is None or self._pixmap.isNull():
            return
        width = max(400, self.scroll.viewport().width() - 8)
        self.lbl_image.setPixmap(self._pixmap.scaledToWidth(width, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_image()

    def _show_residue(self, item):
        from pymol import cmd

        label = item.data(QtCore.Qt.UserRole)
        if not label or not self._object:
            return
        chain, _sep, rest = str(label).rpartition(":")
        resi = rest[3:] or rest                       # after the 3-letter name: number + insertion code
        resi = "\\" + resi if resi.startswith("-") else resi
        sel = f"({self._object}) and resi {resi}" + (f" and chain {chain}" if chain else "")
        cmd.show("sticks", f"({sel}) and not hydro")
        cmd.color("red", f"({sel}) and elem C")
        cmd.zoom(sel, 6)
        cmd.select("rama_residue", sel, enable=1)

    def _save_image(self):
        if not self._png:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Ramachandran plot",
                                                        os.path.basename(self._png), "PNG (*.png)")
        if path:
            shutil.copyfile(self._png, path)
