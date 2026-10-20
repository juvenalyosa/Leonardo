"""
Cryptic Pockets window: run Leonardo's cryptic-site analysis on a PyMOL object and browse the sites (click
one to show it).  The analysis is ``cryptic_pocket_runtime``; the object is coloured by probability.
"""
from __future__ import annotations

import os
import time

from pymol.Qt import QtCore, QtWidgets

from . import cryptic_pocket_runtime as cp
from .protein_prep_dialog import _polymer_objects

TITLE = "Cryptic Pockets"


class CrypticPocketDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, workspace=None, submit_task=None):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.resize(820, 620)
        self._workspace = workspace or (lambda: os.path.join(os.path.expanduser("~"), "leonardo_cryptic_pockets"))
        self._submit_task = submit_task
        self._object = ""
        self._result = None
        outer = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Finds where pockets can open with the protein's motions: an ensemble from the normal modes of an "
            "elastic network, a buried-pocket grid in every conformation, and a probability per residue "
            "(calibrated on CryptoBench). Sites are ranked; each says whether it is already open, grows or forms "
            "with motion. Predictions to follow up with MD or docking, not proof.")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Object:"))
        self.cmb_object = QtWidgets.QComboBox()
        self.cmb_object.setMinimumWidth(200)
        row.addWidget(self.cmb_object, 1)
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh_objects)
        row.addWidget(btn_refresh)
        row.addWidget(QtWidgets.QLabel("Chains:"))
        self.edit_chains = QtWidgets.QLineEdit()
        self.edit_chains.setPlaceholderText("all (or A,B)")
        self.edit_chains.setMaximumWidth(110)
        row.addWidget(self.edit_chains)
        self.btn_run = QtWidgets.QPushButton("Find cryptic pockets")
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self._on_run)
        row.addWidget(self.btn_run)
        outer.addLayout(row)
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        outer.addWidget(self.lbl_status)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["#", "Kind", "Residues", "Pocket A³ (structure → motion)",
                                              "Mean probability", "Lining residues"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._show_site)
        outer.addWidget(self.table, 1)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        bottom.addWidget(btn_close)
        outer.addLayout(bottom)
        self.refresh_objects()

    def refresh_objects(self):
        current = self.cmb_object.currentText()
        self.cmb_object.clear()
        objs = _polymer_objects()
        self.cmb_object.addItems(objs)
        if current in objs:
            self.cmb_object.setCurrentText(current)

    def _on_run(self):
        from pymol import cmd

        obj = self.cmb_object.currentText().strip()
        if not obj:
            QtWidgets.QMessageBox.information(self, TITLE, "Load a protein first.")
            return
        chains = [c.strip() for c in self.edit_chains.text().split(",") if c.strip()]
        sel = f"({obj}) and polymer.protein and not hydro and not alt B+C+D"
        if chains:
            sel += " and chain " + "+".join(chains)
        if cmd.count_atoms(sel + " and name CA") < 10:
            QtWidgets.QMessageBox.information(self, TITLE, "Need a protein with at least 10 residues.")
            return
        folder = self._workspace()
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime(f"{obj}_%Y%m%d_%H%M%S.pdb"))
        cmd.save(path, sel, state=1)
        self._object = obj
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Computing the ensemble and its pockets…")

        def work():
            return cp.analyze(cp.read_chain(path))

        if self._submit_task is None:
            try:
                self._done(work())
            except Exception as exc:
                self._failed(str(exc))
            self.btn_run.setEnabled(True)
            return
        self._submit_task(work, self._done, self._failed, lambda: self.btn_run.setEnabled(True))

    def _failed(self, msg):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"Failed: {msg}")

    def _done(self, result):
        from .agent.tools.cryptic import show_result

        self.btn_run.setEnabled(True)
        self._result = result
        show_result(self._object, result, top=len(result["sites"]))
        engine = "C++ grid" if result["extension"] else "numpy grid (build the C++ one: python -m leonardo.cryptic_build)"
        self.lbl_status.setText(
            f"<b>{self._object}</b>: {len(result['sites'])} candidate site(s); coloured by probability "
            f"(blue → red); selections {self._object}_cryptic_N. {engine}.")
        self.table.setRowCount(0)
        for s in result["sites"]:
            r = self.table.rowCount()
            self.table.insertRow(r)
            vals = [str(s["rank"]), s["kind"], str(s["n_residues"]),
                    f"{s['pocket_volume_structure_A3']:.0f} → {s['pocket_volume_ensemble_A3']:.0f}",
                    f"{s['mean_probability']:.2f}",
                    ", ".join(f"{n}{k.split('_', 1)[1]}" for n, k in zip(s["residue_names"], s["residues"]))]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.table.resizeColumnsToContents()

    def _show_site(self):
        from pymol import cmd

        rows = {i.row() for i in self.table.selectedItems()}
        if not rows or not self._result:
            return
        site = self._result["sites"][min(rows)]
        name = f"{self._object}_cryptic_{site['rank']}"
        if name in cmd.get_names("selections"):
            cmd.show("sticks", f"{name} and not name N+C+O")
            cmd.zoom(name, 5)
            cmd.enable(name)
