"""
Cryptic Pockets window: run the cryptic-site analysis (Crypsis 0.2) on a PyMOL object and browse the sites (click
one to show it).  The analysis is ``cryptic_pocket_runtime``; the object is coloured by probability.
"""
from __future__ import annotations

import os
import time

from pymol.Qt import QtCore, QtWidgets

from . import cryptic_pocket_runtime as cp
from .protein_prep_dialog import _polymer_objects

TITLE = "Cryptic Pockets"
# (label, model, esm): the hybrid entries need PyTorch + fair-esm
MODEL_CHOICES = [("Physics (fast, no extra packages)", "physics", "650M"),
                 ("Hybrid + ESM-2 650M (most accurate, ~5.4 GB)", "hybrid", "650M"),
                 ("Hybrid + ESM-2 150M (~1.4 GB)", "hybrid", "150M"),
                 ("v1 (Leonardo's original logistic model)", "v1", "650M")]


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
            "Finds where pockets can open with the protein's motions (Crypsis 0.2): ensembles from the normal "
            "modes of an elastic network, a buried-pocket grid in every conformation, physical descriptors of "
            "each residue and a probability from models trained on CryptoBench (test AUC 0.86 physics, 0.89 "
            "hybrid). Sites are ranked; each says whether it is already open, grows or forms with motion. "
            "Predictions to follow up with MD or docking, not proof.")
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
        row.addWidget(QtWidgets.QLabel("Model:"))
        self.cmb_model = QtWidgets.QComboBox()
        for label, _model, _esm in MODEL_CHOICES:
            self.cmb_model.addItem(label)
        if not cp.have_esm():
            for i, (_label, model, _esm) in enumerate(MODEL_CHOICES):
                if model == "hybrid":
                    item = self.cmb_model.model().item(i)
                    item.setEnabled(False)
                    item.setToolTip("needs fair-esm in Leonardo's environment: run the installer again, or "
                                    "pip install fair-esm")
        best = cp.default_model()
        self.cmb_model.setCurrentIndex(next(i for i, c in enumerate(MODEL_CHOICES)
                                            if (c[1], c[2]) == best or (c[1] == best[0] == "physics")))
        row.addWidget(self.cmb_model)
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
        self.btn_box = QtWidgets.QPushButton("Use as docking box")
        self.btn_box.setToolTip("Centre Leonardo's docking box on the selected site (Pocket & Docking uses it)")
        self.btn_box.setEnabled(False)
        self.btn_box.clicked.connect(self._use_as_box)
        bottom.addWidget(self.btn_box)
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
        _label, model, esm = MODEL_CHOICES[max(0, self.cmb_model.currentIndex())]
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Computing the ensembles and their pockets…"
                                + (" then ESM-2 (the first run downloads its weights)" if model == "hybrid" else ""))

        def work():
            return cp.analyze(cp.read_chain(path), model=model, esm=esm)

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
        model = f"hybrid + ESM-2 {result['esm']}" if result.get("esm") else result.get("model", "")
        self.lbl_status.setText(
            f"<b>{self._object}</b>: {len(result['sites'])} candidate site(s), {model} model; coloured by "
            f"probability (blue → red); selections {self._object}_cryptic_N. {engine}.")
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
        self.btn_box.setEnabled(True)
        name = f"{self._object}_cryptic_{site['rank']}"
        if name in cmd.get_names("selections"):
            cmd.show("sticks", f"{name} and not name N+C+O")
            cmd.zoom(name, 5)
            cmd.enable(name)

    def _use_as_box(self):
        rows = {i.row() for i in self.table.selectedItems()}
        if not rows or not self._result:
            return
        site = self._result["sites"][min(rows)]
        box = site["docking_box"]
        host = self.parent()
        spins = ("spin_cx", "spin_cy", "spin_cz", "spin_sx", "spin_sy", "spin_sz")
        if host is None or not all(hasattr(host, a) for a in spins):
            QtWidgets.QMessageBox.information(self, TITLE, "Open Leonardo's main window first.")
            return
        for attr, v in zip(spins, list(box["center"]) + list(box["size"])):
            getattr(host, attr).setValue(float(v))
        host._dock_grid_user_defined = True
        self.lbl_status.setText(
            f"Docking box set on site #{site['rank']}: centre {box['center']}, size {box['size']} A. "
            "Open Pocket & Docking to dock a ligand there.")
