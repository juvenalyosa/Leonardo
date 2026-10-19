"""
Prepare Protein window: analyze what a structure needs, choose the options, run the preparation in the
background and see what changed (the prepared structure is loaded as a new PyMOL object).

The work is done by ``protein_prep_runtime`` in a child process (the same one the assistant's
``prepare_protein`` tool uses); this window only collects options and shows the report.
"""
from __future__ import annotations

import os
import re
import time

from pymol.Qt import QtCore, QtWidgets

from . import protein_prep_runtime as rt
from . import protein_prep_view as view

TITLE = "Prepare Protein"


def _polymer_objects():
    from pymol import cmd

    out = []
    for name in cmd.get_names("objects"):
        try:
            if cmd.get_type(name) == "object:molecule" and cmd.count_atoms(f"({name}) and polymer") > 0:
                out.append(name)
        except Exception:
            continue
    return out


class ProteinPrepDialog(QtWidgets.QDialog):
    """Analyze -> options -> run -> report, for one PyMOL object."""

    def __init__(self, parent=None, workspace=None, submit_task=None, infer_pdb_id=None, python=None):
        super().__init__(parent)
        self.setWindowTitle(TITLE)
        self.resize(980, 820)
        self._workspace = workspace or (lambda: os.path.join(os.path.expanduser("~"), "leonardo_protein_prep"))
        self._submit_task = submit_task
        self._infer_pdb_id = infer_pdb_id
        self._python = python
        self._proc = None
        self._job = None
        self._analysis = None
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self._poll)
        self._build_ui()
        self.refresh_objects()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Makes a structure ready for docking, MD or QM/MM: one conformation per residue, missing loops and "
            "side chains, protonation at your pH from PROPKA pKa values, Asn/Gln/His flips and His tautomers "
            "from reduce, hydrogens and a restrained minimization. The result is a new object; the original is "
            "not changed.")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        # structure
        box = QtWidgets.QGroupBox("1) Structure")
        row = QtWidgets.QHBoxLayout(box)
        row.addWidget(QtWidgets.QLabel("Object:"))
        self.cmb_object = QtWidgets.QComboBox()
        self.cmb_object.setMinimumWidth(220)
        row.addWidget(self.cmb_object, 1)
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_objects)
        row.addWidget(self.btn_refresh)
        self.btn_analyze = QtWidgets.QPushButton("Analyze")
        self.btn_analyze.setToolTip("Find gaps, missing atoms, alternate locations, ligands... without changing anything")
        self.btn_analyze.clicked.connect(self._on_analyze)
        row.addWidget(self.btn_analyze)
        outer.addWidget(box)

        self.tree_analysis = QtWidgets.QTreeWidget()
        self.tree_analysis.setHeaderLabels(["Finding", "Details"])
        self.tree_analysis.setColumnWidth(0, 300)
        self.tree_analysis.setMinimumHeight(170)
        outer.addWidget(self.tree_analysis)

        # options
        opts = QtWidgets.QGroupBox("2) Options")
        grid = QtWidgets.QGridLayout(opts)
        grid.addWidget(QtWidgets.QLabel("pH:"), 0, 0)
        self.spin_ph = QtWidgets.QDoubleSpinBox()
        self.spin_ph.setRange(0.0, 14.0)
        self.spin_ph.setDecimals(1)
        self.spin_ph.setSingleStep(0.1)
        self.spin_ph.setValue(7.4)
        grid.addWidget(self.spin_ph, 0, 1)
        self.cb_gaps = QtWidgets.QCheckBox("Build missing loops")
        self.cb_gaps.setChecked(True)
        grid.addWidget(self.cb_gaps, 1, 0, 1, 2)
        self.cb_termini = QtWidgets.QCheckBox("Also build missing termini (often disordered)")
        grid.addWidget(self.cb_termini, 2, 0, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Loop builder:"), 3, 0)
        self.cmb_gap_method = QtWidgets.QComboBox()
        for label, key in (("Automatic (MODELLER, PDBFixer without it)", "auto"), ("MODELLER", "modeller"),
                           ("PDBFixer", "pdbfixer")):
            self.cmb_gap_method.addItem(label, key)
        grid.addWidget(self.cmb_gap_method, 3, 1)
        self.cb_propka = QtWidgets.QCheckBox("Protonation from PROPKA pKa values")
        self.cb_propka.setChecked(True)
        grid.addWidget(self.cb_propka, 0, 2, 1, 2)
        self.cb_flips = QtWidgets.QCheckBox("Flip Asn/Gln/His, His tautomers (reduce)")
        self.cb_flips.setChecked(True)
        grid.addWidget(self.cb_flips, 1, 2, 1, 2)
        self.cb_hets = QtWidgets.QCheckBox("Keep ligands, ions and cofactors")
        self.cb_hets.setChecked(True)
        grid.addWidget(self.cb_hets, 2, 2, 1, 2)
        self.cb_waters = QtWidgets.QCheckBox("Keep crystal waters")
        grid.addWidget(self.cb_waters, 3, 2, 1, 2)
        self.cb_minimize = QtWidgets.QCheckBox("Restrained minimization (hydrogens and built atoms)")
        self.cb_minimize.setChecked(True)
        grid.addWidget(self.cb_minimize, 4, 2, 1, 2)
        grid.addWidget(QtWidgets.QLabel("New object:"), 4, 0)
        self.txt_name = QtWidgets.QLineEdit()
        self.txt_name.setPlaceholderText("<object>_prep")
        grid.addWidget(self.txt_name, 4, 1)
        grid.addWidget(QtWidgets.QLabel("Full sequences\n(only without SEQRES):"), 5, 0)
        self.txt_seq = QtWidgets.QPlainTextEdit()
        self.txt_seq.setPlaceholderText("A: MKTAYIAKQR...   (one chain per line; a PDB entry's sequence is fetched "
                                        "from RCSB automatically)")
        self.txt_seq.setMaximumHeight(64)
        grid.addWidget(self.txt_seq, 5, 1, 1, 3)
        outer.addWidget(opts)

        run_row = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        run_row.addWidget(self.lbl_status, 1)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        run_row.addWidget(self.btn_cancel)
        self.btn_run = QtWidgets.QPushButton("Run Preparation")
        self.btn_run.clicked.connect(self._on_run)
        run_row.addWidget(self.btn_run)
        outer.addLayout(run_row)

        # results
        res = QtWidgets.QGroupBox("3) What changed")
        rl = QtWidgets.QVBoxLayout(res)
        self.tabs = QtWidgets.QTabWidget()
        self.tbl_steps = QtWidgets.QTableWidget(0, 2)
        self.tbl_steps.setHorizontalHeaderLabels(["Step", "Result"])
        self.tbl_steps.horizontalHeader().setStretchLastSection(True)
        self.tbl_steps.verticalHeader().setVisible(False)
        self.tabs.addTab(self.tbl_steps, "Steps")
        self.tbl_prot = QtWidgets.QTableWidget(0, 4)
        self.tbl_prot.setHorizontalHeaderLabels(["Residue", "pKa", "State", "Why"])
        self.tbl_prot.horizontalHeader().setStretchLastSection(True)
        self.tbl_prot.verticalHeader().setVisible(False)
        self.tabs.addTab(self.tbl_prot, "Protonation")
        self.txt_notes = QtWidgets.QPlainTextEdit()
        self.txt_notes.setReadOnly(True)
        self.tabs.addTab(self.txt_notes, "Notes")
        rl.addWidget(self.tabs)
        self.lbl_legend = QtWidgets.QLabel("Colours in PyMOL: " + view.LEGEND + ".")
        self.lbl_legend.setWordWrap(True)
        rl.addWidget(self.lbl_legend)
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn_folder = QtWidgets.QPushButton("Open Output Folder")
        self.btn_folder.setEnabled(False)
        self.btn_folder.clicked.connect(self._on_open_folder)
        btns.addWidget(self.btn_folder)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btns.addWidget(btn_close)
        rl.addLayout(btns)
        outer.addWidget(res, 1)

    def showEvent(self, event):
        super().showEvent(event)
        try:
            from . import ui_theme

            ui_theme.polish_window(self, fit=False)
        except Exception:
            pass

    # ------------------------------------------------------------ helpers --
    def refresh_objects(self):
        current = self.cmb_object.currentText()
        self.cmb_object.blockSignals(True)
        self.cmb_object.clear()
        for name in _polymer_objects():
            self.cmb_object.addItem(name)
        idx = self.cmb_object.findText(current)
        if idx >= 0:
            self.cmb_object.setCurrentIndex(idx)
        self.cmb_object.blockSignals(False)

    def _object(self):
        name = self.cmb_object.currentText().strip()
        if not name:
            raise ValueError("Load a protein structure in PyMOL first.")
        return name

    def _save_input(self, obj, folder):
        from pymol import cmd

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{obj}_input.pdb")
        cmd.save(path, obj, state=1)          # alternate locations and occupancies included
        return path

    def _pdb_id(self, obj):
        if callable(self._infer_pdb_id):
            try:
                return str(self._infer_pdb_id(obj) or "")
            except Exception:
                pass
        m = re.match(r"^([0-9][A-Za-z0-9]{3})", obj)
        return m.group(1).upper() if m else ""

    def _sequences(self):
        out = {}
        for line in self.txt_seq.toPlainText().splitlines():
            m = re.match(r"^\s*([A-Za-z0-9]{1,4})\s*[:=]\s*([A-Za-z\s]+)$", line)
            if m:
                out[m.group(1)] = re.sub(r"[^A-Za-z]", "", m.group(2)).upper()
        return out

    def _run_bg(self, fn, on_ok, on_err):
        if callable(self._submit_task):
            self._submit_task(fn, on_ok, on_err, lambda: None)
        else:
            try:
                on_ok(fn())
            except Exception as exc:
                on_err(str(exc))

    # ------------------------------------------------------------ analyze --
    def _on_analyze(self):
        try:
            obj = self._object()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, TITLE, str(exc))
            return
        folder = os.path.join(self._workspace(), "analysis")
        path = self._save_input(obj, folder)
        pdb_id = self._pdb_id(obj)
        sequences = self._sequences()
        self.btn_analyze.setEnabled(False)
        self.lbl_status.setText(f"Analyzing {obj}...")

        def job():
            seqs = dict(sequences)
            if not seqs and pdb_id:
                try:
                    seqs = rt.fetch_rcsb_sequences(pdb_id)
                except Exception:
                    seqs = {}
            return rt.analyze_structure(path, sequences=seqs or None)

        def ok(result):
            self.btn_analyze.setEnabled(True)
            self._analysis = result
            self._fill_analysis(obj, result)
            self.lbl_status.setText("Analysis done. Choose the options and run the preparation.")

        def err(text):
            self.btn_analyze.setEnabled(True)
            self.lbl_status.setText(f"Analysis failed: {str(text).strip().splitlines()[-1] if text else ''}")

        self._run_bg(job, ok, err)

    def _fill_analysis(self, obj, a):
        t = self.tree_analysis
        t.clear()

        def item(label, detail="", children=()):
            it = QtWidgets.QTreeWidgetItem([label, detail])
            for c in children:
                it.addChild(QtWidgets.QTreeWidgetItem(list(c)))
            t.addTopLevelItem(it)
            return it

        item("Chains", ", ".join(f"{c['chain']} ({c['kind']}, {c['residues']} res)" for c in a["chains"]))
        internal = [g for g in a["gaps"] if g["where"] == "internal"]
        ends = [g for g in a["gaps"] if g["where"] != "internal"]
        item(f"Missing loops: {len(internal)}", f"{sum(g['length'] for g in internal)} residues",
             [(f"{g['chain']}: {g['after']} - {g['before']}", f"{g['length']} res  {g['sequence']}")
              for g in internal])
        item(f"Missing termini: {len(ends)}", f"{sum(g['length'] for g in ends)} residues",
             [(f"{g['chain']}: {g['where']}", f"{g['length']} res  {g['sequence']}") for g in ends])
        if a["chain_breaks_without_sequence"]:
            item(f"Chain breaks of unknown sequence: {len(a['chain_breaks_without_sequence'])}",
                 "give the full sequence to build them",
                 [(f"{b['chain']}: {b['after']} - {b['before']}", f"{b['distance_A']} A")
                  for b in a["chain_breaks_without_sequence"]])
        item(f"Incomplete side chains: {len(a['missing_atoms'])}", "",
             [(m["residue"], ", ".join(m["atoms"])) for m in a["missing_atoms"]])
        item(f"Alternate locations: {len(a['altlocs'])}", "the most occupied is kept",
             [(r["residue"], f"{r['altlocs']} -> {r['kept']} (occupancy {r['occupancy']})") for r in a["altlocs"]])
        item("Modified residues", ", ".join(a["nonstandard"]) or "none")
        item("Hydrogens in the file", str(a["hydrogens_present"]) + (" (rebuilt)" if a["hydrogens_present"] else ""))
        item("Disulfides", ", ".join(a["disulfides"]) or "none")
        item(f"Ligands / ions: {len(a['hetero_groups'])}", "",
             [(f"{h['resn']} {h['resi']}:{h['chain']}", h["kind"]) for h in a["hetero_groups"]])
        item("Waters", str(a["waters"]))
        tools = a.get("tools", {})
        item("Tools", "  ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in tools.items()))
        for i in range(t.topLevelItemCount()):
            if t.topLevelItem(i).childCount() <= 8:
                t.topLevelItem(i).setExpanded(True)

    # ---------------------------------------------------------------- run --
    def _on_run(self):
        if self._proc is not None:
            return
        try:
            obj = self._object()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, TITLE, str(exc))
            return
        from pymol import cmd

        name = re.sub(r"[^A-Za-z0-9_]", "_", self.txt_name.text().strip() or f"{obj}_prep")[:60]
        if name in cmd.get_names("objects"):
            QtWidgets.QMessageBox.warning(self, TITLE, f"An object named {name} already exists; choose another "
                                          "name or delete it.")
            return
        workdir = os.path.join(self._workspace(), time.strftime(f"{name}_%Y%m%d_%H%M%S"))
        path = self._save_input(obj, workdir)
        args = {"pdb_in": path, "pdb_out": os.path.join(workdir, f"{name}.pdb"), "ph": float(self.spin_ph.value()),
                "fill_gaps": self.cb_gaps.isChecked(), "fill_termini": self.cb_termini.isChecked(),
                "gap_method": str(self.cmb_gap_method.currentData()), "sequences": self._sequences() or None,
                "keep_waters": self.cb_waters.isChecked(), "keep_hets": self.cb_hets.isChecked(),
                "flips": self.cb_flips.isChecked(), "use_propka": self.cb_propka.isChecked(),
                "minimize": self.cb_minimize.isChecked(), "workdir": os.path.join(workdir, "work"),
                "pdb_id": self._pdb_id(obj)}
        try:
            self._proc, log_path, result_json = rt.launch_job(args, workdir, python=self._python)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, TITLE, f"Could not start the preparation: {exc}")
            return
        self._job = {"obj": obj, "name": name, "args": args, "log": log_path, "result": result_json,
                     "workdir": workdir, "t0": time.time()}
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText(f"Preparing {obj}...")
        self._timer.start()

    def _poll(self):
        if self._proc is None:
            self._timer.stop()
            return
        job = self._job
        if self._proc.poll() is None:
            msg = rt.job_progress(job["log"])
            self.lbl_status.setText(f"[{time.time() - job['t0']:.0f} s] {msg or 'starting'}")
            return
        self._timer.stop()
        report, error = rt.job_result(self._proc, job["log"], job["result"])
        self._proc = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if report is None:
            self.lbl_status.setText("Preparation failed: " + (error.strip().splitlines()[-1] if error else ""))
            self.txt_notes.setPlainText(error)
            self.tabs.setCurrentWidget(self.txt_notes)
            return
        self._show_report(job, report)

    def _show_report(self, job, report):
        from pymol import cmd

        cmd.load(job["args"]["pdb_out"], job["name"])
        view.show_prepared(job["name"], report, cmd)
        cmd.disable(job["obj"])
        cmd.zoom(job["name"])
        self.refresh_objects()
        self.lbl_status.setText(f"{job['name']}: prepared at pH {report['ph']:g} in {time.time() - job['t0']:.0f} s "
                                f"({report.get('hydrogens', 0)} hydrogens).")
        self.tbl_steps.setRowCount(0)
        for s in report.get("steps", []):
            r = self.tbl_steps.rowCount()
            self.tbl_steps.insertRow(r)
            self.tbl_steps.setItem(r, 0, QtWidgets.QTableWidgetItem(s["step"]))
            self.tbl_steps.setItem(r, 1, QtWidgets.QTableWidgetItem(s["detail"]))
        self.tbl_steps.resizeColumnToContents(0)
        self.tbl_prot.setRowCount(0)
        unusual = ("ASH", "GLH", "HIP", "LYN", "CYM", "CYX")
        rows = sorted(report.get("protonation", []), key=lambda d: (d["state"] not in unusual, d["residue"]))
        for d in rows:
            if d["state"] in ("ASP", "GLU", "LYS", "CYS") and d.get("pka") is None:
                continue
            r = self.tbl_prot.rowCount()
            self.tbl_prot.insertRow(r)
            vals = [d["residue"], "" if d.get("pka") is None else f"{d['pka']:.2f}", d["state"], d.get("why", "")]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(v))
                if d["state"] in unusual:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                self.tbl_prot.setItem(r, c, it)
        self.tbl_prot.resizeColumnsToContents()
        notes = list(report.get("warnings", []))
        for key, label in (("built_residues", "Built residues"), ("flips", "Flipped side chains"),
                           ("covalent_links", "Covalent links to ligands"), ("metal_ligands", "Metal ligands"),
                           ("kept_modified", "Modified residues kept")):
            vals = report.get(key) or []
            if vals:
                if key == "flips":
                    vals = [f"{f['resn']}{f['resi']}:{f['chain']}" for f in vals]
                notes.append(f"{label}: " + ", ".join(str(v) for v in vals))
        notes.append(f"Output: {job['args']['pdb_out']}")
        self.txt_notes.setPlainText("\n\n".join(notes))
        self.btn_folder.setEnabled(True)
        self._last_folder = job["workdir"]

    def _on_cancel(self):
        if self._proc is not None:
            try:
                from .agent.jobs import stop_process

                stop_process(self._proc, getattr(self._proc, "_leonardo_log", None))
            except Exception:
                self._proc.terminate()
            self._proc = None
        self._timer.stop()
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Cancelled.")

    def _on_open_folder(self):
        folder = getattr(self, "_last_folder", "")
        if folder and os.path.isdir(folder):
            from pymol.Qt import QtGui

            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))

    def closeEvent(self, event):
        if self._proc is not None and self._proc.poll() is None:
            ans = QtWidgets.QMessageBox.question(self, TITLE, "A preparation is running. Stop it and close?")
            if ans != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self._on_cancel()
        super().closeEvent(event)
