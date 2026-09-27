"""
"pKa & Protonation States" window.

Qt front end of :mod:`leonardo.pka_runtime`. The dialog does not know the rest of
Leonardo: the main window passes callables (hooks) for background execution, reading
the current PyMOL selection or docking ligand, and handing the chosen state to docking
or parametrization. The chosen protonation state has to be fixed before partial
charges and force-field parameters are assigned, which is why the hand-off actions
live here.
"""

import os

from pymol.Qt import QtCore, QtGui, QtWidgets

from . import pka_runtime as P

METHODS = [
    ("Empirical (fast, class-typical pKa)", "empirical"),
    ("PM7 / COSMO (semiempirical QM, calibrated)", "pm7"),
    ("Psi4 HF/3-21G + PCM (experimental: slower and less accurate than PM7)", "psi4"),
]
SITE_COLUMNS = ["Atom", "Class", "Type", "pKa", "±", "Source", "Warnings"]
STATE_COLUMNS = ["Population", "Charge", "SMILES"]


def _mol_from_file(path):
    from rdkit import Chem

    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".sdf", ".sd"):
        return next((m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m is not None), None)
    if ext == ".mol":
        return Chem.MolFromMolFile(str(path), removeHs=False)
    if ext == ".mol2":
        return Chem.MolFromMol2File(str(path), removeHs=False)
    if ext == ".pdb":
        return Chem.MolFromPDBFile(str(path), removeHs=False)
    raise P.PkaError(f"unsupported file type: {ext}")


def site_depiction_png(mol, sites, size=(420, 300)):
    """2D drawing (PNG bytes) with ionizable sites highlighted and labelled with their pKa."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    parent = P.neutral_parent(mol)
    draw_mol = Chem.Mol(parent)
    draw_mol.RemoveAllConformers()
    colors = {}
    for s in sites:
        idx = int(s["atom"])
        atom = draw_mol.GetAtomWithIdx(idx)
        atom.SetProp("atomNote", f"pKa {float(s.get('pka', s['empirical_pka'])):.1f}")
        colors[idx] = (0.95, 0.55, 0.55) if s["kind"] == "acid" else (0.55, 0.7, 0.95)
    drawer = rdMolDraw2D.MolDraw2DCairo(int(size[0]), int(size[1]))
    drawer.drawOptions().annotationFontScale = 0.8
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, draw_mol, highlightAtoms=list(colors),
                                       highlightAtomColors=colors)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


class PkaDialog(QtWidgets.QDialog):
    """Compute site pKa values and microstate populations; hand a state to other modules."""

    def __init__(self, parent=None, submit_task=None, load_selection=None, current_docking_ligand=None,
                 use_for_docking=None, use_for_parametrization=None, cache_dir=None, default_ph=7.4,
                 default_method="empirical", set_default_method=None, backend_options=None):
        super().__init__(parent)
        self.setWindowTitle("pKa & Protonation States")
        self.resize(1020, 820)
        self._submit_task = submit_task
        self._load_selection = load_selection
        self._current_docking_ligand = current_docking_ligand
        self._use_for_docking = use_for_docking
        self._use_for_parametrization = use_for_parametrization
        self._cache_dir = cache_dir
        self._set_default_method = set_default_method
        self._backend_options = backend_options
        self._mol = None
        self._mol_label = ""
        self._result = None
        self._cancel = False
        self._busy = False
        self._build_ui(default_ph)
        idx = self.cmb_method.findData(str(default_method))
        if idx >= 0:
            self.cmb_method.setCurrentIndex(idx)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self, default_ph):
        root = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Protonation must be decided before partial charges and force-field parameters: "
            "it fixes the net charge and the atom types. Pick the molecule, the method and the pH, "
            "then hand the chosen state to docking or parametrization."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        grp_in = QtWidgets.QGroupBox("Molecule")
        lay_in = QtWidgets.QGridLayout(grp_in)
        lay_in.addWidget(QtWidgets.QLabel("SMILES"), 0, 0)
        self.edit_smiles = QtWidgets.QLineEdit()
        self.edit_smiles.setPlaceholderText("e.g. NC(=N)c1ccccc1  (or load a file / PyMOL selection)")
        lay_in.addWidget(self.edit_smiles, 0, 1, 1, 3)
        self.btn_file = QtWidgets.QPushButton("Load File...")
        self.btn_sel = QtWidgets.QPushButton("From PyMOL Selection")
        self.btn_dock = QtWidgets.QPushButton("Use Docking Ligand")
        self.btn_file.clicked.connect(self._on_load_file)
        self.btn_sel.clicked.connect(self._on_load_selection)
        self.btn_dock.clicked.connect(self._on_load_docking_ligand)
        self.btn_sel.setEnabled(callable(self._load_selection))
        self.btn_dock.setEnabled(callable(self._current_docking_ligand))
        lay_in.addWidget(self.btn_file, 1, 1)
        lay_in.addWidget(self.btn_sel, 1, 2)
        lay_in.addWidget(self.btn_dock, 1, 3)
        self.lbl_source = QtWidgets.QLabel("No molecule loaded.")
        self.lbl_source.setWordWrap(True)
        lay_in.addWidget(self.lbl_source, 2, 0, 1, 4)
        root.addWidget(grp_in)

        grp_opt = QtWidgets.QGroupBox("Calculation")
        lay_opt = QtWidgets.QHBoxLayout(grp_opt)
        lay_opt.addWidget(QtWidgets.QLabel("Method"))
        self.cmb_method = QtWidgets.QComboBox()
        for label, key in METHODS:
            self.cmb_method.addItem(label, key)
        lay_opt.addWidget(self.cmb_method, 1)
        lay_opt.addWidget(QtWidgets.QLabel("pH"))
        self.spin_ph = QtWidgets.QDoubleSpinBox()
        self.spin_ph.setRange(0.0, 14.0)
        self.spin_ph.setSingleStep(0.1)
        self.spin_ph.setDecimals(2)
        self.spin_ph.setValue(float(default_ph))
        self.spin_ph.valueChanged.connect(self._refresh_states)
        lay_opt.addWidget(self.spin_ph)
        self.btn_run = QtWidgets.QPushButton("Compute pKa")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        lay_opt.addWidget(self.btn_run)
        lay_opt.addWidget(self.btn_cancel)
        root.addWidget(grp_opt)
        self.cb_default_method = QtWidgets.QCheckBox(
            "Use this method for automatic protonation in all modules (docking, parametrization, MD ligands, "
            "virtual screening, lead optimization; virtual screening libraries use PM7 instead of Psi4)"
        )
        self.cb_default_method.setEnabled(callable(self._set_default_method))
        self.cb_default_method.toggled.connect(self._on_default_method_toggled)
        self.cmb_method.currentIndexChanged.connect(self._on_default_method_toggled)
        root.addWidget(self.cb_default_method)
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        left = QtWidgets.QWidget()
        lay_left = QtWidgets.QVBoxLayout(left)
        self.lbl_depiction = QtWidgets.QLabel()
        self.lbl_depiction.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_depiction.setMinimumHeight(200)
        self.lbl_depiction.setMaximumHeight(260)
        lay_left.addWidget(self.lbl_depiction)
        self.tbl_sites = QtWidgets.QTableWidget(0, len(SITE_COLUMNS))
        self.tbl_sites.setHorizontalHeaderLabels(SITE_COLUMNS)
        self.tbl_sites.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_sites.verticalHeader().setVisible(False)
        lay_left.addWidget(self.tbl_sites, 1)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        lay_right = QtWidgets.QVBoxLayout(right)
        self.canvas = None
        self.figure = None
        try:
            try:
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
            except ImportError:
                from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(5, 3.2), tight_layout=True)
            self.canvas = Canvas(self.figure)
            lay_right.addWidget(self.canvas, 1)
        except Exception:
            lay_right.addWidget(QtWidgets.QLabel("matplotlib not available: population plot disabled"))
        self.tbl_states = QtWidgets.QTableWidget(0, len(STATE_COLUMNS))
        self.tbl_states.setHorizontalHeaderLabels(STATE_COLUMNS)
        self.tbl_states.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_states.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_states.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.tbl_states.verticalHeader().setVisible(False)
        self.tbl_states.horizontalHeader().setStretchLastSection(True)
        lay_right.addWidget(self.tbl_states, 1)
        split.addWidget(right)
        root.addWidget(split, 1)

        self.txt_warnings = QtWidgets.QPlainTextEdit()
        self.txt_warnings.setReadOnly(True)
        self.txt_warnings.setMaximumHeight(70)
        self.txt_warnings.setPlaceholderText("Warnings (difficult sites, fallbacks, mixed populations) appear here.")
        root.addWidget(self.txt_warnings)

        row = QtWidgets.QHBoxLayout()
        self.btn_to_docking = QtWidgets.QPushButton("Use Selected State for Docking")
        self.btn_to_param = QtWidgets.QPushButton("Use Selected State for Parametrization")
        self.btn_export = QtWidgets.QPushButton("Export SDF...")
        self.btn_to_docking.clicked.connect(lambda: self._hand_off(self._use_for_docking, "docking"))
        self.btn_to_param.clicked.connect(lambda: self._hand_off(self._use_for_parametrization, "parametrization"))
        self.btn_export.clicked.connect(self._on_export)
        for b in (self.btn_to_docking, self.btn_to_param, self.btn_export):
            b.setEnabled(False)
            row.addWidget(b)
        row.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        row.addWidget(btn_close)
        root.addLayout(row)

    def _on_default_method_toggled(self, *_args):
        if callable(self._set_default_method) and self.cb_default_method.isChecked():
            self._set_default_method(str(self.cmb_method.currentData()))

    # -------------------------------------------------------------- input --
    def set_molecule(self, mol, label=""):
        from rdkit import Chem

        if mol is None:
            raise P.PkaError("could not read the molecule")
        P.neutral_parent(mol)   # validates
        self._mol = Chem.Mol(mol)
        self._mol_label = str(label or Chem.MolToSmiles(Chem.RemoveHs(mol)))
        self._result = None
        self.edit_smiles.blockSignals(True)
        self.edit_smiles.setText(Chem.MolToSmiles(Chem.RemoveHs(mol)))
        self.edit_smiles.blockSignals(False)
        has_3d = mol.GetNumConformers() > 0
        self.lbl_source.setText(f"Loaded: {self._mol_label}" + (" (3D coordinates kept)" if has_3d else " (2D; 3D built on export)"))
        self._clear_results()

    def _molecule_from_inputs(self):
        from rdkit import Chem

        smi = str(self.edit_smiles.text() or "").strip()
        if self._mol is not None and smi == Chem.MolToSmiles(Chem.RemoveHs(self._mol)):
            return self._mol
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is None:
            raise P.PkaError("enter a valid SMILES or load a molecule")
        self.set_molecule(mol, label=smi)
        return self._mol

    def _on_load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Molecule", "",
                                                        "Molecules (*.sdf *.mol *.mol2 *.pdb)")
        if path:
            self._guarded(lambda: self.set_molecule(_mol_from_file(path), label=os.path.basename(path)))

    def _on_load_selection(self):
        def _load():
            mol, label = self._load_selection()
            self.set_molecule(mol, label=label)
        self._guarded(_load)

    def _on_load_docking_ligand(self):
        def _load():
            path = self._current_docking_ligand()
            if not path:
                raise P.PkaError("no docking ligand is set")
            self.set_molecule(_mol_from_file(path), label=f"docking ligand {os.path.basename(path)}")
        self._guarded(_load)

    def _guarded(self, fn):
        try:
            fn()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "pKa & Protonation", str(exc))

    # ---------------------------------------------------------------- run --
    def _on_run(self):
        if self._busy:
            return
        try:
            mol = self._molecule_from_inputs()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "pKa & Protonation", str(exc))
            return
        method = str(self.cmb_method.currentData())
        ph = float(self.spin_ph.value())
        self._cancel = False
        self._set_busy(True, f"Computing pKa with {method}...")

        options = self._backend_options(method) if callable(self._backend_options) else None

        def job():
            return P.compute_pka(mol, method=method, focus_ph=ph, cache_dir=self._cache_dir,
                                 cancel=lambda: self._cancel, backend_options=options)

        if callable(self._submit_task):
            self._submit_task(job, self._on_result, self._on_error, lambda: self._set_busy(False))
        else:    # synchronous fallback (tests, no thread pool)
            try:
                self._on_result(job())
            except Exception as exc:
                self._on_error(str(exc))
            finally:
                self._set_busy(False)

    def _on_cancel(self):
        self._cancel = True
        self.lbl_status.setText("Cancelling after the running conformers finish...")

    def _set_busy(self, busy, text=None):
        self._busy = bool(busy)
        self.btn_run.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        if text is not None:
            self.lbl_status.setText(text)

    def _on_error(self, text):
        self.lbl_status.setText(f"pKa calculation failed: {str(text).strip().splitlines()[-1] if text else ''}")

    def _on_result(self, result):
        self._result = dict(result)
        prov = self._result.get("provenance", {})
        self.lbl_status.setText(
            f"Method: {self._result.get('method')} | calibration: {prov.get('calibration_version', 'none')} | "
            f"cache: {prov.get('cache', '-')} | macroscopic pKa: "
            + (", ".join(f"{v:.2f}" for v in self._result.get("macro_pkas", [])) or "none in 0-14")
        )
        self._fill_sites()
        self._refresh_states()
        for b in (self.btn_to_docking, self.btn_to_param, self.btn_export):
            b.setEnabled(True)
        self.btn_to_docking.setEnabled(callable(self._use_for_docking))
        self.btn_to_param.setEnabled(callable(self._use_for_parametrization))

    # ------------------------------------------------------------ results --
    def _clear_results(self):
        self.tbl_sites.setRowCount(0)
        self.tbl_states.setRowCount(0)
        self.txt_warnings.setPlainText("")
        self.lbl_depiction.clear()
        if self.figure is not None:
            self.figure.clear()
            self.canvas.draw_idle()
        for b in (self.btn_to_docking, self.btn_to_param, self.btn_export):
            b.setEnabled(False)

    def _fill_sites(self):
        res = self._result or {}
        sites = list(res.get("sites", []))
        self.tbl_sites.setRowCount(0)
        for s in sites:
            r = self.tbl_sites.rowCount()
            self.tbl_sites.insertRow(r)
            elem = self._mol_atom_symbol(int(s["atom"]))
            vals = [f"{elem}{int(s['atom']) + 1}", s["class"], s["kind"], f"{float(s['pka']):.2f}",
                    f"{float(s.get('sd', float('nan'))):.1f}", str(s.get("pka_source", "")), ", ".join(s.get("flags", []))]
            for c, v in enumerate(vals):
                self.tbl_sites.setItem(r, c, QtWidgets.QTableWidgetItem(str(v)))
        self.tbl_sites.resizeColumnsToContents()
        try:
            png = site_depiction_png(self._mol, sites)
            pix = QtGui.QPixmap()
            pix.loadFromData(png, "PNG")
            self.lbl_depiction.setPixmap(pix)
        except Exception:
            self.lbl_depiction.setText("(depiction unavailable)")

    def _mol_atom_symbol(self, idx):
        try:
            return P.neutral_parent(self._mol).GetAtomWithIdx(idx).GetSymbol()
        except Exception:
            return "?"

    def _state_smiles(self, state):
        from rdkit import Chem

        try:
            m = P.molecule_for_state(self._mol, self._result, state, embed_if_2d=False)
            return Chem.MolToSmiles(Chem.RemoveHs(m))
        except Exception:
            return "?"

    def _refresh_states(self, *_args):
        res = self._result
        if not res:
            return
        ph = float(self.spin_ph.value())
        rows = P.states_at_ph(res, ph, min_population=0.01)
        self.tbl_states.setRowCount(0)
        for row in rows:
            r = self.tbl_states.rowCount()
            self.tbl_states.insertRow(r)
            charge = row["charge"]
            vals = [f"{100 * row['population']:.1f}%", "" if charge is None else f"{int(charge):+d}",
                    self._state_smiles(row["state"])]
            for c, v in enumerate(vals):
                item = QtWidgets.QTableWidgetItem(v)
                item.setData(QtCore.Qt.UserRole, list(row["state"]))
                self.tbl_states.setItem(r, c, item)
        if self.tbl_states.rowCount():
            self.tbl_states.selectRow(0)
        warnings = list(res.get("warnings", []))
        if rows and rows[0]["population"] < 0.6 and res.get("enumerated_atoms"):
            warnings.append(f"Mixed population at pH {ph:.2f}: the dominant state is only "
                            f"{100 * rows[0]['population']:.0f}%. A single parametrized state is an approximation.")
        self.txt_warnings.setPlainText("\n".join(warnings))
        self._plot(ph)

    def _plot(self, ph):
        if self.figure is None or not self._result:
            return
        import numpy as np

        grid = list(np.linspace(0.0, 14.0, 141))
        curves = P.population_curves(self._result, grid, min_population=0.05)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for st in curves["states"]:
            ax.plot(curves["ph"], st["populations"], label=f"{self._state_smiles(st['state'])} ({st['charge']:+d})")
        ax.axvline(ph, color="0.4", linestyle="--", linewidth=1)
        ax.set_xlabel("pH")
        ax.set_ylabel("population")
        ax.set_xlim(0, 14)
        ax.set_ylim(-0.02, 1.02)
        if curves["states"] and curves["mean_charge"][0] is not None:
            ax2 = ax.twinx()
            ax2.plot(curves["ph"], curves["mean_charge"], color="k", linewidth=1.2, linestyle=":")
            ax2.set_ylabel("mean net charge")
        if curves["states"]:
            ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
        self.canvas.draw_idle()

    # ------------------------------------------------------------ hand-off --
    def selected_state(self):
        rows = self.tbl_states.selectionModel().selectedRows() if self.tbl_states.selectionModel() else []
        r = rows[0].row() if rows else 0
        item = self.tbl_states.item(r, 0)
        return list(item.data(QtCore.Qt.UserRole)) if item is not None else []

    def selected_state_molecule(self):
        state = self.selected_state()
        mol = P.molecule_for_state(self._mol, self._result, state, embed_if_2d=True)
        pops = {tuple(r["state"]): r["population"] for r in P.states_at_ph(self._result, float(self.spin_ph.value()), 0.0)}
        from rdkit import Chem

        info = {
            "method": self._result.get("method"),
            "ph": float(self.spin_ph.value()),
            "state": state,
            "population": float(pops.get(tuple(state), float("nan"))),
            "charge": int(Chem.GetFormalCharge(mol)),
            "sites": self._result.get("sites", []),
            "warnings": self._result.get("warnings", []),
            "calibration_version": self._result.get("provenance", {}).get("calibration_version", "none"),
            "source": self._mol_label,
        }
        return mol, info

    def _hand_off(self, callback, target):
        if not callable(callback) or not self._result:
            return
        try:
            mol, info = self.selected_state_molecule()
            message = callback(mol, info)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "pKa & Protonation", f"Could not send the state to {target}: {exc}")
            return
        self.lbl_status.setText(str(message or f"State sent to {target} (net charge {info['charge']:+d})."))

    def _on_export(self):
        if not self._result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Protonated Molecule", "", "SDF (*.sdf)")
        if not path:
            return
        try:
            write_state_sdf(*self.selected_state_molecule(), path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "pKa & Protonation", str(exc))
            return
        self.lbl_status.setText(f"Exported {path}")


def write_state_sdf(mol, info, path):
    """Write a protonated state to SDF with its provenance as properties."""
    from rdkit import Chem

    m = Chem.Mol(mol)
    m.SetProp("_Name", str(info.get("source", "ligand"))[:80])
    m.SetProp("LEONARDO_PROTONATION_METHOD", str(info.get("method")))
    m.SetProp("LEONARDO_PROTONATION_PH", f"{float(info.get('ph', 7.4)):.2f}")
    m.SetProp("LEONARDO_PROTONATION_POPULATION", f"{float(info.get('population', float('nan'))):.3f}")
    m.SetProp("LEONARDO_NET_CHARGE", str(int(info.get("charge", 0))))
    m.SetProp("LEONARDO_PKA_CALIBRATION", str(info.get("calibration_version", "none")))
    writer = Chem.SDWriter(str(path))
    writer.write(m)
    writer.close()
    return str(path)
