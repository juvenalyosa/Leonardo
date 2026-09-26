"""QM/MM reaction paths inside the enzyme: relaxed scans, transition-state search and frequencies.

MOPAC's own optimizers (``EF``, ``TS``, ``SADDLE``, ``IRC``) keep the external potential of ``mol.in`` frozen at
the starting geometry: as the QM atoms move, the embedding no longer matches their positions.  Here the geometry
is driven from Python instead, and **every energy/force evaluation recomputes the electrostatic potential, the
link-atom positions and the QM–MM Lennard-Jones** at the current coordinates
(:func:`leonardo.qmmm_mopac_runtime.qmmm_step_file_based`).  The MM environment stays fixed (the usual
"fixed-environment" QM/MM reaction-path approximation).

Tools (ASE, gradients only):

* :meth:`EmbeddedQMMMModel.optimize` — L-BFGS minimization of the QM region, optionally with a distance restraint.
* :meth:`EmbeddedQMMMModel.relaxed_distance_scan` — relaxed scan of the distance between two atoms: at every target
  distance the whole QM region is re-optimized (starting from the previous point).
* :meth:`EmbeddedQMMMModel.refine_ts_dimer` — dimer method (minimum-mode following) to converge a first-order saddle
  point from a guess such as the highest point of a relaxed scan.
* :meth:`EmbeddedQMMMModel.vibrations` — numerical frequencies of the QM region; a transition state must have
  exactly one imaginary mode.

Units: energies in kJ/mol, coordinates in Angstrom (ASE internally works in eV and eV/Angstrom).
"""
from __future__ import annotations

import os

import numpy as np

from .qmmm_mopac_runtime import mopac_validate_ts_frequencies, qmmm_link_atom_plan, qmmm_step_file_based

KJ_MOL_PER_EV = 96.48533212
_F_KJ_NM_TO_EV_A = 1.0 / KJ_MOL_PER_EV / 10.0        # kJ/mol/nm -> eV/Angstrom


class EmbeddedQMMMModel:
    """QM region (MOPAC) moving inside a fixed MM environment (point charges + Lennard-Jones)."""

    def __init__(self, xyz_full_ang, atomic_numbers_full, qm_indices, mm_indices, mm_charges_e, workdir,
                 mopac_executable="mopac", method="PM7", qm_total_charge=0, qm_multiplicity=1, bond_pairs=None,
                 lj_sigma_nm=None, lj_epsilon_kj=None, fixed_qm_indices=None, dielectric=1.0, timeout_s=600,
                 extra_keywords=""):
        self.xyz_ref = np.asarray(xyz_full_ang, dtype=float).reshape((-1, 3)).copy()
        self.z = [int(v) for v in np.asarray(atomic_numbers_full).reshape(-1)]
        self.qm = [int(i) for i in qm_indices]
        self.mm = [int(i) for i in mm_indices]
        q_all = np.asarray(mm_charges_e, dtype=float).reshape(-1)
        self.q_mm = q_all[self.mm] if q_all.size == len(self.z) else q_all
        if self.q_mm.size != len(self.mm):
            raise ValueError("mm_charges_e must have one value per atom or per MM atom.")
        self.workdir = str(workdir)
        os.makedirs(self.workdir, exist_ok=True)
        self.exe = str(mopac_executable)
        self.keywords = " ".join(t for t in (str(method).upper(), "1SCF GRAD AUX QMMM", str(extra_keywords)) if t.strip())
        self.charge, self.mult = int(qm_total_charge), int(qm_multiplicity)
        self.bond_pairs = bond_pairs
        self.sig, self.eps = lj_sigma_nm, lj_epsilon_kj
        self.include_lj = lj_sigma_nm is not None and lj_epsilon_kj is not None
        self.dielectric, self.timeout_s = float(dielectric), int(timeout_s)
        # the link-atom plan (and so the fixed ratio g) is set once: the chain rule for link forces is then exact
        self.link_plan = qmmm_link_atom_plan(self.xyz_ref, self.z, self.qm, self.mm, bond_pairs=bond_pairs)
        self.fixed = sorted({self.qm.index(int(i)) for i in list(fixed_qm_indices or []) if int(i) in self.qm})
        self.n_calls = 0

    # ------------------------------------------------------------------ energy and forces
    def full_coordinates(self, qm_xyz):
        xyz = self.xyz_ref.copy()
        xyz[self.qm] = np.asarray(qm_xyz, dtype=float).reshape((-1, 3))
        return xyz

    def evaluate(self, qm_xyz, restraint=None, tag="calc"):
        """Energy (kJ/mol) and forces on the QM atoms (kJ/mol/nm).  ``restraint`` = (i, j, d0_ang, k_kj_mol_ang2)
        with global atom indices; its energy is added to the total and also reported separately."""
        self.n_calls += 1
        xyz = self.full_coordinates(qm_xyz)
        r = qmmm_step_file_based(
            xyz, self.z, self.qm, self.mm, self.q_mm, mopac_executable=self.exe, mopac_keywords=self.keywords,
            qm_total_charge=self.charge, qm_multiplicity=self.mult, workdir=os.path.join(self.workdir, tag),
            dielectric=self.dielectric, timeout_s=self.timeout_s, link_plan=self.link_plan,
            lj_sigma_nm=self.sig, lj_epsilon_kj=self.eps, include_lj=self.include_lj)
        f = (np.asarray(r["forces_qm_internal_kj_mol_nm"]) + np.asarray(r["forces_qmmm_elec_qm_kj_mol_nm"])
             + np.asarray(r["forces_lj_qm_kj_mol_nm"]))
        e = float(r["energy_qmmm_total_kj_mol"])
        e_restr = 0.0
        if restraint is not None:
            i, j, d0, k = restraint
            a, b = self.qm.index(int(i)), self.qm.index(int(j))
            v = xyz[int(j)] - xyz[int(i)]
            d = float(np.linalg.norm(v))
            e_restr = 0.5 * float(k) * (d - float(d0)) ** 2
            dE_dd = float(k) * (d - float(d0))                   # kJ/mol/Angstrom
            unit = v / max(d, 1.0e-12)
            f[a] += dE_dd * unit * 10.0                          # force = -dE/dx, converted to kJ/mol/nm
            f[b] -= dE_dd * unit * 10.0
        if self.fixed:
            f[self.fixed] = 0.0
        return e + e_restr, f, {"energy_kj_mol": e, "restraint_kj_mol": e_restr, "raw": r}

    # ------------------------------------------------------------------ ASE bridge
    def ase_atoms(self, qm_xyz, restraint=None, tag="calc"):
        from ase import Atoms
        from ase.calculators.calculator import Calculator, all_changes
        from ase.constraints import FixAtoms

        model = self

        class _QMMMCalculator(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, list(properties), system_changes)
                e, f, terms = model.evaluate(self.atoms.get_positions(), restraint=restraint, tag=tag)
                self.terms = terms
                self.results = {"energy": e / KJ_MOL_PER_EV, "forces": f * _F_KJ_NM_TO_EV_A}

        from .qmmm_mopac_runtime import _symbol_from_atomic_number

        atoms = Atoms(symbols=[_symbol_from_atomic_number(self.z[i]) for i in self.qm],
                      positions=np.asarray(qm_xyz, dtype=float).reshape((-1, 3)))
        if self.fixed:
            atoms.set_constraint(FixAtoms(indices=self.fixed))
        atoms.calc = _QMMMCalculator()
        return atoms

    def optimize(self, qm_xyz, restraint=None, fmax_kj_mol_ang=2.0, steps=300, tag="opt"):
        from ase.optimize import LBFGS

        atoms = self.ase_atoms(qm_xyz, restraint=restraint, tag=tag)
        opt = LBFGS(atoms, logfile=os.path.join(self.workdir, f"{tag.replace('/', '_')}.log"), maxstep=0.15)
        converged = bool(opt.run(fmax=fmax_kj_mol_ang / KJ_MOL_PER_EV, steps=int(steps)))
        xyz = atoms.get_positions().copy()
        e_tot, f, terms = self.evaluate(xyz, restraint=restraint, tag=f"{tag}/final")
        return {"qm_xyz": xyz, "energy_kj_mol": terms["energy_kj_mol"], "restraint_kj_mol": terms["restraint_kj_mol"],
                "converged": converged, "steps": int(opt.get_number_of_steps()),
                "max_force_kj_mol_ang": float(np.max(np.linalg.norm(f, axis=1)) / 10.0)}

    # ------------------------------------------------------------------ reaction paths
    def relaxed_distance_scan(self, atom_i, atom_j, distances_ang, k_kj_mol_ang2=20000.0, fmax_kj_mol_ang=2.0,
                              steps=300, qm_xyz_start=None):
        """Relaxed scan of d(i, j): at each target the whole QM region is optimized under a harmonic restraint.
        The reported energy excludes the restraint.  Both atoms must belong to the QM region.

        The restraint must be stiffer than the bond it pulls (a covalent X–H stretch is ~4000–7000 kJ/mol/Å²),
        otherwise the achieved distance lags behind the target: the default 20000 kJ/mol/Å² keeps it within a few
        hundredths of an Angstrom.  Always read ``distance_ang`` (achieved), not only ``target_distance_ang``."""
        if int(atom_i) not in self.qm or int(atom_j) not in self.qm:
            raise ValueError("Both scanned atoms must be in the QM region.")
        xyz = np.asarray(self.xyz_ref[self.qm] if qm_xyz_start is None else qm_xyz_start, dtype=float).copy()
        rows = []
        for n, d0 in enumerate(list(distances_ang)):
            res = self.optimize(xyz, restraint=(int(atom_i), int(atom_j), float(d0), float(k_kj_mol_ang2)),
                                fmax_kj_mol_ang=fmax_kj_mol_ang, steps=steps, tag=f"scan/point_{n:03d}")
            xyz = res["qm_xyz"]
            full = self.full_coordinates(xyz)
            rows.append({"point": n + 1, "target_distance_ang": float(d0),
                         "distance_ang": float(np.linalg.norm(full[int(atom_j)] - full[int(atom_i)])),
                         "energy_kj_mol": float(res["energy_kj_mol"]), "restraint_kj_mol": float(res["restraint_kj_mol"]),
                         "converged": bool(res["converged"]), "steps": int(res["steps"]), "qm_xyz": xyz.copy()})
        e = np.array([r["energy_kj_mol"] for r in rows])
        for r in rows:
            r["energy_rel_kj_mol"] = float(r["energy_kj_mol"] - e[0])
        k_max = int(np.argmax(e))
        return {"ok": True, "rows": rows, "highest_point": dict(rows[k_max]), "lowest_point": dict(rows[int(np.argmin(e))]),
                "barrier_estimate_kj_mol": float(e[k_max] - e[0]), "n_energy_evaluations": int(self.n_calls),
                "interior_maximum": bool(0 < k_max < len(rows) - 1)}

    def refine_ts_dimer(self, qm_xyz_guess, mode_guess=None, fmax_kj_mol_ang=2.0, steps=300, seed=0):
        """Converge a first-order saddle point with the dimer method (ASE ``MinModeTranslate``).
        ``mode_guess`` (n_qm × 3) is the expected reaction direction, e.g. the scanned bond."""
        from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

        atoms = self.ase_atoms(qm_xyz_guess, tag="ts/calc")
        mask = [k not in self.fixed for k in range(len(self.qm))]
        control = DimerControl(initial_eigenmode_method="displacement",
                               displacement_method="vector", logfile=os.path.join(self.workdir, "dimer.log"),
                               mask=mask, dimer_separation=0.01, max_num_rot=3, maximum_translation=0.1,
                               trial_trans_step=0.01)
        d_atoms = MinModeAtoms(atoms, control)
        vec = (np.asarray(mode_guess, dtype=float).reshape((-1, 3)) if mode_guess is not None
               else np.random.default_rng(seed).normal(size=(len(self.qm), 3)))
        vec[self.fixed] = 0.0
        d_atoms.displace(displacement_vector=vec / np.linalg.norm(vec) * 0.01)
        opt = MinModeTranslate(d_atoms, logfile=os.path.join(self.workdir, "dimer_translate.log"))
        converged = bool(opt.run(fmax=fmax_kj_mol_ang / KJ_MOL_PER_EV, steps=int(steps)))
        xyz = d_atoms.get_positions().copy()
        e, f, terms = self.evaluate(xyz, tag="ts/final")
        return {"qm_xyz": xyz, "energy_kj_mol": float(terms["energy_kj_mol"]), "converged": converged,
                "curvature_ev_ang2": float(d_atoms.get_curvature()), "steps": int(opt.get_number_of_steps()),
                "max_force_kj_mol_ang": float(np.max(np.linalg.norm(f, axis=1)) / 10.0),
                "eigenmode": np.asarray(d_atoms.get_eigenmode(), dtype=float)}

    def vibrations(self, qm_xyz, delta_ang=0.01, threshold_cm=80.0):
        """Numerical harmonic frequencies of the free QM atoms (central differences of the QM/MM forces).

        Imaginary frequencies smaller than ``threshold_cm`` are not counted: finite differences leave residual
        translations/rotations of a weakly held QM region (and SCF noise) at a few tens of cm⁻¹ with either sign,
        while a reaction mode is typically several hundred cm⁻¹.  The full list is returned for inspection."""
        from ase.vibrations import Vibrations

        atoms = self.ase_atoms(qm_xyz, tag="vib/calc")
        free = [k for k in range(len(self.qm)) if k not in self.fixed]
        vib = Vibrations(atoms, indices=free, name=os.path.join(self.workdir, "vib", "vib"), delta=float(delta_ang), nfree=2)
        vib.clean()
        vib.run()
        freqs = vib.get_vibrations().get_frequencies()
        signed = np.array([-abs(f.imag) if abs(f.imag) > 1.0e-6 else float(f.real) for f in freqs])
        return {"freq_cm_signed": signed, "validation": mopac_validate_ts_frequencies(freq_cm_signed=signed,
                                                                                    threshold_cm=threshold_cm)}


def bond_direction_mode(model, atom_i, atom_j):
    """Unit displacement guess for the dimer: atoms i and j moving apart along their bond (global indices)."""
    xyz = model.xyz_ref
    v = xyz[int(atom_j)] - xyz[int(atom_i)]
    v = v / max(float(np.linalg.norm(v)), 1.0e-12)
    mode = np.zeros((len(model.qm), 3))
    mode[model.qm.index(int(atom_i))] = -v
    mode[model.qm.index(int(atom_j))] = v
    return mode


__all__ = ["EmbeddedQMMMModel", "bond_direction_mode", "KJ_MOL_PER_EV"]
