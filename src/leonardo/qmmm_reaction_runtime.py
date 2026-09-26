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
        """Energy (kJ/mol) and forces on the QM atoms (kJ/mol/nm).  ``restraint`` (global atom indices) is either
        ``(i, j, d0_ang, k)`` on the distance d(i, j) or ``(a, b, c, xi0_ang, k)`` on the transfer coordinate
        xi = d(a, b) − d(b, c); k in kJ/mol/Å².  Its energy is added to the total and also reported separately."""
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
            if len(restraint) == 4:                              # (i, j, d0, k): distance d(i, j)
                i, j, d0, k = restraint
                terms_q = [(int(i), int(j), +1.0)]
            else:                                                # (a, b, c, xi0, k): xi = d(a, b) - d(b, c)
                a_, b_, c_, d0, k = restraint
                terms_q = [(int(a_), int(b_), +1.0), (int(b_), int(c_), -1.0)]
            value = 0.0
            grads = []
            for i, j, sign in terms_q:
                v = xyz[j] - xyz[i]
                d = float(np.linalg.norm(v))
                value += sign * d
                grads.append((i, j, sign * v / max(d, 1.0e-12)))
            e_restr = 0.5 * float(k) * (value - float(d0)) ** 2
            dE_dq = float(k) * (value - float(d0))                # kJ/mol/Angstrom
            for i, j, unit in grads:                              # d(d_ij)/dx_j = unit, d/dx_i = -unit
                f[self.qm.index(i)] += dE_dq * unit * 10.0        # force = -dE/dx, converted to kJ/mol/nm
                f[self.qm.index(j)] -= dE_dq * unit * 10.0
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

    def relaxed_transfer_scan(self, atom_a, atom_b, atom_c, xi_values_ang, k_kj_mol_ang2=5000.0,
                              fmax_kj_mol_ang=2.0, steps=300, qm_xyz_start=None):
        """Relaxed scan of the transfer coordinate xi = d(a, b) − d(b, c): group/atom ``b`` moves from ``a`` to ``c``
        (negative xi: b bonded to a; positive: bonded to c).  Unlike a single-distance scan, both the breaking and
        the forming bond are driven together, which avoids the sudden jumps (false maxima) that a one-distance scan
        shows for transfer reactions (proton, phosphoryl, hydride…)."""
        for atom in (atom_a, atom_b, atom_c):
            if int(atom) not in self.qm:
                raise ValueError("The three transfer atoms must be in the QM region.")
        xyz = np.asarray(self.xyz_ref[self.qm] if qm_xyz_start is None else qm_xyz_start, dtype=float).copy()
        rows = []
        for n, xi0 in enumerate(list(xi_values_ang)):
            res = self.optimize(xyz, restraint=(int(atom_a), int(atom_b), int(atom_c), float(xi0), float(k_kj_mol_ang2)),
                                fmax_kj_mol_ang=fmax_kj_mol_ang, steps=steps, tag=f"scan/point_{n:03d}")
            xyz = res["qm_xyz"]
            full = self.full_coordinates(xyz)
            d_ab = float(np.linalg.norm(full[int(atom_b)] - full[int(atom_a)]))
            d_bc = float(np.linalg.norm(full[int(atom_c)] - full[int(atom_b)]))
            rows.append({"point": n + 1, "target_xi_ang": float(xi0), "xi_ang": d_ab - d_bc, "d_ab_ang": d_ab,
                         "d_bc_ang": d_bc, "energy_kj_mol": float(res["energy_kj_mol"]),
                         "restraint_kj_mol": float(res["restraint_kj_mol"]), "converged": bool(res["converged"]),
                         "steps": int(res["steps"]), "qm_xyz": xyz.copy()})
        e = np.array([r["energy_kj_mol"] for r in rows])
        for r in rows:
            r["energy_rel_kj_mol"] = float(r["energy_kj_mol"] - e[0])
        k_max = int(np.argmax(e))
        return {"ok": True, "rows": rows, "highest_point": dict(rows[k_max]), "lowest_point": dict(rows[int(np.argmin(e))]),
                "barrier_estimate_kj_mol": float(e[k_max] - e[0]), "n_energy_evaluations": int(self.n_calls),
                "interior_maximum": bool(0 < k_max < len(rows) - 1)}

    def refine_ts_dimer(self, qm_xyz_guess, mode_guess=None, fmax_kj_mol_ang=2.0, steps=300, seed=0,
                        max_rotations=8, max_positive_curvature_steps=25, max_energy_rise_kj_mol=105.0):
        """Converge a first-order saddle point with the dimer method (ASE ``MinModeTranslate``).

        ``mode_guess`` (n_qm × 3) is the expected reaction direction.  The best guess is the tangent of a path
        around its highest point (:func:`path_tangent_mode`, e.g. from a relaxed scan); a single bond direction is a
        weak guess.  Safeguards: while the dimer has not found a negative curvature it climbs uphill by design, so
        the search stops (``converged=False`` with ``stopped_reason``) if the curvature stays positive for
        ``max_positive_curvature_steps`` steps or the energy rises more than ``max_energy_rise_kj_mol`` above the
        starting point, instead of returning a meaningless geometry.
        """
        from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

        atoms = self.ase_atoms(qm_xyz_guess, tag="ts/calc")
        mask = [k not in self.fixed for k in range(len(self.qm))]
        control = DimerControl(initial_eigenmode_method="displacement",
                               displacement_method="vector", logfile=os.path.join(self.workdir, "dimer.log"),
                               mask=mask, dimer_separation=0.01, max_num_rot=int(max_rotations), maximum_translation=0.1,
                               trial_trans_step=0.01)
        d_atoms = MinModeAtoms(atoms, control)
        vec = (np.asarray(mode_guess, dtype=float).reshape((-1, 3)).copy() if mode_guess is not None
               else np.random.default_rng(seed).normal(size=(len(self.qm), 3)))
        vec[self.fixed] = 0.0
        d_atoms.displace(displacement_vector=vec / np.linalg.norm(vec) * 0.01)
        opt = MinModeTranslate(d_atoms, logfile=os.path.join(self.workdir, "dimer_translate.log"))
        e_start = None
        positive_run = 0
        stopped_reason = ""
        converged = False
        fmax_ev = fmax_kj_mol_ang / KJ_MOL_PER_EV
        for converged in opt.irun(fmax=fmax_ev, steps=int(steps)):
            e_now = float(d_atoms.get_potential_energy()) * KJ_MOL_PER_EV
            e_start = e_now if e_start is None else e_start
            curvature = float(d_atoms.get_curvature())
            positive_run = positive_run + 1 if curvature > 0.0 else 0
            if converged:
                break
            if positive_run >= int(max_positive_curvature_steps):
                stopped_reason = (f"no negative curvature after {positive_run} steps: the start is not near a saddle "
                                  "point or the mode guess is poor (use the tangent of a relaxed scan)")
                break
            if e_now - e_start > float(max_energy_rise_kj_mol):
                stopped_reason = f"energy rose {e_now - e_start:.1f} kJ/mol above the start: search abandoned"
                break
        converged = bool(converged) and not stopped_reason
        xyz = d_atoms.get_positions().copy()
        e, f, terms = self.evaluate(xyz, tag="ts/final")
        return {"qm_xyz": xyz, "energy_kj_mol": float(terms["energy_kj_mol"]), "converged": converged,
                "stopped_reason": stopped_reason, "curvature_ev_ang2": float(d_atoms.get_curvature()),
                "steps": int(opt.get_number_of_steps()),
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


def path_tangent_mode(path_qm_xyz, k):
    """Tangent of a reaction path at frame ``k`` (central difference of neighbouring frames): the direction in
    which *all* QM atoms move through the barrier region, the best initial mode for the dimer."""
    frames = [np.asarray(x, dtype=float).reshape((-1, 3)) for x in path_qm_xyz]
    k = int(k)
    a, b = frames[max(0, k - 1)], frames[min(len(frames) - 1, k + 1)]
    t = b - a
    n = float(np.linalg.norm(t))
    if n <= 1.0e-12:
        raise ValueError("Degenerate path tangent: neighbouring frames are identical.")
    return t / n


def bond_direction_mode(model, atom_i, atom_j):
    """Unit displacement guess for the dimer: atoms i and j moving apart along their bond (global indices)."""
    xyz = model.xyz_ref
    v = xyz[int(atom_j)] - xyz[int(atom_i)]
    v = v / max(float(np.linalg.norm(v)), 1.0e-12)
    mode = np.zeros((len(model.qm), 3))
    mode[model.qm.index(int(atom_i))] = -v
    mode[model.qm.index(int(atom_j))] = v
    return mode


__all__ = ["EmbeddedQMMMModel", "bond_direction_mode", "path_tangent_mode", "KJ_MOL_PER_EV"]
