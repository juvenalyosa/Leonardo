"""
Reaction paths inside the enzyme: QST2-type path -> transition-state optimization -> IRC, all with the QM region
embedded in the fixed MM environment (``EmbeddedQMMMModel``: MOPAC electrostatic embedding + Lennard-Jones).

The reaction is given by the bonds it forms and breaks (global atom indices, all in the QM region). The workflow:

1. reactant: the starting structure, optimized;
2. product: built from the reactant by pulling the forming bonds to a covalent length and the breaking bonds
   apart, then optimized without restraints and checked (the bonds must really be formed/broken);
3. path between the two minima (the two-endpoint "QST2" idea): image-dependent pair potential (IDPP)
   interpolation and a climbing-image nudged elastic band (CI-NEB) in the enzyme;
4. transition state: the climbing image refined with the dimer method (``refine_ts_dimer``), then numerical
   frequencies: exactly one imaginary mode is required;
5. IRC: steepest descent in mass-weighted coordinates from the transition state, both directions along the
   imaginary mode, each end then optimized; the path must connect the reactant and the product.

Unlike a one-coordinate scan, the path cannot follow a different reaction (e.g. a leaving group dissociating
without the nucleophile bonding): both ends are fixed minima with the requested bonding.
Energies are QM/MM potential energies with the protein fixed (kJ/mol), not free energies.
"""
from __future__ import annotations

import os

import numpy as np

from .qmmm_reaction_runtime import KJ_MOL_PER_EV, path_tangent_mode

# single-bond covalent radii (Angstrom; Cordero et al. 2008) for the bond targets and checks
_COVALENT_R = {1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 11: 1.66, 12: 1.41, 14: 1.11, 15: 1.07,
               16: 1.05, 17: 1.02, 19: 2.03, 20: 1.76, 26: 1.32, 29: 1.32, 30: 1.22, 34: 1.20, 35: 1.20, 53: 1.39}
_F_KJ_NM_TO_EV_A = 1.0 / (KJ_MOL_PER_EV * 10.0)
BONDED_FACTOR = 1.25       # a bond exists when d <= 1.25 x (sum of covalent radii)
BROKEN_FACTOR = 1.55       # a bond is broken when d >= 1.55 x (sum of covalent radii)


def covalent_length(z_i, z_j):
    return _COVALENT_R.get(int(z_i), 0.8) + _COVALENT_R.get(int(z_j), 0.8)


def _distance(full_xyz, i, j):
    return float(np.linalg.norm(np.asarray(full_xyz[int(j)]) - np.asarray(full_xyz[int(i)])))


def bond_states(model, qm_xyz, form_bonds, break_bonds):
    """Distances of the reacting bonds and whether the structure is reactant-like, product-like or neither."""
    full = model.full_coordinates(qm_xyz)
    rows = []
    for kind, pairs in (("form", form_bonds), ("break", break_bonds)):
        for i, j in pairs:
            d = _distance(full, i, j)
            ref = covalent_length(model.z[int(i)], model.z[int(j)])
            rows.append({"kind": kind, "atoms": [int(i), int(j)], "distance_A": round(d, 4),
                         "bonded": d <= BONDED_FACTOR * ref, "broken": d >= BROKEN_FACTOR * ref})
    reactant_like = all((not r["bonded"]) if r["kind"] == "form" else r["bonded"] for r in rows)
    product_like = all(r["bonded"] if r["kind"] == "form" else r["broken"] for r in rows)
    return {"bonds": rows, "reactant_like": bool(reactant_like), "product_like": bool(product_like)}


class _MultiRestraintCalculator:
    """ASE calculator: the model's QM/MM energy and forces plus harmonic distance restraints [(i, j, d0, k)]."""

    def __new__(cls, model, restraints=(), tag="calc"):
        from ase.calculators.calculator import Calculator, all_changes

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, list(properties), system_changes)
                xyz = self.atoms.get_positions()
                e, f, terms = model.evaluate(xyz, restraint=None, tag=tag)
                f = np.array(f, dtype=float)
                full = model.full_coordinates(xyz)
                e_r = 0.0
                for i, j, d0, k in restraints:
                    v = full[int(j)] - full[int(i)]
                    d = float(np.linalg.norm(v))
                    e_r += 0.5 * float(k) * (d - float(d0)) ** 2
                    grad = float(k) * (d - float(d0)) * v / max(d, 1.0e-12)      # kJ/mol/Angstrom
                    if int(i) in model.qm:
                        f[model.qm.index(int(i))] += grad * 10.0                     # -> kJ/mol/nm
                    if int(j) in model.qm:
                        f[model.qm.index(int(j))] -= grad * 10.0
                if model.fixed:
                    f[model.fixed] = 0.0
                self.terms = dict(terms, restraint_kj_mol=e_r)
                self.results = {"energy": (e + e_r) / KJ_MOL_PER_EV, "forces": f * _F_KJ_NM_TO_EV_A}

        return _Calc()


def _atoms(model, qm_xyz, restraints=(), tag="calc"):
    from ase import Atoms
    from ase.constraints import FixAtoms

    from .qmmm_mopac_runtime import _symbol_from_atomic_number

    atoms = Atoms(symbols=[_symbol_from_atomic_number(model.z[i]) for i in model.qm],
                  positions=np.asarray(qm_xyz, dtype=float).reshape((-1, 3)))
    if model.fixed:
        atoms.set_constraint(FixAtoms(indices=model.fixed))
    atoms.calc = _MultiRestraintCalculator(model, restraints, tag)
    return atoms


def _optimize(model, qm_xyz, restraints=(), fmax_kj_mol_ang=2.0, steps=400, tag="opt"):
    from ase.optimize import LBFGS

    atoms = _atoms(model, qm_xyz, restraints, tag)
    opt = LBFGS(atoms, logfile=os.path.join(model.workdir, f"{tag.replace('/', '_')}.log"), maxstep=0.15)
    converged = bool(opt.run(fmax=fmax_kj_mol_ang / KJ_MOL_PER_EV, steps=int(steps)))
    xyz = atoms.get_positions().copy()
    e, f, terms = model.evaluate(xyz, tag=f"{tag}/final")
    return {"qm_xyz": xyz, "energy_kj_mol": float(terms["energy_kj_mol"]), "converged": converged,
            "steps": int(opt.get_number_of_steps()),
            "max_force_kj_mol_ang": float(np.max(np.linalg.norm(np.asarray(f, dtype=float), axis=1)) / 10.0)}


def build_product(model, reactant_xyz, form_bonds, break_bonds, k_kj_mol_ang2=2000.0, stages=4, fmax_kj_mol_ang=2.0,
                  steps=400):
    """Product minimum from the reactant: the forming bonds are pulled to their covalent length and the breaking
    bonds stretched to 1.8 x covalent, in ``stages`` restrained optimizations, then everything is released."""
    full = model.full_coordinates(reactant_xyz)
    xyz = np.asarray(reactant_xyz, dtype=float).copy()
    pairs = [(int(i), int(j), covalent_length(model.z[int(i)], model.z[int(j)]), "form") for i, j in form_bonds]
    pairs += [(int(i), int(j), 1.8 * covalent_length(model.z[int(i)], model.z[int(j)]), "break")
              for i, j in break_bonds]
    for s in range(1, int(stages) + 1):
        f = s / float(stages)
        restraints = [(i, j, (1 - f) * _distance(full, i, j) + f * target, k_kj_mol_ang2) for i, j, target, _k in pairs]
        xyz = _optimize(model, xyz, restraints, fmax_kj_mol_ang=fmax_kj_mol_ang, steps=steps,
                        tag=f"product/stage_{s}")["qm_xyz"]
    free = _optimize(model, xyz, (), fmax_kj_mol_ang=fmax_kj_mol_ang, steps=steps, tag="product/free")
    return dict(free, check=bond_states(model, free["qm_xyz"], form_bonds, break_bonds))


def neb_path(model, reactant_xyz, product_xyz, n_images=9, fmax_kj_mol_ang=5.0, steps=300, spring_ev_ang2=0.1):
    """Climbing-image NEB between two QM-region minima inside the fixed enzyme (IDPP interpolation)."""
    from ase.mep import NEB
    from ase.optimize import FIRE

    images = [_atoms(model, reactant_xyz, tag="neb/r")]
    for k in range(int(n_images)):
        images.append(_atoms(model, reactant_xyz, tag=f"neb/image_{k + 1:02d}"))
    images.append(_atoms(model, product_xyz, tag="neb/p"))
    # climbing is switched on only once a plain band has settled and shows an interior maximum: a climbing image
    # started on the interpolated band (or on a band without a barrier) runs up the slope instead
    neb = NEB(images, k=float(spring_ev_ang2), climb=False, remove_rotation_and_translation=False,
              method="improvedtangent")
    try:
        neb.interpolate(method="idpp", mic=False)
    except Exception:
        neb.interpolate(mic=False)
    for img in images[1:-1]:              # interpolation can move the frozen (link-bonded) atoms: put them back
        if model.fixed:
            pos = img.get_positions()
            pos[model.fixed] = np.asarray(reactant_xyz)[model.fixed]
            img.set_positions(pos)
    opt = FIRE(neb, logfile=os.path.join(model.workdir, "neb.log"))
    plain_steps = max(1, int(steps) // 3)
    opt.run(fmax=2.0 * fmax_kj_mol_ang / KJ_MOL_PER_EV, steps=plain_steps)
    used = int(opt.get_number_of_steps())
    band = [float(img.get_potential_energy()) for img in images[1:-1]]
    ends = (float(images[0].get_potential_energy()), float(images[-1].get_potential_energy()))
    if band and max(band) > max(ends):                     # an interior maximum: refine it by climbing
        neb.climb = True
        opt = FIRE(neb, logfile=os.path.join(model.workdir, "neb_climb.log"))
    converged = bool(opt.run(fmax=fmax_kj_mol_ang / KJ_MOL_PER_EV, steps=max(1, int(steps) - used)))
    used += int(opt.get_number_of_steps())
    frames = [img.get_positions().copy() for img in images]
    energies = []
    for k, xyz in enumerate(frames):
        e, _f, terms = model.evaluate(xyz, tag=f"neb/final_{k:02d}")
        energies.append(float(terms["energy_kj_mol"]))
    return {"frames": frames, "energies_kj_mol": energies, "converged": converged,
            "steps": used, "climbing": bool(neb.climb), "climbing_image": int(np.argmax(energies))}


def _masses(model):
    from ase.data import atomic_masses

    return np.array([float(atomic_masses[int(model.z[i])]) for i in model.qm])


def irc(model, ts_xyz, mode, step_amu_ang=0.05, max_steps=200, fmax_kj_mol_ang=2.0, initial_step_ang=0.05):
    """Intrinsic reaction coordinate from a transition state: steepest descent in mass-weighted coordinates
    (Euler steps of fixed length in sqrt(amu) Angstrom) in both directions of the imaginary mode, each end then
    optimized to its minimum. Returns both branches, with s < 0 on the 'minus' side."""
    m = _masses(model)
    sqrt_m = np.sqrt(m)[:, None]
    mode = np.asarray(mode, dtype=float).reshape((-1, 3)).copy()
    if model.fixed:
        mode[model.fixed] = 0.0
    mode /= max(float(np.linalg.norm(mode)), 1.0e-12)
    e_ts, _f, terms = model.evaluate(ts_xyz, tag="irc/ts")
    e_ts = float(terms["energy_kj_mol"])
    branches = {}
    for sign, label in ((+1.0, "plus"), (-1.0, "minus")):
        xyz = np.asarray(ts_xyz, dtype=float) + sign * float(initial_step_ang) * mode
        s = float(initial_step_ang) * float(np.linalg.norm(sqrt_m * mode))
        rows = []
        step = float(step_amu_ang)
        _e, f, terms = model.evaluate(xyz, tag=f"irc/{label}_000")
        e = float(terms["energy_kj_mol"])
        rows.append({"s_amu_ang": s, "energy_kj_mol": e, "qm_xyz": xyz.copy()})
        n = 0
        f_peak = 0.0
        while n < int(max_steps):
            f = np.asarray(f, dtype=float) / 10.0                   # kJ/mol/Angstrom
            if model.fixed:
                f[model.fixed] = 0.0
            f_max = float(np.max(np.linalg.norm(f, axis=1)))
            # reached the valley floor: small forces once they have passed their maximum (a broad, flat saddle
            # also has small forces next to the transition state, where the descent has only just begun)
            if f_max < float(fmax_kj_mol_ang) and f_max < f_peak:
                break
            f_peak = max(f_peak, f_max)
            g_mw = f / sqrt_m                                       # mass-weighted force (descent direction)
            norm = float(np.linalg.norm(g_mw))
            if norm < 1.0e-12:
                break
            trial = xyz + (step * g_mw / norm) / sqrt_m             # step of fixed mass-weighted length
            n += 1
            _e, f_new, terms = model.evaluate(trial, tag=f"irc/{label}_{n:03d}")
            e_new = float(terms["energy_kj_mol"])
            if e_new > e + 1.0e-3:
                # overshoot across the valley (or SCF noise on a flat stretch): halve the step and retry
                step *= 0.5
                if step < 0.01 * float(step_amu_ang):
                    break
                continue
            xyz, e, f = trial, e_new, f_new
            s += step
            rows.append({"s_amu_ang": s, "energy_kj_mol": e, "qm_xyz": xyz.copy()})
            step = min(float(step_amu_ang), step * 1.5)              # recover the step length after a cut
        end = _optimize(model, rows[-1]["qm_xyz"], (), fmax_kj_mol_ang=fmax_kj_mol_ang, steps=400,
                        tag=f"irc/{label}_min")
        branches[label] = {"rows": rows, "minimum": end}
    path = ([dict(r, s_amu_ang=-r["s_amu_ang"]) for r in reversed(branches["minus"]["rows"])]
            + [{"s_amu_ang": 0.0, "energy_kj_mol": e_ts, "qm_xyz": np.asarray(ts_xyz, dtype=float).copy()}]
            + branches["plus"]["rows"])
    return {"path": path, "ts_energy_kj_mol": e_ts, "plus_minimum": branches["plus"]["minimum"],
            "minus_minimum": branches["minus"]["minimum"]}


def reaction_path(model, form_bonds, break_bonds, n_images=9, fmax_kj_mol_ang=2.0, neb_fmax_kj_mol_ang=5.0,
                  neb_steps=300, irc_step_amu_ang=0.05, irc_max_steps=200, threshold_cm=80.0):
    """Full workflow (module docstring). ``form_bonds``/``break_bonds``: [(i, j), ...] global atom indices."""
    form_bonds = [(int(i), int(j)) for i, j in form_bonds]
    break_bonds = [(int(i), int(j)) for i, j in break_bonds]
    for i, j in form_bonds + break_bonds:
        if i not in model.qm or j not in model.qm:
            raise ValueError("every forming/breaking bond must have both atoms in the QM region")
    out = {"ok": False, "form_bonds": form_bonds, "break_bonds": break_bonds}
    react = _optimize(model, model.xyz_ref[model.qm], (), fmax_kj_mol_ang=fmax_kj_mol_ang, tag="reactant")
    out["reactant"] = {"energy_kj_mol": react["energy_kj_mol"], "converged": react["converged"],
                       "max_force_kj_mol_ang": react["max_force_kj_mol_ang"],
                       "check": bond_states(model, react["qm_xyz"], form_bonds, break_bonds)}
    if not out["reactant"]["check"]["reactant_like"]:
        out["error"] = "the optimized starting structure is not a reactant for the given bonds"
        return out
    prod = build_product(model, react["qm_xyz"], form_bonds, break_bonds, fmax_kj_mol_ang=fmax_kj_mol_ang)
    out["product"] = {"energy_kj_mol": prod["energy_kj_mol"], "converged": prod["converged"],
                      "max_force_kj_mol_ang": prod["max_force_kj_mol_ang"], "check": prod["check"]}
    if not prod["converged"] and prod["max_force_kj_mol_ang"] > 5.0 * float(fmax_kj_mol_ang):
        # the bond distances alone do not make a minimum: on a slope with no product well the optimizer stalls
        # with the bonds "formed", and the path, TS and IRC would then be searched towards a point that is not one
        out["error"] = (f"no stable product with the requested bonds: its optimization stopped on a slope (residual "
                        f"force {prod['max_force_kj_mol_ang']:.1f} kJ/mol/A), not at a minimum")
        return out
    if not prod["check"]["product_like"]:
        out["error"] = "no stable product with the requested bonds: it relaxed back or to another structure"
        return out
    neb = neb_path(model, react["qm_xyz"], prod["qm_xyz"], n_images=n_images, fmax_kj_mol_ang=neb_fmax_kj_mol_ang,
                   steps=neb_steps)
    e_r = float(react["energy_kj_mol"])
    out["path"] = {"energies_rel_kj_mol": [round(e - e_r, 3) for e in neb["energies_kj_mol"]],
                   "converged": neb["converged"], "steps": neb["steps"], "climbing_image": neb["climbing_image"],
                   "bond_distances": [bond_states(model, x, form_bonds, break_bonds)["bonds"] for x in neb["frames"]]}
    k = neb["climbing_image"]
    if not 0 < k < len(neb["frames"]) - 1:
        out["error"] = "the path has no interior maximum: no barrier between reactant and product"
        return out
    tangent = path_tangent_mode(neb["frames"], k)
    ts = model.refine_ts_dimer(neb["frames"][k], mode_guess=tangent, fmax_kj_mol_ang=fmax_kj_mol_ang)
    freq = model.vibrations(ts["qm_xyz"], threshold_cm=threshold_cm) if ts["converged"] else None
    out["ts"] = {"energy_kj_mol": ts["energy_kj_mol"], "converged": ts["converged"],
                 "stopped_reason": ts["stopped_reason"],
                 "frequencies_cm1": None if freq is None else [float(v) for v in np.sort(freq["freq_cm_signed"])[:8]],
                 "frequency_validation": None if freq is None else dict(freq["validation"]),
                 "bonds": bond_states(model, ts["qm_xyz"], form_bonds, break_bonds)["bonds"]}
    if not ts["converged"] or not (freq and freq["validation"].get("imaginary_mode_count") == 1):
        out["error"] = "transition-state optimization did not give a first-order saddle point"
        return out
    path = irc(model, ts["qm_xyz"], ts.get("eigenmode", tangent), step_amu_ang=irc_step_amu_ang,
               max_steps=irc_max_steps, fmax_kj_mol_ang=fmax_kj_mol_ang)
    ends = {}
    for side in ("plus", "minus"):
        mini = path[f"{side}_minimum"]
        ends[side] = dict(bond_states(model, mini["qm_xyz"], form_bonds, break_bonds),
                          energy_kj_mol=float(mini["energy_kj_mol"]))
    r_side = "minus" if ends["minus"]["reactant_like"] else "plus"
    p_side = "plus" if r_side == "minus" else "minus"
    connects = bool(ends[r_side]["reactant_like"] and ends[p_side]["product_like"])
    e_r_irc, e_p_irc, e_ts = ends[r_side]["energy_kj_mol"], ends[p_side]["energy_kj_mol"], float(ts["energy_kj_mol"])
    flip = -1.0 if r_side == "plus" else 1.0              # report the IRC from reactant (s < 0) to product (s > 0)
    rows = sorted(({"s_amu_ang": flip * r["s_amu_ang"], "energy_rel_ts_kj_mol": r["energy_kj_mol"] - e_ts,
                    "bonds": [b["distance_A"] for b in bond_states(model, r["qm_xyz"], form_bonds, break_bonds)["bonds"]]}
                   for r in path["path"]), key=lambda r: r["s_amu_ang"])
    out["irc"] = {"connects_reactant_and_product": connects, "reactant_end": ends[r_side], "product_end": ends[p_side],
                  "n_points": len(rows), "rows": rows}
    # the IRC geometries from the reactant end to the product end (for animations and analysis), completed at both
    # ends by the optimized minima the IRC leads to (where the valley is too flat for steepest descent to finish)
    frames = sorted(((flip * r["s_amu_ang"], r["qm_xyz"], r["energy_kj_mol"]) for r in path["path"]),
                    key=lambda t: t[0])
    sqrt_m = np.sqrt(_masses(model))[:, None]
    for side, at_start in ((r_side, True), (p_side, False)):
        mini = path[f"{side}_minimum"]
        edge = frames[0] if at_start else frames[-1]
        gap = float(np.linalg.norm(sqrt_m * (np.asarray(mini["qm_xyz"]) - np.asarray(edge[1]))))
        if gap > 1.0e-6:
            point = (edge[0] - gap if at_start else edge[0] + gap, np.asarray(mini["qm_xyz"]),
                     float(mini["energy_kj_mol"]))
            frames = [point] + frames if at_start else frames + [point]
    out["irc_frames"] = {"s_amu_ang": [f[0] for f in frames], "qm_xyz": [np.asarray(f[1]) for f in frames],
                         "energy_rel_reactant_kj_mol": [f[2] - e_r_irc for f in frames],
                         "ts_index": int(np.argmin([abs(f[0]) for f in frames]))}
    out.update({
        "ok": connects,
        "barrier_kj_mol": e_ts - e_r_irc, "reverse_barrier_kj_mol": e_ts - e_p_irc,
        "reaction_energy_kj_mol": e_p_irc - e_r_irc,
        "imaginary_cm1": float(freq["validation"].get("primary_imag_cm1") or 0.0),
        "n_energy_evaluations": int(getattr(model, "n_calls", 0)),
        "qm_xyz": {"reactant": react["qm_xyz"], "product": prod["qm_xyz"], "ts": ts["qm_xyz"]},
    })
    if not connects:
        out["error"] = "the IRC from the transition state does not connect the reactant and the product"
    return out


__all__ = ["reaction_path", "build_product", "neb_path", "irc", "bond_states", "covalent_length"]
