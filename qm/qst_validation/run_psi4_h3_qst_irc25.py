import csv
import json
import os
import sys

import numpy as np
import psi4


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from psi4_runtime import psi4_build_irc_profile_payload  # noqa: E402
from psi4_runtime import psi4_qst_build_guess_atoms  # noqa: E402
from psi4_runtime import psi4_validate_irc_ts_frequencies  # noqa: E402


BOHR_TO_ANGSTROM = 0.529177210903


def _h3_h_exchange_endpoints():
    """Small radical reaction: H + H2 -> H2 + H."""
    reactant = [
        {"name": "H1", "element": "H", "x": -0.37, "y": 0.0, "z": 0.0},
        {"name": "H2", "element": "H", "x": 0.37, "y": 0.0, "z": 0.0},
        {"name": "H3", "element": "H", "x": 2.20, "y": 0.0, "z": 0.0},
    ]
    product = [
        {"name": "H1", "element": "H", "x": -2.20, "y": 0.0, "z": 0.0},
        {"name": "H2", "element": "H", "x": -0.37, "y": 0.0, "z": 0.0},
        {"name": "H3", "element": "H", "x": 0.37, "y": 0.0, "z": 0.0},
    ]
    ts_guess = [
        {"name": "H1", "element": "H", "x": -0.93, "y": 0.0, "z": 0.0},
        {"name": "H2", "element": "H", "x": 0.00, "y": 0.0, "z": 0.0},
        {"name": "H3", "element": "H", "x": 0.93, "y": 0.0, "z": 0.0},
    ]
    return reactant, product, ts_guess


def _atoms_to_symbols_coords(atoms):
    symbols = [str(atom.get("element", "C")) for atom in atoms]
    coords = np.asarray(
        [[float(atom.get("x", 0.0)), float(atom.get("y", 0.0)), float(atom.get("z", 0.0))] for atom in atoms],
        dtype=float,
    )
    return symbols, coords


def _mol_from_symbols_coords(symbols, coords_ang, charge=0, multiplicity=2):
    lines = [f"{int(charge)} {int(multiplicity)}"]
    for sym, xyz in zip(symbols, np.asarray(coords_ang, dtype=float)):
        lines.append(f"{sym} {float(xyz[0]): .10f} {float(xyz[1]): .10f} {float(xyz[2]): .10f}")
    lines.extend(["units angstrom", "symmetry c1", "no_reorient", "no_com"])
    return psi4.geometry("\n".join(lines))


def _write_xyz(path, symbols, coords_ang, comments=None):
    coords = np.asarray(coords_ang, dtype=float)
    if coords.ndim == 2:
        coords = coords.reshape((1, coords.shape[0], coords.shape[1]))
    comments = list(comments or [])
    with open(path, "w", encoding="utf-8") as fh:
        for frame_idx in range(int(coords.shape[0])):
            fh.write(f"{int(coords.shape[1])}\n")
            comment = comments[frame_idx] if frame_idx < len(comments) else f"frame {frame_idx}"
            fh.write(str(comment) + "\n")
            for sym, xyz in zip(symbols, coords[frame_idx]):
                fh.write(f"{sym:2s} {float(xyz[0]): .10f} {float(xyz[1]): .10f} {float(xyz[2]): .10f}\n")


def _write_profile_csv(path, profile):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["point", "coordinate_amu05_bohr", "energy_hartree", "relative_energy_kcal_mol", "branch"])
        rows = zip(
            profile.get("point", []),
            profile.get("coordinate_amu05_bohr", []),
            profile.get("energy_hartree", []),
            profile.get("relative_energy_kcal_mol", []),
            profile.get("branch", []),
        )
        for row in rows:
            writer.writerow(row)


def _plot_profile(path, profile, title):
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(path), ".matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(profile.get("coordinate_amu05_bohr", []), dtype=float)
    y = np.asarray(profile.get("relative_energy_kcal_mol", []), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(x, y, color="#2f2f2f", linewidth=1.7, marker="o", markersize=3.5)
    ax.axvline(0.0, color="#8c8c8c", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("IRC coordinate (amu$^{1/2}$ bohr)")
    ax.set_ylabel("Relative energy (kcal/mol)")
    ax.grid(True, color="#d9d9d9", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#7a7a7a")
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


def _extract_frequency_arrays(wfn):
    fa = getattr(wfn, "frequency_analysis", {}) or {}
    omega = fa.get("omega") if isinstance(fa, dict) else None
    raw = omega.data if hasattr(omega, "data") else omega
    data = np.asarray(raw, dtype=complex).reshape(-1) if raw is not None else np.asarray([], dtype=complex)
    return [float(v.real) for v in data], [float(abs(v.imag)) for v in data]


def _history_value(history, names):
    if not isinstance(history, dict):
        return None
    for name in names:
        if name in history:
            return history[name]
    return None


def _history_to_arrays(history, natoms):
    raw_coords = _history_value(history, ("coordinates", "geometry", "geometries"))
    raw_energies = _history_value(history, ("energy", "energies"))
    coords = []
    for item in list(raw_coords) if raw_coords is not None else []:
        arr = np.asarray(item, dtype=float)
        if arr.size == int(natoms) * 3:
            coords.append(arr.reshape((int(natoms), 3)) * float(BOHR_TO_ANGSTROM))
    energies = np.asarray(list(raw_energies) if raw_energies is not None else [], dtype=float).reshape(-1)
    n = int(min(len(coords), energies.size))
    return np.asarray(coords[:n], dtype=float), energies[:n]


def _run_ts(label, symbols, coords_ang, out_dir):
    psi4.core.clean()
    psi4.set_output_file(os.path.join(out_dir, f"{label}.out"), False)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "pk",
            "reference": "uhf",
            "opt_type": "ts",
            "geom_maxiter": 80,
            "full_hess_every": 0,
            "g_convergence": "gau_loose",
            "print_trajectory_xyz_file": True,
        }
    )
    mol = _mol_from_symbols_coords(symbols, coords_ang)
    energy, _wfn = psi4.optimize("hf", molecule=mol, return_wfn=True)
    geom_ang = np.asarray(mol.geometry().to_array(), dtype=float) * float(BOHR_TO_ANGSTROM)
    _write_xyz(os.path.join(out_dir, f"{label}.xyz"), symbols, geom_ang, [f"{label} optimized TS"])
    return {"energy_hartree": float(energy), "coords_ang": geom_ang}


def _run_frequency(symbols, coords_ang, out_dir):
    psi4.core.clean()
    psi4.set_output_file(os.path.join(out_dir, "ts_frequency.out"), False)
    psi4.set_options({"basis": "sto-3g", "scf_type": "pk", "reference": "uhf"})
    mol = _mol_from_symbols_coords(symbols, coords_ang)
    energy, wfn = psi4.frequency("hf", molecule=mol, return_wfn=True)
    freq_real, freq_imag = _extract_frequency_arrays(wfn)
    validation = psi4_validate_irc_ts_frequencies(freq_real, freq_imag, threshold_cm=10.0)
    return {
        "energy_hartree": float(energy),
        "freq_real_cm1": freq_real,
        "freq_imag_cm1": freq_imag,
        "validation": validation,
    }


def _run_irc_branch(direction, symbols, coords_ang, out_dir):
    psi4.core.clean()
    psi4.set_output_file(os.path.join(out_dir, f"irc_{direction.lower()}.out"), False)
    psi4.set_options(
        {
            "basis": "sto-3g",
            "scf_type": "pk",
            "reference": "uhf",
            "opt_type": "irc",
            "irc_direction": str(direction).upper(),
            "irc_points": 25,
            "irc_step_size": 0.2,
            "geom_maxiter": 100,
            "print_trajectory_xyz_file": True,
        }
    )
    mol = _mol_from_symbols_coords(symbols, coords_ang)
    _energy, _wfn, history = psi4.optimize("hf", molecule=mol, return_wfn=True, return_history=True)
    return _history_to_arrays(history, len(symbols))


def _downsample_indices(n_items, max_items):
    if n_items <= max_items:
        return list(range(int(n_items)))
    return [int(round(v)) for v in np.linspace(0, int(n_items) - 1, int(max_items))]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "test_outputs", "qst_irc_examples", "psi4_h3")
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    psi4.set_memory("1 GB")
    psi4.set_num_threads(2)

    reactant, product, ts_guess = _h3_h_exchange_endpoints()
    qst2 = psi4_qst_build_guess_atoms(reactant, product, n_images=9)
    qst3 = psi4_qst_build_guess_atoms(reactant, product, ts_guess_atoms=ts_guess, ts_weight=0.8, n_images=9)
    symbols, qst2_coords = _atoms_to_symbols_coords(qst2["guess_atoms"])
    _symbols3, qst3_coords = _atoms_to_symbols_coords(qst3["guess_atoms"])

    _write_xyz(os.path.join(out_dir, "qst2_guess.xyz"), symbols, qst2_coords, ["QST2 synchronous-transit guess"])
    _write_xyz(os.path.join(out_dir, "qst3_guess.xyz"), symbols, qst3_coords, ["QST3 synchronous-transit guess"])

    qst2_ts = _run_ts("qst2_ts", symbols, qst2_coords, out_dir)
    qst3_ts = _run_ts("qst3_ts", symbols, qst3_coords, out_dir)
    freq = _run_frequency(symbols, qst3_ts["coords_ang"], out_dir)
    if not bool(freq["validation"].get("ok", False)):
        raise RuntimeError(f"Refined TS failed frequency validation: {freq['validation']}")

    back_coords, back_energies = _run_irc_branch("BACKWARD", symbols, qst3_ts["coords_ang"], out_dir)
    fwd_coords, fwd_energies = _run_irc_branch("FORWARD", symbols, qst3_ts["coords_ang"], out_dir)
    if back_coords.size == 0 or fwd_coords.size == 0:
        raise RuntimeError("Psi4 IRC did not return usable coordinate histories.")
    coords = np.concatenate([back_coords[::-1], fwd_coords[1:]], axis=0)
    energies = np.concatenate([back_energies[::-1], fwd_energies[1:]], axis=0)
    keep = _downsample_indices(int(coords.shape[0]), 25)
    coords_25 = coords[keep, :, :]
    energies_25 = energies[keep]
    profile = psi4_build_irc_profile_payload(coords_25, energies_25, symbols)
    if not bool(profile.get("ok", False)):
        raise RuntimeError(f"Failed building Psi4 IRC profile: {profile}")

    profile_csv = os.path.join(out_dir, "psi4_h3_irc_profile.csv")
    profile_png = os.path.join(out_dir, "psi4_h3_irc_profile.png")
    path_xyz = os.path.join(out_dir, "psi4_h3_irc_path_25.xyz")
    _write_profile_csv(profile_csv, profile)
    _write_xyz(path_xyz, symbols, coords_25, [f"Psi4 H3 IRC point {i}" for i in range(int(coords_25.shape[0]))])
    _plot_profile(profile_png, profile, "Psi4 H + H2 IRC")

    summary = {
        "ok": True,
        "system": "H + H2 -> H2 + H",
        "charge": 0,
        "multiplicity": 2,
        "method": "UHF/STO-3G",
        "qst2_note": qst2.get("note", ""),
        "qst3_note": qst3.get("note", ""),
        "qst2_ts_energy_hartree": qst2_ts["energy_hartree"],
        "qst3_ts_energy_hartree": qst3_ts["energy_hartree"],
        "frequency_validation": freq["validation"],
        "irc_raw_points": int(coords.shape[0]),
        "irc_profile_points": int(profile.get("n_points", 0)),
        "irc_csv": profile_csv,
        "irc_plot_png": profile_png,
        "irc_xyz": path_xyz,
    }
    summary_path = os.path.join(out_dir, "psi4_h3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
